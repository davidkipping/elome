"""Secondary eclipses in the batched models (make_eclipse_flux) and in
make_transit_target: the parameter layout, the model against the
frontend's transit and eclipse, the two timings, the orbit helpers against
a brute-force orbit, the two precisions and gradients."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome
from elome.anvil import (PenalizedLogLike, import_engine, make_eclipse_flux,
                         make_transit_target, transit_param_names)
from elome.ld import hybrid4_to_q_np, u_to_q_np

engine, _ = import_engine()
T0, PER, RP, A, INC, FP, RSTAR = 2500.123456, 4.2345, 0.11, 11.3, 88.6, 2e-3, 1.2
U = [0.4, 0.25]
W4 = [0.2, 0.25, 0.1, 0.05]
DAY_S = 86_400.0


def _kh(ecc, w):
    s = math.sqrt(ecc)
    return s * math.cos(math.radians(w)), s * math.sin(math.radians(w))


def _b(ecc, w):
    """The impact parameter at transit, Elómë's b."""
    return (A * math.cos(math.radians(INC)) * (1.0 - ecc ** 2)
            / (1.0 + ecc * math.sin(math.radians(w))))


def _delay(ecc, w, rstar):
    return 0.0 if rstar is None else elome.light_travel_delay(
        A, _b(ecc, w), rstar, *_kh(ecc, w))


def _t_occ(ecc, w, rstar=None):
    return T0 + elome.occultation_offset(*_kh(ecc, w)) * PER + _delay(ecc, w, rstar)


def _times(ecc, w, rstar=None, n=300, seed=1, transit=True):
    """A transit two orbits on and an eclipse three on (eclipse only:
    two eclipses)."""
    rng = np.random.default_rng(seed)
    to = _t_occ(ecc, w, rstar)
    parts = [to + 3 * PER + rng.uniform(-0.25, 0.25, n)]
    parts.append((T0 if transit else to) + 2 * PER + rng.uniform(-0.25, 0.25, n))
    return np.sort(np.concatenate(parts))


def _frontend(t, ecc, w, transittype, law="quadratic", rstar=None,
              exp_time=0.0, n_sub=1):
    """(transit + eclipse - fp) / (1 + fp) from TransitModel, the eclipse
    seen the light-travel delay late."""
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, RP, A, INC, ecc, w
    p.limb_dark, p.u, p.fp = law, (U if law == "quadratic" else W4), FP
    ekw = dict(exp_time=exp_time, supersample_factor=n_sub) if exp_time else {}
    sec = elome.TransitModel(p, t - _delay(ecc, w, rstar), transittype="secondary",
                             **ekw).light_curve(p)
    dev = sec - 1.0 - FP
    if transittype == "both":
        tkw = dict(exp_time=exp_time, integration="contact", n_gl=5) if exp_time else {}
        dev = dev + elome.TransitModel(p, t, **tkw).light_curve(p) - 1.0
    return dev / (1.0 + FP)


def _target(transittype, orbit="circular", ecc=0.0, w=90.0, timing="transit",
            law="quadratic", rstar=None, exp_time=0.0, n_sub=1, seed=1):
    t = _times(ecc, w, rstar, seed=seed, transit=transittype == "both")
    flux = _frontend(t, ecc, w, transittype, law, rstar, exp_time, n_sub)
    y = 1.0 + flux + 1e-4 * np.random.default_rng(seed).standard_normal(t.size)
    guess = (T0 if timing == "transit" else _t_occ(ecc, w, rstar)) + 0.003
    kw = dict(eccentric=True) if orbit == "sampled" else (
        dict(ecc=ecc, w=w) if orbit == "fixed" else {})
    if exp_time:
        kw.update(exp_time=exp_time, n_sub=n_sub)
    tt = make_transit_target(t, y, 1e-4, guess, PER - 0.0005,
                             transittype=transittype, timing=timing,
                             limb_dark=law, rstar=rstar, **kw)
    return tt, flux


def _truth(tt, ecc=0.0, w=90.0, rstar=None):
    kw = dict(r=RP, a=A, b=_b(ecc, w), fp=FP)
    if tt.eccentric:
        kw["secosw"], kw["sesinw"] = _kh(ecc, w)
    if tt.transittype == "both":
        q = (u_to_q_np(*U) if tt.limb_dark == "quadratic"
             else hybrid4_to_q_np(np.array(W4)))
        kw.update({f"q{j + 1}": float(v) for j, v in enumerate(q)})
    t0 = T0 if tt.timing == "transit" else _t_occ(ecc, w, rstar)
    return tt.model_params(t0=t0, period=PER, **kw)


def _model64(tt, v):
    with mx.stream(mx.cpu):
        return np.array(tt.model_fn(mx.array(v[None], dtype=mx.float64),
                                    mx.array(tt.x64, dtype=mx.float64)))[0]


class TestLayout:
    def test_names(self):
        assert transit_param_names(transittype="secondary") == [
            "t0_off", "p_off", "r", "b", "a", "fp", "df0"]
        assert transit_param_names(transittype="secondary", timing="occultation",
                                   eccentric=True) == [
            "tocc_off", "p_off", "r", "b", "a", "secosw", "sesinw", "fp", "df0"]
        assert transit_param_names("hybrid4", transittype="both") == [
            "t0_off", "p_off", "r", "b", "a", "q1", "q2", "q3", "q4", "fp", "df0"]

    @pytest.mark.parametrize("kw,match", [
        (dict(transittype="tertiary"), "transittype must be"),
        (dict(timing="periastron"), "timing must be"),
        (dict(timing="occultation"), "times a secondary eclipse"),
        (dict(transittype="both", limb_dark="hybrid4", n_spots=1), "unspotted"),
        (dict(transittype="secondary", limb_dark="hybrid4"), "no limb darkening"),
    ])
    def test_bad_layouts(self, kw, match):
        with pytest.raises(ValueError, match=match):
            transit_param_names(**kw)


CASES = [
    # transittype, orbit, ecc, w, timing, law, rstar
    ("secondary", "circular", 0.0, 90.0, "transit", "quadratic", None),
    ("secondary", "circular", 0.0, 90.0, "occultation", "quadratic", RSTAR),
    ("secondary", "fixed", 0.3, 63.0, "transit", "quadratic", RSTAR),
    ("secondary", "sampled", 0.6, 270.0, "occultation", "quadratic", None),
    ("both", "circular", 0.0, 90.0, "transit", "quadratic", RSTAR),
    ("both", "fixed", 0.3, -40.0, "occultation", "quadratic", None),
    ("both", "sampled", 0.3, 63.0, "transit", "hybrid4", RSTAR),
    ("both", "sampled", 0.6, 200.0, "occultation", "quadratic", RSTAR),
]


class TestModel:
    @pytest.mark.parametrize("case", CASES)
    def test_is_the_frontend(self, case):
        tt_type, orbit, ecc, w, timing, law, rstar = case
        tt, flux = _target(tt_type, orbit, ecc, w, timing, law, rstar)
        assert tt.transittype == tt_type and tt.timing == timing
        m = _model64(tt, _truth(tt, ecc, w, rstar))
        assert np.max(np.abs(m - flux)) < 1e-12
        # the eclipse is in the data, total: its depth is fp / (1 + fp)
        assert np.sum(np.abs(m + FP / (1.0 + FP)) < 1e-15) > 10

    def test_timings_are_one_model(self):
        """The same physical orbit in either timing gives the same curve."""
        ecc, w = 0.4, 130.0
        out = []
        for timing in ("transit", "occultation"):
            tt, flux = _target("both", "sampled", ecc, w, timing, rstar=RSTAR)
            out.append(_model64(tt, _truth(tt, ecc, w, RSTAR)))
        assert np.max(np.abs(out[0] - out[1])) < 1e-13
        assert np.max(np.abs(out[0] - flux)) < 1e-12

    def test_out_of_eclipse_is_zero(self):
        tt, _ = _target("secondary")
        m = _model64(tt, _truth(tt))
        far = np.abs(np.abs(tt.x64[0]) - 0.5 * PER) > 0.2
        assert far.any() and np.max(np.abs(m[far])) == 0.0

    def test_exposures(self):
        """The transit by the contact rule, the eclipse supersampled."""
        exp = 29.4 / 1440.0
        tt, flux = _target("both", "sampled", 0.3, 63.0, exp_time=exp, n_sub=9)
        assert np.max(np.abs(_model64(tt, _truth(tt, 0.3, 63.0)) - flux)) < 1e-12

    def test_light_travel_moves_only_the_eclipse(self):
        tt, _ = _target("both", rstar=RSTAR)
        v = _truth(tt, rstar=RSTAR)
        plain = make_eclipse_flux(tt.period_ref, "both")
        with mx.stream(mx.cpu):
            vx = (mx.array(v[None], dtype=mx.float64), mx.array(tt.x64, dtype=mx.float64))
            d = np.array(tt.model_fn(*vx) - plain(*vx))[0]
        near_transit = np.abs(tt.x64[0]) < 0.3
        assert np.max(np.abs(d[near_transit])) == 0.0
        assert np.max(np.abs(d[~near_transit])) > 1e-5

    def test_engine_contract_without_a_target(self):
        """The model function alone, as any sampler would call it."""
        fn = make_eclipse_flux(PER, "secondary")
        v = mx.array([[0.0, 0.0, RP, 0.3, A, FP, 0.0]] * 3)
        x = mx.array(np.stack([np.linspace(1.8, 2.4, 50), np.zeros(50)]),
                     dtype=mx.float32)
        assert fn(v, x).shape == (3, 50)


class TestOrbitHelpers:
    @staticmethod
    def _observed_conjunctions(ecc, w_deg, a, b, rstar):
        """Brute force: Kepler's equation by Newton on a fine grid, the
        sky-plane conjunctions (X = 0) by bisection, each seen at
        t - Z / c (Z toward the observer; Winn 2010's frame)."""
        w = math.radians(w_deg)
        ci = b * (1.0 + ecc * math.sin(w)) / (a * (1.0 - ecc ** 2))
        si = math.sqrt(1.0 - ci * ci)

        def pos(t):                                  # t in orbits since periastron
            M = 2.0 * math.pi * t
            E = M
            for _ in range(60):
                E -= (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
            f = 2.0 * math.atan2(math.sqrt(1 + ecc) * math.sin(E / 2),
                                 math.sqrt(1 - ecc) * math.cos(E / 2))
            r = a * (1.0 - ecc * math.cos(E))
            return -r * math.cos(w + f), r * math.sin(w + f) * si

        grid = np.linspace(0.0, 1.0, 4001)
        X = np.array([pos(t)[0] for t in grid])
        found = []
        for i in np.nonzero(np.sign(X[:-1]) != np.sign(X[1:]))[0]:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if np.sign(pos(mid)[0]) == np.sign(pos(lo)[0]):
                    lo = mid
                else:
                    hi = mid
            found.append((lo, pos(lo)[1]))
        (t_tr, z_tr), = [c for c in found if c[1] > 0]
        (t_oc, z_oc), = [c for c in found if c[1] < 0]
        lt = rstar * 6.957e8 / 299_792_458.0 / DAY_S       # R* / c in days
        return (t_oc - t_tr) % 1.0, (z_tr - z_oc) * lt

    @pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 63.0), (0.6, 270.0),
                                       (0.5, 200.0), (0.8, -30.0)])
    def test_against_a_brute_force_orbit(self, ecc, w):
        b = 0.4
        off, dt = self._observed_conjunctions(ecc, w, A, b, RSTAR)
        k, h = _kh(ecc, w)
        assert abs(elome.occultation_offset(k, h) - off) < 1e-10
        assert abs(elome.light_travel_delay(A, b, RSTAR, k, h) - dt) < 1e-9 * dt
        if ecc == 0.0:                           # 2 a sin i / c
            si = math.sqrt(1.0 - (b / A) ** 2)
            assert abs(dt - 2 * A * si * RSTAR * 6.957e8 / 299_792_458.0 / DAY_S) < 1e-15

    def test_offset_dev_in_float32(self):
        """occultation_offset - 1/2 accurate relative to itself in float32,
        what an occultation-timed model shifts its transits by."""
        rng = np.random.default_rng(3)
        e = rng.uniform(0.0, 0.9, 2000)
        w = rng.uniform(-math.pi, math.pi, 2000)
        k, h = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
        d32 = np.array(elome.eclipse.occultation_offset_dev(
            mx.array(k.astype(np.float32)), mx.array(h.astype(np.float32))))
        k32, h32 = k.astype(np.float32).astype(float), h.astype(np.float32).astype(float)
        d64 = np.array([elome.eclipse.occultation_offset_dev(a, b) for a, b in zip(k32, h32)])
        # a few ulp of the offset itself (5e-7 of it at worst, e -> 0.9)
        assert np.max(np.abs(d32 - d64)) < 1e-6 * np.abs(d64).max()
        small = e < 0.01
        assert np.max(np.abs(d32 - d64)[small]) < 1e-9

    def test_occultation_center_times(self):
        from elome.orbit import occultation_center_times
        t = np.array([10.0, 10.0 + 0.5 * PER + 0.1, 10.0 + 3 * PER - 0.2])
        x = occultation_center_times(t, 10.0, PER)
        assert np.allclose(x[0], [0.0, 0.1, -0.2]) and np.allclose(x[2], [1, 0, 1])
        assert np.allclose(x[1], [0, 1, 3])

    def test_mlx_matches_floats_and_differentiates(self):
        with mx.stream(mx.cpu):
            for ecc, w in [(0.0, 90.0), (0.3, 63.0), (0.7, 250.0)]:
                k, h = _kh(ecc, w)
                km, hm = (mx.array([v], dtype=mx.float64) for v in (k, h))
                am, bm = mx.array([A], dtype=mx.float64), mx.array([0.4], dtype=mx.float64)
                assert abs(float(elome.occultation_offset(km, hm)[0])
                           - elome.occultation_offset(k, h)) < 1e-14
                assert abs(float(elome.light_travel_delay(am, bm, RSTAR, km, hm)[0])
                           - elome.light_travel_delay(A, 0.4, RSTAR, k, h)) < 1e-18
                # a Python-float orbit is kept in float64, not rounded to float32
                assert abs(float(elome.light_travel_delay(am, bm, RSTAR, k, h)[0])
                           - elome.light_travel_delay(A, 0.4, RSTAR, k, h)) < 1e-18
        for dt in (mx.float32,):
            z = mx.zeros((1,), dtype=dt)
            g = mx.grad(lambda kh: mx.sum(elome.occultation_offset(kh[0], kh[1])
                                          + elome.light_travel_delay(
                                              mx.array([A], dtype=dt),
                                              mx.array([0.4], dtype=dt), RSTAR,
                                              kh[0], kh[1])))(mx.stack([z, z]))
            assert np.isfinite(np.array(g)).all()


class TestTarget:
    @pytest.mark.parametrize("timing,gate", [("transit", 0.1), ("occultation", 1.0)])
    def test_precision_gradient_and_compile(self, timing, gate):
        """A joint fit timed by its occultation on a sampled eccentric orbit
        places the transit by occultation_offset_dev computed in float32,
        a few ulp of it times P (milliseconds here): coherent per chain on
        the steep transit, so its float32 log-likelihood is ~10x noisier
        than the transit-timed model's (0.4 against 0.03 here)."""
        tt, _ = _target("both", "sampled", 0.3, 63.0, timing=timing, rstar=RSTAR)
        assert isinstance(tt.loglike, PenalizedLogLike) and tt.compile_step
        u0 = tt.transform.from_model_np(_truth(tt, 0.3, 63.0, RSTAR))
        rng = np.random.default_rng(0)
        u = mx.array((u0 + 0.003 * rng.standard_normal((16, u0.size))).astype(np.float32))
        assert engine.validate_precision(tt.target, u).max_abs_err < gate
        g = np.array(mx.grad(lambda uu: mx.sum(tt.target.log_prob(uu)))(u))
        assert np.isfinite(g).all() and (np.abs(g).max(0) > 0).all()
        lp = np.array(tt.target.log_prob(u))
        lpc = np.array(mx.compile(tt.target.log_prob)(u))
        assert np.max(np.abs(lp - lpc)) < 1e-3 * np.max(np.abs(lp))

    def test_eclipse_alone_gradient(self):
        tt, _ = _target("secondary", "circular")
        u0 = tt.transform.from_model_np(_truth(tt))
        u = mx.array(np.tile(u0, (4, 1)).astype(np.float32))
        g = np.array(mx.grad(lambda uu: mx.sum(tt.target.log_prob(uu)))(u))
        assert np.isfinite(g).all() and (np.abs(g).max(0) > 0).all()

    def test_fixed_orbit_barrier(self):
        tt, _ = _target("secondary", "fixed", 0.6, 270.0)
        assert isinstance(tt.loglike, PenalizedLogLike) and not tt.eccentric
        v = _truth(tt, 0.6, 270.0)
        bad = v.copy()
        bad[tt.param_names.index("a")] = 2.0            # periastron inside the star
        lp = [float(np.array(tt.loglike(mx.array(x[None].astype(np.float32))))[0])
              for x in (v, bad)]
        assert lp[1] < lp[0] - 1e3

    def test_reported_time_is_the_sampled_events(self):
        tt, _ = _target("secondary", timing="occultation", rstar=RSTAR)
        v = _truth(tt, rstar=RSTAR)
        phys = tt.transform.to_physical(v[None])[0]
        assert abs(phys[0] - _t_occ(0.0, 90.0, RSTAR)) < 1e-9

    def test_options_refused(self):
        t = _times(0.0, 90.0, n=40)
        y = np.ones_like(t)
        bad = [
            (dict(ecc=0.3, w=40.0), "for a fit with a secondary eclipse"),
            (dict(rstar=1.0), "for a fit with a secondary eclipse"),
            (dict(transittype="both", ecc=0.3), "both ecc= and w="),
            (dict(transittype="both", ecc=1.2, w=0.0), "ecc must be in"),
            (dict(transittype="both", ecc=0.3, w=1.0, eccentric=True), "one or the other"),
            (dict(transittype="both", rstar=-1.0), "rstar"),
            (dict(transittype="both", exp_time=0.02), "n_sub > 1"),
            (dict(transittype="secondary", core="graph"), "core selects"),
            (dict(transittype="secondary", limb_dark="hybrid4"), "no limb darkening"),
            (dict(timing="occultation"), "times a secondary eclipse"),
        ]
        for kw, match in bad:
            with pytest.raises(ValueError, match=match):
                make_transit_target(t, y, 1e-4, T0, PER, **kw)
        with pytest.raises(ValueError, match="unknown parameter"):
            make_transit_target(t, y, 1e-4, T0, PER, bounds={"fp": (0.0, 0.1)})


@pytest.mark.slow
def test_chees_smoke_on_a_joint_fit():
    """A short ChEES-HMC run on a transit and an eclipse: no divergences,
    every parameter within 4 sd of the truth it started at."""
    tt, _ = _target("both", "sampled", 0.3, 63.0, rstar=RSTAR)
    v = _truth(tt, 0.3, 63.0, RSTAR)
    u0 = tt.transform.from_model_np(v)
    rng = np.random.default_rng(2)
    u = mx.array((u0 + 0.002 * rng.standard_normal((64, u0.size))).astype(np.float32))
    res = engine.run(engine.ChEESHMC(tt.target, max_leapfrog=32), tt.target, u,
                     n_warmup=60, n_samples=40, seed=1, compile_step=tt.compile_step)
    assert res.extras.get("n_divergent", 0) == 0
    draws = tt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    assert (np.abs(draws.mean(0) - v) < 4.0 * (draws.std(0) + 1e-12)).all()
