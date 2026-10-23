"""make_transit_target with a hybrid limb-darkening law, and on a spotted
star: the parameter layout, the model against the frontend, the two
precisions, gradients, and the spot-specific options."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome
from elome.anvil import (PARAM_NAMES, PARAM_NAMES_ECC, PenalizedLogLike,
                         import_engine, make_transit_target,
                         transit_param_names)
from elome.ld import hybrid2_to_q_np, hybrid4_to_q_np, q_from_simplex_np

engine, _ = import_engine()
T0, PER, RP, A, INC = 2500.123456, 4.2345, 0.11, 11.3, 88.6
W = {"hybrid2": [0.3, 0.05], "hybrid4": [0.2, 0.25, 0.1, 0.05],
     "hybrid5": [0.2, 0.2, 0.1, 0.05, 0.05]}
SPOTS = [[15.0, -20.0, 9.0, 0.5], [-40.0, 10.0, 12.0, 0.7]]
PROT, ISTAR, LAM = 9.0, 75.0, 20.0
EP = T0 + PER                       # the spots' epoch


def _q(law):
    w = np.array(W[law])
    if law == "hybrid2":
        return np.array(hybrid2_to_q_np(*w))
    return hybrid4_to_q_np(w) if law == "hybrid4" else q_from_simplex_np(w)


def _photometry(law="hybrid4", spots=False, ecc=0.0, w=90.0, exp_time=0.0,
                n=300, yerr=2e-4, seed=1, star_alone=False):
    """Three transits through the frontend (spotted: on EP's star; with
    ``star_alone`` the same times with the planet half an orbit away)."""
    rng = np.random.default_rng(seed)
    t = np.sort(np.concatenate([T0 + k * PER + rng.uniform(-0.2, 0.2, n)
                                for k in range(3)]))
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, RP, A, INC, ecc, w
    if star_alone:
        p.t0 = T0 + 0.5 * PER
    p.limb_dark, p.u = law, W[law]
    if spots:
        p.spots, p.prot, p.istar, p.lam, p.spot_epoch = SPOTS, PROT, ISTAR, LAM, EP
    kw = dict(exp_time=exp_time, integration="contact", n_gl=5) if exp_time else {}
    flux = elome.TransitModel(p, t, **kw).light_curve(p)
    return t, flux + yerr * rng.standard_normal(t.size), yerr, flux


def _truth(tt, ecc=0.0, w=90.0):
    b = A * math.cos(math.radians(INC))
    kw = dict(r=RP, a=A, b=b)
    if tt.eccentric:
        kw["b"] = b * (1.0 - ecc ** 2) / (1.0 + ecc * math.sin(math.radians(w)))
        kw["secosw"] = math.sqrt(ecc) * math.cos(math.radians(w))
        kw["sesinw"] = math.sqrt(ecc) * math.sin(math.radians(w))
    kw.update({f"q{j + 1}": float(v) for j, v in enumerate(_q(tt.limb_dark))})
    for j, (lon, lat, rad, c) in enumerate(SPOTS[:tt.n_spots], 1):
        kw.update({f"lon{j}": lon, f"sinlat{j}": math.sin(math.radians(lat)),
                   f"rad{j}": rad, f"contrast{j}": c})
    if tt.n_spots:
        kw.update(prot=PROT, cos_istar=math.cos(math.radians(ISTAR)), lam=LAM)
    return tt.model_params(t0=T0, period=PER, **kw)


def _model64(tt, v):
    with mx.stream(mx.cpu):
        return np.array(tt.model_fn(mx.array(v[None], dtype=mx.float64),
                                    mx.array(tt.x64, dtype=mx.float64)))[0]


def _target(law="hybrid4", spots=False, **kw):
    t, y, yerr, flux = _photometry(law, spots, kw.pop("ecc", 0.0),
                                   kw.pop("w", 90.0), kw.get("exp_time", 0.0))
    extra = dict(n_spots=2, bounds={"prot": (5.0, 15.0)}, spot_epoch=EP) if spots else {}
    if spots and kw.get("relative", True):
        # relative to the spotted star's own level: the frontend's flux over
        # its star-alone flux at the same times
        star = _photometry(law, spots, star_alone=True)[3]
        flux, y = flux / star, y / star
    tt = make_transit_target(t, y, yerr, T0 + 0.003, PER - 0.0005,
                             limb_dark=law, **extra, **kw)
    return tt, flux


class TestLayout:
    def test_quadratic_names_unchanged(self):
        assert transit_param_names() == PARAM_NAMES
        assert transit_param_names(eccentric=True) == PARAM_NAMES_ECC

    def test_hybrid_and_spot_names(self):
        assert transit_param_names("hybrid2") == [
            "t0_off", "p_off", "r", "b", "a", "q1", "q2", "df0"]
        names = transit_param_names("hybrid5", eccentric=True, n_spots=2)
        assert names[5:10] == ["q1", "q2", "q3", "q4", "q5"]
        assert names[10:12] == ["secosw", "sesinw"]
        assert names[12:20] == ["lon1", "sinlat1", "rad1", "contrast1",
                                "lon2", "sinlat2", "rad2", "contrast2"]
        assert names[20:] == ["prot", "cos_istar", "lam", "df0"]

    @pytest.mark.parametrize("kw,match", [
        (dict(limb_dark="quadratic", n_spots=1), "hybrid law"),
        (dict(limb_dark="nonlinear"), "unknown limb_dark"),
        (dict(limb_dark="hybrid4", n_spots=-1), "n_spots"),
    ])
    def test_bad_layouts(self, kw, match):
        with pytest.raises(ValueError, match=match):
            transit_param_names(**kw)


class TestHybridTarget:
    @pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
    def test_model_is_the_frontend(self, law):
        tt, flux = _target(law)
        assert tt.compile_step and tt.spot_epoch is None
        assert np.max(np.abs(_model64(tt, _truth(tt)) - (flux - 1.0))) < 1e-12

    def test_eccentric_model_and_barrier(self):
        tt, flux = _target("hybrid5", ecc=0.3, w=63.0, eccentric=True)
        assert isinstance(tt.loglike, PenalizedLogLike)
        v = _truth(tt, 0.3, 63.0)
        assert np.max(np.abs(_model64(tt, v) - (flux - 1.0))) < 1e-12
        bad = v.copy()
        bad[tt.param_names.index("a")] = 1.05          # periastron inside
        lp = [float(np.array(tt.loglike(mx.array(x[None].astype(np.float32))))[0])
              for x in (v, bad)]
        assert lp[1] < lp[0] - 1e3

    def test_exposures(self):
        exp = 29.4 / 1440.0
        tt, flux = _target("hybrid4", exp_time=exp)
        assert np.max(np.abs(_model64(tt, _truth(tt)) - (flux - 1.0))) < 1e-12

    def test_precision_and_gradient(self):
        tt, _ = _target("hybrid4")
        u0 = tt.transform.from_model_np(_truth(tt))
        rng = np.random.default_rng(0)
        u = mx.array((u0 + 0.003 * rng.standard_normal((16, u0.size))).astype(np.float32))
        rep = engine.validate_precision(tt.target, u)
        assert rep.max_abs_err < 0.1
        g = np.array(mx.grad(lambda uu: mx.sum(tt.target.log_prob(uu)))(u))
        assert np.isfinite(g).all() and (np.abs(g).max(0) > 0).all()

    def test_options_refused(self):
        t, y, yerr, _ = _photometry(n=50)
        with pytest.raises(ValueError, match="core selects"):
            make_transit_target(t, y, yerr, T0, PER, limb_dark="hybrid4",
                                core="graph")
        with pytest.raises(ValueError, match="exp_time needs a hybrid law"):
            make_transit_target(t, y, yerr, T0, PER, exp_time=0.02)


@pytest.fixture(scope="module")
def spotted():
    return _target("hybrid4", spots=True)


class TestSpottedTarget:
    def test_model_is_the_frontend(self, spotted):
        """relative (the default): the frontend's flux over its flux with
        the planet half an orbit away; out of transit exactly 0."""
        tt, flux = spotted
        assert not tt.compile_step and tt.n_spots == 2
        assert abs(tt.spot_epoch - EP) < 1e-9 and tt.x64.shape[0] == 3
        m = _model64(tt, _truth(tt))
        assert np.max(np.abs(m - (flux - 1.0))) < 1e-11
        oot = np.abs(tt.x64[0]) > 0.1
        assert np.max(np.abs(m[oot])) < 1e-15

    def test_unspotted_normalisation(self):
        tt, flux = _target("hybrid4", spots=True, relative=False)
        assert np.max(np.abs(_model64(tt, _truth(tt)) - (flux - 1.0))) < 1e-11

    def test_chi2_precision_and_gradient(self, spotted):
        tt, _ = spotted
        v = _truth(tt)
        with mx.stream(mx.cpu):
            lp = float(np.array(tt.loglike.hi(mx.array(v[None], dtype=mx.float64)))[0])
        assert abs(-2.0 * (lp + tt.loglike.log_offset_const) / tt.loglike.n_data - 1.0) < 0.1
        u0 = tt.transform.from_model_np(v)
        rng = np.random.default_rng(1)
        u = mx.array((u0 + 0.003 * rng.standard_normal((8, u0.size))).astype(np.float32))
        assert engine.validate_precision(tt.target, u).max_abs_err < 0.1
        g = np.array(mx.grad(lambda uu: mx.sum(tt.target.log_prob(uu)))(u))
        assert np.isfinite(g).all() and (np.abs(g).max(0) > 0).all()

    def test_default_epoch_and_bounds(self):
        t, y, yerr, _ = _photometry(spots=True, n=50)
        tt = make_transit_target(t, y, yerr, T0, PER, limb_dark="hybrid4",
                                 n_spots=2, bounds={"prot": (5.0, 15.0),
                                                    "rad": (2.0, 20.0),
                                                    "rad2": (3.0, 25.0)})
        # the predicted mid-transit nearest the data's middle
        assert abs(tt.spot_epoch - (T0 + PER)) < 1e-9
        lo = dict(zip(tt.param_names, tt.transform.model_np(
            np.full(len(tt.param_names), -50.0))))
        assert abs(lo["rad1"] - 2.0) < 1e-6 and abs(lo["rad2"] - 3.0) < 1e-6
        with pytest.raises(ValueError, match="'prot' has no default box"):
            make_transit_target(t, y, yerr, T0, PER, limb_dark="hybrid4", n_spots=1)
        with pytest.raises(ValueError, match="unknown parameter"):
            make_transit_target(t, y, yerr, T0, PER, limb_dark="hybrid4", n_spots=1,
                                bounds={"prot": (5.0, 15.0), "radius": (1.0, 2.0)})

    def test_compile_refused_loudly(self, spotted):
        tt, _ = spotted
        u = mx.array(tt.transform.from_model_np(_truth(tt))[None].astype(np.float32))
        with pytest.raises(ValueError, match="mx.compile"):
            mx.eval(mx.compile(tt.target.log_prob)(u))


@pytest.mark.slow
def test_chees_smoke_on_a_spotted_star(spotted):
    """A short ChEES-HMC run, uncompiled as the target asks: no
    divergences, every parameter within 4 sd of the truth it started at."""
    tt, _ = spotted
    v = _truth(tt)
    u0 = tt.transform.from_model_np(v)
    rng = np.random.default_rng(2)
    u = mx.array((u0 + 0.002 * rng.standard_normal((64, u0.size))).astype(np.float32))
    res = engine.run(engine.ChEESHMC(tt.target, max_leapfrog=32), tt.target, u,
                     n_warmup=60, n_samples=40, seed=1, compile_step=tt.compile_step)
    assert res.extras.get("n_divergent", 0) == 0
    draws = tt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    assert (np.abs(draws.mean(0) - v) < 4.0 * (draws.std(0) + 1e-12)).all()
