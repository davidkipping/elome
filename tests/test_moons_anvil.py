"""make_transit_target with a moon: the parameter layout, the model against
TransitModel, exposures, the two precisions and gradients."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome
from elome.anvil import import_engine, make_transit_target, transit_param_names
from elome.ld import hybrid4_to_q_np, u_to_q_np

engine, _ = import_engine()
T0, PER, RP, A, INC = 2500.123456, 12.345, 0.08, 25.0, 89.4
MOON = dict(moon_r=0.35, moon_a=6.0, moon_per=0.9, moon_q=0.02, moon_inc=88.0,
            moon_node=10.0, moon_phase=40.0)
LAWS = {"quadratic": [0.4, 0.25], "hybrid4": [0.2, 0.25, 0.1, 0.05]}
EXP = 30.0 / 1440.0
BOUNDS = {"moon_p_off": (-0.05, 0.05)}


def _params(law, ecc=0.0, w=90.0, moon_ecc=0.0, moon_w=0.0):
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, RP, A, INC, ecc, w
    p.limb_dark, p.u = law, LAWS[law]
    for k, v in MOON.items():
        setattr(p, k, v)
    p.moon_ecc, p.moon_w = moon_ecc, moon_w
    return p


def _target(law="quadratic", ecc=0.0, w=90.0, moon_ecc=0.0, moon_w=0.0, n=300,
            yerr=2e-4, **kw):
    """Three transits through TransitModel, and the target built on them."""
    rng = np.random.default_rng(1)
    t = np.sort(np.concatenate([T0 + k * PER + rng.uniform(-0.45, 0.45, n)
                                for k in range(3)]))
    p = _params(law, ecc, w, moon_ecc, moon_w)
    mkw = {}
    if kw.get("exp_time"):
        rule = kw.get("integration", "contact")
        mkw = (dict(exp_time=kw["exp_time"], integration="contact", n_gl=kw.get("n_gl", 5))
               if rule == "contact" else
               dict(exp_time=kw["exp_time"], supersample_factor=kw.get("n_sub", 1)))
    flux = elome.TransitModel(p, t, **{"dtype": mx.float64, **mkw}).light_curve(p)
    tt = make_transit_target(t, flux + yerr * rng.standard_normal(t.size), yerr,
                             T0 + 0.003, PER - 0.0005, limb_dark=law, moon=True,
                             moon_period_guess=MOON["moon_per"] + 0.0002,
                             eccentric=ecc > 0.0, moon_eccentric=moon_ecc > 0.0,
                             bounds=BOUNDS, **kw)
    return tt, flux


def _truth(tt, ecc=0.0, w=90.0, moon_ecc=0.0, moon_w=0.0):
    b = A * math.cos(math.radians(INC))
    kw = dict(r=RP, a=A, b=b)
    if tt.eccentric:
        kw["b"] = b * (1.0 - ecc ** 2) / (1.0 + ecc * math.sin(math.radians(w)))
        kw["secosw"] = math.sqrt(ecc) * math.cos(math.radians(w))
        kw["sesinw"] = math.sqrt(ecc) * math.sin(math.radians(w))
    u = LAWS[tt.limb_dark]
    q = u_to_q_np(*u) if tt.limb_dark == "quadratic" else hybrid4_to_q_np(np.array(u))
    kw.update({f"q{j + 1}": float(v) for j, v in enumerate(q)})
    kw.update(moon_r=MOON["moon_r"], moon_a=MOON["moon_a"], moon_per=MOON["moon_per"],
              moon_q=MOON["moon_q"], cos_moon_inc=math.cos(math.radians(MOON["moon_inc"])),
              moon_node=MOON["moon_node"], moon_phase=MOON["moon_phase"])
    if "moon_secosw" in tt.param_names:
        kw["moon_secosw"] = math.sqrt(moon_ecc) * math.cos(math.radians(moon_w))
        kw["moon_sesinw"] = math.sqrt(moon_ecc) * math.sin(math.radians(moon_w))
    return tt.model_params(t0=T0, period=PER, **kw)


def _model(tt, v, dtype=mx.float64):
    if dtype == mx.float64:
        with mx.stream(mx.cpu):
            return np.array(tt.model_fn(mx.array(v[None], dtype=dtype),
                                        mx.array(tt.x64, dtype=dtype)))[0]
    return np.array(tt.model_fn(mx.array(v[None], dtype=dtype),
                                mx.array(tt.x64, dtype=dtype)), dtype=np.float64)[0]


class TestLayout:
    def test_names(self):
        names = transit_param_names("hybrid4", eccentric=True, moon=True,
                                    moon_eccentric=True)
        assert names[9:11] == ["secosw", "sesinw"]
        assert names[11:] == ["moon_r", "moon_a", "moon_p_off", "moon_q", "cos_moon_inc",
                              "moon_node", "moon_phase", "moon_secosw", "moon_sesinw",
                              "df0"]

    @pytest.mark.parametrize("kw,match", [
        (dict(moon_eccentric=True), "moon=True"),
        (dict(moon=True, n_spots=1, limb_dark="hybrid4"), "unspotted"),
        (dict(moon=True, transittype="both"), "primary"),
    ])
    def test_bad_layouts(self, kw, match):
        with pytest.raises(ValueError, match=match):
            transit_param_names(**kw)

    def test_target_checks(self):
        t = T0 + np.linspace(-0.3, 0.3, 50)
        y = np.ones_like(t)
        with pytest.raises(ValueError, match="moon_period_guess"):
            make_transit_target(t, y, 1e-4, T0, PER, moon=True, bounds=BOUNDS)
        with pytest.raises(ValueError, match="moon=True"):
            make_transit_target(t, y, 1e-4, T0, PER, moon_period_guess=1.0)
        with pytest.raises(ValueError, match="no default box"):
            make_transit_target(t, y, 1e-4, T0, PER, moon=True, moon_period_guess=1.0)


class TestModel:
    @pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
    def test_model_is_transit_model(self, law):
        tt, flux = _target(law)
        assert not tt.compile_step and tt.x64.shape[0] == 3
        assert np.max(np.abs(_model(tt, _truth(tt)) - (flux - 1.0))) < 1e-12
        assert flux.min() < 1.0 - 5e-3

    def test_eccentric_planet_and_moon(self):
        orb = dict(ecc=0.2, w=63.0, moon_ecc=0.1, moon_w=200.0)
        tt, flux = _target("hybrid4", **orb)
        assert np.max(np.abs(_model(tt, _truth(tt, **orb)) - (flux - 1.0))) < 1e-12

    @pytest.mark.parametrize("kw", [dict(exp_time=EXP),
                                    dict(exp_time=EXP, integration="supersample", n_sub=5)])
    def test_exposures(self, kw):
        tt, flux = _target("quadratic", **kw)
        assert np.max(np.abs(_model(tt, _truth(tt)) - (flux - 1.0))) < 1e-12

    def test_model_params_and_report(self):
        tt, _ = _target()
        v = _truth(tt)
        assert v[tt.param_names.index("moon_p_off")] == pytest.approx(-0.0002, abs=1e-15)
        rep = tt.transform.to_physical(v[None])[0]
        assert rep[tt.param_names.index("moon_p_off")] == pytest.approx(MOON["moon_per"],
                                                                        abs=1e-14)

    @pytest.mark.parametrize("law,tol", [("quadratic", 5e-7), ("hybrid4", 1e-7)])
    def test_float32(self, law, tol):
        tt, _ = _target(law, exp_time=EXP)
        v = _truth(tt)
        assert np.max(np.abs(_model(tt, v, mx.float32) - _model(tt, v))) < tol

    def test_many_chains_at_once(self):
        tt, _ = _target("hybrid4", exp_time=EXP)
        v = _truth(tt)
        rng = np.random.default_rng(3)
        V = v[None] + 1e-3 * rng.standard_normal((4, v.size)) * (np.abs(v) + 0.1)
        with mx.stream(mx.cpu):
            F = np.array(tt.model_fn(mx.array(V, dtype=mx.float64),
                                     mx.array(tt.x64, dtype=mx.float64)))
        G = np.stack([_model(tt, x) for x in V])
        assert np.max(np.abs(F - G)) < 1e-15


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_log_prob_gradients_are_finite(dtype):
    tt, _ = _target("hybrid4", ecc=0.2, w=63.0, moon_ecc=0.1, moon_w=200.0, exp_time=EXP)
    u0 = tt.transform.from_model_np(_truth(tt, 0.2, 63.0, 0.1, 200.0))
    rng = np.random.default_rng(4)
    U = (u0[None] + 0.01 * rng.standard_normal((3, u0.size))).astype(
        np.float64 if dtype == mx.float64 else np.float32)
    ctx = mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.default_stream(
        mx.default_device()))
    with ctx:
        lp, g = mx.value_and_grad(lambda u: mx.sum(tt.target.log_prob(u)))(
            mx.array(U, dtype=dtype))
        assert np.isfinite(float(lp)) and np.all(np.isfinite(np.array(g)))


@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
def test_finite_across_the_prior_box(law):
    """Draws from the whole box -- a moon inside the planet, edge-on orbits,
    eccentric moons -- give finite fluxes and gradients."""
    tt, _ = _target(law, ecc=0.2, w=63.0, moon_ecc=0.1, moon_w=200.0, exp_time=EXP)
    U = tt.transform.from_model_np(tt.transform.sample_prior_model(
        64, np.random.default_rng(5))).astype(np.float32)
    lp, g = mx.value_and_grad(lambda u: mx.sum(tt.target.log_prob(u)))(mx.array(U))
    assert np.isfinite(float(lp)) and np.all(np.isfinite(np.array(g)))


@pytest.mark.parametrize("law", ["quadratic", "hybrid4"])
@pytest.mark.parametrize("exp_time", [0.0, EXP], ids=["none", "contact"])
def test_moon_r_zero_at_the_box_edge(law, exp_time):
    """moon_r = 0, the box's edge (and below float32's range): finite
    gradients for every chain in either precision."""
    from elome.anvil import make_moon_transit_flux
    from elome.moons import moon_turns_ref
    from elome.orbit import epoch_center_times
    per, mper = 12.345, 0.9
    ep = epoch_center_times(np.linspace(-0.3, 0.3, 40), 0.0, per)
    x = np.vstack([ep[0], ep[1], moon_turns_ref(ep[1], per, mper)])
    fn = make_moon_transit_flux(per, mper, law, exp_time=exp_time)
    vals = dict(t0_off=0.0, p_off=0.0, r=0.08, b=0.2, a=25.0, moon_r=0.0, moon_a=6.0,
                moon_p_off=0.0, moon_q=0.02, cos_moon_inc=0.05, moon_node=10.0,
                moon_phase=0.0, df0=1.0)
    names = transit_param_names(law, moon=True)
    V = np.array([[vals.get(n, 0.3) for n in names]] * 2)
    V[1, names.index("moon_r")] = 1e-32
    for dt, st in ((mx.float64, mx.cpu), (mx.float32, mx.gpu)):
        with mx.stream(st):
            g = mx.grad(lambda th: mx.sum(fn(th, mx.array(x, dtype=dt))))(mx.array(V, dtype=dt))
        assert np.all(np.isfinite(np.array(g))), dt


def test_float32_far_from_the_reference_periods():
    """1000 orbits from t0 with the moon's period up to 0.2 d off its
    reference (a sampler's aliases): k d_ratio, hundreds of turns, in
    float-float arithmetic keeps float32 at its photometry's level (it was
    up to 8.7e-6 off)."""
    from elome.anvil import make_moon_transit_flux
    from elome.moons import moon_turns_ref
    from elome.orbit import epoch_center_times
    per, mper = 12.345, 0.9
    t = np.concatenate([k * per + np.linspace(-0.25, 0.25, 60) for k in (0, 10, 400, 1000)])
    ep = epoch_center_times(t, 0.0, per)
    x = np.vstack([ep[0], ep[1], moon_turns_ref(ep[1], per, mper)])
    names = transit_param_names("hybrid4", moon=True)
    fn = make_moon_transit_flux(per, mper, "hybrid4")
    vals = dict(t0_off=0.0, r=0.08, b=0.2, a=25.0, moon_r=0.4, moon_a=6.0, moon_q=0.02,
                cos_moon_inc=0.05, moon_node=10.0, moon_phase=40.0, df0=1.0)
    V = []
    for moff in (0.0, 0.01, 0.05, 0.2):
        for poff in (0.0, 1e-3):
            V.append([{**vals, "moon_p_off": moff, "p_off": poff}.get(n, 0.3) for n in names])
    V = np.array(V, dtype=np.float32).astype(np.float64)      # the same values in both
    with mx.stream(mx.cpu):
        f64 = np.array(fn(mx.array(V, dtype=mx.float64), mx.array(x, dtype=mx.float64)))
    f32 = np.array(fn(mx.array(V, dtype=mx.float32), mx.array(x, dtype=mx.float32)))
    assert np.abs(f32 - f64).max() < 1e-7                      # measured 4.2e-8


def test_reference_periods_are_checked():
    from elome.anvil import make_moon_transit_flux
    for per, mper in ((10.0, 0.0), (10.0, -1.0), (10.0, np.nan), (0.0, 1.0)):
        with pytest.raises(ValueError, match="finite and > 0"):
            make_moon_transit_flux(per, mper)


@pytest.mark.slow
def test_chees_smoke_with_a_moon():
    """A short ChEES-HMC run, uncompiled as the target asks: no divergences,
    every parameter within 4 sd of the truth it started at."""
    tt, _ = _target("quadratic")
    v = _truth(tt)
    u0 = tt.transform.from_model_np(v)
    rng = np.random.default_rng(2)
    u = mx.array((u0 + 0.002 * rng.standard_normal((32, u0.size))).astype(np.float32))
    res = engine.run(engine.ChEESHMC(tt.target, max_leapfrog=32), tt.target, u,
                     n_warmup=60, n_samples=40, seed=1, compile_step=tt.compile_step)
    assert res.extras.get("n_divergent", 0) == 0
    draws = tt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
    assert (np.abs(draws.mean(0) - v) < 4.0 * (draws.std(0) + 1e-12)).all()
