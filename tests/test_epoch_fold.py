"""A float32 model's per-point epoch fold: absolute mission times over any
baseline at float32's accuracy.

Each time is split on the host, in float64, into its orbit number k and
its residual d from its own transit of the build ephemeris (t0_b, per_b);
a call at (t0, per) evaluates d - k (per - per_b) (TransitModel._set_fold).
Without it a float32 light curve's error grew with the span of the data
(1.4e-5 over 900 d, 3.4e-5 over 2000 d at raw BJD), since the times and
the period's float32 rounding entered the phase k orbits from t0.
"""

import copy
import itertools
import math

import mlx.core as mx
import numpy as np
import pytest

import elome

# float32's own error near t0, which the fold now holds at any span: up to
# 2.6e-7 for the quadratic kernel at rp = 0.1 (2.4e-7 at a 10 d span
# before the fold), well under for the hybrid laws
TOL = 3e-7
EXP = 40.0 / 86400.0


def params(t0=2500.37, per=3.0, b=0.3, ecc=0.0, law="quadratic",
           u=(0.4804, 0.1867), a=None):
    p = elome.TransitParams()
    p.t0, p.per, p.rp = t0, per, 0.1
    p.a = a if a is not None else max(3.0, 9.0 * (per / 3.0) ** (2.0 / 3.0))
    p.inc = math.degrees(math.acos(b / p.a))
    p.ecc, p.w, p.fp = ecc, 40.0, 1e-3
    p.limb_dark, p.u = law, list(u)
    return p


def err(p, t, **kw):
    f32 = elome.TransitModel(p, t, dtype=mx.float32, **kw).light_curve(p)
    f64 = elome.TransitModel(p, t, dtype=mx.float64, **kw).light_curve(p)
    return float(np.max(np.abs(f32 - f64)))


def test_the_error_does_not_grow_with_the_span():
    """Spans of 10 d to 4000 d, at offsets 0, 2500 d and raw BJD, for
    short, medium and long periods, central and grazing chords, circular
    and eccentric orbits, supersampled and under the contact rule."""
    worst = 0.0
    for per, b, ecc, span, lo, integ in itertools.product(
            (0.3, 3.0, 30.0), (0.0, 0.8), (0.0, 0.3), (10.0, 2000.0, 4000.0),
            (0.0, 2500.0, 2.46e6), ("supersample", "contact")):
        t = lo + np.linspace(0.0, span, 40_000)
        kw = dict(integration=integ, exp_time=EXP)
        if integ == "supersample":
            kw["supersample_factor"] = 3
        e = err(params(t0=lo + 0.37 * per, per=per, b=b, ecc=ecc), t, **kw)
        assert e < TOL, (per, b, ecc, span, lo, integ, e)
        worst = max(worst, e)
    assert worst > 0.0


@pytest.mark.parametrize("case", ["hybrid2", "secondary", "oblate", "ring"])
def test_every_periodic_route_folds(case):
    t = 2500.0 + np.linspace(0.0, 2000.0, 100_000)
    kw = {}
    if case == "secondary":
        p = params()
        kw["transittype"] = "secondary"
    else:
        p = params(law="hybrid2", u=(0.5, 0.3))
    if case == "oblate":
        p.f, p.theta = 0.2, 20.0
    if case == "ring":
        p.ring_in, p.ring_out, p.ring_f, p.theta, p.ring_tau = 1.5, 2.2, 0.3, 20.0, 1.0
    assert elome.TransitModel(p, t, dtype=mx.float32, **kw)._fold is not None
    assert err(p, t, **kw) < TOL


def test_float64_and_the_aperiodic_routes_keep_their_times():
    """float64 keeps its zero offset; a spotted star turns with prot and a
    moon's route has its own references, so neither folds on per."""
    t = 2500.0 + np.linspace(0.0, 20.0, 1000)
    p = params()
    assert elome.TransitModel(p, t, dtype=mx.float64)._fold is None
    s = params(law="hybrid2", u=(0.5, 0.3))
    s.spots, s.prot = [[10.0, 0.0, 10.0, 0.5]], 11.0
    assert elome.TransitModel(s, t, dtype=mx.float32)._fold is None
    mo = params(law="hybrid2", u=(0.5, 0.3))
    mo.moon_r, mo.moon_a, mo.moon_per, mo.moon_q = 0.35, 6.0, 0.9, 0.02
    assert elome.TransitModel(mo, t, dtype=mx.float32)._fold is None


@pytest.mark.parametrize("case", ["quadratic", "eccentric", "hybrid2", "contact",
                                  "secondary"])
def test_a_changed_ephemeris_matches_a_model_built_there(case):
    t = 2500.0 + np.linspace(0.0, 2000.0, 100_000)
    kw, pk = {}, {}
    if case == "eccentric":
        pk["ecc"] = 0.3
    if case == "hybrid2":
        pk.update(law="hybrid2", u=(0.5, 0.3))
    if case == "contact":
        kw.update(integration="contact", exp_time=EXP)
    if case == "secondary":
        kw["transittype"] = "secondary"
    p = params(**pk)
    m = elome.TransitModel(p, t, dtype=mx.float32, **kw)
    q = copy.copy(p)
    q.t0, q.per = p.t0 + 1e-3, p.per + 1e-6
    ref = elome.TransitModel(q, t, dtype=mx.float64, **kw).light_curve(q)
    assert np.max(np.abs(m.light_curve(q) - ref)) < TOL
    assert m._fold == (p.t0, p.per)          # a small change keeps the fold


def test_a_distant_ephemeris_refolds():
    """k (per - per_b) past a quarter period: the model refolds on the
    call's ephemeris rather than carry it into float32."""
    t = 2500.0 + np.linspace(0.0, 2000.0, 100_000)
    p = params()
    m = elome.TransitModel(p, t, dtype=mx.float32)
    q = copy.copy(p)
    q.t0, q.per = 2500.9, 3.07
    ref = elome.TransitModel(q, t, dtype=mx.float64).light_curve(q)
    assert np.max(np.abs(m.light_curve(q) - ref)) < TOL
    assert m._fold == (2500.9, 3.07)
    assert np.max(np.abs(m.light_curve(p) - elome.TransitModel(
        p, t, dtype=mx.float64).light_curve(p))) < TOL


@pytest.mark.parametrize("case", ["quadratic", "hybrid2", "contact", "eccentric",
                                  "oblate", "ring"])
def test_light_curves_fold_each_set(case):
    t = 2500.0 + np.linspace(0.0, 2000.0, 50_000)
    kw, pk = {}, {}
    if case == "hybrid2":
        pk.update(law="hybrid2", u=(0.5, 0.3))
    if case == "contact":
        kw.update(integration="contact", exp_time=EXP)
    if case == "eccentric":
        pk["ecc"] = 0.2
    if case in ("oblate", "ring"):
        pk.update(law="hybrid2", u=(0.5, 0.3))
    sets = []
    for j in range(6):
        q = params(**pk)
        q.t0, q.per = q.t0 + 1e-3 * j, q.per + 2e-6 * (j - 3)
        if case == "oblate":
            q.f, q.theta = 0.2, 20.0
        if case == "ring":
            q.ring_in, q.ring_out, q.ring_f, q.theta, q.ring_tau = 1.5, 2.2, 0.3, 20.0, 1.0
        sets.append(q)
    m = elome.TransitModel(sets[0], t, dtype=mx.float32, **kw)
    ref = elome.TransitModel(sets[0], t, dtype=mx.float64, **kw).light_curves(sets)
    got = m.light_curves(sets)
    assert np.max(np.abs(got - ref)) < 1.5 * TOL
    if case in ("quadratic", "contact", "eccentric"):
        # bit for bit where both take the same route (a circular hybrid
        # light_curve takes the graph, light_curves the kernel)
        one = np.stack([m.light_curve(q) for q in sets])
        assert np.array_equal(got, one)


@pytest.mark.parametrize("case", ["quadratic", "eccentric", "hybrid2", "contact"])
def test_gradients_in_t0_and_per_over_years(case):
    """d/dt0 and d/dper over 2000 d: float32 against float64. The period's
    gradient is the sensitive one, k multiplying it; a float32 array
    period stays exact (per - hi - lo) and in the graph."""
    t = 2500.0 + np.linspace(0.0, 2000.0, 100_000)
    kw, pk = {}, {}
    if case == "eccentric":
        pk["ecc"] = 0.3
    if case == "hybrid2":
        pk.update(law="hybrid2", u=(0.5, 0.3))
    if case == "contact":
        kw.update(integration="contact", exp_time=EXP)
    p = params(**pk)
    w = np.random.default_rng(1).standard_normal(t.size)

    def grads(dtype, per_dtype):
        m = elome.TransitModel(p, t, dtype=dtype, **kw)
        W = mx.array(w, dtype=dtype)

        def loss(t0, per):
            q = copy.copy(p)
            q.t0, q.per = t0, per
            return mx.sum(m.light_curve_mx(q).astype(mx.float64) * W.astype(mx.float64))
        with mx.stream(mx.cpu):
            gt = mx.grad(lambda x: loss(x, p.per))(mx.array(p.t0, dtype=mx.float64))
            mx.eval(gt)
        if per_dtype == mx.float64:
            with mx.stream(mx.cpu):
                gp = mx.grad(lambda x: loss(p.t0, x))(mx.array(p.per, dtype=mx.float64))
                mx.eval(gp)
        else:
            gp = mx.grad(lambda x: mx.sum(
                m.light_curve_mx(_with(p, per=x)) * W))(mx.array(p.per, dtype=mx.float32))
        return float(gt), float(gp)

    g64 = grads(mx.float64, mx.float64)
    for per_dtype in (mx.float32, mx.float64):
        g32 = grads(mx.float32, per_dtype)
        assert abs(g32[0] / g64[0] - 1.0) < 3e-4, (per_dtype, g32, g64)
        assert abs(g32[1] / g64[1] - 1.0) < 3e-4, (per_dtype, g32, g64)


def _with(p, **kw):
    q = copy.copy(p)
    for k, v in kw.items():
        setattr(q, k, v)
    return q
