"""The fp32 Metal kernels for starspots (metal_spots) against the fp64
graph: per point (columns and dual-number gradients, on the geometry
cases, at random and on the degenerate lanes), as light curves on both
exposure rules and orbits, in their gradients, in the contact rule's
event search, and the routing."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome as el
from elome import metal as M
from elome import spots
from elome.hybrid import HYBRID2, HYBRID4, HYBRID5, get_law, hybrid_norms

pytestmark = pytest.mark.skipif(not M.metal_available(), reason="needs Metal")

import test_spots as TS                     # noqa: E402  (the geometry cases)
import test_spots_api as TA                 # noqa: E402  (the model cases)

D = math.radians


def ref64(law, X, Y, rp, n, al, c, wct):
    """B and the gradient of wct . B in (X, Y, rp, then per spot n, alpha,
    contrast) by the fp64 graph and its autodiff."""
    K = n.shape[0]
    f = lambda v: mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)
    with mx.stream(mx.cpu):
        def F(X, Y, rp, n, al, c):
            B = spots._shape_cols(law, K)(
                X, Y, rp, *[v for k in range(K) for v in (n[k, 0], n[k, 1], n[k, 2], al[k])],
                *[c[k] for k in range(K)])
            return mx.sum(B * f(wct)), B
        args = [f(X), f(Y), f(rp), f(n), f(al), f(c)]
        B = np.array(F(*args)[1])
        g = mx.grad(lambda *a: F(*a)[0], argnums=list(range(6)))(*args)
        out = [g[0].item(), g[1].item(), g[2].item()]
        for k in range(K):
            out += list(np.array(g[3][k])) + [g[4][k].item(), g[5][k].item()]
    return B, np.array(out)


def tangent(g, n):
    """The n-gradients projected on the sphere: only those are used (the
    kernel takes (a^2 - b^2) / 4 on the sphere, so its off-sphere component
    is not the graph's)."""
    g = np.array(g, dtype=np.float64)
    for k in range(n.shape[0]):
        v = g[3 + 5 * k:6 + 5 * k]
        g[3 + 5 * k:6 + 5 * k] = v - (v @ n[k]) * n[k]
    return g


def errors(law, X, Y, rp, n, al, c, seed=0):
    """(column error / norms, gradient error / its scale), kernel vs fp64."""
    from elome import metal_spots as MS
    wct = np.random.default_rng(seed).normal(size=law.n_col)
    B32, g32 = MS._point_cols([X], [Y], rp, n[None], al, c, law, wct=wct, grad=True)
    B64, g64 = ref64(law, X, Y, rp, n, al, c, wct)
    assert np.all(np.isfinite(B32)) and np.all(np.isfinite(g32))
    g32, g64 = tangent(g32[0], n), tangent(g64, n)
    return (np.max(np.abs(B32[0] - B64) / hybrid_norms(law)),
            np.max(np.abs(g32 - g64)) / np.max(np.abs(g64)))


def case_arrays(case):
    (X, Y, rp), sp = case
    n = np.array([spots.spot_vector(D(lo), D(la)) for lo, la, _, _ in sp])
    return X, Y, rp, n, np.array([D(s[2]) for s in sp]), np.array([s[3] for s in sp])


# ---------------------------------------------------------------------------
# per point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", [HYBRID5, HYBRID4, HYBRID2], ids=lambda L: L.name)
@pytest.mark.parametrize("name", list(TS.CASES))
def test_point_columns_and_gradients(name, law):
    """Columns to the fp32 floor (hybrid5's innermost pole, (mu^2 + eps)^-2
    with eps = 0.0016, on a thin spot at the limb: 3.3e-7 of its norm, the
    fp32 graph's 1.5e-7), the dual-number gradient to 1e-5 of its scale."""
    eB, eg = errors(law, *case_arrays(TS.CASES[name]), seed=len(name))
    assert eB < 5e-7
    assert eg < 1e-5


def test_random_configurations():
    """A planet near one of one to three random spots: the fp32 floor is
    a rim grazing the limb or the planet straddling it under hybrid5's
    innermost pole (1.5e-6 of a column norm, 7e-5 of a gradient)."""
    rng = np.random.default_rng(0)
    worst = [0.0, 0.0]
    for _ in range(120):
        K = int(rng.integers(1, 4))
        n = spots.spot_vector(np.radians(rng.uniform(-100, 100, K)),
                              np.radians(rng.uniform(-60, 60, K)))
        al, c = np.radians(rng.uniform(2, 25, K)), rng.uniform(0.2, 0.9, K)
        j = int(rng.integers(K))
        X, Y = n[j, 0] + rng.normal(0, 0.1), n[j, 1] + rng.normal(0, 0.1)
        e = errors(HYBRID5, X, Y, rng.uniform(0.03, 0.2), n, al, c, seed=int(rng.integers(99)))
        worst = [max(w, v) for w, v in zip(worst, e)]
    assert worst[0] < 3e-6
    assert worst[1] < 2e-4


@pytest.mark.parametrize("off", [1e-3, 1e-5, 0.0])
def test_spot_at_the_disc_centre(off):
    """A spot's rim frame turns as 1 / |n_xy| near the disc centre: the
    kernel and the fp32 graph take the turn and (a^2 - b^2) / 4 in forms
    that stay finite there, so the gradient in the spot's longitude and
    latitude is fp64's to 2e-5, at the centre itself included."""
    from elome import metal_spots as MS
    law = HYBRID5
    wct = np.random.default_rng(3).normal(size=law.n_col)
    lon, lat = off, 0.3 * off

    def graph(dt, stream):
        with mx.stream(stream):
            def F(lon, lat):
                n = spots.spot_vector(lon, lat)
                c = lambda v: mx.array(v, dtype=dt)
                B = spots._shape_cols(law, 1)(c(0.05), c(0.02), c(0.1), n[0], n[1], n[2],
                                              c(0.15), c(0.4))
                return mx.sum(B * mx.array(wct, dtype=dt))
            g = mx.grad(F, argnums=[0, 1])(mx.array(lon, dtype=dt), mx.array(lat, dtype=dt))
            return np.array([g[0].item(), g[1].item()])
    g64 = graph(mx.float64, mx.cpu)
    g32 = graph(mx.float32, mx.gpu)
    n = spots.spot_vector(lon, lat)
    dn = np.array([[math.cos(lat) * math.cos(lon), 0.0, -math.cos(lat) * math.sin(lon)],
                   [-math.sin(lat) * math.sin(lon), math.cos(lat), -math.sin(lat) * math.cos(lon)]])
    _, gk = MS._point_cols([0.05], [0.02], 0.1, n[None, None], [0.15], [0.4], law, wct=wct,
                           grad=True)
    gk = dn @ gk[0][3:6]
    assert abs(g64[0]) > 0.09                       # the centre lane is not dropped
    assert np.max(np.abs(gk - g64)) < 2e-5 * np.max(np.abs(g64))
    assert np.max(np.abs(g32 - g64)) < 1e-4 * np.max(np.abs(g64))


DEGENERATE = {
    "spot at the centre": ((0.05, 0.02, 0.1), (0.0, 0.0, 1.0), 0.15),
    "spot centred on the limb": ((0.95, 0.0, 0.1), (1.0, 0.0, 0.0), 0.1),
    "rim touching the limb": ((0.95, 0.0, 0.1), (math.cos(0.1), 0.0, math.sin(0.1)), 0.1),
    "rim touching the limb from behind": ((0.95, 0.0, 0.1), (math.cos(0.1), 0.0, -math.sin(0.1)), 0.1),
    "planet at the centre": ((0.0, 0.0, 0.1), (0.3, 0.2, math.sqrt(0.87)), 0.15),
    "planet on a centred rim": ((0.0, 0.0, math.sin(0.15)), (0.0, 0.0, 1.0), 0.15),
    "spot behind the star": ((0.3, 0.0, 0.1), (0.3, 0.0, -math.sqrt(0.91)), 0.15),
    "hemisphere": ((0.3, 0.0, 0.1), (0.6, 0.0, 0.8), 0.5 * math.pi),
    "spot of 0.1 degree": ((0.3, 0.2, 0.1), (0.3, 0.25, math.sqrt(1 - 0.1525)), D(0.1)),
}


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_degenerate_geometries(name):
    """Finite and at the fp32 floor where the graph switches branches: the
    centred rim, the rim's limb points merging, a great-circle rim (whose
    y0 = 0 does not make it centred), a cap wholly behind the star."""
    (X, Y, rp), nv, al = DEGENERATE[name]
    n = np.array([nv]) / np.linalg.norm(nv)
    eB, eg = errors(HYBRID5, X, Y, rp, n, np.array([al]), np.array([0.4]), seed=len(name))
    assert eB < 3e-7
    assert eg < 1e-5


def test_inactive_points_are_zero():
    from elome.metal_spots import spot_cols_kernel
    with mx.stream(mx.gpu):
        X = mx.array([0.1, 0.2, 0.3], dtype=mx.float32)
        n = mx.array(spots.spot_vector(np.array([0.1]), np.array([0.2])), dtype=mx.float32)
        act = mx.array([True, False, True])

        def f(X):
            return mx.sum(spot_cols_kernel(X, 0.0, 0.1, n, 0.2, 0.5, HYBRID4, active=act))
        B = spot_cols_kernel(X, 0.0, 0.1, n, 0.2, 0.5, HYBRID4, active=act)
        g = mx.grad(f)(X)
        assert np.all(np.array(B)[1] == 0.0) and np.all(np.array(B)[0] != 0.0)
        assert np.array(g)[1] == 0.0 and np.array(g)[0] != 0.0


# ---------------------------------------------------------------------------
# light curves
# ---------------------------------------------------------------------------

P, LAT, EXP = TA.P, TA.LAT_CHORD, TA.EXP
MODELS = {
    "two spots": dict(),
    "spot smaller than the planet": dict(spots=[[8.0, LAT, 5.0, 0.3], [40.0, 15.0, 10.0, 0.6]],
                                         prot=4.0),
    "fast oblique rotator": dict(spots=[[8.0, LAT, 12.0, 0.4]], prot=1.0, lam=30.0,
                                 istar=70.0),
    "eccentric": dict(spots=[[-5.0, LAT, 4.0, 0.2]], prot=3.0, ecc=0.3, w=50.0),
}
RULES = {"none": dict(), "contact": dict(integration="contact", exp_time=EXP)}
TT = np.linspace(-0.12, 0.12, 121)


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("name", list(MODELS))
def test_kernel_matches_the_fp64_model(name, rule):
    p = P(**MODELS[name])
    ref = el.TransitModel(p, TT, **RULES[rule]).light_curve(p)
    with mx.stream(mx.gpu):
        got = el.TransitModel(p, TT, dtype=mx.float32, **RULES[rule]).light_curve(p)
    assert 1.0 - ref.min() > 0.01
    assert np.abs(got - ref).max() < 2e-7


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("name", ["spot smaller than the planet", "eccentric"])
def test_gradients_match_fp64(name, rule):
    """light_curve_mx's gradient in the spot table, the weights, the orbit
    and the star, through the kernels' VJP and the contact rule's live
    event times: fp64's to 1e-5 of its scale (t0 is the fp32 time floor,
    the graph's as well)."""
    names = ["rp", "inc", "a", "t0", "per", "prot", "istar", "lam", "spot_epoch"]
    p0 = P(**MODELS[name], istar=70.0, lam=20.0, spot_epoch=0.01)
    t = np.linspace(-0.08, 0.08, 41)
    wts = np.random.default_rng(2).normal(size=t.size)

    def grads(dt, stream):
        m = el.TransitModel(p0, t, dtype=dt, **RULES[rule])

        def F(sp, u, *vals):
            p = P(**MODELS[name])
            for k, v in zip(names, vals):
                setattr(p, k, v)
            p.spots, p.u = sp, u
            return mx.sum(m.light_curve_mx(p) * mx.array(wts, dtype=dt))
        with mx.stream(stream):
            args = ([mx.array(np.array(p0.spots), dtype=dt), mx.array(np.array(p0.u), dtype=dt)]
                    + [mx.array(float(getattr(p0, k)), dtype=dt) for k in names])
            g = mx.grad(F, argnums=list(range(len(args))))(*args)
            return [np.array(v, dtype=np.float64) for v in g]
    g64, g32 = grads(mx.float64, mx.cpu), grads(mx.float32, mx.gpu)
    scale = max(np.abs(v).max() for v in g64)
    for k, a, b in zip(["spots", "u"] + names, g32, g64):
        assert np.abs(a - b).max() < 1e-5 * scale, k


def test_events_match_the_graph():
    """The kernel's event search finds the graph's events (fp32, the same
    samples and bisection) and their live times agree."""
    from elome import spots_tau as ST
    p = P(**MODELS["spot smaller than the planet"])
    law = get_law(p.limb_dark)
    f = lambda v: mx.array(np.atleast_1d(np.asarray(v, dtype=np.float64)), dtype=mx.float32)
    tt = mx.array(np.linspace(-0.04, 0.04, 33)[None, :], dtype=mx.float32)
    sp4 = f(np.radians(np.array(p.spots)) * [1, 1, 1, 0] + [0, 0, 0, 1] * np.array(p.spots))[None]
    b = p.a * math.cos(D(p.inc))
    args = [f(p.per), f(p.a), f(b), f(p.rp), f(p.prot), f(D(90.0)), f(0.0)]
    off = mx.full(tt.shape, 0.37, dtype=mx.float32)        # the star's clock, ahead
    half = 0.5 * EXP
    S = ST.sample_count(EXP, p.a, p.per, p.rp)
    with mx.stream(mx.gpu):
        ek, ekf = ST._kernel_events(tt, off, half, *args, sp4, law, S)
        eg, egf = ST._events_fn(mx.float32, S, False, False)(tt - half, tt + half, off,
                                                             *args, sp4)
        assert np.array_equal(np.array(ek), np.array(ekf))      # the frozen copy
        assert np.array_equal(np.array(eg), np.array(egf))
        ek, eg = np.array(ek), np.array(eg)
    ek, eg = np.sort(ek, axis=-1), np.sort(eg, axis=-1)
    assert np.array_equal(np.isfinite(ek), np.isfinite(eg))
    fin = np.isfinite(eg)
    assert fin.sum() >= 8
    assert np.max(np.abs(ek[fin] - eg[fin])) < 1e-6


def test_light_curves_batched_equals_the_loop():
    seq = [P(**MODELS["spot smaller than the planet"]),
           P(spots=[[20.0, LAT + 3.0, 8.0, 0.5], [-30.0, 10.0, 6.0, 0.7]], prot=2.0, lam=20.0)]
    with mx.stream(mx.gpu):
        for rule in RULES.values():
            m = el.TransitModel(seq[0], TT, dtype=mx.float32, **rule)
            got = m.light_curves(seq)
            for i, p in enumerate(seq):
                assert np.abs(got[i] - m.light_curve(p)).max() < 1e-7


def test_ld_basis_columns():
    from elome.spots_tau import flux_dev_spots_t
    p = P()
    t = np.linspace(-0.1, 0.1, 41)
    sp = np.radians(np.array(p.spots)) * [1, 1, 1, 0] + [0, 0, 0, 1] * np.array(p.spots)
    args = (0.0, 0.0, p.per, p.a, p.a * math.cos(D(p.inc)), p.rp, p.prot, D(90.0), 0.0)
    B64 = np.array(flux_dev_spots_t(mx.array(t, dtype=mx.float64), *args, sp, p.limb_dark,
                                    None, basis=True))
    with mx.stream(mx.gpu):
        B32 = np.array(flux_dev_spots_t(mx.array(t, dtype=mx.float32), *args, sp, p.limb_dark,
                                        None, basis=True))
    assert np.abs(B32 - B64).max() < 3e-7 * hybrid_norms(get_law(p.limb_dark)).max()


def test_fp32_model_takes_the_kernels_and_use_metal_false_the_graph():
    from elome import metal_spots as MS
    p = P(**MODELS["spot smaller than the planet"])
    seen = []
    orig_c, orig_e = MS._cols_core, MS.spot_events_kernel

    def spy_c(*a, **kw):
        seen.append("cols")
        return orig_c(*a, **kw)

    def spy_e(*a, **kw):
        seen.append("events")
        return orig_e(*a, **kw)
    MS._cols_core, MS.spot_events_kernel = spy_c, spy_e
    try:
        with mx.stream(mx.gpu):
            k = el.TransitModel(p, TT, dtype=mx.float32, **RULES["contact"]).light_curve(p)
            assert "cols" in seen and "events" in seen
            seen.clear()
            g = el.TransitModel(p, TT, dtype=mx.float32, use_metal=False,
                                **RULES["contact"]).light_curve(p)
            assert not seen
        el.TransitModel(p, TT, **RULES["contact"]).light_curve(p)
        assert not seen
    finally:
        MS._cols_core, MS.spot_events_kernel = orig_c, orig_e
    ref = el.TransitModel(p, TT, **RULES["contact"]).light_curve(p)
    assert np.abs(k - ref).max() < 2e-7 and np.abs(g - ref).max() < 2e-7
