"""The moon's fp32 Metal kernels (metal_moons): the lens and each disc as
accurate as the fp32 graph against the fp64 one, values and gradients,
with finite gradients where the graph's are 0/0, and light curves the
graph's to float32's rounding."""

import numpy as np
import mlx.core as mx
import pytest

import elome
import elome.metal as M
from elome import metal_moons as MM
from elome import moons as MN
from elome.hybrid import declared, get_law

from test_moons_api import EXP, HYB, QUAD, sets, windows

pytestmark = pytest.mark.skipif(not M.metal_available(), reason="needs Metal")

LAWS = {"hybrid4": get_law("hybrid4").eps, "moments": ()}


def _geometries(L=4000, seed=0):
    """Overlapping discs anywhere on the star, across the limb and nested."""
    rng = np.random.default_rng(seed)
    zc, th = rng.uniform(0.0, 1.15, L), rng.uniform(0, 2 * np.pi, L)
    rp = rng.uniform(0.02, 0.3, L)
    rs = rp * rng.uniform(0.05, 1.5, L)
    d, ph = (rp + rs) * rng.uniform(0.0, 1.05, L), rng.uniform(0, 2 * np.pi, L)
    xp, yp = zc * np.cos(th), zc * np.sin(th)
    return [xp, yp, rp, xp + d * np.cos(ph), yp + d * np.sin(ph), rs], rng


def _lens(H, eps, W, dt, kernel):
    st = mx.gpu if dt == mx.float32 else mx.cpu
    with mx.stream(st):
        v = [mx.array(a, dtype=dt) for a in H]
        w = mx.array(W, dtype=dt)
        if kernel:
            fn = lambda *a: MM.lens_cols_kernel(*a, eps)              # noqa: E731
        else:
            fn = lambda *a: MN._lens_cols([a[:3], a[3:]], eps, dt)   # noqa: E731
        val = np.stack([np.array(c, np.float64) for c in fn(*v)], -1)
        g = mx.grad(lambda *a: mx.sum(mx.stack(fn(*a), -1) * w), argnums=tuple(range(6)))(*v)
        return val, np.stack([np.array(c, np.float64) for c in g], -1)


@pytest.mark.parametrize("law", list(LAWS))
def test_the_lens_kernel_is_as_accurate_as_the_graph(law):
    eps = LAWS[law]
    H, rng = _geometries()
    W = rng.normal(size=(H[0].size, 3 + len(eps)))
    v64, g64 = _lens(H, eps, W, mx.float64, False)
    v32, g32 = _lens(H, eps, W, mx.float32, False)
    vk, gk = _lens(H, eps, W, mx.float32, True)
    assert np.all(np.isfinite(vk)) and np.all(np.isfinite(gk))
    for c in range(v64.shape[1]):
        assert np.abs(vk[:, c] - v64[:, c]).max() <= 2.0 * np.abs(v32[:, c] - v64[:, c]).max()
    for j in range(6):
        e_k, e_32 = np.abs(gk[:, j] - g64[:, j]), np.abs(g32[:, j] - g64[:, j])
        assert e_k.max() <= 2.0 * e_32.max()
        assert np.quantile(e_k, 0.999) <= 2.0 * np.quantile(e_32, 0.999)


DEGENERATE = {   # xp, yp, rp, xm, ym, rs
    "concentric": (0.5, 0.2, 0.1, 0.5, 0.2, 0.05),
    "planet centred": (0.0, 0.0, 0.1, 0.05, 0.0, 0.04),
    "both centred": (0.0, 0.0, 0.1, 0.0, 0.0, 0.04),
    "moon centred": (0.03, 0.0, 0.1, 0.0, 0.0, 0.04),
}


@pytest.mark.parametrize("name", list(DEGENERATE))
def test_the_lens_kernel_where_the_graph_gradient_is_0_over_0(name):
    """Concentric discs, or a centre on the star's: the graph's gradient is
    NaN there (a masked 0/0), the kernel's finite and the derivative of the
    float64 values."""
    eps = LAWS["hybrid4"]
    c = np.array(DEGENERATE[name])
    W = np.array([[1.0, -0.7, 0.4, 0.3, -0.2]])
    vk, gk = _lens([[x] for x in c], eps, W, mx.float32, True)

    def val(cc):
        return float(np.sum(_lens([[x] for x in cc], eps, W, mx.float64, False)[0] * W))
    h = 1e-5
    fd = np.array([(val(c + h * np.eye(6)[j]) - val(c - h * np.eye(6)[j])) / (2 * h)
                   for j in range(6)])
    v64 = _lens([[x] for x in c], eps, W, mx.float64, False)[0]
    assert np.abs(vk - v64).max() < 3e-7
    assert np.abs(gk[0] - fd).max() < 3e-7


@pytest.mark.parametrize("large", [False, True])
def test_the_disc_kernel_is_the_graph(large):
    """Each disc's shape columns, with bodies behind the star and moons of
    zero size, and a planet larger than the star."""
    rng = np.random.default_rng(3)
    L = 4000
    z, th = rng.uniform(0.0, 2.2, L), rng.uniform(0, 2 * np.pi, L)
    x, y = z * np.cos(th), z * np.sin(th)
    r = rng.uniform(0.5, 1.6, L) if large else rng.uniform(0.0, 0.4, L)
    r[:20], r[20:25] = 0.0, 1e-9
    front = rng.uniform(size=L) > 0.1
    W = rng.normal(size=(L, 5))
    law = get_law("hybrid4")

    def run(dt, kernel, monkey):
        if not kernel:
            monkey.setattr(MN, "_metal", lambda dt: None)
        with mx.stream(mx.gpu if dt == mx.float32 else mx.cpu), declared(large):
            v = [mx.array(a, dtype=dt) for a in (x, y, r)]
            f, w = mx.array(front), mx.array(W, dtype=dt)
            fn = lambda *a: mx.stack(MN.disc_shape_cols(*a, law, f), -1)   # noqa: E731
            val = np.array(fn(*v), np.float64)
            g = mx.grad(lambda *a: mx.sum(fn(*a) * w), argnums=(0, 1, 2))(*v)
        monkey.undo()
        return val, np.stack([np.array(c, np.float64) for c in g], -1)
    with pytest.MonkeyPatch.context() as mp:
        v64, g64 = run(mx.float64, False, mp)
        v32, g32 = run(mx.float32, False, mp)
        vk, gk = run(mx.float32, True, mp)
    assert np.all(np.isfinite(gk))
    assert np.abs(vk - v64).max() <= 2.5 * np.abs(v32 - v64).max()
    assert np.abs(gk - g64).max() <= 2.0 * np.abs(g32 - g64).max()


@pytest.mark.parametrize("law", [HYB, QUAD], ids=["hybrid", "quadratic"])
@pytest.mark.parametrize("contact", [False, True], ids=["instant", "contact"])
def test_light_curves_by_the_kernels_are_the_graph(monkeypatch, law, contact):
    ps = sets(law)
    kw = dict(exp_time=EXP, integration="contact", n_gl=5) if contact else {}
    with mx.stream(mx.gpu):
        m = elome.TransitModel(ps[0], windows(ps[0], step=EXP), dtype=mx.float32, **kw)
        fast = m.light_curves(ps)
        monkeypatch.setattr(MN, "_metal", lambda dt: None)
        graph = m.light_curves(ps)
    with mx.stream(mx.cpu):
        exact = elome.TransitModel(ps[0], windows(ps[0], step=EXP), **kw).light_curves(ps)
    assert np.abs(fast - exact).max() <= 1.5 * np.abs(graph - exact).max() + 3e-8


def _quad_lens(H, W, dt, kernel):
    st = mx.gpu if dt == mx.float32 else mx.cpu

    def fn(*a):
        if kernel:
            return MM.lens_quad_kernel(*a)
        m0, m2, m4 = MN._lens_cols([a[:3], a[3:]], (), dt)
        return [m0, m2, MN._lens_mu1(m0, m2, m4, *a[:2], a[2], *a[3:5], a[5], dt)]
    with mx.stream(st):
        v = [mx.array(a, dtype=dt) for a in H]
        w = mx.array(W, dtype=dt)
        val = np.stack([np.array(c, np.float64) for c in fn(*v)], -1)
        g = mx.grad(lambda *a: mx.sum(mx.stack(fn(*a), -1) * w), argnums=tuple(range(6)))(*v)
        return val, np.stack([np.array(c, np.float64) for c in g], -1)


def test_the_quadratic_lens_kernel_is_as_accurate_as_the_graph():
    """mu^0, mu^2 and the closure's mu^1, nested discs and slivers included."""
    H, rng = _geometries(seed=1)
    W = rng.normal(size=(H[0].size, 3))
    v64, g64 = _quad_lens(H, W, mx.float64, False)
    v32, g32 = _quad_lens(H, W, mx.float32, False)
    vk, gk = _quad_lens(H, W, mx.float32, True)
    assert np.all(np.isfinite(vk)) and np.all(np.isfinite(gk))
    for c in range(3):
        assert np.abs(vk[:, c] - v64[:, c]).max() <= 2.0 * np.abs(v32[:, c] - v64[:, c]).max()
    for j in range(6):
        e_k, e_32 = np.abs(gk[:, j] - g64[:, j]), np.abs(g32[:, j] - g64[:, j])
        assert np.quantile(e_k, 0.999) <= 2.0 * np.quantile(e_32, 0.999)


@pytest.mark.parametrize("large", [False, True])
def test_the_quadratic_disc_kernel_is_the_graph(large):
    """A disc that may exceed the star takes the graph itself (the kernel's
    terms of order r cancel in float32)."""
    rng = np.random.default_rng(4)
    L = 4000
    z, th = rng.uniform(0.0, 2.2, L), rng.uniform(0, 2 * np.pi, L)
    x, y = z * np.cos(th), z * np.sin(th)
    r = rng.uniform(0.5, 1.6, L) if large else rng.uniform(0.0, 0.4, L)
    r[:20] = 0.0
    u1, u2 = rng.uniform(0.0, 0.8, L), rng.uniform(-0.2, 0.4, L)
    front = rng.uniform(size=L) > 0.1
    W = rng.normal(size=L)

    def run(dt, kernel, monkey):
        if not kernel:
            monkey.setattr(MN, "_metal", lambda dt: None)
        with mx.stream(mx.gpu if dt == mx.float32 else mx.cpu), declared(large):
            v = [mx.array(a, dtype=dt) for a in (x, y, r, u1, u2)]
            f, w = mx.array(front), mx.array(W, dtype=dt)
            fn = lambda *a: MN.disc_flux_quad(*a, f)                  # noqa: E731
            val = np.array(fn(*v), np.float64)
            g = mx.grad(lambda *a: mx.sum(fn(*a) * w), argnums=tuple(range(5)))(*v)
        monkey.undo()
        return val, np.stack([np.array(c, np.float64) for c in g], -1)
    with pytest.MonkeyPatch.context() as mp:
        v64, g64 = run(mx.float64, False, mp)
        v32, g32 = run(mx.float32, False, mp)
        vk, gk = run(mx.float32, True, mp)
    assert np.all(np.isfinite(gk))
    if large:
        assert np.array_equal(vk, v32) and np.array_equal(gk, g32)
    assert np.abs(vk - v64).max() <= 2.5 * np.abs(v32 - v64).max()
    assert np.abs(gk - g64).max() <= 2.5 * np.abs(g32 - g64).max()


@pytest.mark.parametrize("law", ["quadratic", "hybrid5"])
def test_kernels_across_the_limb(law):
    """Overlaps straddling the limb, by the kernels against float64: the
    docs' 6e-7 and 1.5e-8 (measured 3.4e-7 and 1.5e-8 over 20,000)."""
    rng = np.random.default_rng(3)
    n, rp, rs = 4000, 0.1, 0.03
    ang, zp = rng.uniform(0, 2 * np.pi, n), rng.uniform(1 - rp, 1 + rp, n)
    xp, yp = zp * np.cos(ang), zp * np.sin(ang)
    a2, D = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, rp + rs, n)
    H = [xp, yp, rp, xp + D * np.cos(a2), yp + D * np.sin(a2), rs]
    hl = get_law("hybrid5")

    def flux(dt):
        with mx.stream(mx.gpu if dt == mx.float32 else mx.cpu), declared(False):
            X = [mx.array(v, dtype=dt) for v in H]
            if law == "quadratic":
                out = (MN.disc_flux_quad(*X[:3], 0.4, 0.25) + MN.disc_flux_quad(*X[3:], 0.4, 0.25)
                       + MN.lens_flux_quad(*X, 0.4, 0.25))
            else:
                c = [a + b + lc for a, b, lc in zip(MN.disc_shape_cols(*X[:3], hl),
                                                    MN.disc_shape_cols(*X[3:], hl),
                                                    MN.lens_shape_cols(*X, hl))]
                out = elome.hybrid.combine_cols(
                    c, mx.array([0.2, 0.15, 0.1, 0.05, 0.1], dtype=dt), hl)
            return np.array(out, np.float64)
    err = np.abs(flux(mx.float32) - flux(mx.float64)).max()
    assert err < (6e-7 if law == "quadratic" else 2e-8), err


def _sky(seed=5, L=3000, big=False):
    """Planets and moons anywhere about the star, overlapping or not, some
    behind it."""
    rng = np.random.default_rng(seed)
    zc, th = rng.uniform(0.0, 1.4, L), rng.uniform(0, 2 * np.pi, L)
    rp = rng.uniform(1.05, 1.5, L) if big else rng.uniform(0.02, 0.3, L)
    rs = rng.uniform(0.01, 0.3, L)
    d, ph = (rp + rs) * rng.uniform(0.0, 1.6, L), rng.uniform(0, 2 * np.pi, L)
    xp, yp = zc * np.cos(th), zc * np.sin(th)
    fp, fm = rng.uniform(size=L) > 0.1, rng.uniform(size=L) > 0.1
    return [xp, yp, xp + d * np.cos(ph), yp + d * np.sin(ph), rp, rs], fp, fm, rng


_HW = [0.2, 0.15, 0.1, 0.05]          # hybrid4 shape weights


def _public(name, H, fp, fm, W, dt, kernel, monkey, large=None):
    if not kernel:
        monkey.setattr(MN, "_metal", lambda dt: None)
    with mx.stream(mx.gpu if dt == mx.float32 else mx.cpu):
        v = [mx.array(a, dtype=dt) for a in H]
        f_p, f_m = mx.array(fp), mx.array(fm)
        if name == "generator_cols_moon":
            fn = lambda *a: mx.stack(MN.generator_cols_moon(                 # noqa: E731
                *a, "hybrid4", f_p, f_m, large_occultor=large), -1)
        elif name == "shape_cols_moon":
            fn = lambda *a: MN.shape_cols_moon(*a, "hybrid4", f_p, f_m,    # noqa: E731
                                               large_occultor=large)
        elif name == "flux_dev_moon":
            fn = lambda *a: MN.flux_dev_moon(*a, mx.array(_HW, dtype=dt), "hybrid4",  # noqa: E731
                                             f_p, f_m, large_occultor=large)
        else:
            fn = lambda *a: MN.flux_dev_moon_quad(*a, 0.4, 0.25, f_p, f_m)  # noqa: E731
        val = np.array(fn(*v), np.float64)
        w = mx.array(W.reshape(val.shape), dtype=dt)
        g = mx.grad(lambda *a: mx.sum(fn(*a) * w), argnums=tuple(range(6)))(*v)
    monkey.undo()
    return val, np.stack([np.array(c, np.float64) for c in g], -1)


@pytest.mark.parametrize("name", ["generator_cols_moon", "shape_cols_moon", "flux_dev_moon",
                                  "flux_dev_moon_quad"])
def test_the_public_sky_functions_by_the_kernels(name):
    """In float32 on the GPU the sky-position functions take the kernels,
    as accurate as the graph against float64."""
    H, fp, fm, rng = _sky()
    law = get_law("hybrid4")
    ncol = {"generator_cols_moon": 3 + len(law.eps), "shape_cols_moon": law.n_col}.get(name, 1)
    W = rng.normal(size=(H[0].size, ncol))
    with pytest.MonkeyPatch.context() as mp:
        v64, g64 = _public(name, H, fp, fm, W, mx.float64, False, mp)
        v32, g32 = _public(name, H, fp, fm, W, mx.float32, False, mp)
        vk, gk = _public(name, H, fp, fm, W, mx.float32, True, mp)
    assert np.all(np.isfinite(vk)) and np.all(np.isfinite(gk))
    assert not np.array_equal(vk, v32)                  # the kernels did run
    assert np.abs(vk - v64).max() <= 2.0 * np.abs(v32 - v64).max()
    e_k, e_32 = np.abs(gk - g64), np.abs(g32 - g64)
    assert np.quantile(e_k, 0.999) <= 2.0 * np.quantile(e_32, 0.999)


@pytest.mark.parametrize("name", ["flux_dev_moon", "flux_dev_moon_quad"])
def test_the_public_sky_functions_for_a_planet_larger_than_its_star(name):
    """The quadratic discs take the graph itself (the kernel's terms of
    order r cancel in float32); the hybrid laws' kernel builds in the large
    form."""
    H, fp, fm, rng = _sky(seed=6, L=1500, big=True)
    W = rng.normal(size=(H[0].size, 1))
    with pytest.MonkeyPatch.context() as mp:
        v64, _ = _public(name, H, fp, fm, W, mx.float64, False, mp, large=True)
        v32, _ = _public(name, H, fp, fm, W, mx.float32, False, mp, large=True)
        vk, gk = _public(name, H, fp, fm, W, mx.float32, True, mp, large=True)
    assert np.all(np.isfinite(gk))
    assert np.abs(vk - v64).max() <= 2.5 * np.abs(v32 - v64).max()
