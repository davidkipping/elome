"""Ringed planets (elome/rings.py): the fp64 reference graph and its fp32
twin against the ring oracle (tests/ring_oracle.py), the identities a
ringed planet must satisfy, and gradients in every parameter."""

import math

import mlx.core as mx
import numpy as np
import pytest

import ring_oracle as RO
from elome import rings
from elome.hybrid import (HYBRID2, HYBRID4, HYBRID5, HybridLaw, combine_cols, even_cols,
                          hybrid_norms, pole_col)
from elome.oblate import shape_cols_oblate

EPS = tuple(HYBRID5.eps) + tuple(HYBRID4.eps) + tuple(HYBRID2.eps)
ALL = HybridLaw("all_poles", EPS, ((1.0, -1.0, 0.0),))     # every pole in one call
NORM = np.array([math.pi, math.pi / 2, math.pi / 3] + [math.pi / (e * (1 + e)) for e in EPS])
NAMES = ["x0", "y0", "rp", "f", "ring_in", "ring_out", "ring_f", "ring_tau"]
TOL = {mx.float64: 1e-14, mx.float32: 2e-7}               # error / column norm

# (x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau)
CASES = {
    "cavity": (0.3, 0.2, 0.08, 0.0, 1.5, 2.5, 0.1, np.inf),
    "inner edge cut": (0.3, 0.2, 0.08, 0.0, 1.2, 2.5, 0.7, np.inf),
    "both edges cut, translucent": (0.2, -0.1, 0.06, 0.0, 1.3, 3.0, 0.9, 0.5),
    "on the limb": (0.95, 0.1, 0.08, 0.0, 1.5, 2.6, 0.6, np.inf),
    "four limb crossings": (0.97, 0.3, 0.05, 0.0, 1.4, 4.0, 0.95, np.inf),
    "planet off the star": (1.10, -0.2, 0.06, 0.0, 1.3, 3.0, 0.5, 1.0),
    "centred": (0.0, 0.0, 0.08, 0.0, 1.3, 2.4, 0.8, np.inf),
    "oblate planet": (0.4, 0.3, 0.08, 0.1, 1.2, 2.2, 0.7, np.inf),
    "oblate planet on the limb": (0.9, -0.3, 0.07, 0.2, 1.3, 2.5, 0.85, 0.3),
    "face-on ring": (0.3, 0.1, 0.08, 0.0, 1.5, 2.5, 0.0, np.inf),
    "face-on ring, centred": (0.0, 0.0, 0.08, 0.0, 1.5, 2.5, 0.0, 0.6),
    "ring covers the star": (0.1, 0.05, 0.4, 0.0, 1.2, 6.0, 0.2, 0.4),
}


def arr(v, dt):
    return mx.array(np.float64(v), dtype=dt)


def cols(cs, dt=mx.float64, law=ALL):
    with mx.stream(mx.cpu):
        g = rings.generator_cols_ring(*[arr(v, dt) for v in cs], law)
        return np.array([c.item() for c in g])


def oracle(cs):
    x0, y0, rp, f, ri, ro, rf, tau = cs
    (Ap, Bp), edges = rings.ring_axes(rp, f, ri, ro, rf)
    return RO.ring_cols(x0, y0, (Ap, Bp), edges, [float(rings.ring_opacity(tau, rf))], EPS)


def err(v, ref):
    return np.max(np.abs(v - ref) / NORM)


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("name", list(CASES))
def test_against_the_oracle(name, dt):
    cs = CASES[name]
    assert err(cols(cs, dt), oracle(cs)) < TOL[dt]


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("c", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
@pytest.mark.parametrize("where", [(0.3, 0.2), (0.96, 0.15)], ids=["inside", "limb"])
def test_nearly_edge_on(where, c, dt):
    """A thin ring cuts short arcs from the planet's outline: their side of
    the ring is read without forming num + den (terms 1e8 apart)."""
    cs = (*where, 0.08, 0.0, 1.5, 3.0, 1.0 - c, np.inf)
    assert err(cols(cs, dt), oracle(cs)) < TOL[dt]


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("delta", [-1e-6, -1e-12, 0.0, 1e-12, 1e-6])
@pytest.mark.parametrize("where", ["inside", "limb", "triple"])
@pytest.mark.parametrize("edge", ["inner", "outer"])
def test_tangencies(edge, where, delta, dt):
    """The planet's circle tangent to a ring edge's minor axis -- where the
    lens appears -- inside the star, on the limb, and with the tangent point
    on the limb: the side of each arc comes from closed-form corners, never
    a point test, so the exact tangency is as accurate as its neighbours."""
    rp, ri, ro = 0.08, 1.4, 2.6
    rr = ri if edge == "inner" else ro
    c = (1.0 + delta) * 1.0 / rr                  # b_edge = rr rp c = rp (1 + delta)
    x0, y0 = {"inside": (0.3, 0.2), "limb": (0.93, 0.1),
              "triple": (0.3, math.sqrt(1 - 0.09) - rp)}[where]
    cs = (x0, y0, rp, 0.0, ri, ro, 1.0 - c, np.inf)
    assert err(cols(cs, dt), oracle(cs)) < TOL[dt]


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("d", [0.0, 1e-30, 1e-20, 1e-12, 1e-6])
def test_centred_on_the_star(d, dt):
    """A circle centred on the star sends the closed-form roots to 0 and
    infinity: below 1e-30 R (fp64) / 1e-9 R (fp32) the first-order
    expansion in the offset takes over."""
    cs = (d * 0.6, d * 0.8, 0.08, 0.0, 1.5, 2.5, 0.7, np.inf)
    v = cols(cs, dt)
    assert np.isfinite(v).all()
    assert err(v, oracle(cs)) < TOL[dt]


@pytest.mark.parametrize("name", ["cavity", "oblate planet", "on the limb", "oblate planet on the limb"])
def test_transparent_ring_is_the_bare_planet(name):
    """tau = 0: the ring blocks nothing, and the planet alone is the
    package's spherical or oblate solution."""
    x0, y0, rp, f, ri, ro, rf, _ = CASES[name]
    v = cols((x0, y0, rp, f, ri, ro, rf, 0.0), law=HYBRID5)
    with mx.stream(mx.cpu):
        if f == 0.0:
            z, r = arr(math.hypot(x0, y0), mx.float64), arr(rp, mx.float64)
            ref = np.array([c.item() for c in even_cols(z, r)]
                           + [pole_col(z, r, e).item() for e in HYBRID5.eps])
        else:
            from elome.oblate import generator_cols_oblate
            g = generator_cols_oblate(arr(x0, mx.float64), arr(y0, mx.float64),
                                      arr(rp, mx.float64), arr(f, mx.float64), HYBRID5)
            ref = np.array([c.item() for c in g])
    norm = np.array([math.pi, math.pi / 2, math.pi / 3] + [math.pi / (e * (1 + e)) for e in HYBRID5.eps])
    assert np.max(np.abs(v - ref) / norm) < 1e-14


def test_opacity_is_linear():
    """F(tau) = F(0) + T(tau) [F(inf) - F(0)], T = 1 - exp(-tau / |cos i|)."""
    cs = list(CASES["both edges cut, translucent"])
    lo, hi = cols(cs[:7] + [0.0]), cols(cs[:7] + [np.inf])
    for tau in (0.1, 0.5, 2.0, 30.0):
        T = float(rings.ring_opacity(tau, cs[6]))
        assert np.max(np.abs(cols(cs[:7] + [tau]) - (lo + T * (hi - lo))) / NORM) < 1e-15


def test_opacity_keeps_the_data_precision():
    """A Python tau with a float64 ring_f is not rounded through float32."""
    exact = -math.expm1(-0.1 / 0.5)
    with mx.stream(mx.cpu):
        for rf in (mx.array(0.5, dtype=mx.float64), 0.5):
            for tau in (0.1, mx.array(0.1, dtype=mx.float64)):
                if isinstance(rf, float) and isinstance(tau, float):
                    v = rings.ring_opacity(tau, rf)
                else:
                    v = rings.ring_opacity(tau, rf).item()
                assert abs(v - exact) < 1e-16


def test_generator_columns_take_host_arguments():
    """Python numbers beside the arrays, as the other arguments allow."""
    x0, y0 = arr(0.3, mx.float64), arr(0.0, mx.float64)
    with mx.stream(mx.cpu):
        host = rings.generator_cols_ring(x0, y0, 0.1, 0.0, 1.5, 2.0, 0.5, 1.0, "hybrid2")
        dev = rings.generator_cols_ring(x0, y0, *[arr(v, mx.float64) for v in
                                                   (0.1, 0.0, 1.5, 2.0, 0.5, 1.0)], "hybrid2")
        assert all(isinstance(c, mx.array) for c in host)
        assert all(np.array_equal(np.array(a), np.array(b)) for a, b in zip(host, dev))


def test_out_of_transit_is_exactly_zero():
    for cs in [(1.6, 0.0, 0.08, 0.0, 1.5, 2.5, 0.5, np.inf), (0.0, 1.9, 0.08, 0.1, 1.2, 2.0, 0.7, 1.0)]:
        assert np.all(cols(cs) == 0.0)


def test_shape_columns_and_flux():
    """The ld_basis contract: flux - 1 == (B @ c) / (N @ c), c = (1, -w),
    for every law, from shape_cols_ring and flux_dev_ring alike."""
    cs = CASES["on the limb"]
    for law, w in ((HYBRID2, [0.4, 0.2]), (HYBRID4, [0.2, 0.2, 0.1, 0.1]),
                   (HYBRID5, [0.2, 0.1, 0.1, 0.1, 0.05])):
        with mx.stream(mx.cpu):
            args = [arr(v, mx.float64) for v in cs]
            B = np.array(rings.shape_cols_ring(*args, law))
            dev = rings.flux_dev_ring(*args, w, law).item()
        c = np.concatenate([[1.0], -np.asarray(w)])
        N = hybrid_norms(law)
        assert abs(dev - (B @ c) / (N @ c)) < 1e-16
        # and the same from the oracle's generator columns
        G = oracle(cs)
        idx = [0, 1, 2] + [3 + EPS.index(e) for e in law.eps]
        S = law.generator_matrix()
        ref = np.concatenate([[G[0]], S @ G[idx]])
        assert np.max(np.abs(B - ref)) < 1e-14 * np.max(np.abs(ref))


GRAD_CASES = ["inner edge cut", "both edges cut, translucent", "four limb crossings",
              "planet off the star", "centred", "oblate planet on the limb",
              "face-on ring", "face-on ring, centred"]


def _grads(cs, corners=True):
    w = mx.array(1.0 / NORM, dtype=mx.float64)

    def F(*args):
        return mx.sum(mx.stack(rings.generator_cols_ring(*args, ALL), -1) * w)
    args = [arr(v, mx.float64) for v in cs]
    saved = rings._h
    if not corners:
        rings._h = lambda arg, s, even: 0.0 * s
    try:
        with mx.stream(mx.cpu):
            ad = np.array([g.item() for g in mx.grad(F, argnums=list(range(8)))(*args)])
            fd = np.zeros(8)
            at = lambda i, d: F(*[arr(v + (d if j == i else 0.0), mx.float64)
                                  for j, v in enumerate(cs)]).item()
            for i in range(8):
                if not np.isfinite(cs[i]):
                    continue
                h = 1e-6 * max(abs(cs[i]), 0.1)
                if i in (3, 6) and cs[i] == 0.0:
                    # f = 0 / ring_f = 0, a circle (the prior's edge): one-sided
                    fd[i] = (-3.0 * at(i, 0.0) + 4.0 * at(i, h) - at(i, 2 * h)) / (2 * h)
                else:
                    fd[i] = (at(i, h) - at(i, -h)) / (2 * h)
    finally:
        rings._h = saved
    return ad, fd


@pytest.mark.parametrize("name", GRAD_CASES)
def test_gradients_match_finite_differences(name):
    cs = CASES[name]
    ad, fd = _grads(cs)
    keep = [i for i in range(8) if np.isfinite(cs[i])]
    assert np.isfinite(ad).all()
    assert np.max(np.abs(ad - fd)[keep]) < 1e-7 * np.max(np.abs(fd[keep]))


@pytest.mark.parametrize("name", ["four limb crossings", "planet off the star"])
def test_corner_terms_are_required(name):
    """Without the gradient-only corner terms, the frozen arc endpoints
    leave O(1) errors: a guard that the terms stay wired in."""
    ad, fd = _grads(CASES[name], corners=False)
    assert np.max(np.abs(ad - fd)) > 0.1 * np.max(np.abs(fd))


def test_gradients_finite_in_fp32():
    w = mx.array((1.0 / NORM).astype(np.float32))
    for cs in CASES.values():
        def F(*args):
            return mx.sum(mx.stack(rings.generator_cols_ring(*args, ALL), -1) * w)
        with mx.stream(mx.cpu):
            g = mx.grad(F, argnums=list(range(8)))(*[arr(v, mx.float32) for v in cs])
        assert all(np.isfinite(x.item()) for x in g), cs


def test_batched_equals_one_at_a_time():
    """Elementwise over broadcast arguments: a batch of positions with
    shared ring parameters equals the points one by one (to rounding: MLX
    vectorises differently)."""
    xs = np.array([0.3, 0.95, 1.1, 0.0])
    ys = np.array([0.2, 0.1, -0.2, 0.0])
    rest = (0.07, 0.0, 1.4, 2.6, 0.75, 0.8)
    with mx.stream(mx.cpu):
        g = rings.generator_cols_ring(mx.array(xs, dtype=mx.float64), mx.array(ys, dtype=mx.float64),
                                      *[arr(v, mx.float64) for v in rest], ALL)
        batch = np.stack([np.array(c) for c in g], -1)
    for i in range(len(xs)):
        assert err(batch[i], cols((xs[i], ys[i], *rest))) < 1e-16


@pytest.mark.parametrize("bad, msg", [
    (dict(ring_f=1.0), "ring_f"), (dict(ring_f=-0.1), "ring_f"),
    (dict(ring_tau=-1.0), "ring_tau"), (dict(ring_out=1.2), "ring_out"),
    (dict(ring_in=0.9), "equator"), (dict(f=0.3, ring_in=1.1), "equator"),
    (dict(f=-0.1), "flattening"), (dict(f=1.0), "flattening")])
def test_parameter_checks(bad, msg):
    p = dict(x0=0.3, y0=0.2, rp=0.08, f=0.0, ring_in=1.5, ring_out=2.5, ring_f=0.5,
             ring_tau=np.inf)
    p.update(bad)
    with mx.stream(mx.cpu), pytest.raises(ValueError, match=msg):
        rings.shape_cols_ring(arr(p.pop("x0"), mx.float64), *p.values(), HYBRID4)
