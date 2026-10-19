"""The fp32 Metal kernels for ringed planets (metal_rings) against the fp64
graph: per point (columns and dual-number gradients), as light curves on
every exposure rule and orbit, in their VJP, in the ld_basis contract, per
chain, and on the degenerate geometries."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome as el
from elome import metal as M
from elome import rings
from elome.hybrid import HYBRID2, HYBRID5, hybrid_norms
from elome.trig import sincos

pytestmark = pytest.mark.skipif(not M.metal_available(), reason="needs Metal")

import test_rings as TR                     # noqa: E402  (the geometry cases)

PER = 3.45
TAU = np.linspace(-0.2, 0.2, 201)
W5 = [0.2, 0.2, 0.1, 0.1, 0.1]
BASE = dict(limb_dark="hybrid5", u=W5, theta=0.4, ring_in=1.4, ring_out=2.6,
            ring_f=0.7, ring_tau=0.6)
CASES = {
    "circular": (8.8, 0.25, 0.08, dict()),
    "eccentric": (8.8, 0.25, 0.08, dict(secosw=0.3, sesinw=0.2)),
    "oblate planet": (8.8, 0.3, 0.08, dict(f=0.2)),
    "edge-on wide": (8.8, 0.06, 0.08, dict(ring_f=0.95, ring_out=4.0, ring_tau=None)),
}
RULES = {"none": dict(), "contact": dict(exp_time=0.02),
         "supersample": dict(exp_time=0.02, integration="supersample", n_sub=7)}


def gpu(tau):
    return mx.array(np.asarray(tau, dtype=np.float32))


@pytest.mark.parametrize("law", [HYBRID5, HYBRID2], ids=lambda L: L.name)
@pytest.mark.parametrize("name", list(TR.CASES))
def test_point_columns_and_gradients(name, law):
    """The device functions at single points: columns to the fp32 floor,
    the dual-number gradient of wct . B to 1e-5 of its scale (fp32 near a
    four-crossing ring), against fp64 (rings.py and its autodiff)."""
    from elome import metal_rings as MR
    from elome.oblate import axes
    x0, y0, rp, f, ri, ro, rf, tau = TR.CASES[name]
    T = float(rings.ring_opacity(tau, rf))
    th = 0.4
    X, Y = x0 * math.cos(th) - y0 * math.sin(th), x0 * math.sin(th) + y0 * math.cos(th)
    wct = np.random.default_rng(len(name)).normal(size=law.n_col)
    B32, g32 = MR._point_cols([X], [Y], rp, f, th, ri, ro, rf, T, law,
                              wct=wct[None, :], grad=True)

    def F(*v):
        Xv, Yv, r, fv, t, a1, a2, c, Tv = v
        st, ct_ = sincos(t)            # MLX's fp64 sin/cos are float32-accurate
        x, y = Xv * ct_ + Yv * st, -Xv * st + Yv * ct_
        Ap, Bp = axes(r, fv)
        occ = rings._region_cols(x, y, [(Ap, Bp), (a1 * r, a1 * r * c), (a2 * r, a2 * r * c)],
                                 [Tv], law, mx.float64)
        S = law.generator_matrix()
        B = [-occ[0]] + [-sum(float(S[j][k]) * occ[k] for k in range(len(occ)))
                         for j in range(law.n_w)]
        return sum(float(wct[k]) * B[k] for k in range(law.n_col)), B
    args = [mx.array(np.float64(v), dtype=mx.float64)
            for v in (X, Y, rp, f, th, ri, ro, 1.0 - rf, T)]
    with mx.stream(mx.cpu):
        B64 = np.array([b.item() for b in F(*args)[1]])
        g64 = np.array([g.item() for g in
                        mx.grad(lambda *v: F(*v)[0], argnums=list(range(9)))(*args)])
    assert np.max(np.abs(B32[0] - B64) / hybrid_norms(law)) < 1e-7
    assert np.max(np.abs(g32[0] - g64)) < 1e-5 * np.max(np.abs(g64))


@pytest.mark.parametrize("rule", list(RULES))
@pytest.mark.parametrize("name", list(CASES))
def test_kernel_matches_the_fp64_graph(name, rule):
    a, b, r, kw = CASES[name]
    k = {**BASE, **kw, **RULES[rule]}
    f64 = np.array(el.flux_dev_from_tau(TAU, PER, a, b, r, **k))
    with mx.stream(mx.gpu):
        f32 = np.array(el.flux_dev_from_tau(gpu(TAU), PER, a, b, r, **k))
    assert -f64.min() > 1e-3
    assert np.abs(f32 - f64).max() < 3e-7


@pytest.mark.parametrize("exp_time", [0.0, 0.02], ids=["none", "contact"])
def test_vjp_matches_fp64_autodiff(exp_time):
    """Gradients in every parameter, the weights included."""
    ct = np.random.default_rng(1).normal(size=TAU.size)
    vals = dict(period=PER, a=8.8, b=0.3, r=0.08, theta=0.4, ring_in=1.4,
                ring_out=2.6, ring_f=0.7, ring_tau=0.6, f=0.2, u=np.array(W5))

    def grads(dt, stream):
        def L(d):
            out = el.flux_dev_from_tau(
                mx.array(TAU, dtype=dt), d["period"], d["a"], d["b"], d["r"],
                limb_dark="hybrid5", u=d["u"], exp_time=exp_time, theta=d["theta"],
                ring_in=d["ring_in"], ring_out=d["ring_out"], ring_f=d["ring_f"],
                ring_tau=d["ring_tau"], f=d["f"])
            return mx.sum(out * mx.array(ct, dtype=dt))
        with mx.stream(stream):
            g = mx.grad(L)({k: mx.array(v, dtype=dt) for k, v in vals.items()})
            return {k: np.atleast_1d(np.array(v, dtype=np.float64)) for k, v in g.items()}
    g64, g32 = grads(mx.float64, mx.cpu), grads(mx.float32, mx.gpu)
    scale = max(np.abs(v).max() for v in g64.values())
    for k in vals:
        assert np.abs(g32[k] - g64[k]).max() < 1e-6 * scale, k


@pytest.mark.parametrize("b", [0.0, 0.3])
def test_face_on_gradients(b):
    """ring_f = 0 and f = 0 -- a uniform prior's edge, both outlines
    circles (and, at b = 0, centred on the star mid-transit): d/d ring_f
    and d/df from the kernel's VJP match fp64 autodiff, and the fp32 graph
    is finite and agrees."""
    ct = np.random.default_rng(1).normal(size=TAU.size)
    vals = dict(r=0.08, ring_f=0.0, f=0.0)

    def grads(dt, stream):
        def L(d):
            out = el.flux_dev_from_tau(
                mx.array(TAU, dtype=dt), PER, 8.8, b, d["r"], limb_dark="hybrid5",
                u=W5, theta=0.4, ring_in=1.4, ring_out=2.6, ring_f=d["ring_f"],
                ring_tau=0.6, f=d["f"])
            return mx.sum(out * mx.array(ct, dtype=dt))
        with mx.stream(stream):
            g = mx.grad(L)({k: mx.array(v, dtype=dt) for k, v in vals.items()})
            return {k: float(v.item()) for k, v in g.items()}
    g64 = grads(mx.float64, mx.cpu)
    for g32 in (grads(mx.float32, mx.gpu), grads(mx.float32, mx.cpu)):
        for k in ("ring_f", "f"):
            assert abs(g64[k]) > 1e-5 * abs(g64["r"]), k
            assert abs(g32[k] - g64[k]) < 2e-4 * abs(g64[k]), (k, g32[k], g64[k])


def test_ld_basis_contract_and_its_vjp():
    kw = dict(limb_dark="hybrid4", exp_time=0.02, theta=0.3, ring_in=1.4,
              ring_out=2.6, ring_f=0.8, ring_tau=0.5)
    w = [0.2, 0.2, 0.1, 0.1]
    with mx.stream(mx.gpu):
        B = np.array(el.flux_dev_from_tau(gpu(TAU), PER, 8.8, 0.25, 0.08, ld_basis=True, **kw))
        dev = np.array(el.flux_dev_from_tau(gpu(TAU), PER, 8.8, 0.25, 0.08, u=w, **kw))
    c = np.concatenate([[1.0], -np.asarray(w)])
    N = hybrid_norms("hybrid4")
    assert np.abs(dev - (B @ c) / (N @ c)).max() < 2e-8
    # the basis VJP through the contract equals the scalar VJP
    ct = np.random.default_rng(2).normal(size=TAU.size).astype(np.float32)
    nc = float(N @ c)

    def Ls(r):
        return mx.sum(el.flux_dev_from_tau(gpu(TAU), PER, 8.8, 0.25, r, u=w, **kw) * mx.array(ct))

    def Lb(r):
        Bm = el.flux_dev_from_tau(gpu(TAU), PER, 8.8, 0.25, r, ld_basis=True, **kw)
        return mx.sum((Bm @ mx.array(c.astype(np.float32))) / nc * mx.array(ct))
    with mx.stream(mx.gpu):
        gs = mx.grad(Ls)(mx.array(0.08, dtype=mx.float32)).item()
        gb = mx.grad(Lb)(mx.array(0.08, dtype=mx.float32)).item()
    assert abs(gs - gb) < 1e-5 * abs(gs)


def test_per_chain_parameters_equal_one_chain_at_a_time():
    rf = np.array([0.5, 0.8, 0.95], dtype=np.float32)
    rt = np.array([np.inf, 0.4, 1.0], dtype=np.float32)
    th = np.array([0.0, 0.7, 2.0], dtype=np.float32)
    kw = dict(limb_dark="hybrid5", u=W5, exp_time=0.02, ring_in=1.4, ring_out=2.6)
    with mx.stream(mx.gpu):
        batch = np.array(el.flux_dev_from_tau(gpu(np.tile(TAU, (3, 1))), PER, 8.8, 0.25, 0.08,
                                              theta=mx.array(th), ring_f=mx.array(rf),
                                              ring_tau=mx.array(rt), **kw))
        for i in range(3):
            one = np.array(el.flux_dev_from_tau(gpu(TAU), PER, 8.8, 0.25, 0.08, theta=float(th[i]),
                                                ring_f=float(rf[i]), ring_tau=float(rt[i]), **kw))
            assert np.abs(batch[i] - one).max() < 1e-7


def test_degenerate_geometries_are_finite():
    """Tangencies, a planet crossing the star's centre, nearly edge-on and
    face-on rings, a ring larger than the star: finite values and
    gradients, exactly 0 out of transit."""
    tau = np.linspace(-0.3, 0.3, 241)
    for kw in (dict(b=0.0), dict(ring_f=1.0 - 1.0 / 1.4),          # b_in = rp: the lens appears
               dict(ring_f=1.0 - 1e-6, ring_out=4.0), dict(ring_f=0.0),
               dict(ring_out=12.0, ring_f=0.3), dict(b=0.0, f=0.2)):
        k = dict(BASE)
        b = kw.pop("b", 0.25)
        k.update(kw)
        with mx.stream(mx.gpu):
            def L(r):
                return mx.sum(el.flux_dev_from_tau(gpu(tau), PER, 8.8, b, r, exp_time=0.02, **k))
            v, g = mx.value_and_grad(L)(mx.array(0.08, dtype=mx.float32))
            out = np.array(el.flux_dev_from_tau(gpu(tau), PER, 8.8, b, 0.08, **k))
        assert np.isfinite(v.item()) and np.isfinite(g.item()), kw
        assert np.all(out[np.abs(tau) > 0.29] == 0.0), kw


def test_fp32_model_takes_the_ring_kernel_and_use_metal_false_the_graph():
    from elome import metal_rings as MR
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.3, PER, 0.0813, 8.8, 88.5
    p.limb_dark, p.u = "hybrid5", list(W5)
    p.theta, p.ring_in, p.ring_out, p.ring_f = 25.0, 1.4, 2.6, 0.7
    t = 0.3 + np.linspace(-0.15, 0.15, 61)
    seen = []
    orig = MR._make_core

    def spy(*a, **kw):
        seen.append(1)
        return orig(*a, **kw)
    MR._make_core = spy
    try:
        k = el.TransitModel(p, t, dtype=mx.float32).light_curve(p)
        assert seen
        seen.clear()
        g = el.TransitModel(p, t, dtype=mx.float32, use_metal=False).light_curve(p)
        assert not seen
    finally:
        MR._make_core = orig
    ref = el.TransitModel(p, t).light_curve(p)
    assert np.abs(k - ref).max() < 3e-7 and np.abs(g - ref).max() < 3e-7
