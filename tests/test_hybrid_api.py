"""Hybrid limb-darkening laws through the batman-style TransitModel.

The laws ride the existing vector-law plumbing (the polynomial law's):
one coefficient vector per model, the same compiled graphs, contact
integration, batched light_curves and the differentiable light_curve_mx.
These tests pin that plumbing for the three laws; the closed forms
themselves are tested in test_hybrid.py.
"""

import contextlib
import math

import numpy as np
import mlx.core as mx
import pytest

import elome
from hybrid_weights import phys_w
from elome import ld
from elome.hybrid import LAWS, flux_dev_hybrid
from elome.metal import metal_available

T = np.linspace(-0.15, 0.15, 601)
P, A, INC, RP = 3.45, 8.8, 87.0, 0.1
EXP = 0.02
WEIGHTS = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
           "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
NAMES = list(WEIGHTS)


def params(law, u=None, **kw):
    p = elome.TransitParams()
    d = dict(t0=0.0, per=P, rp=RP, a=A, inc=INC, ecc=0.0, w=90.0,
             limb_dark=law, u=list(WEIGHTS[law]) if u is None else list(u))
    d.update(kw)
    for k, v in d.items():
        setattr(p, k, v)
    return p


def circular_z(t, t0=0.0, per=P, a=A, inc=INC):
    phi = 2 * math.pi * (t - t0) / per
    b = a * math.cos(math.radians(inc))
    z = np.sqrt((a * np.sin(phi)) ** 2 + (b * np.cos(phi)) ** 2)
    return z, np.cos(phi) > 0


@pytest.mark.parametrize("law", NAMES)
def test_light_curve_is_flux_dev_hybrid_on_the_orbit(law):
    m = elome.TransitModel(params(law), T)
    got = m.light_curve(params(law))
    z, front = circular_z(T)
    with mx.stream(mx.cpu):
        f = np.asarray(flux_dev_hybrid(
            mx.array(np.where(front, z, 2 + z), dtype=mx.float64),
            mx.array(RP, dtype=mx.float64), WEIGHTS[law], law))
    assert np.abs(got - (1 + f)).max() < 1e-13
    assert got.min() < 0.99 and np.isfinite(got).all()


@pytest.mark.parametrize("law", NAMES)
def test_weights_update_without_rebuild(law):
    m = elome.TransitModel(params(law), T)
    f1 = m.light_curve(params(law))
    u = list(WEIGHTS[law])
    u[0] += 0.3
    f2 = m.light_curve(params(law, u=u))
    assert not np.allclose(f1, f2)
    assert np.array_equal(f2, elome.TransitModel(params(law, u=u), T)
                          .light_curve(params(law, u=u)))


@pytest.mark.parametrize("law", NAMES)
def test_wrong_or_changed_count_raises(law):
    n = LAWS[law].n_w
    with pytest.raises(ValueError, match=f"{law} takes {n} weights"):
        elome.TransitModel(params(law, u=[0.1] * (n + 1)), T)
    m = elome.TransitModel(params(law), T)
    with pytest.raises(ValueError, match="coefficient count changed"):
        m.light_curve(params(law, u=[0.1] * (n - 1)))
    with pytest.raises(ValueError, match="law changed"):
        m.light_curve(params("quadratic", u=[0.4, 0.25]))


@pytest.mark.parametrize("law", NAMES)
def test_eccentric_runs(law):
    p = params(law, ecc=0.3, w=63.0)
    m = elome.TransitModel(p, T)
    f = m.light_curve(p)
    assert f.min() < 0.99 and np.isfinite(f).all()
    assert list(m._compiled) == [(False, False)]          # fp64: the graph


# ---------------------------------------------------------------------------
# fp32 GPU: the tau kernel (_sph_tau_fn); the z-input kernel behind _photom
# serves the (e, w) orbit and direct _hybrid_dev calls
# ---------------------------------------------------------------------------

MODES = {"plain": {}, "contact": dict(exp_time=EXP, integration="contact"),
         "supersample": dict(exp_time=EXP, supersample_factor=5)}


@contextlib.contextmanager
def kernel_spy():
    """Records, per trace of the hybrid z-kernel entry point, whether the
    GPU stream was active -- i.e. whether that trace took the kernel."""
    import elome.api as api
    from elome import metal as M
    orig, calls = api.flux_dev_metal_hybrid, []

    def spy(z, r, law, u, basis=False):
        calls.append(bool(M._gpu_stream_active()))
        return orig(z, r, law, u, basis=basis)

    api.flux_dev_metal_hybrid = spy
    try:
        yield calls
    finally:
        api.flux_dev_metal_hybrid = orig


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circ", "e0.3"])
def test_fp32_gpu_model_runs_the_tau_kernel(law, mode, ecc):
    """The whole model -- orbit, exposure rule, photometry -- runs in
    metal_hybrid's tau kernel (the "sph-tau" cache key; the z-kernel is
    never traced) and agrees with the fp32 graph (use_metal=False) at fp32
    noise and with the fp64 model within 2e-6. The exception: on a circular
    orbit without the contact rule light_curve, which no gradient reaches,
    takes the faster forward, the circular graph into the z-kernel, and
    light_curve_mx still takes the tau kernel."""
    p = params(law, ecc=ecc, w=63.0)
    m = elome.TransitModel(p, T, dtype=mx.float32, **MODES[mode])
    with kernel_spy() as calls:
        got = m.light_curve(p)
    orbit = "circ" if ecc == 0.0 else "kh"
    if ecc == 0.0 and mode != "contact":
        assert calls == [True]
        assert list(m._compiled) == [(True, False)]
        with kernel_spy() as calls:
            raw = np.array(m.light_curve_mx(p), dtype=np.float64)
        assert calls == []
        assert list(m._compiled) == [(True, False), ("sph-tau", orbit)]
        assert np.abs(raw.reshape(T.size, -1).mean(axis=1) - got).max() < 1e-6
    else:
        assert calls == []
        assert list(m._compiled) == [("sph-tau", orbit)]
    graph = elome.TransitModel(p, T, dtype=mx.float32, use_metal=False,
                                     **MODES[mode]).light_curve(p)
    f64 = elome.TransitModel(p, T, **MODES[mode]).light_curve(p)
    assert np.abs(got - graph).max() < 1e-6
    assert np.abs(got - f64).max() < 2e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("mode", ["plain", "contact"])
def test_fp32_light_curves_match_the_loop(law, mode):
    n = 4
    rng = np.random.default_rng(5)
    W = phys_w(law, rng, n)
    rp = np.linspace(0.08, 0.12, n)
    m = elome.TransitModel(params(law), T, dtype=mx.float32,
                                 **MODES[mode])
    pa = params(law)
    pa.rp, pa.u = rp, W
    got = m.light_curves(pa)
    # The same per-point kernel on both sides, but light_curves forms z on
    # the anchored orbit (k = h = 0) and light_curve on the circular graph:
    # two fp32 expressions, apart by ~a*eps, times |dF/dz| <~ 0.1, plus the
    # rounding of 1 + dev -- about one fp32 ulp of 1 (measured 0.5 plain,
    # 1.0 contact, over seeds). Gate at four.
    ulp = float(np.finfo(np.float32).eps)
    for j in range(n):
        ref = m.light_curve(params(law, u=W[j], rp=float(rp[j])))
        assert np.abs(got[j] - ref).max() <= 4 * ulp, j


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("mode", ["plain", "contact"])
def test_a_cpu_stream_first_call_does_not_pin_the_graph(mode):
    """The CPU-stream-first-call trap: a first call on the CPU stream runs
    hybrid.py's graph (no kernel can run there, so the z-kernel entry point
    is never called), and the later GPU calls still take the tau kernel --
    the route is chosen per call, from the active stream. (light_curve_mx:
    light_curve itself takes the circular graph on the GPU without the
    contact rule.)"""
    law = "hybrid5"
    m = elome.TransitModel(params(law), T, dtype=mx.float32,
                                 **MODES[mode])
    with kernel_spy() as calls:
        with mx.stream(mx.cpu):
            a = np.array(m.light_curve_mx(params(law)))
        b = np.array(m.light_curve_mx(params(law)))
        m.light_curve_mx(params(law))
    assert calls == []
    assert list(m._compiled) == [(True, False), ("sph-tau", "circ")]
    assert np.abs(a - b).max() < 1e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_float_radius_with_per_set_weights_takes_the_kernel_path():
    """_hybrid_dev's batched branch passes a Python-float radius to the
    kernel as is (an earlier version reshaped it unguarded; mx.reshape rejects
    floats). Every frontend route casts rp to an array before _photom,
    so this guards the direct-call contract, not a frontend regression."""
    m = elome.TransitModel(params("hybrid5"), T, dtype=mx.float32)
    z = mx.array(np.linspace(0.5, 1.2, 8, dtype=np.float32)).reshape(2, 4)
    w = mx.array(np.array([WEIGHTS["hybrid5"]] * 2, np.float32))
    with kernel_spy() as calls:
        out = m._hybrid_dev(z, 0.1, w)
        mx.eval(out)
    assert calls == [True]
    ref = flux_dev_hybrid(z, 0.1, w, "hybrid5")
    assert out.shape == (2, 4)
    assert np.abs(np.asarray(out - ref, np.float64)).max() < 1e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_one_weight_row_broadcasts_on_the_kernel_route():
    """A (1, n_w) weight row with n > 1 rows of z: the kernel entry point
    broadcasts it (an earlier version reshaped it by the z row count and raised)."""
    m = elome.TransitModel(params("hybrid4"), T, dtype=mx.float32)
    z = mx.array(np.linspace(0.5, 1.2, 8, dtype=np.float32)).reshape(2, 4)
    w1 = mx.array(np.array([WEIGHTS["hybrid4"]], np.float32))        # (1, 4)
    rp = mx.array([0.1, 0.11], dtype=mx.float32)
    with kernel_spy() as calls:
        out = m._hybrid_dev(z, rp, w1)
        mx.eval(out)
    assert calls == [True]
    ref = flux_dev_hybrid(z, rp[:, None], w1, "hybrid4")
    assert np.abs(np.asarray(out - ref, np.float64)).max() < 1e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("wshape", [(4,), (1, 4)])
def test_one_set_on_1d_z_is_one_kernel_row(wshape):
    """A scalar radius with one set of weights on (m,) separations is one
    (1, m) row of the kernel grid, not m single-point rows (an earlier version sent a
    (1, n_w) row down the batched branch with n = m)."""
    import elome.metal_hybrid as MH
    m = elome.TransitModel(params("hybrid4"), T, dtype=mx.float32)
    z = mx.array(np.linspace(0.5, 1.2, 8, dtype=np.float32))
    w = mx.array(np.reshape(WEIGHTS["hybrid4"], wshape), dtype=mx.float32)
    shapes, orig = [], MH._z_core

    def spy(law, basis):
        core = orig(law, basis)

        def wrapped(z2d, rc, *a):
            shapes.append((z2d.shape, rc.shape) + tuple(x.shape for x in a))
            return core(z2d, rc, *a)
        return wrapped

    MH._z_core = spy
    try:
        out = m._hybrid_dev(z, 0.1, w)
        mx.eval(out)
    finally:
        MH._z_core = orig
    assert shapes == [((1, 8), (1,), (1, 4))]
    assert out.shape == (8,)


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("mode", ["plain", "contact"])
def test_fp32_cpu_stream_fallback_is_the_graph_bitwise(law, mode):
    """An fp32 model with Metal on, run on the CPU stream, falls back to
    hybrid.py's graph -- the very graph use_metal=False runs -- not to
    the kernel entry point's own fallback (they once were 2.4e-7 apart). Both
    sides on the CPU stream: the same graph rounds differently on the two
    devices, and that is not what is under test."""
    p = params(law)
    m = elome.TransitModel(p, T, dtype=mx.float32, **MODES[mode])
    g = elome.TransitModel(p, T, dtype=mx.float32, use_metal=False,
                                 **MODES[mode])
    with kernel_spy() as calls:
        with mx.stream(mx.cpu):
            a = m.light_curve(p)
            b = g.light_curve(p)
    assert calls == []
    assert np.array_equal(a, b)


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
def test_fp32_light_curve_mx_gradients_through_the_kernel(law):
    """Gradients in rp, a, inc and every weight through the tau kernel's
    VJP against the fp64 model's, condition-scaled."""
    n_w = LAWS[law].n_w
    w0 = np.array(WEIGHTS[law])
    ct = np.random.default_rng(6).normal(size=T.size)

    def grads(dtype):
        m = elome.TransitModel(params(law), T, dtype=dtype)
        ctx = mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.gpu)
        with ctx:
            c = mx.array(ct, dtype=dtype)

            def loss(rp, a, inc, w):
                return mx.sum(c * m.light_curve_mx(
                    params(law, u=w, rp=rp, a=a, inc=inc)))

            g = mx.grad(loss, argnums=(0, 1, 2, 3))(
                *[mx.array(v, dtype=dtype) for v in (RP, A, INC)],
                mx.array(w0, dtype=dtype))
            mx.eval(g)
        assert (("sph-tau", "circ") in m._compiled) == (dtype == mx.float32)
        return np.concatenate([np.atleast_1d(np.asarray(x, np.float64))
                               for x in g])

    g32, g64 = grads(mx.float32), grads(mx.float64)
    assert g32.shape == (3 + n_w,)
    scale = np.abs(g64).max()
    assert np.abs(g32 - g64).max() / scale < 2e-3


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_array_eccentricity_route_uses_the_kernel_too():
    law = "hybrid4"
    m = elome.TransitModel(params(law, ecc=0.3, w=63.0), T,
                                 dtype=mx.float32)
    with kernel_spy() as calls:
        out = np.asarray(m.light_curve_mx(params(law, ecc=mx.array(0.3),
                                                 w=63.0)), dtype=np.float64)
    assert calls == [True] and list(m._compiled) == ["ew"]
    assert np.abs(out - m.light_curve(params(law, ecc=0.3, w=63.0))).max() < 1e-6


@pytest.mark.parametrize("law", NAMES)
def test_contact_integration(law):
    """The contact rule needs nothing law-specific: n_gl = 7 agrees with
    n_gl = 40 (which is converged) to its own truncation, ~2e-8 on this
    heavily smeared exposure (a sixth of T14; the documented quadratic
    figure is 9e-8 at n_gl = 5), and with a heavy supersampling to the
    latter's own O(1/N) accuracy."""
    def lc(**kw):
        return elome.TransitModel(params(law), T, exp_time=EXP,
                                        **kw).light_curve(params(law))
    c7 = lc(integration="contact", n_gl=7)
    c40 = lc(integration="contact", n_gl=40)
    s = lc(integration="supersample", supersample_factor=1001)
    assert np.abs(c7 - c40).max() < 1e-7
    assert np.abs(c7 - s).max() < 5e-6
    assert np.abs(c7 - lc(integration="supersample")).max() > 1e-5


@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("form", ["sequence", "array"])
def test_light_curves_per_set_weights(law, form):
    n = 4
    rng = np.random.default_rng(1)
    W = phys_w(law, rng, n)
    rp = np.linspace(0.08, 0.12, n)
    m = elome.TransitModel(params(law), T)
    if form == "sequence":
        got = m.light_curves([params(law, u=W[j], rp=float(rp[j]))
                              for j in range(n)])
    else:
        pa = params(law)
        pa.rp, pa.u = rp, W
        got = m.light_curves(pa)
    assert got.shape == (n, T.size)
    for j in range(n):
        ref = m.light_curve(params(law, u=W[j], rp=float(rp[j])))
        assert np.abs(got[j] - ref).max() < 1e-15, j


@pytest.mark.parametrize("law", NAMES)
def test_light_curves_contact_mode(law):
    m = elome.TransitModel(params(law), T, exp_time=EXP,
                                 integration="contact", n_gl=7)
    ps = [params(law, rp=r) for r in (0.08, 0.1, 0.12)]
    got = m.light_curves(ps)
    for j, p in enumerate(ps):
        assert np.abs(got[j] - m.light_curve(p)).max() < 1e-15


@pytest.mark.parametrize("law", NAMES)
def test_fp32_model_matches_fp64(law):
    if not metal_available():
        pytest.skip("Metal unavailable")
    p = params(law)
    f32 = elome.TransitModel(p, T, dtype=mx.float32).light_curve(p)
    f64 = elome.TransitModel(p, T).light_curve(p)
    assert np.abs(f32 - f64).max() < 2e-6


@pytest.mark.parametrize("law", NAMES)
def test_light_curve_mx_gradient_in_every_weight(law):
    """light_curve_mx differentiates in each w_j through the affine
    shape map; FD in fp64 on the CPU."""
    m = elome.TransitModel(params(law), T)
    n = LAWS[law].n_w
    w0 = np.array(WEIGHTS[law])
    ct = np.random.default_rng(2).normal(size=T.size)
    with mx.stream(mx.cpu):
        c = mx.array(ct)
        g = mx.grad(lambda w: mx.sum(c * m.light_curve_mx(params(law, u=w))))(
            mx.array(w0, dtype=mx.float64))
        g = np.asarray(g)
        # h = 1e-4: the flux is smooth in w (a ratio of two affine forms),
        # and the innermost pole's gradient is ~1e-6, which a 1e-6 step
        # cannot resolve against O(10) sums
        for j in range(n):
            h = 1e-4
            wp, wm = w0.copy(), w0.copy()
            wp[j] += h
            wm[j] -= h
            fd = np.sum(ct * (np.asarray(m.light_curve_mx(params(law, u=wp)))
                              - np.asarray(m.light_curve_mx(params(law, u=wm))))) / (2 * h)
            scale = max(abs(fd), 1e-3 * np.abs(g).max())
            assert abs(g[j] - fd) / scale < 1e-6, j


@pytest.mark.parametrize("law", NAMES)
def test_secondary_eclipse_ignores_limb_darkening(law):
    p = params(law, fp=1e-3, t_secondary=P / 2)
    m = elome.TransitModel(p, T + P / 2, transittype="secondary")
    q = params("quadratic", u=[0.4, 0.25], fp=1e-3, t_secondary=P / 2)
    mq = elome.TransitModel(q, T + P / 2, transittype="secondary")
    assert np.array_equal(m.light_curve(p), mq.light_curve(q))


def test_priors_feed_the_model():
    """The documented recipe: weights from the uniform prior maps."""
    rng = np.random.default_rng(3)
    w5 = ld.simplex_from_q_np(rng.random(5))
    w2 = ld.hybrid2_from_q_np(rng.random(), rng.random())
    for law, w in (("hybrid5", w5), ("hybrid2", np.asarray(w2))):
        p = params(law, u=w)
        f = elome.TransitModel(p, T).light_curve(p)
        assert np.isfinite(f).all() and f.min() < 0.99 and f.max() <= 1.0



# ---------------------------------------------------------------------------
# custom laws: a HybridLaw object wherever a name goes
# ---------------------------------------------------------------------------

CUSTOM = {
    "hybrid2-type": elome.HybridLaw("custom2", [0.15], [[1, -1, 0]]),
    "two-pole": elome.HybridLaw("custom3", (0.05, 0.5), ((1, 0, -1),)),
}


@pytest.mark.parametrize("which", list(CUSTOM))
@pytest.mark.parametrize("dtype", [None, mx.float32], ids=["fp64", "fp32"])
def test_transit_model_takes_a_hybrid_law_object(which, dtype):
    """light_curve, light_curves and light_curve_mx with a custom law equal
    flux_dev_hybrid on the model's own separations."""
    if dtype is mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    law = CUSTOM[which]
    w = phys_w(law, np.random.default_rng(3))
    p = params("hybrid2")
    p.limb_dark, p.u = law, list(w)
    m = elome.TransitModel(p, T, dtype=dtype)
    f = m.light_curve(p)
    z, front = circular_z(T)
    with mx.stream(mx.cpu):
        ref = 1.0 + np.asarray(flux_dev_hybrid(
            mx.array(np.where(front, z, 2 + z), dtype=mx.float64),
            mx.array(RP, dtype=mx.float64), list(w), law))
    tol = 1e-13 if dtype is None else 2e-6
    assert np.abs(f - ref).max() <= tol
    assert np.abs(np.asarray(m.light_curve_mx(p), np.float64) - f).max() <= 1e-6
    pa = params("hybrid2")
    pa.limb_dark, pa.u = law, np.stack([w, w])
    pa.rp = np.array([RP, RP])
    lcs = m.light_curves(pa)
    assert np.abs(lcs - f[None, :]).max() <= (1e-14 if dtype is None else 2e-7)


def test_an_equal_law_object_is_the_same_law():
    """_check_law compares laws by value: a fresh, equal object is fine."""
    law = CUSTOM["hybrid2-type"]
    p = params("hybrid2")
    p.limb_dark, p.u = law, [0.3, 0.2]
    m = elome.TransitModel(p, T)
    p.limb_dark = elome.HybridLaw("custom2", (0.15,), ((1.0, -1.0, 0.0),))
    m.light_curve(p)
    p.limb_dark = elome.HybridLaw("custom2", (0.2,), ((1.0, -1.0, 0.0),))
    with pytest.raises(ValueError, match="law changed"):
        m.light_curve(p)


def test_a_law_registered_after_import_is_supported():
    from elome import hybrid as Hm
    law = elome.HybridLaw("late2", (0.25,), ((1.0, -1.0, 0.0),))
    Hm.LAWS["late2"] = law
    try:
        p = params("hybrid2")
        p.limb_dark, p.u = "late2", [0.3, 0.2]
        a = elome.TransitModel(p, T).light_curve(p)
        p.limb_dark = law
        b = elome.TransitModel(p, T).light_curve(p)
        assert np.array_equal(a, b)
    finally:
        del Hm.LAWS["late2"]



def test_a_name_and_its_law_object_are_one_law():
    """Built with "hybrid2", called with elome.HYBRID2 (and the
    reverse): the same law, not "law changed" (an earlier version compared
    raw fields)."""
    for build, call in (("hybrid2", elome.HYBRID2),
                        (elome.HYBRID2, "hybrid2")):
        p = params("hybrid2")
        p.limb_dark = build
        m = elome.TransitModel(p, T)
        a = m.light_curve(p)
        p.limb_dark = call
        assert np.array_equal(m.light_curve(p), a)


def test_a_model_keeps_its_law_when_the_registry_changes():
    """The law is resolved once, at construction: re-binding or deleting
    the name later cannot make one model trace two laws (an earlier version re-read
    the registry on every new graph)."""
    from elome import hybrid as Hm
    law = elome.HybridLaw("x", (0.2,), ((1.0, -1.0, 0.0),))
    Hm.LAWS["x"] = law
    try:
        p = params("hybrid2")
        p.limb_dark, p.u = "x", [0.3, 0.2]
        m = elome.TransitModel(p, T)
        circ = m.light_curve(p)
        Hm.LAWS["x"] = elome.HybridLaw("x", (0.5,), ((1.0, -1.0, 0.0),))
        with pytest.raises(ValueError, match="law changed"):
            m.light_curve(p)                       # "x" now means another law
        p.limb_dark = law
        assert np.array_equal(m.light_curve(p), circ)
        del Hm.LAWS["x"]
        pe = params("hybrid2", ecc=0.2, w=63.0)    # a graph traced only now
        pe.limb_dark, pe.u = law, [0.3, 0.2]
        e = m.light_curve(pe)
        ref = elome.TransitModel(pe, T).light_curve(pe)
        assert np.array_equal(e, ref)
    finally:
        Hm.LAWS.pop("x", None)


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circ", "e0.3"])
def test_a_hybrid_law_eclipse_is_an_eclipse(ecc):
    """A secondary eclipse sees a uniform disc whatever the law: on the
    tau route a hybrid-law eclipse takes the eclipse kernel, not the hybrid
    transit kernel, and equals the quadratic-law model's bit for bit."""
    sq = math.sqrt(ecc)
    c = float(elome.occultation_offset(sq * math.cos(math.radians(63.0)),
                                       sq * math.sin(math.radians(63.0)))) * P
    t = np.linspace(c - 0.2, c + 0.2, 801)
    p = params("hybrid4", ecc=ecc, w=63.0, fp=2e-3)
    m = elome.TransitModel(p, t, transittype="secondary", dtype=mx.float32)
    got = m.light_curve(p)
    q = params("hybrid4", ecc=ecc, w=63.0, fp=2e-3)
    q.limb_dark, q.u = "quadratic", [0.4, 0.2]
    ref = elome.TransitModel(q, t, transittype="secondary", dtype=mx.float32).light_curve(q)
    assert list(m._compiled) == [("sph-tau", "circ" if ecc == 0.0 else "kh")]
    assert got.min() < 1.0 + 1e-3
    assert np.array_equal(got, ref)


def test_spherical_tau_routes_never_touch_the_oblate_or_ring_kernels():
    """TransitModel's spherical tau-kernel routes -- a hybrid law under
    every exposure rule, the quadratic contact rule -- import none of the
    oblate or ring code."""
    import subprocess
    import sys
    code = r'''
import sys, numpy as np, mlx.core as mx
import elome as el
t = np.linspace(-0.12, 0.12, 201)
for law, u in (("hybrid4", [0.2, 0.2, 0.1, 0.1]), ("quadratic", [0.4, 0.25])):
    for ecc in (0.0, 0.3):
        for kw in ({}, dict(exp_time=0.02, supersample_factor=3),
                   dict(exp_time=0.02, integration="contact", n_gl=5)):
            p = el.TransitParams()
            p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 87.5, ecc, 63.0
            p.limb_dark, p.u = law, u
            m = el.TransitModel(p, t, dtype=mx.float32, **kw)
            m.light_curve(p)
            m.light_curves([p, p])
mods = ("elome.metal_oblate", "elome.metal_rings", "elome.rings_tau")
print("IMPORTED:" + ",".join(m for m in mods if m in sys.modules))
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("IMPORTED:"), out.stdout
