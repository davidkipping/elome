"""Secondary eclipses through the sampler API:
flux_dev_from_tau(..., transittype="secondary", fp=).

1 + dev is TransitModel(transittype="secondary")'s flux, for spherical and
oblate planets, circular and eccentric orbits, with and without
supersampling; per-chain batches, both precisions, gradients in tau and
every parameter, under mx.compile as a sampler runs it.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import elome as el
from elome.exposure import contact_geometry

T0, PER, A, INC, R, FP = 0.3, 3.45, 8.8, 88.0, 0.1, 2e-3
F, TH = 0.3, 35.0
EXP = 0.02
TAU = np.linspace(-0.5 * PER, 0.5 * PER, 2001)       # one whole orbit


def _b(ecc, w, inc=INC):
    """The impact parameter at conjunction, as TransitModel derives it."""
    return float(contact_geometry(A, ecc, ecc * math.sin(math.radians(w)),
                                  math.cos(math.radians(inc)),
                                  sqrt=math.sqrt, maximum=max)[1])


def _kw(ecc, w, oblate):
    kw = {}
    if ecc:
        kw.update(secosw=math.sqrt(ecc) * math.cos(math.radians(w)),
                  sesinw=math.sqrt(ecc) * math.sin(math.radians(w)))
    if oblate:
        kw.update(f=F, theta=math.radians(TH))
    return kw


def _params(ecc, w, oblate):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, R, A, INC, ecc, w
    p.limb_dark, p.u, p.fp = "quadratic", [0.4, 0.2], FP
    if oblate:
        p.f, p.theta = F, TH
    return p


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
@pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 40.0), (0.6, -70.0)])
@pytest.mark.parametrize("ss", [1, 7], ids=["none", "supersample"])
def test_equals_transit_model(oblate, ecc, w, ss):
    t = T0 + 2 * PER + np.linspace(0.0, PER, 2001)
    p = _params(ecc, w, oblate)
    mkw = dict(supersample_factor=ss, exp_time=EXP) if ss > 1 else {}
    ref = el.TransitModel(p, t, transittype="secondary", **mkw).light_curve(p)
    tau = (t - T0) - PER * np.round((t - T0) / PER)
    xkw = dict(exp_time=EXP, integration="supersample", n_sub=ss) if ss > 1 else {}
    dev = np.asarray(el.flux_dev_from_tau(tau, PER, A, _b(ecc, w), R,
                                          transittype="secondary", fp=FP,
                                          **_kw(ecc, w, oblate), **xkw))
    assert ref.min() < 1.0 + 0.5 * FP and ref.max() == 1.0 + FP
    assert np.abs(1.0 + dev - ref).max() < 1e-15


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
def test_per_chain_batches_equal_the_loop(oblate):
    """Per-chain (n,) parameters, fp and the shape included, on (m,) and
    (n, m) times; mixed circular and eccentric chains in one batch."""
    rp = np.array([0.1, 0.08, 0.12])
    fp = np.array([2e-3, 1e-3, 5e-4])
    b = np.array([0.3, 0.0, 0.5])
    k, h = np.array([0.0, 0.3, -0.2]), np.array([0.0, 0.2, 0.4])
    kw = dict(f=np.array([0.1, 0.3, 0.0]), theta=np.array([0.0, 0.6, 2.0])) if oblate else {}
    with mx.stream(mx.cpu):
        arr = lambda x: mx.array(x, dtype=mx.float64)
        got = np.asarray(el.flux_dev_from_tau(
            TAU, PER, A, arr(b), arr(rp), transittype="secondary", fp=arr(fp),
            secosw=arr(k), sesinw=arr(h), **{n: arr(v) for n, v in kw.items()}))
        t2 = mx.broadcast_to(arr(TAU), (3, TAU.size))
        got2 = np.asarray(el.flux_dev_from_tau(
            t2, PER, A, arr(b), arr(rp), transittype="secondary", fp=arr(fp),
            secosw=arr(k), sesinw=arr(h), **{n: arr(v) for n, v in kw.items()}))
    loop = np.stack([np.asarray(el.flux_dev_from_tau(
        TAU, PER, A, b[i], rp[i], transittype="secondary", fp=fp[i],
        secosw=k[i], sesinw=h[i], **{n: v[i] for n, v in kw.items()}))
        for i in range(3)])
    assert got.shape == (3, TAU.size)
    assert np.abs(got - loop).max() < 1e-18
    assert np.array_equal(got, got2)
    assert np.all(loop.min(axis=1) < fp * 0.5)            # every chain is eclipsed


def test_one_dimensional_tau_gives_one_dimensional_flux():
    dev = el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="secondary", fp=FP)
    assert dev.shape == TAU.shape
    dev = np.asarray(dev)
    assert dev[0] == pytest.approx(0.0, abs=1e-18)                  # at +-P/2: totality
    assert dev[TAU.size // 2] == FP                                 # at transit


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
def test_periodic_across_the_wrap(oblate):
    """The eclipse sits at tau = +-P/2 on a circular orbit, where a caller's
    tau wraps: tau and tau - P give the same flux, to the rounding of the
    phase (~1e-14 of fp)."""
    tau = np.linspace(0.3 * PER, 0.5 * PER, 401)
    kw = _kw(0.0, 90.0, oblate)
    up = np.asarray(el.flux_dev_from_tau(tau, PER, A, 0.3, R, transittype="secondary",
                                         fp=FP, **kw))
    dn = np.asarray(el.flux_dev_from_tau(tau - PER, PER, A, 0.3, R,
                                         transittype="secondary", fp=FP, **kw))
    assert up.min() < 0.5 * FP
    assert np.abs(up - dn).max() < 1e-16


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_fp32_matches_fp64(oblate, ecc):
    kw = _kw(ecc, 40.0, oblate)
    d64 = np.asarray(el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="secondary",
                                          fp=FP, **kw))
    d32 = np.asarray(el.flux_dev_from_tau(mx.array(TAU, dtype=mx.float32), PER, A, 0.3, R,
                                          transittype="secondary", fp=FP, **kw))
    assert np.abs(d32 - d64).max() < 3e-7


def test_limb_darkening_arguments_are_ignored():
    ref = np.asarray(el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="secondary",
                                          fp=FP))
    for kw in (dict(u1=0.4, u2=0.2), dict(limb_dark="hybrid4", u=[0.3, 0.2, 0.1, 0.1])):
        got = np.asarray(el.flux_dev_from_tau(TAU, PER, A, 0.3, R,
                                              transittype="secondary", fp=FP, **kw))
        assert np.array_equal(got, ref)
    obl = np.asarray(el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="secondary",
                                          fp=FP, f=F, theta=0.6, u1=0.4, u2=0.2))
    assert obl.min() < 0.5 * FP


_NAMES = ("shift", "period", "a", "b", "r", "fp", "secosw", "sesinw", "f", "theta")


def _loss(tau, ct, oblate):
    def loss(shift, period, a, b, r, fp, k, h, f, theta):
        kw = dict(f=f, theta=theta) if oblate else {}
        dev = el.flux_dev_from_tau(tau + shift, period, a, b, r, transittype="secondary",
                                   fp=fp, secosw=k, sesinw=h, exp_time=EXP,
                                   integration="supersample", n_sub=3, **kw)
        return mx.sum(ct * dev)
    return loss


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
def test_gradients_match_finite_differences(oblate):
    """tau (as a shift of every time), the orbit, the planet, fp and the
    shape, against central differences (Richardson), on an eccentric orbit
    through ingress and egress."""
    x = [0.0, PER, A, 0.3, R, FP, 0.3, 0.2, F, 0.6]
    n = len(x) if oblate else len(x) - 2
    with mx.stream(mx.cpu):
        tau = mx.array(np.linspace(-0.5 * PER, 0.5 * PER, 3001), dtype=mx.float64)
        ct = mx.array(np.random.default_rng(3).normal(size=3001), dtype=mx.float64)
        loss = _loss(tau, ct, oblate)
        args = [mx.array(v, dtype=mx.float64) for v in x]
        g = mx.grad(loss, argnums=tuple(range(n)))(*args)
        for i in range(n):
            def fd(step):
                up, dn = list(args), list(args)
                up[i] = args[i] + step
                dn[i] = args[i] - step
                return (float(loss(*up)) - float(loss(*dn))) / (2 * step)
            step = 1e-5 * max(abs(x[i]), 0.01)
            ref = (4 * fd(step / 2) - fd(step)) / 3
            assert abs(float(g[i]) - ref) <= 2e-4 * max(abs(ref), 1e-6), _NAMES[i]


@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_compiled_value_and_grad_equal_eager(oblate, dtype):
    """As a sampler runs it: value and gradient under mx.compile, per-chain
    parameters, equal to the eager call."""
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        tau = mx.array(np.linspace(-0.5 * PER, 0.5 * PER, 1001), dtype=dtype)
        ct = mx.array(np.random.default_rng(4).normal(size=1001), dtype=dtype)

        def loss(r, fp, f):
            kw = dict(f=f, theta=0.6) if oblate else {}
            dev = el.flux_dev_from_tau(tau, PER, A, 0.3, r, transittype="secondary",
                                       fp=fp, secosw=0.3, sesinw=0.2, **kw)
            return mx.sum(ct * dev)
        args = [mx.array(v, dtype=dtype) for v in ([0.1, 0.12], [2e-3, 1e-3], [0.3, 0.1])]
        eager = mx.value_and_grad(loss, argnums=(0, 1, 2))(*args)
        comp = mx.compile(mx.value_and_grad(loss, argnums=(0, 1, 2)))(*args)
        mx.eval(eager, comp)
    tol = 1e-12 if dtype == mx.float64 else 1e-5
    v0, g0 = eager
    v1, g1 = comp
    assert abs(float(v0) - float(v1)) <= tol * max(abs(float(v0)), 1.0)
    for a, b in zip(g0, g1):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(a)) and np.abs(a - b).max() <= tol * max(np.abs(a).max(), 1.0)
    if not oblate:
        assert np.all(np.asarray(g0[2]) == 0.0)


def test_errors():
    kw = dict(transittype="secondary", fp=FP)
    with pytest.raises(ValueError, match="pass f= too"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, theta=0.5, **kw)
    with pytest.raises(ValueError, match="needs fp="):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="secondary")
    with pytest.raises(ValueError, match="pass transittype='secondary'"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, 0.4, 0.2, fp=FP)
    with pytest.raises(ValueError, match="transittype must be"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, transittype="tertiary", fp=FP)
    with pytest.raises(ValueError, match="no ring or spots"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, ring_in=1.5, ring_out=2.0, ring_f=0.5,
                             **kw)
    with pytest.raises(ValueError, match="no ring or spots"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, spots=np.array([[0.1, 0.1, 0.1, 0.5]]),
                             spot_time=TAU, prot=10.0, limb_dark="hybrid2", **kw)
    with pytest.raises(ValueError, match="ld_basis"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, ld_basis=True, **kw)
    with pytest.raises(ValueError, match="not available yet"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, f=F, exp_time=EXP, **kw)
    with pytest.raises(ValueError, match="r_eff"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, 0.6, f=0.5, **kw)
    # exp_time = 0 makes the default contact rule instantaneous: allowed
    el.flux_dev_from_tau(TAU, PER, A, 0.3, R, integration="contact", **kw)


# ---------------------------------------------------------------------------
# the fp32 Metal kernels (metal_eclipse, metal_oblate)
# ---------------------------------------------------------------------------

needs_metal = pytest.mark.skipif(not el.metal_available(),
                                 reason="the Metal kernels need a GPU")
#: per-chain sets: the flattening spans the sphere, below f_sw, and a
#: strongly flattened planet; b from central to grazing
_KN = dict(a=[7.5, 7.8, 8.1, 8.4, 8.7, 9.0], b=[0.0, 0.17, 0.34, 0.51, 0.68, 0.85],
           r=[0.06, 0.076, 0.092, 0.108, 0.124, 0.14], fp=[FP] * 6,
           f=[0.0, 1e-6, 0.02, 0.1, 0.25, 0.4], theta=[-1.0, -0.54, -0.08, 0.38, 0.84, 1.3],
           secosw=[0.25] * 6, sesinw=[-0.15] * 6)


def _kernel_run(dtype, oblate, ecc, ss):
    """Values, gradients in every parameter and tau's, of a weighted sum
    through the eclipse; fp32-rounded inputs in either precision."""
    names = ["a", "b", "r", "fp"] + (["f", "theta"] if oblate else []) + (
        ["secosw", "sesinw"] if ecc else [])
    c = float(el.occultation_offset(0.25, -0.15)) * PER if ecc else 0.5 * PER
    tau = mx.array(np.tile(np.linspace(c - 0.12, c + 0.12, 1500), (6, 1)).astype(np.float32),
                   dtype=dtype)
    ps = [mx.array(np.asarray(_KN[k], np.float32), dtype=dtype) for k in names]
    ct = mx.array(np.random.default_rng(5).normal(size=(6, 1500)), dtype=dtype)
    xkw = dict(exp_time=EXP, integration="supersample", n_sub=ss) if ss > 1 else {}

    def dev(t, *args):
        d = dict(zip(names, args))
        return el.flux_dev_from_tau(t, PER, d.pop("a"), d.pop("b"), d.pop("r"),
                                    transittype="secondary", **d, **xkw)
    out = dev(tau, *ps)
    g = mx.grad(lambda t, *a: mx.sum(ct * dev(t, *a)),
                argnums=tuple(range(len(ps) + 1)))(tau, *ps)
    mx.eval(out, g)
    return names, np.asarray(out, np.float64), [np.asarray(x, np.float64) for x in g]


@needs_metal
@pytest.mark.parametrize("ss", [1, 5], ids=["none", "supersample"])
@pytest.mark.parametrize("ecc", [False, True], ids=["circular", "eccentric"])
@pytest.mark.parametrize("oblate", [False, True], ids=["spherical", "oblate"])
def test_kernel_matches_the_fp64_graph(oblate, ecc, ss):
    """The fp32 kernels against the fp64 graph on the same (fp32-rounded)
    inputs. The values' error is fp32's at a phase near pi (~1e-5 of fp at
    the contacts, as the fp32 graph's); the gradients' is relative to each
    one's largest element."""
    with mx.stream(mx.gpu):
        names, v32, g32 = _kernel_run(mx.float32, oblate, ecc, ss)
    with mx.stream(mx.cpu):
        _, v64, g64 = _kernel_run(mx.float64, oblate, ecc, ss)
    assert np.any(v64 < 0.5 * FP) and np.any((v64 > 0.01 * FP) & (v64 < 0.99 * FP))
    assert np.abs(v32 - v64).max() < 5e-5 * FP
    for name, a, b in zip(["tau"] + names, g32, g64):
        assert np.abs(a - b).max() <= 1e-3 * np.abs(b).max(), name


@needs_metal
def test_kernel_edges():
    """A point planet is always visible, a planet in front of the star or
    clear of it adds exactly nothing, totality is fp (1 + -1), and an
    oblate set outside the domain is NaN alone."""
    with mx.stream(mx.gpu):
        tau = mx.array(np.linspace(-0.5 * PER, 0.5 * PER, 4001), dtype=mx.float32)
        r = mx.array([0.0, 0.1, 0.1, 0.6], dtype=mx.float32)
        f = mx.array([0.2, 0.0, 0.3, 0.5], dtype=mx.float32)
        for kw in ({}, dict(f=f, theta=0.4)):
            def loss(r):
                return mx.sum(el.flux_dev_from_tau(tau, PER, A, 0.0, r,
                                                   transittype="secondary", fp=FP, **kw))
            out = np.asarray(el.flux_dev_from_tau(tau, PER, A, 0.0, r,
                                                  transittype="secondary", fp=FP, **kw))
            g = np.asarray(mx.grad(loss)(r))
            assert np.all(out[0] == np.float32(FP)) and g[0] == 0.0
            front = np.abs(np.asarray(tau)) < 0.25 * PER
            assert np.all(out[1:3][:, front] == np.float32(FP))
            assert np.abs(out[1:3].min(axis=1)).max() < 1e-6 * FP
            if kw:
                assert np.all(np.isnan(out[3])) and np.all(np.isfinite(out[:3]))


@needs_metal
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_transit_model_kernel_matches_its_graph(ecc):
    """TransitModel's fp32 oblate eclipse, the kernel against use_metal=False's
    graph: within two ulps of fp32's 1 (the flux is 1 + fp (...))."""
    p = _params(ecc, 40.0, True)
    t = T0 + 2 * PER + np.linspace(0.0, PER, 3001)
    k = el.TransitModel(p, t, transittype="secondary", dtype=mx.float32).light_curve(p)
    g = el.TransitModel(p, t, transittype="secondary", dtype=mx.float32,
                        use_metal=False).light_curve(p)
    assert k.min() < 1.0 + 0.5 * FP
    assert np.abs(k - g).max() <= 2.0 * 2.0 ** -23


@needs_metal
@pytest.mark.parametrize("ss", [1, 7], ids=["none", "supersample"])
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_transit_model_spherical_eclipse_takes_the_kernel(ecc, ss):
    """TransitModel's fp32 spherical eclipse on the GPU runs the eclipse
    kernel: within two ulps of fp32's 1 of use_metal=False's graph, and
    within fp32's rounding of the fp64 model."""
    p = _params(ecc, 40.0, False)
    t = T0 + 2 * PER + np.linspace(0.0, PER, 3001)
    mkw = dict(supersample_factor=ss, exp_time=EXP) if ss > 1 else {}
    m = el.TransitModel(p, t, transittype="secondary", dtype=mx.float32, **mkw)
    k = m.light_curve(p)
    assert ("sph-tau", "kh" if ecc else "circ") in m._compiled
    g32 = el.TransitModel(p, t, transittype="secondary", dtype=mx.float32,
                          use_metal=False, **mkw)
    g = g32.light_curve(p)
    assert not any(key[0] == "sph-tau" for key in g32._compiled if isinstance(key, tuple))
    d = el.TransitModel(p, t, transittype="secondary", **mkw).light_curve(p)
    assert k.min() < 1.0 + 0.5 * FP
    assert np.abs(k - g).max() <= 2.0 * 2.0 ** -23
    assert np.abs(k - d).max() < 4e-7


@needs_metal
def test_transit_model_spherical_eclipse_batches_and_routes():
    """light_curves runs light_curve's kernel route row by row: equal to the
    loop for circular sets, within two ulps when an eccentric set takes the
    whole batch down the (k, h) orbit. An array ecc keeps its (e, w)
    graph."""
    rng = np.random.default_rng(2)
    t = T0 + 2 * PER + np.linspace(0.0, PER, 1501)

    def sets(eccs):
        out = []
        for e in eccs:
            p = _params(e, rng.uniform(0.0, 360.0), False)
            p.rp, p.inc, p.fp = rng.uniform(0.07, 0.13), rng.uniform(87.5, 88.5), rng.uniform(1e-3, 3e-3)
            out.append(p)
        return out
    m = el.TransitModel(_params(0.0, 90.0, False), t, transittype="secondary",
                        dtype=mx.float32)
    for eccs, exact in (([0.0] * 3, True), ([0.2, 0.4, 0.1], True),
                        ([0.0, 0.3, 0.0, 0.5], False)):
        s = sets(eccs)
        batched = m.light_curves(s)
        looped = np.stack([m.light_curve(p) for p in s])
        if exact:
            assert np.array_equal(batched, looped)
        else:
            assert np.abs(batched - looped).max() <= 2.0 * 2.0 ** -23
    p = _params(0.3, 40.0, False)
    p.ecc = mx.array(0.3, dtype=mx.float32)
    m2 = el.TransitModel(_params(0.3, 40.0, False), t, transittype="secondary",
                         dtype=mx.float32)
    mx.eval(m2.light_curve_mx(p))
    assert "ew" in m2._compiled and not any(
        isinstance(key, tuple) and key[0] == "sph-tau" for key in m2._compiled)


@needs_metal
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_transit_model_spherical_eclipse_gradients(ecc):
    """light_curve_mx's gradients in t0, per, rp, a, fp and inc on the
    kernel route against the fp64 model. Each against a cotangent its
    gradient does not cancel on: a positive bump over the eclipse for rp,
    a, fp and inc (one sign across it), an odd one for t0 and per. Dense
    times: d/drp is singular at the contacts, so a coarse grid leaves it
    to a few points there and their fp32 rounding (the fp32 graph's too)."""
    t = T0 + 2 * PER + np.linspace(0.0, PER, 30001)
    ph = ((t - T0) / PER) % 1.0
    c = float(el.occultation_offset(*_kw(ecc, 40.0, False).values())) if ecc else 0.5
    x = (ph - c) / 0.02
    bump = np.exp(-0.5 * x * x)

    def grads(dt, stream, wgt):
        def loss(t0, per, rp, a, fp, inc):
            q = _params(ecc, 40.0, False)
            q.t0, q.per, q.rp, q.a, q.fp, q.inc = t0, per, rp, a, fp, inc
            m = el.TransitModel(q, t, transittype="secondary", dtype=dt)
            return mx.sum(m.light_curve_mx(q) * mx.array(wgt, dtype=dt))
        with mx.stream(stream):
            xs = [mx.array(v, dtype=dt) for v in (T0, PER, R, A, FP, INC)]
            return dict(zip(["t0", "per", "rp", "a", "fp", "inc"],
                            (float(np.asarray(g, np.float64))
                             for g in mx.grad(loss, argnums=tuple(range(6)))(*xs))))
    for wgt, names in ((bump, ("rp", "a", "fp", "inc")), (bump * x, ("t0", "per"))):
        g32, g64 = grads(mx.float32, mx.gpu, wgt), grads(mx.float64, mx.cpu, wgt)
        for name in names:
            assert abs(g32[name] - g64[name]) <= 1e-3 * abs(g64[name]), name


def test_spherical_eclipse_never_touches_the_oblate_kernels():
    """A spherical eclipse's kernel route -- the sampler API's and
    TransitModel's -- imports none of the oblate or ring code."""
    import subprocess
    import sys
    code = r'''
import sys, numpy as np, mlx.core as mx
import elome as el
tau = mx.array(np.linspace(1.6, 1.85, 101), dtype=mx.float32)
for kw in ({}, dict(secosw=0.2, sesinw=0.1, exp_time=0.02, integration="supersample", n_sub=3)):
    mx.eval(el.flux_dev_from_tau(tau, 3.45, 8.8, 0.3, 0.1, transittype="secondary",
                                 fp=2e-3, **kw))
p = el.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 88.0, 0.0, 90.0
p.limb_dark, p.u, p.fp = "quadratic", [0.4, 0.2], 2e-3
t = np.linspace(1.6, 1.85, 101)
for ecc, ss in ((0.0, 1), (0.2, 3)):
    p.ecc = ecc
    m = el.TransitModel(p, t, transittype="secondary", dtype=mx.float32,
                        supersample_factor=ss, exp_time=0.02)
    m.light_curve(p)
    m.light_curves([p, p])
mods = ("elome.metal_oblate", "elome.metal_rings", "elome.rings_tau")
print("IMPORTED:" + ",".join(m for m in mods if m in sys.modules))
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("IMPORTED:"), out.stdout


# ---------------------------------------------------------------------------
# the contact rule: a spherical planet's eclipse is the transit of the orbit
# turned by pi (eclipse._eclipse_dev_contact)
# ---------------------------------------------------------------------------

_ORBITS = [(0.0, 90.0), (0.3, 40.0), (0.6, -70.0)]


def _around_eclipse(ecc, w, half=0.15, n=601):
    k = math.sqrt(ecc) * math.cos(math.radians(w))
    h = math.sqrt(ecc) * math.sin(math.radians(w))
    c = float(el.occultation_offset(k, h)) * PER
    tau = c + np.linspace(-half, half, n)
    return tau - PER * np.round(tau / PER)


def _contact(tau, ecc, w, r=R, **kw):
    return el.flux_dev_from_tau(tau, PER, A, _b(ecc, w), r, transittype="secondary",
                                fp=FP, exp_time=EXP, **_kw(ecc, w, False), **kw)


@pytest.mark.parametrize("ecc,w", _ORBITS)
def test_contact_rule_converges_to_the_exposure_average(ecc, w):
    """fp64: n_gl = 7 is within 1e-5 of the depth of n_gl = 40, which a
    dense supersample of the eclipse's own graph approaches as O(1/N)."""
    tau = _around_eclipse(ecc, w)
    ref = np.asarray(_contact(tau, ecc, w, n_gl=40))
    assert np.abs(np.asarray(_contact(tau, ecc, w, n_gl=7)) - ref).max() < 1e-5 * FP
    ss = np.asarray(_contact(tau, ecc, w, integration="supersample", n_sub=2001))
    assert np.abs(ss - ref).max() < 2e-3 * FP


@pytest.mark.parametrize("ecc,w", _ORBITS)
def test_contact_rule_is_transit_model_s(ecc, w):
    """The two implementations agree: TransitModel's graph splits about the
    eclipse on the orbit itself (api._eclipse_split), the sampler API runs
    the flipped transit; the same nodes, so the same flux to rounding."""
    t = T0 + _around_eclipse(ecc, w)
    p = _params(ecc, w, False)
    m = el.TransitModel(p, t, transittype="secondary", exp_time=EXP,
                        integration="contact", n_gl=7)
    got = 1.0 + np.asarray(_contact(t - T0, ecc, w, n_gl=7))
    assert np.abs(m.light_curve(p) - got).max() < 1e-12 * FP


def test_contact_rule_point_planet():
    """r = 0: always visible, F - 1 = fp exactly, with finite gradients."""
    tau = _around_eclipse(0.3, 40.0)
    assert np.all(np.asarray(_contact(tau, 0.3, 40.0, r=0.0)) == FP)
    g = mx.grad(lambda r: mx.sum(_contact(mx.array(tau), 0.3, 40.0, r=r)))(mx.array(0.0))
    assert np.isfinite(float(g))


@needs_metal
@pytest.mark.parametrize("r", [0.01, R, 1.5])
@pytest.mark.parametrize("ecc,w", _ORBITS)
def test_contact_rule_fp32_matches_fp64(ecc, w, r):
    """The fp32 kernel route within 1e-4 of the depth of fp64 (measured
    1-5e-5: fp32's floor, against 2-11% for 15-sample supersampling)."""
    tau = _around_eclipse(ecc, w, half=0.3 if r > 1 else 0.15)
    f64 = np.asarray(_contact(tau, ecc, w, r=r, n_gl=7))
    f32 = np.asarray(_contact(mx.array(tau, dtype=mx.float32), ecc, w, r=r, n_gl=7),
                     np.float64)
    assert np.abs(f32 - f64).max() < 1e-4 * FP


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
@pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 40.0)])
def test_contact_rule_gradients(ecc, w, dtype):
    """Gradients in every parameter. The contacts are detached split points
    (as the transit's), so a gradient is that of the quadrature with its
    split held, which converges to the exposure average's with n_gl: fp64
    at n_gl = 40 against central differences of itself, at n_gl = 7 within
    2e-3 of them; fp32 against fp64 at n_gl = 7. A positive bump over the
    eclipse as cotangent, so no gradient cancels to nothing."""
    if dtype == mx.float32 and not el.metal_available():
        pytest.skip("the fp32 route needs a GPU")
    x = np.linspace(-0.15, 0.15, 4001)
    tau = _around_eclipse(ecc, w, n=4001)
    bump = np.exp(-0.5 * (x / 0.06) ** 2)
    names = ["per", "a", "b", "r", "fp"] + (["secosw", "sesinw"] if ecc else [])
    v0 = dict(per=PER, a=A, b=_b(ecc, w), r=R, fp=FP, **_kw(ecc, w, False))

    def total(dt, n_gl, *vals):
        d = dict(zip(names, vals))
        out = el.flux_dev_from_tau(mx.array(tau, dtype=dt), d.pop("per"), d.pop("a"),
                                   d.pop("b"), d.pop("r"), transittype="secondary",
                                   exp_time=EXP, n_gl=n_gl, **d)
        return mx.sum(mx.array(bump, dtype=dt) * out)

    def grads(dt, n_gl):
        args = [mx.array(v0[k], dtype=dt) for k in names]
        g = mx.grad(lambda *a: total(dt, n_gl, *a), argnums=tuple(range(len(names))))(*args)
        return np.array([float(x) for x in g])

    with mx.stream(mx.cpu):
        g64 = grads(mx.float64, 7)
        if dtype == mx.float64:
            g40 = grads(mx.float64, 40)
            for j, k in enumerate(names):
                hstep = 1e-6 * max(abs(v0[k]), 1e-3)
                hi = [v0[n] + (hstep if n == k else 0.0) for n in names]
                lo = [v0[n] - (hstep if n == k else 0.0) for n in names]
                fd = (float(total(mx.float64, 40, *hi))
                      - float(total(mx.float64, 40, *lo))) / (2 * hstep)
                assert abs(g40[j] - fd) <= 1e-5 * abs(fd) + 1e-9, k
                assert abs(g64[j] - fd) <= 2e-3 * abs(fd) + 1e-9, k
    if dtype == mx.float32:
        g32 = grads(mx.float32, 7)
        assert np.all(np.abs(g32 - g64) <= 2e-3 * np.abs(g64) + 1e-9), (names, g32, g64)


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_contact_rule_compiled_equals_eager(dtype):
    if dtype == mx.float32 and not el.metal_available():
        pytest.skip("the fp32 route needs a GPU")
    tau = mx.array(_around_eclipse(0.3, 40.0), dtype=dtype)
    k, h = (mx.array(v, dtype=dtype) for v in _kw(0.3, 40.0, False).values())

    def f(r, fp, kk, hh):
        return mx.sum(el.flux_dev_from_tau(tau, PER, A, _b(0.3, 40.0), r, transittype="secondary",
                                           fp=fp, secosw=kk, sesinw=hh, exp_time=EXP))
    args = [mx.array(R, dtype=dtype), mx.array(FP, dtype=dtype), k, h]
    ctx = mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.gpu)
    with ctx:
        vg = mx.value_and_grad(f, argnums=(0, 1, 2, 3))
        e_v, e_g = vg(*args)
        c_v, c_g = mx.compile(vg)(*args)
        tol = 1e-12 if dtype == mx.float64 else 1e-5      # as the test above
        assert float(mx.abs(e_v - c_v)) <= tol * max(abs(float(e_v)), 1.0)
        for a_, b_ in zip(e_g, c_g):
            assert np.isfinite(float(a_))
            assert float(mx.abs(a_ - b_)) <= tol * max(abs(float(a_)), 1.0)


def _model(ecc, w, **kw):
    p = _params(ecc, w, False)
    t = T0 + _around_eclipse(ecc, w)
    return p, t, el.TransitModel(p, t, transittype="secondary", exp_time=EXP,
                                 integration="contact", n_gl=7, **kw)


@needs_metal
@pytest.mark.parametrize("ecc,w", _ORBITS)
def test_transit_model_eclipse_contact_routes(ecc, w):
    """fp32 takes the tau kernels; fp32 without Metal the graph; both within
    1e-4 of the depth of fp64, plus the rounding of a flux near 1 + fp
    (two fp32 ulps of 1). An array ecc takes the (e, w) graph and agrees
    with the (k, h) one to rounding."""
    tol = 1e-4 * FP + 2 * float(np.finfo(np.float32).eps)
    p, t, m64 = _model(ecc, w)
    f64 = m64.light_curve(p)
    _, _, m32 = _model(ecc, w, dtype=mx.float32)
    f32 = m32.light_curve(p)
    assert list(m32._compiled) == [("sph-tau", "circ" if ecc == 0.0 else "kh")]
    assert np.abs(f32 - f64).max() < tol
    _, _, mg = _model(ecc, w, dtype=mx.float32, use_metal=False)
    assert np.abs(mg.light_curve(p) - f64).max() < tol
    q = _params(mx.array(ecc, dtype=mx.float64), mx.array(w, dtype=mx.float64), False)
    with mx.stream(mx.cpu):
        ew = np.asarray(m64.light_curve_mx(q))
    assert np.abs(ew - f64).max() < 1e-12 * FP


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
@pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 40.0)])
def test_transit_model_eclipse_contact_batches(ecc, w, dtype):
    if dtype == mx.float32 and not el.metal_available():
        pytest.skip("the fp32 route needs a GPU")
    p, t, m = _model(ecc, w, dtype=dtype)
    pa = _params(ecc, w, False)
    pa.rp, pa.fp = np.array([0.08, 0.1, 0.12]), np.array([1e-3, 2e-3, 5e-4])
    got = m.light_curves(pa)
    for j in range(3):
        q = _params(ecc, w, False)
        q.rp, q.fp = float(pa.rp[j]), float(pa.fp[j])
        tol = 1e-15 if dtype == mx.float64 else 4 * float(np.finfo(np.float32).eps)
        assert np.abs(got[j] - m.light_curve(q)).max() <= tol, j


def test_transit_model_eclipse_contact_de_at_zero():
    """The (e, w) route's d/de at e = 0 is finite and is the one-sided
    derivative (the occultation offset from e cos w, no sqrt(e))."""
    p, t, m = _model(0.0, 90.0)
    with mx.stream(mx.cpu):
        def f(e):
            q = _params(e, mx.array(30.0, dtype=mx.float64), False)
            return mx.sum(m.light_curve_mx(q))
        g = float(mx.grad(f)(mx.array(0.0, dtype=mx.float64)))
        fd = (float(f(mx.array(1e-7, dtype=mx.float64)))
              - float(f(mx.array(0.0, dtype=mx.float64)))) / 1e-7
    assert np.isfinite(g) and abs(g - fd) <= 1e-4 * abs(fd)
