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
    with pytest.raises(ValueError, match="integration='supersample'"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, R, exp_time=EXP, **kw)
    with pytest.raises(ValueError, match="r_eff"):
        el.flux_dev_from_tau(TAU, PER, A, 0.3, 0.6, f=0.5, **kw)
    # exp_time = 0 makes the default contact rule instantaneous: allowed
    el.flux_dev_from_tau(TAU, PER, A, 0.3, R, integration="contact", **kw)
