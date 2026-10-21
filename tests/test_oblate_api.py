"""Oblate planets: TransitParams.f / .theta through TransitModel.

The model forms time since each point's own mid-transit and calls
flux_dev_from_tau(f=, theta=) (the oblate kernels on an fp32 GPU model,
the exact graph otherwise), with its own exposure rule. Pinned against an
independent reference (a numpy orbit and a polar-ray quadrature,
tests/oblate_oracle.py) in absolute times, against the spherical model at
f = 0, across precisions and surfaces, and in its
gradients.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import elome as el
from elome import metal as M

import oblate_oracle as OR

KERNEL = M.metal_available() and M._gpu_stream_active()

W5 = [0.2, 0.2, 0.1, 0.1, 0.1]
T0, PER = 0.3, 3.45
T = T0 + 2 * PER + np.linspace(-0.15, 0.15, 241)
EXP = 0.02


def P(**kw):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, 0.1, 8.8, 88.0, 0.0, 90.0
    p.limb_dark, p.u = "hybrid5", list(W5)
    p.f, p.theta = 0.3, 35.0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def spherical(**kw):
    p = P(**kw)
    p.f = p.theta = None
    return p


@pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 40.0), (0.6, -70.0)])
def test_matches_the_reference(ecc, w):
    """Absolute times, t0 the time of inferior conjunction, inc and w in
    degrees, theta in degrees."""
    p = P(ecc=ecc, w=w)
    got = el.TransitModel(p, T).light_curve(p)
    X, Y, front = OR.sky_position(T - T0, PER, 8.8, math.radians(88.0), ecc, math.radians(w))
    th = math.radians(35.0)
    x0, y0 = X * math.cos(th) + Y * math.sin(th), -X * math.sin(th) + Y * math.cos(th)
    A = 0.1 / math.sqrt(0.7)
    ref = np.ones(len(T))
    near = front & (np.hypot(x0, y0) < 1 + A)
    ref[near] += OR.law_flux_dev("hybrid5", W5, x0[near], y0[near], A, 0.7 * A)
    assert 1.0 - ref.min() > 1e-3
    assert np.abs(got - ref).max() < 1e-13


def test_theta_is_in_degrees_and_defaults_to_zero():
    p = P(theta=35.0)
    got = el.TransitModel(p, T).light_curve(p)
    tau = (T - T0) - PER * np.round((T - T0) / PER)
    ref = 1.0 + np.asarray(el.flux_dev_from_tau(
        tau, PER, 8.8, 8.8 * math.cos(math.radians(88.0)), 0.1, limb_dark="hybrid5",
        u=W5, f=0.3, theta=math.radians(35.0)))
    assert np.abs(got - ref).max() < 1e-15
    a, b = P(theta=None), P(theta=0.0)
    assert np.array_equal(el.TransitModel(a, T).light_curve(a),
                          el.TransitModel(b, T).light_curve(b))


@pytest.mark.parametrize("kw", [{}, {"supersample_factor": 7, "exp_time": EXP},
                                {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "supersample", "contact"])
def test_f_zero_is_the_spherical_model(kw):
    o, s = P(f=0.0), spherical()
    got = el.TransitModel(o, T, **kw).light_curve(o)
    ref = el.TransitModel(s, T, **kw).light_curve(s)
    assert np.abs(got - ref).max() < 1e-14


@pytest.mark.parametrize("kw", [{}, {"supersample_factor": 7, "exp_time": EXP},
                                {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "supersample", "contact"])
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_fp32_model_matches_fp64(kw, ecc):
    p = P(ecc=ecc, w=40.0)
    d64 = el.TransitModel(p, T, **kw).light_curve(p)
    d32 = el.TransitModel(p, T, dtype=mx.float32, **kw).light_curve(p)
    assert np.abs(d32 - d64).max() < 3e-7


@pytest.mark.skipif(not KERNEL, reason="needs Metal and the GPU stream")
def test_fp32_model_takes_the_oblate_kernel():
    from elome import metal_oblate as MO
    seen = []
    orig = MO._make_core

    def spy(*a, **k):
        seen.append(1)
        return orig(*a, **k)
    MO._make_core = spy
    try:
        p = P(rp=0.1013)               # a fresh value, not a cached trace
        el.TransitModel(p, T, dtype=mx.float32).light_curve(p)
    finally:
        MO._make_core = orig
    assert seen


@pytest.mark.parametrize("kw", [{}, {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "contact"])
def test_light_curves_equal_the_loop(kw):
    m = el.TransitModel(P(), T, **kw)
    sets = [P(f=0.1, theta=0.0), P(f=0.4, theta=80.0, ecc=0.2, w=10.0),
            P(rp=0.12, f=0.2, theta=150.0, u=[0.3, 0.1, 0.1, 0.05, 0.05])]
    loop = np.stack([m.light_curve(q) for q in sets])
    assert np.abs(m.light_curves(sets) - loop).max() < 1e-15
    pa = P()
    pa.f, pa.theta = np.array([0.1, 0.4, 0.2]), np.array([0.0, 80.0, 150.0])
    pa.rp = np.array([0.1, 0.1, 0.12])
    arr = m.light_curves(pa)
    ref = np.stack([m.light_curve(P(f=f, theta=t, rp=r))
                    for f, t, r in zip(pa.f, pa.theta, pa.rp)])
    assert np.abs(arr - ref).max() < 1e-15


def test_light_curve_mx_equals_light_curve_and_differentiates():
    p = P(ecc=0.2, w=30.0)
    ct = np.random.default_rng(0).normal(size=T.shape)
    with mx.stream(mx.cpu):
        m = el.TransitModel(p, T)
        assert np.abs(np.asarray(m.light_curve_mx(p)) - m.light_curve(p)).max() < 1e-15
        names = ("f", "theta", "rp", "t0", "inc", "ecc", "w")
        x = [0.3, 35.0, 0.1, T0, 88.0, 0.2, 30.0]

        def loss(*v):
            q = P()
            for nm, val in zip(names, v):
                setattr(q, nm, val)
            q.u = [v[-1]] + W5[1:]
            return mx.sum(mx.array(ct) * m.light_curve_mx(q))
        args = [mx.array(v, dtype=mx.float64) for v in x + [W5[0]]]
        g = mx.grad(loss, argnums=tuple(range(8)))(*args)
        for i in range(8):
            def fd(h):
                up = [mx.array(a) for a in args]
                dn = [mx.array(a) for a in args]
                up[i] = args[i] + h
                dn[i] = args[i] - h
                return (float(loss(*up)) - float(loss(*dn))) / (2 * h)
            h = 1e-4 * max(abs(float(args[i])), 0.01)
            ref = (4 * fd(h / 2) - fd(h)) / 3
            assert abs(float(g[i]) - ref) <= 2e-4 * max(abs(ref), 1e-6), i


def test_errors():
    with pytest.raises(ValueError, match="hybrid law"):
        el.TransitModel(P(limb_dark="quadratic", u=[0.4, 0.2]), T)
    with pytest.raises(ValueError, match="set params.f too"):
        q = spherical()
        q.theta = 30.0
        el.TransitModel(q, T)
    with pytest.raises(ValueError, match="primary-transit only"):
        el.TransitModel(P(fp=1e-4), T, transittype="secondary")
    m = el.TransitModel(P(), T)
    with pytest.raises(ValueError, match="build a new TransitModel"):
        m.light_curve(spherical())
    ms = el.TransitModel(spherical(), T)
    with pytest.raises(ValueError, match="build a new TransitModel"):
        ms.light_curve(P())
    with pytest.raises(ValueError, match="r_eff"):
        m.light_curve(P(rp=0.6, f=0.5))


def test_out_of_domain_array_is_nan():
    with mx.stream(mx.cpu):
        m = el.TransitModel(P(), T)
        out = np.asarray(m.light_curve_mx(P(rp=mx.array(0.6, dtype=mx.float64),
                                            f=mx.array(0.5, dtype=mx.float64))))
    assert np.isnan(out[len(T) // 2])


def test_spherical_models_unchanged_by_the_new_fields():
    """A TransitParams without f / theta attributes still works, and a
    spherical model's batch keys are the spherical ones."""
    p = spherical()
    del p.f, p.theta
    m = el.TransitModel(p, T)
    assert m._batch_keys == el.TransitModel._BATCH_KEYS
    ref = el.TransitModel(spherical(), T).light_curve(spherical())
    assert np.array_equal(m.light_curve(p), ref)


# ---------------------------------------------------------------------------
# review fixes: regressions
# ---------------------------------------------------------------------------

def _one_sided_fd_ecc(m, w, h=1e-6):
    """d/d(ecc) at e = 0 from the right, second order, fp64."""
    def L(e):
        return float(mx.sum(m.light_curve_mx(P(ecc=mx.array(e, dtype=mx.float64),
                                               w=w))))
    return (-3 * L(0.0) + 4 * L(h) - L(2 * h)) / (2 * h)


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_array_ecc_gradient_at_zero_is_the_one_sided_derivative(dtype):
    """An array ecc enters as (e, w), exact at e = 0 (going through
    sqrt(e) and would return -0.06 against the true +1.02 here)."""
    with mx.stream(mx.cpu):
        ref = _one_sided_fd_ecc(el.TransitModel(P(w=40.0), T), 40.0)
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        m = el.TransitModel(P(w=40.0), T, dtype=dtype)
        g = float(mx.grad(lambda e: mx.sum(m.light_curve_mx(P(ecc=e, w=40.0))))(
            mx.array(0.0, dtype=dtype)))
    tol = 1e-6 if dtype == mx.float64 else 1e-4
    assert abs(g - ref) <= tol * abs(ref)


def test_array_and_python_ecc_routes_agree():
    with mx.stream(mx.cpu):
        m = el.TransitModel(P(), T)
        for e, w in ((0.2, 30.0), (0.6, -70.0)):
            arr = np.asarray(m.light_curve_mx(P(ecc=mx.array(e, dtype=mx.float64), w=w)))
            assert np.abs(arr - m.light_curve(P(ecc=e, w=w))).max() < 1e-13


def test_array_ecc_out_of_range_raises_or_is_nan():
    with mx.stream(mx.cpu):
        m = el.TransitModel(P(), T)
        for e in (1.5, -0.2, 1.0):
            with pytest.raises(ValueError, match="eccentricity"):
                m.light_curve_mx(P(ecc=mx.array(e, dtype=mx.float64)))
        comp = mx.compile(lambda e: m.light_curve_mx(P(ecc=e)))
        assert np.isnan(np.asarray(comp(mx.array(1.5, dtype=mx.float64)))).all()
        assert np.isfinite(np.asarray(comp(mx.array(0.3, dtype=mx.float64)))).all()


def test_ecc_above_0999_raises_or_is_nan():
    """The orbit is rebuilt from the impact parameter, whose cos i holds to
    e = 0.999; above it the curve was quietly wrong (2% of the depth at
    e = 0.9995). The spherical model is exact there and still accepts it."""
    kw = dict(ecc=0.9995, w=300.0, inc=87.0)
    with mx.stream(mx.cpu):
        m = el.TransitModel(P(ecc=0.99, w=300.0, inc=87.0), T)
        with pytest.raises(ValueError, match=r"\[0, 0.999\]"):
            m.light_curve(P(**kw))
        with pytest.raises(ValueError, match=r"\[0, 0.999\]"):
            m.light_curves([P(**dict(kw, ecc=0.5)), P(**kw)])
        with pytest.raises(ValueError, match=r"\[0, 0.999\]"):
            m.light_curve_mx(P(**dict(kw, ecc=mx.array(0.9995, dtype=mx.float64))))
        comp = mx.compile(lambda e: m.light_curve_mx(P(**dict(kw, ecc=e))))
        assert np.isnan(np.asarray(comp(mx.array(0.9995, dtype=mx.float64)))).all()
        assert np.isfinite(np.asarray(comp(mx.array(0.999, dtype=mx.float64)))).all()
        assert np.isfinite(m.light_curve(P(**dict(kw, ecc=0.999)))).all()
        s = spherical(**kw)
        assert np.isfinite(el.TransitModel(s, T).light_curve(s)).all()


@pytest.mark.skipif(not KERNEL, reason="needs Metal and the GPU stream")
@pytest.mark.parametrize("kw", [{}, {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "contact"])
def test_use_metal_false_runs_the_graph(kw):
    from elome import metal_oblate as MO
    seen = []
    orig = MO.flux_dev_from_tau_oblate_kernel

    def spy(*a, **k):
        seen.append(1)
        return orig(*a, **k)
    MO.flux_dev_from_tau_oblate_kernel = spy
    try:
        p = P(rp=0.1021)
        off = el.TransitModel(p, T, dtype=mx.float32, use_metal=False, **kw).light_curve(p)
        assert not seen
        on = el.TransitModel(p, T, dtype=mx.float32, **kw).light_curve(p)
        assert seen
    finally:
        MO.flux_dev_from_tau_oblate_kernel = orig
    assert np.abs(on - off).max() < 3e-7


def _spy_graph(monkeypatch):
    """The (n, m) of every block oblate_tau's graph evaluates."""
    from elome import oblate_tau
    shapes, orig = [], oblate_tau._tau_graph

    def spy(tau, *a, **k):
        shapes.append(tuple(tau.shape))
        return orig(tau, *a, **k)
    monkeypatch.setattr(oblate_tau, "_tau_graph", spy)
    return shapes


@pytest.mark.parametrize("pkw,mkw", [
    ({}, {}), (dict(ecc=0.3, w=40.0), {}),
    ({}, dict(integration="contact", exp_time=EXP, n_gl=2)),
    (dict(ecc=0.3, w=40.0), dict(integration="contact", exp_time=EXP, n_gl=2)),
    ({}, dict(supersample_factor=3, exp_time=EXP))],
    ids=["circular", "eccentric", "contact", "eccentric contact", "supersampled"])
def test_long_grids_are_evaluated_in_blocks(pkw, mkw, monkeypatch):
    """The fp64 graph holds ~1-3 KB per evaluation point until evaluated,
    so it is built in blocks of at most _OBLATE_BLOCK points (times x sets x
    exposure nodes). flux_dev_from_tau gives the same curve bit for bit;
    TransitModel, whose blocked call compiles one block instead of the
    whole call, to rounding."""
    from elome import oblate_tau
    p = P(**pkw)
    m = el.TransitModel(p, T, **mkw)
    sets = [p, P(**{**pkw, "f": 0.1, "theta": 80.0, "rp": 0.08})]
    whole, whole_sets = m.light_curve(p), m.light_curves(sets)
    nodes = 5 * 2 if mkw.get("integration") == "contact" else 1
    monkeypatch.setattr(oblate_tau, "_OBLATE_BLOCK", 50 * nodes)
    assert np.abs(m.light_curve(p) - whole).max() < 1e-15
    assert np.abs(m.light_curves(sets) - whole_sets).max() < 1e-15

    tau = np.linspace(-0.15, 0.15, 230)
    kw = dict(limb_dark="hybrid5", f=0.3, theta=0.6, exp_time=EXP,
              integration=mkw.get("integration", "none"), n_gl=2)
    if "ecc" in pkw:
        kw.update(secosw=0.4, sesinw=0.3)
    with mx.stream(mx.cpu):
        t2 = mx.broadcast_to(mx.array(tau, dtype=mx.float64), (2, tau.size))
        rp = mx.array([0.1, 0.08], dtype=mx.float64)
        monkeypatch.setattr(oblate_tau, "_OBLATE_BLOCK", 1 << 18)
        ref = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rp, u=W5, **kw))
        monkeypatch.setattr(oblate_tau, "_OBLATE_BLOCK", 50 * nodes)
        shapes = _spy_graph(monkeypatch)
        got = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rp, u=W5, **kw))
    assert np.array_equal(got, ref)
    assert shapes == 2 * [(1, 46)] * 5      # each set's 230 times, 5 equal blocks


def test_many_short_rows_are_blocked_by_sets(monkeypatch):
    """Sets that fit a block whole go in groups of rows (padded with the
    last), bit for bit; the basis columns are blocked as well."""
    from elome import oblate_tau
    tau = np.linspace(-0.15, 0.15, 9)
    rp = 0.06 + 0.004 * np.arange(7)
    kw = dict(limb_dark="hybrid5", f=0.2, theta=0.4)
    with mx.stream(mx.cpu):
        t2 = mx.broadcast_to(mx.array(tau, dtype=mx.float64), (7, 9))
        rpa = mx.array(rp, dtype=mx.float64)
        whole = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, u=W5, **kw))
        basis = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, ld_basis=True, **kw))
        monkeypatch.setattr(oblate_tau, "_OBLATE_BLOCK", 20)
        shapes = _spy_graph(monkeypatch)
        assert np.array_equal(np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, u=W5,
                                                            **kw)), whole)
        assert shapes == [(2, 9)] * 4       # 7 rows of 9 points, 2 per block
        assert np.array_equal(np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa,
                                                            ld_basis=True, **kw)), basis)
    assert whole.shape == (7, 9) and basis.shape == (7, 9, 6)
