"""Ringed planets through TransitModel and flux_dev_from_tau.

The model forms time since each point's own mid-transit and evaluates
rings_tau's graph, with its own exposure rule (the contact rule split at
the kinks inside each exposure). Pinned against elome.rings at sky
positions from an independent numpy orbit (tests/oblate_oracle.py) in
absolute times, against the bare planet for a transparent ring, against
converged supersampling for the exposure rule, across surfaces and
precisions, and in its gradients.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome as el
from elome import rings, rings_tau

import oblate_oracle as OR

W5 = [0.2, 0.2, 0.1, 0.1, 0.1]
T0, PER = 0.3, 3.45
T = T0 + 2 * PER + np.linspace(-0.2, 0.2, 121)
EXP = 0.02


def P(**kw):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, 0.08, 8.8, 88.5, 0.0, 90.0
    p.limb_dark, p.u = "hybrid5", list(W5)
    p.theta, p.ring_in, p.ring_out, p.ring_f = 25.0, 1.4, 2.6, 0.7
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def reference(p, t):
    """elome.rings at sky positions from the numpy orbit."""
    X, Y, front = OR.sky_position(t - p.t0, p.per, p.a, math.radians(p.inc), p.ecc,
                                  math.radians(p.w))
    th = math.radians(p.theta or 0.0)
    x0 = X * math.cos(th) + Y * math.sin(th)
    y0 = -X * math.sin(th) + Y * math.cos(th)
    tau = math.inf if p.ring_tau is None else p.ring_tau
    with mx.stream(mx.cpu):
        a = lambda v: mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)
        dev = rings.flux_dev_ring(a(np.where(front, x0, 9.0)), a(np.where(front, y0, 0.0)),
                                  p.rp, p.f or 0.0, p.ring_in, p.ring_out, p.ring_f, tau,
                                  p.u, p.limb_dark)
        return 1.0 + np.array(dev)


@pytest.mark.parametrize("kw", [dict(), dict(ecc=0.3, w=40.0), dict(f=0.2, ring_tau=0.4),
                                dict(ring_f=0.95, ring_out=4.0, inc=89.6)],
                         ids=["circular", "eccentric", "oblate, translucent", "edge-on, wide"])
def test_matches_the_reference(kw):
    """Absolute times, t0 the time of inferior conjunction, inc, w and
    theta in degrees."""
    p = P(**kw)
    got = el.TransitModel(p, T).light_curve(p)
    ref = reference(p, T)
    assert 1.0 - ref.min() > 1e-3
    assert np.abs(got - ref).max() < 1e-14


@pytest.mark.parametrize("kw", [{}, {"supersample_factor": 5, "exp_time": EXP}],
                         ids=["none", "supersample"])
@pytest.mark.parametrize("f", [None, 0.2])
def test_transparent_ring_is_the_bare_planet(kw, f):
    """ring_tau = 0: the spherical or oblate model, to rounding."""
    p = P(ring_tau=0.0, f=f)
    bare = P(f=f)
    bare.ring_in = bare.ring_out = bare.ring_f = None
    if f is None:
        bare.theta = None
    got = el.TransitModel(p, T, **kw).light_curve(p)
    ref = el.TransitModel(bare, T, **kw).light_curve(bare)
    assert np.abs(got - ref).max() < 1e-14


def test_contact_rule_converges():
    """A nearly edge-on ring wider than its distance to the limb, whose
    outer edge crosses it four times: the contact rule, split at each
    exposure's kinks, against supersampling extrapolated in 1/N (which
    converges only that slowly across a kink)."""
    p = P(ring_f=0.95, ring_out=4.0, inc=89.6)
    t = T0 + 2 * PER + np.linspace(-0.17, 0.17, 41)
    L = lambda **kw: el.TransitModel(p, t, exp_time=EXP, **kw).light_curve(p)
    c5 = L(integration="contact")
    s1, s2 = (L(integration="supersample", supersample_factor=n) for n in (801, 3201))
    ext = s2 + (s2 - s1) / 3.0
    assert 1.0 - c5.min() > 1e-3
    assert np.abs(c5 - ext).max() < 2e-7


def test_kinks_are_where_the_outlines_touch_the_limb():
    """A spherical planet with a face-on ring: every outline is a circle,
    whose contacts are elementary -- sin^2 phi = ((1 +- R)^2 - b^2)/(a^2 - b^2)."""
    a, b, rp, ri, ro = 8.8, 0.3, 0.08, 1.5, 3.0
    k = np.array(rings_tau.ring_kinks(a, b, rp, 0.0, 0.0, ri, ro, 0.0))[0]
    want = []
    for R in (rp, ri * rp, ro * rp):
        for Z in (1 + R, 1 - R):
            s2 = (Z * Z - b * b) / (a * a - b * b)
            want += [-math.asin(math.sqrt(s2)), math.asin(math.sqrt(s2))]
    assert np.allclose(np.sort(k[np.isfinite(k)]), np.sort(want), atol=1e-10)


def test_theta_is_in_degrees_and_matches_flux_dev_from_tau():
    p = P(theta=35.0)
    got = el.TransitModel(p, T).light_curve(p)
    tau = (T - T0) - PER * np.round((T - T0) / PER)
    ref = 1.0 + np.asarray(el.flux_dev_from_tau(
        tau, PER, 8.8, 8.8 * math.cos(math.radians(88.5)), 0.08, limb_dark="hybrid5",
        u=W5, theta=math.radians(35.0), ring_in=1.4, ring_out=2.6, ring_f=0.7))
    assert np.abs(got - ref).max() < 1e-15


def test_ld_basis_contract():
    tau = np.linspace(-0.15, 0.15, 31)
    kw = dict(limb_dark="hybrid4", ring_in=1.4, ring_out=2.6, ring_f=0.8, ring_tau=0.5,
              theta=0.3, exp_time=EXP)
    B = np.array(el.flux_dev_from_tau(tau, PER, 8.8, 0.25, 0.08, ld_basis=True, **kw))
    w = [0.2, 0.2, 0.1, 0.1]
    dev = np.array(el.flux_dev_from_tau(tau, PER, 8.8, 0.25, 0.08, u=w, **kw))
    c = np.concatenate([[1.0], -np.asarray(w)])
    N = el.hybrid_norms("hybrid4")
    assert np.abs(dev - (B @ c) / (N @ c)).max() < 1e-16


@pytest.mark.parametrize("kw", [{}, {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "contact"])
def test_light_curves_equal_the_loop(kw):
    m = el.TransitModel(P(), T, **kw)
    sets = [P(), P(ring_f=0.9, ring_tau=0.3, theta=80.0, ecc=0.2, w=10.0),
            P(rp=0.07, ring_in=1.2, ring_out=3.0, u=[0.3, 0.1, 0.1, 0.05, 0.05])]
    loop = np.stack([m.light_curve(q) for q in sets])
    assert np.abs(m.light_curves(sets) - loop).max() < 1e-15
    pa = P()
    pa.ring_f, pa.ring_tau = np.array([0.7, 0.9, 0.5]), np.array([np.inf, 0.3, 1.0])
    pa.theta = np.array([25.0, 80.0, 150.0])
    arr = m.light_curves(pa)
    ref = np.stack([m.light_curve(P(ring_f=f, ring_tau=t, theta=th))
                    for f, t, th in zip(pa.ring_f, pa.ring_tau, pa.theta)])
    assert np.abs(arr - ref).max() < 1e-15


def test_light_curve_mx_differentiates_in_the_ring():
    """Gradients in every ring field and theta, vs central differences."""
    base = dict(ring_in=1.4, ring_out=2.6, ring_f=0.8, ring_tau=0.6, theta=25.0, rp=0.08)
    t = T0 + 2 * PER + np.linspace(-0.12, 0.12, 25)
    ct = np.random.default_rng(0).normal(size=t.shape)
    m = el.TransitModel(P(**base), t)

    def loss(vals):
        p = P(**{k: v if isinstance(v, mx.array) else mx.array(v, dtype=mx.float64)
                 for k, v in vals.items()})
        return mx.sum(m.light_curve_mx(p) * mx.array(ct))
    with mx.stream(mx.cpu):
        g = mx.grad(loss)({k: mx.array(v, dtype=mx.float64) for k, v in base.items()})
        for k, v in base.items():
            h = 1e-6 * abs(v)
            up, dn = dict(base), dict(base)
            up[k], dn[k] = v + h, v - h
            fd = (loss(up).item() - loss(dn).item()) / (2 * h)
            assert abs(g[k].item() - fd) < 1e-6 * max(abs(fd), 1e-3), k


@pytest.mark.parametrize("kw", [{}, {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "contact"])
def test_fp32_model_matches_fp64(kw):
    p = P(ecc=0.2, w=40.0, ring_tau=0.5)
    d64 = el.TransitModel(p, T, **kw).light_curve(p)
    d32 = el.TransitModel(p, T, dtype=mx.float32, **kw).light_curve(p)
    assert np.abs(d32 - d64).max() < 3e-7



def _spy_graph(monkeypatch):
    """The (sets, times) of each ring-graph block, in call order."""
    shapes, orig = [], rings_tau._tau_graph

    def spy(tau, *a, **kw):
        shapes.append(tuple(tau.shape))
        return orig(tau, *a, **kw)
    monkeypatch.setattr(rings_tau, "_tau_graph", spy)
    return shapes


def _blocks(total, most):
    nb = -(-total // most)
    return [-(-total // nb)] * nb          # equal blocks, the last padded


@pytest.mark.parametrize("pkw,mkw", [
    ({}, {}), (dict(ecc=0.3, w=40.0), {}), (dict(f=0.2, ring_tau=0.4), {}),
    ({}, dict(integration="contact", exp_time=EXP, n_gl=2)),
    ({}, dict(supersample_factor=3, exp_time=EXP)),
    (dict(ecc=0.2, w=40.0), dict(dtype=mx.float32, use_metal=False))],
    ids=["circular", "eccentric", "oblate, translucent", "contact", "supersampled",
         "fp32 graph"])
def test_long_grids_are_evaluated_in_blocks(pkw, mkw, monkeypatch):
    """The ring graph holds ~20 KB per evaluation point until evaluated, so
    it is built in blocks of at most _RING_BLOCK points -- times x sets x
    exposure nodes: the same curve, bit for bit."""
    p = P(**pkw)
    m = el.TransitModel(p, T, **mkw)
    sets = [p, P(**{**pkw, "ring_f": 0.9, "ring_tau": 0.3, "theta": 80.0})]
    whole, whole_sets = m.light_curve(p), m.light_curves(sets)
    nodes = 5 * 2 if mkw.get("integration") == "contact" else 1
    monkeypatch.setattr(rings_tau, "_RING_BLOCK", 50 * nodes)
    shapes = _spy_graph(monkeypatch)
    n = m._t_mx.size
    one = [(1, s) for s in _blocks(n, 50)]
    assert np.array_equal(m.light_curve(p), whole)
    assert shapes == one
    assert np.array_equal(m.light_curves(sets), whole_sets)
    assert shapes == 3 * one                # each set's times in blocks


def test_many_short_rows_are_blocked_by_sets(monkeypatch):
    """Sets that fit a block whole go in groups of rows (padded with the
    last); flux_dev_from_tau and the basis columns are blocked as well."""
    tau = np.linspace(-0.15, 0.15, 9)
    rp = 0.06 + 0.004 * np.arange(7)
    kw = dict(limb_dark="hybrid5", theta=0.4, ring_in=1.4, ring_out=2.6, ring_f=0.7)
    with mx.stream(mx.cpu):
        t2 = mx.broadcast_to(mx.array(tau, dtype=mx.float64), (7, 9))
        rpa = mx.array(rp, dtype=mx.float64)
        whole = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, u=W5, **kw))
        basis = np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, ld_basis=True, **kw))
        monkeypatch.setattr(rings_tau, "_RING_BLOCK", 20)
        shapes = _spy_graph(monkeypatch)
        assert np.array_equal(np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa, u=W5,
                                                            **kw)), whole)
        assert shapes == [(2, 9)] * 4       # 7 rows of 9 points, 2 per block
        assert np.array_equal(np.array(el.flux_dev_from_tau(t2, PER, 8.8, 0.3, rpa,
                                                            ld_basis=True, **kw)), basis)
    assert whole.shape == (7, 9) and basis.shape == (7, 9, 6)


def test_contact_kinks_are_found_once_per_set(monkeypatch):
    """The contact rule's kinks depend on the parameters alone."""
    calls, orig = [], rings_tau._ring_kinks

    def spy(*a, **kw):
        calls.append(a[0].shape)
        return orig(*a, **kw)
    p = P()
    m = el.TransitModel(p, T, integration="contact", exp_time=EXP, n_gl=2)
    monkeypatch.setattr(rings_tau, "_RING_BLOCK", 100)
    monkeypatch.setattr(rings_tau, "_ring_kinks", spy)
    m.light_curves([p, P(rp=0.07)])
    assert calls == [(2,)]


@pytest.mark.skipif(not el.metal_available(), reason="needs Metal")
def test_ring_kernels_are_not_blocked(monkeypatch):
    """The fp32 ring kernels hold no per-point graph: no graph at all."""
    monkeypatch.setattr(rings_tau, "_RING_BLOCK", 50)
    p = P()
    m = el.TransitModel(p, T, dtype=mx.float32)
    shapes = _spy_graph(monkeypatch)
    m.light_curve(p)
    m.light_curves([p, P(rp=0.07)])
    assert shapes == []


def test_errors():
    with pytest.raises(ValueError, match="hybrid law"):
        el.TransitModel(P(limb_dark="quadratic", u=[0.4, 0.2]), T)
    with pytest.raises(ValueError, match="ring_in"):
        el.TransitModel(P(ring_in=None), T)
    with pytest.raises(ValueError, match="ring_out"):
        p = P(ring_out=None)
        el.TransitModel(p, T).light_curve(p)
    with pytest.raises(ValueError, match="primary"):
        el.TransitModel(P(fp=1e-4), T, transittype="secondary")
    with pytest.raises(ValueError, match="ring_f"):
        p = P(ring_f=1.0)
        el.TransitModel(p, T).light_curve(p)
    with pytest.raises(ValueError, match="equator"):
        p = P(ring_in=0.9)
        el.TransitModel(p, T).light_curve(p)
    for f in (-0.1, 1.0, 1.2):              # as the oblate path, not NaN
        with pytest.raises(ValueError, match="flattening"):
            p = P(f=f)
            el.TransitModel(p, T).light_curve(p)
        with pytest.raises(ValueError, match="flattening"):
            el.TransitModel(P(f=0.2), T).light_curves([P(f=0.2), P(f=f)])
    m = el.TransitModel(P(), T)
    bare = P()
    bare.ring_in = bare.ring_out = bare.ring_f = bare.theta = None
    with pytest.raises(ValueError, match="ring_in changed"):
        m.light_curve(bare)
    with pytest.raises(ValueError, match="pass ring_in"):
        el.flux_dev_from_tau(np.zeros(3), PER, 8.8, 0.2, 0.08, limb_dark="hybrid5", u=W5,
                             ring_out=2.0)


def test_theta_alone_is_allowed_with_a_ring():
    """theta orients the ring of a spherical planet (no f needed)."""
    p = P(f=None, theta=60.0)
    assert 1.0 - el.TransitModel(p, T).light_curve(p).min() > 1e-3


def test_oblate_and_spherical_models_never_touch_the_ring_code():
    """The ring model may cost more; the oblate and spherical models must
    not pay for it: building and running them (fp64 graph, fp32, contact
    rule, flux_dev_from_tau) imports none of the ring modules."""
    import subprocess
    import sys
    code = r'''
import sys, math, numpy as np, mlx.core as mx
import elome as el
t = 0.3 + np.linspace(-0.15, 0.15, 31)
for f in (None, 0.3):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.3, 3.45, 0.1, 8.8, 88.0
    p.limb_dark, p.u = "hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1]
    p.f, p.theta = f, (None if f is None else 30.0)
    for dt in (mx.float64, mx.float32):
        for kw in ({}, {"integration": "contact", "exp_time": 0.02}):
            el.TransitModel(p, t, dtype=dt, **kw).light_curve(p)
el.flux_dev_from_tau(np.linspace(-0.1, 0.1, 9), 3.45, 8.8, 0.3, 0.1,
                     limb_dark="hybrid4", u=[0.2, 0.2, 0.1, 0.1], f=0.2, theta=0.4)
bad = [m for m in ("elome.rings", "elome.rings_tau", "elome.metal_rings") if m in sys.modules]
print("IMPORTED:" + ",".join(bad))
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "IMPORTED:\n" in out.stdout or out.stdout.strip().endswith("IMPORTED:"), out.stdout
