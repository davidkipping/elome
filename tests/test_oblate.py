"""Oblate planets: the reference MLX graph (elome/oblate.py).

The truth is an independent polar-ray quadrature (tests/oblate_oracle.py,
fp64 and 30-digit mpmath), itself checked against closed forms here. The
contract: <= 1e-12 of each column's unocculted flux in fp64 over the
topology classes and the limb- and pole-tangency stress sets; f -> 0
continuous with the spherical path; the ld_basis identity; gradients
against finite differences, finite everywhere.
"""

import functools
import math

import numpy as np
import mlx.core as mx
import pytest

import elome as el
from elome import oblate as O
from elome.hybrid import HYBRID2, HybridLaw, LAWS, hybrid_norms, shape_cols

import oblate_oracle as OR

EPS = tuple(sorted({e for name in ("hybrid2", "hybrid4", "hybrid5") for e in LAWS[name].eps}))
ALLPOLES = HybridLaw("allpoles", EPS, ((1.0, -1.0, 0.0),))
NORM = np.concatenate([[math.pi, math.pi / 2, math.pi / 3],
                       math.pi / (np.array(EPS) * (1.0 + np.array(EPS)))])
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
     "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}


def f64(v):
    return mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)


def cpu_np(fn):
    """Run fn on the CPU stream and return numpy float64."""
    with mx.stream(mx.cpu):
        out = fn()
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)


def generators(x0, y0, r, f, dtype=mx.float64):
    stream = mx.cpu if dtype == mx.float64 else mx.default_device()
    with mx.stream(stream):
        g = O.generator_cols_oblate(*(mx.array(np.asarray(v, np.float64), dtype=dtype)
                                      for v in (x0, y0, r, f)), ALLPOLES)
        mx.eval(*g)
        return np.stack([np.asarray(v.astype(mx.float32) if dtype == mx.float32 else v,
                                    dtype=np.float64) for v in g], -1)


def reference(x0, y0, A, B):
    return OR.columns(x0, y0, A, B, EPS)


@functools.lru_cache(maxsize=None)
def stress(name):
    """(x0, y0, A, B, reference columns) of one stress set, in the
    planetary domain."""
    x0, y0, A, B = OR.planets(np.array(OR.stress_sets(EPS)[name]).T)
    return x0, y0, A, B, reference(x0, y0, A, B)


# ---------------------------------------------------------------------------
# geometry and domain
# ---------------------------------------------------------------------------

def test_axes_and_domain():
    A, B = O.axes(0.1, 0.3)
    Av, Bv = O.axes(0.1, np.array([0.1, 0.3]))        # numpy f
    assert np.allclose(Av * Bv, 0.01, rtol=1e-15, atol=0)
    assert abs(A * B - 0.01) < 1e-17 and abs(B / A - 0.7) < 1e-15
    assert O.max_r_eff(0.5) == pytest.approx(0.5 ** 1.5)
    with pytest.raises(ValueError, match="r_eff"):
        el.flux_dev_oblate(f64([0.5]), f64([0.0]), 0.6, 0.5, W["hybrid4"], "hybrid4")
    for bad_f in (-0.1, 1.0):
        with pytest.raises(ValueError, match="flattening"):
            el.flux_dev_oblate(f64([0.5]), f64([0.0]), 0.1, bad_f, W["hybrid4"], "hybrid4")


def test_principal_frame_is_the_rotation_by_theta():
    """theta is the sky angle of the long axis from X: the principal frame
    rotates the sky by -theta."""
    rng = np.random.default_rng(0)
    X, Y, th = rng.normal(size=50), rng.normal(size=50), rng.uniform(0, np.pi, 50)
    x0, y0 = O.principal_frame(X, Y, th)
    assert np.allclose(x0, X * np.cos(th) + Y * np.sin(th), atol=1e-15, rtol=0)
    assert np.allclose(y0, -X * np.sin(th) + Y * np.cos(th), atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
# the reference itself
# ---------------------------------------------------------------------------

def test_the_reference_meets_closed_forms():
    """The quadrature against what is known exactly: a circle's lens area
    (f = 0), an ellipse wholly on the disc (area pi A B, and the mu^2
    moment pi A B (1 - x0^2 - y0^2 - (A^2 + B^2)/4)), a centred planet,
    and its own 30-digit version."""
    for d, r in ((0.0, 0.2), (0.5, 0.1), (0.95, 0.1), (1.0, 0.1), (1.05, 0.1), (0.3, 0.5)):
        got = OR.columns(d, 0.0, r, r, ())[0, 0]
        assert abs(got + OR.lens_area(d, r)) < 1e-14, (d, r)
    for x0, y0, A, B in ((0.3, -0.2, 0.2, 0.12), (0.0, 0.0, 0.3, 0.15), (-0.5, 0.6, 0.1, 0.05)):
        c = OR.columns(x0, y0, A, B, ())[0]
        area = math.pi * A * B
        assert abs(c[0] + area) < 1e-15
        assert abs(c[1] + area * (1 - x0 * x0 - y0 * y0 - (A * A + B * B) / 4)) < 1e-15
    for cfg in ((0.95, 0.1, 0.12, 0.09), (-0.4, 0.75, 0.2, 0.12), (0.0, 0.99, 0.15, 0.1)):
        pytest.importorskip("mpmath")
        eps = (EPS[0], EPS[-1])
        norm = np.concatenate([NORM[:3], [NORM[3], NORM[-1]]])
        err = np.abs(OR.columns_mp(*cfg, eps) - OR.columns(*cfg, eps)[0]) / norm
        assert err.max() < 1e-14, cfg


def test_the_reference_is_the_spherical_path_at_f_zero():
    """A = B: the quadrature equals the spherical closed forms, whose own
    truth is test_hybrid's 40-digit radial oracle."""
    z = np.array([0.0, 0.3, 0.85, 0.95, 1.02, 1.08])
    r = 0.1
    ref = OR.columns(z, 0.0, r, r, EPS)
    with mx.stream(mx.cpu):
        sph = np.asarray(shape_cols(f64(z), f64(r), ALLPOLES))
    # shape_cols is in the shape basis: [E0, 1 - mu^2, Pi_eps...]
    E0, E1 = ref[:, 0], ref[:, 1]
    assert np.abs(sph[:, 0] - E0).max() < 1e-14
    assert np.abs(sph[:, 1] - (E0 - E1)).max() < 1e-14
    for k, e in enumerate(EPS):                  # Pi_eps = (P_eps - p^-2 E0) / N_eps
        Pi = (ref[:, 3 + k] - E0 / (1 + e) ** 2) / (e ** -2 - (1 + e) ** -2)
        assert np.abs(sph[:, 2 + k] - Pi).max() < 1e-14 * max(1.0, 1 / e), e


# ---------------------------------------------------------------------------
# against the reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["two_int", "inside", "disjoint", "origin_in_partial",
                                  "limb_tangent", "pole_tangent"])
@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_generators_match_the_reference(name, dtype):
    """Every generator column, every topology class and tangency set
    (planetary domain), to 1e-12 (fp64) / 1e-6 (fp32) of its unocculted
    flux."""
    x0, y0, A, B, ref = stress(name)
    r, f = np.sqrt(A * B), 1.0 - B / A
    got = generators(x0, y0, r, f, dtype)
    err = np.abs(got - ref) / NORM
    assert np.isfinite(got).all()
    assert err.max() < (1e-12 if dtype == mx.float64 else 1e-6)


@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
def test_law_light_curve_matches_the_reference(law):
    """flux_dev_oblate along tilted transit chords, sky frame and theta
    convention included."""
    for r, f, th, b in ((0.1, 0.1, 0.4, 0.3), (0.2, 0.3, 1.0, 0.6), (0.05, 0.5, 2.5, 0.0)):
        A, B = r / math.sqrt(1 - f), r * math.sqrt(1 - f)
        X = np.linspace(-(1 + A) * 1.02, (1 + A) * 1.02, 301)
        Y = np.full_like(X, b)
        x0, y0 = X * math.cos(th) + Y * math.sin(th), -X * math.sin(th) + Y * math.cos(th)
        got = cpu_np(lambda: el.flux_dev_oblate(f64(x0), f64(y0), r, f, W[law], law))
        ref = OR.law_flux_dev(law, W[law], x0, y0, A, B)
        assert ref.min() < -1e-3                            # a transit is tested
        assert np.abs(got - ref).max() < 1e-12, (r, f, th, b)


def test_generators_match_mpmath_quadrature():
    """Spot checks at 30 digits: a partial and an inside configuration,
    every column."""
    pytest.importorskip("mpmath")
    cfg = [(0.95, 0.1, 0.12, 0.09), (0.3, 0.2, 0.1, 0.08)]
    x0, y0, A, B = np.array(cfg).T
    got = generators(x0, y0, np.sqrt(A * B), 1.0 - B / A)
    for i, c in enumerate(cfg):
        ref = OR.columns_mp(*c, EPS)
        assert (np.abs(got[i] - ref) / NORM).max() < 1e-12, c


@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9, -1e-9, -1e-6, -1e-3])
def test_inside_and_partial_meet_at_the_limb(gap):
    """Where |centre| + A crosses 1 the inside closed forms hand over to the
    partial-regime arcs: both sides agree with the reference."""
    rng = np.random.default_rng(int(abs(gap) * 1e12) % 1000)
    cfg = []
    for _ in range(20):
        r, f = rng.uniform(0.02, 0.3), rng.uniform(0.01, 0.5)
        f = min(f, 1 - r ** (2 / 3) - 1e-3)
        A, B = r / math.sqrt(1 - f), r * math.sqrt(1 - f)
        t = rng.uniform(0, 2 * np.pi)
        d = 1.0 - A - gap                        # |centre| + A = 1 - gap
        cfg.append((d * math.cos(t), d * math.sin(t), A, B))
    x0, y0, A, B = np.array(cfg).T
    got = generators(x0, y0, np.sqrt(A * B), 1.0 - B / A)
    assert (np.abs(got - reference(x0, y0, A, B)) / NORM).max() < 1e-12


# ---------------------------------------------------------------------------
# f -> 0, the ld_basis contract, zero outside
# ---------------------------------------------------------------------------

def test_small_f_is_the_spherical_path():
    """Below f_sw the spherical closed forms (exactly the spherical path's
    values); above it the oblate path, continuous across the switch."""
    X = np.linspace(0.0, 1.12, 300)
    x0, y0 = O.principal_frame(X, 0.3, 0.7)
    sph = cpu_np(lambda: shape_cols(f64(np.hypot(x0, y0)), 0.1, "hybrid5"))
    for f in (0.0, 1e-12, 0.99 * O._F_SW[mx.float64]):
        got = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, f, "hybrid5"))
        assert np.abs(got - sph).max() < 1e-15
    above = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1,
                                               1.01 * O._F_SW[mx.float64], "hybrid5"))
    assert np.abs(above - sph).max() < 1e-11          # the flattening's own ~4e-3 f


def test_ld_basis_identity():
    rng = np.random.default_rng(3)
    X = np.linspace(-1.2, 1.2, 200)
    x0, y0 = O.principal_frame(X, 0.2, 1.1)
    for law in ("hybrid2", "hybrid4", "hybrid5"):
        Bc = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.12, 0.25, law))
        w = rng.dirichlet(np.ones(LAWS[law].n_w + 1))[:-1]
        c = np.concatenate([[1.0], -w])
        got = cpu_np(lambda: el.flux_dev_oblate(f64(x0), f64(y0), 0.12, 0.25, list(w), law))
        assert np.abs(got - (Bc @ c) / (hybrid_norms(law) @ c)).max() < 1e-15


def test_exactly_zero_out_of_transit():
    X = np.linspace(1.2, 3.0, 50)
    x0, y0 = O.principal_frame(X, 0.1, 0.4)
    got = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, 0.3, "hybrid5"))
    assert np.array_equal(got, np.zeros_like(got))


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

POINTS = [(0.3, 0.2, 0.1, 0.2), (0.95, 0.1, 0.1, 0.3), (-0.6, 0.75, 0.15, 0.25),
          (0.02, 0.01, 0.12, 0.1)]


@pytest.mark.parametrize("pt", POINTS, ids=["inside", "partial-a", "partial-b", "near-centre"])
def test_gradients_match_finite_differences(pt):
    law, w = "hybrid5", W["hybrid5"]

    def fn(x, y, r, f, ww):
        return mx.sum(el.flux_dev_oblate(x, y, r, f, ww, law))

    with mx.stream(mx.cpu):
        g = mx.grad(fn, argnums=(0, 1, 2, 3, 4))(*map(f64, pt), f64(w))
        mx.eval(g)
        h = 1e-6
        for i in range(4):
            up, dn = list(pt), list(pt)
            up[i] += h
            dn[i] -= h
            fd = (float(fn(*map(f64, up), f64(w))) - float(fn(*map(f64, dn), f64(w)))) / (2 * h)
            assert abs(float(g[i]) - fd) <= 1e-6 * max(abs(fd), 1e-3), (i, float(g[i]), fd)
        for j in range(5):
            wu, wd = list(w), list(w)
            wu[j] += h
            wd[j] -= h
            fd = (float(fn(*map(f64, pt), f64(wu))) - float(fn(*map(f64, pt), f64(wd)))) / (2 * h)
            assert abs(float(np.asarray(g[4])[j]) - fd) <= 1e-6 * max(abs(fd), 1e-3), j


@pytest.mark.parametrize("dtype", [mx.float64,
                                   pytest.param(mx.float32, marks=pytest.mark.slow)],
                         ids=["fp64", "fp32"])
def test_gradients_finite_on_a_boundary_sweep(dtype):
    """Exact tangencies, a centred planet, f = 0 and either side of f_sw,
    the inside/partial seam, out of transit: every gradient finite."""
    cfgs = []
    for gap in (0.0, 1e-9, -1e-9, 1e-6):
        for kind in ("external", "internal"):
            x0, y0, a, b = OR.tangent_config(0.12, 0.08, 1.1, kind, gap)
            cfgs.append((x0, y0, math.sqrt(a * b), 1 - b / a))
    sw = O._F_SW[dtype]
    cfgs += [(0.0, 0.0, 0.1, 0.2), (0.3, 0.0, 0.1, 0.0), (0.3, 0.0, 0.1, 0.99 * sw),
             (0.3, 0.0, 0.1, 1.01 * sw), (1.2, 0.1, 0.1, 0.2),
             (1.0 - 0.1 / math.sqrt(0.8), 0.0, 0.1, 0.2)]
    stream = mx.cpu if dtype == mx.float64 else mx.default_device()
    with mx.stream(stream):
        for c in cfgs:
            def fn(x, y, r, f):
                return mx.sum(el.flux_dev_oblate(x, y, r, f, W["hybrid5"], "hybrid5"))
            g = mx.grad(fn, argnums=(0, 1, 2, 3))(*(mx.array(v, dtype=dtype) for v in c))
            mx.eval(g)
            for gi in g:
                assert np.isfinite(np.asarray(gi.astype(mx.float32))).all(), c


# ---------------------------------------------------------------------------
# calling forms
# ---------------------------------------------------------------------------

def test_weight_forms_agree():
    X = np.linspace(-1.1, 1.1, 64)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    w = W["hybrid4"]
    host = cpu_np(lambda: el.flux_dev_oblate(f64(x0), f64(y0), 0.1, 0.2, w, "hybrid4"))
    traced = cpu_np(lambda: el.flux_dev_oblate(f64(x0), f64(y0), 0.1, 0.2, f64(w), "hybrid4"))
    rows = cpu_np(lambda: el.flux_dev_oblate(f64(np.stack([x0, x0])), f64(np.stack([y0, y0])),
                                             0.1, 0.2, f64([w, w]), "hybrid4"))
    assert np.abs(host - traced).max() < 1e-16
    assert np.abs(rows - host[None]).max() < 1e-16


def test_data_contract_and_keyword_calls():
    """numpy float64 data is fp64 (on the CPU stream, put there for us);
    float32 stays float32; the data may be passed by name."""
    X = np.linspace(-1.1, 1.1, 32)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    a = el.flux_dev_oblate(x0, y0, 0.1, 0.2, W["hybrid2"], "hybrid2")
    assert a.dtype == mx.float64
    b = el.flux_dev_oblate(x0=x0, y0=y0, r=0.1, f=0.2, w=W["hybrid2"], law="hybrid2")
    with mx.stream(mx.cpu):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    c = el.flux_dev_oblate(x0.astype(np.float32), y0.astype(np.float32), 0.1, 0.2,
                           W["hybrid2"], "hybrid2")
    assert c.dtype == mx.float32


def test_a_custom_law_object():
    law = HybridLaw("custom2", (0.15,), HYBRID2.shapes)
    X = np.linspace(-1.1, 1.1, 32)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    B = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, 0.2, law))
    assert B.shape == (32, 3) and np.isfinite(B).all() and B.min() < 0


def test_fp64_graph_compiles():
    X = np.linspace(-1.1, 1.1, 64)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    with mx.stream(mx.cpu):
        xa, ya = f64(x0), f64(y0)
        eager = el.flux_dev_oblate(xa, ya, 0.1, 0.2, W["hybrid5"], "hybrid5")
        comp = mx.compile(lambda x, y: el.flux_dev_oblate(x, y, 0.1, 0.2, W["hybrid5"], "hybrid5"))
        got = comp(xa, ya)
        assert np.abs(np.asarray(got) - np.asarray(eager)).max() < 1e-15


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_theta_gradient_through_the_sky_frame(dtype):
    """theta enters through principal_frame: autodiff vs central FD (fp64),
    and finite at theta = 0 and pi/2 along a whole chord (both dtypes)."""
    X = np.linspace(-1.15, 1.15, 81)
    stream = mx.cpu if dtype == mx.float64 else mx.default_device()

    def fn(th, b):
        x0, y0 = O.principal_frame(mx.array(X, dtype=dtype), b, th)
        return mx.sum(el.flux_dev_oblate(x0, y0, 0.1, 0.25, W["hybrid4"], "hybrid4"))

    with mx.stream(stream):
        for th in (0.0, 0.4, math.pi / 2):
            g = mx.grad(fn)(mx.array(th, dtype=dtype), mx.array(0.3, dtype=dtype))
            mx.eval(g)
            assert np.isfinite(float(g.astype(mx.float32)))
            if dtype == mx.float64 and th == 0.4:
                h = 1e-6
                fd = (float(fn(mx.array(th + h, dtype=dtype), mx.array(0.3, dtype=dtype)))
                      - float(fn(mx.array(th - h, dtype=dtype), mx.array(0.3, dtype=dtype)))) / (2 * h)
                assert abs(float(g) - fd) <= 1e-6 * max(abs(fd), 1e-3)
