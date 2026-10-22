"""Starspots crossed by a planet (elome/spots.py): the fp64 reference graph
and its fp32 twin against the spot oracle (tests/spot_oracle.py), the
identities a spotted star must satisfy, and gradients in every
parameter."""

import math

import mlx.core as mx
import numpy as np
import pytest

import spot_oracle as SO
from elome import spots
from elome.hybrid import (HYBRID2, HYBRID4, HYBRID5, HybridLaw, even_cols, hybrid_norms,
                          pole_col)

EPS = tuple(HYBRID5.eps) + tuple(HYBRID4.eps) + tuple(HYBRID2.eps)
ALL = HybridLaw("all_poles", EPS, ((1.0, -1.0, 0.0),))     # every pole in one call
NORM = np.array([math.pi, math.pi / 2, math.pi / 3] + [math.pi / (e * (1 + e)) for e in EPS])
TOL = {mx.float64: 1e-14, mx.float32: 1e-6}               # error / column norm
D = math.radians
FAR = (5.0, 5.0, 0.1)                                      # a planet off the star

# planet (X, Y, rp); spots [(lon, lat, alpha, c)] in degrees
CASES = {
    "centre spot alone": (FAR, [(0, 0, 10, 0.3)]),
    "planet across a rim": ((0.52, 0.30, 0.1), [(20, 15, 12, 0.4)]),
    "spot straddling the limb, planet on it": ((0.93, 0.15, 0.08), [(80, 10, 15, 0.5)]),
    "planet straddling the limb and a rim": ((0.97, 0.05, 0.1), [(70, 3, 12, 0.3)]),
    "planet inside the spot": ((0.0, 0.0, 0.03), [(0, 0, 20, 0.6)]),
    "spot inside the planet": ((0.1, 0.1, 0.2), [(6, 6, 3, 0.2)]),
    "umbra in penumbra": ((0.3, -0.1, 0.1), [(15, -5, 8, 0.7), (15, -5, 4, 0.5)]),
    "overlapping spot and facula": ((-0.2, 0.3, 0.12), [(-10, 18, 9, 0.4), (-18, 14, 7, 1.3)]),
    "planet off the star, spot behind the limb": ((1.05, 0.0, 0.1), [(95, 0, 10, 0.4)]),
    "large spot covering the disc centre": ((-0.4, -0.3, 0.15), [(-20, -10, 40, 0.8)]),
    "thin limb spot, planet across it": ((0.0, 0.985, 0.06), [(0, 84, 8, 0.4)]),
    "three spots": ((0.5, 0.4, 0.09), [(25, 25, 6, 0.5), (40, 20, 9, 0.6), (-30, -30, 15, 0.2)]),
    "planet across two overlapping spots": ((0.3, 0.3, 0.1), [(18, 15, 8, 0.4), (24, 19, 7, 0.6)]),
}


def oracle(case):
    planet, sp = case
    pl = None if planet == FAR else planet
    return SO.columns(pl, [(SO.spot_vector(D(lo), D(la)), D(al), c) for lo, la, al, c in sp], EPS)


def cols(case, dt=mx.float64, law=ALL):
    (X, Y, rp), sp = case
    n = np.array([SO.spot_vector(D(lo), D(la)) for lo, la, _, _ in sp])
    al = np.array([D(s[2]) for s in sp])
    c = np.array([s[3] for s in sp])
    with mx.stream(mx.cpu):
        g = spots.generator_cols_spots(mx.array(X, dtype=dt), mx.array(Y, dtype=dt),
                                       mx.array(rp, dtype=dt), mx.array(n, dtype=dt),
                                       mx.array(al, dtype=dt), mx.array(c, dtype=dt), law)
        return np.array([v.item() for v in g])


def err(v, ref):
    return np.max(np.abs(v - ref) / NORM)


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("name", list(CASES))
def test_against_the_oracle(name, dt):
    assert err(cols(CASES[name], dt), oracle(CASES[name])) < TOL[dt]


def _random_case(rng):
    """A spot anywhere on the visible hemisphere or just behind the limb,
    a planet near it (often across its rim), a second spot half the time."""
    lon, lat = rng.uniform(-100, 100), rng.uniform(-80, 80)
    al = rng.uniform(1.0, 25.0)
    sp = [(lon, lat, al, rng.uniform(0.0, 1.4))]
    n = SO.spot_vector(D(lon), D(lat))
    rp = rng.uniform(0.02, 0.2)
    off = rng.uniform(0, 2 * math.pi)
    d = math.cos(D(al)) * math.hypot(n[0], n[1])
    cx, cy = d * n[0] / max(math.hypot(n[0], n[1]), 1e-12), d * n[1] / max(math.hypot(n[0], n[1]), 1e-12)
    rr = math.sin(D(al)) * rng.uniform(0.0, 1.3)
    planet = (cx + rr * math.cos(off), cy + rr * math.sin(off), rp)
    if rng.uniform() < 0.5:
        sp.append((lon + rng.uniform(-15, 15), lat + rng.uniform(-15, 15), rng.uniform(1.0, 15.0),
                   rng.uniform(0.0, 1.4)))
    return planet, sp


@pytest.mark.parametrize("seed", range(6))
def test_random_configurations(seed):
    rng = np.random.default_rng(seed)
    for _ in range(8):
        case = _random_case(rng)
        assert err(cols(case), oracle(case)) < 1e-13, case


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("alpha", [15.0, 3.0, 0.3])
def test_spot_rotating_over_the_limb(alpha, dt):
    """Centre from 0.95 alpha in front of the limb to 0.95 alpha behind,
    with a planet across its visible part: the rim and limb arcs cancel to
    a sliver of order alpha^3 (product-form limb points)."""
    for frac in np.linspace(-0.95, 0.95, 9):
        lon = 90 + frac * alpha
        n = SO.spot_vector(D(lon), D(5))
        planet = (0.999 * n[0] / math.hypot(n[0], n[1]), 0.999 * n[1] / math.hypot(n[0], n[1]),
                  0.6 * math.sin(D(alpha)) + 0.01)
        case = (planet, [(lon, 5, alpha, 0.3)])
        assert err(cols(case, dt), oracle(case)) < TOL[dt], frac


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("d", [0.0, 1e-30, 1e-16, 1e-10, 1e-6, 1e-3])
def test_spot_at_disc_centre(d, dt):
    """|n_xy| -> 0: the rim is centred, mu_b -> 0 and the closed-form roots
    run to 0 and infinity; the centred expansion takes over."""
    case = ((0.05, 0.02, 0.08), [(math.degrees(d), 0.5 * math.degrees(d), 12, 0.4)])
    v = cols(case, dt)
    assert np.isfinite(v).all()
    assert err(v, oracle(case)) < TOL[dt]


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("dz", [-1e-3, -1e-9, 0.0, 1e-9, 1e-3])
def test_spot_centred_on_the_limb(dz, dt):
    """n_z -> 0 from either side: the rim's ellipse flattens to a segment
    and its orientation flips."""
    case = ((0.93, 0.3, 0.08), [(90.0 - math.degrees(dz), 18, 10, 0.4)])
    assert err(cols(case, dt), oracle(case)) < TOL[dt]


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("delta", [-1e-6, -1e-12, 0.0, 1e-12, 1e-6])
@pytest.mark.parametrize("kind", ["outside", "inside"])
def test_planet_tangent_to_a_rim(kind, delta, dt):
    """The planet touching a rim from outside or inside, at a point on the
    rim's long axis: the corners merge and are dropped together."""
    lon, lat, al = 25.0, 10.0, 12.0
    n = SO.spot_vector(D(lon), D(lat))
    nxy = math.hypot(n[0], n[1])
    ux, uy = n[1] / nxy, -n[0] / nxy                    # the rim's long axis
    cx, cy = math.cos(D(al)) * n[0], math.cos(D(al)) * n[1]
    a, rp = math.sin(D(al)), 0.06
    r = a + rp * (1 + delta) if kind == "outside" else a - rp * (1 + delta)
    case = ((cx + r * ux, cy + r * uy, rp), [(lon, lat, al, 0.4)])
    assert err(cols(case, dt), oracle(case)) < TOL[dt]


@pytest.mark.parametrize("name", ["planet across a rim", "umbra in penumbra", "three spots",
                                  "planet straddling the limb and a rim"])
def test_unspotted_is_the_spherical_closed_form(name):
    """Every spot at contrast 1: the planet alone, the spherical solution."""
    (X, Y, rp), sp = CASES[name]
    v = cols(((X, Y, rp), [s[:3] + (1.0,) for s in sp]))
    with mx.stream(mx.cpu):
        z, r = mx.array([math.hypot(X, Y)], dtype=mx.float64), mx.array(rp, dtype=mx.float64)
        ref = np.array([c.item() for c in even_cols(z, r)] + [pole_col(z, r, e).item() for e in EPS])
    assert err(v, ref) < 1e-14


def test_contrast_is_linear():
    """Each spot's columns are affine in its contrast."""
    planet, sp = CASES["overlapping spot and facula"]
    base = cols((planet, [sp[0][:3] + (1.0,), sp[1]]))
    dark = cols((planet, [sp[0][:3] + (0.0,), sp[1]]))
    for c in (0.2, 0.7, 1.5):
        got = cols((planet, [sp[0][:3] + (c,), sp[1]]))
        assert err(got, base + (1.0 - c) * (dark - base)) < 1e-15


def test_planet_off_the_star_is_the_modulation():
    """Out of transit the columns are the spots' own: sum (1 - c) cap."""
    sp = CASES["three spots"][1]
    ref = sum((1 - c) * SO.cap_cols(SO.spot_vector(D(lo), D(la)), D(al), EPS)
              for lo, la, al, c in sp)
    assert err(cols((FAR, sp)), ref) < 1e-14


def test_spots_on_the_far_side_are_exactly_zero():
    assert np.all(cols((FAR, [(180, 10, 20, 0.3), (150, -30, 10, 0.5)])) == 0.0)


def test_host_arguments_and_shared_spots():
    """Python numbers beside arrays, and spot fields shared by a batch of
    planet positions, equal the points one by one."""
    sp = [(20, 15, 12, 0.4), (30, 5, 6, 0.7)]
    n = mx.array(np.array([SO.spot_vector(D(a), D(b)) for a, b, _, _ in sp]), dtype=mx.float64)
    al = [D(s[2]) for s in sp]
    c = [s[3] for s in sp]
    xs, ys = np.array([0.35, 0.45, 2.0, 0.0]), np.array([0.25, 0.2, 0.0, 0.0])
    with mx.stream(mx.cpu):
        g = spots.generator_cols_spots(mx.array(xs, dtype=mx.float64), mx.array(ys, dtype=mx.float64),
                                       0.1, n, mx.array(al, dtype=mx.float64),
                                       mx.array(c, dtype=mx.float64), ALL)
        batch = np.stack([np.array(v) for v in g], -1)
    for i in range(len(xs)):
        assert err(batch[i], cols(((xs[i], ys[i], 0.1), sp))) < 1e-16


def test_per_point_spots():
    """Spot fields with the point shape (..., K): each point its own spots."""
    cases = [CASES["planet across a rim"], CASES["spot straddling the limb, planet on it"]]
    X = mx.array([c[0][0] for c in cases], dtype=mx.float64)
    Y = mx.array([c[0][1] for c in cases], dtype=mx.float64)
    rp = mx.array([c[0][2] for c in cases], dtype=mx.float64)
    n = mx.array(np.array([[SO.spot_vector(D(s[0][0]), D(s[0][1]))] for _, s in cases]), dtype=mx.float64)
    al = mx.array([[D(s[0][2])] for _, s in cases], dtype=mx.float64)
    c = mx.array([[s[0][3]] for _, s in cases], dtype=mx.float64)
    with mx.stream(mx.cpu):
        g = spots.generator_cols_spots(X, Y, rp, n, al, c, ALL)
        batch = np.stack([np.array(v) for v in g], -1)
    for i, case in enumerate(cases):
        assert err(batch[i], cols(case)) < 1e-16


def test_shape_columns_and_flux():
    """The ld_basis contract: flux - 1 == (B @ c) / (N @ c), c = (1, -w),
    with N the unspotted star's norms, for every law."""
    (X, Y, rp), sp = CASES["planet across two overlapping spots"]
    n = mx.array(np.array([SO.spot_vector(D(a), D(b)) for a, b, _, _ in sp]), dtype=mx.float64)
    al = mx.array([D(s[2]) for s in sp], dtype=mx.float64)
    cc = mx.array([s[3] for s in sp], dtype=mx.float64)
    G = oracle(CASES["planet across two overlapping spots"])
    for law, w in ((HYBRID2, [0.4, 0.2]), (HYBRID4, [0.2, 0.2, 0.1, 0.1]),
                   (HYBRID5, [0.2, 0.1, 0.1, 0.1, 0.05])):
        with mx.stream(mx.cpu):
            args = [mx.array(v, dtype=mx.float64) for v in (X, Y, rp)]
            B = np.array(spots.shape_cols_spots(*args, n, al, cc, law))
            dev = spots.flux_dev_spots(*args, n, al, cc, w, law).item()
        c = np.concatenate([[1.0], -np.asarray(w)])
        assert abs(dev - (B @ c) / (hybrid_norms(law) @ c)) < 1e-16
        idx = [0, 1, 2] + [3 + EPS.index(e) for e in law.eps]
        ref = np.concatenate([[G[0]], law.generator_matrix() @ G[idx]])
        assert np.max(np.abs(B - ref)) < 1e-14 * np.max(np.abs(ref))


def test_spot_vector():
    lon, lat = 0.3, -0.4
    v = spots.spot_vector(lon, lat)
    with mx.stream(mx.cpu):
        m = np.array(spots.spot_vector(mx.array(lon, dtype=mx.float64), mx.array(lat, dtype=mx.float64)))
    assert np.allclose(v, SO.spot_vector(lon, lat), atol=1e-16, rtol=0)
    assert np.allclose(m, v, atol=1e-16, rtol=0)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _flat(case):
    """Parameters as one flat list: X, Y, rp, then lon, lat, alpha, c per spot (radians)."""
    (X, Y, rp), sp = case
    out = [X, Y, rp]
    for lo, la, al, c in sp:
        out += [D(lo), D(la), D(al), c]
    return out


def _F(dt, corners=True):
    w = mx.array((1.0 / NORM), dtype=dt)

    def F(*p):
        K = (len(p) - 3) // 4
        lon = mx.stack([p[3 + 4 * k] for k in range(K)])
        lat = mx.stack([p[4 + 4 * k] for k in range(K)])
        al = mx.stack([p[5 + 4 * k] for k in range(K)])
        c = mx.stack([p[6 + 4 * k] for k in range(K)])
        n = spots.spot_vector(lon, lat)
        return mx.sum(mx.stack(spots.generator_cols_spots(p[0], p[1], p[2], n, al, c, ALL), -1) * w)
    return F


def _grads(case, corners=True):
    from elome import rings
    p = _flat(case)
    F = _F(mx.float64)
    saved = spots._h
    if not corners:
        spots._h = lambda arg, s, even: 0.0 * s
    try:
        with mx.stream(mx.cpu):
            args = [mx.array(v, dtype=mx.float64) for v in p]
            ad = np.array([g.item() for g in mx.grad(F, argnums=list(range(len(p))))(*args)])
            at = lambda i, d: F(*[mx.array(v + (d if j == i else 0.0), dtype=mx.float64)
                                  for j, v in enumerate(p)]).item()
            fd = np.zeros(len(p))
            for i in range(len(p)):
                h = 1e-6 * max(abs(p[i]), 0.1)
                fd[i] = (at(i, h) - at(i, -h)) / (2 * h)
    finally:
        spots._h = saved
    return ad, fd


GRAD_CASES = ["planet across a rim", "spot straddling the limb, planet on it",
              "planet straddling the limb and a rim", "overlapping spot and facula",
              "planet off the star, spot behind the limb", "three spots",
              "planet across two overlapping spots", "centre spot alone",
              "thin limb spot, planet across it"]


@pytest.mark.parametrize("name", GRAD_CASES)
def test_gradients_match_finite_differences(name):
    ad, fd = _grads(CASES[name])
    assert np.isfinite(ad).all()
    assert np.max(np.abs(ad - fd)) < 1e-7 * np.max(np.abs(fd))


@pytest.mark.parametrize("name", ["planet across a rim", "spot straddling the limb, planet on it"])
def test_corner_terms_are_required(name):
    """Without the gradient-only corner terms the frozen arc endpoints
    leave O(1) errors: a guard that the terms stay wired in."""
    ad, fd = _grads(CASES[name], corners=False)
    assert np.max(np.abs(ad - fd)) > 0.1 * np.max(np.abs(fd))


@pytest.mark.parametrize("dt", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_gradients_finite_on_degenerate_lanes(dt):
    """Centred spots, spots centred on the limb, tangent planets, spots
    behind the star and concentric spots: no NaN or inf in any gradient."""
    lanes = [((0.05, 0.02, 0.08), [(0.0, 0.0, 12, 0.4)]),
             ((0.93, 0.3, 0.08), [(90.0, 18, 10, 0.4)]),
             ((0.3, -0.1, 0.1), [(15, -5, 8, 0.7), (15, -5, 4, 0.5)]),
             (FAR, [(180, 10, 20, 0.3)]),
             ((0.0, 0.0, 0.03), [(0, 0, 20, 0.6)])] + list(CASES.values())
    F = _F(dt)
    for case in lanes:
        p = _flat(case)
        with mx.stream(mx.cpu):
            g = mx.grad(F, argnums=list(range(len(p))))(*[mx.array(v, dtype=dt) for v in p])
        assert all(np.isfinite(x.item()) for x in g), case


def test_n_must_be_vectors():
    with mx.stream(mx.cpu), pytest.raises(ValueError, match="unit vector"):
        spots.generator_cols_spots(mx.array(0.1, dtype=mx.float64), 0.0, 0.1,
                                   mx.array([0.0, 0.0, 1.0], dtype=mx.float64), 0.1, 0.5, HYBRID2)


def test_other_models_never_touch_the_spot_code():
    """Spherical, oblate and ringed models (fp64 graph, fp32, contact rule)
    import none of the spot modules: they must not pay for them."""
    import subprocess
    import sys
    code = r'''
import sys, numpy as np, mlx.core as mx
import elome as el
t = 0.3 + np.linspace(-0.15, 0.15, 31)
for f, ring in ((None, None), (0.3, None), (None, 1.5)):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.3, 3.45, 0.1, 8.8, 88.0
    p.limb_dark, p.u = "hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1]
    p.f, p.theta = f, (None if f is None and ring is None else 30.0)
    if ring is not None:
        p.ring_in, p.ring_out, p.ring_f = ring, 2.2, 0.6
    for dt in (mx.float64, mx.float32):
        for kw in ({}, {"integration": "contact", "exp_time": 0.02}):
            el.TransitModel(p, t, dtype=dt, **kw).light_curve(p)
bad = [m for m in sys.modules if m.startswith("elome.spot")]
print("IMPORTED:" + ",".join(bad))
'''
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("IMPORTED:"), out.stdout
