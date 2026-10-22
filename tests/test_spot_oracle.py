"""The spot oracle (tests/spot_oracle.py) checked before anything is checked
against it, and the geometry a spot model builds on, checked against the
oblate primitives.

The oracle: its two methods against each other, against the spherical
closed forms for a planet alone, against the projected area of a cap at
disc centre, and its weight bookkeeping (c = 1, a spot behind the star).

The geometry: a spot's rim on the sky is an ellipse in the oblate code's
form, parametrised by the rim angle psi, with x0 = 0, y0 = cos(alpha)
|n_xy|, a = sin(alpha), b = sin(alpha) |n_z| in the frame (x along
(n_y, -n_x), y along n_xy); along it mu = mu_a + mu_b sin(psi), linear,
because the rim lies on the sphere. So s = 1 - mu^2 on the rim, and the
pole quartic D(z) = z^2 (s - p) = -z^2 (mu^2 + eps) factorises as
(mu_b^2 / 4) times two quadratics z^2 + 2 i kappa z - 1,
kappa = (mu_a +- i sqrt(eps)) / mu_b,
each with root product -1: one root inside the unit circle, one outside.
With those roots the oblate arc integrals give a visible spot's occulted
flux exactly, and the oblate crossing solver, in the frame centred on the
planet and scaled by its radius, gives the planet-rim corners."""

import math

import mlx.core as mx
import numpy as np
import pytest

import spot_oracle as SO
from elome.hybrid import HYBRID2, HYBRID4, HYBRID5, even_cols, pole_col
from elome.oblate import (_C, _N_MAX, _coeffs, _even_arc, _limb_candidates, _pole_arc,
                          _q_coeffs, _roots)

EPS = tuple(HYBRID5.eps) + tuple(HYBRID4.eps) + tuple(HYBRID2.eps)
# each column's full-disc flux: an absolute error over it is a flux error
# per unit weight
NORM = np.array([math.pi, math.pi / 2, math.pi / 3] + [math.pi / (e * (1 + e)) for e in EPS])
D = math.radians
F64 = mx.float64


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def spot(lon, lat, alpha, c):
    return (SO.spot_vector(D(lon), D(lat)), D(alpha), c)


CONFIGS = {
    "centre spot alone": (None, [spot(0, 0, 10, 0.3)]),
    "planet across a rim": ((0.52, 0.30, 0.1), [spot(20, 15, 12, 0.4)]),
    "spot straddling the limb, planet on it": ((0.93, 0.15, 0.08), [spot(80, 10, 15, 0.5)]),
    "planet straddling the limb and a rim": ((0.97, 0.05, 0.1), [spot(70, 3, 12, 0.3)]),
    "planet inside the spot": ((0.0, 0.0, 0.03), [spot(0, 0, 20, 0.6)]),
    "spot inside the planet": ((0.1, 0.1, 0.2), [spot(6, 6, 3, 0.2)]),
    "umbra in penumbra": ((0.3, -0.1, 0.1), [spot(15, -5, 8, 0.7), spot(15, -5, 4, 0.5)]),
    "overlapping spot and facula": ((-0.2, 0.3, 0.12), [spot(-10, 18, 9, 0.4),
                                                       spot(-18, 14, 7, 1.3)]),
    "planet off the star, spot behind the limb": ((1.05, 0.0, 0.1), [spot(95, 0, 10, 0.4)]),
    "large spot covering the disc centre": ((-0.4, -0.3, 0.15), [spot(-20, -10, 40, 0.8)]),
    "thin limb spot, planet across it": ((0.0, 0.985, 0.06), [spot(0, 84, 8, 0.4)]),
    "three spots": ((0.5, 0.4, 0.09), [spot(25, 25, 6, 0.5), spot(40, 20, 9, 0.6),
                                       spot(-30, -30, 15, 0.2)]),
}


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CONFIGS)
def test_two_methods_agree(name):
    planet, spots = CONFIGS[name]
    a = SO.columns(planet, spots, EPS, "star")
    b = SO.columns(planet, spots, EPS, "split")
    assert rel(b, a) < 1e-12


@pytest.mark.parametrize("name", [k for k, v in CONFIGS.items() if v[0] is not None])
def test_unspotted_is_the_spherical_closed_form(name):
    """No spots, or every spot at contrast 1: the planet alone."""
    planet, spots = CONFIGS[name]
    xp, yp, rp = planet
    with mx.stream(mx.cpu):
        z = mx.array([math.hypot(xp, yp)], dtype=F64)
        r = mx.array(rp, dtype=F64)
        ref = np.array([c.item() for c in even_cols(z, r)]
                       + [pole_col(z, r, e).item() for e in EPS])
    clear = [(n, a, 1.0) for n, a, _ in spots]
    for method in ("star", "split"):
        assert rel(SO.columns(planet, [], EPS, method), ref) < 1e-12
        assert rel(SO.columns(planet, clear, EPS, method), ref) < 1e-12


@pytest.mark.parametrize("alpha", [0.5, 5.0, 30.0, 80.0])
def test_cap_at_disc_centre(alpha):
    """A cap facing the observer projects to a disc of radius sin(alpha):
    the mu^0 column is -pi sin^2(alpha), and mu^2 n is the disc's closed
    form, -pi (1 - (1 - R^2)^(n+1)) / (n + 1)."""
    R2 = math.sin(D(alpha)) ** 2
    got = SO.cap_cols(SO.spot_vector(0, 0), D(alpha), EPS)
    for k in range(3):
        assert got[k] == pytest.approx(-math.pi * (1 - (1 - R2) ** (k + 1)) / (k + 1),
                                       rel=1e-13)
    for j, e in enumerate(EPS):     # (1 + eps - rho^2)^-2 over the disc rho < R
        ref = -math.pi * (1.0 / (1.0 + e - R2) - 1.0 / (1.0 + e))
        assert got[3 + j] == pytest.approx(ref, rel=1e-12)


def test_spot_behind_the_star_is_zero():
    sp = [spot(180, 20, 25, 0.3)]
    for method in ("star", "split"):
        assert np.all(SO.columns(None, sp, EPS, method) == 0.0)


def test_mirror_symmetry():
    """Reflecting the sky in y = 0 (and the spot with it) changes nothing."""
    planet, spots = CONFIGS["planet across a rim"]
    mir = [(n * np.array([1, -1, 1]), a, c) for n, a, c in spots]
    a = SO.columns(planet, spots, EPS)
    b = SO.columns((planet[0], -planet[1], planet[2]), mir, EPS)
    assert rel(b, a) < 1e-13


# ---------------------------------------------------------------------------
# the geometry, against the oblate primitives
# ---------------------------------------------------------------------------

def rim_frame(n, alpha):
    """The rim as an oblate-code ellipse: (x0, y0, a, b), mu = mu_a + mu_b
    sin(psi), and the frame's x axis on the sky."""
    nxy = math.hypot(n[0], n[1])
    sg = 1.0 if n[2] >= 0 else -1.0
    ex = (n[1] / nxy, -n[0] / nxy)
    return ((0.0, math.cos(alpha) * nxy, math.sin(alpha), math.sin(alpha) * abs(n[2])),
            math.cos(alpha) * n[2], -sg * math.sin(alpha) * nxy, ex)


def closed_roots(mu_a, mu_b, eps):
    """The pole quartic's four roots: per sign, z^2 + 2 i kappa z - 1 = 0,
    the larger root without cancellation and its partner as -1/z."""
    out = []
    for sg in (1, -1):
        kap = complex(mu_a, sg * math.sqrt(eps)) / mu_b
        w = np.sqrt(complex(1.0 - kap * kap))
        big = max((-1j * kap + w, -1j * kap - w), key=abs)
        out += [big, -1.0 / big]
    return out


def limb_points(n, alpha, mu_a, mu_b):
    """(lo, hi) of the rim where mu > 0, and the limb arc dth inside the
    cap. Both from the product form q = sin^2(alpha) - n_z^2
    = (sin(alpha) - |n_z|)(sin(alpha) + |n_z|) = |n_xy|^2 - cos^2(alpha):
    mu = 0 at sin(psi) = -mu_a / mu_b, |cos(psi)| = sqrt(q) / (sin(alpha) |n_xy|),
    and dth = 2 atan2(sqrt(q), cos(alpha)). Asin and acos near 1 would
    lose half the digits of a small spot on the limb."""
    sa, nz = math.sin(alpha), n[2]
    q = (sa - abs(nz)) * (sa + abs(nz))
    if q <= 0.0:                                # no limb points
        return ((0.0, 2 * math.pi) if mu_a > 0 else (0.0, 0.0)), 0.0
    r = math.sqrt(q)
    p1 = math.atan2(-mu_a / mu_b, r / (sa * math.hypot(n[0], n[1])))
    p2 = math.pi - p1
    arc = (p1, p2) if mu_a + mu_b * math.sin(0.5 * (p1 + p2)) > 0 else (p2, p1 + 2 * math.pi)
    return arc, 2.0 * math.atan2(r, math.cos(alpha))


def random_spots(seed, k):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        v = rng.normal(size=3)
        n = v / np.linalg.norm(v)
        if n[2] < -0.5 or abs(n[2]) < 1e-3 or math.hypot(n[0], n[1]) < 1e-3:
            continue
        out.append((n, rng.uniform(0.01, 0.6)))
    return out


def A1(v):
    return mx.array([float(v)], dtype=F64)


@pytest.mark.parametrize("seed", range(4))
def test_rim_is_an_oblate_ellipse_with_linear_mu(seed):
    for n, al in random_spots(seed, 50):
        (x0, y0, a, b), mu_a, mu_b, ex = rim_frame(n, al)
        e1 = np.array([ex[0], ex[1], 0.0])
        e2 = np.cross(n, e1) * (1.0 if n[2] >= 0 else -1.0)
        psi = np.linspace(0, 2 * math.pi, 37)
        P = math.cos(al) * n + math.sin(al) * (np.cos(psi)[:, None] * e1
                                               + np.sin(psi)[:, None] * e2)
        ey = (n[0] / math.hypot(n[0], n[1]), n[1] / math.hypot(n[0], n[1]))
        x, y = P[:, 0] * ex[0] + P[:, 1] * ex[1], P[:, 0] * ey[0] + P[:, 1] * ey[1]
        assert np.max(np.abs(x - (x0 + a * np.cos(psi)))) < 1e-15
        assert np.max(np.abs(y - (y0 + b * np.sin(psi)))) < 1e-15
        mu = mu_a + mu_b * np.sin(psi)
        assert np.max(np.abs(P[:, 2] - mu)) < 1e-15
        assert np.max(np.abs(x * x + y * y - (1 - mu * mu))) < 1e-15


@pytest.mark.parametrize("seed", range(2))
def test_pole_quartic_factorises(seed):
    """The closed-form roots are the oblate solver's, one inside the unit
    circle per factor."""
    with mx.stream(mx.cpu):
        for n, al in random_spots(10 + seed, 40):
            (x0, y0, a, b), mu_a, mu_b, _ = rim_frame(n, al)
            S0, S1, S2, _, _ = _coeffs(A1(x0), A1(y0), A1(a), A1(b))
            for e in EPS:
                zc = closed_roots(mu_a, mu_b, e)
                inner = [abs(z) < 1 for z in zc]
                assert inner[0] != inner[1] and inner[2] != inner[3]
                zs = [complex(z.re.item(), z.im.item()) for z in _roots(S0, S1, S2, 1.0 + e)]
                for z in zc:
                    assert min(abs(z - w) for w in zs) < 1e-8 * max(1.0, abs(z))


def green_cap(n, al):
    """A visible cap's deviation columns by Green's theorem: the rim's
    visible arc through the oblate arc integrals with the closed-form
    roots (counter-clockwise for a cap facing the observer, clockwise for
    one facing away), plus h(1) times the limb arc inside the cap."""
    (x0, y0, a, b), mu_a, mu_b, _ = rim_frame(n, al)
    (lo, hi), dth = limb_points(n, al, mu_a, mu_b)
    S0, S1, S2, K0, K1 = _coeffs(A1(x0), A1(y0), A1(a), A1(b))
    cols = [v.item() for v in _even_arc(_q_coeffs(S0, S1, S2, K0, K1, _N_MAX), A1(lo), A1(hi))]
    h1 = [1.0 / (2.0 * (k + 1)) for k in range(3)]
    for e in EPS:
        z0 = [_C(A1(z.real), A1(z.imag)) for z in closed_roots(mu_a, mu_b, e)]
        cols.append(_pole_arc(S0, S1, S2, K0, K1, 1.0 + e, z0, A1(lo), A1(hi)).item())
        h1.append(1.0 / (2.0 * (1.0 + e) * e))
    # a cap facing away (n_z < 0) shows the sliver between its rim's front
    # arc and the limb, outside the ellipse: the arc is walked clockwise
    sg = 1.0 if n[2] >= 0 else -1.0
    return -(sg * np.array(cols) + dth * np.array(h1))


@pytest.mark.parametrize("cfg", [(0, 0.5, 10), (30, 20, 12), (60, -40, 5), (75, 10, 20),
                                 (85, 30, 8), (100, -5, 15), (-120, 70, 25), (45, 45, 0.5),
                                 (89, 0, 1), (10, 88, 3)])
def test_green_cap_is_the_oracle(cfg):
    lon, lat, al = cfg
    n, a = SO.spot_vector(D(lon), D(lat)), D(al)
    with mx.stream(mx.cpu):
        got = green_cap(n, a)
    ref = SO.cap_cols(n, a, EPS)
    assert np.max(np.abs(got - ref)) < 1e-12 * max(np.max(np.abs(ref)), 1e-6)


@pytest.mark.parametrize("dt,tol", [(F64, 1e-14), (mx.float32, 2e-6)])
@pytest.mark.parametrize("alpha", [20.0, 3.0, 1.0, 0.1, 0.03])
def test_green_cap_across_the_limb(alpha, dt, tol):
    """A spot rotating over the limb, centre from 0.95 alpha in front to
    0.95 alpha behind: the rim and limb arcs cancel to a sliver of order
    alpha^3, and the error stays at rounding in both precisions."""
    global A1
    keep = A1
    A1 = lambda v: mx.array([float(v)], dtype=dt)
    try:
        for frac in np.linspace(-0.95, 0.95, 11):
            n, a = SO.spot_vector(D(90 + frac * alpha), D(5)), D(alpha)
            with mx.stream(mx.cpu):
                got = green_cap(n, a)
            ref = SO.columns(None, [(n, a, 0.0)], EPS, "star")
            assert np.max(np.abs(got - ref) / NORM) < tol
    finally:
        A1 = keep


@pytest.mark.parametrize("cfg,count", [
    (((0.52, 0.30, 0.1), (20, 15, 12)), 2),      # across a rim
    (((0.0, 0.985, 0.06), (0, 84, 8)), 4),       # thin limb spot
    (((0.1, 0.1, 0.2), (6, 6, 3)), 0),           # spot inside the planet
    (((0.0, 0.0, 0.03), (0, 2, 20)), 0),         # planet inside the spot
    (((0.3, -0.1, 0.12), (15, -5, 8)), 2),       # rim and planet of similar size
])
def test_planet_rim_corners_by_the_oblate_solver(cfg, count):
    """In the frame centred on the planet and scaled by its radius the
    planet is the unit circle and the rim an oblate-code ellipse, so the
    crossing solver's candidates are the planet-rim corners."""
    (xp, yp, rp), (lon, lat, al) = cfg
    n, a_ = SO.spot_vector(D(lon), D(lat)), D(al)
    (x0, y0, a, b), _, _, ex = rim_frame(n, a_)
    ey = (n[0] / math.hypot(n[0], n[1]), n[1] / math.hypot(n[0], n[1]))
    px, py = xp * ex[0] + yp * ex[1], xp * ey[0] + yp * ey[1]
    with mx.stream(mx.cpu):
        phis, valid = _limb_candidates(A1((x0 - px) / rp), A1((y0 - py) / rp),
                                       A1(a / rp), A1(b / rp), F64)
        got = [p.item() for p, v in zip(phis, valid) if v.item()]
    pts = [(x0 + a * math.cos(p), y0 + b * math.sin(p)) for p in got]
    sky = sorted((u * ex[0] + v * ey[0], u * ex[1] + v * ey[1]) for u, v in pts)
    ref = sorted((P[0], P[1]) for P in SO.planet_rim_points(n, a_, (xp, yp, rp), front=False))
    assert len(sky) == len(ref) == count
    for g, r in zip(sky, ref):
        assert math.hypot(g[0] - r[0], g[1] - r[1]) < 1e-12
