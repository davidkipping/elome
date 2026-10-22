"""An independent reference for planets crossing starspots, for the tests.

Shares no code with ``elome``. A spot is a circular cap on the stellar
surface: unit centre vector n (sky frame: x, y on the sky, z towards the
observer), angular radius alpha, intensity contrast c (the spot's
intensity is c times the photosphere's at the same mu). The planet is an
opaque disc (xp, yp, rp) on the sky. The blocked fraction is

    w = 1 inside the planet, else sum_k (1 - c_k) [visible spot k],

so overlapping spots add (an umbra inside a penumbra is two concentric
caps), and a spot's visible region is the set of disc points whose FRONT
surface point lies in the cap -- bounded by the cap's rim where mu > 0 and
by the limb, not by the rim's projected ellipse alone.

Columns follow the package's generator order: mu^0, mu^2, mu^4, then one
double pole (mu^2 + eps)^-2 per eps; deviation form (minus the integral of
w I over the disc). Two unrelated methods:

* ``method="star"`` -- polar rays about the star's centre. On a ray at
  angle phi, with s = sin t, a spot's condition P . n >= cos(alpha) is
  R cos(t - t0) >= cos(alpha): one interval in t, so one chord. The planet
  is a circle's chord. The ray is split at every chord end, each piece's
  weight read at its midpoint, and integrated with the closed-form radial
  primitives; the angle is split at every tangent ray, limb point and
  corner (planet-rim, rim-rim) and integrated adaptively.
* ``method="split"`` -- the weight decomposed,
      planet + sum_k (1 - c_k) (cap_k - planet n cap_k),
  with the planet by rays about its own centre (the ring oracle's
  ``centre`` method), each cap alone as a one-dimensional integral over mu
  (the cap's share of the circle of constant mu on the sphere has a
  closed-form azimuth length), and each lens planet n cap_k by rays about
  the planet's centre, the cap's chord on each from a quadratic.

Both methods agree to ~1e-14 for spots from a few tenths of a degree up.
Below that their arccos edges (a cap's chord, its share of a circle of
constant mu) lose digits as 1e-16 / alpha^2: about 1e-9 relative at 0.01
degrees. Test against them above ~0.1 degrees.
"""

import math

import numpy as np

from oblate_oracle import _limb_angles, _tangent_angles, even_prim, pole_prim
from ring_oracle import _adaptive, _radial_even, _radial_pole, _turn, region_cols

__all__ = ["columns", "cap_cols", "lens_cols", "planet_rim_points", "rim_rim_points",
           "spot_vector"]

_HALF_PI = 0.5 * math.pi


def _prims(eps):
    return [even_prim(n) for n in range(3)] + [pole_prim(e) for e in eps]


def spot_vector(lon, lat):
    """Unit vector of a spot at sky-frame longitude lon (from the line of
    sight, towards +x) and latitude lat (towards +y), radians."""
    return np.array([math.cos(lat) * math.sin(lon), math.sin(lat),
                     math.cos(lat) * math.cos(lon)])


def _basis(n):
    """(e1, e2) completing n to an orthonormal frame, e1 horizontal."""
    nxy = math.hypot(n[0], n[1])
    e1 = np.array([1.0, 0.0, 0.0]) if nxy == 0.0 else np.array([n[1], -n[0], 0.0]) / nxy
    return e1, np.cross(n, e1)


# ---------------------------------------------------------------------------
# corners
# ---------------------------------------------------------------------------

def _rim_fourier(n, alpha):
    """The rim P(psi) = cos(alpha) n + sin(alpha) (cos psi e1 + sin psi e2)."""
    e1, e2 = _basis(n)
    return math.cos(alpha) * n, math.sin(alpha) * e1, math.sin(alpha) * e2


def planet_rim_points(n, alpha, planet, front=True):
    """3D points where a spot's rim meets the planet's outline on the sky:
    (x - xp)^2 + (y - yp)^2 = rp^2 along the rim is a trigonometric
    quadratic in psi, a quartic in z = e^{i psi}. ``front``: only mu >= 0."""
    xp, yp, rp = planet
    P0, Pc, Ps = _rim_fourier(n, alpha)
    U0, Uc, Us = P0[0] - xp, Pc[0], Ps[0]
    V0, Vc, Vs = P0[1] - yp, Pc[1], Ps[1]
    a0 = U0 * U0 + V0 * V0 + 0.5 * (Uc * Uc + Us * Us + Vc * Vc + Vs * Vs) - rp * rp
    a1, b1 = 2 * (U0 * Uc + V0 * Vc), 2 * (U0 * Us + V0 * Vs)
    a2 = 0.5 * (Uc * Uc - Us * Us + Vc * Vc - Vs * Vs)
    b2 = Uc * Us + Vc * Vs

    def g(psi):
        return (a0 + a1 * math.cos(psi) + b1 * math.sin(psi)
                + a2 * math.cos(2 * psi) + b2 * math.sin(2 * psi))

    def dg(psi):
        return (-a1 * math.sin(psi) + b1 * math.cos(psi)
                - 2 * a2 * math.sin(2 * psi) + 2 * b2 * math.cos(2 * psi))

    coef = [(a2 - 1j * b2) / 2, (a1 - 1j * b1) / 2, a0, (a1 + 1j * b1) / 2, (a2 + 1j * b2) / 2]
    out = []
    for z in np.roots(coef):
        if abs(abs(z) - 1.0) > 1e-6:
            continue
        psi = math.atan2(z.imag, z.real)
        for _ in range(4):
            d = dg(psi)
            if d == 0.0:
                break
            step = g(psi) / d
            if abs(step) > 1e-6:
                break
            psi -= step
        P = P0 + Pc * math.cos(psi) + Ps * math.sin(psi)
        if not front or P[2] >= 0.0:
            out.append(P)
    return out


def rim_rim_points(n1, a1, n2, a2):
    """3D points where two rims meet on the sphere: P . n1 = cos a1,
    P . n2 = cos a2, |P| = 1; front side only."""
    d = float(n1 @ n2)
    if abs(d) >= 1.0 - 1e-15:
        return []
    c1, c2 = math.cos(a1), math.cos(a2)
    det = 1.0 - d * d
    a, b = (c1 - d * c2) / det, (c2 - d * c1) / det
    base = a * n1 + b * n2
    t2 = (1.0 - base @ base) / det
    if t2 < 0.0:
        return []
    m = np.cross(n1, n2)
    return [P for P in (base + math.sqrt(t2) * m, base - math.sqrt(t2) * m) if P[2] >= 0.0]


def _rim_limb_points(n, alpha):
    """Sky points where a rim meets the limb (mu = 0): the limb arc inside
    the cap is azimuth phi_n +- acos(cos(alpha) / |n_xy|)."""
    nxy = math.hypot(n[0], n[1])
    if nxy <= math.cos(alpha):
        return []
    g, h = math.atan2(n[1], n[0]), math.acos(math.cos(alpha) / nxy)
    return [np.array([math.cos(g + s * h), math.sin(g + s * h), 0.0]) for s in (1, -1)]


def _rim_ellipse(n, alpha):
    """The rim's projection: centre, semi-axes, and the long axis's angle
    on the sky (a degenerate b = 0 for a spot centred on the limb)."""
    e1, _ = _basis(n)
    return (math.cos(alpha) * n[0], math.cos(alpha) * n[1], math.sin(alpha),
            math.sin(alpha) * abs(n[2]), math.atan2(e1[1], e1[0]))


# ---------------------------------------------------------------------------
# method "star": rays about the star's centre
# ---------------------------------------------------------------------------

def _circle_chord(phi, xp, yp, rp):
    c, s = np.cos(phi), np.sin(phi)
    b = c * xp + s * yp
    disc = b * b - (xp * xp + yp * yp - rp * rp)
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    return np.clip(b - sq, 0, 1), np.clip(b + sq, 0, 1), hit


def _spot_chord(phi, n, alpha):
    """The ray's chord in the visible spot: s = sin t, t in [0, pi/2]."""
    m = np.cos(phi) * n[0] + np.sin(phi) * n[1]
    R = np.hypot(m, n[2])
    ca = math.cos(alpha)
    hit = R > ca
    delta = np.arccos(np.clip(ca / np.where(hit, R, 1.0), -1.0, 1.0))
    t0 = np.arctan2(m, n[2])
    tlo, thi = np.maximum(t0 - delta, 0.0), np.minimum(t0 + delta, _HALF_PI)
    hit = hit & (thi > tlo)
    return np.where(hit, np.sin(tlo), 0.0), np.where(hit, np.sin(thi), 0.0), hit


def _star_integrand(planet, spots, prims):
    def f(phi):
        chords = ([_circle_chord(phi, *planet)] if planet is not None else [])
        chords += [_spot_chord(phi, n, al) for n, al, _ in spots]
        ends = [np.zeros_like(phi), np.ones_like(phi)]
        for lo, hi, hit in chords:
            ends += [np.where(hit, lo, 0.0), np.where(hit, hi, 0.0)]
        ends = np.sort(np.stack(ends, -1), -1)
        out = np.zeros((phi.size, len(prims)))
        off = 1 if planet is not None else 0
        for k in range(ends.shape[1] - 1):
            lo, hi = ends[:, k], ends[:, k + 1]
            m = 0.5 * (lo + hi)
            ins = [hit & (a < m) & (m < b) for a, b, hit in chords]
            w = np.zeros(phi.size)
            for j, (_, _, c) in enumerate(spots):
                w = w + (1.0 - c) * ins[off + j]
            if planet is not None:
                w = np.where(ins[0], 1.0, w)
            w = np.where(hi > lo, w, 0.0)
            for j, p in enumerate(prims):
                out[:, j] += w * (p(hi) - p(lo))
        return out
    return f


def _star_breakpoints(planet, spots):
    angles = []
    if planet is not None:
        xp, yp, rp = planet
        angles += _tangent_angles(xp, yp, rp, rp) + _limb_angles(xp, yp, rp, rp)
    for i, (n, al, _) in enumerate(spots):
        nxy, g = math.hypot(n[0], n[1]), math.atan2(n[1], n[0])
        v2 = math.cos(al) ** 2 - n[2] ** 2                  # tangent rays: R = cos(alpha)
        if nxy > 0.0 and v2 >= 0.0:
            v = math.sqrt(v2) / nxy
            if v <= 1.0:
                for h in (math.acos(v), math.acos(-v)):
                    angles += [g + h, g - h]
        pts = _rim_limb_points(n, al)
        if planet is not None:
            pts += planet_rim_points(n, al, planet)
        for n2, al2, _ in spots[i + 1:]:
            pts += rim_rim_points(n, al, n2, al2)
        angles += [math.atan2(P[1], P[0]) for P in pts]
    start = 0.123456789
    return _turn(angles, start)


# ---------------------------------------------------------------------------
# method "split": each cap over mu, each lens by rays about the planet
# ---------------------------------------------------------------------------

def _cap_integrand(n, alpha, eps):
    nxy, ca = math.hypot(n[0], n[1]), math.cos(alpha)

    def f(mu):
        st = np.sqrt(np.maximum(1.0 - mu * mu, 0.0))
        if nxy == 0.0:
            L = np.where(mu * n[2] >= ca, 2 * math.pi, 0.0)
        else:
            den = st * nxy
            kap = (ca - mu * n[2]) / np.where(den > 0, den, 1.0)
            kap = np.where(den > 0, kap, np.where(ca - mu * n[2] <= 0, -2.0, 2.0))
            L = 2.0 * np.arccos(np.clip(kap, -1.0, 1.0))
        cols = [mu ** (2 * k + 1) * L for k in range(3)]
        cols += [mu * L / (mu * mu + e) ** 2 for e in eps]
        return np.stack(cols, -1)
    return f


def cap_cols(n, alpha, eps, rtol=1e-15):
    """Deviation columns of a spot's visible cap alone (weight 1): the
    integral of I(mu) mu over the cap's front part, dOmega = dmu dphi."""
    n = np.asarray(n, dtype=np.float64)
    beta = math.acos(max(-1.0, min(1.0, n[2])))
    pts = [0.0, 1.0] + [math.cos(t) for t in (beta + alpha, abs(beta - alpha))]
    pts = np.unique(np.clip(pts, 0.0, 1.0))
    f = _cap_integrand(n, alpha, tuple(eps))
    ncol = 3 + len(eps)
    return -sum(_adaptive(a, b, f, ncol, rtol) for a, b in zip(pts[:-1], pts[1:]) if b > a)


def _lens_integrand(planet, n, alpha, eps):
    xp, yp, rp = planet
    c0 = xp * xp + yp * yp
    ca = math.cos(alpha)
    L00 = ca - (xp * n[0] + yp * n[1])
    nz2 = n[2] * n[2]

    def inside(q_x, q_y):
        r2 = q_x * q_x + q_y * q_y
        mu = np.sqrt(np.maximum(1.0 - r2, 0.0))
        return (r2 < 1.0) & (q_x * n[0] + q_y * n[1] + mu * n[2] >= ca)

    def f(psi):
        cs, sn = np.cos(psi), np.sin(psi)
        b = xp * cs + yp * sn
        disc = b * b - c0 + 1.0
        meets = disc > 0
        sq = np.sqrt(np.where(meets, disc, 0.0))
        t1, t2 = -b - sq, -b + sq                       # the star's chord
        ell = cs * n[0] + sn * n[1]
        # n_z^2 (1 - |q|^2) = (cos(alpha) - q . n_xy)^2, a quadratic in rho
        A2 = -(nz2 + ell * ell)
        A1 = -2 * b * nz2 + 2 * L00 * ell
        A0 = nz2 * (1 - c0) - L00 * L00
        dq = A1 * A1 - 4 * A2 * A0
        real = dq >= 0
        sd = np.sqrt(np.where(real, dq, 0.0))
        A2s = np.where(A2 != 0, A2, -1.0)
        r1, r2 = (-A1 - sd) / (2 * A2s), (-A1 + sd) / (2 * A2s)
        ends = [np.zeros_like(psi), np.full_like(psi, rp)]
        for t in (t1, t2):
            ends.append(np.clip(np.where(meets, t, 0.0), 0, rp))
        for r in (r1, r2):
            ends.append(np.clip(np.where(real, r, 0.0), 0, rp))
        ends = np.sort(np.stack(ends, -1), -1)
        out = np.zeros((psi.size, 3 + len(eps)))
        for k in range(ends.shape[1] - 1):
            lo, hi = ends[:, k], ends[:, k + 1]
            m = 0.5 * (lo + hi)
            w = ((hi > lo) & meets & (t1 < m) & (m < t2)
                 & inside(xp + m * cs, yp + m * sn)).astype(float)
            for j in range(3):
                out[:, j] += w * _radial_even(j, b, c0, lo, hi)
            for j, e in enumerate(eps):
                out[:, 3 + j] += np.where(w != 0, w * _radial_pole(e, b, c0, lo, hi), 0.0)
        return out
    return f


def _lens_breakpoints(planet, n, alpha):
    xp, yp, rp = planet
    pts = planet_rim_points(n, alpha, planet) + _rim_limb_points(n, alpha)
    pts += [np.array([math.cos(t), math.sin(t), 0.0]) for t in _limb_angles(xp, yp, rp, rp)]
    angles = [math.atan2(P[1] - yp, P[0] - xp) for P in pts]
    ex, ey, A, B, th = _rim_ellipse(n, alpha)               # tangent rays to the rim
    if B > 1e-12 * A:
        c, s = math.cos(th), math.sin(th)
        X, Y = (ex - xp) * c + (ey - yp) * s, -(ex - xp) * s + (ey - yp) * c
        angles += [t + th for t in _tangent_angles(X, Y, A, B)]
    d = math.hypot(xp, yp)
    if d > 1.0:                                             # the star's tangent rays
        back, g = math.atan2(-yp, -xp), math.asin(1.0 / d)
        angles += [back - g, back + g]
    return _turn(angles, 0.123456789)


def lens_cols(planet, n, alpha, eps, rtol=1e-15):
    """Deviation columns of the planet's disc n spot k's visible region."""
    n = np.asarray(n, dtype=np.float64)
    f = _lens_integrand(planet, n, alpha, tuple(eps))
    pts = _lens_breakpoints(planet, n, alpha)
    ncol = 3 + len(eps)
    return -sum(_adaptive(a, b, f, ncol, rtol) for a, b in zip(pts[:-1], pts[1:]) if b > a)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def columns(planet, spots, eps, method="star", rtol=1e-15):
    """Deviation-form generator columns (3 + len(eps),) of a spotted star
    crossed by a planet. ``planet``: (xp, yp, rp) or None; ``spots``: a
    list of (n, alpha, c), n a unit 3-vector."""
    spots = [(np.asarray(n, dtype=np.float64), float(a), float(c)) for n, a, c in spots]
    eps = tuple(eps)
    ncol = 3 + len(eps)
    if method == "star":
        f = _star_integrand(planet, spots, _prims(eps))
        pts = _star_breakpoints(planet, spots)
        return -sum(_adaptive(a, b, f, ncol, rtol) for a, b in zip(pts[:-1], pts[1:]) if b > a)
    if method != "split":
        raise ValueError(method)
    out = np.zeros(ncol)
    if planet is not None:
        xp, yp, rp = planet
        out += region_cols(xp, yp, [(rp, rp)], lambda ins: ins[0].astype(float), eps,
                           "centre", rtol)
    for n, al, c in spots:
        part = cap_cols(n, al, eps, rtol)
        if planet is not None:
            part = part - lens_cols(planet, n, al, eps, rtol)
        out += (1.0 - c) * part
    return out
