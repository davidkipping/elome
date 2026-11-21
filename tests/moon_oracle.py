"""An independent reference for a planet and its moon, for the tests.

Shares no code with ``elome``: the flux two opaque discs block of a
radially symmetric intensity I(rho) is integrated in polar coordinates
about the star's centre,

    occ = int_0^1 I(rho) rho m(rho) d rho,

with m(rho) the angle of the circle of radius rho covered by the union of
the discs. Each disc covers one interval of that circle, cos(theta - g) >
(rho^2 + d^2 - R^2) / (2 rho d), or all or none of it; the union's measure
is |A| + |B| - |A n B| from the intervals in closed form. m(rho) has
square-root kinks where a disc's interval opens or closes (rho = |d - R|,
d + R) and where the two intervals' ends cross (the radii of the discs'
intersection points), so the integral is split there and each piece
integrated adaptively (scipy's QUADPACK).
"""

import math

import numpy as np
from scipy.integrate import quad

__all__ = ["occulted", "generator_cols", "quad_flux_dev"]

TWO_PI = 2.0 * math.pi


def _interval(rho, x, y, R):
    """(centre, half-width) of the arc of the circle of radius rho inside
    the disc (x, y, R); half-width pi: the whole circle, 0: none."""
    d = math.hypot(x, y)
    if rho + d <= R:
        return 0.0, math.pi
    if rho <= d - R or rho >= d + R or rho == 0.0:
        return 0.0, 0.0
    c = (rho * rho + d * d - R * R) / (2.0 * rho * d)
    return math.atan2(y, x), math.acos(max(-1.0, min(1.0, c)))


def _overlap(g1, h1, g2, h2):
    if h1 == 0.0 or h2 == 0.0:
        return 0.0
    if h1 == math.pi:
        return 2.0 * h2
    if h2 == math.pi:
        return 2.0 * h1
    dl = abs((g1 - g2 + math.pi) % TWO_PI - math.pi)
    lim = 2.0 * min(h1, h2)
    side = lambda sep: max(0.0, min(h1 + h2 - sep, lim))
    return min(side(dl) + side(TWO_PI - dl), lim)


def _measure(rho, discs, which):
    (g1, h1), (g2, h2) = (_interval(rho, *dsc) for dsc in discs)
    both = _overlap(g1, h1, g2, h2)
    if which == "lens":
        return both
    return 2.0 * h1 + 2.0 * h2 - both


def _breaks(discs):
    pts = {0.0, 1.0}
    for x, y, R in discs:
        d = math.hypot(x, y)
        pts.update([abs(d - R), d + R])
    (x1, y1, R1), (x2, y2, R2) = discs
    D = math.hypot(x2 - x1, y2 - y1)
    if abs(R1 - R2) < D < R1 + R2:
        a = (D * D + R1 * R1 - R2 * R2) / (2.0 * D)
        hh = math.sqrt(max(R1 * R1 - a * a, 0.0))
        ux, uy = (x2 - x1) / D, (y2 - y1) / D
        for sgn in (1.0, -1.0):
            pts.add(math.hypot(x1 + a * ux - sgn * hh * uy, y1 + a * uy + sgn * hh * ux))
    return sorted(p for p in pts if 0.0 <= p <= 1.0)


def occulted(discs, intensity, which="union"):
    """int over (union, or lens) of the discs, clipped to the star, of
    intensity(mu); discs = [(x, y, R), (x, y, R)]."""
    pts = _breaks(discs)
    total = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi - lo < 1e-15:
            continue
        f = lambda rho: intensity(math.sqrt(max(1.0 - rho * rho, 0.0))) * rho * _measure(
            rho, discs, which)
        v, _ = quad(f, lo, hi, epsabs=1e-16, epsrel=1e-14, limit=400)
        total += v
    return total


def generator_cols(discs, eps, which="union"):
    """Deviation columns (minus the occulted flux) of mu^0, mu^2, mu^4 and
    each double pole (mu^2 + e)^-2: the package's generator order."""
    fs = [lambda mu, n=n: mu ** (2 * n) for n in range(3)]
    fs += [lambda mu, e=e: (mu * mu + e) ** -2 for e in eps]
    return np.array([-occulted(discs, f, which) for f in fs])


def quad_flux_dev(discs, u1, u2):
    """F - 1 for the quadratic law."""
    I = lambda mu: 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    return -occulted(discs, I) / (math.pi * (1.0 - u1 / 3.0 - u2 / 6.0))
