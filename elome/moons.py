"""A planet with one moon: the sky positions of the planet and the moon.

**Orbits.** In the nested two-body approximation (Kipping 2011), the
barycentre of the planet and the moon follows the spherical path's orbit:
t0, per, a, inc and (ecc, w) describe the barycentre, t0 its inferior
conjunction. The moon goes round the planet on a Keplerian ellipse of its
own, unperturbed by the star. The planet and the moon sit on either side of
the barycentre in the ratio of their masses, q = M_moon / M_planet:

    planet = B - q / (1 + q) d,     moon = B + 1 / (1 + q) d,

with d the moon's position relative to the planet, of semi-major axis
a_ps (the planet-moon separation, in stellar radii here).

**Sky frame.** As ``oblate_tau.py``'s: X along the orbit (the direction of
motion at transit), Y across, Z towards the observer, in stellar radii. The
barycentre is at X = -u, Y = -v cos i, Z = v sin i, with u = r cos(w + f),
v = r sin(w + f) its orbital-plane Cartesians, so in front of the star
where v > 0.

**The moon's elements are referred to the sky**, as the planet's are (the
convention of gefera, Pandora and planetplanet): d is the planet's formula
with the moon's own inclination and argument of periastron, then turned by
the node angle about the line of sight, from X towards Y:

    d = R_z(node) (-u_m, -v_m cos(moon_inc), v_m sin(moon_inc)).

moon_inc = 90 degrees is edge-on, and a moon in the planet's orbital
plane, going round in the same sense, has moon_inc = inc and node = 0. No
element of the planet's orbit enters d, so the moon's orientation does not
move when (ecc, w) do. The mutual inclination iota between the two orbits
follows from

    cos(iota) = cos(inc) cos(moon_inc) + sin(inc) sin(moon_inc) cos(node)

(``mutual_inclination``). Turning the node, w_m and the phase by half a
turn mirrors the moon's orbit through the sky plane: every sky position
is unchanged and only the depths Z change sign. A transit of opaque discs
blocks the same area whichever is in front, so the two give the same light
curve, while one of them goes round the planet the other way (a
prograde moon's mirror is retrograde, exactly so when the planet's orbit is
edge-on). The sense of revolution is not measurable from the light curve,
and a sampler should take the node over half a turn.

**Phase.** ``phase`` is the moon's mean argument of latitude at t0 less a
quarter turn, M_m(t0) + w_m - pi / 2: on a circular orbit, the angle the
moon has gone round since its conjunction (w_m + f_m = pi / 2, where it is
nearest the observer for 0 < moon_inc < pi), and phase = 0 puts the moon at
its conjunction at t0 -- in front of the planet, on an edge-on orbit. So
the mirror above is (node, w_m, phase) -> (node, w_m, phase) + pi for any
eccentricity, and phase moves the moon by an amount of order e_m only when
w_m changes. The solve is anchored at the conjunction, as the planet's at
transit (``anchored.py``), which gives the moon's orbit the planet's
conditioning at small eccentricity: smooth in (e_m, w_m) through e_m = 0.
The anchored phase is the mean phase plus f_c - M_c, the true minus the
mean anomaly at conjunction, an O(e_m) quantity (``moon_orbit``).

**Float32.** A moon goes round many times per orbit of the planet, so its
phase at the k-th transit, 2 pi k P / P_m, is far too large to form in
float32 from traced periods: at k = 1000 and P / P_m = 30 the ulp alone is
2e-3 of a turn. The phase is split as the transit times are
(``orbit.epoch_center_times``): the turns k P_ref / P_m,ref of fixed
reference periods are reduced on the host in float64 (``moon_turns_ref``),
and the graph adds only k times the small offset of the ratio from its
reference, formed from the period offsets:

    k P / P_m - k P_ref / P_m,ref = k (p_off P_m,ref - P_ref m_off) / (P_m P_m,ref),

with P = P_ref + p_off and P_m = P_m,ref + m_off.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .anchored import _one_minus_cos, anchor_constants_ew, solve_sincos_delta
from .dtypes import fp64_on_cpu
from .flux import flux_dev
from .hybrid import (_combine, _generator_terms, _may_be_large, combine_cols, declared,
                     get_law, lens_geometry)
from .rings import _SAMPLES, _TINY, _decisive
from .trig import sincos

__all__ = ["moon_turns_ref", "moon_phase", "moon_offset", "barycentre_sky",
           "body_positions", "anchored_constants", "moon_orbit", "moon_orbit_kh",
           "mutual_inclination", "moon_period", "check_moon", "generator_cols_moon",
           "shape_cols_moon", "flux_dev_moon", "flux_dev_moon_quad"]

_TWO_PI = 2.0 * math.pi
_sg = mx.stop_gradient
# G / (3 pi) in m^3 kg^-1 day^-2: P_m^2 = (a_ps / R_p)^3 / (_G3PI rho_p (1 + q))
_G3PI = 6.67430e-11 * 86400.0 ** 2 / (3.0 * math.pi)


def _sincos_any(x):
    if isinstance(x, mx.array):
        return sincos(x)
    return math.sin(x), math.cos(x)


def _sqrt_any(x):
    if isinstance(x, mx.array):
        return mx.sqrt(mx.maximum(x, 0.0))
    return math.sqrt(max(x, 0.0))


# --------------------------------------------------------------------------
# the moon's phase
# --------------------------------------------------------------------------

def moon_turns_ref(k, per_ref, moon_per_ref) -> np.ndarray:
    """Float64 host preprocessing: k P_ref / P_m,ref reduced to [-1/2, 1/2)
    turns, for orbit numbers k (``orbit.epoch_center_times``' second row)."""
    x = np.asarray(k, np.float64) * (float(per_ref) / float(moon_per_ref))
    return x - np.floor(x + 0.5)


def moon_phase(tau, k, turns_ref, per_ref, moon_per_ref, p_off, m_off, phase):
    """The moon's mean phase 2 pi (t - t0) / P_m + phase, its first term
    reduced to a turn.

    tau: time from the barycentre's mid-transit of orbit k (``orbit.
    tau_from_epochs``); turns_ref: ``moon_turns_ref(k, per_ref,
    moon_per_ref)``; p_off, m_off: the planet's and the moon's period offsets
    from the references; phase: the moon's phase at t0 [radians]."""
    moon_per = moon_per_ref + m_off
    d_ratio = (p_off * moon_per_ref - per_ref * m_off) / (moon_per * moon_per_ref)
    turns = turns_ref + k * d_ratio + tau / moon_per
    turns = turns - mx.round(turns)
    return _TWO_PI * turns + phase


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

def _orbit_uv(phi, a, ecc):
    """(u, v) = r (cos(w + f), sin(w + f)) at anchored phase phi on an orbit
    of semi-major axis a; ``ecc`` None (circular) or the anchored constants
    (``anchored.anchor_constants`` / ``anchor_constants_ew``)."""
    if ecc is None:
        sd, cd = sincos(phi)
        return -(a * sd), a * cd
    e, ecw, esw, es, ec, B1, A2, B2 = ecc
    sd, cd = solve_sincos_delta(phi, es, ec)
    omc = _one_minus_cos(sd, cd)
    return a * (-ecw * omc - B1 * sd), a * (A2 * cd - B2 * sd - esw)


def moon_orbit(ecc, w, dtype):
    """(anchored constants, f_c - M_c) of the moon's orbit (e_m, w_m
    [radians]), or None for a circular one (ecc None or a host 0). f_c - M_c
    = Delta + e sin E_c with Delta = f_c - E_c in ``anchored._anchor_core``'s
    cancellation-free form: O(e), smooth through e = 0."""
    consts = anchored_constants(ecc, w, dtype)
    if consts is None:
        return None
    e, w = consts[0], (w if isinstance(w, mx.array) else mx.array(float(w), dtype=dtype))
    s2, c2 = sincos(0.5 * (0.5 * math.pi - w))
    kk = mx.sqrt(mx.maximum((1.0 - e) / (1.0 + e), 0.0))
    one_m_kk = 2.0 * e / ((1.0 + e) * (1.0 + kk))
    delta = 2.0 * mx.arctan2(one_m_kk * s2 * c2, c2 * c2 + kk * s2 * s2)
    return consts, delta + consts[3]


def moon_orbit_kh(k, h):
    """``moon_orbit`` from (k, h) = sqrt(e_m) (cos w_m, sin w_m), a sampler's
    pair: w_m from the guarded (sin w_m, cos w_m), as
    ``anchored.anchor_constants`` forms it, so e_m = 0 stays an interior
    point with finite gradients."""
    e = k * k + h * h
    pos = e > 0.0
    den = mx.where(pos, mx.maximum(mx.sqrt(mx.maximum(e, 1e-30)), 1e-30), mx.ones_like(e))
    w = mx.arctan2(mx.where(pos, h / den, mx.zeros_like(h)),
                   mx.where(pos, k / den, mx.ones_like(k)))
    return moon_orbit(e, w, e.dtype)


def moon_offset(phi_m, a_ps, moon_inc, node, orbit=None):
    """The moon's position relative to the planet, (dX, dY, dZ), in the sky
    frame. phi_m: mean phase (``moon_phase``); a_ps: semi-major axis of the
    relative orbit [stellar radii]; moon_inc, node [radians]; orbit: None
    (circular) or ``moon_orbit(e_m, w_m)``."""
    if orbit is None:
        u, v = _orbit_uv(phi_m, a_ps, None)
    else:
        consts, off = orbit
        u, v = _orbit_uv(phi_m + off, a_ps, consts)
    si, ci = _sincos_any(moon_inc)
    sn, cn = _sincos_any(node)
    vc = v * ci
    return -(cn * u - sn * vc), -(sn * u + cn * vc), v * si


def barycentre_sky(phi, a, inc, ecc=None):
    """(X, Y, Z) of the barycentre at phase phi = 2 pi tau / P; inc in
    radians; ``ecc`` None (circular) or the anchored constants of (e, w)."""
    si, ci = _sincos_any(inc)
    u, v = _orbit_uv(phi, a, ecc)
    return -u, -(v * ci), v * si


def body_positions(XB, YB, ZB, dX, dY, dZ, q):
    """((Xp, Yp, Zp), (Xm, Ym, Zm)): the planet and the moon on either side
    of the barycentre, q = M_moon / M_planet."""
    fp = q / (1.0 + q)
    fm = 1.0 / (1.0 + q)
    return ((XB - fp * dX, YB - fp * dY, ZB - fp * dZ),
            (XB + fm * dX, YB + fm * dY, ZB + fm * dZ))


def anchored_constants(ecc, w, dtype):
    """The anchored constants of an orbit (e, w [radians]), or None for a
    circular one (ecc None or a host 0). Host values are cast to ``dtype``:
    the constants' mx ops would make float32 scalars of Python floats."""
    if ecc is None or (not isinstance(ecc, mx.array) and float(ecc) == 0.0):
        return None
    cast = lambda v: v if isinstance(v, mx.array) else mx.array(float(v), dtype=dtype)
    return anchor_constants_ew(cast(ecc), cast(w))


# --------------------------------------------------------------------------
# host helpers
# --------------------------------------------------------------------------

def mutual_inclination(inc, moon_inc, node):
    """(iota, iota_mirror) in degrees: the mutual inclination of the moon's
    orbit to the barycentre's, and that of its mirror (node + 180 degrees),
    which gives the same light curve. Arguments in degrees; numpy arrays
    broadcast."""
    i, im, n = (np.radians(np.asarray(x, np.float64)) for x in (inc, moon_inc, node))
    a = np.cos(i) * np.cos(im)
    b = np.sin(i) * np.sin(im) * np.cos(n)
    return (np.degrees(np.arccos(np.clip(a + b, -1.0, 1.0))),
            np.degrees(np.arccos(np.clip(a - b, -1.0, 1.0))))


def moon_period(moon_a, rho_p, q=0.0):
    """The moon's orbital period [days] from Kepler's third law:
    moon_a the planet-moon semi-major axis in planet radii, rho_p the
    planet's mean density [kg m^-3], q = M_moon / M_planet."""
    moon_a, rho_p, q = (np.asarray(x, np.float64) for x in (moon_a, rho_p, q))
    return np.sqrt(moon_a ** 3 / (_G3PI * rho_p * (1.0 + q)))


def check_moon(moon_r, moon_a, moon_q, moon_ecc=0.0):
    """Host-value checks. moon_r and moon_a are in planet radii: the moon
    must lie outside the planet at periastron, moon_a (1 - e) > 1 + moon_r."""
    vals = [np.asarray(v, np.float64) for v in (moon_r, moon_a, moon_q, moon_ecc)]
    r, a, q, e = vals
    if not all(np.all(np.isfinite(v)) for v in vals):
        raise ValueError("moon parameters must be finite")
    if np.any(r < 0.0):
        raise ValueError("moon_r must be >= 0")
    if np.any(q < 0.0):
        raise ValueError("moon_q (M_moon / M_planet) must be >= 0")
    if np.any((e < 0.0) | (e >= 1.0)):
        raise ValueError("moon_ecc must be in [0, 1)")
    if np.any(a * (1.0 - e) <= 1.0 + r):
        raise ValueError("the moon must orbit outside the planet: "
                         "moon_a (1 - moon_ecc) > 1 + moon_r (planet radii)")


# --------------------------------------------------------------------------
# photometry
# --------------------------------------------------------------------------
#
# Two opaque discs on the star, the planet P (radius rp) and the moon M (rs),
# block
#
#     occ(P u M) = occ(P) + occ(M) - occ(L),     L = P n M n star,
#
# each occ the occulted flux of the disc's part on the star. occ(P) and
# occ(M) are the spherical path's elementary solutions, exact for any law
# and any radius; occ(L), the lens the two discs share, is 0 unless they
# overlap on the star.
#
# The lens by Green's theorem (``rings.py``): with h(s) = (1/2s) int_0^s I,
#
#     occ(L) = sum over each circle's sub-arcs inside the star and inside the
#              other circle of int h(s) (x dy - y dx)
#            + sum over the limb's arcs inside both circles of h(1) dtheta.
#
# Each circle is split at its two limb points and its two corners with the
# other circle, each point computed once and shared by every curve through
# it (``_meet``); each arc's membership, of the star and of the other
# circle, is read at three points (``rings._SAMPLES``) -- an arc has no
# corner with the curve it is tested against, so every interior point gives
# the same answer except where the curves merely touch.
#
# On a circle of radius r whose centre is z from the star's, with psi the
# angle from its point nearest the star's centre, s = z^2 + r^2 - 2 z r
# cos(psi) and x dy - y dx = (s + r^2 - z^2) / 2 dpsi, so every column of an
# arc is elementary (``_arc_cols``): mu^{2n}'s h_n(s) is a polynomial in s,
# its integrand at most a cubic in cos(psi); a pole's h = 1 / (2p (p - s))
# gives
#
#     int (s + r^2 - z^2) / (4p (p - s)) dpsi
#         = -dpsi / 4p + (p + r^2 - z^2) / 4p int dpsi / (a + b cos psi),
#
# a = p - z^2 - r^2, b = 2 z r: an atan, an atanh, or their common series
# near a = b (``_J``) -- the forms of ``hybrid._pole_terms``, at any limits.
# The arcs' ends are live (the shared points move with the discs), so the
# gradients carry the ends' motion; only the membership is frozen.
#
# For the hybrid laws every column of every piece is elementary, and the
# light curve is exact. The quadratic law's mu^1 term over the lens would
# need incomplete elliptic integrals, so the lens alone takes a small-body
# closure (``_lens_mu``); the discs stay exact.

# a crossing pair closer to a tangency than this half-chord (times the
# smaller radius) counts as none: the circles then touch, and the sliver
# between them, of area ~ R^2 THIN^3, is below the precision's rounding of
# the columns; nearer the touch, where each crossing angle carries rounding
# amplified as 1 / (half-chord), the arcs on either side would be split at
# points whose membership tests are themselves at rounding level
_THIN = {mx.float64: 1e-4, mx.float32: 1e-2}
# half-width in x = (a - b) / (a + b) tan^2(psi / 2) of the series bridge
# across a = b (as hybrid's): six terms truncate at x^6 / 13
_X_SER = {mx.float64: 1e-3, mx.float32: 0.05}


def _meet(c1, c2, dt):
    """Where circle c1 = (x, y, R) meets c2: the two points (live), computed
    once so that every curve through a corner is split at the same point,
    and whether they exist -- not within _THIN of a tangency."""
    (x1, y1, R1), (x2, y2, R2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    D2 = dx * dx + dy * dy
    D = mx.sqrt(mx.maximum(D2, _TINY[dt]))
    a = (D2 + (R1 - R2) * (R1 + R2)) / (2.0 * D)       # along c1 -> c2 from c1
    hh2 = (R1 - a) * (R1 + a)
    thin = _THIN[dt] * mx.minimum(R1, R2)
    has = (hh2 > thin * thin) & (D2 > 0.0)
    hh = mx.where(has, mx.sqrt(mx.where(has, hh2, 1.0)), 0.0)
    ux, uy = dx / D, dy / D
    bx, by = x1 + a * ux, y1 + a * uy
    return [(bx - hh * uy, by + hh * ux), (bx + hh * uy, by - hh * ux)], has


def _sorted_arcs(bps, valid, start, period):
    """Sub-arcs between a closed curve's valid breakpoints (live angles,
    sorted by their frozen values), as (lo, hi, ok) lists of fixed length:
    the last wraps round by ``period``; with no breakpoint at all, one
    whole turn from ``start``."""
    K = len(bps)
    B = mx.stack(bps, -1)
    V = mx.stack(valid, -1)
    order = _sg(mx.argsort(mx.where(V, _sg(B), mx.inf), axis=-1))
    B = mx.take_along_axis(B, order, axis=-1)
    n = mx.sum(V.astype(mx.int32), axis=-1)
    first = B[..., 0]
    los, his, oks = [], [], []
    for i in range(K):
        lo = B[..., i]
        hi = B[..., i + 1] if i + 1 < K else first
        hi = mx.where(i + 1 < n, hi, first + period)
        whole = (n == 0) & (i == 0)
        los.append(mx.where(whole, start, lo))
        his.append(mx.where(whole, start + period, hi))
        oks.append((i < n) | whole)
    return los, his, oks


def _J(psi, amb, apb, Q, dt):
    """An antiderivative of 1 / (a + b cos psi) at psi in (-2 pi, 2 pi),
    continuous wherever a + b cos psi > 0 -- every point of an arc on the
    star; amb = a - b, apb = a + b > 0 (a circle reaching the star), Q =
    amb apb:

        Q > 0:  (2 / sqrt Q) atan2(sqrt(a - b) sin(psi/2), sqrt(a + b) cos(psi/2))
        Q < 0:  (2 / sqrt(-Q)) atanh(sqrt(b - a) tan(psi/2) / sqrt(a + b))
        |x| < _X_SER, x = (a - b) / (a + b) tan^2(psi/2), |psi| < pi:
                (2 / (a + b)) tan(psi/2) sum_k (-x)^k / (2k + 1),

    the series bridging Q = 0, where both closed forms are 0/0."""
    sh, ch = sincos(0.5 * psi)
    t = sh / mx.where(ch > 0.0, ch, 1.0)
    x = (amb / apb) * t * t
    ser = (mx.abs(x) < _X_SER[dt]) & (ch > 0.0)
    xs = mx.where(ser, x, 0.0)
    acc = 0.0 * xs + 1.0 / 11.0
    for k in range(4, -1, -1):
        acc = 1.0 / (2 * k + 1) - xs * acc
    f_ser = 2.0 * t / apb * acc
    pos = Q > 0.0
    Qp = mx.where(pos & ~ser, Q, 1.0)
    ap = mx.where(pos & ~ser, amb, 1.0)
    f_pos = 2.0 / mx.sqrt(Qp) * mx.arctan2(mx.sqrt(ap) * sh, mx.sqrt(apb) * ch)
    neg = ~pos & ~ser
    Qn = mx.where(neg, -Q, 1.0)
    w = mx.where(neg, mx.sqrt(mx.where(neg, -amb, 1.0)) * t / mx.sqrt(apb), 0.0)
    f_neg = 1.0 / mx.sqrt(Qn) * mx.log1p(2.0 * w / (1.0 - w))
    return mx.where(ser, f_ser, mx.where(pos, f_pos, f_neg))


# mu^{2n}'s integrand h_n(s) (s + d) / 2, d = r^2 - z^2, as polynomial
# coefficients in s: h_0 = 1/2, h_1 = (2 - s)/4, h_2 = (3 - 3s + s^2)/6
def _even_poly(n, d):
    if n == 0:
        return [0.25 * d, 0.25 + 0.0 * d]
    if n == 1:
        return [0.25 * d, (2.0 - d) / 8.0, -0.125 + 0.0 * d]
    return [0.25 * d, (1.0 - d) / 4.0, (d - 3.0) / 12.0, 1.0 / 12.0 + 0.0 * d]


def _arc_cols(z, r, lo, hi, eps):
    """The generator columns [mu^0, mu^2, mu^4, poles...] of an arc [lo,
    hi] of a circle of radius r, centre z from the star's, psi from its
    point nearest the star's centre, (lo, hi) in (-2 pi, 2 pi)."""
    dt = lo.dtype
    S, b, d = z * z + r * r, 2.0 * z * r, r * r - z * z
    sl, cl = sincos(lo)
    sh, chh = sincos(hi)
    # int cos^k: psi, sin, psi/2 + sin cos/2, sin - sin^3/3
    A = [hi - lo, sh - sl, 0.5 * (hi - lo) + 0.5 * (sh * chh - sl * cl),
         (sh - sl) - (sh * sh * sh - sl * sl * sl) / 3.0]
    cols = []
    for n in range(3):
        ps = _even_poly(n, d)
        # (S - b c)^j -> coefficients in c
        col = 0.0 * lo
        for j, pj in enumerate(ps):
            for kk in range(j + 1):
                col = col + pj * math.comb(j, kk) * S ** (j - kk) * (-b) ** kk * A[kk]
        cols.append(col)
    for e in eps:
        p = 1.0 + e
        sp = math.sqrt(p)
        amb = (sp - z - r) * (sp + z + r)
        dzr = mx.abs(z - r)
        apb = (sp - dzr) * (sp + dzr)
        # a circle wholly beyond sqrt(p) has no arc on the star: stand-ins,
        # so its masked lanes carry no NaN into the gradient
        reach = apb > 0.0
        amb, apb = mx.where(reach, amb, 0.5), mx.where(reach, apb, 1.0)
        J = _J(hi, amb, apb, amb * apb, dt) - _J(lo, amb, apb, amb * apb, dt)
        cols.append(-(hi - lo) / (4.0 * p) + (p + d) / (4.0 * p) * J)
    return cols


def _lens_cols(circles, eps, dt):
    """Occulted generator columns (positive) [mu^0, mu^2, mu^4, poles...] of
    the lens of two circles on the star; circles = [(x0, y0, R), (x1, y1,
    R1)], each (n,)."""
    ncol = 3 + len(eps)
    occ = [0.0 * circles[0][0] for _ in range(ncol)]
    zero = 0.0 * circles[0][0]
    star = (zero, zero, zero + 1.0)
    limb = [_meet(circles[k], star, dt) for k in range(2)]
    corners = _meet(circles[0], circles[1], dt)
    limb_th, limb_ok = [], []
    for k in range(2):
        x0, y0, R = circles[k]
        ox, oy, oR = (_sg(v) for v in circles[1 - k])
        z2 = x0 * x0 + y0 * y0
        cen = _sg(z2) < _TINY[dt]
        z = mx.sqrt(mx.where(cen, 1.0, z2)) * (~cen).astype(dt)
        # towards the star's centre from this one's (any direction if centred)
        ux = mx.where(cen, -1.0, -x0 / mx.where(cen, 1.0, z))
        uy = mx.where(cen, 0.0, -y0 / mx.where(cen, 1.0, z))
        psi = lambda pt: mx.arctan2(ux * (pt[1] - y0) - uy * (pt[0] - x0),
                                    ux * (pt[0] - x0) + uy * (pt[1] - y0))
        pts, has = limb[k]
        for pt in pts:
            limb_th.append(mx.arctan2(pt[1], pt[0]))
            limb_ok.append(has)
        bps = [psi(pt) for pt in pts] + [psi(pt) for pt in corners[0]]
        bv = [has, has, corners[1], corners[1]]
        los, his, oks = _sorted_arcs(bps, bv, -math.pi + 0.0 * z, _TWO_PI)
        LO, HI, OK = mx.stack(los, -1), mx.stack(his, -1), mx.stack(oks, -1)
        # the wrapping arc into (-2 pi, 2 pi)
        shift = ((LO > 0.0) & (HI > math.pi)).astype(dt) * _TWO_PI   # not where(c, 2pi, 0): fp32
        LO, HI = LO - shift, HI - shift
        # membership at three points of each arc, frozen
        gLO, gHI = _sg(LO), _sg(HI)
        b = lambda v: _sg(v)[..., None]
        gx, gy, gR, gux, guy = b(x0), b(y0), b(R), b(ux), b(uy)
        m_star, m_other = [], []
        for t in _SAMPLES:
            sp_, cp_ = sincos(gLO + t * (gHI - gLO))
            xs = gx + gR * (gux * cp_ - guy * sp_)
            ys = gy + gR * (gux * sp_ + guy * cp_)
            m_star.append(1.0 - (xs * xs + ys * ys))
            u_, v_ = (xs - ox[..., None]) / oR[..., None], (ys - oy[..., None]) / oR[..., None]
            m_other.append(1.0 - (u_ * u_ + v_ * v_))
        keep = OK & (_decisive(m_star) > 0.0) & (_decisive(m_other) > 0.0)
        cols = _arc_cols(z[..., None], R[..., None], mx.where(keep, LO, 0.0),
                         mx.where(keep, HI, 0.0), eps)
        for idx in range(ncol):
            occ[idx] = occ[idx] + mx.sum(mx.where(keep, cols[idx], 0.0), axis=-1)
    # the limb's arcs inside both circles
    los, his, oks = _sorted_arcs(limb_th, limb_ok, 0.0 * zero, _TWO_PI)
    h1 = [1.0 / (2.0 * (n + 1.0)) for n in range(3)] + [1.0 / (2.0 * (1.0 + e) * e) for e in eps]
    g = [tuple(_sg(v) for v in c) for c in circles]
    for lo, hi, ok in zip(los, his, oks):
        ms = [[], []]
        glo, ghi = _sg(lo), _sg(hi)
        for t in _SAMPLES:
            sm, cm = sincos(glo + t * (ghi - glo))
            for j, (gx0, gy0, gR) in enumerate(g):
                u, v = (cm - gx0) / gR, (sm - gy0) / gR
                ms[j].append(1.0 - (u * u + v * v))
        w = ok & (_decisive(ms[0]) > 0.0) & (_decisive(ms[1]) > 0.0)
        for idx in range(ncol):
            occ[idx] = occ[idx] + mx.where(w, h1[idx] * (hi - lo), 0.0)
    return occ


def _separation(x, y, r, front):
    """z = |(x, y)|, floored for the backward pass at z = 0 as the orbit's
    separations are, and pushed beyond 1 + r off the front side."""
    eps = 1.2e-7 if x.dtype == mx.float32 else 2.3e-16
    z = mx.sqrt(mx.maximum(x * x + y * y, (10.0 * eps) ** 2))
    return z if front is None else mx.where(front, z, z + 2.0 + r)


def _both_front(front_p, front_m):
    if front_p is None:
        return front_m
    return front_p if front_m is None else front_p & front_m


def _large(rp, large):
    if large is not None:
        return bool(large)
    flag = _may_be_large(rp)
    return flag is None or bool(flag)


def generator_cols_moon(xp, yp, xm, ym, rp, rs, law, front_p=None, front_m=None,
                        large_occultor=None):
    """Generator deviation columns [E0, E1, E2, P_k...] (minus the occulted
    flux of mu^0, mu^2, mu^4 and each pole of ``law``) of a planet at (xp,
    yp) and a moon at (xm, ym) on the sky, radii rp, rs [stellar radii].
    front_p, front_m: whether each body is on the observer's side of the
    star (None: yes). ``large_occultor``: whether rp may exceed 1, as
    ``hybrid.flux_dev_hybrid`` takes it."""
    law = get_law(law)
    dt = xp.dtype
    yp, xm, ym, rp, rs = (_cast(v, dt) for v in (yp, xm, ym, rp, rs))
    zero = 0.0 * (xp + yp + xm + ym + rp + rs)
    xp, yp, xm, ym, rp, rs = (v + zero for v in (xp, yp, xm, ym, rp, rs))
    with declared(large_occultor):
        gp = _generator_terms(lens_geometry(_separation(xp, yp, rp, front_p), rp), law)[0]
        gm = _generator_terms(lens_geometry(_separation(xm, ym, rs, front_m), rs), law)[0]
        large = _large(rp, large_occultor)
    lens = _lens_cols([(xp, yp, rp), (xm, ym, rs)], law.eps, dt)
    both = _both_front(front_p, front_m)
    if both is not None:
        lens = [mx.where(both, c, 0.0) for c in lens]
    return [a + b + c for a, b, c in zip(gp, gm, lens)]


@fp64_on_cpu(any_arg=True)
def shape_cols_moon(xp, yp, xm, ym, rp, rs, law, front_p=None, front_m=None,
                    large_occultor=None) -> mx.array:
    """Shape-basis columns [E0, S_1..S_n] of a planet and its moon, stacked
    on a trailing axis -- the ``ld_basis`` contract of the other paths
    (flux - 1 == (B @ c) / (N @ c), c = (1, -w), N = hybrid_norms(law)).
    Arguments as ``generator_cols_moon``; all are cast to xp's dtype."""
    law = get_law(law)
    gen = generator_cols_moon(xp, yp, xm, ym, rp, rs, law, front_p, front_m,
                              large_occultor)
    S = law.generator_matrix()
    return mx.stack([gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)], axis=-1)


@fp64_on_cpu(any_arg=True)
def flux_dev_moon(xp, yp, xm, ym, rp, rs, w, law, front_p=None, front_m=None,
                  large_occultor=None) -> mx.array:
    """F - 1 of a planet and its moon for a hybrid law with shape weights w
    (as ``hybrid.flux_dev_hybrid`` takes them); arguments as
    ``generator_cols_moon``."""
    B = shape_cols_moon(xp, yp, xm, ym, rp, rs, law, front_p, front_m, large_occultor)
    return combine_cols([B[..., k] for k in range(B.shape[-1])], w, law)


def _lens_mu(m0, m2, m4, dt):
    """The lens's integral of mu, from its exact moments m_n = int mu^n dA
    (n = 0, 2, 4), by a distribution of x = mu^2 over the lens that matches
    its mean and variance:

    * uniform on [a, b] = mean -+ sqrt(3 var) while a >= 0: int mu = m0
      (2/3)(a + sqrt(ab) + b) / (sqrt a + sqrt b), the mean of sqrt over
      [a, b] written without cancellation;
    * on a lens reaching the limb (a < 0), the density x^k on [0, b]:
      E[x^2] / E[x]^2 = 1 + 1 / ((k + 1)(k + 3)) gives k, b = E[x] (k + 2) /
      (k + 1) and E[sqrt x] = sqrt(b) (k + 1) / (k + 3/2). At k = 0 it is
      the uniform [0, b] of the first branch, so the two join continuously.

    Exact for a lens at one mu; on a 0.03 moon its error is ~1e-3 ppm on
    the disc and up to ~0.7 ppm while the lens straddles the limb."""
    tiny = _TINY[dt]
    # an empty lens takes stand-in moments (mean 1/2, variance 1/20): a masked
    # lane must not poison the gradient with sqrt'(0)
    ok = m0 > tiny
    m0s = mx.where(ok, m0, 1.0)
    mean = mx.maximum(mx.where(ok, m2, 0.5) / m0s, tiny)
    var = mx.where(ok, m4, 0.3) / m0s - mean * mean
    spread = var > 0.0
    half = mx.where(spread, mx.sqrt(3.0 * mx.where(spread, var, 1.0)), 0.0)
    var = mx.where(spread, var, 0.0)
    limb = half > mean
    a = mx.where(limb, mean, mean - half)        # the uniform branch kept finite
    b = mean + half
    ra, rb = mx.sqrt(a), mx.sqrt(b)
    uni = (2.0 / 3.0) * (a + ra * rb + b) / (ra + rb)
    # q = E[x]^2 / var < 3 and k + 1 = sqrt(1 + q) - 1 without cancellation,
    # E[sqrt x] = sqrt(E[x] (k + 1) (k + 2)) / (k + 3/2) without dividing by
    # k + 1: a lens barely on the star (mean -> 0) takes k -> -1
    q = mx.where(limb, mean * mean / mx.where(limb, var, 1.0), 1.0)
    kp1 = q / (mx.sqrt(1.0 + q) + 1.0)
    pw = mx.sqrt(mx.maximum(mean * kp1 * (kp1 + 1.0), tiny)) / (kp1 + 0.5)
    return mx.where(ok, m0s * mx.where(limb, pw, uni), 0.0)


@fp64_on_cpu(any_arg=True)
def flux_dev_moon_quad(xp, yp, xm, ym, rp, rs, u1, u2, front_p=None,
                       front_m=None) -> mx.array:
    """F - 1 of a planet and its moon for the quadratic law
    I = 1 - u1 (1 - mu) - u2 (1 - mu)^2: each disc exact (ALFM19), the lens
    they share by its exact mu^0 and mu^2 moments and ``_lens_mu``'s
    closure for mu^1. Arguments as ``generator_cols_moon``."""
    dt = xp.dtype
    yp, xm, ym, rp, rs = (_cast(v, dt) for v in (yp, xm, ym, rp, rs))
    zero = 0.0 * (xp + yp + xm + ym + rp + rs)
    xp, yp, xm, ym, rp, rs = (v + zero for v in (xp, yp, xm, ym, rp, rs))
    dev = (flux_dev(_separation(xp, yp, rp, front_p), rp, u1, u2)
           + flux_dev(_separation(xm, ym, rs, front_m), rs, u1, u2))
    m0, m2, m4 = _lens_cols([(xp, yp, rp), (xm, ym, rs)], (), dt)
    occ = (1.0 - u1 - u2) * m0 + (u1 + 2.0 * u2) * _lens_mu(m0, m2, m4, dt) - u2 * m2
    both = _both_front(front_p, front_m)
    if both is not None:
        occ = mx.where(both, occ, 0.0)
    return dev + occ / (math.pi * (1.0 - u1 / 3.0 - u2 / 6.0))


# the pieces, for routes that take the lens on a subset of points
# (``moons_tau``): each returns its contribution to the shape columns
# [E0, S_1..S_n] (hybrid) or to F - 1 (quadratic)

def _shape_of(gen, law):
    S = law.generator_matrix()
    return [gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)]


def disc_shape_cols(x, y, r, law, front=None):
    """One disc's shape columns (deviation form), under the caller's
    large_occultors declaration."""
    return _shape_of(_generator_terms(lens_geometry(_separation(x, y, r, front), r), law)[0], law)


def lens_shape_cols(xp, yp, rp, xm, ym, rs, law):
    """The lens's shape columns, positive (the flux the discs' columns count
    twice): add them to the two discs'."""
    return _shape_of(_lens_cols([(xp, yp, rp), (xm, ym, rs)], law.eps, xp.dtype), law)


def disc_flux_quad(x, y, r, u1, u2, front=None):
    """One disc's F - 1, quadratic law."""
    return flux_dev(_separation(x, y, r, front), r, u1, u2)


def lens_flux_quad(xp, yp, rp, xm, ym, rs, u1, u2):
    """The lens's flux (positive, normalised as F - 1): add it to the two
    discs' F - 1."""
    dt = xp.dtype
    m0, m2, m4 = _lens_cols([(xp, yp, rp), (xm, ym, rs)], (), dt)
    occ = (1.0 - u1 - u2) * m0 + (u1 + 2.0 * u2) * _lens_mu(m0, m2, m4, dt) - u2 * m2
    return occ / (math.pi * (1.0 - u1 / 3.0 - u2 / 6.0))


def _cast(v, dt):
    if isinstance(v, mx.array):
        return v if v.dtype == dt else v.astype(dt)
    return mx.array(float(v), dtype=dt)
