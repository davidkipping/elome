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
from .oblate import _C, _N_MAX, _coeffs, _even_arc, _even_arc_gl, _mod2pi, _pole_arc, _q_coeffs
from .rings import _SAMPLES, _TINY, _arcs, _centred, _circle_roots, _decisive, _h
from .trig import sincos

__all__ = ["moon_turns_ref", "moon_phase", "moon_offset", "barycentre_sky",
           "body_positions", "anchored_constants", "moon_orbit", "mutual_inclination",
           "moon_period", "check_moon", "generator_cols_moon", "shape_cols_moon",
           "flux_dev_moon", "flux_dev_moon_quad"]

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
# other circle (closed form), and each arc's membership, of the star and of
# the other circle, is read at three points (``rings._SAMPLES``): an arc
# has no corner with the curve it is tested against, so every interior
# point gives the same answer except where the curves merely touch. The
# arcs are ``oblate.py``'s per-arc primitives with a circle's closed-form
# roots (``rings._circle_roots``); gradients by frozen endpoints plus the
# corner term, as ``rings._region_cols``.
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


def _meet(c1, c2, dt):
    """Where circle c1 = (x, y, R) meets c2: the two points, computed once
    so that every curve through a corner is split at the same point, and
    whether they exist -- not within _THIN of a tangency. Frozen inputs."""
    (x1, y1, R1), (x2, y2, R2) = c1, c2
    dx, dy = x2 - x1, y2 - y1
    D2 = dx * dx + dy * dy
    D = mx.sqrt(mx.maximum(D2, _TINY[dt]))
    a = (D2 + (R1 - R2) * (R1 + R2)) / (2.0 * D)       # along c1 -> c2 from c1
    hh2 = (R1 - a) * (R1 + a)
    thin = _THIN[dt] * mx.minimum(R1, R2)
    has = (hh2 > thin * thin) & (D2 > 0.0)
    hh = mx.sqrt(mx.maximum(hh2, 0.0))
    ux, uy = dx / D, dy / D
    bx, by = x1 + a * ux, y1 + a * uy
    return [(bx - hh * uy, by + hh * ux), (bx + hh * uy, by - hh * ux)], has


def _angle_on(c, pt):
    return _mod2pi(mx.arctan2(pt[1] - c[1], pt[0] - c[0]))


def _lens_cols(circles, eps, dt, large=False):
    """Occulted generator columns (positive) [mu^0, mu^2, mu^4, poles...] of
    the lens of two circles on the star; circles = [(x0, y0, R), (x1, y1,
    R1)], each (n,). ``large``: a circle may be larger than the star (its
    even columns then by ``oblate._even_arc_gl``)."""
    g = [tuple(_sg(v) for v in c) for c in circles]
    ncol = 3 + len(eps)
    occ = [0.0 * circles[0][0] for _ in range(ncol)]
    limb_th, limb_ok = [], []
    hs = [(n, True) for n in range(3)] + [(1.0 + e, False) for e in eps]
    zero = 0.0 * g[0][0]
    star = (zero, zero, zero + 1.0)
    limb = [_meet(g[k], star, dt) for k in range(2)]
    corners = _meet(g[0], g[1], dt)
    for k in range(2):
        gx0, gy0, gR = g[k]
        ox, oy, oR = g[1 - k]
        x0, y0, R = circles[k]
        pts, has = limb[k]
        lp = [_angle_on(g[k], pt) for pt in pts]
        lv = [has, has]
        for pt in pts:
            limb_th.append(_angle_on(star, pt))
            limb_ok.append(has)
        cp = [_angle_on(g[k], pt) for pt in corners[0]]
        cv = [corners[1], corners[1]]
        los, his, oks = _arcs(lp + cp, lv + cv)
        LO, HI = _sg(mx.stack(los, -1)), _sg(mx.stack(his, -1))
        OK = mx.stack(oks, -1)
        b = lambda v: v[..., None]
        m_star, m_other = [], []
        for t in _SAMPLES:
            sm_, cm_ = sincos(LO + t * (HI - LO))
            xs, ys = b(gx0) + b(gR) * cm_, b(gy0) + b(gR) * sm_
            m_star.append(1.0 - (xs * xs + ys * ys))
            ux, uy = (xs - b(ox)) / b(oR), (ys - b(oy)) / b(oR)
            m_other.append(1.0 - (ux * ux + uy * uy))
        inside = (_decisive(m_star) > 0.0) & (_decisive(m_other) > 0.0)
        kappa = mx.where(OK & inside, 1.0, 0.0).astype(dt)
        # the arcs: live coefficients, frozen endpoints
        shp = LO.shape
        bx = lambda v: mx.broadcast_to(v[..., None], shp)
        X0, Y0, A = bx(x0), bx(y0), bx(R)
        S0, S1, S2, K0, K1 = _coeffs(X0, Y0, A, A)
        cols = list(_even_arc(_q_coeffs(S0, S1, S2, K0, K1, _N_MAX), LO, HI))
        if large:
            big = bx(gR > 1.0)
            gl = _even_arc_gl(X0, Y0, A, A, LO, HI, _N_MAX)
            cols = [mx.where(big, g_, c_) for g_, c_ in zip(gl, cols)]
        cen = _centred(gx0, gy0, gR, dt)
        cenb = bx(cen)
        pS = _coeffs(mx.where(cenb, _sg(0.5 * A), X0), mx.where(cenb, 0.0, Y0), A, A)
        sl, cl = sincos(LO)
        sh, ch = sincos(HI)
        for e in eps:
            p = 1.0 + e
            zc = _circle_roots(mx.where(cen, 0.5 * gR, gx0), mx.where(cen, 0.0, gy0), gR, p, dt)
            z0 = [_C(mx.broadcast_to(_sg(z.re)[..., None], shp),
                     mx.broadcast_to(_sg(z.im)[..., None], shp)) for z in zc]
            v = _pole_arc(*pS, p, z0, LO, HI)
            # a circle centred on the star, first order in its offset
            R2 = A * A
            c0 = R2 / (2.0 * p * (p - R2))
            c1 = A * (p + R2) / (2.0 * p * (p - R2) ** 2)
            vc = c0 * (HI - LO) + c1 * (X0 * (sh - sl) + Y0 * (cl - ch))
            cols.append(mx.where(cenb, vc, v))
        # gradient-only corner terms at both ends of every arc
        gX0, gY0, gA = _sg(X0), _sg(Y0), _sg(A)

        def C(sn, cs):
            xp, yp = gX0 + gA * cs, gY0 + gA * sn
            xl, yl = X0 + A * cs, Y0 + A * sn
            return xp * yl - yp * xl, xp * xp + yp * yp
        Cl, s_lo = C(sl, cl)
        Ch, s_hi = C(sh, ch)
        for idx, (arg, even) in enumerate(hs):
            corner = _h(arg, s_lo, even) * Cl - _h(arg, s_hi, even) * Ch
            corner = corner - _sg(corner)
            occ[idx] = occ[idx] + mx.sum(kappa * mx.where(OK, cols[idx] + corner, 0.0), axis=-1)
    # the limb's arcs inside both circles
    los, his, oks = _arcs(limb_th, limb_ok)
    h1 = [1.0 / (2.0 * (n + 1.0)) for n in range(3)] + [1.0 / (2.0 * (1.0 + e) * e) for e in eps]
    for lo, hi, ok in zip(los, his, oks):
        ms = [[], []]
        for t in _SAMPLES:
            sm, cm = sincos(lo + t * (hi - lo))
            for j, (gx0, gy0, gR) in enumerate(g):
                u, v = (cm - gx0) / gR, (sm - gy0) / gR
                ms[j].append(1.0 - (u * u + v * v))
        w = mx.where(ok & (_decisive(ms[0]) > 0.0) & (_decisive(ms[1]) > 0.0), 1.0, 0.0).astype(dt)
        dth = _sg(hi - lo)
        for idx in range(ncol):
            occ[idx] = occ[idx] + w * h1[idx] * dth
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
    lens = _lens_cols([(xp, yp, rp), (xm, ym, rs)], law.eps, dt, large)
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
    rv = mx.where(limb, var / (mean * mean), 1.0)           # E[x^2]/E[x]^2 - 1 > 1/3
    k = mx.sqrt(1.0 + 1.0 / rv) - 2.0
    bk = mean * (k + 2.0) / (k + 1.0)
    pw = mx.sqrt(bk) * (k + 1.0) / (k + 1.5)
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
    m0, m2, m4 = _lens_cols([(xp, yp, rp), (xm, ym, rs)], (), dt, _large(rp, None))
    occ = (1.0 - u1 - u2) * m0 + (u1 + 2.0 * u2) * _lens_mu(m0, m2, m4, dt) - u2 * m2
    both = _both_front(front_p, front_m)
    if both is not None:
        occ = mx.where(both, occ, 0.0)
    return dev + occ / (math.pi * (1.0 - u1 / 3.0 - u2 / 6.0))


def _cast(v, dt):
    if isinstance(v, mx.array):
        return v if v.dtype == dt else v.astype(dt)
    return mx.array(float(v), dtype=dt)
