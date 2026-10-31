"""Oblate planets on an orbit: sky positions, contacts, exposures.

``metal.flux_dev_from_tau(..., f=, theta=)`` delegates here. The light
curve is ``oblate.shape_cols_oblate`` at the planet's sky position, put in
its principal frame, and integrated over each exposure as the spherical
path does; this module adds only what the orbit and the exposure need.

**Sky frame.** X along the orbit (the direction of motion at transit), Y
across, both in stellar radii (the convention is pinned by a test against
an independent numpy orbit):

    circular:   X = a sin phi,   Y = -b cos phi,   in front iff cos phi > 0
    eccentric:  X = -u,          Y = -v cos i,     in front iff v > 0

with (u, v) the transit-anchored orbit's sky-plane Cartesians
(``anchored.py``; at e = 0 they reduce to the circular ones). theta is the
sky angle of the planet's long axis from X, and
``oblate.principal_frame(X, Y, theta)`` gives the centre in the planet's
own frame. The far side is replaced by a fixed point well clear of the
star, so it is exactly 0 with zero gradient, as the spherical path's
pushed separation is.

**Contacts, per side.** A tilted ellipse's ingress and egress differ, and a
grazing one can have both outer contacts on one side of conjunction, so
nothing here is mirrored. Along the transit chord (a straight line to
within (duration / period)^2) two functions of phase are convex:

    q(phi) = (distance from the star's centre to the planet's disc)^2
    M(phi) = (largest distance from it to the planet's boundary)^2

The outer contacts are q = 1 and the inner ones M = 1. Newton's method on
a convex function, started on the outside of a root, approaches it
monotonically and never overshoots; so the outer contacts start from the
circumscribed (radius A) circle's contacts, where q >= 1, and the inner
ones from the outer contacts, where M >= 1. The derivatives come from the
envelope theorem: dq/dphi = 2 P* . dc/dphi at the extremal boundary point
P*. Contacts are quadrature split points only -- interior points of a
continuous integrand, detached as the spherical ones are -- so a solve
that has not converged (a near-tangent graze, whose dip is tiny) costs
quadrature accuracy there, never a wrong model. A contact that does not
exist collapses onto a point inside the transit.

**A tidally locked planet** (``tidal_q=``). A triaxial ellipsoid whose
long axis r1 points at the star at every instant, on any orbit, with one
shape parameter q: r1 = r2 (1 + 3 q), r3 = r2 (1 - q) (Correia 2014), and
the radius argument the volumetric (r1 r2 r3)^(1/3). This is ellc's "love"
model (Maxted 2016), term for term:

* q is the shape at the separation a. The volume is fixed and the bulge
  follows the separation r, so q(r) solves

      q (1 + 3 q)(1 - q) = Q (a / r)^3,   Q = h_f R_V^3 / (2 mu a^3)

  with mu = Mp / M*: ellc's q = h_f r2^3 / (2 mu r^3), with r2 from the
  volume. Q is the q of h_f's first-order relation, and only Q reaches the
  shape, so the mass ratio is needed for h_f alone (``tidal_q_from_hf``).
* With ``mass_ratio=``, the ellipsoid's centre moves towards the star by
  ellc's D = r2^5 / (mu a^4) (Chandrasekhar 1933's third-order tide, with
  Delta_3 = 1 and no dependence on the separation), applied to the planet's
  barycentric position, so that the sky position is scaled by
  1 - D / ((1 + mu) r). On a circular orbit that is the same model with
  a -> a - D / (1 + mu) at the same inclination (``_tidal_offset``).

Any ellipsoid projects to an ellipse, so each instant is the oblate problem
with its own (r, f, theta), the outline's shape matrix being

    S = r1^2 e1 e1^T + r2^2 e2' e2'^T + r3^2 e3' e3'^T

with, in the sky frame above and nu the angle along the orbit from
conjunction (phi on a circular orbit; s, c = sin, cos nu; ci, si = cos,
sin i),

    e1 = (-s, ci c), e2 = (c, ci s), e3 = (0, si),
    e2' = e2 cos tw + e3 sin tw, e3' = -e2 sin tw + e3 cos tw.

So nu carries the nose to the star, and the twist tw (``theta=``, 0 in
ellc) turns the other two axes about the nose, in a frame that goes round
with the orbit: r2 along the orbit and r3 along its normal at tw = 0; at
conjunction on an edge-on orbit, tw is the outline's sky angle, as the
oblate theta. Written S / r2^2 = 1 + q T, the factor q comes out of the
whole anisotropy, so f and theta stay well defined through the sphere: the
outline's angle jumps by 90 degrees there as f passes through 0, which the
ellipse does not notice (``tidal_outline``). The contacts solve the same
functions with the shape at the current phase; their Newton slope leaves
out the outline's own motion, so a step contracts at about r q / a rather
than quadratically, and the root is still the true contact.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .dtypes import fp64_on_cpu
from .hybrid import HybridLaw, combine_cols, get_law
from .oblate import _check_domain, principal_frame, shape_cols_oblate
from .trig import sincos

__all__ = ["sky_circular", "sky_anchored", "tidal_outline",
           "tidal_q_from_hf", "tidal_hf_from_q",
           "contact_offsets_oblate", "contact_offsets_tidal",
           "flux_dev_from_tau_oblate", "eclipse_dev_from_tau_oblate"]

_TWO_PI = 2.0 * math.pi
_N_PSI = 16              # boundary samples per extremum search
_N_PSI_NEWTON = 3        # Newton steps polishing the sampled extremum
_N_CONTACT = {mx.float64: 14, mx.float32: 10}     # Newton steps in phase
_FAR = 4.0               # far-side stand-in, |(X, Y)| > 1 + A for A < 1
_T2_MIN = 1e-30          # the floor on a locked outline's squared anisotropy
_TIDAL_G_RANGE = (-0.09, 0.6)   # q (1 + 3 q)(1 - q) on its monotonic branch


# --------------------------------------------------------------------------
# sky positions
# --------------------------------------------------------------------------

def sky_circular(phi, a, b):
    """(X, Y, dX/dphi, dY/dphi, front) on a circular orbit."""
    s, c = sincos(phi)
    return a * s, -(b * c), a * c, b * s, c > 0.0


def sky_anchored(phi, a, ci, consts):
    """(X, Y, dX/dphi, dY/dphi, front) on the transit-anchored eccentric
    orbit; ``consts`` = ``anchored.anchor_constants(k, h)``. The lines are
    ``anchored.separation_anchored``'s, kept separate so that its compiled
    graphs are untouched."""
    from .anchored import _one_minus_cos, solve_sincos_delta
    e, ecw, esw, es, ec, B1, A2, B2 = consts
    sd, cd = solve_sincos_delta(phi, es, ec)
    omc = _one_minus_cos(sd, cd)
    u = a * (-ecw * omc - B1 * sd)
    v = a * (A2 * cd - B2 * sd - esw)
    D = 1.0 + es * sd - ec * cd            # d delta / d phi = 1 / D
    du = a * (-ecw * sd - B1 * cd) / D
    dv = a * (-A2 * sd - B2 * cd) / D
    return -u, -(v * ci), -du, -(dv * ci), v > 0.0


def _sky(phi, a, b, ecc):
    """``ecc`` is None (circular) or (ci, anchor constants)."""
    if ecc is None:
        return sky_circular(phi, a, b)
    ci, consts = ecc
    return sky_anchored(phi, a, ci, consts)


# --------------------------------------------------------------------------
# a tidally locked planet's outline
# --------------------------------------------------------------------------

def tidal_outline(s, c, ci, rv, q, tw):
    """(r, f, theta) of a locked planet's outline, for
    ``shape_cols_oblate`` and ``principal_frame``: the area-equivalent
    radius, the flattening and the long axis's sky angle [rad]. (s, c): sin,
    cos of the angle along the orbit from conjunction; ci = cos i; rv the
    volumetric radius; q in (-1/3, 1); tw the twist about the nose [rad].
    MLX arrays. Differentiable in all of them through q = 0, where f = 0:
    an f and theta taken from S itself are not."""
    s2, c2, ci2 = s * s, c * c, ci * ci
    si = mx.sqrt(mx.maximum(1.0 - ci2, 1e-12))
    stw, ctw = sincos(tw)
    sig, chi, kap = stw * stw, ctw * ctw, stw * ctw
    # (1 + 3 q)^2 = 1 + q al, (1 - q)^2 = 1 - q be
    al, be = 6.0 + 9.0 * q, 2.0 - q
    Txx = al * s2 - be * sig * c2
    Txy = be * kap * si * c - (al + be * sig) * ci * s * c
    Tyy = (al * ci2 * c2 - be * sig * ci2 * s2 - be * chi * si * si
           + 2.0 * be * kap * ci * si * s)
    m, t1, t2 = 0.5 * (Txx + Tyy), 0.5 * (Txx - Tyy), Txy
    # q has a sign and f has not: at q < 0 the long axis is 90 degrees away
    sgn = mx.where(q >= 0.0, 1.0, -1.0)
    T2 = t1 * t1 + t2 * t2
    round_ = T2 < _T2_MIN                  # a circular outline at any q
    T = mx.sqrt(mx.where(round_, _T2_MIN, T2))
    D = sgn * q * T
    hs = 1.0 + q * m
    Ah, Bh = mx.sqrt(hs + D), mx.sqrt(hs - D)
    f = 2.0 * D / (Ah * (Ah + Bh))
    r = _tidal_r2(rv, q) * mx.sqrt(Ah * Bh)
    th = 0.5 * mx.arctan2(mx.where(round_, 0.0, sgn * t2),
                          mx.where(round_, 1.0, sgn * t1))
    return r, f, th


def _tidal_r2(rv, q):
    """r2, the axis along the orbit, from the volumetric radius."""
    return rv * mx.power((1.0 + 3.0 * q) * (1.0 - q), -1.0 / 3.0)


def _tidal_g(q):
    """q (1 + 3 q)(1 - q): q's volume-corrected form, the quantity that goes
    as the inverse cube of the separation."""
    return q * (1.0 + 3.0 * q) * (1.0 - q)


def _tidal_q_solve(g):
    """q with ``_tidal_g(q)`` = g, on the branch through 0, by Newton under
    stop_gradient and one live step, which gives the exact d/dg."""
    g0 = mx.stop_gradient(g)
    q = g0 / (1.0 + 2.0 * g0)
    for _ in range(5):
        q = q - (_tidal_g(q) - g0) / (1.0 + q * (4.0 - 9.0 * q))
    q = mx.stop_gradient(q)
    return q - (_tidal_g(q) - g) / (1.0 + q * (4.0 - 9.0 * q))


def tidal_q_from_hf(hf, mass_ratio, a, rv):
    """``tidal_q`` from the fluid Love number, by ellc's relation: q =
    h_f r2^3 / (2 mu a^3) with r2 the axis along the orbit, whose form in
    the volumetric radius is the cubic q (1 + 3 q)(1 - q) = h_f R_V^3 /
    (2 mu a^3). mass_ratio = mu = Mp / M*; a and rv in the same unit.
    MLX arrays in, MLX out, differentiable; numbers in, numbers out."""
    Q = 0.5 * hf / mass_ratio * (rv / a) ** 3
    if isinstance(Q, mx.array):
        return _tidal_q_solve(Q)
    Q = np.asarray(Q, dtype=np.float64)
    q = Q / (1.0 + 2.0 * Q)
    for _ in range(40):
        q = q - (_tidal_g(q) - Q) / (1.0 + q * (4.0 - 9.0 * q))
    return float(q) if q.ndim == 0 else q


def tidal_hf_from_q(q, mass_ratio, a, rv):
    """The inverse of ``tidal_q_from_hf``: 2 mu (a / r2)^3 q."""
    return 2.0 * mass_ratio * (a / rv) ** 3 * _tidal_g(q)


def _tidal_cm(mass_ratio):
    """1 / (mu (1 + mu)), the offset's coefficient, 0 without a mass ratio."""
    pos = mass_ratio > 0.0
    mu = mx.where(pos, mass_ratio, 1.0)
    return mx.where(pos, 1.0 / (mu * (1.0 + mu)), 0.0)


def _tidal_offset(a, b, rv, q, cm):
    """(a, b) with the offset applied: on a circular orbit it is a constant
    scale of the sky position, so the same model at a smaller a and the
    same inclination."""
    r2 = _tidal_r2(rv, q)
    a1 = a - cm * (r2 / a) ** 4 * r2
    return a1, b * a1 / a


def _tidal_instant(phi, a, b, ecc, rv, q, tw, cm):
    """(X, Y, dX/dphi, dY/dphi, front, r, f, theta): ``_sky`` and the
    outline. On a circular orbit (a, b) already hold the offset. On an
    eccentric one the shape and the offset follow the separation, nu comes
    from the position at cos i = 1, where (X, Y) lies in the orbit's plane;
    the derivatives, for the contacts' slope, leave out the small offset."""
    if ecc is None:
        X, Y, dX, dY, front = sky_circular(phi, a, b)
        s, c = sincos(phi)
        return (X, Y, dX, dY, front, *tidal_outline(s, c, b / a, rv, q, tw))
    ci, consts = ecc
    X, Y, dX, dY, front = sky_anchored(phi, a, 1.0, consts)
    rr = mx.sqrt(X * X + Y * Y)
    qn = _tidal_q_solve(_tidal_g(q) * (a / rr) ** 3)
    r2 = _tidal_r2(rv, qn)
    sc = 1.0 - cm * (r2 / a) ** 4 * r2 / rr
    return (X * sc, Y * ci * sc, dX, dY * ci, front,
            *tidal_outline(X / rr, -Y / rr, ci, rv, qn, tw))


def _tidal_bound(rv, q, e=None):
    """A circle containing every outline at any phase, the offset included
    (it is small next to the 1% added): the ellipsoid's longest axis, at
    periastron on an eccentric orbit."""
    if e is not None:
        q = _tidal_q_solve(_tidal_g(q) / (1.0 - e) ** 3)
    return 1.01 * _tidal_r2(rv, q) * mx.maximum(1.0 + 3.0 * q, 1.0 - q)


# --------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------

def _boundary_ext(x0, y0, A, B, kind):
    """(s*, psi*) with s(psi) = (x0 + A cos psi)^2 + (y0 + B sin psi)^2 at
    its minimum (kind = -1) or maximum (+1) over the boundary: the best of
    _N_PSI samples, polished by Newton on ds/dpsi."""
    dt = x0.dtype
    grid = mx.array(np.linspace(0.0, _TWO_PI, _N_PSI, endpoint=False), dtype=dt)
    xs, ys = x0[..., None], y0[..., None]
    sg, cg = sincos(grid)
    sv = (xs + A[..., None] * cg) ** 2 + (ys + B[..., None] * sg) ** 2
    k = mx.argmax(sv, axis=-1) if kind > 0 else mx.argmin(sv, axis=-1)
    psi = mx.take(grid, k)
    for _ in range(_N_PSI_NEWTON):
        sp, cp = sincos(psi)
        xx, yy = x0 + A * cp, y0 + B * sp
        d1 = 2.0 * (-xx * A * sp + yy * B * cp)
        d2 = 2.0 * (A * A * sp * sp - xx * A * cp + B * B * cp * cp - yy * B * sp)
        # only a step toward the extremum sought: d2 must have its sign
        ok = (d2 * kind) < 0.0
        psi = psi - mx.where(ok, d1 / mx.where(ok, d2, 1.0), 0.0)
    sp, cp = sincos(psi)
    return (x0 + A * cp) ** 2 + (y0 + B * sp) ** 2, psi


def _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind, tidal=None):
    """q (kind = -1: squared distance to the disc, 0 inside it) or M
    (kind = +1: squared largest boundary distance) at phase phi, and its
    phase derivative; ``tidal`` = (rv, q, tw, cm) replaces the fixed
    outline by the locked planet's at phi, the slope leaving out its
    motion."""
    if tidal is None:
        X, Y, dX, dY, _ = _sky(phi, a, b, ecc)
    else:
        X, Y, dX, dY, _, r, f, th = _tidal_instant(phi, a, b, ecc, *tidal)
        A = r / mx.sqrt(1.0 - f)
        B = A * (1.0 - f)
        sth, cth = sincos(th)
    x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
    dx0, dy0 = dX * cth + dY * sth, -dX * sth + dY * cth
    s, psi = _boundary_ext(x0, y0, A, B, kind)
    sp, cp = sincos(psi)
    slope = 2.0 * ((x0 + A * cp) * dx0 + (y0 + B * sp) * dy0)
    if kind < 0:
        inside = (x0 / A) ** 2 + (y0 / B) ** 2 < 1.0
        s = mx.where(inside, 0.0, s)
        slope = mx.where(inside, 0.0, slope)
    return s, slope


def _newton_contact(phi, a, b, ecc, A, B, cth, sth, kind, side, lim, n,
                    tidal=None):
    """Newton on g(phi) - 1 = 0 from phi, on the outside of the root
    (side -1: ingress, the root lies at larger phi; +1: egress)."""
    for _ in range(n):
        g, dg = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind, tidal)
        # outside the root the slope points away from it: dg * side > 0
        ok = (dg * side) > 0.0
        step = mx.where(ok, (g - 1.0) / mx.where(ok, dg, 1.0), 0.0)
        phi = phi - mx.clip(step, -lim, lim)
    g, _ = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind, tidal)
    return phi, g


def contact_offsets_oblate(a, b, r, f, theta, k=None, h=None, ci=None,
                           tidal=False, mass_ratio=None):
    """The four contact phases (ingress outer, ingress inner, egress inner,
    egress outer) of an oblate planet, as offsets from mid-transit in
    orbital phase (phi = 2 pi (t - t0) / P); ordered, detached.

    a: a/R*; b: impact parameter at conjunction; r: area-equivalent radius;
    f: flattening; theta: the long axis's sky angle [rad]; (k, h) =
    (secosw, sesinw) for the transit-anchored eccentric orbit, with ci =
    cos i (derived from (a, b, k, h) if omitted). Each argument a Python
    number, a numpy array or an mx.array, broadcast together; the result
    has their broadcast shape. Computed in float64 (on the CPU stream)
    unless every array argument is float32. A missing contact collapses
    onto a point inside the transit (the inner pair of a graze), or all
    four onto one point (no transit). ``tidal``: (r, f, theta) hold
    (rv, q, tw) (``contact_offsets_tidal``), with its ``mass_ratio``.
    """
    if (k is None) != (h is None):
        raise ValueError("pass both k and h (secosw, sesinw), or neither")
    vals = [a, b, r, f, theta] + ([k, h] if k is not None else []) \
        + ([ci] if ci is not None else []) \
        + ([mass_ratio] if mass_ratio is not None else [])
    arrays = [v for v in vals if isinstance(v, mx.array)]
    dt = (mx.float32 if arrays and all(v.dtype == mx.float32 for v in arrays)
          else mx.float64)
    shape = np.broadcast_shapes(*(tuple(v.shape) if isinstance(v, mx.array)
                                  else np.shape(v) for v in vals))
    n = int(np.prod(shape)) if shape else 1

    def norm(v):
        with mx.stream(mx.cpu):
            x = (v.astype(dt) if isinstance(v, mx.array)
                 else mx.array(np.asarray(v, dtype=np.float64), dtype=dt))
            return mx.reshape(mx.broadcast_to(x, shape), (n,))

    ctx = mx.stream(mx.cpu) if dt == mx.float64 else _nullctx()
    with ctx:
        a, b, r, f, theta = (norm(v) for v in (a, b, r, f, theta))
        if k is not None:
            k, h = norm(k), norm(h)
            ci = M._ecc_shape(a, b, k, h)[2] if ci is None else norm(ci)
        cm = None
        if tidal:
            cm = _tidal_cm(norm(0.0 if mass_ratio is None else mass_ratio))
            if k is None:
                a, b = _tidal_offset(a, b, r, f, cm)
        out = _contact_offsets_oblate(a, b, r, f, theta, k, h, ci, tidal=tidal,
                                      cm=cm)
        return tuple(mx.reshape(c, shape) for c in out)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def contact_offsets_tidal(a, b, rv, q, theta=0.0, k=None, h=None, ci=None,
                          mass_ratio=None):
    """``contact_offsets_oblate`` for a tidally locked planet of volumetric
    radius rv, shape q and twist theta about its nose [rad]: its four
    contact phases, ordered and detached. Inputs as there; ``mass_ratio``
    as ``flux_dev_from_tau``'s."""
    return contact_offsets_oblate(a, b, rv, q, theta, k, h, ci, tidal=True,
                                  mass_ratio=mass_ratio)


def _contact_offsets_oblate(a, b, r, f, theta, k=None, h=None, ci=None,
                            consts=None, tidal=False, cm=None):
    """contact_offsets_oblate's work, on canonical inputs: (n,) mx arrays
    of one dtype on the active stream (the graph, oblate kernels and tests
    call it directly). ``consts``, if given, replaces anchor_constants(k, h)
    -- the (e, w) route. ``tidal``: (r, f, theta) hold (rv, q, tw), with the
    offset's coefficient cm, and (a, b) hold the circular offset already;
    the starts come from ``_tidal_bound``."""
    from .exposure import contact_offsets, contact_offsets_anchored
    a, b, r, f, theta = (mx.stop_gradient(v) for v in (a, b, r, f, theta))
    dt = a.dtype
    if tidal:
        A = _tidal_bound(r, f, None if k is None else
                         mx.stop_gradient(k * k + h * h))
    else:
        A = r / mx.sqrt(1.0 - f)
    B = A * (1.0 - f)
    if k is None:
        ecc = None
        cA = contact_offsets(A, a, b)
    else:
        from .anchored import anchor_constants
        k, h, ci = (mx.stop_gradient(v) for v in (k, h, ci))
        consts = tuple(mx.stop_gradient(c) for c in
                       (anchor_constants(k, h) if consts is None else consts))
        ecc = (ci, consts)
        cA = contact_offsets_anchored(A, a, b, k, h, ci, consts=consts)
    tid = (r, f, theta, mx.stop_gradient(cm)) if tidal else None
    sth, cth = sincos(theta)
    lo, hi = cA[0], cA[3]
    lim = mx.maximum(hi - lo, 1e-12)
    n = _N_CONTACT[dt]
    tol = 1e-8 if dt == mx.float64 else 1e-4

    # Each pair's ingress and egress solves are independent: run them as
    # one stacked (2m,) chain, ingress rows first, which halves the graph.
    # Every op is elementwise or per row, so the bits are the separate
    # solves'.
    m = max(v.shape[0] if v.ndim else 1 for v in
            (a, b, A, B, cth, sth, lo, hi, *(() if ecc is None
                                              else (ecc[0], *ecc[1]))))

    def t2(*vs):
        return mx.concatenate([mx.broadcast_to(v, (m,) + tuple(v.shape[1:]))
                               for v in vs], axis=0)

    a2, b2, A2, B2, cth2, sth2 = (t2(v, v) for v in (a, b, A, B, cth, sth))
    tid2 = None if tid is None else tuple(t2(v, v) for v in tid)
    ecc2 = None if ecc is None else (t2(ecc[0], ecc[0]),
                                     tuple(t2(c, c) for c in ecc[1]))
    side = t2(mx.array(-1.0, dtype=dt), mx.array(1.0, dtype=dt))

    def pair(start, end, kind, lim):
        c, g = _newton_contact(t2(start, end), a2, b2, ecc2, A2, B2, cth2,
                               sth2, kind, side, t2(lim, lim), n, tid2)
        return c[:m], c[m:], g[:m], g[m:]

    c1, c4, g1, g4 = pair(lo, hi, -1, lim)
    outer = (mx.abs(g1 - 1.0) < tol) & (mx.abs(g4 - 1.0) < tol) & (c1 < c4)
    mid = 0.5 * (lo + hi)
    c1, c4 = mx.where(outer, c1, mid), mx.where(outer, c4, mid)

    lim_i = mx.maximum(c4 - c1, 1e-12)
    c2, c3, g2, g3 = pair(c1, c4, 1, lim_i)
    inner = (outer & (mx.abs(g2 - 1.0) < tol) & (mx.abs(g3 - 1.0) < tol)
             & (c1 <= c2) & (c2 < c3) & (c3 <= c4))
    mid = 0.5 * (c1 + c4)
    c2, c3 = mx.where(inner, c2, mid), mx.where(inner, c3, mid)
    return tuple(mx.stop_gradient(c) for c in (c1, c2, c3, c4))


def _check_tidal_domain(rv, q):
    """q in (-1/3, 1) and the ellipsoid's long axis squared within its short
    axis, which is the oblate domain A^2 / B <= 1 at every phase, for host
    values, raising; array arguments are checked in the graph
    (_in_tidal_domain: NaN outside), as _check_domain's."""
    if isinstance(rv, mx.array) or isinstance(q, mx.array):
        return
    rv = np.asarray(rv, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if np.any(q <= -1.0 / 3.0) or np.any(q >= 1.0):
        raise ValueError(f"tidal_q must be in (-1/3, 1); got {q}")
    u, v = 1.0 + 3.0 * q, 1.0 - q
    r2 = rv * (u * v) ** (-1.0 / 3.0)
    if np.any(r2 * np.maximum(u, v) ** 2 > np.minimum(u, v) * (1.0 + 1e-12)):
        raise ValueError(
            "the ellipsoid's long axis squared exceeds its short axis: the "
            "planet could cross the stellar limb more than twice, outside "
            "the oblate path's domain")


def _in_tidal_domain(rv, q, e=None):
    """The domain as a mask, for array arguments (see _check_tidal_domain);
    with an eccentricity, at periastron, and with the separation's cubic on
    its monotonic branch."""
    qs = mx.clip(q, -0.3, 0.99)            # finite outside; ok masks those
    ok = (q > -1.0 / 3.0) & (q < 1.0)
    if e is not None:
        lo, hi = _TIDAL_G_RANGE
        gp = _tidal_g(qs) / (1.0 - mx.minimum(e, 0.999)) ** 3
        ok = ok & (gp > lo) & (gp < hi)
        qs = _tidal_q_solve(mx.clip(gp, lo, hi))
    u, v = 1.0 + 3.0 * qs, 1.0 - qs
    return ok & (_tidal_r2(rv, qs) * mx.maximum(u, v) ** 2
                 <= mx.minimum(u, v) * (1.0 + 1e-12))


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def _orbit_consts(a, b, k, h, ew=False):
    """(cos i, anchored constants) of the transit-anchored orbit, from b
    at conjunction and (k, h) = sqrt(e) (cos w, sin w) -- or, with ``ew``,
    from (k, h) holding (e, w [rad]) themselves (anchor_constants_ew):
    exact in d/de at e = 0, where the (k, h) map is singular (TransitModel's
    array-ecc route). cos i is metal._ecc_shape's, written in (e, e sin w)."""
    from .anchored import anchor_constants, anchor_constants_ew, cos_inc_from_b
    if not ew:
        return M._ecc_shape(a, b, k, h)[2], anchor_constants(k, h)
    e, w = k, h
    esw = e * sincos(w)[0]
    ci = cos_inc_from_b(a, b, e, esw, M._E_MAX)
    return ci, anchor_constants_ew(e, w)


def _contact_kh(k, h, ew):
    """(k, h) for the detached contact starts; on the (e, w) route,
    sqrt(e) (cos w, sin w) -- the contacts carry no gradient."""
    if not ew:
        return k, h
    s, c = sincos(h)
    sq = mx.sqrt(mx.maximum(k, 0.0))
    return mx.stop_gradient(sq * c), mx.stop_gradient(sq * s)


def _tau_graph(tau, period, a, b, r, f, theta, w2d, exp_time, mode, n_gl,
               n_sub, law, basis=False, k=None, h=None, ew=False,
               contacts=None, tidal=False, cm=None):
    """The oblate counterpart of ``metal_hybrid._tau_graph``: the same
    exposure rules, contacts detached and the period detached in the node
    map exactly as there. ``contacts``: the contact rule's (n, 4)
    (_graph_contacts), computed here if not given. ``tidal``: (r, f, theta)
    hold (rv, q, tw), each node having its own outline; cm is the offset's
    coefficient, read on an eccentric orbit ((a, b) hold a circular one's)."""
    from .exposure import exposure_nodes
    law = get_law(law)
    ecc = k is not None
    pars = (period, a, b, r, f, theta) + ((k, h) if ecc else ())
    cm_col = None if cm is None else (lambda nk: mx.reshape(cm, (-1,) + (1,) * nk))

    def col(nk):
        return [mx.reshape(p, (-1,) + (1,) * nk) for p in pars]

    def orbit(cols):
        per, av, bv = cols[:3]
        if not ecc:
            return per, av, bv, None
        kv, hv = cols[-2:]
        return per, av, bv, _orbit_consts(av, bv, kv, hv, ew)

    def inst(tt, nk):
        cols = col(nk)
        per, av, bv, eo = orbit(cols)
        rv, fv, thv = cols[3:6]
        phi = (_TWO_PI / per) * tt
        if tidal:
            X, Y, _, _, front, rv, fv, thv = _tidal_instant(
                phi, av, bv, eo, rv, fv, thv, cm_col(nk))
        else:
            X, Y, _, _, front = _sky(phi, av, bv, eo)
        X = mx.where(front, X, _FAR)
        Y = mx.where(front, Y, 0.0)
        sth, cth = sincos(thv)
        x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
        return shape_cols_oblate(x0, y0, rv, fv, law)

    if mode == M._INT_NONE or exp_time == 0.0:
        Bc = inst(tau, 1)
    elif mode == M._INT_SUPER:
        half = 0.5 * exp_time
        off = (np.linspace(-half, half, int(n_sub)) if n_sub > 1
               else np.zeros(1))
        nodes = tau[..., None] + mx.array(off, dtype=tau.dtype)
        Bc = mx.mean(inst(nodes, 2), axis=-2)
    else:
        per1 = col(1)[0]
        if contacts is None:
            contacts = _graph_contacts(a, b, r, f, theta, k, h, ew, tidal, cm)
        cs = tuple(contacts[:, j:j + 1] for j in range(4))
        T, W = exposure_nodes(tau, tau * 0.0, mx.stop_gradient(per1),
                              exp_time, cs, int(n_gl), dtype=tau.dtype)
        Bc = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


def _graph_contacts(a, b, r, f, theta, k=None, h=None, ew=False,
                    tidal=False, cm=None):
    """The contact rule's (n, 4) contact phases for _tau_graph's canonical
    (n,) inputs, detached."""
    if k is None:
        cs = _contact_offsets_oblate(a, b, r, f, theta, tidal=tidal, cm=cm)
    else:
        ci, consts = _orbit_consts(a, b, k, h, ew)
        kc, hc = _contact_kh(k, h, ew)
        cs = _contact_offsets_oblate(a, b, r, f, theta, kc, hc, ci,
                                     consts=consts, tidal=tidal, cm=cm)
    return mx.stack([mx.reshape(c, (-1,)) for c in cs], axis=1)


# Evaluation points (sets x times x exposure nodes) per block of the fp64
# graph (_in_blocks): ~1-3 KB of intermediates each, so ~0.3-0.8 GB.
_OBLATE_BLOCK = 1 << 18


def _nodes_per_time(exp_time, mode, n_gl, n_sub):
    """Points the graph evaluates per time (``_tau_graph``'s exposure rule:
    the contact rule's five pieces of n_gl nodes)."""
    if mode == M._INT_NONE or exp_time == 0.0:
        return 1
    if mode == M._INT_SUPER:
        return max(int(n_sub), 1)
    return 5 * int(n_gl)


_block_fns: dict = {}


def _block_fn(exp_time, mode, n_gl, n_sub, law, basis, ew, present,
              compiled, tidal=False):
    """``run(tt, rows)`` for _in_blocks: _tau_graph on one block's rows
    (``present``: which of them are arrays). ``compiled``: under
    mx.compile, cached per configuration -- every block has one shape, so
    one trace serves them all, where compiling a whole blocked call traces
    every block (minutes at 30 blocks)."""
    def raw(tt, *arrs):
        it = iter(arrs)
        *p6, w, kk, hh, c, cm = [next(it) if p else None for p in present]
        return _tau_graph(tt, *p6, w, exp_time, mode, n_gl, n_sub, law,
                          basis=basis, k=kk, h=hh, ew=ew, contacts=c,
                          tidal=tidal, cm=cm)
    fn = raw
    if compiled:
        key = (float(exp_time), int(mode), int(n_gl), int(n_sub), law,
               bool(basis), bool(ew), present, bool(tidal))
        fn = _block_fns.get(key)
        if fn is None:
            fn = _block_fns[key] = mx.compile(raw)
    return lambda tt, rows: fn(tt, *[x for x in rows if x is not None])


def _pad(x, size, axis):
    """``x`` padded to ``size`` along ``axis`` (0 or 1) with its last slice."""
    k = size - x.shape[axis]
    if k == 0:
        return x
    last = x[-1:] if axis == 0 else x[:, -1:]
    shape = list(x.shape)
    shape[axis] = k
    return mx.concatenate([x, mx.broadcast_to(last, shape)], axis=axis)


def _in_blocks(run, tau2d, rows, nodes, block, per_time=None):
    """``run(tau2d, rows)`` -- the oblate or ringed graph, (n, m[, ncol]) --
    in blocks of at most ``block`` evaluation points, concatenated lazily.
    The graphs hold their fp64 intermediates until evaluated (~1-3 KB per
    point oblate, ~20-24 KB ringed: ~24 GB at 10^6 ringed points); MLX
    evaluates the blocks one after another and frees each block's
    intermediates before the next, so the peak is one block's, at no cost
    in time. Each flux depends only on its own row (``rows``: the per-set
    (n, ...) arrays, or None) and time. Blocks are whole rows
    while a row fits, else one row's times; equal-sized, padded with the
    last row or time and the padding dropped, so one compiled shape and one
    cached buffer set serve every block (a shorter last block would
    allocate, and cache, a second set). ``per_time``: (n, m, ...) arrays
    sliced and padded with tau2d, passed on as ``run(tt, rows, *per_time)``."""
    n, m = tau2d.shape
    per_row = nodes * m
    pts = list(per_time or ())

    def call(tt, rws, pt):
        return run(tt, rws, *pt) if per_time is not None else run(tt, rws)
    if n * per_row <= block:
        return call(tau2d, rows, pts)

    def equal(total, most):
        nb = -(-total // max(most, 1))
        return nb, -(-total // nb)

    def sl(x, i, j):
        return None if x is None else x[i:j]
    if per_row <= block:
        nb, size = equal(n, block // per_row)
        tp = _pad(tau2d, nb * size, 0)
        rp = [None if x is None else _pad(x, nb * size, 0) for x in rows]
        pp = [_pad(x, nb * size, 0) for x in pts]
        return mx.concatenate(
            [call(tp[i * size:(i + 1) * size],
                  [sl(x, i * size, (i + 1) * size) for x in rp],
                  [x[i * size:(i + 1) * size] for x in pp])
             for i in range(nb)], axis=0)[:n]
    nb, size = equal(m, block // nodes)
    tp = _pad(tau2d, nb * size, 1)
    pp = [_pad(x, nb * size, 1) for x in pts]
    return mx.concatenate(
        [mx.concatenate([call(tp[j:j + 1, i * size:(i + 1) * size],
                              [sl(x, j, j + 1) for x in rows],
                              [x[j:j + 1, i * size:(i + 1) * size] for x in pp])
                         for i in range(nb)], axis=1)
         for j in range(n)], axis=0)[:, :m]


@fp64_on_cpu
def flux_dev_from_tau_oblate(tau, period, a, b, r, f, theta, law, u,
                             exp_time, mode, n_gl, n_sub, basis=False,
                             k=None, h=None, ew=False, use_metal=True,
                             compile_blocks=False, tidal=False,
                             mass_ratio=None):
    """``metal.flux_dev_from_tau`` with ``f=`` (that function validates the
    keywords and delegates here). fp32 data on the GPU stream takes the
    oblate Metal kernels (``metal_oblate``) unless ``use_metal`` is False;
    anything else runs this graph, fp64 on the CPU stream, in blocks of at
    most _OBLATE_BLOCK points (_in_blocks) -- each block compiled once for
    all (they share one shape) with ``compile_blocks``, TransitModel's
    route for a call too large to compile whole. ``ew``: (k, h) hold (e, w
    [rad]) -- TransitModel's array-ecc route (_orbit_consts). ``tidal``:
    (r, f, theta) hold (rv, q, tw), a tidally locked planet, with its
    ``mass_ratio`` (None: no offset)."""
    from .metal_hybrid import _canon_w
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    (_check_tidal_domain if tidal else _check_domain)(r, f)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r, f, theta) + ((k, h) if ecc else ())
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    if isinstance(mass_ratio, mx.array) and mass_ratio.ndim >= 1:
        n_param = max(n_param, mass_ratio.shape[0])
    n = max(tau2d.shape[0], n_param)
    pc = [M._canon_param(p, n, tau2d.dtype) for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    kh = pc[-2:] if ecc else [None, None]
    cm = dom = None
    if tidal:
        cm = _tidal_cm(M._canon_param(0.0 if mass_ratio is None else mass_ratio,
                                      n, tau2d.dtype))
        dom = _in_tidal_domain(pc[3], pc[4], None if not ecc else
                               kh[0] if ew else kh[0] * kh[0] + kh[1] * kh[1])
        if not ecc:
            pc[1], pc[2] = _tidal_offset(pc[1], pc[2], pc[3], pc[4], cm)
    w2d = (None if basis else
           _canon_w(u, n, law.n_w, tau2d.dtype, law.name))
    if (use_metal and tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        from .metal_oblate import flux_dev_from_tau_oblate_kernel
        from .oblate import _in_domain
        out = flux_dev_from_tau_oblate_kernel(
            tau2d, *pc[:6], w2d, exp_time, mode, n_gl, n_sub, law,
            basis=basis, k=kh[0], h=kh[1], ew=ew, tidal=tidal, cm=cm)
        ok = (dom if tidal else _in_domain(pc[3], pc[4]))[:, None]
        out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    else:
        # the contact rule's contacts depend on the parameters alone: once
        # per set, not per block
        contact = mode not in (M._INT_NONE, M._INT_SUPER) and exp_time != 0.0
        cs = (_graph_contacts(*pc[1:6], kh[0], kh[1], ew, tidal, cm) if contact
              else None)

        rows = [*pc[:6], w2d, *kh, cs, cm]
        run = _block_fn(exp_time, mode, n_gl, n_sub, law, basis, ew,
                        tuple(x is not None for x in rows), compile_blocks,
                        tidal)
        out = _in_blocks(run, tau2d, rows,
                         _nodes_per_time(exp_time, mode, n_gl, n_sub),
                         _OBLATE_BLOCK)
        if tidal:
            ok = dom[:, None]
            out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    return out[0] if squeeze and n == 1 else out


# --------------------------------------------------------------------------
# secondary eclipses
# --------------------------------------------------------------------------

# no shapes and no poles: the shape basis is the uniform column E0 alone
_AREA = HybridLaw("area", (), ())


def _eclipse_graph(tau, period, a, b, r, f, theta, k=None, h=None, ew=False):
    """The visible fraction of a uniform oblate planet's disc, minus 1, on
    (n, m) tau and (n,) parameters. The planet is behind the star, so the
    near side of the orbit takes the far-side stand-in and the far side is
    live. E0 is minus the area of the planet's outline inside the star's
    disc, so the fraction is 1 + E0 / (pi r^2) (r is the area-equivalent
    radius), the spherical path's rule; r = 0 is a point planet, always
    visible. theta is the sky angle at transit: the outline's projection
    is the same all along the orbit."""
    per, av, bv, rv, fv, thv = (mx.reshape(p, (-1, 1))
                                for p in (period, a, b, r, f, theta))
    eo = None
    if k is not None:
        eo = _orbit_consts(av, bv, mx.reshape(k, (-1, 1)),
                           mx.reshape(h, (-1, 1)), ew)
    X, Y, _, _, front = _sky((_TWO_PI / per) * tau, av, bv, eo)
    X = mx.where(front, _FAR, X)
    Y = mx.where(front, 0.0, Y)
    sth, cth = sincos(thv)
    x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
    e0 = shape_cols_oblate(x0, y0, rv, fv, _AREA)[..., 0]
    pos = rv > 0.0
    return mx.where(pos, e0 / mx.where(pos, math.pi * rv * rv, 1.0), 0.0)


_eclipse_block_fns: dict = {}


def _eclipse_block_fn(ecc, ew, compiled):
    """``run(tt, rows)`` for _in_blocks, as _block_fn's."""
    def raw(tt, *rows):
        return _eclipse_graph(tt, *rows[:6], *(rows[6:] if ecc else ()),
                              ew=ew)
    fn = raw
    if compiled:
        key = (bool(ecc), bool(ew))
        fn = _eclipse_block_fns.get(key)
        if fn is None:
            fn = _eclipse_block_fns[key] = mx.compile(raw)
    return lambda tt, rows: fn(tt, *rows)


@fp64_on_cpu
def eclipse_dev_from_tau_oblate(tau, period, a, b, r, f, theta, k=None,
                                h=None, ew=False, compile_blocks=False):
    """A uniform oblate planet's secondary eclipse: the visible fraction of
    its disc minus 1 (0 out of eclipse, -1 in totality), so that the flux
    is 1 + fp (1 + this). ``tau`` is (n, m) or (m,), the time since each
    point's mid-*transit* within half a period of it, as
    flux_dev_from_tau_oblate's (the orbit places the eclipse: near
    +-period / 2 on a circular orbit); arguments as there. The graph only,
    with no kernel and no exposure rule (TransitModel supersamples on its
    own grid); fp64 on the CPU stream in blocks of at most _OBLATE_BLOCK
    points, each compiled once with ``compile_blocks``."""
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    _check_domain(r, f)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r, f, theta) + ((k, h) if ecc else ())
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    n = max(tau2d.shape[0], n_param)
    rows = [M._canon_param(p, n, tau2d.dtype) for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    out = _in_blocks(_eclipse_block_fn(ecc, ew, compile_blocks), tau2d, rows,
                     1, _OBLATE_BLOCK)
    return out[0] if squeeze and n == 1 else out
