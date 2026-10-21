"""Exposure-time averaging by contact-split Gauss-Legendre quadrature.

A finite exposure measures the flux averaged over [t - dt/2, t + dt/2].
The usual fix (batman's, and ``TransitModel(supersample_factor=N)``) is
to average N uniformly spaced samples. That converges only as O(1/N)
because the light curve is *not* smooth inside the window: its
derivative jumps at each contact, where the planet's limb crosses the
stellar limb.

ALFM19's recipe -- and Limbdark.jl's ``integrate_lightcurve`` -- is to
split the integration at the contact times and integrate each smooth
piece separately. There the pieces are handled by adaptive Simpson; an
adaptive depth is data-dependent control flow, which would break
batching and mx.compile, so here each piece gets a *fixed-order*
Gauss-Legendre rule instead. On a smooth piece an n-point rule is exact
for polynomials of degree 2n-1 and converges geometrically, so a handful
of nodes beats hundreds of uniform samples.

The split is done branchlessly: the window always has five
sub-intervals, with edges

    t1 <= clamp(tc1) <= clamp(tc2) <= clamp(tc3) <= clamp(tc4) <= t2

after clamping the four contact times into the window. Contacts outside
it collapse to a zero-width sub-interval that contributes nothing, so
grazing transits (two contacts), full transits (four) and windows that
straddle no contact at all are all the same code path -- and the node
positions stay differentiable in the parameters.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = ["gauss_legendre", "contact_offsets", "contact_geometry",
           "contact_offsets_anchored", "exposure_nodes"]


def gauss_legendre(n: int):
    """(nodes, weights) on [-1, 1], float64 host-side."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype(np.float64), w.astype(np.float64)


def contact_offsets(r, a_sky, b):
    """Contact times relative to transit centre, as phase offsets
    phi = 2 pi (t - t0) / P.

    For a circular orbit z^2 = a^2 sin^2(phi) + b^2 cos^2(phi) with
    b = a cos i, so z = Z is solved exactly by

        sin^2(phi) = (Z^2 - b^2) / (a^2 - b^2).

    For an eccentric orbit the same expression is used with ``a_sky``
    set to the sky-speed-equivalent semi-major axis
    a (1 + e sin w) / sqrt(1 - e^2) and ``b`` the true impact parameter
    at conjunction. That linearises the contact, which is all the split
    needs: an error dtau in a split point leaves a residual kink of
    order dtau^3 in the quadrature, against the O(dtau) error of not
    splitting at all.

    Returns (phi_1, phi_2, phi_3, phi_4), ordered. The inner pair sits at
    z = |1 - r|: the planet fully inside the disc for r < 1, the disc fully
    covered for r > 1 (total occultation). A grazing transit
    (b > |1 - r|) collapses the inner pair onto the transit centre, which
    the branchless five-interval split absorbs.
    """
    a2mb2 = mx.maximum(a_sky * a_sky - b * b, 1e-30)
    out = []
    for sign in (1.0, -1.0):          # Z = 1 + r (outer), |1 - r| (inner)
        Z = 1.0 + sign * r
        if sign < 0.0:
            Z = mx.abs(Z) if isinstance(Z, mx.array) else abs(Z)
        s2 = (Z * Z - b * b) / a2mb2
        # A collapsed contact (s2 <= 0: the inner pair of a grazing
        # transit) sits where d sqrt and d arcsin are infinite, and the
        # clip's zero cotangent then makes 0 * inf = NaN. So the active
        # branch sees a sanitised argument and the collapsed value is
        # detached -- the same values, bit for bit, with finite gradients.
        inside = mx.logical_and(s2 > 0.0, s2 < 1.0)
        s2_in = mx.where(inside, s2, mx.full(s2.shape, 0.5, dtype=s2.dtype))
        edge = mx.stop_gradient(mx.arcsin(mx.sqrt(mx.clip(s2, 0.0, 1.0))))
        out.append(mx.where(inside, mx.arcsin(mx.sqrt(s2_in)), edge))
    phi_out, phi_in = out
    return -phi_out, -phi_in, phi_in, phi_out


def contact_geometry(a, ecc, esw, ci, sqrt=None, maximum=None):
    """(a_sky, b_conj) for ``contact_offsets``, from the orbital elements.

    ``a_sky`` is the sky-velocity-equivalent semi-major axis
    a (1 + e sin w) / sqrt(1 - e^2) and ``b_conj`` the impact parameter at
    inferior conjunction, a (1 - e^2) / (1 + e sin w) cos i. At e = 0 both
    reduce to a and a cos i.

    Written once and called from all three contact sites (the eager
    frontend, the compiled frontend branch and the batched path), which
    previously carried three copies that already differed in how they
    floored ``esw`` and ``beta``. Pass ``e sin w`` rather than w so the
    caller can build it however it likes -- h * sqrt(e) in the anchored
    parameterisation, e * sin(w) from elements -- and pass the sqrt /
    maximum for the flavour of array in play (math, numpy or mlx).
    """
    if sqrt is None:
        sqrt, maximum = mx.sqrt, mx.maximum
    beta = sqrt(maximum(1.0 - ecc * ecc, 1e-30))
    one_p = 1.0 + esw
    return a * one_p / beta, a * (1.0 - ecc * ecc) / one_p * ci


def exposure_nodes(t, t0, period, exp_time, contacts, n_gl: int,
                   dtype=mx.float64):
    """Quadrature nodes and weights for each exposure window.

    ``t`` are exposure mid-times, ``contacts`` the four phase offsets
    from ``contact_offsets``. Returns (times, weights), both with one
    extra trailing axis of length 5 * n_gl, the weights already
    normalised so that summing weight * flux gives the average. A leading
    batch axis is allowed throughout (many parameter sets at once), in
    which case t and the scalars broadcast against each other.
    """
    xg, wg = gauss_legendre(n_gl)
    xg = mx.array(xg, dtype=dtype)
    wg = mx.array(wg, dtype=dtype)

    half = 0.5 * exp_time
    t1 = t - half
    t2 = t + half
    # transit centre of the epoch each exposure belongs to
    n_ep = mx.round((t - t0) / period)
    tc_mid = t0 + n_ep * period
    scale = period / (2.0 * math.pi)

    edges = [t1]
    for phi in contacts:
        tc = tc_mid + phi * scale
        edges.append(mx.minimum(mx.maximum(tc, t1), t2))
    edges.append(t2)
    # enforce monotonicity after clamping (contacts are already ordered,
    # but clamping two of them to the same bound must not invert them)
    for i in range(1, len(edges)):
        edges[i] = mx.maximum(edges[i], edges[i - 1])

    times, weights = [], []
    inv_dt = 1.0 / exp_time
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mid = 0.5 * (lo + hi)
        halfw = 0.5 * (hi - lo)
        times.append(mid[..., None] + halfw[..., None] * xg)
        weights.append(halfw[..., None] * wg * inv_dt)
    axis = times[0].ndim - 1
    W = mx.concatenate(weights, axis=axis)
    # The sub-interval widths sum to the exposure only up to rounding in
    # the clamped edges, which left the out-of-transit flux at 1 + 2e-14
    # instead of exactly 1. Renormalising restores that contract (the
    # weights are a partition of unity by construction, so this only
    # removes round-off).
    W = W / mx.sum(W, axis=axis, keepdims=True)
    return mx.concatenate(times, axis=axis), W


# secant steps to the closest approach, and outward doublings to bracket a
# contact: with n_iter = 5 Newton steps, float64 contacts are within 1e-12
# of the transit's duration of a bisected root on 2,600 random orbits
# (e <= 0.95, grazing and off-conjunction transits included); 4 steps is
# the minimum for that, 3 gives 2e-10
_NSEC, _NEXP = 3, 2


def contact_offsets_anchored(r, a, b, k, h, ci, n_iter: int = 5,
                             consts=None):
    """Exact contact phases of the transit-anchored eccentric orbit.

    ``contact_offsets`` with ``contact_geometry``'s sky-equivalent a is a
    linearisation: measured, it misplaces a contact by up to 2.3e-3 d on a
    grazing e = 0.5 orbit. A split that misses the kink costs the
    quadrature accuracy (20x there at n_gl = 5) and -- worse for a sampler
    -- breaks the frozen-split gradient, whose integrand dF/dtheta then
    jumps *inside* a Gauss-Legendre piece (~1% on d/dperiod at e = 0.7).

    On an eccentric orbit the closest approach is not at conjunction: the
    planet's distance changes across the transit, so z^2(phi) has its
    minimum at phi_min != 0, below b^2. A grazing transit can then sit
    almost entirely on one side of conjunction, or happen at all with
    b > 1 + r. So, in three steps, all on the anchored solve itself:

    1. phi_min from secant steps on d(z^2)/dphi, started at conjunction
       with the quadratic model z^2 ~ b^2 + a_sky^2 phi^2;
    2. a contact exists where z(phi_min) < Z, Z = 1 + r (outer pair) or
       |1 - r| (inner pair);
    3. each root of z^2 = Z^2 is bracketed between phi_min and a phase
       found by stepping outwards, and refined by ``n_iter`` Newton steps
       that fall back to bisection when a step would leave the bracket,
       with

           d(z^2)/dphi = 2 (u du/ddelta + ci^2 v dv/ddelta) / D,
           D = 1 + es sin delta - ec cos delta = 1 - e cos E > 0.

    The four roots run as one stacked (4, ...) solve. A contact with no
    root (the inner pair of a grazing transit, all four without a transit)
    collapses onto phi_min, the transit's centre. The phases carry the
    implicit-function gradient -dg/dtheta / (dg/dphi) at the root, not the
    iterations'. At e = 0 the linearisation is exact and is returned as
    is, so circular sets are bit-identical to ``contact_offsets``.
    (k, h) = (secosw, sesinw), ci = cos i; ``consts``, if given, replaces
    (k, h) with an ``anchor_constants[_ew]`` tuple. Returns (phi_1, phi_2,
    phi_3, phi_4), ordered.
    """
    from .anchored import _one_minus_cos, anchor_constants, solve_sincos_delta
    if consts is None:
        consts = anchor_constants(k, h)
    e, ecw, esw, es, ec, B1, A2, B2 = consts
    a_sky, _ = contact_geometry(a, e, esw, ci)
    lin = contact_offsets(r, a_sky, b)

    def z2_slope(phi, c):
        """z^2 and d(z^2)/dphi at phase phi, for the constants c."""
        a_, ci_, ecw_, esw_, es_, ec_, B1_, A2_, B2_ = c
        sd, cd = solve_sincos_delta(phi, es_, ec_)
        omc = _one_minus_cos(sd, cd)
        u = a_ * (-ecw_ * omc - B1_ * sd)
        v = a_ * (A2_ * cd - B2_ * sd - esw_)
        dud = a_ * (-ecw_ * sd - B1_ * cd)
        dvd = a_ * (-A2_ * sd - B2_ * cd)
        D = 1.0 + es_ * sd - ec_ * cd
        return u * u + (v * ci_) ** 2, 2.0 * (u * dud + ci_ * ci_ * v * dvd) / D

    # the search runs detached; only the last step below carries gradients
    zero = mx.stop_gradient(lin[0] * 0.0 + e * 0.0)
    live = tuple(x + zero for x in (a, ci, ecw, esw, es, ec, B1, A2, B2))
    frozen = tuple(mx.stop_gradient(x) for x in live)
    asq = mx.stop_gradient(a_sky * a_sky + zero)
    rl = r + zero
    r_ = mx.stop_gradient(rl)
    half = math.pi / 2

    # 1. the closest approach
    p0 = zero
    s0 = z2_slope(p0, frozen)[1]
    p1 = mx.clip(-s0 / (2.0 * asq), -half, half)
    z1, s1 = z2_slope(p1, frozen)
    for _ in range(_NSEC):
        ds = s1 - s0
        ok = ds != 0.0
        step = mx.where(ok, s1 * (p1 - p0) / mx.where(ok, ds, 1.0), 0.0)
        p0, s0 = p1, s1
        p1 = mx.clip(p1 - mx.clip(step, -0.5, 0.5), -half, half)
        z1, s1 = z2_slope(p1, frozen)
    pm, zm2 = p1, z1

    # 2.-3. the four roots, stacked: (outer, inner, inner, outer)
    zo, zi = 1.0 + r_, mx.abs(1.0 - r_)
    Z2 = mx.stack([zo, zi, zi, zo]) ** 2
    sgn = mx.array([-1.0, -1.0, 1.0, 1.0], dtype=zero.dtype).reshape(
        (4,) + (1,) * zero.ndim)
    exists = zm2 < Z2
    dq = mx.sqrt(mx.maximum(Z2 - zm2, 0.0) / asq)    # quadratic model
    p_in = mx.broadcast_to(pm, Z2.shape)
    p_out = pm + sgn * mx.minimum(2.0 * dq + 1e-6, half)
    for _ in range(_NEXP):                            # until outside
        grow = (z2_slope(p_out, frozen)[0] - Z2) <= 0.0
        p_out = mx.where(grow, pm + mx.clip(2.0 * (p_out - pm), -half, half),
                         p_out)
    phi = pm + sgn * dq
    for _ in range(int(n_iter)):
        z2, dz2 = z2_slope(phi, frozen)
        g = z2 - Z2
        outside = g > 0.0
        p_out = mx.where(outside, phi, p_out)
        p_in = mx.where(outside, p_in, phi)
        okd = dz2 != 0.0
        newton = phi - g / mx.where(okd, dz2, 1.0)
        within = okd & ((newton - p_in) * (newton - p_out) <= 0.0)
        phi = mx.where(within, newton, 0.5 * (p_in + p_out))
    root = mx.stop_gradient(mx.where(exists, phi, pm))

    # the implicit-function gradient, with the value left as it is
    z2, dz2 = z2_slope(root, live)
    Zl2 = mx.stack([(1.0 + rl) ** 2, (1.0 - rl) ** 2, (1.0 - rl) ** 2,
                    (1.0 + rl) ** 2])
    okg = exists & (mx.abs(dz2) > 0.0)
    corr = mx.where(okg, -(z2 - Zl2) / mx.where(okg, dz2, 1.0), 0.0)
    root = root + (corr - mx.stop_gradient(corr))

    out = [root[0], root[1], root[2], root[3]]
    # roots on either side of phi_min are ordered by construction; keep the
    # edge clamp's monotonicity guaranteed all the same
    for i in range(1, 4):
        out[i] = mx.maximum(out[i], out[i - 1])
    eccentric = e > 0.0
    return tuple(mx.where(eccentric, o, l) for o, l in zip(out, lin))
