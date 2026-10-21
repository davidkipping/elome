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
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .dtypes import fp64_on_cpu
from .hybrid import combine_cols, get_law
from .oblate import _check_domain, principal_frame, shape_cols_oblate
from .trig import sincos

__all__ = ["sky_circular", "sky_anchored", "contact_offsets_oblate",
           "flux_dev_from_tau_oblate"]

_TWO_PI = 2.0 * math.pi
_N_PSI = 16              # boundary samples per extremum search
_N_PSI_NEWTON = 3        # Newton steps polishing the sampled extremum
_N_CONTACT = {mx.float64: 14, mx.float32: 10}     # Newton steps in phase
_FAR = 4.0               # far-side stand-in, |(X, Y)| > 1 + A for A < 1


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


def _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind):
    """q (kind = -1: squared distance to the disc, 0 inside it) or M
    (kind = +1: squared largest boundary distance) at phase phi, and its
    phase derivative."""
    X, Y, dX, dY, _ = _sky(phi, a, b, ecc)
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


def _newton_contact(phi, a, b, ecc, A, B, cth, sth, kind, side, lim, n):
    """Newton on g(phi) - 1 = 0 from phi, on the outside of the root
    (side -1: ingress, the root lies at larger phi; +1: egress)."""
    for _ in range(n):
        g, dg = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind)
        # outside the root the slope points away from it: dg * side > 0
        ok = (dg * side) > 0.0
        step = mx.where(ok, (g - 1.0) / mx.where(ok, dg, 1.0), 0.0)
        phi = phi - mx.clip(step, -lim, lim)
    g, _ = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind)
    return phi, g


def contact_offsets_oblate(a, b, r, f, theta, k=None, h=None, ci=None):
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
    four onto one point (no transit).
    """
    if (k is None) != (h is None):
        raise ValueError("pass both k and h (secosw, sesinw), or neither")
    vals = [a, b, r, f, theta] + ([k, h] if k is not None else []) \
        + ([ci] if ci is not None else [])
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
        out = _contact_offsets_oblate(a, b, r, f, theta, k, h, ci)
        return tuple(mx.reshape(c, shape) for c in out)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _contact_offsets_oblate(a, b, r, f, theta, k=None, h=None, ci=None,
                            consts=None):
    """contact_offsets_oblate's work, on canonical inputs: (n,) mx arrays
    of one dtype on the active stream (the graph, oblate kernels and tests
    call it directly). ``consts``, if given, replaces anchor_constants(k, h)
    -- the (e, w) route."""
    from .exposure import contact_offsets, contact_offsets_anchored
    a, b, r, f, theta = (mx.stop_gradient(v) for v in (a, b, r, f, theta))
    dt = a.dtype
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
    sth, cth = sincos(theta)
    lo, hi = cA[0], cA[3]
    lim = mx.maximum(hi - lo, 1e-12)
    n = _N_CONTACT[dt]
    tol = 1e-8 if dt == mx.float64 else 1e-4

    c1, g1 = _newton_contact(lo, a, b, ecc, A, B, cth, sth, -1, -1.0, lim, n)
    c4, g4 = _newton_contact(hi, a, b, ecc, A, B, cth, sth, -1, 1.0, lim, n)
    outer = (mx.abs(g1 - 1.0) < tol) & (mx.abs(g4 - 1.0) < tol) & (c1 < c4)
    mid = 0.5 * (lo + hi)
    c1, c4 = mx.where(outer, c1, mid), mx.where(outer, c4, mid)

    lim_i = mx.maximum(c4 - c1, 1e-12)
    c2, g2 = _newton_contact(c1, a, b, ecc, A, B, cth, sth, 1, -1.0, lim_i, n)
    c3, g3 = _newton_contact(c4, a, b, ecc, A, B, cth, sth, 1, 1.0, lim_i, n)
    inner = (outer & (mx.abs(g2 - 1.0) < tol) & (mx.abs(g3 - 1.0) < tol)
             & (c1 <= c2) & (c2 < c3) & (c3 <= c4))
    mid = 0.5 * (c1 + c4)
    c2, c3 = mx.where(inner, c2, mid), mx.where(inner, c3, mid)
    return tuple(mx.stop_gradient(c) for c in (c1, c2, c3, c4))


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
               contacts=None):
    """The oblate counterpart of ``metal_hybrid._tau_graph``: the same
    exposure rules, contacts detached and the period detached in the node
    map exactly as there. ``contacts``: the contact rule's (n, 4)
    (_graph_contacts), computed here if not given."""
    from .exposure import exposure_nodes
    law = get_law(law)
    ecc = k is not None
    pars = (period, a, b, r, f, theta) + ((k, h) if ecc else ())

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
        X, Y, _, _, front = _sky((_TWO_PI / per) * tt, av, bv, eo)
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
            contacts = _graph_contacts(a, b, r, f, theta, k, h, ew)
        cs = tuple(contacts[:, j:j + 1] for j in range(4))
        T, W = exposure_nodes(tau, tau * 0.0, mx.stop_gradient(per1),
                              exp_time, cs, int(n_gl), dtype=tau.dtype)
        Bc = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


def _graph_contacts(a, b, r, f, theta, k=None, h=None, ew=False):
    """The contact rule's (n, 4) contact phases for _tau_graph's canonical
    (n,) inputs, detached."""
    if k is None:
        cs = _contact_offsets_oblate(a, b, r, f, theta)
    else:
        ci, consts = _orbit_consts(a, b, k, h, ew)
        kc, hc = _contact_kh(k, h, ew)
        cs = _contact_offsets_oblate(a, b, r, f, theta, kc, hc, ci,
                                     consts=consts)
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
              compiled):
    """``run(tt, rows)`` for _in_blocks: _tau_graph on one block's rows
    (``present``: which of them are arrays). ``compiled``: under
    mx.compile, cached per configuration -- every block has one shape, so
    one trace serves them all, where compiling a whole blocked call traces
    every block (minutes at 30 blocks)."""
    def raw(tt, *arrs):
        it = iter(arrs)
        *p6, w, kk, hh, c = [next(it) if p else None for p in present]
        return _tau_graph(tt, *p6, w, exp_time, mode, n_gl, n_sub, law,
                          basis=basis, k=kk, h=hh, ew=ew, contacts=c)
    fn = raw
    if compiled:
        key = (float(exp_time), int(mode), int(n_gl), int(n_sub), law,
               bool(basis), bool(ew), present)
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


def _in_blocks(run, tau2d, rows, nodes, block):
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
    allocate, and cache, a second set)."""
    n, m = tau2d.shape
    per_row = nodes * m
    if n * per_row <= block:
        return run(tau2d, rows)

    def equal(total, most):
        nb = -(-total // max(most, 1))
        return nb, -(-total // nb)

    def sl(x, i, j):
        return None if x is None else x[i:j]
    if per_row <= block:
        nb, size = equal(n, block // per_row)
        tp = _pad(tau2d, nb * size, 0)
        rp = [None if x is None else _pad(x, nb * size, 0) for x in rows]
        return mx.concatenate(
            [run(tp[i * size:(i + 1) * size],
                 [sl(x, i * size, (i + 1) * size) for x in rp])
             for i in range(nb)], axis=0)[:n]
    nb, size = equal(m, block // nodes)
    tp = _pad(tau2d, nb * size, 1)
    return mx.concatenate(
        [mx.concatenate([run(tp[j:j + 1, i * size:(i + 1) * size],
                             [sl(x, j, j + 1) for x in rows])
                         for i in range(nb)], axis=1)
         for j in range(n)], axis=0)[:, :m]


@fp64_on_cpu
def flux_dev_from_tau_oblate(tau, period, a, b, r, f, theta, law, u,
                             exp_time, mode, n_gl, n_sub, basis=False,
                             k=None, h=None, ew=False, use_metal=True,
                             compile_blocks=False):
    """``metal.flux_dev_from_tau`` with ``f=`` (that function validates the
    keywords and delegates here). fp32 data on the GPU stream takes the
    oblate Metal kernels (``metal_oblate``) unless ``use_metal`` is False;
    anything else runs this graph, fp64 on the CPU stream, in blocks of at
    most _OBLATE_BLOCK points (_in_blocks) -- each block compiled once for
    all (they share one shape) with ``compile_blocks``, TransitModel's
    route for a call too large to compile whole. ``ew``: (k, h) hold (e, w
    [rad]) -- TransitModel's array-ecc route (_orbit_consts)."""
    from .metal_hybrid import _canon_w
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    _check_domain(r, f)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r, f, theta) + ((k, h) if ecc else ())
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(tau2d.shape[0], n_param)
    pc = [M._canon_param(p, n, tau2d.dtype) for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    kh = pc[-2:] if ecc else [None, None]
    w2d = (None if basis else
           _canon_w(u, n, law.n_w, tau2d.dtype, law.name))
    if (use_metal and tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        from .metal_oblate import flux_dev_from_tau_oblate_kernel
        from .oblate import _in_domain
        out = flux_dev_from_tau_oblate_kernel(
            tau2d, *pc[:6], w2d, exp_time, mode, n_gl, n_sub, law,
            basis=basis, k=kh[0], h=kh[1], ew=ew)
        ok = _in_domain(pc[3], pc[4])[:, None]
        out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    else:
        # the contact rule's contacts depend on the parameters alone: once
        # per set, not per block
        contact = mode not in (M._INT_NONE, M._INT_SUPER) and exp_time != 0.0
        cs = _graph_contacts(*pc[1:6], kh[0], kh[1], ew) if contact else None

        rows = [*pc[:6], w2d, *kh, cs]
        run = _block_fn(exp_time, mode, n_gl, n_sub, law, basis, ew,
                        tuple(x is not None for x in rows), compile_blocks)
        out = _in_blocks(run, tau2d, rows,
                         _nodes_per_time(exp_time, mode, n_gl, n_sub),
                         _OBLATE_BLOCK)
    return out[0] if squeeze and n == 1 else out
