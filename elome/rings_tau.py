"""Ringed planets on an orbit: sky positions, kinks, exposures.

``metal.flux_dev_from_tau(..., ring_in=, ...)`` and ``TransitModel`` with
ring parameters delegate here. The light curve is ``rings.shape_cols_ring``
at the planet's sky position in its principal frame (sky frame and theta
convention: ``oblate_tau``), integrated over each exposure. This is the
graph (fp64 on the CPU stream), or for fp32 data on the GPU the fused
kernels of ``metal_rings``.

**Kinks.** The light curve is smooth except where an outline -- the planet
or a ring edge -- is tangent to the stellar limb, i.e. where its number of
limb crossings changes: first and last contact, the planet fully on the
disc, a nearly edge-on ring going from two crossings to four and back, a
ring edge larger than the star closing round it. Per outline, the count is
sampled at _N_SAMPLE phases across the window in which its circumscribed
circle can touch the star, and every change is bisected to a phase (at
most _N_EVENTS per outline). Two changes closer than the sampling step
that cancel are missed -- a brief 2 -> 4 -> 2 episode, whose effect on an
exposure average is correspondingly small. Kinks are split points of the
exposure quadrature only, detached, as the spherical and oblate contacts
are.

**Exposures.** ``integration="contact"`` splits each exposure window at
the kinks that fall inside it -- at most _K_SPLIT of them, nearest first --
and integrates each piece by Gauss-Legendre: (_K_SPLIT + 1) * n_gl nodes
per exposure whatever the number of kinks along the transit, the same
count as the spherical and oblate rule.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .dtypes import fp64_on_cpu
from .exposure import contact_offsets, contact_offsets_anchored, gauss_legendre
from .hybrid import combine_cols, get_law
from .oblate import _F_SW
from .oblate_tau import _contact_kh, _orbit_consts, _sky
from .rings import _check_ring, _crossings, ring_axes, shape_cols_ring
from .trig import sincos

__all__ = ["ring_kinks", "flux_dev_from_tau_ring"]

_TWO_PI = 2.0 * math.pi
_N_SAMPLE = 48                                    # count samples per outline
_N_EVENTS = 8                                     # kinks kept per outline
_N_BISECT = {mx.float64: 40, mx.float32: 24}
_K_SPLIT = 4                                      # split points per exposure
# Evaluation points (sets x times x exposure nodes) per block of the graph
# (_in_blocks): ~1 GB of fp64 intermediates and cache.
_RING_BLOCK = 1 << 15


# --------------------------------------------------------------------------
# kinks
# --------------------------------------------------------------------------

def _count(phi, a, b, ecc, A, B, cth, sth, dt):
    """Limb crossings of an outline (semi-axes A, B) at orbital phase phi
    (0 on the far side)."""
    X, Y, _, _, front = _sky(phi, a, b, ecc)
    x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
    A, B = A + 0.0 * x0, B + 0.0 * x0
    circ = (A - B) <= _F_SW[dt] * A
    _, valid = _crossings(x0, y0, A, B, circ, dt)
    n = sum(v.astype(mx.int32) for v in valid)
    return mx.where(front, n, 0)


def _outline_kinks(a, b, ecc, A, B, cth, sth, window, dt):
    """(phases, ok), each (n, _N_EVENTS): where this outline's crossing
    count changes, bisected from a sampled count."""
    lo, hi = window
    S = _N_SAMPLE
    u = mx.array(np.linspace(0.0, 1.0, S), dtype=dt)
    ph = lo + (hi - lo) * u                                   # (n, S)
    cnt = _count(ph, a, b, ecc, A, B, cth, sth, dt)
    change = cnt[:, 1:] != cnt[:, :-1]                        # (n, S - 1)
    idx = mx.arange(S - 1)
    key = mx.sort(mx.where(change, idx, S), axis=-1)[:, :_N_EVENTS]
    ok = key < S
    k = mx.minimum(key, S - 2)
    plo = mx.take_along_axis(ph[:, :-1], k, axis=-1)
    phi = mx.take_along_axis(ph[:, 1:], k, axis=-1)
    clo = mx.take_along_axis(cnt[:, :-1], k, axis=-1)
    for _ in range(_N_BISECT[dt]):
        mid = 0.5 * (plo + phi)
        same = _count(mid, a, b, ecc, A, B, cth, sth, dt) == clo
        plo = mx.where(same, mid, plo)
        phi = mx.where(same, phi, mid)
    return 0.5 * (plo + phi), ok


def _ring_kinks(a, b, r, f, theta, ring_in, ring_out, ring_f, k=None, h=None,
                ci=None, consts=None):
    """ring_kinks' work on canonical (n,) inputs: (n, 3 * _N_EVENTS) kink
    phases, sorted, +inf where an outline has fewer. Detached."""
    sg = mx.stop_gradient
    a, b, r, f, theta, ring_in, ring_out, ring_f = (
        sg(v)[:, None] for v in (a, b, r, f, theta, ring_in, ring_out, ring_f))
    dt = a.dtype
    if k is None:
        ecc = None
    else:
        from .anchored import anchor_constants
        k, h, ci = (sg(v)[:, None] for v in (k, h, ci))
        consts = (tuple(sg(c) for c in anchor_constants(k, h)) if consts is None
                  else tuple(sg(c)[:, None] if c.ndim == 1 else sg(c) for c in consts))
        ecc = (ci, consts)
    sth, cth = sincos(theta)
    planet, edges = ring_axes(r, f, ring_in, ring_out, ring_f)
    # The three outlines solved as one (3 n, 1) chain: each row's arithmetic
    # is unchanged, with a third of the ops of three solves (eager, the
    # sampled counts and bisections are ~2e5 small ops per outline).
    outlines = [planet] + edges
    n = max(v.shape[0] for v in (a, b, cth, *(x for o in outlines for x in o)))

    def t3(v):
        return mx.concatenate([mx.broadcast_to(v, (n,) + v.shape[1:])] * 3, axis=0)
    A, B = (mx.concatenate([mx.broadcast_to(o[j], (n, 1)) for o in outlines],
                           axis=0) for j in (0, 1))
    a, b, cth, sth = (t3(v) for v in (a, b, cth, sth))
    if ecc is None:
        cA = contact_offsets(A, a, b)
    else:
        k, h = t3(k), t3(h)
        ecc = (t3(ecc[0]), tuple(t3(c) for c in ecc[1]))
        cA = contact_offsets_anchored(A, a, b, k, h, ecc[0], consts=ecc[1])
    lo, hi = cA[0], cA[3]
    pad = 0.02 * (hi - lo) + 1e-9
    ph, ok = _outline_kinks(a, b, ecc, A, B, cth, sth, (lo - pad, hi + pad), dt)
    out = mx.split(mx.where(ok, ph, mx.inf), 3, axis=0)
    return sg(mx.sort(mx.concatenate(out, axis=-1), axis=-1))


def ring_kinks(a, b, r, f, theta, ring_in, ring_out, ring_f, k=None, h=None):
    """The phases (offsets from mid-transit, phi = 2 pi (t - t0) / P) at
    which a ringed planet's light curve has a kink -- an outline tangent to
    the limb -- sorted, +inf in unused slots: shape (n, 24) for n parameter
    sets broadcast from the arguments (Python numbers, numpy or mx arrays).
    a: a/R*; b: impact parameter at conjunction; r, f, theta: the planet;
    ring_in, ring_out, ring_f: the ring; (k, h) = (secosw, sesinw) for an
    eccentric orbit. Float64 on the CPU stream unless every array argument
    is float32."""
    vals = [a, b, r, f, theta, ring_in, ring_out, ring_f] + ([k, h] if k is not None else [])
    arrays = [v for v in vals if isinstance(v, mx.array)]
    dt = (mx.float32 if arrays and all(v.dtype == mx.float32 for v in arrays)
          else mx.float64)
    shape = np.broadcast_shapes(*(tuple(v.shape) if isinstance(v, mx.array)
                                  else np.shape(v) for v in vals))
    n = int(np.prod(shape)) if shape else 1

    def norm(v):
        x = (v.astype(dt) if isinstance(v, mx.array)
             else mx.array(np.asarray(v, dtype=np.float64), dtype=dt))
        return mx.reshape(mx.broadcast_to(x, shape), (n,))

    with mx.stream(mx.cpu):
        cols = [norm(v) for v in vals]
        if k is not None:
            kk, hh = cols[8], cols[9]
            ci = M._ecc_shape(cols[0], cols[1], kk, hh)[2]
            return _ring_kinks(*cols[:8], kk, hh, ci)
        return _ring_kinks(*cols[:8])


# --------------------------------------------------------------------------
# exposures
# --------------------------------------------------------------------------

def _exposure_nodes_kinks(t, period, exp_time, kinks, n_gl, dtype):
    """Nodes and weights over each exposure window [t - e/2, t + e/2],
    split at the kinks inside it (the earliest _K_SPLIT; more in one
    exposure would need kinks closer than exp_time / 5). t:
    (n, ...) times since mid-transit; kinks: (n, E) phase offsets. Returns
    (times, weights) with a trailing axis of (_K_SPLIT + 1) * n_gl,
    weights summing to 1."""
    xg, wg = gauss_legendre(n_gl)
    xg, wg = mx.array(xg, dtype=dtype), mx.array(wg, dtype=dtype)
    half = 0.5 * exp_time
    t1, t2 = t - half, t + half
    tc = mx.round(t / period) * period                    # the epoch's centre
    extra = (1,) * (t.ndim - 1)
    kn = mx.reshape(kinks, (kinks.shape[0],) + extra + (kinks.shape[-1],))
    per = mx.reshape(period, (-1,) + extra + (1,)) if period.ndim else period
    te = tc[..., None] + kn * (per / _TWO_PI)
    inside = (te > t1[..., None]) & (te < t2[..., None])
    key = mx.sort(mx.where(inside, te, mx.inf), axis=-1)[..., :_K_SPLIT]
    edges = [t1] + [mx.where(mx.isinf(key[..., j]), t2, key[..., j]) for j in range(_K_SPLIT)] + [t2]
    for i in range(1, len(edges)):
        edges[i] = mx.maximum(edges[i], edges[i - 1])
    times, weights = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mid, hw = 0.5 * (lo + hi), 0.5 * (hi - lo)
        times.append(mid[..., None] + hw[..., None] * xg)
        weights.append(hw[..., None] * wg / exp_time)
    W = mx.concatenate(weights, axis=-1)
    W = W / mx.sum(W, axis=-1, keepdims=True)
    return mx.concatenate(times, axis=-1), W


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def _ring_ok(r, f, ring_in, ring_out, ring_f, ring_tau):
    """The parameters' domain as a mask (arrays are never read back:
    traced values outside it give NaN, as the oblate path's do)."""
    return ((f >= 0.0) & (f < 1.0) & (ring_f >= 0.0) & (ring_f < 1.0)
            & (ring_tau >= 0.0) & (ring_out > ring_in)
            & (ring_in * mx.sqrt(1.0 - mx.minimum(mx.maximum(f, 0.0), 0.999999))
               >= 1.0 - 1e-12) & (r > 0.0))


def _graph_kinks(a, b, r, f, theta, ring_in, ring_out, ring_f, k=None,
                 h=None, ew=False):
    """The contact rule's kinks for _tau_graph's canonical (n,) inputs
    (detached: they only place the quadrature's split points)."""
    geo = (a, b, r, f, theta, ring_in, ring_out, ring_f)
    if k is None:
        return _ring_kinks(*geo)
    ci, consts = _orbit_consts(a, b, k, h, ew)
    kc, hc = _contact_kh(k, h, ew)
    return _ring_kinks(*geo, kc, hc, ci, consts=consts)


def _tau_graph(tau, period, a, b, r, f, theta, ring_in, ring_out, ring_f,
               ring_tau, w2d, exp_time, mode, n_gl, n_sub, law, basis=False,
               k=None, h=None, ew=False, kinks=None):
    """oblate_tau._tau_graph for a ringed planet; ``kinks`` the contact
    rule's (_graph_kinks), computed here if not given."""
    law = get_law(law)
    ecc = k is not None
    pars = (period, a, b, r, f, theta, ring_in, ring_out, ring_f, ring_tau) \
        + ((k, h) if ecc else ())

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
        rv, fv, thv, riv, rov, rfv, rtv = cols[3:10]
        X, Y, _, _, front = _sky((_TWO_PI / per) * tt, av, bv, eo)
        far = 2.0 + 2.0 * mx.stop_gradient(rov * rv)          # clear of the outer edge
        X = mx.where(front, X, far)
        Y = mx.where(front, Y, 0.0)
        sth, cth = sincos(thv)
        x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
        return shape_cols_ring(x0, y0, rv, fv, riv, rov, rfv, rtv, law)

    if mode == M._INT_NONE or exp_time == 0.0:
        Bc = inst(tau, 1)
    elif mode == M._INT_SUPER:
        half = 0.5 * exp_time
        off = (np.linspace(-half, half, int(n_sub)) if n_sub > 1
               else np.zeros(1))
        nodes = tau[..., None] + mx.array(off, dtype=tau.dtype)
        Bc = mx.mean(inst(nodes, 2), axis=-2)
    else:
        per1 = mx.stop_gradient(col(1)[0])
        if kinks is None:
            kinks = _graph_kinks(a, b, r, f, theta, ring_in, ring_out, ring_f,
                                 k, h, ew)
        T, W = _exposure_nodes_kinks(tau, per1, exp_time, kinks, int(n_gl),
                                     tau.dtype)
        Bc = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


def _nodes_per_time(exp_time, mode, n_gl, n_sub):
    """Points the graph evaluates per time (``_tau_graph``'s exposure rule)."""
    if mode == M._INT_NONE or exp_time == 0.0:
        return 1
    if mode == M._INT_SUPER:
        return max(int(n_sub), 1)
    return (_K_SPLIT + 1) * int(n_gl)


def _pad(x, size, axis):
    """``x`` padded to ``size`` along ``axis`` (0 or 1) with its last slice."""
    k = size - x.shape[axis]
    if k == 0:
        return x
    last = x[-1:] if axis == 0 else x[:, -1:]
    shape = list(x.shape)
    shape[axis] = k
    return mx.concatenate([x, mx.broadcast_to(last, shape)], axis=axis)


def _in_blocks(run, tau2d, rows, nodes):
    """``run(tau2d, rows)`` -- the ring graph, (n, m[, ncol]) -- in blocks of
    at most _RING_BLOCK evaluation points, concatenated lazily. The graph
    holds ~20-24 KB of fp64 intermediates per point until evaluated (~24 GB
    at 10^6 points); MLX evaluates the blocks one after another and frees
    each block's intermediates before the next, so the peak is one block's,
    at no cost in time. Each flux depends only on its own row (``rows``:
    the per-set (n, ...) arrays, or None) and time. Blocks are whole rows
    while a row fits, else one row's times; equal-sized, padded with the
    last row or time and the padding dropped, so one compiled shape and one
    cached buffer set serve every block (a shorter last block would
    allocate, and cache, a second set)."""
    n, m = tau2d.shape
    per_row = nodes * m
    if n * per_row <= _RING_BLOCK:
        return run(tau2d, rows)

    def equal(total, most):
        nb = -(-total // max(most, 1))
        return nb, -(-total // nb)

    def sl(x, i, j):
        return None if x is None else x[i:j]
    if per_row <= _RING_BLOCK:
        nb, size = equal(n, _RING_BLOCK // per_row)
        tp = _pad(tau2d, nb * size, 0)
        rp = [None if x is None else _pad(x, nb * size, 0) for x in rows]
        return mx.concatenate(
            [run(tp[i * size:(i + 1) * size],
                 [sl(x, i * size, (i + 1) * size) for x in rp])
             for i in range(nb)], axis=0)[:n]
    nb, size = equal(m, _RING_BLOCK // nodes)
    tp = _pad(tau2d, nb * size, 1)
    return mx.concatenate(
        [mx.concatenate([run(tp[j:j + 1, i * size:(i + 1) * size],
                             [sl(x, j, j + 1) for x in rows])
                         for i in range(nb)], axis=1)
         for j in range(n)], axis=0)[:, :m]


@fp64_on_cpu
def flux_dev_from_tau_ring(tau, period, a, b, r, f, theta, ring_in, ring_out,
                           ring_f, ring_tau, law, u, exp_time, mode, n_gl,
                           n_sub, basis=False, k=None, h=None, ew=False,
                           use_metal=True):
    """``metal.flux_dev_from_tau`` with ``ring_in=`` (that function validates
    the keywords and delegates here). fp32 data on the GPU stream takes the
    ring Metal kernels (``metal_rings``) unless ``use_metal`` is False;
    anything else runs this graph, fp64 on the CPU stream. ``ew``: (k, h)
    hold (e, w [rad]) -- TransitModel's array-ecc route."""
    from .metal_hybrid import _canon_w
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    _check_ring(f, ring_in, ring_out, ring_f, ring_tau)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r, f, theta, ring_in, ring_out, ring_f, ring_tau) \
        + ((k, h) if ecc else ())
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
        from .metal_rings import flux_dev_from_tau_ring_kernel
        out = flux_dev_from_tau_ring_kernel(
            tau2d, *pc[:10], w2d, exp_time, mode, n_gl, n_sub, law,
            basis=basis, k=kh[0], h=kh[1], ew=ew)
    else:
        # the contact rule's kinks depend on the parameters alone: once per
        # set, not per block
        contact = mode not in (M._INT_NONE, M._INT_SUPER) and exp_time != 0.0
        kinks = (_graph_kinks(*pc[1:9], kh[0], kh[1], ew) if contact
                 else None)

        def run(tt, rows):
            *p10, w, kk, hh, kn = rows
            return _tau_graph(tt, *p10, w, exp_time, mode, n_gl, n_sub, law,
                              basis=basis, k=kk, h=hh, ew=ew, kinks=kn)
        out = _in_blocks(run, tau2d, [*pc[:10], w2d, *kh, kinks],
                         _nodes_per_time(exp_time, mode, n_gl, n_sub))
    ok = _ring_ok(*pc[3:5], *pc[6:10])[:, None]
    out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    return out[0] if squeeze and n == 1 else out
