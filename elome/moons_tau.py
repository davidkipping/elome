"""A planet and its moon on their orbits: the light curve in time.

``TransitModel`` with ``params.moon_r`` delegates here. Positions are
``moons.py``'s, the photometry its disc and lens pieces.

**Times.** As the anvil data contract (``orbit.epoch_center_times``): each
point's time from a reference transit, dt = t - t0_ref - k P_ref, and its
orbit number k, float64 on the host against references fixed when the
model is built, with the moon's turns k P_ref / P_m,ref reduced there too
(``moons.moon_turns_ref``). The graph forms tau = dt - (t0_off + k p_off)
-- not re-wrapped to the nearest transit, since the moon's phase counts
from k -- and the moon's phase from the period offsets
(``moons.moon_phase``), so float32 holds the moon's position at any
distance from t0.

**What is evaluated.** A point whose barycentre is too far from the star
for either body to reach it at any time in its exposure is exactly 0, and
the lens is taken only at the nodes where the two discs overlap with both
on the star: frozen choices made from host values, as
``TransitModel._event_view``'s, so the light curve and its gradients are
those of the full model.

**Exposures.** Instantaneous, or the contact rule (``exp_time`` > 0): each
exposure split at the light curve's kinks inside it -- at most _K_SPLIT,
earliest first -- and each piece integrated by Gauss-Legendre. The kinks
are where a disc touches the limb (z = 1 +- r, for the planet and the
moon), where the discs touch each other (d = rp +- rs) near the star, and
where a corner of the two discs crosses the limb. The moon moves between
transits, so they are found per exposure: eight smooth event functions are
sampled at n_sample + 1 times across the window, the sign changes read on
the host, and only those refined (regula falsi, then bisection where it
is slow). Only the pieces with width are evaluated, each with its lane's
parameters. The step (``sample_count``) is a quarter of the smaller radius
at the fastest sky speed; a pair of kinks closer than a step that cancel
-- a brief graze -- is missed, at a cost as small as its effect.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import moons
from .exposure import gauss_legendre
from .hybrid import combine_cols, declared

__all__ = ["flux_dev_moon_t", "sample_count"]

_TWO_PI = 2.0 * math.pi
_K_SPLIT = 8                                       # split points per exposure
_S_MIN, _S_MAX = 4, 256                            # event samples per exposure
_N_BISECT = {mx.float64: 30, mx.float32: 16}       # the kinks' bracket, in halvings
_N_FALSI = 8                                       # Illinois steps before them
_BLOCK = 1 << 14                                   # evaluation nodes per block
_EVENT_BLOCK = 1 << 20                             # event samples per block (frozen)
_sg = mx.stop_gradient


def sample_count(exp_time, a, per, rp, a_ps, moon_per, rs, ecc=0.0, moon_ecc=0.0):
    """Event samples per exposure: the step is a quarter of the smaller disc
    radius at the fastest sky speed (the barycentre's at periastron plus the
    moon's about the planet at its periastron). Host values."""
    v = (_TWO_PI * a / per * math.sqrt((1.0 + ecc) / max(1.0 - ecc, 1e-12))
         + _TWO_PI * a_ps / moon_per * math.sqrt((1.0 + moon_ecc) / max(1.0 - moon_ecc, 1e-12)))
    r = min(x for x in (rp, rs) if x > 0.0) if max(rp, rs) > 0.0 else 1.0
    return int(min(max(math.ceil(exp_time * v / (0.25 * r)), _S_MIN), _S_MAX))


def _geometry(P):
    """pos(dt, k, turns, delta) -> (xp, yp, front_p, xm, ym, front_m, zB)
    for the parameter dict P (graph values)."""
    def pos(dt, k, turns, delta):
        tau = dt + delta - (P["t0_off"] + k * P["p_off"])
        XB, YB, ZB = moons.barycentre_sky((_TWO_PI / P["per"]) * tau, P["a"], P["inc"],
                                          P["ecc"])
        phi_m = moons.moon_phase(tau, k, turns, P["per_ref"], P["mper_ref"], P["p_off"],
                                 P["m_off"], P["phase"])
        d = moons.moon_offset(phi_m, P["a_ps"], P["moon_inc"], P["node"], P["orbit"])
        (xp, yp, zp), (xm, ym, zm) = moons.body_positions(XB, YB, ZB, *d, P["q"])
        return xp, yp, zp > 0.0, xm, ym, zm > 0.0, mx.sqrt(XB * XB + YB * YB)
    return pos


def _freeze(v):
    """Stop gradients through a parameter: an array, a tuple of them (the
    anchored constants, the moon's orbit) or a host value."""
    if isinstance(v, mx.array):
        return _sg(v)
    if isinstance(v, tuple):
        return tuple(_freeze(c) for c in v)
    return v


def _frozen(P):
    return {k: _freeze(v) for k, v in P.items()}


def _event_g(pos, P, dtype):
    """g(delta, k, turns, dt) -> (..., 9): the event functions, whose sign
    changes are the kinks: z_p^2 - (1 +- rp)^2, z_m^2 - (1 +- rs)^2,
    d^2 - (rp +- rs)^2 and |c|^2 - 1 for the discs' two corners c, then
    h^2, positive where the corners exist (no event function itself). Each
    is continuous in time -- without corners, c is the point on the line of
    centres the corners merge at -- so every root found moves with the
    parameters as ``_live`` takes it to. A root of c where there are no
    corners is no kink, and ``_events`` drops it; another that is no kink
    (the discs touching off the star) only splits a smooth stretch."""
    rp, rs = P["rp"], P["rs"]
    tiny = 1e-300 if dtype == mx.float64 else 1e-30

    def g(delta, kk, tt, dtt):
        xp, yp, _, xm, ym, _, _ = pos(dtt, kk, tt, delta)
        zp2, zm2 = xp * xp + yp * yp, xm * xm + ym * ym
        dx, dy = xm - xp, ym - yp
        d2 = dx * dx + dy * dy
        D = mx.sqrt(mx.maximum(d2, tiny))
        a_ = (d2 + (rp - rs) * (rp + rs)) / (2.0 * D)
        hh2 = (rp - a_) * (rp + a_)
        has = hh2 > 0.0
        hh = mx.where(has, mx.sqrt(mx.where(has, hh2, 1.0)), 0.0)  # no sqrt'(0) on masked lanes
        bx, by = xp + a_ * dx / D, yp + a_ * dy / D
        c1 = (bx - hh * dy / D) ** 2 + (by + hh * dx / D) ** 2 - 1.0
        c2 = (bx + hh * dy / D) ** 2 + (by - hh * dx / D) ** 2 - 1.0
        gs = [zp2 - (1.0 + rp) ** 2, zp2 - (1.0 - rp) ** 2,
              zm2 - (1.0 + rs) ** 2, zm2 - (1.0 - rs) ** 2,
              d2 - (rp + rs) ** 2, d2 - (rp - rs) ** 2, c1, c2, hh2]
        return mx.stack(gs, axis=-1)
    return g


def _at(P, ix):
    """P's per-lane (lanes, 1) columns at the lanes ix; shared values kept."""
    return _lanes(P, lambda v: mx.take(v, ix, axis=0))


def _brackets(P_frozen, sets, dt, k, turns, hw0, n_sample, dtype):
    """The kinks' brackets, read on the host from the event functions with
    the frozen parameters, sampled at n_sample + 1 times across each exposure: per kink its lane,
    its event function and its bracket (lo, hi) as offsets from the exposure
    centre, host arrays ordered by lane, at most _K_SPLIT per lane (the
    earliest). The samples are taken in blocks of lanes (no gradient is
    kept, so each block's memory is freed with it)."""
    j = np.arange(n_sample + 1, dtype=np.float64) / n_sample
    deltas = -hw0 + 2.0 * hw0 * j
    dm = mx.array(deltas, dtype=dtype)[None, :]
    step = max(1, _EVENT_BLOCK // (n_sample + 1))
    found = []
    for i in range(0, dt.shape[0], step):
        e = lambda v: v[i:i + step, None]                           # noqa: E731
        Pb = _lanes(P_frozen, lambda v: v[i:i + step]) if sets else P_frozen
        g = _event_g(_geometry(Pb), Pb, dtype)
        G = g(dm, e(k), e(turns), e(dt))                            # (block, J + 1, 9)
        S = G[..., :8] > 0.0
        change = S[:, :-1] != S[:, 1:]                              # (block, J, 8)
        # the corners' functions only where there are corners at either
        # end: a sign change of the merged point (the same in both) is no kink
        corners = (G[:, :-1, 8:] > 0.0) | (G[:, 1:, 8:] > 0.0)
        change = np.array(mx.concatenate([change[..., :6], change[..., 6:] & corners], axis=-1))
        lb, jb, fb = np.nonzero(change)                             # lane, time, function order
        found.append((lb + i, jb, fb))
    lane, jj, ff = (np.concatenate(c) for c in zip(*found))
    if lane.size:
        start = np.searchsorted(lane, lane, side="left")
        keep = np.arange(lane.size) - start < _K_SPLIT
        lane, jj, ff = lane[keep], jj[keep], ff[keep]
    return lane, ff, deltas[jj], deltas[jj + 1]


def _refine(gf, lo, hi, dtype):
    """Each kink's time in its bracket (lo, hi) (host arrays), where its event
    function gf changes sign: regula falsi with the Illinois rule, which
    keeps the bracket and narrows it superlinearly, then bisection on any
    bracket not yet as narrow as _N_BISECT halvings make it (a tangency's
    slow root). The final bracket's midpoint, a host array."""
    tol = (hi - lo) * 2.0 ** -_N_BISECT[dtype]
    a, b = mx.array(lo, dtype=dtype), mx.array(hi, dtype=dtype)
    fa, fb = gf(a), gf(b)
    side = mx.zeros(a.shape, dtype=mx.int32)
    for _ in range(_N_FALSI):
        den = fb - fa
        c = mx.where(den != 0.0, (a * fb - b * fa) / mx.where(den != 0.0, den, 1.0),
                     0.5 * (a + b))
        c = mx.minimum(mx.maximum(c, a), b)
        fc = gf(c)
        right = (fc > 0.0) == (fb > 0.0)                            # c replaces b
        a, b, fa, fb = (mx.where(right, a, c), mx.where(right, c, b),
                        mx.where(right, mx.where(side == -1, 0.5 * fa, fa), fc),
                        mx.where(right, fc, mx.where(side == 1, 0.5 * fb, fb)))
        side = mx.where(right, -1, 1)
    a, b, fa = np.array(a), np.array(b), np.array(fa)
    slow = np.nonzero(b - a > tol)[0]
    if slow.size:
        n = int(np.ceil(np.log2(np.max((b - a)[slow] / tol[slow]))))
        lo_, hi_ = mx.array(a[slow], dtype=dtype), mx.array(b[slow], dtype=dtype)
        pos = mx.array(fa[slow] > 0.0)
        for _ in range(n):
            mid = 0.5 * (lo_ + hi_)
            same = (gf(mid, slow) > 0.0) == pos
            lo_, hi_ = mx.where(same, mid, lo_), mx.where(same, hi_, mid)
        a[slow], b[slow] = np.array(lo_), np.array(hi_)
    return 0.5 * (a + b)


def _picker(P, d_s, k_s, t_s, lane, ff, sets, dtype):
    """gf(delta, sub=None): each kink's own event function, with P (live or
    frozen) at the kink's lane, at offsets delta (one per kink, or per kink
    of the subset sub)."""
    ix = mx.array(lane.astype(np.int32))
    Pk = _at(P, ix) if sets else P
    cols = [mx.take(v, ix)[:, None] for v in (k_s, t_s, d_s)]
    f_all = mx.array(ff.astype(np.int32))

    def gf(delta, sub=None):
        Pe, (kk, tt, dtt), f = Pk, cols, f_all
        if sub is not None:
            sx = mx.array(np.asarray(sub, np.int32))
            Pe = _at(Pk, sx) if sets else Pk
            kk, tt, dtt = (mx.take(v, sx, axis=0) for v in cols)
            f = mx.take(f_all, sx)
        G = _event_g(_geometry(Pe), Pe, dtype)(delta[:, None], kk, tt, dtt)[:, 0, :]
        return mx.take_along_axis(G, f[:, None], axis=-1)[:, 0]
    return gf


def _pieces(PL, P_frozen, d_s, k_s, t_s, hw0, n_sample, n_gl, sets, dtype):
    """The contact rule's pieces, only those with width: per piece its lane
    (a host array) and its Gauss-Legendre offsets and normalised weights
    (pieces, n_gl). The kinks are found with the frozen parameters and
    moved with the live ones by one gradient-only Newton step t - [g(t) -
    sg(g(t))] / g'(t) on each kink's own function: the value stays the
    root, the gradient is the implicit one, -dg/dtheta / g', so the nodes
    move with the parameters and the gradient is that of the quadrature
    evaluated."""
    m = d_s.shape[0]
    lane, ff, lo, hi = _brackets(P_frozen, sets, d_s, k_s, t_s, hw0, n_sample, dtype)
    if lane.size:
        gf = _picker(P_frozen, d_s, k_s, t_s, lane, ff, sets, dtype)
        ev = _refine(gf, lo, hi, dtype)
        eps = (1e-7 if dtype == mx.float64 else 1e-3) * hw0
        evm = mx.array(ev, dtype=dtype)
        slope = np.array((gf(evm + eps) - gf(evm - eps)) / (2.0 * eps))
        keep = np.nonzero(slope != 0.0)[0]
        lane, ff, ev, slope = lane[keep], ff[keep], ev[keep], slope[keep]
    if lane.size:
        gl = _picker(PL, d_s, k_s, t_s, lane, ff, sets, dtype)(mx.array(ev, dtype=dtype))
        live = mx.array(ev, dtype=dtype) - (gl - _sg(gl)) / mx.array(slope, dtype=dtype)
        live = mx.minimum(mx.maximum(live, -hw0), hw0)
        order = np.lexsort((ev, lane))                              # by lane, then time
        lane, live = lane[order], mx.take(live, mx.array(order.astype(np.int32)))
    n_ev = np.bincount(lane, minlength=m)
    # edges per lane: -hw, its kinks in time order, hw; source vector
    # [kinks..., -hw, hw]
    first = np.concatenate([[0], np.cumsum(n_ev)[:-1]])
    pl = np.repeat(np.arange(m), n_ev + 1)                          # each piece's lane
    jp = np.arange(pl.size) - np.repeat(first + np.arange(m), n_ev + 1)   # its index in the lane
    ne, base = lane.size, first[pl] + jp
    lo_ix = np.where(jp == 0, ne, base - 1)
    hi_ix = np.where(jp == n_ev[pl], ne + 1, base)
    hwv = mx.array([-hw0, hw0], dtype=dtype)
    src = mx.concatenate([live, hwv]) if ne else hwv
    lo_e = mx.take(src, mx.array(lo_ix.astype(np.int32)))
    hi_e = mx.take(src, mx.array(hi_ix.astype(np.int32)))
    xg, wg = gauss_legendre(n_gl)
    mid, half = 0.5 * (lo_e + hi_e), 0.5 * (hi_e - lo_e)
    T = mid[:, None] + half[:, None] * mx.array(xg, dtype=dtype)
    # each lane's halves sum to hw exactly (the kinks telescope)
    W = half[:, None] * mx.array(wg / (hw0 * np.sum(wg)), dtype=dtype)
    return pl, T, W


def _photometry(pos, P, kind, large, dt, k, turns, T):
    """Per-node contributions on points (m,) at offsets T (m, n): the shape
    columns (hybrid; a list) or F - 1 (quadratic). P and kind may hold
    per-point (m, 1) columns (``_lanes``)."""
    e = lambda v: mx.broadcast_to(v[:, None], T.shape)
    xp, yp, fp, xm, ym, fm, _ = pos(e(dt), e(k), e(turns), T)
    rp, rs = P["rp"], P["rs"]
    full = lambda v: mx.broadcast_to(v, T.shape)
    with declared(large):
        if kind[0] == "hybrid":
            law = kind[1]
            cols = [a + b for a, b in zip(moons.disc_shape_cols(xp, yp, rp, law, fp),
                                          moons.disc_shape_cols(xm, ym, rs, law, fm))]
        else:
            u1, u2 = kind[1], kind[2]
            cols = [moons.disc_flux_quad(xp, yp, rp, u1, u2, fp)
                    + moons.disc_flux_quad(xm, ym, rs, u1, u2, fm)]
    # the lens, where the discs overlap with both on the star (host choice)
    gx = [np.array(_sg(v)) for v in (xp, yp, xm, ym)]
    grp, grs = (np.broadcast_to(np.array(_sg(v)), T.shape) for v in (rp, rs))
    zp_, zm_ = np.hypot(gx[0], gx[1]), np.hypot(gx[2], gx[3])
    dd = np.hypot(gx[2] - gx[0], gx[3] - gx[1])
    need = (np.array(fp) & np.array(fm) & (dd < (grp + grs) * (1.0 + 1e-9))
            & (zp_ < 1.0 + grp) & (zm_ < 1.0 + grs)).ravel()
    idx = np.nonzero(need)[0]
    if idx.size:
        ix = mx.array(idx.astype(np.int32))
        flat = (lambda v: mx.take(full(v).reshape(-1), ix)
                if isinstance(v, mx.array) and v.ndim else v)
        L = [flat(v) for v in (xp, yp, xm, ym)]
        lrp, lrs = flat(rp), flat(rs)
        if kind[0] == "hybrid":
            lens = moons.lens_shape_cols(L[0], L[1], lrp, L[2], L[3], lrs, kind[1])
        else:
            lens = [moons.lens_flux_quad(L[0], L[1], lrp, L[2], L[3], lrs, flat(kind[1]),
                                         flat(kind[2]))]
        cols = [(c.reshape(-1).at[ix].add(lc)).reshape(T.shape) for c, lc in zip(cols, lens)]
    return cols


def _map_sets(v, fn):
    """fn applied to every per-set array (ndim >= 1) in v: an array, a tuple
    of them (the anchored constants, the moon's orbit) or a host value."""
    if isinstance(v, mx.array):
        return fn(v) if v.ndim else v
    if isinstance(v, tuple):
        return tuple(_map_sets(c, fn) for c in v)
    return v


def _lanes(P, fn):
    return {key: _map_sets(v, fn) for key, v in P.items()}


def flux_dev_moon_t(dt, k, turns, P, kind, large=False, exp_time=0.0, n_gl=7,
                    n_sample=None, n_sets=None):
    """F - 1 of a planet and its moon at times (dt, k, turns), (m,) each --
    host float64 arrays, or mx arrays (anvil's rows) -- from
    ``orbit.epoch_center_times`` and ``moons.moon_turns_ref``
    against the references in P. P: the graph parameters -- t0_off, p_off,
    per, a, inc [rad], ecc (``moons.anchored_constants`` or None), rp, rs,
    a_ps (stellar radii), q, m_off, moon_inc, node, phase [rad], orbit
    (``moons.moon_orbit`` or None) -- and the host values per_ref, mper_ref
    (the references), speed (the barycentre's largest sky speed, an upper
    bound) and optionally moon_e (the moon's eccentricity, for the reach
    bound; None or absent: 1). kind: ("hybrid", law, w) or ("quad", u1, u2). ``large``: rp
    may exceed 1. ``exp_time`` > 0: the contact rule, ``n_gl`` nodes per
    piece, ``n_sample`` event samples per exposure (None: 64).

    ``n_sets``: many parameter sets at the same times, each value in P and
    kind (n_sets,) -- (n_sets, n_w) for the hybrid weights -- or shared (a
    0-d array or host value), and the result (n_sets, m). Each evaluated
    (set, point) pair is a lane, its parameters gathered to (lanes, 1)
    columns, so one pass serves every set; the host choices (reach, the
    lens's nodes, the number of splits) are taken over all of them."""
    dtype = P["a"].dtype if isinstance(P["a"], mx.array) else mx.float64
    m = dt.shape[0]
    n = 1 if n_sets is None else int(n_sets)
    hw0 = 0.5 * exp_time
    dt_m, k_m, tu_m = (v.astype(dtype) if isinstance(v, mx.array) else mx.array(v, dtype=dtype)
                       for v in (dt, k, turns))
    # the lanes either body can reach the star from, in their exposure: each
    # body lies within its share of the moon's apoapsis of the barycentre,
    # which moves at most speed hw in half an exposure
    host = lambda v: np.broadcast_to(np.asarray(                    # noqa: E731
        _sg(v) if isinstance(v, mx.array) else 1.0 if v is None else v,
        np.float64).reshape(-1), (n,))
    a_ps, rp_h, rs_h, e_m = (host(P.get(key)) for key in ("a_ps", "rp", "rs", "moon_e"))
    q_h = np.maximum(host(P["q"]), 0.0)
    R = a_ps * (1.0 + e_m) / (1.0 + q_h)
    reach = 1.0 + np.maximum(rs_h + R, rp_h + q_h * R) + P["speed"] * hw0
    col = _lanes(_frozen(P), lambda v: v.reshape(-1, 1))           # (n, 1) columns
    tau = dt_m[None] - (col["t0_off"] + k_m[None] * col["p_off"])
    XB, YB, _ = moons.barycentre_sky((_TWO_PI / col["per"]) * tau, col["a"], col["inc"],
                                     col["ecc"])
    zB = np.broadcast_to(np.array(mx.sqrt(XB * XB + YB * YB)), (n, m))
    sel = np.nonzero((zB < reach[:, None]).reshape(-1))[0]
    out_cols = None
    if sel.size:
        six = mx.array(sel.astype(np.int32))
        pix = mx.array((sel % m).astype(np.int32))
        sets = n_sets is not None
        if not sets:
            PL, lkind = P, kind
        else:
            sx = mx.array((sel // m).astype(np.int32))
            gather = lambda v: mx.take(v, sx, axis=0)[:, None]
            PL = _lanes(P, gather)
            lkind = kind if kind[0] == "hybrid" else (kind[0],) + tuple(
                _map_sets(v, gather) for v in kind[1:])
        d_s, k_s, t_s = (mx.take(v, pix) for v in (dt_m, k_m, tu_m))
        if exp_time > 0.0:
            # the pieces with width, each evaluated at its lane's values
            pl, T, W = _pieces(PL, _frozen(PL), d_s, k_s, t_s, hw0, int(n_sample or 64), n_gl,
                               sets, dtype)
        else:
            pl = np.arange(sel.size)
            T = mx.zeros((sel.size, 1), dtype=dtype)
            W = mx.ones((sel.size, 1), dtype=dtype)
        step = max(1, _BLOCK // T.shape[1])
        sub = None
        for i in range(0, pl.size, step):
            j = slice(i, min(i + step, pl.size))
            ix = mx.array(pl[j].astype(np.int32))
            if exp_time > 0.0:
                take = lambda v: mx.take(v, ix, axis=0)             # noqa: E731
            else:
                take = lambda v: v[j]                               # noqa: E731
            Pj, kj = PL, lkind
            if sets:
                Pj = _lanes(PL, take)
                if lkind[0] == "quad":
                    kj = (lkind[0],) + tuple(_map_sets(v, take) for v in lkind[1:])
            cols = _photometry(_geometry(Pj), Pj, kj, large, take(d_s), take(k_s), take(t_s),
                               T[j])
            part = [mx.sum(c * W[j], axis=-1) for c in cols]
            if sub is None:
                sub = [mx.zeros((sel.size,), dtype=dtype) for _ in part]
            sub = [s_.at[ix].add(c) for s_, c in zip(sub, part)]
        out_cols = [mx.zeros((n * m,), dtype=dtype).at[six].add(s_) for s_ in sub]
        if n_sets is not None:
            out_cols = [c.reshape(n, m) for c in out_cols]
    shape = (m,) if n_sets is None else (n, m)
    if out_cols is None:
        return mx.zeros(shape, dtype=dtype)
    if kind[0] == "hybrid":
        return combine_cols(out_cols, kind[2], kind[1])
    return out_cols[0]
