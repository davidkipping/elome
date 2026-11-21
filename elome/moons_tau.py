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
sampled at n_sample + 1 times across the window and each sign change
bisected. The step (``sample_count``) is a quarter of the smaller radius
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
_K_SPLIT = 6                                       # split points per exposure
_S_MIN, _S_MAX = 4, 256                            # event samples per exposure
_N_BISECT = {mx.float64: 30, mx.float32: 16}
_BLOCK = 1 << 14                                   # evaluation nodes per block
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
    """g(delta, k, turns, dt) -> (..., 8): the event functions, whose sign
    changes are the kinks: z_p^2 - (1 +- rp)^2, z_m^2 - (1 +- rs)^2,
    d^2 - (rp +- rs)^2 and |c|^2 - 1 for the discs' two corners c. Each is
    continuous in time -- without corners, c is the point on the line of
    centres the corners merge at -- so every root found moves with the
    parameters as ``_live`` takes it to; a root that is no kink (the discs
    touching off the star, a merged point crossing the limb) only splits a
    smooth stretch."""
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
              d2 - (rp + rs) ** 2, d2 - (rp - rs) ** 2, c1, c2]
        return mx.stack(gs, axis=-1)
    return g


def _events(g, dt, k, turns, hw, n_sample, dtype):
    """Kink times (offsets from each exposure's centre), their validity and
    which event function each is a root of, (m, _K_SPLIT) each, earliest
    first: found with the frozen event functions g."""
    j = mx.arange(n_sample + 1, dtype=dtype) / n_sample
    hw2 = hw[:, None]
    deltas = -hw2 + 2.0 * hw2 * j[None, :]                         # (m, J + 1)
    e = lambda v: v[:, None]
    G = g(deltas, e(k), e(turns), e(dt))                            # (m, J + 1, 8)
    change = (G[:, :-1, :] > 0.0) != (G[:, 1:, :] > 0.0)            # (m, J, 8)
    m, J, F = change.shape
    key = mx.where(change, mx.arange(J * F, dtype=mx.int32).reshape(1, J, F), J * F)
    key = mx.sort(key.reshape(m, J * F), axis=-1)[:, :_K_SPLIT]
    ok = key < J * F
    keyc = mx.where(ok, key, 0)
    jj, ff = keyc // F, keyc % F
    lo = mx.take_along_axis(deltas, jj, axis=-1)
    hi = mx.take_along_axis(deltas, jj + 1, axis=-1)
    kk, tt, dtt = (mx.broadcast_to(e(v), lo.shape) for v in (k, turns, dt))

    def gf(delta):
        return mx.take_along_axis(g(delta, kk, tt, dtt), ff[..., None], axis=-1)[..., 0]
    glo = gf(lo)
    for _ in range(_N_BISECT[dtype]):
        mid = 0.5 * (lo + hi)
        same = (gf(mid) > 0.0) == (glo > 0.0)
        lo, hi = mx.where(same, mid, lo), mx.where(same, hi, mid)
    return _sg(0.5 * (lo + hi)), ok, ff


def _live(events, ok, ff, g_live, g_frozen, dt, k, turns, hw, dtype):
    """The events moved with the parameters: one gradient-only Newton step
    t - [g(t) - sg(g(t))] / g'(t) on each event's own function, live -- the
    value stays the bisected root, and the gradient is the implicit one,
    -dg/dtheta / g', so the contact rule's nodes move with the parameters
    and the gradient is that of the quadrature evaluated."""
    e = lambda v: mx.broadcast_to(v[:, None], events.shape)
    kk, tt, dtt = e(k), e(turns), e(dt)

    def pick(gfun, delta):
        return mx.take_along_axis(gfun(delta, kk, tt, dtt), ff[..., None], axis=-1)[..., 0]
    eps = (1e-7 if dtype == mx.float64 else 1e-3) * hw[:, None]
    slope = _sg((pick(g_frozen, events + eps) - pick(g_frozen, events - eps)) / (2.0 * eps))
    ok = ok & (mx.abs(slope) > 0.0)
    gl = pick(g_live, events)
    step = (gl - _sg(gl)) / mx.where(ok, slope, 1.0)
    return mx.where(ok, events - step, events), ok


def _nodes(events, ok, hw, n_k, n_gl, dtype):
    """Gauss-Legendre offsets and weights (normalised) over each exposure
    split at its first n_k events: (m, (n_k + 1) n_gl) each."""
    hw2 = hw[:, None]
    ev = mx.where(ok[:, :n_k], events[:, :n_k], hw2)
    ev = mx.minimum(mx.maximum(ev, -hw2), hw2)
    # sorted by argsort and gathered: the event times carry gradients
    ev = mx.take_along_axis(ev, _sg(mx.argsort(ev, axis=-1)), axis=-1)
    edges = mx.concatenate([-hw2, ev, hw2], axis=-1)
    lo, hi = edges[:, :-1], edges[:, 1:]
    xg, wg = (mx.array(v, dtype=dtype) for v in gauss_legendre(n_gl))
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    T = (mid[..., None] + half[..., None] * xg).reshape(lo.shape[0], -1)
    W = (half[..., None] * wg).reshape(lo.shape[0], -1)
    return T, W / mx.sum(W, axis=-1, keepdims=True)


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
    (the references) and speed (the barycentre's largest sky speed, an
    upper bound). kind: ("hybrid", law, w) or ("quad", u1, u2). ``large``: rp
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
    # the lanes either body can reach the star from, in their exposure
    hmax = lambda v: float(np.max(np.array(_sg(v)))) if isinstance(v, mx.array) else float(v)
    reach = (1.0 + 2.0 * hmax(P["a_ps"]) + hmax(P["rp"]) + hmax(P["rs"])
             + P["speed"] * 2.0 * hw0)
    col = _lanes(_frozen(P), lambda v: v.reshape(-1, 1))           # (n, 1) columns
    tau = dt_m[None] - (col["t0_off"] + k_m[None] * col["p_off"])
    XB, YB, _ = moons.barycentre_sky((_TWO_PI / col["per"]) * tau, col["a"], col["inc"],
                                     col["ecc"])
    zB = np.broadcast_to(np.array(mx.sqrt(XB * XB + YB * YB)), (n, m)).reshape(-1)
    sel = np.nonzero(zB < reach)[0]
    out_cols = None
    if sel.size:
        six = mx.array(sel.astype(np.int32))
        pix = mx.array((sel % m).astype(np.int32))
        if n_sets is None:
            PL, lkind = P, kind
        else:
            sx = mx.array((sel // m).astype(np.int32))
            gather = lambda v: mx.take(v, sx, axis=0)[:, None]
            PL = _lanes(P, gather)
            lkind = kind if kind[0] == "hybrid" else (kind[0],) + tuple(
                _map_sets(v, gather) for v in kind[1:])
        pos = _geometry(PL)
        frozen = _geometry(_frozen(PL))
        d_s, k_s, t_s = (mx.take(v, pix) for v in (dt_m, k_m, tu_m))
        if exp_time > 0.0:
            hw = mx.full(d_s.shape, hw0, dtype=dtype)
            g_frozen = _event_g(frozen, _frozen(PL), dtype)
            ev, ok, ff = _events(g_frozen, d_s, k_s, t_s, hw, int(n_sample or 64), dtype)
            n_k = int(np.max(np.sum(np.array(ok), axis=-1))) if ok.size else 0
            ev, ok = _live(ev, ok, ff, _event_g(pos, PL, dtype), g_frozen, d_s, k_s, t_s, hw,
                           dtype)
            T, W = _nodes(ev, ok, hw, n_k, n_gl, dtype)
        else:
            T = mx.zeros((sel.size, 1), dtype=dtype)
            W = mx.ones((sel.size, 1), dtype=dtype)
        per_pt = T.shape[1]
        step = max(1, _BLOCK // per_pt)
        blocks = []
        for i in range(0, sel.size, step):
            j = slice(i, min(i + step, sel.size))
            if n_sets is None:
                Pj, kj = PL, lkind
            else:
                Pj = _lanes(PL, lambda v: v[j])
                kj = (lkind[0],) + tuple(_map_sets(v, lambda c: c[j]) for v in lkind[1:]) \
                    if lkind[0] == "quad" else lkind
            cols = _photometry(_geometry(Pj), Pj, kj, large, d_s[j], k_s[j], t_s[j], T[j])
            blocks.append([mx.sum(c * W[j], axis=-1) for c in cols])
        sub = [mx.concatenate([b[c] for b in blocks]) for c in range(len(blocks[0]))]
        out_cols = [mx.zeros((n * m,), dtype=dtype).at[six].add(s) for s in sub]
        if n_sets is not None:
            out_cols = [c.reshape(n, m) for c in out_cols]
    shape = (m,) if n_sets is None else (n, m)
    if out_cols is None:
        return mx.zeros(shape, dtype=dtype)
    if kind[0] == "hybrid":
        return combine_cols(out_cols, kind[2], kind[1])
    return out_cols[0]
