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
on the star (where one lies wholly inside the other, the union is the
larger disc): frozen choices made from host values, as
``TransitModel._event_view``'s, so the light curve and its gradients are
those of the full model.

**Exposures.** Instantaneous, or the contact rule (``exp_time`` > 0): each
exposure split at the light curve's kinks inside it -- at most _K_SPLIT,
earliest first -- and each piece integrated by Gauss-Legendre. The kinks
are where a disc touches the limb (z = 1 +- r, for the planet and the
moon), where the discs touch each other (d = rp +- rs) near the star, and
where a corner of the two discs crosses the limb. The moon moves between
transits, so they are found per exposure: on the exposures whose bodies
can reach a contact within them (``_screen``), eight smooth event
functions are sampled at n_sample + 1 times across the window, the sign
changes read on the host, and only those refined (regula falsi, then bisection where it
is slow), on the GPU compiled from host copies of the parameters. Only
the pieces with width are evaluated, each with its lane's parameters.
The step (``sample_count``) is a quarter of the smaller radius at the
fastest sky speed; a pair of kinks closer than a step that cancel -- a
brief graze -- is missed, at a cost as small as its effect.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import moons
from .dtypes import _gpu_stream_active
from .exposure import gauss_legendre
from .hybrid import combine_cols, declared

__all__ = ["flux_dev_moon_t", "sample_count"]

_TWO_PI = 2.0 * math.pi
_K_SPLIT = 8                                       # split points per exposure
_S_MIN, _S_MAX = 4, 256                            # event samples per exposure
_N_BISECT = {mx.float64: 30, mx.float32: 16}       # the kinks' bracket, in halvings
_N_FALSI = 10                                      # Illinois steps before them
# evaluation nodes per block: a forward light curve's memory is bounded by
# it (no gradient keeps the blocks); fewer blocks build less graph
_BLOCK = 1 << 18
_EVENT_BLOCK = 1 << 18                             # event samples per block (frozen)
_sg = mx.stop_gradient


def sample_count(exp_time, a, per, rp, a_ps, moon_per, rs, ecc=0.0, moon_ecc=0.0):
    """Event samples per exposure: the step is a quarter of the smaller disc
    radius at the fastest sky speed (the barycentre's at periastron plus the
    moon's about the planet at its periastron). Host values."""
    v = (_TWO_PI * a / per * math.sqrt((1.0 + ecc) / max(1.0 - ecc, 1e-12))
         + _TWO_PI * a_ps / moon_per * math.sqrt((1.0 + moon_ecc) / max(1.0 - moon_ecc, 1e-12)))
    r = min(x for x in (rp, rs) if x > 0.0) if max(rp, rs) > 0.0 else 1.0
    return int(min(max(math.ceil(exp_time * v / (0.25 * r)), _S_MIN), _S_MAX))


def _prep(P, dtype):
    """The positions' constants per set, once, from P's values (graph or
    host): 2 pi / P, the angles' (sin, cos), the bodies' shares of the
    moon's offset, its period; each an array, so a frozen copy of them is
    a compiled stage's input."""
    c = lambda v: v if isinstance(v, (mx.array, tuple)) or v is None \
        else mx.array(float(v), dtype=dtype)                        # noqa: E731
    q = P["q"]
    Q = dict(t0_off=P["t0_off"], p_off=P["p_off"], w=_TWO_PI / P["per"], a=P["a"],
             ecc=P["ecc"], inc=moons._sincos_any(P["inc"]), a_ps=P["a_ps"],
             orbit=P["orbit"], moon_inc=moons._sincos_any(P["moon_inc"]),
             node=moons._sincos_any(P["node"]), fp=q / (1.0 + q), fm=1.0 / (1.0 + q),
             moon_per=P["mper_ref"] + P["m_off"], phase=P["phase"], rp=P["rp"], rs=P["rs"])
    return {key: tuple(c(x) for x in v) if key in ("inc", "moon_inc", "node") else c(v)
            for key, v in Q.items()}


def _lane_cols(P, k, turns, dtype):
    """Per lane, (lanes, 1) each: t0_off + k p_off, and the moon's turns
    fixed per orbit (``moons._turns_base``; in float32 its float-float
    part, which is no compiled graph's). P: (lanes, 1) columns or shared."""
    kk, tt = k[:, None], turns[:, None]
    base, _ = moons._turns_base(kk, tt, P["per_ref"], P["mper_ref"], P["p_off"], P["m_off"],
                                dtype)
    return P["t0_off"] + kk * P["p_off"], base


def _pos(Q, dt, o, u, delta):
    """(xp, yp, front_p, xm, ym, front_m) at times dt + delta of lanes with
    ``_lane_cols``' (o, u), for ``_prep``'s constants Q."""
    tau = dt + delta - o
    XB, YB, ZB = moons._barycentre(Q["w"] * tau, Q["a"], Q["inc"], Q["ecc"])
    phi_m = moons._phase_at(tau, u, Q["moon_per"], Q["phase"])
    d = moons._offset(phi_m, Q["a_ps"], Q["moon_inc"], Q["node"], Q["orbit"])
    (xp, yp, zp), (xm, ym, zm) = moons._bodies(XB, YB, ZB, *d, Q["fp"], Q["fm"])
    return xp, yp, zp > 0.0, xm, ym, zm > 0.0


def _reach_z(Q, dt, k):
    """The barycentre's separation at times dt of orbits k: a frozen stage."""
    tau = dt - (Q["t0_off"] + k * Q["p_off"])
    XB, YB, _ = moons._barycentre(Q["w"] * tau, Q["a"], Q["inc"], Q["ecc"])
    return mx.sqrt(XB * XB + YB * YB)


# --------------------------------------------------------------------------
# compiled frozen stages
# --------------------------------------------------------------------------

_KERNELS = {}


def _spec(v, leaves):
    if isinstance(v, mx.array):
        leaves.append(v)
        return "a"
    if v is None:
        return None
    if isinstance(v, dict):
        return ("d", tuple((key, _spec(v[key], leaves)) for key in sorted(v)))
    return ("t", tuple(_spec(c, leaves) for c in v))


def _unspec(s, it):
    if s == "a":
        return next(it)
    if s is None:
        return None
    if s[0] == "d":
        return {key: _unspec(c, it) for key, c in s[1]}
    return tuple(_unspec(c, it) for c in s[1])


def _run(fn, *args):
    """fn(*args), on the GPU compiled once per function and structure of
    args: trees (dicts, tuples) of arrays and None. Only for frozen stages
    fed host copies (``_host``): under a traced gradient mx.compile does
    nothing for arrays computed from traced ones. MLX retraces per input
    shape, hence the bucketed lane counts (``_bucket``). On the CPU eager:
    there the compiled kernels are C++, built on first use at ~0.2 s each
    (the event functions make over a hundred), for a gain of a third."""
    leaves = []
    specs = tuple(_spec(a, leaves) for a in args)
    if not _compiling(mx.float64 if any(v.dtype == mx.float64 for v in leaves)
                      else mx.float32):
        return fn(*args)
    key = (fn, specs)
    f = _KERNELS.get(key)
    if f is None:
        def flat(*xs):
            it = iter(xs)
            return fn(*(_unspec(sp, it) for sp in specs))
        f = _KERNELS[key] = mx.compile(flat)
    return f(*leaves)


def _walk(v, fn):
    """v (a tree) with fn applied to each of its arrays, in order."""
    if isinstance(v, mx.array):
        return fn(v)
    if isinstance(v, dict):
        return {key: _walk(c, fn) for key, c in v.items()}
    if isinstance(v, (tuple, list)):
        return type(v)(_walk(c, fn) for c in v)
    return v


def _host(v):
    """A fresh copy of every array in v (a tree), no graph behind it: the
    arrays evaluated together, once, then copied."""
    frozen = []
    _walk(v, lambda x: frozen.append(_sg(x)))
    if frozen:
        mx.eval(*frozen)
    copies = iter([mx.array(np.asarray(x), dtype=x.dtype) for x in frozen])
    return _walk(v, lambda x: next(copies))


def _compiling(dtype):
    return dtype != mx.float64 and _gpu_stream_active()


def _bucket(n, dtype):
    """A stage's lane count for n lanes: compiled, a power of two."""
    return 1 << max(3, (int(n) - 1).bit_length()) if _compiling(dtype) else int(n)


def _padded(ix, size):
    """Indices ix, repeated at the last to ``size``."""
    return np.concatenate([ix, np.full(size - ix.size, ix[-1], dtype=ix.dtype)])


def _event_g(Q):
    """g(delta, dt, o, u) -> (..., 9): the event functions, whose sign
    changes are the kinks: z_p^2 - (1 +- rp)^2, z_m^2 - (1 +- rs)^2,
    d^2 - (rp +- rs)^2 and |c|^2 - 1 for the discs' two corners c, then
    h^2, positive where the corners exist (no event function itself). Each
    is continuous in time -- without corners, c is the point on the line of
    centres the corners merge at -- so every root found moves with the
    parameters as ``_pieces`` takes it to. A root of c where there are no
    corners is no kink, and ``_brackets`` drops it; another that is no kink
    (the discs touching off the star) only splits a smooth stretch."""
    rp, rs = Q["rp"], Q["rs"]

    def g(delta, dt, o, u):
        tiny = 1e-300 if dt.dtype == mx.float64 else 1e-30
        xp, yp, _, xm, ym, _ = _pos(Q, dt, o, u, delta)
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


def _changes(Q, dt, o, u, dm):
    """The event functions' sign changes between samples dm (1, J + 1) of
    lanes (dt, o, u), (lanes, J) bytes, bit f set where function f changes
    sign (one byte to read back per interval, not eight): the corners' only
    where there are corners at either end, a sign change of the merged
    point (the same in both) being no kink. A frozen stage."""
    G = _event_g(Q)(dm, dt, o, u)                                   # (lanes, J + 1, 9)
    S = G[..., :8] > 0.0
    change = S[:, :-1] != S[:, 1:]
    corners = (G[:, :-1, 8:] > 0.0) | (G[:, 1:, 8:] > 0.0)
    change = mx.concatenate([change[..., :6], change[..., 6:] & corners], axis=-1)
    bits = mx.array([1 << f for f in range(8)], dtype=mx.uint8)
    return mx.sum(change.astype(mx.uint8) * bits, axis=-1).astype(mx.uint8)


def _centres(Q, dt, o, u):
    """(xp, yp, xm, ym) at each lane's exposure centre, (lanes, 4)."""
    xp, yp, _, xm, ym, _ = _pos(Q, dt, o, u, 0.0 * dt)
    return mx.concatenate([xp, yp, xm, ym], axis=-1)


def _screen(Q, cols, hw0, screen, dtype):
    """The lanes whose exposure may hold a kink, host indices: from the
    bodies at its centre and how far they can move in half an exposure.
    screen: per lane (rp, rs, v_b, v_m), host arrays -- the radii, the
    barycentre's largest sky speed and the moon's about the planet (each
    body moves at most v_b + v_m, their separation at most v_m). A lane is
    kept where an event function can change sign: a disc crossing either
    of its contacts with the limb, the discs either tangency, or a corner
    crossing the limb -- which needs the discs overlapping with both
    reaching the limb (a corner appearing mid-exposure is a tangency).
    Every sign change the event search can find is kept, spurious ones
    included, so its result is unchanged."""
    rp, rs, vb, vm = screen
    c = np.array(_run(_centres, Q, *cols), dtype=np.float64)
    zp, zm = np.hypot(c[:, 0], c[:, 1]), np.hypot(c[:, 2], c[:, 3])
    d = np.hypot(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1])
    slack = 1e-5 if dtype == mx.float32 else 1e-10
    with np.errstate(invalid="ignore"):
        mb, mr = (vb + vm) * hw0 + slack, vm * hw0 + slack           # a body, the separation
        near = lambda x, level, move: np.abs(x - level) <= move     # noqa: E731
        keep = (near(zp, 1.0 + rp, mb) | near(zp, np.abs(1.0 - rp), mb)
                | near(zm, 1.0 + rs, mb) | near(zm, np.abs(1.0 - rs), mb)
                | near(d, rp + rs, mr) | near(d, np.abs(rp - rs), mr)
                | ((d - mr < rp + rs) & (d + mr > np.abs(rp - rs))
                   & near(zp, 1.0, rp + mb) & near(zm, 1.0, rs + mb))
                | ~np.isfinite(mb + mr))
    return np.nonzero(keep)[0]


def _pick(Q, dt, o, u, f, delta):
    """Each kink's own event function f at its offset delta, (kinks,)."""
    G = _event_g(Q)(delta[:, None], dt, o, u)[:, 0, :]
    return mx.take_along_axis(G, f[:, None], axis=-1)[:, 0]


def _brackets(Q, cols, hw0, n_sample, dtype):
    """The kinks' brackets, read on the host from the event functions with
    frozen values (Q, lane columns cols = (dt, o, u)), sampled at n_sample +
    1 times across each exposure: per kink its lane, its event function and
    its bracket (lo, hi) as offsets from the exposure centre, host arrays
    ordered by lane, at most _K_SPLIT per lane (the earliest). The samples
    are taken in blocks of lanes (``_run``, each a bucketed count)."""
    j = np.arange(n_sample + 1, dtype=np.float64) / n_sample
    deltas = -hw0 + 2.0 * hw0 * j
    dm = mx.array(deltas, dtype=dtype)[None, :]
    m = cols[0].shape[0]
    step = max(1, _EVENT_BLOCK // (n_sample + 1))
    found = []
    for i in range(0, m, step):
        c = min(step, m - i)
        ix = mx.array(np.minimum(np.arange(i, i + min(_bucket(c, dtype), step)), m - 1)
                      .astype(np.int32))
        take = lambda v: mx.take(v, ix, axis=0)                     # noqa: E731
        packed = np.array(_run(_changes, _lanes(Q, take), *(take(v) for v in cols), dm))[:c]
        nz = np.flatnonzero(packed)
        bits = np.unpackbits(packed.reshape(-1)[nz][:, None], axis=1, bitorder="little")
        r, fb = np.nonzero(bits)
        lb, jb = np.divmod(nz[r], packed.shape[1])                  # lane, time, function order
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
        # c on the root itself (common: near its root an O(1) function falls
        # on the rounding grid): the bracket collapses there, else the
        # secant would return that end every step, the bracket frozen
        zero = fc == 0.0
        a, b = mx.where(zero, c, a), mx.where(zero, c, b)
    mx.eval(a, b, fa)
    a, b, fa = np.array(a), np.array(b), np.array(fa)
    slow = np.nonzero(b - a > tol)[0]
    if slow.size:
        n = int(np.ceil(np.log2(np.max((b - a)[slow] / tol[slow]))))
        sp = _padded(slow, _bucket(slow.size, dtype))
        gs = gf.at(sp)
        lo_, hi_ = mx.array(a[sp], dtype=dtype), mx.array(b[sp], dtype=dtype)
        pos = mx.array(fa[sp] > 0.0)
        for _ in range(n):
            mid = 0.5 * (lo_ + hi_)
            same = (gs(mid) > 0.0) == pos
            lo_, hi_ = mx.where(same, mid, lo_), mx.where(same, hi_, mid)
        a[slow], b[slow] = np.array(lo_)[:slow.size], np.array(hi_)[:slow.size]
    return 0.5 * (a + b)


class _Picker:
    """gf(delta): each kink's own event function at offsets delta (one per
    kink), with Q and the lane columns cols at the kinks' lanes -- compiled
    for frozen values (``frozen``), else live. gf.at(sub): the same for the
    kinks sub."""

    def __init__(self, Q, cols, f, frozen):
        self.Q, self.cols, self.f, self.frozen = Q, cols, f, frozen

    @classmethod
    def of(cls, Q, cols, lane, ff, frozen):
        ix = mx.array(lane.astype(np.int32))
        take = lambda v: mx.take(v, ix, axis=0)                     # noqa: E731
        return cls(_lanes(Q, take), [take(v) for v in cols], mx.array(ff.astype(np.int32)),
                   frozen)

    def at(self, sub):
        sx = mx.array(np.asarray(sub, np.int32))
        take = lambda v: mx.take(v, sx, axis=0)                     # noqa: E731
        return _Picker(_lanes(self.Q, take), [take(v) for v in self.cols], take(self.f),
                       self.frozen)

    def __call__(self, delta):
        if self.frozen:
            return _run(_pick, self.Q, *self.cols, self.f, delta)
        return _pick(self.Q, *self.cols, self.f, delta)


def _pieces(QL, cols, QH, cols_h, hw0, n_sample, n_gl, dtype, screen=None):
    """The contact rule's pieces, only those with width: per piece its lane
    (a host array) and its Gauss-Legendre offsets and normalised weights
    (pieces, n_gl). The kinks are found with the frozen parameters and
    moved with the live ones by one gradient-only Newton step t - [g(t) -
    sg(g(t))] / g'(t) on each kink's own function: the value stays the
    root, the gradient is the implicit one, -dg/dtheta / g', so the nodes
    move with the parameters and the gradient is that of the quadrature
    evaluated. QL, cols: the live values (``_prep``'s constants, the
    lane columns), QH, cols_h their frozen copies; screen: ``_screen``'s
    bounds, the event search then only on the lanes that may hold a kink."""
    m = cols[0].shape[0]
    sub = np.arange(m) if screen is None else _screen(QH, cols_h, hw0, screen, dtype)
    if sub.size == m:
        lane, ff, lo, hi = _brackets(QH, cols_h, hw0, n_sample, dtype)
    elif sub.size:
        sx = mx.array(sub.astype(np.int32))
        take = lambda v: mx.take(v, sx, axis=0)                     # noqa: E731
        lane, ff, lo, hi = _brackets(_lanes(QH, take), [take(v) for v in cols_h], hw0,
                                     n_sample, dtype)
        lane = sub[lane]
    else:
        lane = ff = lo = hi = np.zeros(0, dtype=np.int64)
    if lane.size:
        # the frozen search on a bucketed count of kinks, the last repeated
        nk, pad = lane.size, _padded(np.arange(lane.size), _bucket(lane.size, dtype))
        gf = _Picker.of(QH, cols_h, lane[pad], ff[pad], True)
        ev = _refine(gf, lo[pad], hi[pad], dtype)
        eps = (1e-7 if dtype == mx.float64 else 1e-3) * hw0
        evm = mx.array(ev, dtype=dtype)
        slope = np.array((gf(evm + eps) - gf(evm - eps)) / (2.0 * eps))[:nk]
        keep = np.nonzero(slope != 0.0)[0]
        lane, ff, ev, slope = lane[keep], ff[keep], ev[:nk][keep], slope[keep]
    if lane.size:
        gl = _Picker.of(QL, cols, lane, ff, False)(mx.array(ev, dtype=dtype))
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


def _photometry(Q, kind, large, cols, T):
    """Per-node contributions on points (m,) at offsets T (m, n): the shape
    columns (hybrid; a list) or F - 1 (quadratic). cols: the points' lane
    columns (dt, o, u), (m, 1) each; Q and kind may hold per-point (m, 1)
    columns (``_lanes``)."""
    xp, yp, fp, xm, ym, fm = _pos(Q, *(mx.broadcast_to(v, T.shape) for v in cols), T)
    rp, rs = Q["rp"], Q["rs"]
    full = lambda v: mx.broadcast_to(v, T.shape)
    with declared(large):
        if kind[0] == "hybrid":
            dp = moons.disc_shape_cols(xp, yp, rp, kind[1], fp)
            dm = moons.disc_shape_cols(xm, ym, rs, kind[1], fm)
        else:
            dp = [moons.disc_flux_quad(xp, yp, rp, kind[1], kind[2], fp)]
            dm = [moons.disc_flux_quad(xm, ym, rs, kind[1], kind[2], fm)]
    # the lens, where the discs overlap with both on the star (host choice)
    reads = [_sg(v) for v in (xp, yp, xm, ym, rp, rs)] + [fp, fm]
    mx.eval(*reads)                                                 # one sync for them all
    gx = [np.array(v) for v in reads[:4]]
    grp, grs = (np.broadcast_to(np.array(v), T.shape) for v in reads[4:6])
    zp_, zm_ = np.hypot(gx[0], gx[1]), np.hypot(gx[2], gx[3])
    dd = np.hypot(gx[2] - gx[0], gx[3] - gx[1])
    need = (np.array(reads[6]) & np.array(reads[7]) & (dd < (grp + grs) * (1.0 + 1e-9))
            & (zp_ < 1.0 + grp) & (zm_ < 1.0 + grs))
    # one disc wholly inside the other: the lens is the smaller disc, so the
    # union is the larger one, whose columns are already here (the lens is
    # C^1 at the internal tangency, so the switch leaves no kink)
    inside = need & (dd <= np.abs(grp - grs))
    if inside.any():
        big_p = mx.array(np.broadcast_to(grp >= grs, T.shape))
        ins = mx.array(inside)
        cols = [mx.where(ins, mx.where(big_p, a, b), a + b) for a, b in zip(dp, dm)]
    else:
        cols = [a + b for a, b in zip(dp, dm)]
    idx = np.nonzero((need & ~inside).ravel())[0]
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
    Q = _prep(P, dtype)
    QH = _host(Q)                                                   # frozen, no graph
    col = _lanes(QH, lambda v: v.reshape(-1, 1))                    # (n, 1) columns
    zB = np.broadcast_to(np.array(_run(_reach_z, col, dt_m[None], k_m[None])), (n, m))
    sel = np.nonzero((zB < reach[:, None]).reshape(-1))[0]
    out_cols = None
    if sel.size:
        six = mx.array(sel.astype(np.int32))
        pix = mx.array((sel % m).astype(np.int32))
        sets = n_sets is not None
        PO = {key: P[key] for key in ("t0_off", "p_off", "m_off", "per_ref", "mper_ref")}
        if not sets:
            QL, QHL, lkind = Q, QH, kind
        else:
            sx = mx.array((sel // m).astype(np.int32))
            gather = lambda v: mx.take(v, sx, axis=0)[:, None]
            QL, QHL, PO = _lanes(Q, gather), _lanes(QH, gather), _lanes(PO, gather)
            lkind = kind if kind[0] == "hybrid" else (kind[0],) + tuple(
                _map_sets(v, gather) for v in kind[1:])
        d_s, k_s, t_s = (mx.take(v, pix) for v in (dt_m, k_m, tu_m))
        lc = [d_s[:, None], *_lane_cols(PO, k_s, t_s, dtype)]       # (lanes, 1) each
        if exp_time > 0.0:
            # the pieces with width, each evaluated at its lane's values
            # the speeds bounding how far the bodies move in an exposure
            si = sel // m
            mper = host(P["mper_ref"] + P["m_off"])
            with np.errstate(divide="ignore", invalid="ignore"):
                v_m = (_TWO_PI * a_ps / mper
                       * np.sqrt((1.0 + e_m) / np.maximum(1.0 - e_m, 0.0)))
            screen = (rp_h[si], rs_h[si], float(P["speed"]), v_m[si])
            pl, T, W = _pieces(QL, lc, QHL, _host(lc), hw0, int(n_sample or 64), n_gl, dtype,
                               screen)
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
            Qj, kj = QL, lkind
            if sets:
                Qj = _lanes(QL, take)
                if lkind[0] == "quad":
                    kj = (lkind[0],) + tuple(_map_sets(v, take) for v in lkind[1:])
            cols = _photometry(Qj, kj, large, [take(v) for v in lc], T[j])
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
