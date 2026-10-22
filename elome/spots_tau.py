"""Starspots on a rotating star, crossed by a planet on an orbit.

``TransitModel`` with ``params.spots`` delegates here. The light curve is
``spots.shape_cols_spots`` at the planet's sky position and the spots'
positions at each time, normalised to the unspotted star: out of transit
it is the spots' rotational modulation, in transit the crossing as well.

**Frames.** The sky frame is ``oblate_tau``'s: X along the planet's motion
at transit, Y the sky projection of the orbit normal, Z towards the
observer (the planet transits at Y = -b). The stellar spin axis is

    s = (sin i* sin lam, sin i* cos lam, cos i*)

-- Fabrycky & Winn (2009)'s convention for the sky-projected obliquity
lam, measured clockwise on the sky from the projected orbit normal to the
projected spin axis, so a published Rossiter-McLaughlin lambda is this
lam; i* is the spin axis's inclination to the line of sight. The star
turns about s (prograde: lam = 0, i* = 90 degrees turns its visible face
along +X, with the planet). A spot at stellar latitude phi_s and longitude
L is at

    n = cos(phi_s) [cos(L + Omega t) e_obs + sin(L + Omega t) e_east]
        + sin(phi_s) s,

e_obs = (-cos i* sin lam, -cos i* cos lam, sin i*) the meridian facing
the observer, e_east = s x e_obs = (cos lam, -sin lam, 0), Omega = 2 pi /
P_rot and t the time since the spots' reference epoch: longitude 0 faces
the observer then, and longitudes increase in the sense of rotation.

**Times.** The planet sees tau, the time since each point's own
mid-transit; the spots see t - epoch, absolute. On the far side of the
orbit the planet is moved off the star and the spots are still evaluated.

**Exposures.** Instantaneous, or the contact rule (``exp_time`` > 0):
each exposure window is split at the light curve's kinks inside it -- at
most _K_SPLIT, earliest first -- and each piece integrated by
Gauss-Legendre. The kinks are the planet's limb contacts (closed form, as
the spherical rule's) and the times at which the number of planet-rim
corners on a spot's visible rim changes: the planet touching a rim, or
an edge of the planet passing a rim's limb point. The star turns between
transits, so these are found per exposure, not per orbit: the count is
sampled at n_sample + 1 times across the window, spot by spot, and each
change bisected. The step should be shorter than the planet takes to move
a fraction of its radius (``sample_count``): a spot smaller than the
planet is wholly covered for a while mid-crossing, and a pair of changes
closer than a step that cancel is missed -- a brief graze, or a spot
barely smaller than the planet, whose effect on the exposure is as
small. Two
slow kinks are not split: a rim reaching the limb (a spot rotating onto
or off the disc) and two spots starting to overlap, both on the rotation
time scale with a curvature, not a slope, discontinuity.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .exposure import contact_offsets, contact_offsets_anchored, gauss_legendre
from .hybrid import combine_cols, get_law
from .oblate import _F_SW, _mod2pi
from .oblate_tau import _contact_kh, _in_blocks, _orbit_consts, _sky
from .rings import _crossings
from .spots import _Rim, shape_cols_spots
from .trig import sincos

__all__ = ["spin_frame", "spot_vectors", "flux_dev_spots_t"]

_TWO_PI = 2.0 * math.pi
# evaluation points per block of the graph (oblate_tau._in_blocks): the
# columns hold their fp64 intermediates until evaluated
_SPOT_BLOCK = 1 << 14
_K_SPLIT = 6                                      # split points per exposure
_S_MIN, _S_MAX = 8, 256                           # count samples per exposure
_E_SPOT = 4                                       # changes bisected per spot
_N_BISECT = {mx.float64: 30, mx.float32: 16}


def spin_frame(istar, lam):
    """(s, e_obs, e_east), each a tuple of three components: the spin
    axis, the meridian facing the observer and east, from the spin axis's
    inclination to the line of sight and the sky-projected obliquity
    (radians; mx arrays)."""
    si, ci = sincos(istar)
    sl, cl = sincos(lam)
    zero = 0.0 * si * sl
    return ((si * sl, si * cl, ci), (-(ci * sl), -(ci * cl), si), (cl + zero, -sl, zero))


def spot_vectors(ts, prot, istar, lam, lon, lat):
    """(..., K, 3) unit centre vectors of K spots at times ts (since the
    spots' epoch), on a star with rotation period prot, spin inclination
    istar and sky-projected obliquity lam; lon, lat (..., K) radians.
    ts, prot, istar and lam broadcast against the leading axes."""
    ex = lambda v: v[..., None]
    return _spot_xyz(ex(ts), ex(prot), ex(istar), ex(lam), lon, lat)


def _spot_xyz(ts, prot, istar, lam, lon, lat):
    """spot_vectors elementwise: every argument broadcast, (..., 3) out."""
    (sx, sy, sz), (ox, oy, oz), (ex, ey, ez) = spin_frame(istar, lam)
    ph = lon + _TWO_PI * ts / prot
    sp, cp = sincos(ph)
    sb, cb = sincos(lat)
    u, v = cb * cp, cb * sp
    return mx.stack([u * ox + v * ex + sb * sx, u * oy + v * ey + sb * sy,
                     u * oz + v * ez + sb * sz], axis=-1)


def _corners(X, Y, front, rp, n, alpha, dt):
    """Planet-rim corners: (rim, [psi], [on the visible rim]) for the four
    candidates, elementwise: X, Y, front the planet, n (..., 3) the spot.
    Frozen inputs."""
    g = _Rim(n[..., 0], n[..., 1], n[..., 2], alpha, dt)
    px = X * g.ux + Y * g.uy
    py = X * g.vx + Y * g.vy
    ab = mx.abs(g.b)
    circ = (g.a - ab) <= _F_SW[dt] * g.a
    phis, valid = _crossings(-px / rp, (g.y0 - py) / rp, g.a / rp, ab / rp, circ, dt)
    psis, vis = [], []
    for ph, v in zip(phis, valid):
        psi = _mod2pi(mx.where(g.nz < 0, -ph, ph))
        mu = g.mu_a + g.mu_b * sincos(psi)[0]
        psis.append(psi)
        vis.append(v & (mu >= 0.0) & front)
    return g, psis, vis


def _corner_count(X, Y, front, rp, n, alpha, dt):
    """Planet-rim corners on the spot's visible rim (mu >= 0)."""
    _, _, vis = _corners(X, Y, front, rp, n, alpha, dt)
    return sum(v.astype(mx.int32) for v in vis)


def sample_count(exp_time, a, per, rp):
    """Count samples per exposure for the contact rule's event search: a
    step of at most a quarter of the time the planet takes to cross its
    own radius at transit, within [_S_MIN, _S_MAX]. Host values."""
    v = _TWO_PI * float(a) / float(per)                       # radii per unit time
    s = math.ceil(4.0 * float(exp_time) * v / max(float(rp), 1e-6))
    return int(min(max(s, _S_MIN), _S_MAX))


def _spot_events(t1, t2, cols, sp, dt, n_sample):
    """(…, _E_SPOT * K) times inside [t1, t2] at which a spot's visible
    corner count changes, +inf where none: sampled, then bisected, spot by
    spot (the spot axis is a batch axis). ``cols(nk)`` gives the per-set
    parameters reshaped for nk trailing axes.

    The times carry the implicit derivative of the event in the
    parameters (value unchanged), so the contact rule's gradient is that of
    the quadrature it evaluates. An event is a root of q = |P - C|^2 - rp^2,
    P a rim point and C the planet's centre: at a tangency (two corners
    merge, the count changing by 2) P is the touching point, at which
    dq/dpsi = 0, so q's partial derivatives at that frozen psi are its
    total ones; where a corner passes the rim's limb point (the count
    changing by 1) P is that limb point, live. Then dt_e = -q_theta / q_t."""
    sg = mx.stop_gradient
    t1, t2 = sg(t1), sg(t2)
    K = sp.shape[-2]
    s4 = sp[:, None, None]                    # (n, 1, 1, K, 4)

    def geom(tt, live):
        f = (lambda v: v) if live else sg
        t0, ep, per, a, b, r, prot, istar, lam, eo = cols(3)
        t0, ep, per, a, b, r, prot, istar, lam = (f(v) for v in (t0, ep, per, a, b, r, prot,
                                                                istar, lam))
        eo = None if eo is None else (f(eo[0]), tuple(f(c) for c in eo[1]))
        tau = tt - t0
        tau = tau - per * mx.round(tau / per)
        X, Y, _, _, front = _sky((_TWO_PI / per) * tau, a, b, eo)
        nv = _spot_xyz(tt - ep, prot, istar, lam, f(s4[..., 0]), f(s4[..., 1]))
        return X, Y, front, r, nv, f(s4[..., 2])

    def count(tt):                           # tt (n, m, J, K) -> (n, m, J, K)
        return _corner_count(*geom(tt, False), dt)

    S = int(n_sample)
    j = mx.array(np.linspace(0.0, 1.0, S + 1), dtype=dt)
    ts = t1[..., None] + (t2 - t1)[..., None] * j                    # (n, m, S+1)
    ts = mx.broadcast_to(ts[..., None], ts.shape + (K,))
    cnt = count(ts)
    change = cnt[..., 1:, :] != cnt[..., :-1, :]                     # (n, m, S, K)
    idx = mx.arange(S)[:, None]
    key = mx.sort(mx.where(change, idx, S), axis=-2)[..., :_E_SPOT, :]
    ok = key < S
    kk = mx.minimum(key, S - 1)
    lo = mx.take_along_axis(ts[..., :-1, :], kk, axis=-2)
    hi = mx.take_along_axis(ts[..., 1:, :], kk, axis=-2)
    clo = mx.take_along_axis(cnt[..., :-1, :], kk, axis=-2)
    for _ in range(_N_BISECT[dt]):
        mid = 0.5 * (lo + hi)
        same = count(mid) == clo
        lo = mx.where(same, mid, lo)
        hi = mx.where(same, hi, mid)
    te = sg(0.5 * (lo + hi))

    # ---- the event's geometry (frozen): its kind and its point on the rim
    chi = count(hi)
    tangency = mx.abs(chi - clo) == 2
    more = mx.where(chi > clo, hi, lo)
    g, psis, vis = _corners(*geom(more, False), dt)
    best = mx.full(te.shape, mx.inf, dtype=dt)
    psi_t = 0.0 * te
    for i in range(4):
        for jj in range(i + 1, 4):
            dpsi = _mod2pi(psis[jj] - psis[i] + math.pi) - math.pi
            dist = mx.where(vis[i] & vis[jj], mx.abs(dpsi), mx.inf)
            closer = dist < best
            best = mx.where(closer, dist, best)
            psi_t = mx.where(closer, psis[i] + 0.5 * dpsi, psi_t)

    def q_at(tt, live, which=None):
        X, Y, front, r, nv, al = geom(tt, live)
        rim = _Rim(nv[..., 0], nv[..., 1], nv[..., 2], al, dt)
        pl = (rim.psi_limb, math.pi - rim.psi_limb)
        if which is None:
            return [_q(rim, X, Y, r, p) for p in pl]
        psi = mx.where(tangency, sg(psi_t), mx.where(which, pl[1], pl[0]))
        return _q(rim, X, Y, r, psi)

    q1, q2 = q_at(te, False)
    which = mx.abs(q2) < mx.abs(q1)                 # the limb point the corner passed
    q_live = q_at(te, True, which)
    _, q_t = mx.jvp(lambda tt: q_at(tt, False, which), [te], [mx.ones_like(te)])
    q_t = sg(q_t[0])
    good = ok & (mx.abs(q_t) > 0.0)
    shift = (q_live - sg(q_live)) / mx.where(good, q_t, 1.0)

    ev = mx.where(ok, te - mx.where(good, shift, 0.0), mx.inf)
    return mx.reshape(ev, ev.shape[:-2] + (_E_SPOT * K,))


def _events_fn(dt, n_sample, ecc, ew):
    """_spot_events as a function of arrays: (t1, t2, t0, epoch, period, a,
    b, r, prot, istar, lam, spots4[, k, h]) -> events. Compiled once per
    configuration on the CPU stream (eager, the event search's many small
    ops cost more than the columns it splits); eager on the GPU."""
    def raw(t1, t2, t0, epoch, period, a, b, r, prot, istar, lam, spots4, *kh):
        k, h = kh if ecc else (None, None)

        def cols(nk):
            sh = (-1,) + (1,) * nk
            c = [mx.reshape(v, sh) for v in (t0, epoch, period, a, b, r, prot, istar, lam)]
            eo = None if k is None else _orbit_consts(c[3], c[4], mx.reshape(k, sh),
                                                      mx.reshape(h, sh), ew)
            return c + [eo]
        return _spot_events(t1, t2, cols, spots4, dt, n_sample)
    if M._gpu_stream_active():
        return raw
    key = (dt, int(n_sample), bool(ecc), bool(ew))
    fn = _EVENTS.get(key)
    if fn is None:
        fn = _EVENTS[key] = mx.compile(raw)
    return fn


_EVENTS: dict = {}


def _q(rim, X, Y, r, psi):
    """|P(psi) - C|^2 - r^2 for the rim point P and the planet's centre C."""
    x, y, _ = rim.sky(psi)
    return (x - X) ** 2 + (y - Y) ** 2 - r * r


def _contact_events(tt, t0, per, a, b, r, k, h, ew):
    """The planet's four limb contacts nearest each time, (n, m, 4), from
    the closed-form phase offsets -- live in the parameters, as the
    spherical rule's."""
    sg = mx.stop_gradient
    if k is None:
        cA = contact_offsets(r, a, b)
    else:
        ci, consts = _orbit_consts(a, b, k, h, ew)
        kc, hc = _contact_kh(k, h, ew)
        cA = contact_offsets_anchored(r, a, b, kc, hc, ci, consts=consts)
    tc = t0 + per * sg(mx.round((tt - t0) / per))                    # the epoch's centre
    return mx.stack([tc + c * (per / _TWO_PI) for c in cA], axis=-1)


def _exposure_nodes(tt, exp_time, events, n_gl, dt):
    """Gauss-Legendre nodes and weights over [t - e/2, t + e/2] split at
    the nearest _K_SPLIT events inside it: (times, weights), a trailing
    axis of (_K_SPLIT + 1) * n_gl, weights summing to 1."""
    xg, wg = gauss_legendre(n_gl)
    xg, wg = mx.array(xg, dtype=dt), mx.array(wg, dtype=dt)
    half = 0.5 * exp_time
    t1, t2 = tt - half, tt + half
    inside = (events > t1[..., None]) & (events < t2[..., None])
    cand = mx.where(inside, events, mx.inf)
    # sorted by argsort and gathered: the event times carry gradients
    order = mx.stop_gradient(mx.argsort(cand, axis=-1))[..., :_K_SPLIT]
    key = mx.take_along_axis(cand, order, axis=-1)
    edges = [t1] + [mx.where(mx.isinf(key[..., j]), t2, key[..., j])
                    for j in range(_K_SPLIT)] + [t2]
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


def _tau_graph(tt, t0, epoch, period, a, b, r, prot, istar, lam, spots4, w2d, law,
               basis=False, k=None, h=None, ew=False, exp_time=0.0, n_gl=5,
               n_sample=_S_MIN):
    """F - 1 (or the basis columns) at times tt (n, m): per-set (n,)
    parameters, spots4 (n, K, 4) of (lon, lat, alpha [rad], contrast).
    ``exp_time`` > 0: the contact rule, n_gl nodes per piece and n_sample
    count samples per exposure."""
    dt = tt.dtype

    def cols(nk):
        sh = (-1,) + (1,) * nk
        c = [mx.reshape(p, sh) for p in (t0, epoch, period, a, b, r, prot, istar, lam)]
        eo = None if k is None else _orbit_consts(c[3], c[4], mx.reshape(k, sh),
                                                  mx.reshape(h, sh), ew)
        return c + [eo]

    def inst(T, nk):
        t0c, ep, per, av, bv, rv, prot_, istar_, lam_, eo = cols(nk)
        tau = T - t0c
        tau = tau - per * mx.round(tau / per)
        X, Y, _, _, front = _sky((_TWO_PI / per) * tau, av, bv, eo)
        far = 2.0 + 2.0 * mx.stop_gradient(rv)                   # clear of the star
        X = mx.where(front, X, far)
        Y = mx.where(front, Y, 0.0)
        sp = mx.reshape(spots4, (spots4.shape[0],) + (1,) * nk + spots4.shape[-2:])
        n = spot_vectors(T - ep, prot_, istar_, lam_, sp[..., 0], sp[..., 1])
        return shape_cols_spots(X, Y, rv, n, sp[..., 2], sp[..., 3], law)

    if exp_time == 0.0:
        Bc = inst(tt, 1)
    else:
        half = 0.5 * exp_time
        ev_fn = _events_fn(dt, n_sample, k is not None, ew)
        sp_ev = ev_fn(tt - half, tt + half, t0, epoch, period, a, b, r, prot, istar, lam,
                      spots4, *(() if k is None else (k, h)))
        c1 = mx.reshape(t0, (-1, 1))
        pc = [mx.reshape(v, (-1, 1)) for v in (period, a, b, r)]
        kh = (None, None) if k is None else (mx.reshape(k, (-1, 1)), mx.reshape(h, (-1, 1)))
        ct_ev = _contact_events(tt, c1, *pc, *kh, ew)
        T, W = _exposure_nodes(tt, exp_time, mx.concatenate([ct_ev, sp_ev], axis=-1),
                               int(n_gl), dt)
        Bc = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


def _spots_ok(spots4):
    """The spots' domain as a mask over sets (arrays are never read back:
    a traced value outside it gives NaN)."""
    lat, al, c = spots4[..., 1], spots4[..., 2], spots4[..., 3]
    ok = (al > 0.0) & (al <= 0.5 * math.pi) & (mx.abs(lat) <= 0.5 * math.pi) & (c >= 0.0)
    return mx.all(ok, axis=-1)


def _check_spots(spots4):
    """Host values only: (K, 4) or (n, K, 4) of (lon, lat, alpha [rad],
    contrast)."""
    if isinstance(spots4, mx.array):
        return
    s = np.asarray(spots4, dtype=np.float64)
    if s.ndim not in (2, 3) or s.shape[-1] != 4:
        raise ValueError("spots must be (K, 4) or (n_sets, K, 4): longitude, "
                         "latitude, angular radius, contrast")
    if not np.all(np.isfinite(s)):
        raise ValueError("spots must be finite")
    al, lat, c = s[..., 2], s[..., 1], s[..., 3]
    if np.any(al <= 0.0) or np.any(al > 0.5 * math.pi):
        raise ValueError("a spot's angular radius must be in (0, 90] degrees")
    if np.any(np.abs(lat) > 0.5 * math.pi):
        raise ValueError("a spot's latitude must be in [-90, 90] degrees")
    if np.any(c < 0.0):
        raise ValueError("a spot's contrast must be >= 0")


@M.fp64_on_cpu
def flux_dev_spots_t(t, t0, epoch, period, a, b, r, prot, istar, lam, spots4, law, u,
                     basis=False, k=None, h=None, ew=False, exp_time=0.0, n_gl=5,
                     n_sample=None):
    """F - 1 of a planet crossing a spotted star, at times t ((m,) or
    (n, m)), normalised to the unspotted star. Per-set parameters are
    scalars or (n,) / (n, 1) arrays: t0 (mid-transit), epoch (the spots'
    reference time), period, a, b (impact parameter at conjunction), r,
    prot, istar and lam (radians); spots4 (K, 4) or (n, K, 4) of
    (longitude, latitude, angular radius [rad], contrast). (k, h) =
    sqrt(e) (cos w, sin w) for an eccentric orbit, or (e, w [rad]) with
    ``ew``. ``exp_time`` > 0: each time an exposure of that length,
    integrated by the contact rule with ``n_gl`` nodes per piece and
    ``n_sample`` count samples per exposure for its event search (None:
    ``sample_count`` from host values of a, period and r, else 64; see the
    module notes). Float64 on the CPU stream unless the data are
    float32."""
    law = get_law(law)
    if t.ndim not in (1, 2):
        raise ValueError(f"t must be (m,) or (n, m); got {t.shape}")
    _check_spots(spots4)
    if n_sample is None and exp_time != 0.0:
        try:
            host = [np.asarray(v, dtype=np.float64) for v in (a, period, r)]
            n_sample = max(sample_count(exp_time, x, P, y)
                           for x, P, y in np.broadcast(*host))
        except (TypeError, ValueError, RuntimeError):
            n_sample = 64
    dt = t.dtype
    squeeze = t.ndim == 1
    t2d = t[None, :] if squeeze else t
    ecc = k is not None
    params = (t0, epoch, period, a, b, r, prot, istar, lam) + ((k, h) if ecc else ())
    sp = spots4 if isinstance(spots4, mx.array) else mx.array(
        np.asarray(spots4, dtype=np.float64), dtype=dt)
    sp = M._as_dtype(sp, dt)
    n_param = max([(p.reshape(-1).shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                    else 1) for p in params] + [sp.shape[0] if sp.ndim == 3 else 1])
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(t2d.shape[0], n_param)
    pc = [M._canon_param(p, n, dt) for p in params]
    if t2d.shape[0] != n:
        t2d = mx.broadcast_to(t2d, (n, t2d.shape[1]))
    sp = mx.broadcast_to(sp if sp.ndim == 3 else sp[None], (n,) + sp.shape[-2:])
    kh = pc[-2:] if ecc else [None, None]
    from .metal_hybrid import _canon_w
    w2d = None if basis else _canon_w(u, n, law.n_w, dt, law.name)

    def run(tt, rows):
        *p9, s4, w, kk, hh = rows
        return _tau_graph(tt, *p9, s4, w, law, basis=basis, k=kk, h=hh, ew=ew,
                          exp_time=exp_time, n_gl=n_gl, n_sample=n_sample or _S_MIN)
    nodes = 1 if exp_time == 0.0 else (_K_SPLIT + 1) * int(n_gl)
    out = _in_blocks(run, t2d, [*pc[:9], sp, w2d, *kh], nodes, _SPOT_BLOCK)
    ok = _spots_ok(sp)[:, None]
    out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    return out[0] if squeeze and n == 1 else out
