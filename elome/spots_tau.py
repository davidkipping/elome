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

__all__ = ["spin_frame", "spot_vectors", "flux_dev_spots_tau", "flux_dev_spots_t"]

_TWO_PI = 2.0 * math.pi
# evaluation points per block of the graph (oblate_tau._in_blocks): the
# columns hold their fp64 intermediates until evaluated
_SPOT_BLOCK = 1 << 14
_SPOT_BLOCK_KERNEL = 1 << 21                      # the kernels hold nothing per point
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


def _spot_events(t1, t2, off, cols, sp, dt, n_sample):
    """(…, _E_SPOT * K) times inside [t1, t2] at which a spot's visible
    corner count changes, +inf where none: sampled, then bisected, spot by
    spot (the spot axis is a batch axis; _search_events), then made live
    (_live_events). Times are the planet's (since mid-transit); the star's
    clock is ahead by ``off`` (n, m). ``cols(nk)`` gives the per-set
    parameters reshaped for nk trailing axes.

    The times carry the implicit derivative of the event in the
    parameters (value unchanged), so the contact rule's gradient is that of
    the quadrature it evaluates. An event is a root of q = |P - C|^2 - rp^2,
    P a rim point and C the planet's centre: at a tangency (two corners
    merge, the count changing by 2) P is the touching point, at which
    dq/dpsi = 0, so q's partial derivatives at that frozen psi are its
    total ones; where a corner passes the rim's limb point (the count
    changing by 1) P is that limb point, live. Then dt_e = -q_theta / q_t."""
    ev = _live_events(*_search_events(t1, t2, off, cols, sp, dt, n_sample),
                      _event_geom(cols, sp, off), dt)
    return mx.reshape(ev, ev.shape[:-2] + (-1,))


def _search_events(t1, t2, off, cols, sp, dt, n_sample):
    """The event search, frozen: (te, ok, tangency, psi_t, which), each
    (..., _E_SPOT, K) -- the times, their validity, their kind, a
    tangency's touching rim parameter and the limb point a corner passed
    (see _live_events)."""
    sg = mx.stop_gradient
    t1, t2 = sg(t1), sg(t2)
    K = sp.shape[-2]
    geom = _event_geom(cols, sp, off)

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

    X, Y, _, r, nv, al = geom(te, False)
    rim = _Rim(nv[..., 0], nv[..., 1], nv[..., 2], al, dt)
    q1, q2 = (_q(rim, X, Y, r, p) for p in (rim.psi_limb, math.pi - rim.psi_limb))
    which = mx.abs(q2) < mx.abs(q1)                 # the limb point the corner passed
    return te, ok, tangency, sg(psi_t), which


def _event_geom(cols, sp, off):
    """geom(tt, live) -> (X, Y, front, r, n, alpha) of the planet and the
    spots at planet times tt (n, m, J, K), the star's clock ahead by off
    (n, m); frozen unless ``live``. ``cols(nk)`` the per-set parameters
    for nk trailing axes, sp (n, K, 4)."""
    sg = mx.stop_gradient
    s4 = sp[:, None, None]                    # (n, 1, 1, K, 4)
    o4 = off[..., None, None]                 # (n, m, 1, 1)

    def geom(tt, live, rates=False):
        f = (lambda v: v) if live else sg
        per, a, b, r, prot, istar, lam, eo = cols(3)
        per, a, b, r, prot, istar, lam = (f(v) for v in (per, a, b, r, prot, istar, lam))
        eo = None if eo is None else (f(eo[0]), tuple(f(c) for c in eo[1]))
        X, Y, dX, dY, front = _sky((_TWO_PI / per) * tt, a, b, eo)
        nv = _spot_xyz(tt + f(o4), prot, istar, lam, f(s4[..., 0]), f(s4[..., 1]))
        out = (X, Y, front, r, nv, f(s4[..., 2]))
        if rates:
            # the planet's sky velocity and the star's angular velocity
            om = _TWO_PI / prot
            out += ((_TWO_PI / per) * dX, (_TWO_PI / per) * dY,
                    tuple(om * c for c in spin_frame(istar, lam)[0]))
        return out
    return geom


def _live_events(te, ok, tangency, psi_t, which, geom, dt):
    """The event times te (frozen) with their implicit derivative in the
    parameters (see _spot_events): ``tangency`` an event's kind, ``psi_t``
    a tangency's touching rim parameter, ``which`` the limb point (pi -
    psi_limb if True) a corner passed; +inf where not ``ok``."""
    sg = mx.stop_gradient

    def q_at(tt, live):
        X, Y, front, r, nv, al = geom(tt, live)
        rim = _Rim(nv[..., 0], nv[..., 1], nv[..., 2], al, dt)
        psi = mx.where(tangency, sg(psi_t),
                       mx.where(which, math.pi - rim.psi_limb, rim.psi_limb))
        return _q(rim, X, Y, r, psi)
    q_live = q_at(te, True)
    q_t = _q_rate(te, tangency, psi_t, which, geom, dt)
    good = ok & (mx.abs(q_t) > 0.0)
    shift = (q_live - sg(q_live)) / mx.where(good, q_t, 1.0)
    return mx.where(ok, te - mx.where(good, shift, 0.0), mx.inf)


def _make_cols(params, k, h, ew):
    """cols(nk): the per-set (period, a, b, r, prot, istar, lam) reshaped
    for nk trailing axes, and the orbit constants (None if circular)."""
    def cols(nk):
        sh = (-1,) + (1,) * nk
        c = [mx.reshape(v, sh) for v in params]
        eo = None if k is None else _orbit_consts(c[1], c[2], mx.reshape(k, sh),
                                                  mx.reshape(h, sh), ew)
        return c + [eo]
    return cols


def _events_fn(dt, n_sample, ecc, ew):
    """The graph's event search and its live correction, as a function of
    arrays: (t1, t2, off, period, a, b, r, prot, istar, lam, spots4[, k,
    h]) -> (live events, their frozen copy or None). The search runs on
    host copies of its inputs in chunks of exposures, each evaluated before
    the next, and the live correction follows it on every exposure at
    once. (Compiling the search trips an MLX 0.32 compile error,
    "unordered_map::at"; chunked and evaluated it is as fast.)"""
    def split(args):
        params, spots4, kh = args[:7], args[7], args[8:]
        k, h = kh if ecc else (None, None)
        return params, spots4, k, h

    def fn(t1, t2, off, *args):
        n, m = t1.shape
        K = args[7].shape[-2]
        B = max(1, _EVENT_SAMPLES // ((int(n_sample) + 1) * K))
        ins, host = _host_copies((t1, t2, off, *args))
        parts = []
        for rows, times in _chunks(n, m, B):
            params, spots4, k, h = split([v[rows] for v in ins[3:]])
            out = _search_events(ins[0][rows, times], ins[1][rows, times],
                                 ins[2][rows, times], _make_cols(params, k, h, ew), spots4,
                                 dt, n_sample)
            if host:
                mx.eval(*out)    # free each chunk's search before the next
            parts.append((rows, out))
        frozen = _assemble(parts, n, m)
        params, spots4, k, h = split(args)
        ev = _live_events(*frozen, _event_geom(_make_cols(params, k, h, ew), spots4, off), dt)
        fz = mx.where(frozen[1], frozen[0], mx.inf) if host else None
        flat = lambda v: mx.reshape(v, v.shape[:-2] + (-1,))     # noqa: E731
        return flat(ev), (None if fz is None else flat(fz))
    return fn


def _host_copies(vs):
    """(fresh host copies of the arrays vs, True), or, under mx.compile or
    vmap where nothing can be read, (stop-gradient views, False). Inside a
    traced gradient an array computed from traced ones keeps its graph,
    and evaluating it early (a search, a mask read back) makes the
    backward pass compute it again; computed from copies it is a constant."""
    try:
        return [mx.array(np.asarray(mx.stop_gradient(v)), dtype=v.dtype) for v in vs], True
    except (RuntimeError, ValueError):
        return [mx.stop_gradient(v) for v in vs], False


# exposures per chunk of the graph's event search: (n_sample + 1) * K count
# samples each, about 2 KB a sample in float64 at its peak
_EVENT_SAMPLES = 1 << 19


def _chunks(n, m, B):
    """(rows, times) slices covering (n, m) in chunks of at most B
    exposures: whole rows while they fit, else one row's times."""
    if n * m <= B:
        return [(slice(0, n), slice(0, m))]
    if m <= B:
        rb = B // m
        return [(slice(i, min(i + rb, n)), slice(0, m)) for i in range(0, n, rb)]
    return [(slice(i, i + 1), slice(j, min(j + B, m))) for i in range(n) for j in range(0, m, B)]


def _assemble(parts, n, m):
    """The chunks' outputs (each a tuple of (rows, times, ...) arrays) as
    full (n, m, ...) arrays, in _chunks' order."""
    if len(parts) == 1:
        return parts[0][1]
    nout = len(parts[0][1])
    rows, cur, r0 = [], [], None
    for sl_rows, out in parts:
        if r0 is not None and sl_rows.start != r0:
            rows.append(cur)
            cur = []
        r0 = sl_rows.start
        cur.append(out)
    rows.append(cur)
    return tuple(mx.concatenate([mx.concatenate([c[q] for c in row], axis=1) for row in rows],
                                axis=0) for q in range(nout))


def _kernel_events(tt, off, half, period, a, b, r, prot, istar, lam, spots4, law,
                   n_sample, k=None, h=None, ew=False):
    """_events_fn's route on the fp32 kernels, for exposures centred on
    planet times tt (n, m), the star's clock ahead by off: the search in
    one launch (``metal_spots.spot_events_kernel``), on host copies of its
    inputs, the live correction (_live_events) as graph operations. (live,
    frozen) events, frozen None under mx.compile or vmap."""
    from .metal_spots import spot_events_kernel
    dt = tt.dtype
    params = (period, a, b, r, prot, istar, lam)
    hv, host = _host_copies([tt, off, *params, spots4] + ([] if k is None else [k, h]))
    ht, hoff, hper, ha, hb, hr, hprot, histar, hlam, hsp = hv[:10]
    if k is None:
        orbit, shape = "circ", hb
    else:
        from .anchored import pack_orbit_constants
        ci, consts = _orbit_consts(ha, hb, hv[10], hv[11], ew)
        orbit, shape = "ecc", pack_orbit_constants(None, None, ci, consts=consts)
    te, code, psit = spot_events_kernel(ht, hoff, half, hper, ha, shape, hr, hprot, histar,
                                        hlam, hsp, law, n_sample, orbit)
    te, code, psit = (mx.transpose(v, (0, 1, 3, 2)) for v in (te, code, psit))
    ok = code > 0.0
    te = mx.where(ok, te, ht[..., None, None])        # finite geometry where none
    geom = _event_geom(_make_cols(params, k, h, ew), spots4, off)
    ev = _live_events(te, ok, code == 1.0, psit, code == 3.0, geom, dt)
    flat = lambda v: mx.reshape(v, v.shape[:-2] + (-1,))         # noqa: E731
    return flat(ev), (flat(mx.where(ok, te, mx.inf)) if host else None)


def _q_rate(te, tangency, psi_t, which, geom, dt):
    """dq/dt at an event, frozen, in closed form (an mx.jvp here, inside a
    traced gradient, breaks the VJP of every compiled function called
    after it in MLX 0.32). The rim point P turns with the star, dP/dt =
    Omega s x P: at a tangency its slide along the rim does not change q
    (dq/dpsi = 0 there); a limb point slides along the rim to stay at
    mu = 0, by T dpsi with T = dP/dpsi and T_z dpsi = -(Omega s x P)_z."""
    X, Y, _, r, nv, al, Xd, Yd, (wx, wy, wz) = geom(te, False, rates=True)
    g = _Rim(nv[..., 0], nv[..., 1], nv[..., 2], al, dt)
    psi = mx.where(tangency, psi_t, mx.where(which, math.pi - g.psi_limb, g.psi_limb))
    px, py, pz = g.sky(psi)
    ux, uy, uz = wy * pz - wz * py, wz * px - wx * pz, wx * py - wy * px
    sp, cp = sincos(psi)
    tx = g.sa * (-sp * g.ux + cp * g.nz * g.vx)
    ty = g.sa * (-sp * g.uy + cp * g.nz * g.vy)
    tz = -g.sa * cp * g.nxy
    ok = (~tangency) & (mx.abs(tz) > 0.0)
    slide = mx.where(ok, -uz / mx.where(ok, tz, 1.0), 0.0)
    ux, uy = ux + tx * slide, uy + ty * slide
    return mx.stop_gradient(2.0 * ((px - X) * (ux - Xd) + (py - Y) * (uy - Yd)))


def _q(rim, X, Y, r, psi):
    """|P(psi) - C|^2 - r^2 for the rim point P and the planet's centre C."""
    x, y, _ = rim.sky(psi)
    return (x - X) ** 2 + (y - Y) ** 2 - r * r


def _contact_events(tt, per, a, b, r, k, h, ew):
    """The planet's four limb contacts around the transit nearest each
    planet time, (n, m, 4), from the closed-form phase offsets -- live in
    the parameters, as the spherical rule's."""
    sg = mx.stop_gradient
    if k is None:
        cA = contact_offsets(r, a, b)
    else:
        ci, consts = _orbit_consts(a, b, k, h, ew)
        kc, hc = _contact_kh(k, h, ew)
        cA = contact_offsets_anchored(r, a, b, kc, hc, ci, consts=consts)
    tc = per * sg(mx.round(tt / per))                                # the epoch's centre
    return mx.stack([tc + c * (per / _TWO_PI) for c in cA], axis=-1)


def _exposure_pieces(tt, exp_time, events):
    """[t - e/2, t + e/2] split at the nearest _K_SPLIT events inside it:
    the pieces' midpoints and half-widths, a trailing axis of
    _K_SPLIT + 1 (empty pieces have half-width 0)."""
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
    lo, hi = mx.stack(edges[:-1], axis=-1), mx.stack(edges[1:], axis=-1)
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def _piece_nodes(mid, hw, exp_time, n_gl, dt):
    """Gauss-Legendre nodes and weights over the pieces (mid, hw), n_gl
    per piece along the trailing axis (unnormalised: they sum to about 1
    over a whole exposure)."""
    xg, wg = gauss_legendre(n_gl)
    xg, wg = mx.array(xg, dtype=dt), mx.array(wg, dtype=dt)
    T = mid[..., None] + hw[..., None] * xg
    W = hw[..., None] * wg / exp_time
    sh = mid.shape[:-1] + (mid.shape[-1] * n_gl,)
    return mx.reshape(T, sh), mx.reshape(W, sh)


def _exposure_nodes(tt, exp_time, events, n_gl, dt):
    """Gauss-Legendre nodes and weights over [t - e/2, t + e/2] split at
    the nearest _K_SPLIT events inside it: (times, weights), a trailing
    axis of (_K_SPLIT + 1) * n_gl, weights summing to 1."""
    T, W = _piece_nodes(*_exposure_pieces(tt, exp_time, events), exp_time, n_gl, dt)
    return T, W / mx.sum(W, axis=-1, keepdims=True)


def _tau_graph(tt, ts, period, a, b, r, prot, istar, lam, spots4, w2d, law,
               basis=False, k=None, h=None, ew=False, exp_time=0.0, mode=M._INT_CONTACT,
               n_gl=5, n_sub=1, n_sample=_S_MIN, kernel=False, events=None, frozen=None):
    """F - 1 (or the basis columns) at planet times tt (n, m, since each
    point's mid-transit) and star times ts (n, m, since the spots' epoch):
    per-set (n,) parameters, spots4 (n, K, 4) of (lon, lat, alpha [rad],
    contrast). ``exp_time`` > 0: an exposure per time, by the contact rule
    (``mode``; n_gl nodes per piece and n_sample count samples for its
    event search) or supersampled (n_sub uniform nodes); ``events`` (n, m,
    ...) the contact rule's split candidates if already found and
    ``frozen`` their constant copy (_all_events). ``kernel``: the columns
    and the event search by the fp32 Metal kernels (``metal_spots``)."""
    dt = tt.dtype
    params = (period, a, b, r, prot, istar, lam)
    off = ts - tt                                   # the star's clock, ahead

    def cols(nk, lead=0):
        sh = (1,) * lead + (-1,) + (1,) * nk
        c = [mx.reshape(p, sh) for p in params]
        eo = None if k is None else _orbit_consts(c[1], c[2], mx.reshape(k, sh),
                                                  mx.reshape(h, sh), ew)
        return c + [eo]

    def inst(T, Ts, nk, active=None, lead=0):
        *c, eo = cols(nk, lead)
        sp = mx.reshape(spots4, (1,) * lead + (spots4.shape[0],) + (1,) * nk
                        + spots4.shape[-2:])
        return node_cols(T, Ts, c, eo, sp, active)

    def node_cols(T, Ts, c, eo, sp, active=None):
        """The columns at planet times T and star times Ts: c the seven
        per-set parameters, eo the orbit constants and sp (..., K, 4) the
        spots, broadcast to T."""
        per, av, bv, rv, prot_, istar_, lam_ = c
        X, Y, _, _, front = _sky((_TWO_PI / per) * T, av, bv, eo)
        far = 2.0 + 2.0 * mx.stop_gradient(rv)                   # clear of the star
        X = mx.where(front, X, far)
        Y = mx.where(front, Y, 0.0)
        n = spot_vectors(Ts, prot_, istar_, lam_, sp[..., 0], sp[..., 1])
        if kernel:
            from .metal_spots import spot_cols_kernel
            return spot_cols_kernel(X, Y, rv, n, sp[..., 2], sp[..., 3], law, active=active)
        return shape_cols_spots(X, Y, rv, n, sp[..., 2], sp[..., 3], law)

    if exp_time == 0.0:
        Bc = inst(tt, ts, 1)
    elif mode == M._INT_SUPER:
        # n_sub uniform nodes, the sub-exposure midpoints (batman's rule)
        x = mx.array((np.arange(n_sub) + 0.5) / n_sub - 0.5, dtype=dt) * exp_time
        Bc = mx.mean(inst(tt[..., None] + x, ts[..., None] + x, 2), axis=-2)
    else:
        if events is None:
            events, frozen = _all_events(tt, ts, *params, spots4, law, k, h, ew, exp_time,
                                         n_sample, kernel)
        Bc = None
        if frozen is not None:
            (htt,), _ = _host_copies([tt])
            mask = _exposure_pieces(htt, exp_time, frozen)[1] > 0.0
            Bc = _active_pieces(*_exposure_pieces(tt, exp_time, events), off, mask,
                                exp_time, int(n_gl), node_cols, params, spots4, k, h, ew)
        if Bc is None:
            T, W = _exposure_nodes(tt, exp_time, events, int(n_gl), dt)
            Ts = T + off[..., None]
        if Bc is None and kernel:
            # every node, nodes outermost: an exposure's empty sub-intervals
            # (most of its nodes) then fill whole SIMD groups, which return
            # at once, where interleaved with the occupied ones they cost as
            # much
            mv = lambda v: mx.moveaxis(v, -1, 0)                 # noqa: E731
            Bt = inst(mv(T), mv(Ts), 1, mv(W > 0.0), lead=1)
            Bc = mx.sum(mx.moveaxis(Bt, 0, -2) * W[..., None], axis=-2)
        elif Bc is None:
            Bc = mx.sum(inst(T, Ts, 2, W > 0.0) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


def _active_pieces(mid, hw, off, mask, exp_time, n_gl, node_cols, params, spots4, k, h, ew):
    """sum_j W_j B(T_j) per exposure over the nodes of its non-empty pieces
    only: most of an exposure's _K_SPLIT + 1 pieces (mid, hw) are empty,
    and gathering the rest before laying out their n_gl nodes saves the
    empty pieces' nodes, orbit, rotation and columns (and the gradient's
    graph over them). Each set's active pieces are gathered to a common
    width (the largest set's, rounded up to one of eight sizes per octave
    so the float64 graph's compiled columns see few shapes; padded with
    weight 0), so the per-set parameters broadcast and their gradients are
    sums along an axis, not scatters into a few rows. ``off`` (n, m) the
    star's clock ahead of the planet's; ``mask``: hw > 0, a constant
    (computed from host copies, so reading it evaluates nothing traced)."""
    mask = np.asarray(mask)
    n, m, J = mid.shape
    mask = mask.reshape(n, m * J)
    count = mask.sum(axis=1)
    top = int(count.max())
    step = 1 << max(int(math.ceil(math.log2(max(top, 8)))) - 3, 0)
    width = -(-top // step) * step
    rows, cols = np.nonzero(mask)                      # row-major: rows sorted
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    pos = np.arange(rows.size) - start[rows]
    order = np.zeros((n, width), dtype=np.int32)
    order[:] = cols[np.maximum(start + count - 1, 0)][:, None]   # pad: a real piece
    order[rows, pos] = cols
    real = np.arange(width)[None, :] < count[:, None]
    ii = mx.array(order)
    take = lambda v: mx.take_along_axis(mx.reshape(v, (n, m * J)), ii, axis=1)   # noqa: E731
    # the weights' normalisation per exposure: they sum to 1 up to rounding
    norm = mx.sum(hw, axis=-1) * (float(np.sum(gauss_legendre(n_gl)[1])) / exp_time)
    T, W = _piece_nodes(take(mid), mx.where(mx.array(real), take(hw), 0.0), exp_time,
                        n_gl, mid.dtype)
    eo_ = mx.array(order // J)
    W = W / mx.repeat(mx.take_along_axis(norm, eo_, axis=1), n_gl, axis=1)
    Ts = T + mx.repeat(mx.take_along_axis(off, eo_, axis=1), n_gl, axis=1)
    c = [mx.reshape(p, (-1, 1)) for p in params]
    eo = None if k is None else _orbit_consts(c[1], c[2], mx.reshape(k, (-1, 1)),
                                              mx.reshape(h, (-1, 1)), ew)
    B = node_cols(T, Ts, c, eo, spots4[:, None])        # (n, width * n_gl, ncol)
    nc = B.shape[-1]
    BW = mx.sum(mx.reshape(B * W[..., None], (n, width, n_gl, nc)), axis=2)
    slot = mx.array((np.arange(n)[:, None] * m + order // J).reshape(-1))
    acc = mx.zeros((n * m, nc), dtype=B.dtype).at[slot].add(mx.reshape(BW, (-1, nc)))
    return mx.reshape(acc, (n, m, nc))


def _all_events(tt, ts, period, a, b, r, prot, istar, lam, spots4, law, k, h, ew,
                exp_time, n_sample, kernel):
    """The contact rule's split candidates for exposures centred on planet
    times tt (n, m), the star at ts: the planet's limb contacts and every
    spot's events, (n, m, 4 + _E_SPOT * K), live in the parameters -- and
    the same values as a constant computed from host copies (None under
    mx.compile or vmap), from which _active_pieces reads its mask without
    evaluating the traced graph early."""
    half = 0.5 * exp_time
    off = ts - tt
    params = (period, a, b, r, prot, istar, lam)
    if kernel:
        sp_ev, sp_fz = _kernel_events(tt, off, half, *params, spots4, law, n_sample, k, h, ew)
    else:
        ev_fn = _events_fn(tt.dtype, n_sample, k is not None, ew)
        sp_ev, sp_fz = ev_fn(tt - half, tt + half, off, *params, spots4,
                             *(() if k is None else (k, h)))

    def contacts(tt, period, a, b, r, k, h):
        pc = [mx.reshape(v, (-1, 1)) for v in (period, a, b, r)]
        kh = (None, None) if k is None else (mx.reshape(k, (-1, 1)), mx.reshape(h, (-1, 1)))
        return _contact_events(tt, *pc, *kh, ew)
    live = mx.concatenate([contacts(tt, period, a, b, r, k, h), sp_ev], axis=-1)
    if sp_fz is None:
        return live, None
    hv, _ = _host_copies([tt, period, a, b, r] + ([] if k is None else [k, h]))
    hk = (None, None) if k is None else tuple(hv[5:7])
    return live, mx.concatenate([contacts(*hv[:5], *hk), sp_fz], axis=-1)


def _spots_ok(spots4):
    """The spots' domain as a mask over sets (arrays are never read back:
    a traced value outside it gives NaN)."""
    lat, al, c = spots4[..., 1], spots4[..., 2], spots4[..., 3]
    ok = (al > 0.0) & (al <= 0.5 * math.pi) & (mx.abs(lat) <= 0.5 * math.pi) & (c >= 0.0)
    return mx.all(ok, axis=-1)


def _check_spots(spots4, degrees=True):
    """Host values only: (K, 4) or (n, K, 4) of (lon, lat, alpha [rad],
    contrast). ``degrees``: the units the messages name."""
    if isinstance(spots4, mx.array):
        return
    s = np.asarray(spots4, dtype=np.float64)
    if s.ndim not in (2, 3) or s.shape[-1] != 4:
        raise ValueError("spots must be (K, 4) or (n_sets, K, 4): longitude, "
                         "latitude, angular radius, contrast")
    if not np.all(np.isfinite(s)):
        raise ValueError("spots must be finite")
    al, lat, c = s[..., 2], s[..., 1], s[..., 3]
    unit = ("(0, 90] degrees", "[-90, 90] degrees") if degrees else ("(0, pi/2] radians",
                                                                     "[-pi/2, pi/2] radians")
    if np.any(al <= 0.0) or np.any(al > 0.5 * math.pi):
        raise ValueError(f"a spot's angular radius must be in {unit[0]}")
    if np.any(np.abs(lat) > 0.5 * math.pi):
        raise ValueError(f"a spot's latitude must be in {unit[1]}")
    if np.any(c < 0.0):
        raise ValueError("a spot's contrast must be >= 0")


def _refuse_transforms(arrays):
    """A spotted star reads values back to the host (the event search's
    sample count, its constant copy, the active nodes' mask) and its graph
    trips MLX 0.32's compiler -- an error or a crash. Refuse mx.compile and
    vmap up front, where nothing can be read."""
    # mx.eval raises under a transform (np.asarray of a traced array there
    # crashes the process)
    for v in arrays:
        if isinstance(v, mx.array) and v.size:
            try:
                mx.eval(mx.stop_gradient(mx.reshape(v, (-1,))[:1]))
            except ValueError as e:
                raise ValueError(
                    "a spotted star cannot be evaluated under mx.compile or mx.vmap: "
                    "call it eagerly (its float32 kernels are already fused)") from e


@M.fp64_on_cpu
def flux_dev_spots_tau(tau, ts, period, a, b, r, prot, istar, lam, spots4, law, u,
                       basis=False, k=None, h=None, ew=False, exp_time=0.0,
                       mode=M._INT_CONTACT, n_gl=5, n_sub=1, n_sample=None,
                       use_metal=True, degrees=False):
    """F - 1 of a planet crossing a spotted star, normalised to the
    unspotted star: the planet at tau ((m,) or (n, m), since each point's
    mid-transit), the star at ts (the same shape, since the spots' epoch).
    Per-set parameters are scalars or (n,) / (n, 1) arrays: period, a, b
    (impact parameter at conjunction), r, prot, istar and lam (radians);
    spots4 (K, 4) or (n, K, 4) of (longitude, latitude, angular radius
    [rad], contrast). (k, h) = sqrt(e) (cos w, sin w) for an eccentric
    orbit, or (e, w [rad]) with ``ew``. ``exp_time`` > 0: each time an
    exposure of that length, by the contact rule (``mode``, ``n_gl`` nodes
    per piece, ``n_sample`` count samples per exposure for its event
    search: None for ``sample_count`` from host values of a, period and r)
    or supersampled (``n_sub`` nodes). Float64 on the CPU stream unless
    tau is float32; float32 on the GPU stream takes the Metal kernels
    (``metal_spots``) unless ``use_metal`` is False. Not under mx.compile
    or vmap."""
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    _check_spots(spots4, degrees)
    _refuse_transforms([tau, ts, period, a, b, r, prot, istar, lam, spots4, u, k, h])
    if exp_time == 0.0:
        mode = M._INT_NONE
    if n_sample is None and mode == M._INT_CONTACT:
        try:
            host = [np.asarray(v, dtype=np.float64) for v in (a, period, r)]
            n_sample = max(sample_count(exp_time, x, P, y)
                           for x, P, y in np.broadcast(*host))
        except (TypeError, ValueError, RuntimeError):
            n_sample = 64
    dt = tau.dtype
    ts = M._as_dtype(ts if isinstance(ts, mx.array) else mx.array(
        np.asarray(ts, dtype=np.float64), dtype=dt), dt)
    squeeze = tau.ndim == 1 and ts.ndim == 1
    t2d = tau[None, :] if tau.ndim == 1 else tau
    s2d = ts[None, :] if ts.ndim == 1 else ts
    if s2d.shape[-1] != t2d.shape[-1]:
        raise ValueError(f"spot_time must have tau's points: {ts.shape} against {tau.shape}")
    ecc = k is not None
    params = (period, a, b, r, prot, istar, lam) + ((k, h) if ecc else ())
    sp = spots4 if isinstance(spots4, mx.array) else mx.array(
        np.asarray(spots4, dtype=np.float64), dtype=dt)
    sp = M._as_dtype(sp, dt)
    n_param = max([(p.reshape(-1).shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                    else 1) for p in params] + [sp.shape[0] if sp.ndim == 3 else 1])
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(t2d.shape[0], s2d.shape[0], n_param)
    pc = [M._canon_param(p, n, dt) for p in params]
    t2d = mx.broadcast_to(t2d, (n, t2d.shape[1]))
    s2d = mx.broadcast_to(s2d, (n, s2d.shape[1]))
    sp = mx.broadcast_to(sp if sp.ndim == 3 else sp[None], (n,) + sp.shape[-2:])
    kh = pc[-2:] if ecc else [None, None]
    from .metal_hybrid import _canon_w
    w2d = None if basis else _canon_w(u, n, law.n_w, dt, law.name)

    kernel = (use_metal and dt == mx.float32 and M._gpu_stream_active()
              and M.metal_available())

    # the contact rule's split candidates: found once for every exposure
    # (each costs a search, not an evaluation per node), then sliced with
    # the times into the blocks of the columns
    per_time = [s2d]
    if mode == M._INT_CONTACT:
        events, frozen = _all_events(t2d, s2d, *pc[:7], sp, law, *kh, ew, exp_time,
                                     n_sample or _S_MIN, kernel)
        per_time += [events] + ([] if frozen is None else [frozen])

    def run(tt, rows, ts, *ev):
        *p7, s4, w, kk, hh = rows
        return _tau_graph(tt, ts, *p7, s4, w, law, basis=basis, k=kk, h=hh, ew=ew,
                          exp_time=exp_time, mode=mode, n_gl=n_gl, n_sub=n_sub,
                          n_sample=n_sample or _S_MIN, kernel=kernel,
                          events=ev[0] if ev else None,
                          frozen=ev[1] if len(ev) > 1 else None)
    # evaluation points per time: under the contact rule the active nodes
    # (_active_pieces), typically one or two pieces of n_gl
    nodes = {M._INT_NONE: 1, M._INT_SUPER: max(int(n_sub), 1)}.get(mode, 2 * int(n_gl))
    out = _in_blocks(run, t2d, [*pc[:7], sp, w2d, *kh], nodes,
                     _SPOT_BLOCK_KERNEL if kernel else _SPOT_BLOCK, per_time=per_time)
    ok = _spots_ok(sp)[:, None]
    out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    return out[0] if squeeze and n == 1 else out


@M.fp64_on_cpu
def flux_dev_spots_t(t, t0, epoch, period, a, b, r, prot, istar, lam, spots4, law, u,
                     basis=False, k=None, h=None, ew=False, exp_time=0.0, n_gl=5,
                     n_sample=None, use_metal=True):
    """``flux_dev_spots_tau`` from absolute times t ((m,) or (n, m)), the
    per-set mid-transit t0 and the spots' epoch (TransitModel's route):
    tau = t - t0 wrapped to the nearest transit, ts = t - epoch.
    ``exp_time`` > 0 takes the contact rule."""
    def ex(v):
        return mx.reshape(v, (-1, 1)) if isinstance(v, mx.array) and v.ndim >= 1 else v
    tau = t - ex(t0)
    per = ex(period)
    tau = tau - per * mx.round(tau / per)
    ts = t - ex(epoch)
    return flux_dev_spots_tau(tau, ts, period, a, b, r, prot, istar, lam, spots4, law, u,
                              basis=basis, k=k, h=h, ew=ew, exp_time=exp_time,
                              mode=M._INT_CONTACT, n_gl=n_gl, n_sample=n_sample,
                              use_metal=use_metal, degrees=True)
