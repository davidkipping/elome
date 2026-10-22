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

**Exposures.** Instantaneous here; TransitModel supersamples on its own
grid. (The contact rule, whose split points include the planet-spot
tangencies, is not available for spots.)
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .hybrid import combine_cols, get_law
from .oblate_tau import _in_blocks, _orbit_consts, _sky
from .spots import shape_cols_spots
from .trig import sincos

__all__ = ["spin_frame", "spot_vectors", "flux_dev_spots_t"]

_TWO_PI = 2.0 * math.pi
# evaluation points per block of the graph (oblate_tau._in_blocks): the
# columns hold their fp64 intermediates until evaluated
_SPOT_BLOCK = 1 << 14


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
    (sx, sy, sz), (ox, oy, oz), (ex, ey, ez) = (
        tuple(c[..., None] for c in v) for v in spin_frame(istar, lam))
    ph = lon + (_TWO_PI * ts / prot)[..., None]
    sp, cp = sincos(ph)
    sb, cb = sincos(lat)
    u, v = cb * cp, cb * sp
    return mx.stack([u * ox + v * ex + sb * sx, u * oy + v * ey + sb * sy,
                     u * oz + v * ez + sb * sz], axis=-1)


def _tau_graph(tt, t0, epoch, period, a, b, r, prot, istar, lam, spots4, w2d, law,
               basis=False, k=None, h=None, ew=False):
    """F - 1 (or the basis columns) at times tt (n, m): per-set (n,)
    parameters, spots4 (n, K, 4) of (lon, lat, alpha [rad], contrast)."""
    col = lambda p: mx.reshape(p, (-1, 1))
    per, av, bv, rv = col(period), col(a), col(b), col(r)
    tau = tt - col(t0)
    tau = tau - per * mx.round(tau / per)
    eo = None if k is None else _orbit_consts(av, bv, col(k), col(h), ew)
    X, Y, _, _, front = _sky((_TWO_PI / per) * tau, av, bv, eo)
    far = 2.0 + 2.0 * mx.stop_gradient(rv)                   # clear of the star
    X = mx.where(front, X, far)
    Y = mx.where(front, Y, 0.0)
    sp = spots4[:, None]                                       # (n, 1, K, 4)
    n = spot_vectors(tt - col(epoch), col(prot), col(istar), col(lam),
                     sp[..., 0], sp[..., 1])
    Bc = shape_cols_spots(X, Y, rv, n, sp[..., 2], sp[..., 3], law)
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
                     basis=False, k=None, h=None, ew=False):
    """F - 1 of a planet crossing a spotted star, at times t ((m,) or
    (n, m)), normalised to the unspotted star. Per-set parameters are
    scalars or (n,) / (n, 1) arrays: t0 (mid-transit), epoch (the spots'
    reference time), period, a, b (impact parameter at conjunction), r,
    prot, istar and lam (radians); spots4 (K, 4) or (n, K, 4) of
    (longitude, latitude, angular radius [rad], contrast). (k, h) =
    sqrt(e) (cos w, sin w) for an eccentric orbit, or (e, w [rad]) with
    ``ew``. Float64 on the CPU stream unless the data are float32."""
    law = get_law(law)
    if t.ndim not in (1, 2):
        raise ValueError(f"t must be (m,) or (n, m); got {t.shape}")
    _check_spots(spots4)
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
        return _tau_graph(tt, *p9, s4, w, law, basis=basis, k=kk, h=hh, ew=ew)
    out = _in_blocks(run, t2d, [*pc[:9], sp, w2d, *kh], 1, _SPOT_BLOCK)
    ok = _spots_ok(sp)[:, None]
    out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    return out[0] if squeeze and n == 1 else out
