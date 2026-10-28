"""Secondary eclipses from time since mid-transit: ``metal.flux_dev_from_tau``
with ``transittype="secondary"`` delegates here.

The planet is a uniform disc of flux ratio fp behind the star, as in
``TransitModel(transittype="secondary")``: F = 1 + fp * (visible fraction
of the planet's disc), so the deviation F - 1 is fp out of eclipse and 0
in totality, and 1 + dev is TransitModel's light curve. ``tau`` keeps its
meaning, each point's time since its own mid-transit; the orbit places the
eclipse (near +-P/2 on a circular orbit, where the wrap of tau is harmless:
the sky position is periodic in it).

A spherical planet's visible fraction is the uniform-disc column s0 of
``solution.sn_dev`` at the planet's separation, with the near side pushed
clear of the star (TransitModel's rule); an oblate planet's is
``oblate_tau.eclipse_dev_from_tau_oblate``, imported only for one. The MLX
graph only, in either precision (no Metal kernel); the exposure rule is
"none" or "supersample", applied here for both.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .dtypes import fp64_on_cpu
from .ellip import dtype_eps
from .solution import sn_dev
from .trig import sincos

__all__ = ["eclipse_dev_from_tau"]

_TWO_PI = 2.0 * math.pi


def _vis_dev_spherical(tau, period, a, b, r, k=None, h=None):
    """Visible fraction of a uniform spherical planet's disc, minus 1, on
    (n, m) tau and (n,) parameters (0 out of eclipse, -1 in totality).
    Circular: the sky position a sin phi, b cos phi; eccentric: the
    transit-anchored orbit, as the transit graph's (metal._tau_graph)."""
    from .anchored import separation_anchored
    per, av, bv, rv = (mx.reshape(p, (-1, 1)) for p in (period, a, b, r))
    phi = (_TWO_PI / per) * tau
    if k is None:
        s, c = sincos(phi)
        eps = dtype_eps(tau.dtype)
        z = mx.sqrt(mx.maximum((av * s) ** 2 + (bv * c) ** 2, (10.0 * eps) ** 2))
        front = c > 0.0
    else:
        kv, hv = mx.reshape(k, (-1, 1)), mx.reshape(h, (-1, 1))
        ci = M._ecc_shape(av, bv, kv, hv)[2]
        z, front = separation_anchored(phi, kv, hv, av, ci)
    # the planet in front of the star is pushed clear of it (TransitModel's
    # _photom); r = 0 is a point planet, always visible
    s0d = sn_dev(mx.where(front, z + 2.0 * (1.0 + rv), z), rv)[0]
    pos = rv > 0.0
    return mx.where(pos, s0d / mx.where(pos, math.pi * rv * rv, 1.0), 0.0)


@fp64_on_cpu
def eclipse_dev_from_tau(tau, period, a, b, r, fp, f=None, theta=None, k=None,
                         h=None, exp_time=0.0, mode=None, n_sub=1):
    """F - 1 of a secondary eclipse, fp (1 + visible-fraction deviation),
    (n, m) -- or (m,) for 1-D tau and no per-chain parameters. ``f`` given:
    an oblate planet (``theta`` defaulting to 0). ``mode``: metal._INT_NONE
    or _INT_SUPER (``n_sub`` uniform nodes over ``exp_time``, batman's
    rule, as the transit graph's)."""
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    oblate = f is not None
    if oblate:
        from .oblate import _check_domain
        _check_domain(r, f)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = ((period, a, b, r, fp) + ((f, 0.0 if theta is None else theta)
                                       if oblate else ())
              + ((k, h) if ecc else ()))
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    n = max(tau2d.shape[0], n_param)
    pc = [M._canon_param(p, n, tau2d.dtype) for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    per_c, a_c, b_c, r_c, fp_c = pc[:5]
    kh = pc[-2:] if ecc else [None, None]
    m = tau2d.shape[1]
    nodes = 1
    tt = tau2d
    if mode == M._INT_SUPER and exp_time != 0.0 and n_sub > 1:
        half = 0.5 * exp_time
        off = mx.array(np.linspace(-half, half, int(n_sub)), dtype=tau2d.dtype)
        nodes = int(n_sub)
        tt = mx.reshape(tau2d[..., None] + off, (n, m * nodes))
    if oblate:
        from .oblate_tau import eclipse_dev_from_tau_oblate
        vis = eclipse_dev_from_tau_oblate(tt, per_c, a_c, b_c, r_c, *pc[5:7],
                                          k=kh[0], h=kh[1])
        vis = mx.reshape(vis, (n, m * nodes))
    else:
        vis = _vis_dev_spherical(tt, per_c, a_c, b_c, r_c, *kh)
    if nodes > 1:
        vis = mx.mean(mx.reshape(vis, (n, m, nodes)), axis=-1)
    out = mx.reshape(fp_c, (-1, 1)) * (1.0 + vis)
    return out[0] if squeeze and n == 1 else out
