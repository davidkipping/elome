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
``oblate_tau.eclipse_dev_from_tau_oblate``, imported only for one. fp32
data on the GPU stream takes the fused Metal kernels (``metal_eclipse``,
``metal_oblate``), with the exposure rule in the kernel; anything else runs
the MLX graph, fp64 on the CPU stream. The exposure rule is "none",
"supersample" or "contact": the eclipse is the transit of the orbit turned
by pi, whose contacts are the eclipse's (``_eclipse_dev_contact``;
``oblate_tau._eclipse_turned``).

Two orbit helpers for timing an eclipse: ``occultation_offset``, the time
from transit to occultation in orbits, and ``light_travel_delay``, how
much later the eclipse is seen than the geometry places it.
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

__all__ = ["eclipse_dev_from_tau", "occultation_offset",
           "occultation_offset_dev", "light_travel_delay"]

_TWO_PI = 2.0 * math.pi
#: the IAU nominal solar radius (m), the speed of light (m/s), a day (s)
_R_SUN, _C_LIGHT, _DAY = 6.957e8, 299_792_458.0, 86_400.0
#: the floor on 1 - e^2, as the anchored orbit's (e <= 0.999)
_ONE_MINUS_E2_MIN = 1.0 - 0.999 ** 2


def _is_mx(*xs):
    return any(isinstance(x, mx.array) for x in xs)


def _as_mx(*xs):
    """MLX arrays, a Python number taking the first array's dtype (not
    float32, which ``mx.array`` would round a float64 orbit to)."""
    dt = next(x.dtype for x in xs if isinstance(x, mx.array))
    return tuple(x if isinstance(x, mx.array) else mx.array(x, dtype=dt)
                 for x in xs)


def occultation_offset(secosw=0.0, sesinw=0.0):
    """The time from mid-transit to mid-occultation in orbits, (0, 1):
    the mean anomaly between the two conjunctions, in closed form (E from
    the true anomaly), with (secosw, sesinw) = sqrt(e) (cos w, sin w). 1/2
    on a circular orbit, where its slope in (secosw, sesinw) is finite.
    MLX arrays in, MLX out (differentiable, either precision); floats in,
    a float out (float64). The geometric offset: add
    ``light_travel_delay / period`` for the observed one."""
    return 0.5 + occultation_offset_dev(secosw, sesinw)


def occultation_offset_dev(secosw=0.0, sesinw=0.0):
    """``occultation_offset`` - 1/2, in (-1/2, 1/2), without the
    cancellation of forming it from the offset: accurate relative to
    itself, so a float32 model can shift times by it times the period.

    With E_tr the eccentric anomaly at transit (true anomaly pi/2 - w)
    and E_occ = E_tr + pi + dE, tan((E_occ - E_tr) / 2)'s difference
    formula gives dE = 2 atan2(e cos w, sqrt(1 - e^2)), and

        M_occ - M_tr - pi = dE + 2 e sin(E_tr + dE / 2) cos(dE / 2).
    """
    if not _is_mx(secosw, sesinw):
        k, h = float(secosw), float(sesinw)
        e = k * k + h * h
        w = math.atan2(h, k) if e > 0.0 else 0.0
        th = 0.5 * math.pi - w
        E_tr = 2.0 * math.atan2(math.sqrt(max(1.0 - e, 0.0)) * math.sin(0.5 * th),
                                math.sqrt(1.0 + e) * math.cos(0.5 * th))
        dE = 2.0 * math.atan2(math.sqrt(e) * k, math.sqrt(max(1.0 - e * e, 0.0)))
        d = (dE + 2.0 * e * math.sin(E_tr + 0.5 * dE) * math.cos(0.5 * dE)) / _TWO_PI
        return d - round(d)
    k, h = _as_mx(secosw, sesinw)
    e = k * k + h * h
    circ = e == 0.0
    w = mx.arctan2(mx.where(circ, 0.0, h), mx.where(circ, 1.0, k))
    s, c = sincos(0.5 * (0.5 * math.pi - w))
    E_tr = 2.0 * mx.arctan2(mx.sqrt(mx.maximum(1.0 - e, 0.0)) * s,
                            mx.sqrt(1.0 + e) * c)
    # e cos w = k sqrt(e), the root floored: its slope is infinite at e = 0
    dE = 2.0 * mx.arctan2(k * mx.sqrt(mx.maximum(e, 1e-30)),
                          mx.sqrt(mx.maximum(1.0 - e * e, 0.0)))
    # sincos, not mx.sin: MLX's float64 sin is good to only ~2e-7
    sn = sincos(E_tr + 0.5 * dE)[0]
    cs = sincos(0.5 * dE)[1]
    d = (dE + 2.0 * e * sn * cs) / _TWO_PI
    return d - mx.round(d)


def _occultation_offset_dev_ew(e, w):
    """``occultation_offset_dev`` from (e, w [rad]) MLX arrays, with e cos w
    in place of secosw sqrt(e): smooth at e = 0, for TransitModel's (e, w)
    orbit."""
    s, c = sincos(0.5 * (0.5 * math.pi - w))
    E_tr = 2.0 * mx.arctan2(mx.sqrt(mx.maximum(1.0 - e, 0.0)) * s,
                            mx.sqrt(1.0 + e) * c)
    dE = 2.0 * mx.arctan2(e * sincos(w)[1], mx.sqrt(mx.maximum(1.0 - e * e, 0.0)))
    sn = sincos(E_tr + 0.5 * dE)[0]
    cs = sincos(0.5 * dE)[1]
    d = (dE + 2.0 * e * sn * cs) / _TWO_PI
    return d - mx.round(d)


def light_travel_delay(a, b, rstar, secosw=0.0, sesinw=0.0):
    """How much later than the orbit's geometry places it a secondary
    eclipse is seen, in days, relative to the transit: the light-travel
    time across the orbit between the two conjunctions,

        (r_tr + r_occ) sin i / c = 2 a sin i (1 - e^2) / (1 - e^2 sin^2 w) / c,

    2 a sin i / c on a circular orbit (Kaplan 2010's leading term; the
    star's reflex motion, of order the mass ratio, is left out). ``a`` in
    stellar radii and ``b`` the impact parameter at transit (Elómë's), so
    cos i = b (1 + e sin w) / (a (1 - e^2)); ``rstar`` the star's radius
    in solar radii. MLX arrays in, MLX out; floats in, a float out."""
    scale = 2.0 * float(rstar) * _R_SUN / (_C_LIGHT * _DAY)
    if not _is_mx(a, b, secosw, sesinw):
        a, b, k, h = float(a), float(b), float(secosw), float(sesinw)
        e = k * k + h * h
        esw = h * math.sqrt(e)
        one_m = max(1.0 - e * e, _ONE_MINUS_E2_MIN)
        ci = b * (1.0 + esw) / (a * one_m)
        si = math.sqrt(max(1.0 - ci * ci, 0.0))
        return scale * a * si * one_m / (1.0 - esw * esw)
    a, b, k, h = _as_mx(a, b, secosw, sesinw)
    e = k * k + h * h
    # floored before the sqrt: its slope is infinite at e = 0
    esw = h * mx.sqrt(mx.maximum(e, 1e-30))
    one_m = mx.maximum(1.0 - e * e, _ONE_MINUS_E2_MIN)
    ci = b * (1.0 + esw) / (a * one_m)
    si = mx.sqrt(mx.maximum(1.0 - ci * ci, 1e-30))
    return scale * a * si * one_m / (1.0 - esw * esw)


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


@mx.compile
def _since_mid_eclipse(tau2d, period, dev):
    """Each point's time since its own mid-eclipse, P (1/2 + dev) after
    mid-transit, within half a period: (n, m) from (n,) period and dev."""
    per = mx.reshape(period, (-1, 1))
    d = (tau2d - 0.5 * per) - per * mx.reshape(dev, (-1, 1))
    return d - per * mx.round(d / per)


@mx.compile
def _eclipse_from_transit(D, r, fp):
    """fp (1 + D / r^2) per row: the eclipse's F - 1 from the transit
    deviation D; the visible fraction is 1 for r = 0 (a point planet)."""
    rc, fc = mx.reshape(r, (-1, 1)), mx.reshape(fp, (-1, 1))
    pos = rc > 0.0
    return fc * (1.0 + mx.where(pos, D / mx.where(pos, rc * rc, 1.0), 0.0))


@mx.compile
def _fp_times(vis, fp):
    """fp (1 + vis) per row: the eclipse's F - 1 from the visible fraction
    - 1, in one pass."""
    return mx.reshape(fp, (-1, 1)) * (1.0 + vis)


def _eclipse_dev_contact(tau2d, period, a, b, r, fp, k, h, exp_time, n_gl):
    """F - 1 of a spherical planet's eclipse under the contact rule, (n, m),
    from (n,) parameters. The eclipse on the orbit
    (e, w) is the transit of (e, w + pi) -- (secosw, sesinw) -> (-k, -h),
    the impact parameter at occultation b_occ -- at the time since
    mid-eclipse, P occultation_offset after mid-transit: the planet's sky
    position at the one is minus its position at the other, the same
    distance from the star. A uniform disc's overlap with the star is
    symmetric in the two discs, so the visible fraction - 1 is the transit
    deviation D of a star without limb darkening (u = 0) over r^2, and the
    transit graph integrates the exposure, the contacts the flipped orbit's
    own. An fp32 GPU runs the same rule on the eclipse's own kernel
    (metal_eclipse.eclipse_vis_contact_kernel: the transit's orbit and
    split, the eclipse's photometry, half the transit kernel's work per
    node)."""
    if (tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        from .metal_eclipse import eclipse_vis_contact_kernel
        vis = eclipse_vis_contact_kernel(tau2d, period, a, b, r, k, h,
                                         exp_time=exp_time, n_gl=n_gl)
        return _fp_times(vis, fp)
    if k is None:
        dev, b_occ, orbit = mx.zeros_like(period), b, {}
    else:
        e, esw, ci = M._ecc_shape(a, b, k, h)
        dev = occultation_offset_dev(k, h)
        b_occ = a * ci * (1.0 - e * e) / (1.0 - esw)
        orbit = dict(secosw=-k, sesinw=-h)
    zero = mx.zeros_like(r)
    D = M.flux_dev_from_tau(_since_mid_eclipse(tau2d, period, dev), period, a,
                            b_occ, r, zero, zero, exp_time=exp_time,
                            integration="contact", n_gl=n_gl, **orbit)
    return _eclipse_from_transit(D, r, fp)


@fp64_on_cpu
def eclipse_dev_from_tau(tau, period, a, b, r, fp, f=None, theta=None, k=None,
                         h=None, exp_time=0.0, mode=None, n_sub=1, n_gl=5):
    """F - 1 of a secondary eclipse, fp (1 + visible-fraction deviation),
    (n, m) -- or (m,) for 1-D tau and no per-chain parameters. ``f`` given:
    an oblate planet (``theta`` defaulting to 0). ``mode``: metal._INT_NONE,
    _INT_SUPER (``n_sub`` uniform nodes over ``exp_time``, batman's rule, as
    the transit graph's) or _INT_CONTACT (``n_gl`` Gauss-Legendre nodes per
    piece of the exposure split at the contacts: _eclipse_dev_contact, and
    oblate_tau.eclipse_dev_from_tau_oblate for an oblate planet). fp32 on
    the GPU stream: the Metal kernels, the exposure rule in them."""
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    oblate = f is not None
    contact = mode == M._INT_CONTACT and exp_time != 0.0
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
    if contact and oblate:
        from .oblate_tau import eclipse_dev_from_tau_oblate
        vis = eclipse_dev_from_tau_oblate(tau2d, per_c, a_c, b_c, r_c, *pc[5:7],
                                          *kh, exp_time=exp_time, mode=mode,
                                          n_gl=int(n_gl))
        out = _fp_times(vis, fp_c)
        return out[0] if squeeze and n == 1 else out
    if contact:
        out = _eclipse_dev_contact(tau2d, per_c, a_c, b_c, r_c, fp_c, *kh,
                                   exp_time, int(n_gl))
        return out[0] if squeeze and n == 1 else out
    if (tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        if oblate:
            from .metal_oblate import eclipse_vis_oblate_kernel
            from .oblate import _in_domain
            vis = eclipse_vis_oblate_kernel(tau2d, per_c, a_c, b_c, r_c,
                                            *pc[5:7], *kh, exp_time=exp_time,
                                            mode=mode, n_sub=n_sub)
            vis = mx.where(_in_domain(r_c, pc[5])[:, None], vis, mx.nan)
        else:
            from .metal_eclipse import eclipse_vis_kernel
            vis = eclipse_vis_kernel(tau2d, per_c, a_c, b_c, r_c, *kh,
                                     exp_time=exp_time, mode=mode,
                                     n_sub=n_sub)
        out = mx.reshape(fp_c, (-1, 1)) * (1.0 + vis)
        return out[0] if squeeze and n == 1 else out
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
