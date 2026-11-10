"""Fused fp32 Metal kernels for secondary eclipses of spherical planets.

The planet, a uniform disc, passes behind the star: its visible fraction
minus 1 is the photometric core's uniform deficit s0d over pi r^2
(``eclipse._vis_dev_spherical``, the graph this transcribes). One thread per
point does the orbit, the overlap and the exposure rule (none, or
supersampling: an eclipse has no contact rule) in registers; the VJP
recomputes forward-style and reduces the per-chain gradients by simd_sum,
as the transit tau kernels do (``metal._TAU_VJP_G``).

The orbit plug-ins are the transit tau kernels' (``metal._TAU_ORBITS``),
read, not changed: a point is evaluated where the planet is behind the
star (``front`` false), and adds exactly nothing in front of it or clear of
the disc. ``s0d`` and its partials are the core's text (``metal._CORE``,
``metal._PHOT_PARTIALS``) for the uniform term alone. Nothing here edits an
existing kernel; the oblate eclipse kernel (``metal_oblate``) includes the
device function below.
"""

from __future__ import annotations

import mlx.core as mx

from . import metal as M

__all__ = ["eclipse_vis_kernel"]


# visible fraction - 1 of a uniform disc of radius r at separation z behind
# the star, and its z and r partials; r <= 0 (or NaN) is a point planet,
# always visible, as the graph's mask
_ECL_FN = """
inline float mp_ecl_vis(float z, float r, thread float *dvdz,
                        thread float *dvdr) {
    *dvdz = 0.0f; *dvdr = 0.0f;
    if (!(r > 0.0f) || z >= 1.0f + r) return 0.0f;
    float r2 = r * r, z2 = z * z;
    bool m_comp = z <= 1.0f - r;
    bool m_tot = z <= r - 1.0f;     // total occultation: r > 1 only
    float s0d, ds0dz, ds0dr;
    if (m_comp) {
        s0d = -MP_PI * r2;
        ds0dz = 0.0f;
        ds0dr = -2.0f * MP_PI * r;
    } else {
        // Kahan sorted-product sqarea: grouping kept verbatim
        float sa = max(z, r);
        float sc = min(z, r);
        float sb2 = min(sa, 1.0f);
        sa = max(sa, 1.0f);
        float sb = max(sb2, sc);
        sc = min(sb2, sc);
        float sqarea = (sa + (sb + sc)) * (sc - (sa - sb))
                     * (sc + (sa - sb)) * (sa + (sb - sc));
        float kite = metal::precise::sqrt(max(sqarea, KITE_FLOOR));
        float kap0 = metal::precise::atan2(kite, r2 + z2 - 1.0f);
        float kap1 = metal::precise::atan2(kite, 1.0f + z2 - r2);
        s0d = -(kap1 + r2 * kap0 - 0.5f * kite);
        ds0dz = kite / z;
        ds0dr = -2.0f * r * kap0;
    }
    if (m_tot) {
        s0d = -MP_PI;
        ds0dz = 0.0f;
        ds0dr = 0.0f;
    }
    float inv = 1.0f / (MP_PI * r2);
    float v = s0d * inv;
    *dvdz = ds0dz * inv;
    *dvdr = ds0dr * inv - 2.0f * v / r;
    return v;
}
"""

_ECL_HEAD = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;

    float tau0 = taui[i];
    float per = perin[y];
    float av  = ain[y];
ORB_LOAD
    float r   = rin[y];
    float hw_exp = 0.5f * expt;
    int n_node = (mode == MODE_SUPER) ? max(nsub, 1) : 1;
"""

# a node's time: the point's own, or batman's uniform supersampling nodes
_ECL_NODE = """        float tt = tau0;
        if (mode == MODE_SUPER) {
            float frac = (nsub == 1) ? 0.5f : (float)j / (float)(nsub - 1);
            tt = (tau0 - hw_exp) + 2.0f * hw_exp * frac;
        }
"""

_ECL_FWD = _ECL_HEAD + """
    float z;
    bool front;
ORB_DECL
    float acc = 0.0f;
    for (int j = 0; j < n_node; ++j) {
""" + _ECL_NODE + """        ORB_Z(tt)
        float dz, dr;
        if (!front) acc += mp_ecl_vis(z, r, &dz, &dr);
    }
    out[i] = acc / (float)n_node;
"""

# per-point gradient in tau; per-chain slots (per, theta..., r)
_ECL_VJP = _ECL_HEAD + """
    float ctv = ct[i];
    float z, dzdphi;
    bool front;
    float dth[NTH];
ORB_DECL
    float p_per = 0.0f, p_r = 0.0f;
    float p_th[NTH];
    for (int k = 0; k < NTH; ++k) p_th[k] = 0.0f;
    float g_tau = 0.0f;
    float cw = ctv / (float)n_node;
    for (int j = 0; j < n_node; ++j) {
""" + _ECL_NODE + """        ORB_ZD(tt)
        if (front) continue;
        float dvdz, dvdr;
        mp_ecl_vis(z, r, &dvdz, &dvdr);
        float dfdtau = dvdz * dzdphi * (MP_TWO_PI / per);
        g_tau += cw * dfdtau;
        p_per += cw * (-dfdtau * tt / per);
        for (int k = 0; k < NTH; ++k) p_th[k] += cw * dvdz * dth[k];
        p_r += cw * dvdr;
    }
    gtau[i] = g_tau;
    int n_live = TH_LIVE;                     // uniform over the simdgroup
    p_per = metal::simd_sum(p_per);
    for (int k = 0; k < NTH; ++k)
        if (k < n_live) p_th[k] = metal::simd_sum(p_th[k]);
    p_r = metal::simd_sum(p_r);
    if (metal::simd_is_first()) {
        uint ngrp = ((uint)npts + 31u) / 32u;
        uint ob = y * NSLOT_C * ngrp + x / 32u;
        gpar[ob] = p_per;
        for (int k = 0; k < NTH; ++k)
            if (k < n_live) gpar[ob + (1u + (uint)k) * ngrp] = p_th[k];
        gpar[ob + (1u + NTH_U) * ngrp] = p_r;
    }
"""


def _ecl_header() -> str:
    return M._subst(_ECL_FN)


def _src(body: str, orbit: str) -> str:
    import re
    plug = M._TAU_ORBITS[orbit]
    nth = len(plug["theta"])
    s = (body.replace("ORB_LOAD\n", plug["load"])
             .replace("ORB_DECL\n", plug["decl"]))
    s = re.sub(r"ORB_ZD\((\w+)\)", lambda m: plug["zd"].replace("TT", m.group(1)), s)
    s = re.sub(r"ORB_Z\((\w+)\)", lambda m: plug["z"].replace("TT", m.group(1)), s)
    s = (s.replace("TH_LIVE", plug["live"])
          .replace("NSLOT_C", f"{nth + 2}u")
          .replace("NTH_U", f"{nth}u")
          .replace("NTH", str(nth))
          .replace("NORB_I", str(M.NORB))
          .replace("NORB_C", f"{M.NORB}u"))
    for marker in ("ORB_", "TH_LIVE", "NTH", "NSLOT"):
        assert marker not in s, f"unexpanded {marker} in eclipse kernel"
    return M._tau_src(s)


def _get_kernels(orbit: str):
    key = ("ecl", orbit)
    if key not in M._kernels:
        hdr = M._HEADER + M._phot_header() + _ecl_header()
        if orbit == "ecc":
            hdr += M._ecc_header()
        ins = (["taui", "perin", "ain"] + M._TAU_ORBITS[orbit]["inputs"]
               + ["rin", "expt", "mode", "nsub", "npts"])
        M._kernels[key] = (
            mx.fast.metal_kernel(
                name=f"mp_ecl_fwd_{orbit}", input_names=ins,
                output_names=["out"], header=hdr,
                source=_src(_ECL_FWD, orbit)),
            mx.fast.metal_kernel(
                name=f"mp_ecl_vjp_{orbit}", input_names=ins[:-1] + ["ct", "npts"],
                output_names=["gtau", "gpar"], header=hdr,
                source=_src(_ECL_VJP, orbit)))
    return M._kernels[key]


def _make_core(exp_time, mode, n_sub, orbit):
    """custom_function over the eclipse kernels. Primals: (tau2d, period,
    a, shape, r), ``shape`` as metal._make_tau_core_g's."""
    key = ("ecl", orbit, float(exp_time), int(mode), int(n_sub))
    if key in M._tau_cores:
        return M._tau_cores[key]
    nth = len(M._TAU_ORBITS[orbit]["theta"])
    nslot = nth + 2

    @mx.custom_function
    def core(tau2d, period, a, shape, r):
        n, m = tau2d.shape
        k = _get_kernels(orbit)[0]
        return k(inputs=[tau2d, period, a, shape, r, exp_time, mode, n_sub,
                         int(m)],
                 output_shapes=[(n, m)], output_dtypes=[mx.float32],
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]

    @core.vjp
    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        tau2d = primals[0]
        n, m = tau2d.shape
        cols = (m + 31) // 32
        k = _get_kernels(orbit)[1]
        gtau, gpar = k(
            inputs=[*primals, exp_time, mode, n_sub, ct, int(m)],
            output_shapes=[(n, m), (n, nslot, cols)],
            output_dtypes=[mx.float32] * 2, init_value=0.0,
            grid=(m, n, 1), threadgroup=(256, 1, 1))
        g = mx.sum(gpar, axis=2)
        if orbit == "circ":
            g_shape = g[:, 2]
        else:
            g_shape = mx.zeros((n, M.NORB), dtype=g.dtype)
            g_shape[:, mx.array(M._ECC_TH_COLS)] = g[:, 2:1 + nth]
        return gtau, g[:, 0], g[:, 1], g_shape, g[:, 1 + nth]

    M._tau_cores[key] = core
    return core


def eclipse_vis_kernel(tau2d, per_c, a_c, b_c, r_c, k=None, h=None,
                       exp_time=0.0, mode=M._INT_NONE, n_sub=1):
    """The fp32 kernel route of ``eclipse._vis_dev_spherical`` with its
    exposure rule: canonical (n,) parameters in, the visible fraction of
    the disc minus 1, (n, m), out, averaged over ``n_sub`` uniform nodes
    across ``exp_time`` for mode metal._INT_SUPER."""
    if not (mode == M._INT_SUPER and exp_time != 0.0 and n_sub > 1):
        mode, exp_time, n_sub = M._INT_NONE, 0.0, 1
    if k is None:
        orbit, shape = "circ", b_c
    else:
        from .anchored import anchor_constants, pack_orbit_constants
        ci = M._ecc_shape(a_c, b_c, k, h)[2]
        shape = pack_orbit_constants(k, h, ci, consts=anchor_constants(k, h))
        orbit = "ecc"
    core = _make_core(exp_time, mode, n_sub, orbit)
    return core(tau2d, per_c, a_c, shape, r_c)
