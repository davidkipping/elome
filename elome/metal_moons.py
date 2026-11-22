"""Fused fp32 Metal kernels for a planet and its moon (hybrid laws, and
the quadratic law's lens moments).

The mathematics is ``moons.py``'s (read its docstring first): each disc
is the spherical solution, and the lens the two discs share on the star
is Green's arcs of the two circles and the limb (``moons._lens_cols``).
This module transcribes the lens per point, in registers, with the
graph's choices: the frame E towards the lens, the arcs' order and their
membership tests are frozen (plain float), the arcs' ends, coefficients
and integrals live. A live value is a float or the dual number ``md``,
which carries d/d(xp, yp, rp, xm, ym, rs): the VJP evaluates the lens once
in dual numbers and contracts them with the cotangent.

Two kernels per law: the lens's generator columns [mu^0, mu^2, mu^4,
P_eps...] at the points where the discs overlap, and their VJP; for the
quadratic law the lens's mu^0, mu^2 and its closure for mu^1
(``moons._lens_mu1``, the smaller disc's exact moments from the hybrid and
quadratic device functions). Each disc is those device functions too
(``metal_hybrid.mp_hyb_cols``, ``metal.mp_phot``), one point per thread
with its own radius.
The orbits, the discs' separations and the exposure quadrature stay MLX
graph operations around them (``moons_tau``), whose autodiff chains these.
"""

from __future__ import annotations

import hashlib
import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .hybrid import _may_be_large, get_law
from .metal_hybrid import _f, _hyb_header, _law_tag

__all__ = ["available", "lens_cols_kernel", "lens_quad_kernel", "disc_cols_kernel",
           "disc_quad_kernel"]


def available(dtype) -> bool:
    """Whether the kernels take this computation: float32 on the GPU."""
    return dtype == mx.float32 and M._gpu_stream_active() and M.metal_available()


# ---------------------------------------------------------------------------
# device code
# ---------------------------------------------------------------------------

_MN_FN = r"""
// =================== dual numbers, six derivatives ======================
// a value and d/d(xp, yp, rp, xm, ym, rs)
struct md {
    float v; float4 a; float2 b;
    md() {}
    md(float x) : v(x), a(float4(0.0f)), b(float2(0.0f)) {}
    md(float x, float4 p, float2 q) : v(x), a(p), b(q) {}
};
inline md operator+(md x, md y) { return md(x.v + y.v, x.a + y.a, x.b + y.b); }
inline md operator+(md x, float y) { return md(x.v + y, x.a, x.b); }
inline md operator+(float x, md y) { return md(x + y.v, y.a, y.b); }
inline md operator-(md x, md y) { return md(x.v - y.v, x.a - y.a, x.b - y.b); }
inline md operator-(md x, float y) { return md(x.v - y, x.a, x.b); }
inline md operator-(float x, md y) { return md(x - y.v, -y.a, -y.b); }
inline md operator-(md x) { return md(-x.v, -x.a, -x.b); }
inline md operator*(md x, md y) {
    return md(x.v * y.v, x.v * y.a + y.v * x.a, x.v * y.b + y.v * x.b);
}
inline md operator*(md x, float y) { return md(x.v * y, x.a * y, x.b * y); }
inline md operator*(float x, md y) { return md(x * y.v, x * y.a, x * y.b); }
inline md operator/(md x, md y) {
    float q = x.v / y.v;
    return md(q, (x.a - q * y.a) / y.v, (x.b - q * y.b) / y.v);
}
inline md operator/(md x, float y) { return md(x.v / y, x.a / y, x.b / y); }
inline md operator/(float x, md y) {
    float q = x / y.v;
    return md(q, (-q / y.v) * y.a, (-q / y.v) * y.b);
}
inline md md_chain(float f, float df, md x) { return md(f, x.a * df, x.b * df); }

inline float mv(float x) { return x; }
inline float mv(md x) { return x.v; }
inline float m_sqrt(float x) { return metal::precise::sqrt(x); }
inline md m_sqrt(md x) {
    float s = metal::precise::sqrt(x.v);
    return md_chain(s, 0.5f / s, x);
}
inline float m_sin(float x) { return metal::precise::sin(x); }
inline md m_sin(md x) { return md_chain(metal::precise::sin(x.v), metal::precise::cos(x.v), x); }
inline float m_cos(float x) { return metal::precise::cos(x); }
inline md m_cos(md x) { return md_chain(metal::precise::cos(x.v), -metal::precise::sin(x.v), x); }
inline float m_atan2(float y, float x) { return metal::precise::atan2(y, x); }
inline md m_atan2(md y, md x) {
    float r2 = x.v * x.v + y.v * y.v;
    return md(metal::precise::atan2(y.v, x.v), (x.v * y.a - y.v * x.a) / r2,
              (x.v * y.b - y.v * x.b) / r2);
}
// log(1 + x) without its rounding near x = 0
inline float m_log1p(float x) {
    float u = 1.0f + x;
    return (u == 1.0f) ? x : metal::precise::log(u) * (x / (u - 1.0f));
}
inline md m_log1p(md x) { return md_chain(m_log1p(x.v), 1.0f / (1.0f + x.v), x); }
inline float m_abs(float x) { return fabs(x); }
inline md m_abs(md x) { return (x.v < 0.0f) ? -x : x; }
inline float m_max(float x, float y) { return max(x, y); }
inline md m_max(md x, float y) { return (x.v > y) ? x : md(y); }

// ============================ the lens ==================================
constant float MN_TINY = 1e-30f;      // rings._TINY
constant float MN_THIN = 1e-2f;       // moons._THIN
constant float MN_XSER = 0.05f;       // moons._X_SER
constant float MN_SAMPLES[3] = {0.5f, 0.25f, 0.75f};   // rings._SAMPLES
#define MN_K CONST_K
#define MN_NC (3 + MN_K)
constant float MN_EPS[CONST_KA] = {CONST_EPS};       // the poles
constant float MN_SP[CONST_KA] = {CONST_SP};         // sqrt(1 + e)
constant float MN_P4[CONST_KA] = {CONST_P4};         // 4 (1 + e)
constant float MN_P2[CONST_KA] = {CONST_P2};         // 2 (1 + e)
constant float MN_H1[3 + CONST_KA] = {CONST_H1};     // F on the limb

// where circle 1 meets circle 2 (moons._meet): the points, whether they
// exist, and each point's offset from either centre
template <typename T> struct mn_meet {
    T px[2], py[2], o1x[2], o1y[2], o2x[2], o2y[2];
    bool has;
};

template <typename T>
inline void mn_meet_of(T x1, T y1, T R1, T x2, T y2, T R2, thread mn_meet<T> &P) {
    T dx = x2 - x1, dy = y2 - y1;
    T D2 = dx * dx + dy * dy;
    T D = m_sqrt(m_max(D2, MN_TINY));
    T a = (D2 + (R1 - R2) * (R1 + R2)) / (2.0f * D);
    T hh2 = (R1 - a) * (R1 + a);
    float thin = MN_THIN * min(mv(R1), mv(R2));
    P.has = (mv(hh2) > thin * thin) && (mv(D2) > 0.0f);
    T hh = P.has ? m_sqrt(hh2) : T(0.0f);
    T ux = dx / D, uy = dy / D;
    T bx = x1 + a * ux, by = y1 + a * uy;
    T a2 = a - D;
    P.px[0] = bx - hh * uy; P.py[0] = by + hh * ux;
    P.px[1] = bx + hh * uy; P.py[1] = by - hh * ux;
    P.o1x[0] = a * ux - hh * uy; P.o1y[0] = a * uy + hh * ux;
    P.o1x[1] = a * ux + hh * uy; P.o1y[1] = a * uy - hh * ux;
    P.o2x[0] = a2 * ux - hh * uy; P.o2y[0] = a2 * uy + hh * ux;
    P.o2x[1] = a2 * ux + hh * uy; P.o2y[1] = a2 * uy - hh * ux;
}

// sub-arcs between four breakpoints, ordered by their frozen values
// (moons._sorted_arcs); vals: a value per breakpoint, at each arc's ends
template <typename T>
inline void mn_sorted(thread const T *bp, thread const bool *val, thread const T *vals,
                      float start, thread T *lo, thread T *hi, thread bool *ok,
                      thread T *vlo, thread T *vhi) {
    int ord[4] = {0, 1, 2, 3};
    float key[4];
    int n = 0;
    for (int i = 0; i < 4; ++i) {
        key[i] = val[i] ? mv(bp[i]) : INFINITY;
        n += val[i] ? 1 : 0;
    }
    for (int i = 1; i < 4; ++i)
        for (int j = i; j > 0 && key[ord[j - 1]] > key[ord[j]]; --j) {
            int s = ord[j]; ord[j] = ord[j - 1]; ord[j - 1] = s;
        }
    T first = bp[ord[0]];
    for (int i = 0; i < 4; ++i) {
        bool last = !(i + 1 < n);
        T h = last ? first + MP_TWO_PI : ((i + 1 < 4) ? bp[ord[i + 1]] : first);
        bool whole = (n == 0) && (i == 0);
        lo[i] = whole ? T(start) : bp[ord[i]];
        hi[i] = whole ? T(start + MP_TWO_PI) : h;
        ok[i] = (i < n) || whole;
        vlo[i] = vals[ord[i]];
        vhi[i] = last ? vals[ord[0]] : ((i + 1 < 4) ? vals[ord[i + 1]] : vals[ord[0]]);
    }
}

// an antiderivative of 1 / (a + b cos psi) (moons._J), from (sin, cos,
// tan) of psi / 2
template <typename T>
inline T mn_J(T sh, T ch, T t, T amb, T apb, T Q) {
    T x = (amb / apb) * t * t;
    if (fabs(mv(x)) < MN_XSER && mv(ch) > 0.0f) {
        T acc = T(1.0f / 11.0f);
        for (int k = 4; k >= 0; --k) acc = 1.0f / (float)(2 * k + 1) - x * acc;
        return 2.0f * t / apb * acc;
    }
    if (mv(Q) > 0.0f)
        return 2.0f / m_sqrt(Q) * m_atan2(m_sqrt(amb) * sh, m_sqrt(apb) * ch);
    T w = m_sqrt(-amb) * t / m_sqrt(apb);
    return 1.0f / m_sqrt(-Q) * m_log1p(2.0f * w / (1.0f - w));
}

// an arc's generator columns (moons._arc_cols)
template <typename T>
inline void mn_arc_cols(T z, T r, T lo, T hi, T sl, T cl, T sh, T chh, thread T *cols) {
    T b = 2.0f * z * r;
    T A[4];
    A[0] = hi - lo;
    A[1] = sh - sl;
    A[2] = 0.5f * (hi - lo) + 0.5f * (sh * chh - sl * cl);
    A[3] = (sh - sl) - (sh * sh * sh - sl * sl * sl) / 3.0f;
    T t0 = (1.0f - z) * (1.0f + z) - r * r;
    T h1[2] = {0.25f * (1.0f + t0), 0.25f * b};
    T h2[3] = {(1.0f + t0 + t0 * t0) / 6.0f, b * (1.0f + 2.0f * t0) / 6.0f, b * b / 6.0f};
    cols[0] = r * r * 0.5f * A[0] - r * z * 0.5f * A[1];
    T c = r * r * h1[0] * A[0];
    c = c + (r * r * h1[1] - r * z * h1[0]) * A[1];
    cols[1] = c + (0.0f - r * z * h1[1]) * A[2];
    c = r * r * h2[0] * A[0];
    c = c + (r * r * h2[1] - r * z * h2[0]) * A[1];
    c = c + (r * r * h2[2] - r * z * h2[1]) * A[2];
    cols[2] = c + (0.0f - r * z * h2[2]) * A[3];
    if (MN_K > 0) {
        T shl = m_sin(0.5f * lo), chl = m_cos(0.5f * lo);
        T shh = m_sin(0.5f * hi), chh2 = m_cos(0.5f * hi);
        T tl = shl / ((mv(chl) > 0.0f) ? chl : T(1.0f));
        T th = shh / ((mv(chh2) > 0.0f) ? chh2 : T(1.0f));
        for (int k = 0; k < MN_K; ++k) {
            float sp = MN_SP[k];
            T amb = (sp - z - r) * (sp + z + r);
            T dzr = m_abs(z - r);
            T apb = (sp - dzr) * (sp + dzr);
            bool reach = mv(apb) > 0.0f;
            amb = reach ? amb : T(0.5f);
            apb = reach ? apb : T(1.0f);
            T Q = amb * apb;
            T J = mn_J(shh, chh2, th, amb, apb, Q) - mn_J(shl, chl, tl, amb, apb, Q);
            T pd = (sp - z) * (sp + z) + r * r;
            cols[3 + k] = -(hi - lo) / MN_P4[k] + pd / MN_P4[k] * J;
        }
    }
}

// F of each column's Green form at mu^2 = t (moons._green_f)
template <typename T>
inline void mn_green(T t, thread T *f) {
    f[0] = T(0.5f);
    f[1] = 0.25f * (1.0f + t);
    f[2] = (1.0f + t + t * t) / 6.0f;
    for (int k = 0; k < MN_K; ++k) f[3 + k] = 1.0f / (MN_P2[k] * (t + MN_EPS[k]));
}

// the margin of largest magnitude (rings._decisive)
inline float mn_decide(float best, float m) { return (fabs(m) > fabs(best)) ? m : best; }

// occulted generator columns (positive) of the lens of two circles on the
// star (moons._lens_cols)
template <typename T>
inline void mn_lens(T X0, T Y0, T R0, T X1, T Y1, T R1, thread T *occ) {
    for (int c = 0; c < MN_NC; ++c) occ[c] = T(0.0f);
    T cx[2] = {X0, X1}, cy[2] = {Y0, Y1}, cr[2] = {R0, R1};
    mn_meet<T> limb[2], cor;
    for (int k = 0; k < 2; ++k)
        mn_meet_of(cx[k], cy[k], cr[k], T(0.0f), T(0.0f), T(1.0f), limb[k]);
    mn_meet_of(X0, Y0, R0, X1, Y1, R1, cor);
    // E, frozen: along the two centres' sum
    float ex = mv(X0) + mv(X1), ey = mv(Y0) + mv(Y1);
    float en = metal::precise::sqrt(ex * ex + ey * ey);
    bool far = en > MN_TINY;
    ex = far ? ex / en : 1.0f;
    ey = far ? ey / en : 0.0f;
    T cmu2[2];
    for (int q = 0; q < 2; ++q) cmu2[q] = 1.0f - (cor.px[q] * cor.px[q] + cor.py[q] * cor.py[q]);
    T limb_th[4];
    bool limb_ok[4];
    for (int k = 0; k < 2; ++k) {
        T x0 = cx[k], y0 = cy[k], R = cr[k];
        float ox = mv(cx[1 - k]), oy = mv(cy[1 - k]), oR = mv(cr[1 - k]);
        T z2 = x0 * x0 + y0 * y0;
        bool cen = mv(z2) < MN_TINY;
        T z = cen ? T(0.0f) : m_sqrt(z2);
        T ux = cen ? T(-1.0f) : -x0 / z;
        T uy = cen ? T(0.0f) : -y0 / z;
        T bp[4], vals[4];
        bool bv[4];
        for (int q = 0; q < 2; ++q) {
            T px = limb[k].px[q], py = limb[k].py[q];
            limb_th[2 * k + q] = m_atan2(ex * py - ey * px, ex * px + ey * py);
            limb_ok[2 * k + q] = limb[k].has;
            T fx = limb[k].o1x[q], fy = limb[k].o1y[q];
            bp[q] = m_atan2(ux * fy - uy * fx, ux * fx + uy * fy);
            bv[q] = limb[k].has;
            vals[q] = T(0.0f);
            T gx_ = (k == 0) ? cor.o1x[q] : cor.o2x[q];
            T gy_ = (k == 0) ? cor.o1y[q] : cor.o2y[q];
            bp[2 + q] = m_atan2(ux * gy_ - uy * gx_, ux * gx_ + uy * gy_);
            bv[2 + q] = cor.has;
            vals[2 + q] = cmu2[q];
        }
        T LO[4], HI[4], VLO[4], VHI[4];
        bool OK[4];
        mn_sorted(bp, bv, vals, -MP_PI, LO, HI, OK, VLO, VHI);
        bool cut = bv[0] || bv[1] || bv[2] || bv[3];
        T t0 = (1.0f - z) * (1.0f + z) - R * R;
        T cxe = x0 - ex, cye = y0 - ey;
        float gx = mv(x0), gy = mv(y0), gR = mv(R), gux = mv(ux), guy = mv(uy);
        for (int i = 0; i < 4; ++i) {
            if (!OK[i]) continue;
            T lo = LO[i], hi = HI[i];
            if (mv(lo) > 0.0f && mv(hi) > MP_PI) { lo = lo - MP_TWO_PI; hi = hi - MP_TWO_PI; }
            // membership at three points of the arc, frozen
            float gl = mv(lo), gh = mv(hi), ds = 0.0f, dother = 0.0f;
            for (int s = 0; s < 3; ++s) {
                float an = gl + MN_SAMPLES[s] * (gh - gl);
                float sp_ = metal::precise::sin(an), cp_ = metal::precise::cos(an);
                float xs = gx + gR * (gux * cp_ - guy * sp_);
                float ys = gy + gR * (gux * sp_ + guy * cp_);
                float ms = 1.0f - (xs * xs + ys * ys);
                float u_ = (xs - ox) / oR, v_ = (ys - oy) / oR;
                float mo = 1.0f - (u_ * u_ + v_ * v_);
                ds = (s == 0) ? ms : mn_decide(ds, ms);
                dother = (s == 0) ? mo : mn_decide(dother, mo);
            }
            if (!(ds > 0.0f && dother > 0.0f)) continue;
            T sl = m_sin(lo), cl = m_cos(lo), sh = m_sin(hi), ch = m_cos(hi);
            T cols[MN_NC];
            mn_arc_cols(z, R, lo, hi, sl, cl, sh, ch, cols);
            // E x X and F at each end, X - E from the centre's offset from E
            T dxh = cxe + R * (ux * ch - uy * sh), dyh = cye + R * (ux * sh + uy * ch);
            T dxl = cxe + R * (ux * cl - uy * sl), dyl = cye + R * (ux * sl + uy * cl);
            T eh = ex * dyh - ey * dxh, el = ex * dyl - ey * dxl;
            T mu0 = t0 + 2.0f * z * R;
            T fh[MN_NC], fl[MN_NC];
            mn_green(cut ? VHI[i] : mu0, fh);
            mn_green(cut ? VLO[i] : mu0, fl);
            for (int c = 0; c < MN_NC; ++c) occ[c] = occ[c] + (cols[c] - (fh[c] * eh - fl[c] * el));
        }
    }
    // the limb's arcs inside both circles
    T zeros[4] = {T(0.0f), T(0.0f), T(0.0f), T(0.0f)};
    T llo[4], lhi[4], vl[4], vh[4];
    bool lok[4];
    mn_sorted(limb_th, limb_ok, zeros, 0.0f, llo, lhi, lok, vl, vh);
    float gx0[2] = {mv(X0), mv(X1)}, gy0[2] = {mv(Y0), mv(Y1)}, gR0[2] = {mv(R0), mv(R1)};
    for (int i = 0; i < 4; ++i) {
        if (!lok[i]) continue;
        float glo = mv(llo[i]), ghi = mv(lhi[i]), d[2] = {0.0f, 0.0f};
        for (int s = 0; s < 3; ++s) {
            float an = glo + MN_SAMPLES[s] * (ghi - glo);
            float sr = metal::precise::sin(an), cr_ = metal::precise::cos(an);
            float cm = ex * cr_ - ey * sr, sm = ey * cr_ + ex * sr;
            for (int j = 0; j < 2; ++j) {
                float u = (cm - gx0[j]) / gR0[j], v = (sm - gy0[j]) / gR0[j];
                float m = 1.0f - (u * u + v * v);
                d[j] = (s == 0) ? m : mn_decide(d[j], m);
            }
        }
        if (!(d[0] > 0.0f && d[1] > 0.0f)) continue;
        T arc = (lhi[i] - llo[i]) - (m_sin(lhi[i]) - m_sin(llo[i]));
        for (int c = 0; c < MN_NC; ++c) occ[c] = occ[c] + MN_H1[c] * arc;
    }
}
"""

_MN_QUAD = r"""
// ====================== the quadratic law's closure =====================
constant float MN_SEP_FLOOR = 1.44e-12f;   // (10 eps)^2 (moons._separation)
constant float MN_RTINY = 1.2e-7f;         // moons._disc
constant float MN_C1 = CONST_C1;           // -2 pi / 3: the linear law's flux to mu^1

// f(z, r) with its partials, carried to the inputs
inline float mn_zr(float f, float fz, float fr, float z, float r) { return f; }
inline md mn_zr(float f, float fz, float fr, md z, md r) {
    return md(f, fz * z.a + fr * r.a, fz * z.b + fr * r.b);
}
template <typename T> inline T mn_clip01(T x) {
    return (mv(x) < 0.0f) ? T(0.0f) : ((mv(x) > 1.0f) ? T(1.0f) : x);
}

// the closure for mu^1 from the moments m_0, m_2, m_4 (moons._lens_mu)
template <typename T> inline T mn_lens_mu(T m0, T m2, T m4) {
    bool ok = mv(m0) > MN_TINY;
    T m0s = ok ? m0 : T(1.0f);
    T mean = m_max((ok ? m2 : T(0.5f)) / m0s, MN_TINY);
    T var = (ok ? m4 : T(0.3f)) / m0s - mean * mean;
    bool spread = mv(var) > 0.0f;
    T hw = spread ? m_sqrt(3.0f * var) : T(0.0f);
    var = spread ? var : T(0.0f);
    bool limb = mv(hw) > mv(mean);
    T a = limb ? mean : mean - hw;
    T b = mean + hw;
    T ra = m_sqrt(a), rb = m_sqrt(b);
    T uni = (2.0f / 3.0f) * (a + ra * rb + b) / (ra + rb);
    T q = limb ? mean * mean / var : T(1.0f);
    T kp1 = q / (m_sqrt(1.0f + q) + 1.0f);
    T pw = m_sqrt(m_max(mean * kp1 * (kp1 + 1.0f), MN_TINY)) / (kp1 + 0.5f);
    return ok ? m0s * (limb ? pw : uni) : T(0.0f);
}

// a disc's occulted moments [mu^0, mu^2, mu^4, mu^1] (moons._lens_mu1's D)
template <typename T> inline void mn_disc_moments(T x, T y, T r, thread T *d) {
    bool ok = mv(r) > MN_RTINY;
    T rr = ok ? r : T(0.1f);
    T z = m_sqrt(m_max(x * x + y * y, MN_SEP_FLOOR));
    float G[HYB_NGEN], Gz[HYB_NGEN], Gr[HYB_NGEN];
    mp_hyb_gen(mv(z), mv(rr), G, Gz, Gr);
    float fz, fr, f1, f2;
    float F = mp_phot_d(mv(z), mv(rr), 1.0f, 0.0f, &fz, &fr, &f1, &f2);
    for (int n = 0; n < 3; ++n) d[n] = ok ? mn_zr(-G[n], -Gz[n], -Gr[n], z, rr) : T(0.0f);
    d[3] = ok ? mn_zr(F * MN_C1, fz * MN_C1, fr * MN_C1, z, rr) : T(0.0f);
}

// the lens's mu^1 (moons._lens_mu1)
template <typename T>
inline T mn_mu1(T m0, T m2, T m4, T X0, T Y0, T R0, T X1, T Y1, T R1) {
    bool small = mv(R1) <= mv(R0);
    T d[4];
    mn_disc_moments(small ? X1 : X0, small ? Y1 : Y0, small ? R1 : R0, d);
    T w = (mv(d[0]) > MN_TINY) ? mn_clip01(m0 / d[0]) : T(0.0f);
    T c0 = d[0] - m0;
    bool rest = mv(c0) > 1e-5f * mv(d[0]) + 3e-8f;
    T c2 = rest ? d[1] - m2 : T(0.0f), c4 = rest ? d[2] - m4 : T(0.0f);
    c0 = rest ? c0 : T(0.0f);
    T t = mn_clip01((w - 0.9f) / 0.1f);
    t = t * t * (3.0f - 2.0f * t);
    return (1.0f - t) * mn_lens_mu(m0, m2, m4) + t * (d[3] - mn_lens_mu(c0, c2, c4));
}
"""

_QLENS_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float m[3];
    mn_lens<float>(xp[i], yp[i], rp[i], xm[i], ym[i], rs[i], m);
    out[3u * i] = m[0];
    out[3u * i + 1u] = m[1];
    out[3u * i + 2u] = mn_mu1<float>(m[0], m[1], m[2], xp[i], yp[i], rp[i], xm[i], ym[i], rs[i]);
"""

_QLENS_VJP_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float2 o = float2(0.0f);
    md X0 = md(xp[i], float4(1.0f, 0.0f, 0.0f, 0.0f), o);
    md Y0 = md(yp[i], float4(0.0f, 1.0f, 0.0f, 0.0f), o);
    md R0 = md(rp[i], float4(0.0f, 0.0f, 1.0f, 0.0f), o);
    md X1 = md(xm[i], float4(0.0f, 0.0f, 0.0f, 1.0f), o);
    md Y1 = md(ym[i], float4(0.0f), float2(1.0f, 0.0f));
    md R1 = md(rs[i], float4(0.0f), float2(0.0f, 1.0f));
    md m[3];
    mn_lens<md>(X0, Y0, R0, X1, Y1, R1, m);
    md mu1 = mn_mu1<md>(m[0], m[1], m[2], X0, Y0, R0, X1, Y1, R1);
    float c0 = ct[3u * i], c1 = ct[3u * i + 1u], c2 = ct[3u * i + 2u];
    float4 ga = c0 * m[0].a + c1 * m[1].a + c2 * mu1.a;
    float2 gb = c0 * m[0].b + c1 * m[1].b + c2 * mu1.b;
    for (int c = 0; c < 4; ++c) g[6u * i + (uint)c] = ga[c];
    g[6u * i + 4u] = gb.x;
    g[6u * i + 5u] = gb.y;
"""

_LENS_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float occ[MN_NC];
    mn_lens<float>(xp[i], yp[i], rp[i], xm[i], ym[i], rs[i], occ);
    for (int c = 0; c < MN_NC; ++c) out[(uint)MN_NC * i + (uint)c] = occ[c];
"""

_LENS_VJP_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    md occ[MN_NC];
    float2 o = float2(0.0f);
    mn_lens<md>(md(xp[i], float4(1.0f, 0.0f, 0.0f, 0.0f), o),
                md(yp[i], float4(0.0f, 1.0f, 0.0f, 0.0f), o),
                md(rp[i], float4(0.0f, 0.0f, 1.0f, 0.0f), o),
                md(xm[i], float4(0.0f, 0.0f, 0.0f, 1.0f), o),
                md(ym[i], float4(0.0f), float2(1.0f, 0.0f)),
                md(rs[i], float4(0.0f), float2(0.0f, 1.0f)), occ);
    float4 ga = float4(0.0f);
    float2 gb = float2(0.0f);
    for (int c = 0; c < MN_NC; ++c) {
        float w = ct[(uint)MN_NC * i + (uint)c];
        ga += w * occ[c].a;
        gb += w * occ[c].b;
    }
    for (int c = 0; c < 4; ++c) g[6u * i + (uint)c] = ga[c];
    g[6u * i + 4u] = gb.x;
    g[6u * i + 5u] = gb.y;
"""


def _mn_header(eps) -> str:
    K = len(eps)
    pad = lambda v: ", ".join(_f(x) for x in v) if K else "0.0f"     # noqa: E731
    src = (_MN_FN.replace("CONST_KA", str(max(K, 1)))
                 .replace("CONST_K", str(K))
                 .replace("CONST_EPS", pad(eps))
                 .replace("CONST_SP", pad([math.sqrt(1.0 + e) for e in eps]))
                 .replace("CONST_P4", pad([4.0 * (1.0 + e) for e in eps]))
                 .replace("CONST_P2", pad([2.0 * (1.0 + e) for e in eps]))
                 .replace("CONST_H1", ", ".join(_f(x) for x in [
                     1.0 / (2.0 * (n + 1.0)) for n in range(3)]
                     + [1.0 / (2.0 * (1.0 + e) * e) for e in eps] + ([0.0] if not K else []))))
    assert "CONST_" not in src, "unexpanded marker in the moon header"
    return src


_LENS_INS = ["xp", "yp", "rp", "xm", "ym", "rs"]


def _lens_kernels(eps):
    """(fwd, vjp) of the lens's generator columns for poles eps, or with eps
    None the quadratic law's (mu^0, mu^2, mu^1)."""
    quad = eps is None
    eps = () if quad else tuple(float(e) for e in eps)
    key = ("moon-lens", "quad" if quad else eps)
    if key not in M._kernels:
        if quad:
            # the smaller disc's moments: the quadratic photometry and the
            # hybrid even columns (any law's; the lens is never a large
            # occultor's, as the graph's even_cols outside a declaration)
            hdr = (M._HEADER + M._phot_header() + _hyb_header(get_law("hybrid2"))
                   + _mn_header(()) + _MN_QUAD.replace("CONST_C1", _f(-2.0 * math.pi / 3.0)))
            tag, srcs = "quad", (_QLENS_SRC, _QLENS_VJP_SRC)
        else:
            hdr = M._HEADER + _mn_header(eps)
            tag = hashlib.sha1(repr(eps).encode()).hexdigest()[:10]
            srcs = (_LENS_SRC, _LENS_VJP_SRC)
        M._kernels[key] = (
            mx.fast.metal_kernel(name=f"mn_lens_{tag}", input_names=_LENS_INS + ["npts"],
                                 output_names=["out"], header=hdr, source=srcs[0]),
            mx.fast.metal_kernel(name=f"mn_lens_vjp_{tag}",
                                 input_names=_LENS_INS + ["ct", "npts"],
                                 output_names=["g"], header=hdr, source=srcs[1]))
    return M._kernels[key]


_CORES: dict = {}


def _lens_core(eps):
    """custom_function (xp, yp, rp, xm, ym, rs) -> (L, 3 + K), each (L,);
    eps None: the quadratic law's (L, 3)."""
    eps = None if eps is None else tuple(float(e) for e in eps)
    if ("lens", eps) in _CORES:
        return _CORES[("lens", eps)]
    nc = 3 if eps is None else 3 + len(eps)

    @mx.custom_function
    def core(xp, yp, rp, xm, ym, rs):
        n = xp.shape[0]
        return _lens_kernels(eps)[0](
            inputs=[xp, yp, rp, xm, ym, rs, n], output_shapes=[(n, nc)],
            output_dtypes=[mx.float32], grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]

    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = primals[0].shape[0]
        g = _lens_kernels(eps)[1](
            inputs=[*primals, ct, n], output_shapes=[(n, 6)],
            output_dtypes=[mx.float32], grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]
        return tuple(g[:, j] for j in range(6))
    core.vjp(_vjp)
    _CORES[("lens", eps)] = core
    return core


def _points(vs):
    """Arrays (or host values) broadcast to their common shape and
    flattened, float32, and that shape."""
    vs = [(v if isinstance(v, mx.array) else mx.array(v)).astype(mx.float32) for v in vs]
    shape = tuple(np.broadcast_shapes(*(v.shape for v in vs)))
    n = int(np.prod(shape)) if shape else 1
    return [mx.reshape(mx.broadcast_to(v, shape), (n,)) for v in vs], shape


def lens_cols_kernel(xp, yp, rp, xm, ym, rs, eps):
    """``moons._lens_cols`` by the fp32 kernel: the lens's occulted
    generator columns [mu^0, mu^2, mu^4, P_eps...] (positive), a list, for
    points (L,) -- rp, rs may be shared -- differentiable in every input."""
    vs, shape = _points((xp, yp, rp, xm, ym, rs))
    out = _lens_core(eps)(*vs)
    return [mx.reshape(out[:, c], shape) for c in range(out.shape[1])]


def lens_quad_kernel(xp, yp, rp, xm, ym, rs):
    """The quadratic law's lens by the fp32 kernel: its occulted mu^0 and
    mu^2 (``moons._lens_cols``) and its mu^1 (``moons._lens_mu1``), a list,
    as ``lens_cols_kernel``."""
    vs, shape = _points((xp, yp, rp, xm, ym, rs))
    out = _lens_core(None)(*vs)
    return [mx.reshape(out[:, c], shape) for c in range(3)]


# ---------------------------------------------------------------------------
# a disc's shape columns, one point per thread with its own radius
# ---------------------------------------------------------------------------

_DISC_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float B[HYB_NCOL];
    mp_hyb_cols(fabs(zin[i]), rin[i], B);
    for (int q = 0; q < HYB_NCOL; ++q) out[(uint)HYB_NCOL * i + (uint)q] = B[q];
"""

_DISC_VJP_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float wct[HYB_NCOL];
    for (int q = 0; q < HYB_NCOL; ++q) wct[q] = ct[(uint)HYB_NCOL * i + (uint)q];
    float dz, dr, d1, d2;
    mp_hyb_cols_bd(fabs(zin[i]), rin[i], wct, &dz, &dr, &d1, &d2);
    gz[i] = dz;
    gr[i] = dr;
"""


def _disc_kernels(law, large):
    key = ("moon-disc", law.definition, large)
    if key not in M._kernels:
        hdr = M._HEADER + _hyb_header(law, large)
        tag = f"{_law_tag(law)}{'_large' if large else ''}"
        M._kernels[key] = (
            mx.fast.metal_kernel(name=f"mn_disc_{tag}", input_names=["zin", "rin", "npts"],
                                 output_names=["out"], header=hdr, source=M._subst(_DISC_SRC)),
            mx.fast.metal_kernel(name=f"mn_disc_vjp_{tag}",
                                 input_names=["zin", "rin", "ct", "npts"],
                                 output_names=["gz", "gr"], header=hdr,
                                 source=M._subst(_DISC_VJP_SRC)))
    return M._kernels[key]


def _disc_core(law, large):
    key = ("disc", law.definition, large)
    if key in _CORES:
        return _CORES[key]
    ncol = law.n_col

    @mx.custom_function
    def core(z, r):
        n = z.shape[0]
        return _disc_kernels(law, large)[0](
            inputs=[z, r, n], output_shapes=[(n, ncol)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]

    def _vjp(primals, cotangent, output):
        z, r = primals
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = z.shape[0]
        return tuple(_disc_kernels(law, large)[1](
            inputs=[z, r, ct, n], output_shapes=[(n,), (n,)],
            output_dtypes=[mx.float32] * 2, grid=(n, 1, 1), threadgroup=(256, 1, 1)))
    core.vjp(_vjp)
    _CORES[key] = core
    return core


def disc_cols_kernel(z, r, law):
    """A disc's shape columns [E0, S_1..S_n] (deviation form) at separation z
    and radius r (broadcast), a list, by the fp32 kernel; under the caller's
    large_occultors declaration, as ``hybrid.shape_cols``."""
    law = get_law(law)
    large = _may_be_large(r) is not False
    z, r = (v if isinstance(v, mx.array) else mx.array(v, dtype=mx.float32) for v in (z, r))
    shape = tuple(np.broadcast_shapes(z.shape, r.shape))
    n = int(np.prod(shape)) if shape else 1
    flat = lambda v: mx.reshape(mx.broadcast_to(v.astype(mx.float32), shape), (n,))  # noqa: E731
    out = _disc_core(law, large)(flat(z), flat(r))
    return [mx.reshape(out[:, c], shape) for c in range(law.n_col)]


_QDISC_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    out[i] = mp_phot(fabs(zin[i]), rin[i], u1in[i], u2in[i]);
"""

_QDISC_VJP_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float dz, dr, d1, d2;
    mp_phot_d(fabs(zin[i]), rin[i], u1in[i], u2in[i], &dz, &dr, &d1, &d2);
    float c = ct[i];
    gz[i] = c * dz;
    gr[i] = c * dr;
    gu1[i] = c * d1;
    gu2[i] = c * d2;
"""


def _qdisc_core():
    if "qdisc" in _CORES:
        return _CORES["qdisc"]
    if ("moon-qdisc",) not in M._kernels:
        hdr = M._HEADER + M._phot_header()
        ins = ["zin", "rin", "u1in", "u2in"]
        M._kernels[("moon-qdisc",)] = (
            mx.fast.metal_kernel(name="mn_qdisc", input_names=ins + ["npts"],
                                 output_names=["out"], header=hdr, source=_QDISC_SRC),
            mx.fast.metal_kernel(name="mn_qdisc_vjp", input_names=ins + ["ct", "npts"],
                                 output_names=["gz", "gr", "gu1", "gu2"], header=hdr,
                                 source=_QDISC_VJP_SRC))
    fwd, vjp = M._kernels[("moon-qdisc",)]

    @mx.custom_function
    def core(z, r, u1, u2):
        n = z.shape[0]
        return fwd(inputs=[z, r, u1, u2, n], output_shapes=[(n,)], output_dtypes=[mx.float32],
                   grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]

    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = primals[0].shape[0]
        return tuple(vjp(inputs=[*primals, ct, n], output_shapes=[(n,)] * 4,
                         output_dtypes=[mx.float32] * 4, grid=(n, 1, 1),
                         threadgroup=(256, 1, 1)))
    core.vjp(_vjp)
    _CORES["qdisc"] = core
    return core


def disc_quad_kernel(z, r, u1, u2):
    """A disc's F - 1 for the quadratic law at separation z, radius r and
    coefficients u1, u2 (broadcast), by the fp32 kernel (``flux.flux_dev``)."""
    vs, shape = _points((z, r, u1, u2))
    return mx.reshape(_qdisc_core()(*vs), shape)
