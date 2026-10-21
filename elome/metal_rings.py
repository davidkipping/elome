"""Fused fp32 Metal kernels for ringed planets (hybrid laws).

The mathematics is ``rings.py``'s (read its docstring first): the occultor
is a weighted region, and its flux is every outline's arcs -- the planet's
and the two ring edges' -- weighted by the jump in the blocked fraction
across them, plus the limb arcs weighted by the fraction there. This
module transcribes it per point, in registers, on top of the oblate
kernels' device functions (``metal_oblate``: the root solves, the per-arc
pole integral, its dual numbers), which it includes and does not change.

Per point and outline:

* **detached** (plain float): the limb crossings (up to four), the corners
  with the other outlines (closed form: concentric, coaxial), the sorted
  sub-arcs and, at each sub-arc's midpoint, which side of the star and of
  the other outlines it lies on (corners decide, never a point test
  against another outline), the weight jump kappa = a + b T as integers
  (a, b), each pole level's roots (an ellipse's by the oblate solver, a
  circle's -- a spherical planet, a face-on ring -- in closed form) and,
  per arc, the near-double pair selection;
* **live** (``rg_outline<T>``, T = float or the dual number ``dv``): the
  coefficients, each arc's even Fourier sums and pole partial fractions
  (the oblate partial branch, the star term left out), and the arc's
  gradient-only corner term h(s) [C(lo) - C(hi)] at both ends, h at the
  corner (s < 1 at a planet-ring corner).

A dual number carries d/d(x0, y0, A, B) of ONE outline, so each outline is
evaluated with its own seeds and the results chained by hand: the
principal frame, the planet's axes from (r, f), the edges' from (r,
ring_in, ring_out, c = 1 - ring_f). The blocked fraction T enters linearly
(kappa = a + b T), so its derivative is the b-weighted sum of the values.
T = 1 - exp(-tau / c) itself is formed outside the kernel (MLX ops, whose
autodiff chains it), since fp32 has no accurate expm1 on the device.

Exposures: the contact rule splits each window at the kinks inside it
(``rings_tau``: every outline's limb tangencies, detached), at most four,
so 5 * n_gl nodes as the other kernels.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from . import metal_oblate as MO
from .hybrid import get_law
from .metal_hybrid import _law_tag

__all__ = ["flux_dev_from_tau_ring_kernel"]

_NKINK = 24                     # rings_tau: 3 outlines x _N_EVENTS


# ---------------------------------------------------------------------------
# device code
# ---------------------------------------------------------------------------

_RG_FN = r"""
// ============================ ringed planets ============================
constant float RG_BIG = 1e8f;           // the root at infinity (rings._BIG)
constant float RG_S1C = 1e-8f;          // a centred circle: R d below this
constant float RG_AL[3][3] = {{0.5f, 0.0f, 0.0f}, {0.5f, -0.25f, 0.0f},
                              {0.5f, -0.5f, 1.0f / 6.0f}};

// the detached geometry of one point: centre, the outlines' axes (0 the
// planet, 1 the inner edge, 2 the outer), and their limb crossings
struct rg_geo {
    float x0, y0;
    float A[3], B[3];
    bool circ[3];
    int nl[3];
    float lp[3][4];
};

inline void rg_sort(thread float *v, int n) {
    for (int i = 1; i < n; ++i) {
        float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; --j; }
        v[j + 1] = key;
    }
}

// all limb crossings of an outline on its own parameter (rings._crossings)
inline int rg_limb(float x0, float y0, float a, float b, bool circ,
                   thread float *ph) {
    if (circ) {
        float d2 = x0 * x0 + y0 * y0;
        if (!(d2 > 0.0f)) return 0;
        float d = metal::precise::sqrt(d2);
        float cq = ((1.0f - a) * (1.0f + a) - d2) / (2.0f * d * a);
        if (!(fabs(cq) < 1.0f)) return 0;
        float g = metal::precise::atan2(y0, x0);
        float h = metal::precise::atan2(metal::precise::sqrt(max(1.0f - cq * cq, 0.0f)), cq);
        ph[0] = ob_mod2pi(g + h);
        ph[1] = ob_mod2pi(g - h);
        return 2;
    }
    float S0 = x0 * x0 + y0 * y0 + 0.5f * (a * a + b * b);
    float2 S1 = float2(a * x0, -(b * y0));
    float S2 = 0.25f * (a * a - b * b);
    float2 z[4];
    if (ob_use_ferrari(S1, S2)) ob_roots_ferrari(S0, S1, S2, 1.0f, z);
    else                        ob_roots_qr(S0, S1, S2, 1.0f, z);
    float sc = 1.0f + (fabs(x0) + a) * (fabs(x0) + a)
                    + (fabs(y0) + b) * (fabs(y0) + b);
    float phis[4]; bool valid[4];
    for (int i = 0; i < 4; ++i) {
        bool cand = fabs(cab(z[i]) - 1.0f) < OB_ZTOL;
        float phi = cand ? carg(z[i]) : 0.0f;
        float g, dg;
        ob_g_dg(phi, x0, y0, a, b, &g, &dg);
        for (int it = 0; it < 3; ++it) {
            float trial = phi - ((fabs(dg) > 0.0f) ? g / dg : 0.0f);
            float gt, dgt;
            ob_g_dg(trial, x0, y0, a, b, &gt, &dgt);
            if (fabs(gt) < fabs(g)) { phi = trial; g = gt; dg = dgt; }
        }
        phis[i] = ob_mod2pi(phi);
        valid[i] = cand && (fabs(g) <= OB_GTOL * sc);
    }
    bool drop[4] = {false, false, false, false};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float d = fabs(ob_mod2pi(phis[i] - phis[j] + MP_PI) - MP_PI);
            if (valid[i] && valid[j] && d < OB_PAIRTOL) { drop[i] = true; drop[j] = true; }
        }
    int cnt = 0;
    for (int i = 0; i < 4; ++i)
        if (valid[i] && !drop[i]) ph[cnt++] = phis[i];
    return cnt;
}

// a circle's roots of D at level p (rings._circle_roots): 0, the root at
// infinity, and q's two roots by Vieta -- not a root and its partner
inline void rg_circle_roots(float x0, float y0, float R, float p,
                            thread float2 *z) {
    float2 s1 = float2(R * x0, -(R * y0));
    float b_ = x0 * x0 + y0 * y0 + R * R - p;
    float disc = b_ * b_ - 4.0f * ca2(s1);
    float2 sq = (disc >= 0.0f) ? float2(metal::precise::sqrt(disc), 0.0f)
                               : float2(0.0f, metal::precise::sqrt(-disc));
    float sgn = (b_ >= 0.0f) ? 1.0f : -1.0f;
    float2 qq = float2(-0.5f * (b_ + sgn * sq.x), -0.5f * sgn * sq.y);
    z[0] = float2(0.0f);
    if (ca2(s1) > 0.0f && ca2(qq) > 0.0f) {
        z[1] = cdiv(qq, s1);
        z[2] = cdiv(cj(s1), qq);
    } else {
        z[1] = float2(RG_BIG, 0.0f);
        z[2] = float2(0.0f);
    }
    z[3] = float2(RG_BIG, 0.0f);
}

inline void rg_roots(float x0, float y0, float a, float b, bool circ, float p,
                     thread float2 *z) {
    if (circ) { rg_circle_roots(x0, y0, a, p, z); return; }
    float S0 = x0 * x0 + y0 * y0 + 0.5f * (a * a + b * b);
    float2 S1 = float2(a * x0, -(b * y0));
    float S2 = 0.25f * (a * a - b * b);
    if (ob_use_ferrari(S1, S2)) ob_roots_ferrari(S0, S1, S2, p, z);
    else                        ob_roots_qr(S0, S1, S2, p, z);
}

// outline k against outline j (rings._pair): F = num cos^2 - den sin^2
inline void rg_pair(float ak, float bk, float aj, float bj, thread float *num,
                    thread float *den, thread bool *cr, thread float *ph) {
    *num = (ak - aj) * (ak + aj) / (aj * aj);
    *den = (bj - bk) * (bj + bk) / (bj * bj);
    *cr = ((*num > 0.0f) && (*den > 0.0f)) || ((*num < 0.0f) && (*den < 0.0f));
    *ph = metal::precise::atan2(metal::precise::sqrt(fabs(*num)),
                                metal::precise::sqrt(fabs(*den)));
}
inline bool rg_inside(float num, float den, bool cr, float c2, float s2) {
    return cr ? (num * c2 < den * s2) : (num < den);
}
// blocked fraction a + b T for flags [planet, inner, outer]
inline void rg_w(thread const bool *f, thread int *a, thread int *b) {
    *a = f[0] ? 1 : 0;
    *b = (!f[0] && f[2] && !f[1]) ? 1 : 0;
}

inline void rg_geom(float X, float Y, float r, float fl, float cth, float sth,
                    float sqf, float ri, float ro, float cr, thread rg_geo &G) {
    G.x0 = X * cth + Y * sth;
    G.y0 = -X * sth + Y * cth;
    bool pc = fl < OB_FSW;
    G.A[0] = pc ? r : r / sqf;
    G.B[0] = pc ? r : r * sqf;
    G.circ[0] = pc;
    bool ec = (1.0f - cr) < OB_FSW;
    float R1 = ri * r, R2 = ro * r;
    G.A[1] = R1; G.B[1] = ec ? R1 : R1 * cr; G.circ[1] = ec;
    G.A[2] = R2; G.B[2] = ec ? R2 : R2 * cr; G.circ[2] = ec;
    for (int k = 0; k < 3; ++k)
        G.nl[k] = rg_limb(G.x0, G.y0, G.A[k], G.B[k], G.circ[k], G.lp[k]);
}

// ============================ live evaluation ============================
// one arc's generator integrals [mu^0, mu^2, mu^4, P_k...] (the oblate
// partial branch, without the star term)
template <typename T>
inline void rg_arc(T x0, T y0, T A, T S0, cx<T> S1, T S2, T K0, cx<T> K1,
                   thread const cx<T> *c0, thread const cx<T> *c1,
                   thread const cx<T> *c2, thread const float2 *zall, bool cen,
                   float lo, float hi, thread T *out) {
    float mid = 0.5f * (lo + hi), hlf = 0.5f * (hi - lo);
    T E[3];
    for (int p = 0; p < 3; ++p) {
        int mmax = 2 * p + 1;
        T acc = ((p == 0) ? c0[0].re : ((p == 1) ? c1[0].re : c2[0].re)) * (hi - lo);
        for (int m = 1; m <= mmax; ++m) {
            cx<T> cm_ = (p == 0) ? c0[m] : ((p == 1) ? c1[m] : c2[m]);
            float2 em = cexpi_live((float)m * mid);
            float sm = metal::precise::sin((float)m * hlf);
            acc = acc + (4.0f / (float)m) * sm * (cm_.re * em.x - cm_.im * em.y);
        }
        E[p] = acc;
    }
    for (int n = 0; n < 3; ++n) {
        T acc = T(0.0f);
        for (int p = 0; p <= n; ++p) acc = acc + RG_AL[n][p] * E[p];
        out[n] = acc;
    }
    float2 w1 = cexpi_live(lo), w2 = cexpi_live(hi);
    for (int k = 0; k < HYB_K; ++k) {
        float p = 1.0f + HYB_POLE[k];
        if (cen) {
            // a circle centred on the star: first order in the offset and
            // in A - B, A B h(s) integrated with s = A^2 cos^2 + B^2 sin^2
            // about R^2 = A B -- the arc's own d/dA, d/dB (A^2 - B^2 = 4 S2)
            T R2 = K0;
            T c0v = R2 / (2.0f * p * (p - R2));
            T c1v = A * (p + R2) / (2.0f * p * (p - R2) * (p - R2));
            T c2v = R2 / (2.0f * p * (p - R2) * (p - R2));
            float sl = metal::precise::sin(lo), cl = metal::precise::cos(lo);
            float sh = metal::precise::sin(hi), ch = metal::precise::cos(hi);
            out[3 + k] = c0v * (hi - lo) + c1v * (x0 * (sh - sl) + y0 * (cl - ch))
                       + c2v * (2.0f * S2) * (sh * ch - sl * cl);
            continue;
        }
        thread const float2 *z0 = &zall[4 * k];
        int pj[2], pl[2];
        bool pg[2];
        ob_pairs(z0, lo, hi, pj, pl, pg);
        bool used[4] = {false, false, false, false};
        for (int r = 0; r < 2; ++r)
            if (pg[r]) { used[pj[r]] = true; used[pl[r]] = true; }
        cx<T> ell = c_const<T>(float2(0.0f));
        for (int q = 0; q < 4; ++q) {
            if (used[q]) continue;
            if (cab(z0[q]) > OB_RECIP) {
                float m = cab(z0[q]);
                float2 wf = (z0[q] / m) / m;
                float S0f = t_val(S0), S2f = t_val(S2);
                float2 S1f = c_val(S1);
                for (int it = 0; it < 2; ++it)
                    wf = wf - cdiv(ob_D(wf, S0f, S1f, S2f, p),
                                   ob_dD(wf, S0f, S1f, S2f, p));
                cx<T> w = ob_live(wf, S0, S1, S2, p);
                cx<T> dDw = ob_dDt(w, S0, S1, S2, p);
                if (ca2(c_val(dDw)) == 0.0f) continue;
                cx<T> R = -c_i(c_conj(ob_Nt(w, K0, K1) / dDw));
                cx<T> one = c_const<T>(float2(1.0f, 0.0f));
                cx<T> a2 = one - c_const<T>(w2) * c_conj(w);
                cx<T> a1 = one - c_const<T>(w1) * c_conj(w);
                ell = ell + R * cx<T>(c_logabs(a2) - c_logabs(a1),
                                      c_arg(a2) - c_arg(a1));
                continue;
            }
            cx<T> z = ob_live(z0[q], S0, S1, S2, p);
            cx<T> dD = ob_dDt(z, S0, S1, S2, p);
            if (ca2(c_val(dD)) == 0.0f) continue;
            cx<T> R = -(ob_Nt(z, K0, K1) / c_i(dD));
            ell = ell + R * ob_dlog(z, lo, hi, cab(z0[q]) < 1.0f, w1, w2);
        }
        cx<T> dc[5] = {c_real(S2), c_conj(S1), c_real(S0 - p), S1, c_real(S2)};
        for (int r = 0; r < 2; ++r) {
            if (!pg[r]) continue;
            float2 z1 = z0[pj[r]], z2 = z0[pl[r]];
            float2 sig0 = z1 + z2, pi0 = cm(z1, z2);
            ob_pair_polish(dc, &sig0, &pi0);
            cx<T> sig, pi_;
            ob_bairstow(dc, sig0, pi0, &sig, &pi_);
            cx<T> Q2 = dc[0];
            cx<T> Q1 = S1 + sig * Q2;
            cx<T> Q0 = sig * Q1 + c_real(S0 - p) - pi_ * Q2;
            cx<T> aN = K1 * sig + c_real(K0);
            cx<T> bN = c_conj(K1) - K1 * pi_;
            cx<T> aQ = Q2 * sig + Q1;
            cx<T> bQ = Q0 - Q2 * pi_;
            cx<T> den = aQ * aQ * pi_ + aQ * bQ * sig + bQ * bQ;
            if (ca2(c_val(den)) == 0.0f) den = c_const<T>(float2(1.0f, 0.0f));
            cx<T> Acoef = c_i((aN * bQ - bN * aQ) / den);
            cx<T> Ccoef = c_scf(c_i((c_scf(aN * aQ * pi_, 2.0f)
                                     + (aN * bQ + bN * aQ) * sig
                                     + c_scf(bN * bQ, 2.0f)) / den), 0.5f);
            cx<T> W1 = c_const<T>(w1), W2 = c_const<T>(w2);
            cx<T> qh = W2 * W2 - sig * W2 + pi_;
            cx<T> ql = W1 * W1 - sig * W1 + pi_;
            T darg = c_arg(qh / ql);
            float rough = t_val(ob_dlog(c_const<float>(z1), lo, hi,
                                        cab(z1) < 1.0f, w1, w2).im)
                        + t_val(ob_dlog(c_const<float>(z2), lo, hi,
                                        cab(z2) < 1.0f, w1, w2).im);
            float wind = metal::rint((rough - t_val(darg)) / MP_TWO_PI);
            cx<T> LQ = cx<T>(c_logabs(qh) - c_logabs(ql), darg + MP_TWO_PI * wind);
            cx<T> Delta = c_scf(sig * sig, 0.25f) - pi_;
            cx<T> c = c_scf(sig, 0.5f);
            cx<T> Jd = ob_J(W2 - c, Delta) - ob_J(W1 - c, Delta);
            ell = ell + c_scf(Acoef * LQ, 0.5f) + Ccoef * Jd;
        }
        out[3 + k] = ell.re / (2.0f * p);
    }
}

// the arc's gradient-only corner terms, h(s) [C(lo) - C(hi)] per column
template <typename T>
inline void rg_corner(T x0, T y0, T A, T B, float x0f, float y0f, float Af,
                      float Bf, float lo, float hi, thread T *out) {
    float cl = metal::precise::cos(lo), sl = metal::precise::sin(lo);
    float ch = metal::precise::cos(hi), sh = metal::precise::sin(hi);
    float xl = x0f + Af * cl, yl = y0f + Bf * sl;
    float xh = x0f + Af * ch, yh = y0f + Bf * sh;
    T vlo = xl * (y0 + B * sl) - yl * (x0 + A * cl);
    T vhi = xh * (y0 + B * sh) - yh * (x0 + A * ch);
    vlo = vlo - t_nograd(vlo);
    vhi = vhi - t_nograd(vhi);
    float slo = xl * xl + yl * yl, shi = xh * xh + yh * yh;
    for (int n = 0; n < 3; ++n) {
        float hl = 0.0f, hh = 0.0f;
        for (int p = n; p >= 0; --p) { hl = hl * slo + RG_AL[n][p]; hh = hh * shi + RG_AL[n][p]; }
        out[n] = hl * vlo - hh * vhi;
    }
    for (int k = 0; k < HYB_K; ++k) {
        float p = 1.0f + HYB_POLE[k];
        out[3 + k] = vlo / (2.0f * p * (p - slo)) - vhi / (2.0f * p * (p - shi));
    }
}

// outline k: sum over its sub-arcs on the star of a * (arc + corner) into
// sa and b * (arc + corner) into sb, kappa = a + b T the weight jump
template <typename T>
inline void rg_outline(int k, T x0, T y0, T A, T B, thread const rg_geo &G,
                       thread T *sa, thread T *sb) {
    float x0f = G.x0, y0f = G.y0, Af = G.A[k], Bf = G.B[k];
    bool circ = G.circ[k];
    float bp[12];
    int n = 0;
    for (int i = 0; i < G.nl[k]; ++i) bp[n++] = G.lp[k][i];
    float num[3], den[3];
    bool cr[3];
    for (int j = 0; j < 3; ++j) {
        cr[j] = false; num[j] = 0.0f; den[j] = 0.0f;
        if (j == k || (k != 0 && j != 0)) continue;
        float ph;
        rg_pair(Af, Bf, G.A[j], G.B[j], &num[j], &den[j], &cr[j], &ph);
        if (cr[j]) {
            bp[n++] = ph; bp[n++] = MP_PI - ph;
            bp[n++] = MP_PI + ph; bp[n++] = MP_TWO_PI - ph;
        }
    }
    rg_sort(bp, n);

    T S0 = x0 * x0 + y0 * y0 + 0.5f * (A * A + B * B);
    cx<T> S1 = cx<T>(A * x0, -(B * y0));
    T S2 = 0.25f * (A * A - B * B);
    T K0 = A * B;
    cx<T> K1 = cx<T>(0.5f * (B * x0), -0.5f * (A * y0));
    cx<T> sgm[5] = {c_real(S2), c_conj(S1), c_real(S0), S1, c_real(S2)};
    cx<T> c0[2] = {c_real(K0), K1};
    cx<T> c1[4], c2[6];
    for (int m = 0; m < 4; ++m) {
        cx<T> acc = c_const<T>(float2(0.0f));
        for (int j = -2; j <= 2; ++j) {
            int q = m - j;
            if (q < -1 || q > 1) continue;
            cx<T> v = (q >= 0) ? c0[q] : c_conj(c0[-q]);
            acc = acc + sgm[j + 2] * v;
        }
        c1[m] = acc;
    }
    for (int m = 0; m < 6; ++m) {
        cx<T> acc = c_const<T>(float2(0.0f));
        for (int j = -2; j <= 2; ++j) {
            int q = m - j;
            if (q < -3 || q > 3) continue;
            cx<T> v = (q >= 0) ? c1[q] : c_conj(c1[-q]);
            acc = acc + sgm[j + 2] * v;
        }
        c2[m] = acc;
    }
    bool cen = circ && (Af * metal::precise::sqrt(x0f * x0f + y0f * y0f) < RG_S1C);
    float2 z[HYB_K * 4];
    for (int q = 0; q < HYB_K; ++q)
        rg_roots(cen ? 0.5f * Af : x0f, cen ? 0.0f : y0f, Af, Bf, circ,
                 1.0f + HYB_POLE[q], &z[4 * q]);
    // a centred circle's live pole branch is not used: keep it finite
    T px = cen ? T(0.5f * Af) : x0, py = cen ? T(0.0f) : y0;
    T pS0 = px * px + py * py + 0.5f * (A * A + B * B);
    cx<T> pS1 = cx<T>(A * px, -(B * py));
    cx<T> pK1 = cx<T>(0.5f * (B * px), -0.5f * (A * py));

    int na = (n == 0) ? 1 : n;
    for (int i = 0; i < na; ++i) {
        float lo = (n == 0) ? 0.0f : bp[i];
        float hi = (n == 0) ? MP_TWO_PI : ((i + 1 < n) ? bp[i + 1] : bp[0] + MP_TWO_PI);
        if (!(hi > lo)) continue;
        float mid = 0.5f * (lo + hi);
        float smid = metal::precise::sin(mid), cmid = metal::precise::cos(mid);
        float xs = x0f + Af * cmid, ys = y0f + Bf * smid;
        if (xs * xs + ys * ys >= 1.0f) continue;
        bool fin[3], fout[3];
        for (int j = 0; j < 3; ++j) {
            bool v;
            if (j == k) { fin[j] = true; fout[j] = false; continue; }
            if (k == 0 || j == 0) v = rg_inside(num[j], den[j], cr[j], cmid * cmid, smid * smid);
            else v = j > k;
            fin[j] = v; fout[j] = v;
        }
        int ai, bi, ao, bo;
        rg_w(fin, &ai, &bi);
        rg_w(fout, &ao, &bo);
        int ka = ai - ao, kb = bi - bo;
        if (ka == 0 && kb == 0) continue;
        T val[HYB_NGEN], cor[HYB_NGEN];
        rg_arc(x0, y0, A, pS0, pS1, S2, K0, pK1, c0, c1, c2, z, cen, lo, hi, val);
        if (!cen) {
            // even columns from the true coefficients (c0..c2 above)
        }
        rg_corner(x0, y0, A, B, x0f, y0f, Af, Bf, lo, hi, cor);
        for (int c = 0; c < HYB_NGEN; ++c) {
            T v = val[c] + cor[c];
            sa[c] = sa[c] + (float)ka * v;
            sb[c] = sb[c] + (float)kb * v;
        }
    }
}

// the limb arcs: sum of w h(1) dtheta, w = a + b T
inline void rg_limb_sum(thread const rg_geo &G, thread float *la,
                        thread float *lb) {
    for (int c = 0; c < HYB_NGEN; ++c) { la[c] = 0.0f; lb[c] = 0.0f; }
    float th[12];
    int n = 0;
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < G.nl[k]; ++i) {
            float p = G.lp[k][i];
            th[n++] = ob_mod2pi(metal::precise::atan2(
                G.y0 + G.B[k] * metal::precise::sin(p),
                G.x0 + G.A[k] * metal::precise::cos(p)));
        }
    rg_sort(th, n);
    float h1[HYB_NGEN];
    for (int c = 0; c < 3; ++c) h1[c] = 1.0f / (2.0f * ((float)c + 1.0f));
    for (int k = 0; k < HYB_K; ++k)
        h1[3 + k] = 1.0f / (2.0f * (1.0f + HYB_POLE[k]) * HYB_POLE[k]);
    int na = (n == 0) ? 1 : n;
    for (int i = 0; i < na; ++i) {
        float lo = (n == 0) ? 0.0f : th[i];
        float hi = (n == 0) ? MP_TWO_PI : ((i + 1 < n) ? th[i + 1] : th[0] + MP_TWO_PI);
        if (!(hi > lo)) continue;
        float mid = 0.5f * (lo + hi);
        float X = metal::precise::cos(mid), Y = metal::precise::sin(mid);
        bool f[3];
        for (int k = 0; k < 3; ++k) {
            float u = (X - G.x0) / G.A[k], v = (Y - G.y0) / G.B[k];
            f[k] = u * u + v * v < 1.0f;
        }
        int a, b;
        rg_w(f, &a, &b);
        float dth = hi - lo;
        for (int c = 0; c < HYB_NGEN; ++c) {
            la[c] += (float)a * h1[c] * dth;
            lb[c] += (float)b * h1[c] * dth;
        }
    }
}

// B = [E0, S_1..S_NW] of a ringed planet at sky (X, Y)
inline void rg_cols(float X, float Y, float r, float fl, float cth, float sth,
                    float sqf, float ri, float ro, float cr, float Tv,
                    thread float *Bc) {
    rg_geo G;
    rg_geom(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, G);
    float gen[HYB_NGEN];
    for (int c = 0; c < HYB_NGEN; ++c) gen[c] = 0.0f;
    float d = metal::precise::sqrt(G.x0 * G.x0 + G.y0 * G.y0);
    if (d - max(G.A[0], G.A[2]) < 1.0f) {
        float sa[HYB_NGEN], sb[HYB_NGEN], la[HYB_NGEN], lb[HYB_NGEN];
        for (int c = 0; c < HYB_NGEN; ++c) { sa[c] = 0.0f; sb[c] = 0.0f; }
        for (int k = 0; k < 3; ++k)
            rg_outline<float>(k, G.x0, G.y0, G.A[k], G.B[k], G, sa, sb);
        rg_limb_sum(G, la, lb);
        for (int c = 0; c < HYB_NGEN; ++c) gen[c] = sa[c] + la[c] + Tv * (sb[c] + lb[c]);
    }
    ob_shape(gen, Bc);
}

// wct . B and its gradient g = d/d(X, Y, r, f, theta, ring_in, ring_out,
// c = 1 - ring_f, T); B returned too
inline void rg_cols_bd(float X, float Y, float r, float fl, float cth,
                       float sth, float sqf, float ri, float ro, float cr,
                       float Tv, thread const float *wct, thread float *Bc,
                       thread float *g) {
    for (int c = 0; c < 9; ++c) g[c] = 0.0f;
    rg_geo G;
    rg_geom(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, G);
    float gen[HYB_NGEN];
    for (int c = 0; c < HYB_NGEN; ++c) gen[c] = 0.0f;
    float d = metal::precise::sqrt(G.x0 * G.x0 + G.y0 * G.y0);
    if (!(d - max(G.A[0], G.A[2]) < 1.0f)) { ob_shape(gen, Bc); return; }
    // wct . B = sum_c u[c] gen[c], B = -[gen_0, S gen]
    float u[HYB_NGEN];
    for (int c = 0; c < HYB_NGEN; ++c) {
        float acc = (c == 0) ? wct[0] : 0.0f;
        for (int j = 0; j < HYB_NW; ++j) acc += wct[j + 1] * HYB_SMAT[j * HYB_NGEN + c];
        u[c] = -acc;
    }
    float gx = 0.0f, gy = 0.0f, gT = 0.0f;
    float gA[3], gB[3];
    for (int k = 0; k < 3; ++k) {
        dv sa[HYB_NGEN], sb[HYB_NGEN];
        for (int c = 0; c < HYB_NGEN; ++c) { sa[c] = dv(0.0f); sb[c] = dv(0.0f); }
        rg_outline<dv>(k, dv(G.x0, float4(1.0f, 0.0f, 0.0f, 0.0f)),
                       dv(G.y0, float4(0.0f, 1.0f, 0.0f, 0.0f)),
                       dv(G.A[k], float4(0.0f, 0.0f, 1.0f, 0.0f)),
                       dv(G.B[k], float4(0.0f, 0.0f, 0.0f, 1.0f)), G, sa, sb);
        float4 t = float4(0.0f);
        for (int c = 0; c < HYB_NGEN; ++c) {
            gen[c] += sa[c].v + Tv * sb[c].v;
            t += u[c] * (sa[c].d + Tv * sb[c].d);
            gT += u[c] * sb[c].v;
        }
        gx += t.x; gy += t.y; gA[k] = t.z; gB[k] = t.w;
    }
    float la[HYB_NGEN], lb[HYB_NGEN];
    rg_limb_sum(G, la, lb);
    for (int c = 0; c < HYB_NGEN; ++c) {
        gen[c] += la[c] + Tv * lb[c];
        gT += u[c] * lb[c];
    }
    ob_shape(gen, Bc);
    // chain: x0 = X c + Y s, y0 = -X s + Y c
    g[0] = gx * cth - gy * sth;
    g[1] = gx * sth + gy * cth;
    g[4] = gx * G.y0 - gy * G.x0;
    // the planet: A = r / sqf, B = r sqf; the edges: A = q r, B = q r c.
    // A circle's outline (below f_sw) still carries separate d/dA, d/dB,
    // so f and ring_f keep their gradients face-on
    g[2] = gA[0] / sqf + gB[0] * sqf;
    g[3] = (gA[0] * G.A[0] - gB[0] * G.B[0]) / (2.0f * sqf * sqf);
    for (int e = 1; e < 3; ++e) {
        float q = (e == 1) ? ri : ro;
        float gR = gA[e] + cr * gB[e];
        g[2] += q * gR;
        g[4 + e] = r * gR;
        g[7] += q * r * gB[e];
    }
    g[8] = gT;
}

// F - 1 = (E0 - sum w_j S_j) * inv_norm
inline float rg_phot(float X, float Y, float r, float fl, float cth, float sth,
                     float sqf, float ri, float ro, float cr, float Tv,
                     thread const float *wv, float inv_norm) {
    float Bc[HYB_NCOL];
    rg_cols(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, Tv, Bc);
    float num = Bc[0];
    for (int j = 0; j < HYB_NW; ++j) num -= wv[j] * Bc[j + 1];
    return num * inv_norm;
}
inline float rg_phot_d(float X, float Y, float r, float fl, float cth,
                       float sth, float sqf, float ri, float ro, float cr,
                       float Tv, thread const float *wv, float inv_norm,
                       thread float *g, thread float *dFdw) {
    float wct[HYB_NCOL], Bc[HYB_NCOL];
    wct[0] = inv_norm;
    for (int j = 0; j < HYB_NW; ++j) wct[j + 1] = -wv[j] * inv_norm;
    rg_cols_bd(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, Tv, wct, Bc, g);
    float num = Bc[0];
    for (int j = 0; j < HYB_NW; ++j) num -= wv[j] * Bc[j + 1];
    float F = num * inv_norm;
    for (int j = 0; j < HYB_NW; ++j)
        dFdw[j] = (-Bc[j + 1] + F * HYB_NORM[j + 1]) * inv_norm;
    return F;
}
"""


def _rg_header(law) -> str:
    src = _RG_FN
    for k, v in MO._OB_CONST.items():
        src = src.replace(k, v)
    return M._subst(src)


# ---------------------------------------------------------------------------
# tau kernels, derived from the oblate ones by asserted substitutions
# ---------------------------------------------------------------------------

def _rep(s, old, new, count=1):
    assert s.count(old) == count, f"anchor not found once: {old[:60]!r}"
    return s.replace(old, new)


_RG_LOADS = """    float ri  = riin[y];
    float ro  = roin[y];
    float cr  = crin[y];
    float Tv  = Tin[y];
"""

_RG_EDGES = """
    float t1 = tau0 - hw_exp, t2 = tau0 + hw_exp;
    float edge[6]; float dedge[6];
    edge[0] = t1;  dedge[0] = 1.0f;
    edge[5] = t2;  dedge[5] = 1.0f;
    int n_iv = 1;
    {
        int ne = 0;
        for (int c = 0; c < NKINK; ++c) {
            float kt = ks[y * NKINK_U + (uint)c];
            if (kt > t1 && kt < t2 && ne < 4) { edge[1 + ne] = kt; dedge[1 + ne] = 0.0f; ++ne; }
        }
        for (int c = ne; c < 4; ++c) { edge[1 + c] = t2; dedge[1 + c] = 1.0f; }
        n_iv = 1 + ne;
    }
    int n_act = (mode == MODE_NONE || mode == MODE_SUPER) ? n_node : n_iv * ngl;
    int chunk = (n_act + npart - 1) / npart;
    int n_lo = (int)part * chunk, n_hi = metal::min(n_act, n_lo + chunk);
"""


# One exposure's quadrature nodes are shared out over `npart` threads (x =
# part * npts + point; the parts are summed outside). Only the nodes of the
# n_iv non-empty sub-intervals are dealt out -- the contact split leaves the
# rest empty unless four kinks fall in the exposure -- so a typical exposure
# costs each thread n_gl / npart evaluations. A ringed evaluation costs ~10x
# an oblate one (~30x with its gradient), and a thread that ran all
# 5 * n_gl of them held the GPU long enough for macOS to abort the command
# buffer ("Impacting Interactivity") whenever the display needed it. S, the
# total weight, is still summed over every node, so each part's A / S is
# its exact share.
_RG_SPLIT = """    uint xv = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint part = xv / (uint)npts;
    uint x = xv - part * (uint)npts;
    if (part >= (uint)npart) return;
    uint i = y * (uint)npts + x;
    uint iv = (y * (uint)npart + part) * (uint)npts + x;
"""


def _ring_head():
    s = _rep(MO._OB_HEAD, "    uint x = thread_position_in_grid.x;\n"
             "    uint y = thread_position_in_grid.y;\n"
             "    if (x >= (uint)npts) return;\n"
             "    uint i = y * (uint)npts + x;\n", _RG_SPLIT)
    return _rep(s, "    float th  = thin[y];\n",
                "    float th  = thin[y];\n" + _RG_LOADS)


def _ring_body(body):
    s = _rep(body, MO._OB_HEAD, _ring_head())
    s = _rep(s, M._TAU_EDGES, _RG_EDGES)
    return s


_RG_FWD = _ring_body(MO._OB_FWD)
_RG_FWD = _rep(_RG_FWD, "        if (w != 0.0f) { ORB_XY(tt) }\n",
               "        if (node >= n_lo && node < n_hi && w != 0.0f) { ORB_XY(tt) }\n")
_RG_FWD = _rep(_RG_FWD, "out[(uint)NACC * i + (uint)q]", "out[(uint)NACC * iv + (uint)q]")

_RG_VJP = _ring_body(MO._OB_VJP)
_RG_VJP = _rep(_RG_VJP, "        if (w != 0.0f || dw != 0.0f) { ORB_XYD(tt) }\n",
               "        if (node >= n_lo && node < n_hi && (w != 0.0f || dw != 0.0f)) "
               "{ ORB_XYD(tt) }\n")
_RG_VJP = _rep(_RG_VJP, "    gtau[i] = g_tau;\n", "    gtau[iv] = g_tau;\n")
_RG_VJP = _rep(_RG_VJP, "        uint ngrp = ((uint)npts + 31u) / 32u;\n"
               "        uint ob = y * NSLOT_C * ngrp + x / 32u;\n",
               "        uint ngrp = ((uint)npart * (uint)npts + 31u) / 32u;\n"
               "        uint ob = y * NSLOT_C * ngrp + xv / 32u;\n")
_RG_VJP = _rep(_RG_VJP, "    float g[5];\n", "    float g[9];\n")
_RG_VJP = _rep(_RG_VJP, "    float a_per = 0.0f, a_r = 0.0f, a_f = 0.0f, a_ts = 0.0f;\n",
               "    float a_per = 0.0f, a_r = 0.0f, a_f = 0.0f, a_ts = 0.0f;\n"
               "    float a_ri = 0.0f, a_ro = 0.0f, a_cr = 0.0f, a_T = 0.0f;\n")
_RG_VJP = _rep(_RG_VJP, "            a_ts += w * g[4];\n",
               "            a_ts += w * g[4];\n"
               "            a_ri += w * g[5]; a_ro += w * g[6];\n"
               "            a_cr += w * g[7]; a_T  += w * g[8];\n")
_RG_VJP = _rep(_RG_VJP, "    float p_per = 0.0f, p_r = 0.0f, p_f = 0.0f, p_ts = 0.0f;\n",
               "    float p_per = 0.0f, p_r = 0.0f, p_f = 0.0f, p_ts = 0.0f;\n"
               "    float p_ri = 0.0f, p_ro = 0.0f, p_cr = 0.0f, p_T = 0.0f;\n")
_RG_VJP = _rep(_RG_VJP, "        p_ts = ctv * a_ts * inv;\n",
               "        p_ts = ctv * a_ts * inv;\n"
               "        p_ri = ctv * a_ri * inv; p_ro = ctv * a_ro * inv;\n"
               "        p_cr = ctv * a_cr * inv; p_T  = ctv * a_T * inv;\n")
_RG_VJP = _rep(_RG_VJP, "    p_ts = metal::simd_sum(p_ts);\n",
               "    p_ts = metal::simd_sum(p_ts);\n"
               "    p_ri = metal::simd_sum(p_ri); p_ro = metal::simd_sum(p_ro);\n"
               "    p_cr = metal::simd_sum(p_cr); p_T  = metal::simd_sum(p_T);\n")
_RG_VJP = _rep(_RG_VJP, "        gpar[ob + (3u + NTH_U) * ngrp] = p_ts;\n",
               "        gpar[ob + (3u + NTH_U) * ngrp] = p_ts;\n"
               "        gpar[ob + (4u + NTH_U) * ngrp] = p_ri;\n"
               "        gpar[ob + (5u + NTH_U) * ngrp] = p_ro;\n"
               "        gpar[ob + (6u + NTH_U) * ngrp] = p_cr;\n"
               "        gpar[ob + (7u + NTH_U) * ngrp] = p_T;\n")
_RG_VJP = _rep(_RG_VJP, "            gpar[ob + (4u + NTH_U + (uint)q) * ngrp] = p_w[q];\n",
               "            gpar[ob + (8u + NTH_U + (uint)q) * ngrp] = p_w[q];\n")

_ARGS = "X, Y, r, fl, cth, sth, sqf, ri, ro, cr, Tv"
_EVAL_F = f"val[0] = rg_phot({_ARGS}, wv, inv_norm);"
_EVAL_B = f"rg_cols({_ARGS}, val);"
_EVAL_D = f"f = rg_phot_d({_ARGS}, wv, inv_norm, g, dFdw);"
_EVAL_BD = ("{ float Bc[HYB_NCOL];\n"
            f"              rg_cols_bd({_ARGS}, wct, Bc, g);\n"
            "              f = 0.0f;\n"
            "              for (int q = 0; q < HYB_NCOL; ++q) f += wct[q] * Bc[q]; }")


def _nslot(law, orbit, basis):
    """gpar slots: per, theta..., r, f, theta_sky, ring_in, ring_out, c, T,
    then the n_w weights."""
    return 8 + MO._OB_ORBITS[orbit]["nth"] + (0 if basis else get_law(law).n_w)


def _src(body, orbit, law, basis, vjp):
    import re
    plug = MO._OB_ORBITS[orbit]
    nth = plug["nth"]
    nw = 0 if basis else get_law(law).n_w
    s = (body.replace("ORB_LOAD\n", plug["load"])
             .replace("W_LOADS", "" if basis else MO._W_LOADS)
             .replace("CT_LOAD", MO._CT_BASIS if basis else MO._CT_SCALAR))
    s = re.sub(r"ORB_XYD\((\w+)\)", lambda m: plug["xyd"].replace("TT", m.group(1)), s)
    s = re.sub(r"ORB_XY\((\w+)\)", lambda m: plug["xy"].replace("TT", m.group(1)), s)
    if vjp:
        s = s.replace("EVAL_D", _EVAL_BD if basis else _EVAL_D)
        s = s.replace("NWS", str(max(nw, 1)))
        if basis:
            s = _rep(s, "for (int q = 0; q < 1; ++q) a_w[q] += w * dFdw[q];", "")
            s = _rep(s, "    for (int q = 0; q < 1; ++q)\n"
                        "            gpar[ob + (8u + NTH_U + (uint)q) * ngrp] = p_w[q];\n", "")
    else:
        s = s.replace("EVAL", _EVAL_B if basis else _EVAL_F)
        s = s.replace("NACC", "HYB_NCOL" if basis else "1")
    s = (s.replace("TH_LIVE", plug["live"])
          .replace("NSLOT_C", f"{_nslot(law, orbit, basis)}u")
          .replace("NKINK_U", f"{_NKINK}u")
          .replace("NKINK", str(_NKINK))
          .replace("NTH_U", f"{nth}u")
          .replace("NTH", str(nth))
          .replace("NORB_I", str(M.NORB))
          .replace("NORB_C", f"{M.NORB}u"))
    for marker in ("ORB_", "TH_LIVE", "NTH", "NSLOT", "EVAL", "NACC", "NWS",
                   "W_LOADS", "CT_LOAD", "NKINK"):
        assert marker not in s, f"unexpanded {marker} in ring tau kernel"
    return M._tau_src(s)


def _header(law, orbit):
    return MO._header(law, orbit) + _rg_header(law)


def _get_kernels(law, orbit: str, basis: bool):
    law = get_law(law)
    key = ("ring", law.definition, orbit, basis)
    if key not in M._kernels:
        hdr = _header(law, orbit)
        ins = (["taui", "perin", "ain"] + MO._OB_ORBITS[orbit]["inputs"]
               + ["rin", "fin", "thin", "riin", "roin", "crin", "Tin"]
               + ([] if basis else ["win"])
               + ["ks", "xg", "wg", "expt", "mode", "ngl", "nsub", "npart",
                  "npts"])
        tag = f"{_law_tag(law)}_{orbit}{'_b' if basis else ''}"
        M._kernels[key] = (
            mx.fast.metal_kernel(
                name=f"mp_rtau_fwd_{tag}", input_names=ins,
                output_names=["out"], header=hdr,
                source=_src(_RG_FWD, orbit, law, basis, False)),
            mx.fast.metal_kernel(
                name=f"mp_rtau_vjp_{tag}", input_names=ins[:-1] + ["ct", "npts"],
                output_names=["gtau", "gpar"], header=hdr,
                source=_src(_RG_VJP, orbit, law, basis, True)))
    return M._kernels[key]


def _make_core(law, exp_time, mode, n_gl, n_sub, orbit, basis):
    """custom_function over the ring tau kernels. Primals: (tau2d, period,
    a, shape, r, f, theta, ring_in, ring_out, c, T, [w2d,] kinks)."""
    law = get_law(law)
    key = ("ring", law.definition, orbit, bool(basis), float(exp_time),
           int(mode), int(n_gl), int(n_sub))
    if key in M._tau_cores:
        return M._tau_cores[key]
    from .exposure import gauss_legendre
    xg_np, wg_np = gauss_legendre(n_gl)
    nth = MO._OB_ORBITS[orbit]["nth"]
    nslot = _nslot(law, orbit, basis)
    ncol = law.n_col
    # threads per exposure (see _RG_SPLIT). Forward: one node each for the
    # n_gl nodes of an exposure with no kink inside, up to 8. The VJP, ~4x
    # dearer an evaluation, one node each for up to 25.
    npart_f = {M._INT_CONTACT: min(max(5, int(n_gl)), 8),
               M._INT_SUPER: max(1, min(8, int(n_sub)))}.get(int(mode), 1)
    npart_b = {M._INT_CONTACT: min(5 * int(n_gl), 25),
               M._INT_SUPER: max(1, min(25, int(n_sub)))}.get(int(mode), 1)

    def _static(dtype):
        return mx.array(xg_np, dtype=dtype), mx.array(wg_np, dtype=dtype)

    def _parts(v, n, m, npart):
        if npart == 1:
            return v
        return mx.sum(v.reshape(n, npart, m, *v.shape[2:]), axis=1)

    def _fwd(*primals):
        tau2d = primals[0]
        n, m = tau2d.shape
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[0]
        npart = npart_f
        out = k(inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub, npart,
                        int(m)],
                output_shapes=[(n, npart * m, ncol) if basis else (n, npart * m)],
                output_dtypes=[mx.float32],
                grid=(npart * m, n, 1), threadgroup=(256, 1, 1))[0]
        return _parts(out, n, m, npart)

    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        tau2d = primals[0]
        n, m = tau2d.shape
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[1]
        npart = npart_b
        cols = (npart * m + 31) // 32
        gtau, gpar = k(
            inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub, npart, ct,
                    int(m)],
            output_shapes=[(n, npart * m), (n, nslot, cols)],
            output_dtypes=[mx.float32] * 2, init_value=0.0,
            grid=(npart * m, n, 1), threadgroup=(256, 1, 1))
        gtau = _parts(gtau, n, m, npart)
        g = mx.sum(gpar, axis=2)
        g_per, g_a = g[:, 0], g[:, 1]
        if orbit == "circ":
            g_shape = g[:, 2]
        else:
            g_shape = mx.zeros((n, M.NORB), dtype=g.dtype)
            g_shape[:, mx.array(M._ECC_TH_COLS)] = g[:, 2:1 + nth]
        o = 1 + nth
        g_r, g_f, g_th = g[:, o], g[:, o + 1], g[:, o + 2]
        g_ri, g_ro, g_c, g_T = g[:, o + 3], g[:, o + 4], g[:, o + 5], g[:, o + 6]
        ws = () if basis else (g[:, o + 7:o + 7 + law.n_w],)
        return (gtau, g_per, g_a, g_shape, g_r, g_f, g_th, g_ri, g_ro, g_c, g_T,
                *ws, mx.zeros_like(primals[-1]))

    if basis:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, f, theta, ri, ro, c, T, ks):
            return _fwd(tau2d, period, a, shape, r, f, theta, ri, ro, c, T, ks)
    else:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, f, theta, ri, ro, c, T, w2d, ks):
            return _fwd(tau2d, period, a, shape, r, f, theta, ri, ro, c, T, w2d, ks)
    core.vjp(_vjp)
    M._tau_cores[key] = core
    return core


def flux_dev_from_tau_ring_kernel(tau2d, per_c, a_c, b_c, r_c, f_c, th_c, ri_c,
                                  ro_c, rf_c, rt_c, w2d, exp_time, mode, n_gl,
                                  n_sub, law, basis=False, k=None, h=None,
                                  ew=False):
    """The fp32 kernel route of ``rings_tau.flux_dev_from_tau_ring``:
    canonical (n,) parameters in, (n, m) or (n, m, ncol) out. The kinks are
    rings_tau's (detached), as tau offsets, solved only for the contact
    rule (``ring_kinks_kernel``, one thread per chain). T = 1 - exp(-tau / c)
    is formed here by MLX ops (autodiff chains it into ring_tau and ring_f)."""
    from .oblate_tau import _contact_kh, _orbit_consts
    from .rings import ring_opacity
    n = tau2d.shape[0]
    ecc = k is not None
    contact = mode == M._INT_CONTACT and exp_time > 0.0
    geo = (a_c, b_c, r_c, f_c, th_c, ri_c, ro_c, rf_c)
    if ecc:
        from .anchored import pack_orbit_constants
        ci, consts = _orbit_consts(a_c, b_c, k, h, ew)
        shape = pack_orbit_constants(None, None, ci, consts=consts)
        if contact:
            kc, hc = _contact_kh(k, h, ew)
            ks = ring_kinks_kernel(law, *geo, kc, hc, ci, consts=consts)
    else:
        shape = b_c
        if contact:
            ks = ring_kinks_kernel(law, *geo)
    if contact:
        ks = mx.stop_gradient(ks * (per_c / (2.0 * math.pi))[:, None])
    else:
        ks = mx.full((n, _NKINK), math.inf, dtype=tau2d.dtype)
    c = 1.0 - rf_c
    T = ring_opacity(rt_c, rf_c)
    orbit = "circ" if not ecc else ("eccew" if ew else "ecc")
    core = _make_core(law, exp_time, mode, n_gl, n_sub, orbit, basis)
    ws = () if basis else (w2d,)
    return core(tau2d, per_c, a_c, shape, r_c, f_c, th_c, ri_c, ro_c, c, T,
                *ws, ks)


# ---------------------------------------------------------------------------
# kinks: one thread per chain (rings_tau.ring_kinks)
# ---------------------------------------------------------------------------
#
# The graph version is thousands of tiny MLX operations -- seconds a call
# on the GPU -- so the kernel route finds its kinks here, by the same
# scheme: the limb-crossing count of each outline sampled across its
# window, every change bisected. The windows (the circumscribed circles'
# contacts) come in.

_RG_KINK_FN = r"""
inline int rg_count(float phi, ORB_ARGS, float cth, float sth, float A,
                    float B, bool circ) {
    float X, Y, dXp, dYp, dXt[NTH], dYt[NTH];
    bool front;
    ORB_XYD
    if (!front) return 0;
    float x0 = X * cth + Y * sth, y0 = -X * sth + Y * cth;
    float tmp[4];
    return rg_limb(x0, y0, A, B, circ, tmp);
}
"""

_RG_KINK_SRC = """
    uint y = thread_position_in_grid.x;
    if (y >= (uint)nchain) return;
    float av = ain[y];
ORB_LOAD
    float r = rin[y], fl = fin[y], th = thin[y];
    float ri = riin[y], ro = roin[y], cr = crin[y];
    float sth = metal::precise::sin(th), cth = metal::precise::cos(th);
    float sqf = metal::precise::sqrt(1.0f - fl);
    bool pc = fl < OB_FSW, ec = (1.0f - cr) < OB_FSW;
    float A[3] = {pc ? r : r / sqf, ri * r, ro * r};
    float B[3] = {pc ? r : r * sqf, ec ? ri * r : ri * r * cr, ec ? ro * r : ro * r * cr};
    bool circ[3] = {pc, ec, ec};
    float ev[NKINK];
    int nk = 0;
    for (int ol = 0; ol < 3; ++ol) {
        float lo = st[6u * y + 2u * (uint)ol], hi = st[6u * y + 2u * (uint)ol + 1u];
        float pad = 0.02f * (hi - lo) + 1e-6f;
        lo -= pad; hi += pad;
        int prev = rg_count(lo, ORB_PASS, cth, sth, A[ol], B[ol], circ[ol]);
        float pprev = lo;
        int ne = 0;
        for (int s = 1; s < NSAMPLE; ++s) {
            float ph = lo + (hi - lo) * (float)s / (float)(NSAMPLE - 1);
            int c = rg_count(ph, ORB_PASS, cth, sth, A[ol], B[ol], circ[ol]);
            if (c != prev && ne < NEVENT) {
                float a = pprev, b = ph;
                for (int it = 0; it < NBISECT; ++it) {
                    float m = 0.5f * (a + b);
                    if (rg_count(m, ORB_PASS, cth, sth, A[ol], B[ol], circ[ol]) == prev) a = m;
                    else b = m;
                }
                ev[nk++] = 0.5f * (a + b);
                ++ne;
            }
            prev = c; pprev = ph;
        }
    }
    rg_sort(ev, nk);
    for (int q = 0; q < NKINK; ++q)
        out[(uint)NKINK * y + (uint)q] = (q < nk) ? ev[q] : INFINITY;
"""


def _kink_kernel(law, orbit):
    from .rings_tau import _N_BISECT, _N_EVENTS, _N_SAMPLE
    law = get_law(law)
    key = ("ring-kinks", law.definition, orbit)
    if key not in M._kernels:
        plug = MO._OB_CONTACT_ORBITS[orbit]

        def fill(src):
            src = (src.replace("ORB_ARGS", plug["args"])
                      .replace("ORB_PASS", plug["pass_"])
                      .replace("ORB_XYD", plug["xyd"])
                      .replace("ORB_LOAD\n", plug["load"])
                      .replace("NSAMPLE", str(_N_SAMPLE))
                      .replace("NEVENT", str(_N_EVENTS))
                      .replace("NBISECT", str(_N_BISECT[mx.float32]))
                      .replace("NKINK", str(_NKINK))
                      .replace("NTH", str(plug["nth"]))
                      .replace("NORB_I", str(M.NORB))
                      .replace("NORB_C", f"{M.NORB}u"))
            for k, v in MO._OB_CONST.items():
                src = src.replace(k, v)
            for marker in ("ORB_", "NSAMPLE", "NEVENT", "NBISECT", "NKINK", "NTH", "NORB"):
                assert marker not in src, f"unexpanded {marker} in ring kink kernel"
            return src
        hdr = M._HEADER + M._phot_header()
        if orbit == "ecc":
            hdr += M._ecc_header() + M._subst(MO._OB_ECC_FN.replace("SOLVE", M._SOLVE))
        hdr += M._subst(MO._OB_ORBIT_FN) + MO._ob_header(law) + _rg_header(law)
        hdr += M._subst(fill(_RG_KINK_FN))
        M._kernels[key] = mx.fast.metal_kernel(
            name=f"mp_rkinks_{_law_tag(law)}_{orbit}",
            input_names=["ain"] + plug["inputs"] + ["rin", "fin", "thin", "riin",
                                                    "roin", "crin", "st", "nchain"],
            output_names=["out"], header=hdr, source=M._subst(fill(_RG_KINK_SRC)))
    return M._kernels[key]


def ring_kinks_kernel(law, a, b, r, f, theta, ring_in, ring_out, ring_f, k=None,
                      h=None, ci=None, consts=None):
    """``rings_tau._ring_kinks`` in one fp32 kernel launch: (n, 24) kink
    phases, sorted, +inf in unused slots; detached."""
    from .exposure import contact_offsets
    from .rings import ring_axes
    sg = mx.stop_gradient
    a, b, r, f, theta, ring_in, ring_out, ring_f = (
        sg(v) for v in (a, b, r, f, theta, ring_in, ring_out, ring_f))
    n = a.shape[0]
    planet, edges = ring_axes(r, f, ring_in, ring_out, ring_f)
    if k is None:
        orbit, shape = "circ", b
        win = [contact_offsets(A, a, b) for A, _ in [planet] + edges]
    else:
        from .anchored import anchor_constants, pack_orbit_constants
        k, h, ci = (sg(v) for v in (k, h, ci))
        consts = tuple(sg(c) for c in (anchor_constants(k, h) if consts is None else consts))
        orbit, shape = "ecc", pack_orbit_constants(k, h, ci, consts=consts)
        # the three outlines' windows in one launch, stacked (3 n,)
        def t3(v):
            return mx.concatenate([mx.broadcast_to(v, (n,))] * 3)
        cs = M.contact_offsets_anchored_kernel(
            mx.concatenate([mx.broadcast_to(A, (n,)) for A, _ in [planet] + edges]),
            t3(a), t3(b), t3(k), t3(h), t3(ci), consts=tuple(t3(c) for c in consts))
        win = [[cs[j * n:(j + 1) * n, i] for i in range(4)] for j in range(3)]
    st = mx.stack([mx.broadcast_to(c, (n,)) for w in win for c in (w[0], w[3])], axis=1)
    out = _kink_kernel(law, orbit)(
        inputs=[a, sg(shape), r, f, theta, ring_in, ring_out, 1.0 - ring_f, st, n],
        output_shapes=[(n, _NKINK)], output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 64), 1, 1))[0]
    return sg(out)


# ---------------------------------------------------------------------------
# per-point access to the device functions (tests and diagnostics)
# ---------------------------------------------------------------------------

_POINT_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float X = Xi[i], Y = Yi[i], r = rri[i], fl = fi[i], th = ti[i];
    float ri = rii[i], ro = roi[i], cr = cri[i], Tv = Ti[i];
    float cth = metal::precise::cos(th), sth = metal::precise::sin(th);
    float sqf = metal::precise::sqrt(1.0f - fl);
    float wct[HYB_NCOL];
    for (int c = 0; c < HYB_NCOL; ++c) wct[c] = wi[(uint)HYB_NCOL * i + (uint)c];
    float Bc[HYB_NCOL], g[9];
    if (grad == 0) {
        rg_cols(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, Tv, Bc);
        for (int c = 0; c < 9; ++c) g[c] = 0.0f;
    } else {
        rg_cols_bd(X, Y, r, fl, cth, sth, sqf, ri, ro, cr, Tv, wct, Bc, g);
    }
    uint o = (uint)(HYB_NCOL + 9) * i;
    for (int c = 0; c < HYB_NCOL; ++c) out[o + (uint)c] = Bc[c];
    for (int c = 0; c < 9; ++c) out[o + (uint)HYB_NCOL + (uint)c] = g[c];
"""


def _point_cols(X, Y, r, f, theta, ring_in, ring_out, ring_f, T, law,
                wct=None, grad=False):
    """The device functions at single points (no orbit, no exposure): the
    shape columns B (n, ncol) and, with ``grad``, the gradient of wct . B in
    (X, Y, r, f, theta, ring_in, ring_out, c = 1 - ring_f, T) (n, 9). Host
    arrays in, numpy out."""
    law = get_law(law)
    key = ("ring-point", law.definition)
    if key not in M._kernels:
        M._kernels[key] = mx.fast.metal_kernel(
            name=f"mp_rpoint_{_law_tag(law)}",
            input_names=["Xi", "Yi", "rri", "fi", "ti", "rii", "roi", "cri", "Ti",
                         "wi", "grad", "npts"],
            output_names=["out"],
            header=M._HEADER + M._phot_header() + MO._ob_header(law) + _rg_header(law),
            source=_POINT_SRC)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    f32 = lambda v: mx.array(np.broadcast_to(np.asarray(v, np.float32), (n,)).copy())
    nc = law.n_col
    w = np.zeros((n, nc)) if wct is None else np.asarray(wct)
    c = 1.0 - np.asarray(ring_f, dtype=np.float64)
    with mx.stream(mx.gpu):
        out = M._kernels[key](
            inputs=[f32(X), f32(Y), f32(r), f32(f), f32(theta), f32(ring_in),
                    f32(ring_out), f32(c), f32(T),
                    mx.array(w.astype(np.float32).reshape(-1)), int(grad), n],
            output_shapes=[(n * (nc + 9),)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]
        mx.eval(out)
    out = np.asarray(out, dtype=np.float64).reshape(n, nc + 9)
    return out[:, :nc], out[:, nc:]
