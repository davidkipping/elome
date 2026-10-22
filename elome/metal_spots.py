"""Fused fp32 Metal kernels for starspots crossed by a planet (hybrid laws).

The mathematics is ``spots.py``'s (read its docstring first): every
outline -- the planet and each spot's rim -- is walked once by Green's
theorem, split at its limb points and its corners with the other
outlines, each sub-arc weighted by the jump in the blocked fraction
across it, plus the limb's arcs weighted by the fraction just inside.
This module transcribes it per point, in registers, on top of the oblate
and ring kernels' device functions (``metal_oblate``, ``metal_rings``:
the crossing solver, the per-arc integrals, the dual numbers), which it
includes and does not change.

Per point and outline:

* **detached** (plain float): the limb points, the planet-rim corners
  (``rg_limb`` in the planet's unit frame, as ``spots.py``), the rim-rim
  corners (closed form on the sphere), the sorted sub-arcs and, at each
  midpoint, the membership tests that fix the weight; each rim's pole
  roots in closed form, a circle's (the planet) by ``rg_circle_roots``;
* **live** (T = float or the dual number ``dv``): the outline's
  coefficients, each arc's integrals (``rg_arc``) and its gradient-only
  corner term h(s) [C(lo) - C(hi)], with C taken in the sky frame (a
  rim's own frame turns with its spot).

A dual number carries four derivatives: the planet is seeded in (X, Y, rp),
a rim directly in its spot's (n_x, n_y, n_z, alpha), so nothing is chained
by hand. The contrasts enter the weights linearly: their derivatives are
the weight-derivative-weighted arc values.

Two kernels per law and spot count: the columns B = [E0, S_1..S_NW] at
sky positions (N points, each with its own spots), and their VJP for a
cotangent on B. The orbit, the star's rotation and the exposure
quadrature stay MLX graph operations around them (``spots_tau``), whose
autodiff chains these. A third kernel is the contact rule's event search:
per (chain, exposure, spot), the visible planet-rim corner count sampled
across the exposure and each change bisected, as ``spots_tau._spot_events``
(whose live correction, a few graph operations, follows it).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import metal as M
from . import metal_oblate as MO
from .hybrid import get_law
from .metal_hybrid import _law_tag
from .metal_rings import _rg_header

__all__ = ["spot_cols_kernel", "spot_events_kernel"]


# ---------------------------------------------------------------------------
# device code
# ---------------------------------------------------------------------------

_SP_FN = r"""
// ============================== starspots ==============================
constant float SP_TINY = 1e-30f;        // |n_xy|^2 below this: a centred spot
constant float SP_S1C = 1e-8f;          // centred outline (rings._S1_CENTRED)
constant float SP_RR = 1e-12f;          // rim-rim: nearly concentric caps

inline float sp_sin(float a) { return metal::precise::sin(a); }
inline float sp_cos(float a) { return metal::precise::cos(a); }
inline dv sp_sin(dv a) { return dv(metal::precise::sin(a.v), metal::precise::cos(a.v) * a.d); }
inline dv sp_cos(dv a) { return dv(metal::precise::cos(a.v), -metal::precise::sin(a.v) * a.d); }

// a rim, frozen (spots._Rim)
struct sp_rim {
    float nx, ny, nz, sa, ca, nxy, ux, uy, vx, vy, y0, a, b, mu_a, mu_b;
    float psi_limb, half_limb, phi_n;
    bool ok, limb, cen, vis;
};

inline void sp_rim_make(float nx, float ny, float nz, float al, thread sp_rim &R) {
    R.nx = nx; R.ny = ny; R.nz = nz;
    float nxy2 = nx * nx + ny * ny;
    R.ok = nxy2 > SP_TINY;
    R.nxy = R.ok ? metal::precise::sqrt(nxy2) : 0.0f;
    float nxs = R.ok ? R.nxy : 1.0f;
    R.ux = R.ok ? ny / nxs : 1.0f;
    R.uy = R.ok ? -nx / nxs : 0.0f;
    R.vx = R.ok ? nx / nxs : 0.0f;
    R.vy = R.ok ? ny / nxs : 1.0f;
    R.sa = metal::precise::sin(al);
    R.ca = metal::precise::cos(al);
    R.y0 = R.ca * R.nxy;
    R.a = R.sa;
    R.b = R.sa * nz;
    R.mu_a = R.ca * nz;
    R.mu_b = -R.sa * R.nxy;
    float q = (R.sa - fabs(nz)) * (R.sa + fabs(nz));
    R.limb = q > 0.0f;
    float r = R.limb ? metal::precise::sqrt(q) : 0.0f;
    float den = R.limb ? R.sa * R.nxy : 1.0f;
    float mbs = R.limb ? R.mu_b : -1.0f;
    R.psi_limb = metal::precise::atan2(-R.mu_a / mbs, r / den);
    R.half_limb = metal::precise::atan2(r, R.ca);
    R.phi_n = metal::precise::atan2(ny, nx);
    R.cen = (R.sa * R.nxy < SP_S1C) || !R.ok;    // |mu_b|: the closed form divides by it
    R.vis = nz > -R.sa;                  // some of the cap is on the front
}

inline void sp_rim_sky(thread const sp_rim &R, float psi, thread float *x,
                       thread float *y, thread float *mu) {
    float s = metal::precise::sin(psi), c = metal::precise::cos(psi);
    float xr = R.a * c, yr = R.y0 + R.b * s;
    *x = xr * R.ux + yr * R.vx;
    *y = xr * R.uy + yr * R.vy;
    *mu = R.mu_a + R.mu_b * s;
}

inline bool sp_own(thread const sp_rim &R, float x, float y, float z) {
    return x * R.nx + y * R.ny + z * R.nz >= R.ca;
}

// the rim parameter of a 3D rim point (spots._Rim.param)
inline float sp_param(thread const sp_rim &R, float x, float y, float z) {
    float c = x * R.ux + y * R.uy;
    float s = R.nz * (x * R.vx + y * R.vy) - R.nxy * z;
    return metal::precise::atan2(s, c);
}

// planet-rim corners on the rim's parameter: rings._crossings in the
// planet's unit frame, the rim with |b| (a cap facing away mirrored)
inline int sp_pr(thread const sp_rim &R, float X, float Y, float rp,
                 thread float *psi) {
    float cx = R.y0 * R.vx, cy = R.y0 * R.vy;
    float dx = X - cx, dy = Y - cy, lim = 1.001f * (rp + R.a);
    if (dx * dx + dy * dy > lim * lim) return 0;
    float px = X * R.ux + Y * R.uy, py = X * R.vx + Y * R.vy;
    float ab = fabs(R.b);
    bool circ = (R.a - ab) <= OB_FSW * R.a;
    float ph[4];
    int n = rg_limb(-px / rp, (R.y0 - py) / rp, R.a / rp, ab / rp, circ, ph);
    for (int i = 0; i < n; ++i) psi[i] = ob_mod2pi((R.nz < 0.0f) ? -ph[i] : ph[i]);
    return n;
}

// rims i and j meet on the sphere (spots._rim_rim): up to two 3D points
inline int sp_rr(thread const sp_rim &Ri, thread const sp_rim &Rj, thread float3 *P) {
    float3 ni = float3(Ri.nx, Ri.ny, Ri.nz), nj = float3(Rj.nx, Rj.ny, Rj.nz);
    float d = dot(ni, nj);
    float det = (1.0f - d) * (1.0f + d);
    if (!(det > SP_RR)) return 0;
    float a_ = (Ri.ca - d * Rj.ca) / det, b_ = (Rj.ca - d * Ri.ca) / det;
    float3 base = a_ * ni + b_ * nj;
    float t2 = (1.0f - dot(base, base)) / det;
    if (!(t2 > 0.0f)) return 0;
    float t = metal::precise::sqrt(t2);
    float3 m = cross(ni, nj);
    P[0] = base + t * m;
    P[1] = base - t * m;
    return 2;
}

// a rim's pole roots in closed form (spots._rim_roots): per sign the
// larger root of z^2 + 2 i kappa z - 1, and its partner -1/z
inline void sp_rim_roots(thread const sp_rim &R, float eps, thread float2 *z) {
    float se = metal::precise::sqrt(eps);
    for (int s = 0; s < 2; ++s) {
        float sgn = (s == 0) ? 1.0f : -1.0f;
        float kre = R.mu_a / R.mu_b, kim = sgn * se / R.mu_b;
        float2 mik = float2(kim, -kre);
        float2 w = csq(float2(1.0f - (kre * kre - kim * kim), -2.0f * kre * kim));
        float2 z1 = mik + w, z2 = mik - w;
        float2 big = (ca2(z1) >= ca2(z2)) ? z1 : z2;
        z[2 * s] = big;
        z[2 * s + 1] = -cdiv(float2(1.0f, 0.0f), big);
    }
}

// the frozen geometry of one point
struct sp_geo {
    float X, Y, rp;
    float c[SP_K];
    sp_rim R[SP_K];
    int npr[SP_K];
    float pr[SP_K][4];
    bool on;                             // the planet overlaps the disc
    int nl;
    float lp[4];
};

inline void sp_geom(float X, float Y, float rp, thread const float *sp,
                    thread sp_geo &G) {
    G.X = X; G.Y = Y; G.rp = rp;
    G.on = X * X + Y * Y < (1.0f + rp) * (1.0f + rp);
    G.nl = G.on ? rg_limb(X, Y, rp, rp, true, G.lp) : 0;
    for (int k = 0; k < SP_K; ++k) {
        sp_rim_make(sp[5 * k], sp[5 * k + 1], sp[5 * k + 2], sp[5 * k + 3], G.R[k]);
        G.c[k] = sp[5 * k + 4];
        G.npr[k] = (G.on && G.R[k].vis) ? sp_pr(G.R[k], X, Y, rp, G.pr[k]) : 0;
    }
}

// a rim, live: its own-frame ellipse and frame, from (n, alpha) of type T
// smooth through the disc centre (|n_xy| is not): y0 = cos(alpha) n . v
// and the frame's turn -n . u = |n_xy| d(phi), a pure tangent, both on the
// frozen frame, and (a^2 - b^2) / 4 on the sphere, where a ~ |b| near the
// centre cancels. Only derivatives along the sphere are used.
template <typename T>
struct sp_live { T y0, a, b, s2, turn; };

template <typename T>
inline void sp_live_make(T nx, T ny, T nz, T al, thread const sp_rim &R,
                         thread sp_live<T> &L) {
    T sa = sp_sin(al), ca = sp_cos(al);
    L.y0 = ca * (nx * R.vx + ny * R.vy);
    T turn = -(nx * R.ux + ny * R.uy);
    L.turn = turn - t_nograd(turn);
    L.a = sa;
    L.b = sa * nz;
    L.s2 = 0.25f * (sa * sa) * (nx * nx + ny * ny);
}

// an ellipse's arc coefficients (as rg_outline), S2 = (A^2 - B^2) / 4 given
template <typename T>
inline void sp_coef(T x0, T y0, T A, T B, T s2, thread T &S0, thread cx<T> &S1,
                    thread T &S2, thread T &K0, thread cx<T> &K1,
                    thread cx<T> *c0, thread cx<T> *c1, thread cx<T> *c2) {
    S0 = x0 * x0 + y0 * y0 + 0.5f * (A * A + B * B);
    S1 = cx<T>(A * x0, -(B * y0));
    S2 = s2;
    K0 = A * B;
    K1 = cx<T>(0.5f * (B * x0), -0.5f * (A * y0));
    cx<T> sgm[5] = {c_real(S2), c_conj(S1), c_real(S0), S1, c_real(S2)};
    c0[0] = c_real(K0); c0[1] = K1;
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
}

// the arc's gradient-only corner terms h(s) [C(lo) - C(hi)], C = xf yl -
// yf xl in the sky frame (frozen f, live l)
template <typename T>
inline void sp_corner(T xl0, T yl0, float xf0, float yf0, T xl1, T yl1,
                      float xf1, float yf1, thread T *out) {
    T vlo = xf0 * yl0 - yf0 * xl0;
    T vhi = xf1 * yl1 - yf1 * xl1;
    vlo = vlo - t_nograd(vlo);
    vhi = vhi - t_nograd(vhi);
    float slo = xf0 * xf0 + yf0 * yf0, shi = xf1 * xf1 + yf1 * yf1;
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

inline float sp_uval(thread const float *u, thread const float *v) {
    float s = 0.0f;
    for (int c = 0; c < HYB_NGEN; ++c) s += u[c] * v[c];
    return s;
}

// the planet's outline into occ; with ``want``, d(u . occ)/d c_k into gc
template <typename T>
inline void sp_planet(T X, T Y, T rp, thread const sp_geo &G, thread T *occ,
                      bool want, thread const float *u, thread float *gc) {
    if (!G.on) return;
    float bp[4 + 4 * SP_K];
    int n = 0;
    for (int i = 0; i < G.nl; ++i) bp[n++] = G.lp[i];
    for (int k = 0; k < SP_K; ++k)
        for (int i = 0; i < G.npr[k]; ++i) {
            float x, y, mu;
            sp_rim_sky(G.R[k], G.pr[k][i], &x, &y, &mu);
            bp[n++] = ob_mod2pi(metal::precise::atan2(y - G.Y, x - G.X));
        }
    rg_sort(bp, n);
    T S0, S2, K0;
    cx<T> S1, K1, c0[2], c1[4], c2[6];
    sp_coef(X, Y, rp, rp, T(0.0f), S0, S1, S2, K0, K1, c0, c1, c2);
    bool cen = G.X * G.X + G.Y * G.Y < (SP_S1C / G.rp) * (SP_S1C / G.rp);
    float2 z[HYB_K * 4];
    if (!cen)
        for (int q = 0; q < HYB_K; ++q)
            rg_circle_roots(G.X, G.Y, G.rp, 1.0f + HYB_POLE[q], &z[4 * q]);
    int na = (n == 0) ? 1 : n;
    for (int i = 0; i < na; ++i) {
        float lo = (n == 0) ? 0.0f : bp[i];
        float hi = (n == 0) ? MP_TWO_PI : ((i + 1 < n) ? bp[i + 1] : bp[0] + MP_TWO_PI);
        if (!(hi > lo)) continue;
        float mid = 0.5f * (lo + hi);
        float xs = G.X + G.rp * metal::precise::cos(mid);
        float ys = G.Y + G.rp * metal::precise::sin(mid);
        float r2 = xs * xs + ys * ys;
        if (!(r2 < 1.0f)) continue;
        float zs = metal::precise::sqrt(1.0f - r2);
        bool fl[SP_K];
        float kappa = 1.0f;
        for (int k = 0; k < SP_K; ++k) {
            fl[k] = G.R[k].vis && sp_own(G.R[k], xs, ys, zs);
            if (fl[k]) kappa -= 1.0f - G.c[k];
        }
        T val[HYB_NGEN], cor[HYB_NGEN];
        rg_arc(X, Y, rp, S0, S1, S2, K0, K1, c0, c1, c2, z, cen, lo, hi, val);
        float cl = metal::precise::cos(lo), sl = metal::precise::sin(lo);
        float ch = metal::precise::cos(hi), sh = metal::precise::sin(hi);
        sp_corner(X + rp * cl, Y + rp * sl, G.X + G.rp * cl, G.Y + G.rp * sl,
                  X + rp * ch, Y + rp * sh, G.X + G.rp * ch, G.Y + G.rp * sh, cor);
        for (int c = 0; c < HYB_NGEN; ++c) occ[c] = occ[c] + kappa * (val[c] + cor[c]);
        if (want) {
            float vv[HYB_NGEN];
            for (int c = 0; c < HYB_NGEN; ++c) vv[c] = t_val(val[c]);
            float s = sp_uval(u, vv);
            for (int k = 0; k < SP_K; ++k) if (fl[k]) gc[k] += s;
        }
    }
}

// rim k's outline into occ; with ``want``, d(u . occ)/d c_k into gc[k]
template <typename T>
inline void sp_rim_outline(int k, thread const sp_live<T> &L, thread const sp_geo &G,
                           thread T *occ, bool want, thread const float *u,
                           thread float *gc) {
    thread const sp_rim &R = G.R[k];
    if (!R.vis) return;
    float bp[6 + 2 * SP_K];
    int n = 0;
    if (R.limb) {
        bp[n++] = ob_mod2pi(R.psi_limb);
        bp[n++] = ob_mod2pi(MP_PI - R.psi_limb);
    }
    for (int i = 0; i < G.npr[k]; ++i) bp[n++] = G.pr[k][i];
    for (int j = 0; j < SP_K; ++j) {
        if (j == k || !G.R[j].vis) continue;
        float3 P[2];
        int m = sp_rr(R, G.R[j], P);
        for (int q = 0; q < m; ++q) bp[n++] = ob_mod2pi(sp_param(R, P[q].x, P[q].y, P[q].z));
    }
    rg_sort(bp, n);
    T S0, S2, K0;
    cx<T> S1, K1, c0[2], c1[4], c2[6];
    T x0 = T(0.0f);
    sp_coef(x0, L.y0, L.a, L.b, L.s2, S0, S1, S2, K0, K1, c0, c1, c2);
    float2 z[HYB_K * 4];
    if (!R.cen)
        for (int q = 0; q < HYB_K; ++q) sp_rim_roots(R, HYB_POLE[q], &z[4 * q]);
    float kappa = 1.0f - G.c[k];
    int na = (n == 0) ? 1 : n;
    for (int i = 0; i < na; ++i) {
        float lo = (n == 0) ? 0.0f : bp[i];
        float hi = (n == 0) ? MP_TWO_PI : ((i + 1 < n) ? bp[i + 1] : bp[0] + MP_TWO_PI);
        if (!(hi > lo)) continue;
        float xm, ym, mum;
        sp_rim_sky(R, 0.5f * (lo + hi), &xm, &ym, &mum);
        if (!(mum > 0.0f)) continue;
        float dx = xm - G.X, dy = ym - G.Y;
        if (dx * dx + dy * dy < G.rp * G.rp) continue;
        T val[HYB_NGEN], cor[HYB_NGEN];
        rg_arc(x0, L.y0, L.a, S0, S1, S2, K0, K1, c0, c1, c2, z, R.cen, lo, hi, val);
        float cl = metal::precise::cos(lo), sl = metal::precise::sin(lo);
        float ch = metal::precise::cos(hi), sh = metal::precise::sin(hi);
        // C in the rim's own frame, plus the frame's turn: C = Pf x R(dphi)
        // Pl gains s dphi at each end, so the arc gains [g(s_lo) - g(s_hi)]
        // dphi with g = h s: as ((s_lo - s_hi) / |n_xy|) (|n_xy| dphi), both
        // finite at the disc centre
        float fx0 = R.a * cl, fy0 = R.y0 + R.b * sl;
        float fx1 = R.a * ch, fy1 = R.y0 + R.b * sh;
        sp_corner(L.a * cl, L.y0 + L.b * sl, fx0, fy0,
                  L.a * ch, L.y0 + L.b * sh, fx1, fy1, cor);
        {
            // (s_lo - s_hi) / |n_xy|
            float slo = fx0 * fx0 + fy0 * fy0, shi = fx1 * fx1 + fy1 * fy1;
            float ds = (R.sa * R.sa) * R.nxy * (sh - sl) * (sh + sl)
                     + 2.0f * R.ca * R.b * (sl - sh);
            for (int nn = 0; nn < 3; ++nn) {
                float acc = 0.0f;
                for (int j = 0; j <= nn; ++j) {
                    float pw = 0.0f, sp_ = 1.0f;
                    for (int ii = 0; ii <= j; ++ii) { pw = pw * slo + sp_; sp_ *= shi; }
                    acc += RG_AL[nn][j] * pw;
                }
                cor[nn] = cor[nn] + (ds * acc) * L.turn;
            }
            for (int q = 0; q < HYB_K; ++q) {
                float p = 1.0f + HYB_POLE[q];
                cor[3 + q] = cor[3 + q] + (ds / (2.0f * (p - slo) * (p - shi))) * L.turn;
            }
        }
        for (int c = 0; c < HYB_NGEN; ++c) occ[c] = occ[c] + kappa * (val[c] + cor[c]);
        if (want) {
            float vv[HYB_NGEN];
            for (int c = 0; c < HYB_NGEN; ++c) vv[c] = t_val(val[c]);
            gc[k] -= sp_uval(u, vv);
        }
    }
}

// the limb's arcs: w h(1) dtheta, w the blocked fraction just inside
inline void sp_limb(thread const sp_geo &G, thread float *occ, bool want,
                    thread const float *u, thread float *gc) {
    float th[4 + 2 * SP_K];
    int n = 0;
    for (int i = 0; i < G.nl; ++i) {
        float p = G.lp[i];
        th[n++] = ob_mod2pi(metal::precise::atan2(G.Y + G.rp * metal::precise::sin(p),
                                                  G.X + G.rp * metal::precise::cos(p)));
    }
    for (int k = 0; k < SP_K; ++k) {
        if (!G.R[k].limb) continue;
        th[n++] = ob_mod2pi(G.R[k].phi_n + G.R[k].half_limb);
        th[n++] = ob_mod2pi(G.R[k].phi_n - G.R[k].half_limb);
    }
    rg_sort(th, n);
    float h1[HYB_NGEN];
    for (int c = 0; c < 3; ++c) h1[c] = 1.0f / (2.0f * ((float)c + 1.0f));
    for (int k = 0; k < HYB_K; ++k)
        h1[3 + k] = 1.0f / (2.0f * (1.0f + HYB_POLE[k]) * HYB_POLE[k]);
    float uh = want ? sp_uval(u, h1) : 0.0f;
    int na = (n == 0) ? 1 : n;
    for (int i = 0; i < na; ++i) {
        float lo = (n == 0) ? 0.0f : th[i];
        float hi = (n == 0) ? MP_TWO_PI : ((i + 1 < n) ? th[i + 1] : th[0] + MP_TWO_PI);
        if (!(hi > lo)) continue;
        float mid = 0.5f * (lo + hi);
        float cm = metal::precise::cos(mid), sm = metal::precise::sin(mid);
        float dx = cm - G.X, dy = sm - G.Y;
        bool in_p = dx * dx + dy * dy < G.rp * G.rp;
        float w = 1.0f;
        float dth = hi - lo;
        if (!in_p) {
            w = 0.0f;
            for (int k = 0; k < SP_K; ++k) {
                if (!(G.R[k].vis && sp_own(G.R[k], cm, sm, 0.0f))) continue;
                w += 1.0f - G.c[k];
                if (want) gc[k] -= uh * dth;
            }
        }
        if (w == 0.0f) continue;
        for (int c = 0; c < HYB_NGEN; ++c) occ[c] += w * h1[c] * dth;
    }
}

// B = [E0, S_1..S_NW] at sky (X, Y), spots sp[5 k ..] = (n_x, n_y, n_z,
// alpha, contrast)
inline void sp_cols(float X, float Y, float rp, thread const float *sp,
                    thread float *Bc) {
    sp_geo G;
    sp_geom(X, Y, rp, sp, G);
    float occ[HYB_NGEN], u[HYB_NGEN], gc[SP_K];
    for (int c = 0; c < HYB_NGEN; ++c) { occ[c] = 0.0f; u[c] = 0.0f; }
    sp_planet<float>(X, Y, rp, G, occ, false, u, gc);
    for (int k = 0; k < SP_K; ++k) {
        if (!G.R[k].vis) continue;
        sp_live<float> L;
        sp_live_make<float>(sp[5 * k], sp[5 * k + 1], sp[5 * k + 2], sp[5 * k + 3],
                            G.R[k], L);
        sp_rim_outline<float>(k, L, G, occ, false, u, gc);
    }
    sp_limb(G, occ, false, u, gc);
    ob_shape(occ, Bc);
}

// B and the gradient g of wct . B in (X, Y, rp, then per spot n_x, n_y,
// n_z, alpha, contrast)
inline void sp_cols_bd(float X, float Y, float rp, thread const float *sp,
                       thread const float *wct, thread float *Bc,
                       thread float *g) {
    sp_geo G;
    sp_geom(X, Y, rp, sp, G);
    // wct . B = sum_c u[c] occ[c], B = -[occ_0, S occ]
    float u[HYB_NGEN];
    for (int c = 0; c < HYB_NGEN; ++c) {
        float acc = (c == 0) ? wct[0] : 0.0f;
        for (int j = 0; j < HYB_NW; ++j) acc += wct[j + 1] * HYB_SMAT[j * HYB_NGEN + c];
        u[c] = -acc;
    }
    float gen[HYB_NGEN], gc[SP_K];
    for (int c = 0; c < HYB_NGEN; ++c) gen[c] = 0.0f;
    for (int k = 0; k < SP_K; ++k) gc[k] = 0.0f;
    for (int c = 0; c < 3 + 5 * SP_K; ++c) g[c] = 0.0f;
    {
        dv occ[HYB_NGEN];
        for (int c = 0; c < HYB_NGEN; ++c) occ[c] = dv(0.0f);
        sp_planet<dv>(dv(X, float4(1.0f, 0.0f, 0.0f, 0.0f)),
                      dv(Y, float4(0.0f, 1.0f, 0.0f, 0.0f)),
                      dv(rp, float4(0.0f, 0.0f, 1.0f, 0.0f)), G, occ, true, u, gc);
        float4 t = float4(0.0f);
        for (int c = 0; c < HYB_NGEN; ++c) { gen[c] += occ[c].v; t += u[c] * occ[c].d; }
        g[0] = t.x; g[1] = t.y; g[2] = t.z;
    }
    for (int k = 0; k < SP_K; ++k) {
        if (!G.R[k].vis) continue;
        dv occ[HYB_NGEN];
        for (int c = 0; c < HYB_NGEN; ++c) occ[c] = dv(0.0f);
        sp_live<dv> L;
        sp_live_make<dv>(dv(sp[5 * k], float4(1.0f, 0.0f, 0.0f, 0.0f)),
                         dv(sp[5 * k + 1], float4(0.0f, 1.0f, 0.0f, 0.0f)),
                         dv(sp[5 * k + 2], float4(0.0f, 0.0f, 1.0f, 0.0f)),
                         dv(sp[5 * k + 3], float4(0.0f, 0.0f, 0.0f, 1.0f)),
                         G.R[k], L);
        sp_rim_outline<dv>(k, L, G, occ, true, u, gc);
        float4 t = float4(0.0f);
        for (int c = 0; c < HYB_NGEN; ++c) { gen[c] += occ[c].v; t += u[c] * occ[c].d; }
        g[3 + 5 * k] = t.x; g[4 + 5 * k] = t.y; g[5 + 5 * k] = t.z; g[6 + 5 * k] = t.w;
    }
    sp_limb(G, gen, true, u, gc);
    for (int k = 0; k < SP_K; ++k) g[7 + 5 * k] = gc[k];
    ob_shape(gen, Bc);
}
"""


def _sp_header(law, K) -> str:
    src = _SP_FN.replace("SP_K", str(int(K)))
    for k, v in MO._OB_CONST.items():
        src = src.replace(k, v)
    return M._subst(src)


def _point_header(law, K) -> str:
    return (M._HEADER + M._phot_header() + MO._ob_header(law) + _rg_header(law)
            + _sp_header(law, K))


# ---------------------------------------------------------------------------
# the columns at sky positions, and their VJP
# ---------------------------------------------------------------------------

_COLS_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float Bc[HYB_NCOL];
    if (act[i] == 0.0f) {
        for (int c = 0; c < HYB_NCOL; ++c) out[(uint)HYB_NCOL * i + (uint)c] = 0.0f;
        return;
    }
    float sp[5 * NSPOT];
    for (int c = 0; c < 5 * NSPOT; ++c) sp[c] = Si[(uint)(5 * NSPOT) * i + (uint)c];
    sp_cols(Xi[i], Yi[i], ri[i], sp, Bc);
    for (int c = 0; c < HYB_NCOL; ++c) out[(uint)HYB_NCOL * i + (uint)c] = Bc[c];
"""

_VJP_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    const int NG = 3 + 5 * NSPOT;
    if (act[i] == 0.0f) {
        for (int c = 0; c < NG; ++c) g[(uint)NG * i + (uint)c] = 0.0f;
        return;
    }
    float sp[5 * NSPOT], wct[HYB_NCOL], Bc[HYB_NCOL], gv[NG];
    for (int c = 0; c < 5 * NSPOT; ++c) sp[c] = Si[(uint)(5 * NSPOT) * i + (uint)c];
    for (int c = 0; c < HYB_NCOL; ++c) wct[c] = ct[(uint)HYB_NCOL * i + (uint)c];
    sp_cols_bd(Xi[i], Yi[i], ri[i], sp, wct, Bc, gv);
    for (int c = 0; c < NG; ++c) g[(uint)NG * i + (uint)c] = gv[c];
"""


def _cols_kernels(law, K):
    law = get_law(law)
    key = ("spot-cols", law.definition, int(K))
    if key not in M._kernels:
        hdr = _point_header(law, K)
        tag = f"{_law_tag(law)}_k{int(K)}"
        ins = ["Xi", "Yi", "ri", "Si", "act"]
        M._kernels[key] = (
            mx.fast.metal_kernel(
                name=f"mp_spcols_{tag}", input_names=ins + ["npts"],
                output_names=["out"], header=hdr,
                source=_COLS_SRC.replace("NSPOT", str(int(K)))),
            mx.fast.metal_kernel(
                name=f"mp_spvjp_{tag}", input_names=ins + ["ct", "npts"],
                output_names=["g"], header=hdr,
                source=_VJP_SRC.replace("NSPOT", str(int(K)))))
    return M._kernels[key]


_CORES: dict = {}


def _cols_core(law, K):
    """custom_function (X, Y, rp, S, act) -> B over N flat points: S (N, K,
    5) of (n_x, n_y, n_z, alpha, contrast); act (N,) zero where a point is
    not needed (its columns and gradient are 0)."""
    law = get_law(law)
    key = (law.definition, int(K))
    if key in _CORES:
        return _CORES[key]
    ncol, ng = law.n_col, 3 + 5 * int(K)

    @mx.custom_function
    def core(X, Y, rp, S, act):
        n = X.shape[0]
        return _cols_kernels(law, K)[0](
            inputs=[X, Y, rp, S, act, n], output_shapes=[(n, ncol)],
            output_dtypes=[mx.float32], grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]

    def _vjp(primals, cotangent, output):
        X, Y, rp, S, act = primals
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = X.shape[0]
        g = _cols_kernels(law, K)[1](
            inputs=[X, Y, rp, S, act, ct, n], output_shapes=[(n, ng)],
            output_dtypes=[mx.float32], grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]
        return (g[:, 0], g[:, 1], g[:, 2], mx.reshape(g[:, 3:], (n, int(K), 5)),
                mx.zeros_like(act))
    core.vjp(_vjp)
    _CORES[key] = core
    return core


def spot_cols_kernel(X, Y, rp, n, alpha, contrast, law, active=None):
    """``spots.shape_cols_spots`` by the fp32 kernel: X, Y, rp of a common
    point shape (broadcast), n (..., K, 3), alpha and contrast (..., K);
    (..., ncol) out, differentiable in every input. ``active`` (point
    shape, optional): where False the columns are 0 and not computed."""
    law = get_law(law)
    f32 = lambda v: (v if isinstance(v, mx.array) else mx.array(v)).astype(mx.float32)
    X, Y, rp, n, alpha, contrast = (f32(v) for v in (X, Y, rp, n, alpha, contrast))
    K = n.shape[-2]
    point = tuple(np.broadcast_shapes(X.shape, Y.shape, rp.shape, n.shape[:-2],
                                      alpha.shape[:-1], contrast.shape[:-1]))
    S = mx.concatenate([mx.broadcast_to(n, point + (K, 3)),
                        mx.broadcast_to(alpha, point + (K,))[..., None],
                        mx.broadcast_to(contrast, point + (K,))[..., None]], axis=-1)
    N = int(np.prod(point)) if point else 1
    flat = lambda v: mx.reshape(mx.broadcast_to(v, point), (N,))
    act = (mx.ones((N,), dtype=mx.float32) if active is None
           else flat(mx.stop_gradient(active).astype(mx.float32)))
    B = _cols_core(law, K)(flat(X), flat(Y), flat(rp), mx.reshape(S, (N, K, 5)), act)
    return mx.reshape(B, point + (law.n_col,))


# ---------------------------------------------------------------------------
# the contact rule's event search: one thread per (chain, exposure, spot)
# ---------------------------------------------------------------------------

_EV_FN = r"""
// the planet and one spot at planet time t (since mid-transit) and star
// time t + ds (since the spots' epoch) (spots_tau's frames)
inline void sp_state(float t, float ds, float per, float av, ORB_ARGS,
                     float prot, float istar, float lam, float lon, float lat,
                     float al, thread float *Xo, thread float *Yo, thread bool *fo,
                     thread sp_rim &R) {
    float tau = t;
    float X, Y;
    bool front;
    ORB_XY
    *Xo = X; *Yo = Y; *fo = front;
    float si = metal::precise::sin(istar), ci = metal::precise::cos(istar);
    float sl = metal::precise::sin(lam), cl = metal::precise::cos(lam);
    float ph = lon + MP_TWO_PI * (t + ds) / prot;
    float sp = metal::precise::sin(ph), cp = metal::precise::cos(ph);
    float sb = metal::precise::sin(lat), cb = metal::precise::cos(lat);
    float u = cb * cp, v = cb * sp;
    float nx = u * (-(ci * sl)) + v * cl + sb * (si * sl);
    float ny = u * (-(ci * cl)) - v * sl + sb * (si * cl);
    float nz = u * si + sb * ci;
    sp_rim_make(nx, ny, nz, al, R);
}

// planet-rim corners on the spot's visible rim (spots_tau._corner_count)
inline int sp_count_at(float t, float ds, float per, float av, ORB_ARGS,
                       float prot, float istar, float lam, float lon, float lat,
                       float al, float r, thread float *psi, thread bool *vis) {
    float X, Y;
    bool front;
    sp_rim R;
    sp_state(t, ds, per, av, ORB_PASS, prot, istar, lam, lon, lat, al, &X, &Y,
             &front, R);
    for (int i = 0; i < 4; ++i) vis[i] = false;
    if (!front) return 0;
    int n = sp_pr(R, X, Y, r, psi);
    int c = 0;
    for (int i = 0; i < n; ++i) {
        float mu = R.mu_a + R.mu_b * metal::precise::sin(psi[i]);
        vis[i] = mu >= 0.0f;
        c += vis[i] ? 1 : 0;
    }
    return c;
}
"""

_EV_SRC = """
    uint xk = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint e = xk / (uint)NSPOT, k = xk - e * (uint)NSPOT;
    if (e >= (uint)m) return;
    uint i = y * (uint)m + e;
    float tc = tci[i], ds = dsi[i];
    float per = perin[y], av = ain[y];
ORB_LOAD
    float r = rin[y], prot = protin[y], istar = isin[y], lam = lamin[y];
    float lon = spin[(y * (uint)NSPOT + k) * 4u], lat = spin[(y * (uint)NSPOT + k) * 4u + 1u];
    float al = spin[(y * (uint)NSPOT + k) * 4u + 2u];
    float t1 = tc - hexp, t2 = tc + hexp;
    int S = nsample;
    float ps[4]; bool vs[4];
    float lo[NEV], hi[NEV]; int clo[NEV];
#define SP_COUNT(TT, PS, VS) sp_count_at(TT, ds, per, av, ORB_PASS, prot, istar, lam, lon, lat, al, r, PS, VS)
    int ne = 0;
    float tp = t1;
    int cp = SP_COUNT(t1, ps, vs);
    for (int s = 1; s <= S; ++s) {
        float ts = t1 + (t2 - t1) * ((float)s / (float)S);
        int c = SP_COUNT(ts, ps, vs);
        if (c != cp && ne < NEV) { lo[ne] = tp; hi[ne] = ts; clo[ne] = cp; ++ne; }
        cp = c; tp = ts;
    }
    uint oo = (i * (uint)NSPOT + k) * (uint)NEV;
    for (int q = 0; q < NEV; ++q) {
        if (q >= ne) { te[oo + q] = INFINITY; code[oo + q] = 0.0f; psit[oo + q] = 0.0f; continue; }
        float a = lo[q], b = hi[q];
        for (int it = 0; it < NBISECT; ++it) {
            float mid = 0.5f * (a + b);
            int c = SP_COUNT(mid, ps, vs);
            if (c == clo[q]) a = mid; else b = mid;
        }
        float tev = 0.5f * (a + b);
        int chi = SP_COUNT(b, ps, vs);
        bool tang = abs(chi - clo[q]) == 2;
        float more = (chi > clo[q]) ? b : a;
        float pm[4]; bool vm[4];
        SP_COUNT(more, pm, vm);
        float best = INFINITY, psi_t = 0.0f;
        for (int u1 = 0; u1 < 4; ++u1)
            for (int u2 = u1 + 1; u2 < 4; ++u2) {
                if (!(vm[u1] && vm[u2])) continue;
                float d = ob_mod2pi(pm[u2] - pm[u1] + MP_PI) - MP_PI;
                if (fabs(d) < best) { best = fabs(d); psi_t = pm[u1] + 0.5f * d; }
            }
        // the limb point the corner passed: |q| smaller at te
        float X, Y; bool front; sp_rim R;
        sp_state(tev, ds, per, av, ORB_PASS, prot, istar, lam, lon, lat, al, &X, &Y, &front, R);
        float x1, y1, x2, y2, mu_;
        sp_rim_sky(R, R.psi_limb, &x1, &y1, &mu_);
        sp_rim_sky(R, MP_PI - R.psi_limb, &x2, &y2, &mu_);
        float q1 = (x1 - X) * (x1 - X) + (y1 - Y) * (y1 - Y) - r * r;
        float q2 = (x2 - X) * (x2 - X) + (y2 - Y) * (y2 - Y) - r * r;
        te[oo + q] = tev;
        code[oo + q] = tang ? 1.0f : ((fabs(q2) < fabs(q1)) ? 3.0f : 2.0f);
        psit[oo + q] = psi_t;
    }
#undef SP_COUNT
"""


def _events_kernel(law, K, orbit):
    from .spots_tau import _E_SPOT, _N_BISECT
    law = get_law(law)
    orbit = "circ" if orbit == "circ" else "ecc"
    key = ("spot-events", law.definition, int(K), orbit)
    if key not in M._kernels:
        plug = MO._OB_ORBITS[orbit]
        args = "float bv" if orbit == "circ" else "thread const float *o"
        pass_ = "bv" if orbit == "circ" else "o"

        def fill(src):
            src = (src.replace("ORB_ARGS", args).replace("ORB_PASS", pass_)
                      .replace("ORB_XY", plug["xy"].replace("TT", "tau"))
                      .replace("ORB_LOAD\n", plug["load"])
                      .replace("NSPOT", str(int(K))).replace("NEV", str(_E_SPOT))
                      .replace("NBISECT", str(_N_BISECT[mx.float32]))
                      .replace("NORB_I", str(M.NORB)).replace("NORB_C", f"{M.NORB}u"))
            for marker in ("ORB_", "NSPOT", "NEV", "NBISECT", "NORB"):
                assert marker not in src, f"unexpanded {marker} in spot event kernel"
            return src
        hdr = MO._header(law, orbit) + _rg_header(law) + _sp_header(law, K) \
            + M._subst(fill(_EV_FN))
        M._kernels[key] = mx.fast.metal_kernel(
            name=f"mp_spev_{_law_tag(law)}_k{int(K)}_{orbit}",
            input_names=["tci", "dsi", "perin", "ain"] + plug["inputs"]
            + ["rin", "protin", "isin", "lamin", "spin", "hexp", "nsample", "m"],
            output_names=["te", "code", "psit"], header=hdr,
            source=M._subst(fill(_EV_SRC)))
    return M._kernels[key]


def spot_events_kernel(tt, ds, half, period, a, shape, r, prot, istar, lam, spots4, law,
                       n_sample, orbit="circ"):
    """The event search of ``spots_tau._spot_events`` in one fp32 launch,
    detached: for exposures centred on planet times tt (n, m, since each
    point's mid-transit) of half-length ``half``, the star's clock ahead of
    the planet's by ds (n, m),
    per spot the first _E_SPOT times at which the visible planet-rim
    corner count changes, (n, m, K, E) each of: the time (+inf if none),
    its kind (0 none, 1 a tangency, 2 / 3 a corner passing the rim's first
    / second limb point) and, for a tangency, the frozen touching psi.
    ``shape`` is b (circular) or the packed orbit constants."""
    from .spots_tau import _E_SPOT
    sg = mx.stop_gradient
    n, m = tt.shape
    K = spots4.shape[-2]
    f32 = lambda v: sg(v).astype(mx.float32)
    ins = [f32(tt), f32(ds), *(f32(v) for v in (period, a, shape, r, prot, istar, lam)),
           f32(spots4), float(half), int(n_sample), int(m)]
    shp = (n, m, K, _E_SPOT)
    te, code, psit = _events_kernel(law, K, orbit)(
        inputs=ins, output_shapes=[shp] * 3, output_dtypes=[mx.float32] * 3,
        grid=(m * K, n, 1), threadgroup=(64, 1, 1))
    return sg(te), sg(code), sg(psit)


# ---------------------------------------------------------------------------
# per-point access to the device functions (tests and diagnostics)
# ---------------------------------------------------------------------------

def _point_cols(X, Y, rp, n, alpha, contrast, law, wct=None, grad=False):
    """The device functions at single points: B (N, ncol) and, with
    ``grad``, the gradient of wct . B in (X, Y, rp, then per spot n_x, n_y,
    n_z, alpha, contrast) (N, 3 + 5 K). Host arrays in (n (N, K, 3)),
    numpy out."""
    law = get_law(law)
    X = np.asarray(X, dtype=np.float32).reshape(-1)
    N = X.shape[0]
    n = np.broadcast_to(np.asarray(n, dtype=np.float32), (N,) + np.shape(n)[-2:])
    K = n.shape[-2]
    al = np.broadcast_to(np.asarray(alpha, dtype=np.float32), (N, K))
    c = np.broadcast_to(np.asarray(contrast, dtype=np.float32), (N, K))
    S = np.concatenate([n, al[..., None], c[..., None]], axis=-1).astype(np.float32)
    f32 = lambda v: mx.array(np.broadcast_to(np.asarray(v, np.float32), (N,)).copy())
    act = mx.ones((N,), dtype=mx.float32)
    k_cols, k_vjp = _cols_kernels(law, K)
    with mx.stream(mx.gpu):
        ins = [f32(X), f32(Y), f32(rp), mx.array(S), act]
        B = k_cols(inputs=ins + [N], output_shapes=[(N, law.n_col)],
                   output_dtypes=[mx.float32], grid=(N, 1, 1), threadgroup=(256, 1, 1))[0]
        g = None
        if grad:
            w = np.zeros((N, law.n_col)) if wct is None else np.broadcast_to(wct, (N, law.n_col))
            g = k_vjp(inputs=ins + [mx.array(np.asarray(w, np.float32)), N],
                      output_shapes=[(N, 3 + 5 * K)], output_dtypes=[mx.float32],
                      grid=(N, 1, 1), threadgroup=(256, 1, 1))[0]
        mx.eval(B, *([] if g is None else [g]))
    B = np.asarray(B, dtype=np.float64)
    return (B, np.asarray(g, dtype=np.float64)) if grad else B
