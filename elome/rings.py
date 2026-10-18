"""Ringed planets for the hybrid limb-darkening laws -- the reference MLX
graph, fp64 and fp32.

A ring is a flat circular annulus in the planet's equatorial plane. On the
sky its edges are concentric ellipses, coaxial with an oblate planet's
outline (the same projected angle theta), with semi-major axes equal to
the ring's true radii and a common axis ratio 1 - ring_f = |cos i_ring|.
Parameters, as for an oblate planet, are projected quantities:

* ``rp``, ``f``: the planet's area-equivalent radius (stellar radii) and
  projected flattening (0: a spherical planet);
* ``ring_in``, ``ring_out``: the ring's inner and outer radii, in units of
  ``rp``; ``ring_in`` must reach past the planet's equator,
  ring_in >= 1/sqrt(1 - f);
* ``ring_f``: the ring's projected flattening, 0 face-on to 1 edge-on. A
  uniform prior on it is the isotropic prior on the ring's orientation.
  An equatorial ring has f <= ring_f (documented, not enforced);
* ``ring_tau``: the ring's normal optical depth (inf: opaque); the blocked
  fraction along the line of sight is T = 1 - exp(-tau / (1 - ring_f)).

The occultor blocks a piecewise-constant fraction w of the star: 1 inside
the planet, T in the ring, 0 elsewhere. By Green's theorem with the
potential h(s) = (1/2s) int_0^s I (``oblate.py``),

    int_star w I dA = sum over each outline's sub-arcs inside the star of
                      (w just inside - w just outside) * int h(s) (x dy - y dx)
                    + sum over the star's limb arcs of w * h(1) * dtheta,

so every outline -- the planet and each ring edge -- is walked once, its
arcs weighted by the jump in w across them. The arcs are split at the
outline's limb crossings (up to four) and at its corners with the other
outlines (closed form: the outlines are concentric and coaxial, so they
meet where cos^2 phi takes one value). Which side of another outline an
arc lies on is decided from that closed form, never from a point test
against the other outline, which is a coin toss at a tangency.

Each arc is one call of ``oblate.py``'s per-arc primitives: the even
moments' Fourier sums and the poles' partial fractions over the roots of
the self-inversive quartic. An ellipse's roots come from the oblate
solver; a circle's (a spherical planet, a face-on ring) in closed form --
D(z) = z q(z) with q quadratic, its roots by Vieta, plus 0 and the root at
infinity, whose residue vanishes. A circle centred on the star (R times
its offset below 1e-28 in fp64, 1e-8 in fp32) takes its first-order
expansion in the offset instead: the roots there run to 0 and infinity.

Gradients as in ``oblate.py``: roots and arc endpoints are frozen; the
endpoints' motion is restored by a gradient-only corner term
kappa h(s) [C(P_lo) - C(P_hi)] per arc and column, h at the corner (s = 1
on the limb, as ``oblate._corner``; s < 1 at a planet-ring corner).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .dtypes import fp64_on_cpu
from .hybrid import _combine, combine_cols, get_law
from .oblate import (_C, _F_SW, _N_MAX, _alpha, _coeffs, _cwhere, _even_arc,
                     _limb_candidates, _mod2pi, _pole_arc, _q_coeffs, _roots,
                     _to_dtype, axes)
from .trig import sincos

__all__ = ["ring_axes", "ring_opacity", "generator_cols_ring", "shape_cols_ring",
           "flux_dev_ring"]

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_sg = mx.stop_gradient

# a circle's far root is ~1/(R d) and D's derivatives carry its fourth
# power: below |S1| = R d of _S1_CENTRED the circle counts as centred (its
# first-order expansion, error ~d^2), and the root at infinity stands at
# _BIG -- its neglected share is ~1/_BIG, below rounding, while _BIG^4
# stays finite in the precision
_S1_CENTRED = {mx.float64: 1e-28, mx.float32: 1e-8}
_BIG = {mx.float64: 1e30, mx.float32: 1e8}
_TINY = {mx.float64: 1e-300, mx.float32: 1e-30}
# stand-in for the ellipse crossing solve on lanes where the outline is a
# circle (the QR seeds divide by S2): a small ellipse straddling the limb
_SAFE_ELL = (1.0, 0.02, 0.1, 0.08)
# stand-in for the ellipse root solve on circle lanes
_SAFE_IN = (0.05, 0.03, 0.1, 0.08)


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------

def ring_axes(rp, f, ring_in, ring_out, ring_f):
    """((A_p, B_p), [(A_in, B_in), (A_out, B_out)]): the planet's and the
    ring edges' semi-axes in stellar radii, long axes along the principal
    frame's x."""
    Ap, Bp = axes(rp, f)
    c = 1.0 - ring_f
    Ri, Ro = ring_in * rp, ring_out * rp
    return (Ap, Bp), [(Ri, Ri * c), (Ro, Ro * c)]


def ring_opacity(ring_tau, ring_f):
    """The blocked fraction along the line of sight, 1 - exp(-tau / |cos i|),
    with |cos i| = 1 - ring_f (expm1: MLX's float64 exp is float32-
    accurate). tau = inf is an opaque ring."""
    if isinstance(ring_tau, mx.array) or isinstance(ring_f, mx.array):
        # an opaque ring (tau = inf) is exactly 1, and sanitized: d/d ring_f
        # of exp(-inf / c) would be 0 * inf
        tau = ring_tau if isinstance(ring_tau, mx.array) else mx.array(ring_tau)
        opaque = mx.isinf(tau)
        t = mx.where(opaque, 0.0, tau)
        return mx.where(opaque, 1.0, -mx.expm1(-t / (1.0 - ring_f)))
    return -np.expm1(-np.asarray(ring_tau, dtype=np.float64) / (1.0 - np.asarray(ring_f, dtype=np.float64)))


def _check_ring(f, ring_in, ring_out, ring_f, ring_tau):
    """Host values only (an mx.array is never read back: see
    oblate._check_domain)."""
    vals = (f, ring_in, ring_out, ring_f, ring_tau)
    if any(isinstance(v, mx.array) for v in vals):
        return
    f, ri, ro, rf, tau = (np.asarray(v, dtype=np.float64) for v in vals)
    if np.any(rf < 0.0) or np.any(rf >= 1.0):
        raise ValueError(f"ring_f must be in [0, 1); got {rf}")
    if np.any(tau < 0.0):
        raise ValueError(f"ring_tau must be >= 0; got {tau}")
    if np.any(ro <= ri):
        raise ValueError("ring_out must exceed ring_in")
    if np.any(ri * np.sqrt(1.0 - f) < 1.0 - 1e-12):
        raise ValueError("ring_in (in units of rp) must reach past the planet's "
                         "equator: ring_in >= 1 / sqrt(1 - f)")


# --------------------------------------------------------------------------
# one outline: crossings, roots
# --------------------------------------------------------------------------

def _crossings(gx0, gy0, ga, gb, circ, dt):
    """Up to four limb crossings of an outline (frozen inputs): candidate
    angles on its own parameter and validity masks. Ellipses by the oblate
    solver; circles in closed form, cos(phi - g) = (1 - d^2 - R^2)/(2 d R)."""
    sx, sy, sa, sb = _SAFE_ELL
    ex0, ey0 = mx.where(circ, sx, gx0), mx.where(circ, sy, gy0)
    ea, eb = mx.where(circ, sa, ga), mx.where(circ, sb, gb)
    phis, valid = _limb_candidates(ex0, ey0, ea, eb, dt)
    d2 = gx0 * gx0 + gy0 * gy0
    d = mx.sqrt(mx.maximum(d2, _TINY[dt]))
    cq = ((1.0 - ga) * (1.0 + ga) - d2) / (2.0 * d * ga)
    has = circ & (mx.abs(cq) < 1.0) & (d2 > 0.0)
    g = mx.arctan2(gy0, gx0)
    half = mx.arctan2(mx.sqrt(mx.maximum(1.0 - cq * cq, 0.0)), mx.where(has, cq, 0.0))
    pc = [_mod2pi(g + half), _mod2pi(g - half)]
    out_p = [mx.where(circ, pc[i], phis[i]) if i < 2 else phis[i] for i in range(4)]
    out_v = [mx.where(circ, has, valid[i]) if i < 2 else valid[i] & ~circ for i in range(4)]
    return out_p, out_v


def _circle_roots(gx0, gy0, R, p, dt):
    """The four roots of D for a circle (S2 = 0): D = z q(z),
    q = S1 z^2 + (S0 - p) z + conj(S1). 0, a large stand-in for the root at
    infinity (zero residue), and q's two roots by Vieta from the
    cancellation-free quadratic formula -- not a root and its reciprocal
    partner: once the circle crosses the level s = p both lie on |z| = 1,
    each its own partner. Frozen inputs; lanes with S1 = 0 (centred) are
    the caller's to replace."""
    s1 = _C(R * gx0, -(R * gy0))
    b_ = gx0 * gx0 + gy0 * gy0 + R * R - p
    disc = b_ * b_ - 4.0 * (s1.re * s1.re + s1.im * s1.im)
    sq = _C(mx.sqrt(mx.maximum(disc, 0.0)), mx.sqrt(mx.maximum(-disc, 0.0)))
    sgn = mx.where(b_ >= 0.0, 1.0, -1.0)
    qq = _C(-0.5 * (b_ + sgn * sq.re), -0.5 * sgn * sq.im)
    ok = s1.abs2() > 0.0
    s1s = _cwhere(ok, s1, 1.0)
    qqs = _cwhere(qq.abs2() > 0.0, qq, 1.0)
    z1 = _cwhere(ok, qq / s1s, _BIG[dt])
    z2 = _cwhere(ok, s1.conj() / qqs, 0.0)
    zero = _C(0.0 * gx0, 0.0 * gx0)
    return [zero, z1, z2, _C(_BIG[dt] + 0.0 * gx0, 0.0 * gx0)]


def _outline_roots(gx0, gy0, ga, gb, circ, p, dt):
    """Frozen roots of D at level p for each lane: the oblate solver on
    ellipses, the closed form on circles (with a safe offset where the
    circle is centred, whose lanes the caller replaces)."""
    sx, sy, sa, sb = _SAFE_IN
    ex0, ey0 = mx.where(circ, sx, gx0), mx.where(circ, sy, gy0)
    ea, eb = mx.where(circ, sa, ga), mx.where(circ, sb, gb)
    S0, S1, S2, _, _ = _coeffs(ex0, ey0, ea, eb)
    ze = [z.sg() for z in _roots(S0, S1, S2, p)]
    cen = _centred(gx0, gy0, ga, dt)
    cx0 = mx.where(cen, 0.5 * ga, gx0)
    zc = _circle_roots(cx0, mx.where(cen, 0.0, gy0), ga, p, dt)
    return [_cwhere(circ, c, e).sg() for c, e in zip(zc, ze)]


def _centred(gx0, gy0, R, dt):
    lim = _S1_CENTRED[dt] / R
    return gx0 * gx0 + gy0 * gy0 < lim * lim


# --------------------------------------------------------------------------
# outline pairs: corners and sides (concentric, coaxial)
# --------------------------------------------------------------------------

def _pair(ak, bk, aj, bj):
    """Outline k against outline j: F(phi) = (a_k cos/a_j)^2 + (b_k sin/b_j)^2
    - 1 = (num + den) cos^2 phi - den on k's own parameter, with
    num = a_k^2/a_j^2 - 1 and den = 1 - b_k^2/b_j^2 in product form
    (F = num cos^2 - den sin^2). They
    cross iff num and den share a sign, at tan^2 phi* = num / den. Returns
    (num, den, cross, phi*)."""
    num = (ak - aj) * (ak + aj) / (aj * aj)
    den = (bj - bk) * (bj + bk) / (bj * bj)
    cross = ((num > 0.0) & (den > 0.0)) | ((num < 0.0) & (den < 0.0))
    phi = mx.arctan2(mx.sqrt(mx.abs(num)), mx.sqrt(mx.abs(den)))
    return num, den, cross, phi


def _inside(num, den, cross, cos2, sin2):
    """Whether k's point at own parameter phi is inside j. With corners,
    the sign of F = num cos^2 - den sin^2 there (the arc lies between
    corners, so away from them) -- without forming num + den, which a
    nearly edge-on ring makes a sum of terms 1e8 apart, rounding away the
    short arcs it cuts from the planet's outline; without corners, the
    whole outline is on one side, read at cos^2 = 1/2 -- never a point near
    a tangency."""
    return mx.where(cross, num * cos2 < den * sin2, num < den)


def _arcs(bps, valid):
    """Sub-arcs of one closed curve between its valid breakpoints (lists),
    as (lo, hi, ok) lists of fixed length len(bps): sorted, the last
    wrapping round; no breakpoint at all is one whole turn."""
    key = mx.sort(mx.stack([mx.where(v, b, mx.inf) for b, v in zip(bps, valid)], -1), axis=-1)
    n = sum(v.astype(mx.int32) for v in valid)
    K = len(bps)
    first = mx.where(n > 0, key[..., 0], 0.0)
    los, his, oks = [], [], []
    for i in range(K):
        lo = key[..., i]
        nxt = key[..., i + 1] if i + 1 < K else first
        hi = mx.where(i + 1 < n, nxt, first + _TWO_PI)
        ok = (i < n) | ((n == 0) & (i == 0))
        lo = mx.where((n == 0) & (i == 0), 0.0, lo)
        hi = mx.where((n == 0) & (i == 0), _TWO_PI, hi)
        los.append(mx.where(ok, lo, 0.0))
        his.append(mx.where(ok, hi, 0.0))
        oks.append(ok)
    return los, his, oks


# --------------------------------------------------------------------------
# the columns
# --------------------------------------------------------------------------

def _h(n_or_p, s, even):
    """The potential h(s) of one generator column: h_n(s) for mu^{2n}, or
    1/(2p(p - s)) for a pole at level p."""
    if even:
        al = _alpha(_N_MAX)[n_or_p]
        acc = 0.0 * s
        for k in range(len(al) - 1, -1, -1):
            acc = acc * s + float(al[k])
        return acc
    p = n_or_p
    return 1.0 / (2.0 * p * (p - s))


def _weight(flags, T):
    """Blocked fraction for membership flags [planet, edge_0, ..., edge_n]
    (edges innermost first): 1 in the planet, else the band's T."""
    band = 0.0 * T[0]
    for k, t in enumerate(T):
        band = mx.where(flags[2 + k] & ~flags[1 + k], t, band)
    return mx.where(flags[0], 1.0, band)


def _region_cols(x0, y0, conics, T, law, dt):
    """Occulted generator columns (positive) of the weighted occultor:
    conics[0] the planet, conics[1:] the ring edges innermost first (all
    concentric at (x0, y0), coaxial); T[k] the blocked fraction between
    edges k and k + 1."""
    eps = law.eps
    n_c = len(conics)
    gx0, gy0 = _sg(x0), _sg(y0)
    g = [(_sg(a), _sg(b)) for a, b in conics]
    circ = [(ga - gb) <= _F_SW[dt] * ga for ga, gb in g]
    cross = [_crossings(gx0, gy0, ga, gb, c, dt) for (ga, gb), c in zip(g, circ)]
    # planet-edge pairs (edges are homothetic: never cross each other)
    pairs = {}
    for k in range(n_c):
        for j in range(n_c):
            if k != j and (k == 0 or j == 0):
                pairs[(k, j)] = _pair(g[k][0], g[k][1], g[j][0], g[j][1])

    ncol = 3 + len(eps)
    occ = [0.0 * x0 for _ in range(ncol)]
    limb_th, limb_ok = [], []
    for k in range(n_c):
        ga, gb = g[k]
        a, b = conics[k]
        phis, valid = cross[k]
        for ph, v in zip(phis, valid):
            s_, c_ = sincos(ph)
            limb_th.append(_mod2pi(mx.arctan2(gy0 + gb * s_, gx0 + ga * c_)))
            limb_ok.append(v)
        bps, bv = list(phis), list(valid)
        for j in range(n_c):
            if (k, j) in pairs:
                num, den, cr, ph = pairs[(k, j)]
                bps += [ph, _PI - ph, _PI + ph, _TWO_PI - ph]
                bv += [cr] * 4
        los, his, oks = _arcs(bps, bv)
        LO, HI = _sg(mx.stack(los, -1)), _sg(mx.stack(his, -1))
        OK = mx.stack(oks, -1)
        mid = 0.5 * (LO + HI)
        sm, cm = sincos(mid)
        xs, ys = gx0[..., None] + ga[..., None] * cm, gy0[..., None] + gb[..., None] * sm
        in_star = xs * xs + ys * ys < 1.0
        cos2, sin2 = cm * cm, sm * sm
        flags_in, flags_out = [], []
        for j in range(n_c):
            if j == k:
                f_in, f_out = mx.ones(LO.shape, dtype=mx.bool_), mx.zeros(LO.shape, dtype=mx.bool_)
            elif (k, j) in pairs:
                num, den, cr, _ = pairs[(k, j)]
                f_in = f_out = _inside(num[..., None], den[..., None], cr[..., None], cos2, sin2)
            else:                                   # edge vs edge: by size
                f_in = f_out = mx.full(LO.shape, j > k, dtype=mx.bool_)
            flags_in.append(f_in)
            flags_out.append(f_out)
        Te = [t[..., None] for t in T]
        kappa = _weight(flags_in, Te) - _weight(flags_out, Te)
        kappa = mx.where(OK & in_star, kappa, 0.0)
        # the arcs: live coefficients, frozen endpoints
        shp = LO.shape
        bx = lambda v: mx.broadcast_to(v[..., None], shp)
        X0, Y0, A, B = bx(x0), bx(y0), bx(a), bx(b)
        S0, S1, S2, K0, K1 = _coeffs(X0, Y0, A, B)
        q = _q_coeffs(S0, S1, S2, K0, K1, _N_MAX)
        cols = list(_even_arc(q, LO, HI))
        cen = bx(_centred(gx0, gy0, ga, dt) & circ[k])
        # the general pole branch on a centred circle's lanes gets a safe,
        # gradient-free offset (its roots are taken there too): a masked
        # lane must not poison the gradient
        pS = _coeffs(mx.where(cen, _sg(0.5 * A), X0), mx.where(cen, 0.0, Y0), A, B)
        sl, cl = sincos(LO)
        sh, ch = sincos(HI)
        for e in eps:
            p = 1.0 + e
            z0 = [_C(mx.broadcast_to(z.re[..., None], shp), mx.broadcast_to(z.im[..., None], shp))
                  for z in _outline_roots(gx0, gy0, ga, gb, circ[k], p, dt)]
            v = _pole_arc(*pS, p, z0, LO, HI)
            R = A
            c0 = R * R / (2.0 * p * (p - R * R))
            c1 = R * (p + R * R) / (2.0 * p * (p - R * R) ** 2)
            vc = c0 * (HI - LO) + c1 * (X0 * (sh - sl) + Y0 * (cl - ch))
            cols.append(mx.where(cen, vc, v))
        # gradient-only corner terms at both ends of every arc
        gX0, gY0, gA, gB = _sg(X0), _sg(Y0), _sg(A), _sg(B)

        def C(sn, cs):
            xp, yp = gX0 + gA * cs, gY0 + gB * sn
            xl, yl = X0 + A * cs, Y0 + B * sn
            return xp * yl - yp * xl, xp * xp + yp * yp
        Cl, s_lo = C(sl, cl)
        Ch, s_hi = C(sh, ch)
        hs = [(n, True) for n in range(3)] + [(1.0 + e, False) for e in eps]
        for idx, (arg, even) in enumerate(hs):
            corner = _h(arg, s_lo, even) * Cl - _h(arg, s_hi, even) * Ch
            corner = corner - _sg(corner)
            occ[idx] = occ[idx] + mx.sum(kappa * mx.where(OK, cols[idx] + corner, 0.0), axis=-1)
    # the limb
    los, his, oks = _arcs(limb_th, limb_ok)
    h1 = [1.0 / (2.0 * (n + 1.0)) for n in range(3)] + [1.0 / (2.0 * (1.0 + e) * e) for e in eps]
    for lo, hi, ok in zip(los, his, oks):
        sm, cm = sincos(0.5 * (lo + hi))
        flags = []
        for ga, gb in g:
            u, v = (cm - gx0) / ga, (sm - gy0) / gb
            flags.append(u * u + v * v < 1.0)
        w = mx.where(ok, _weight(flags, T), 0.0)
        dth = _sg(hi - lo)
        for idx in range(ncol):
            occ[idx] = occ[idx] + w * h1[idx] * dth
    return occ


def generator_cols_ring(x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau, law):
    """Generator deviation columns [E0, E1, E2, P_k...] (minus the occulted
    flux of mu^0, mu^2, mu^4 and each pole of ``law``) of a ringed planet,
    elementwise over broadcast arguments. (x0, y0): the planet's centre in
    the principal frame (``oblate.principal_frame``)."""
    law = get_law(law)
    dt = x0.dtype
    # broadcast without arithmetic: ring_tau may be inf (inf * 0 is NaN)
    shape = np.broadcast_shapes(*[v.shape for v in (x0, y0, rp, f, ring_in, ring_out,
                                                    ring_f, ring_tau) if isinstance(v, mx.array)])
    zero = mx.zeros(shape, dtype=dt)
    x0, y0 = x0 + zero, y0 + zero
    planet, edges = ring_axes(rp, f, ring_in, ring_out, ring_f)
    conics = [(planet[0] + zero, planet[1] + zero)] + [(a + zero, b + zero) for a, b in edges]
    T = [ring_opacity(ring_tau, ring_f) + zero]
    occ = _region_cols(x0, y0, conics, T, law, dt)
    return [-o for o in occ]


@fp64_on_cpu(any_arg=True)
def shape_cols_ring(x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau, law) -> mx.array:
    """Shape-basis columns [E0, S_1..S_n] of a ringed planet, stacked on a
    trailing axis -- the ``ld_basis`` contract of the spherical and oblate
    paths (flux - 1 == (B @ c) / (N @ c), c = (1, -w),
    N = hybrid_norms(law)). Arguments as ``generator_cols_ring``; all are
    cast to x0's dtype."""
    law = get_law(law)
    dt = x0.dtype
    _check_ring(f, ring_in, ring_out, ring_f, ring_tau)
    args = [_to_dtype(v, dt) for v in (y0, rp, f, ring_in, ring_out, ring_f, ring_tau)]
    from .metal import _gpu_stream_active
    if _gpu_stream_active():
        return _shape_cols(law)(x0, *args)
    return _shape_cols_compiled(law)(x0, *args)


def _shape_cols(law):
    S = law.generator_matrix()

    def cols(x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau):
        gen = generator_cols_ring(x0, y0, rp, f, ring_in, ring_out, ring_f,
                                  ring_tau, law)
        return mx.stack([gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)],
                        axis=-1)
    return cols


# On the CPU the columns' graph is compiled once per law (fused: ~2x faster
# in float64, ~3x for short light curves). Only this piece: MLX 0.32's
# mx.compile fails ("unordered_map::at") when the orbit feeding it is traced
# in the same graph, so the callers around it stay eager. Not on the GPU,
# where MLX's fused Metal kernels for this graph abort (the float32 GPU
# route is metal_rings' own kernels; the graph there is use_metal=False).
_COMPILED = {}


def _shape_cols_compiled(law):
    fn = _COMPILED.get(law.definition)
    if fn is None:
        fn = _COMPILED[law.definition] = mx.compile(_shape_cols(law))
    return fn


@fp64_on_cpu(any_arg=True)
def flux_dev_ring(x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau, w, law) -> mx.array:
    """F - 1 for a ringed planet and a hybrid law with shape weights w (a
    host sequence, (n_w,) or (..., n_w)); arguments as ``shape_cols_ring``."""
    B = shape_cols_ring(x0, y0, rp, f, ring_in, ring_out, ring_f, ring_tau, law)
    return combine_cols([B[..., k] for k in range(B.shape[-1])], w, law)
