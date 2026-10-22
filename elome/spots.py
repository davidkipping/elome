"""Starspots crossed by a spherical planet, for the hybrid limb-darkening
laws -- the reference MLX graph, fp64 and fp32.

A spot is a circular cap on the stellar surface: unit centre n (sky frame:
x, y on the sky, z towards the observer, mu = z on the surface), angular
radius alpha and contrast c, its intensity c times the photosphere's at
the same mu (c < 1 dark, c > 1 a facula). The planet is an opaque disc of
radius rp at (X, Y). The blocked fraction is

    w = 1 inside the planet, else sum_k (1 - c_k) [visible spot k],

so overlapping spots add (an umbra in a penumbra is two concentric caps),
and the columns are those of the integral of w I over the disc: the flux
is normalised to the unspotted star, F - 1 = (B . c) / (N . c) with the
``ld_basis`` contract of the other paths, and carries the spots'
rotational modulation out of transit as well as the crossing in it.

A spot's visible region is the set of disc points whose front surface
point lies in the cap: bounded by the rim where mu > 0 and by the limb.
The rim, P(psi) = cos(alpha) n + sin(alpha) (cos(psi) e1 + sin(psi) e2)
with e1 = (n_y, -n_x, 0) / |n_xy| and e2 = (n_z n_xy / |n_xy|, -|n_xy|),
is on the sky an ellipse in ``oblate.py``'s form in the frame
(x along e1, y along n_xy): centre (0, cos(alpha) |n_xy|), semi-axes
a = sin(alpha), b = sin(alpha) n_z -- signed, so a cap facing away is
walked clockwise, keeping the cap on the arc's left. Along it

    mu = mu_a + mu_b sin(psi),  mu_a = cos(alpha) n_z,  mu_b = -sin(alpha) |n_xy|,

linear, because the rim lies on the sphere; so s = x^2 + y^2 = 1 - mu^2,
and a pole's quartic D(z) = z^2 (s - p) = -z^2 (mu^2 + eps) splits into
(mu_b^2 / 4) (z^2 + 2 i kappa z - 1) for kappa = (mu_a +- i sqrt(eps)) /
mu_b, each with root product -1: closed-form roots, no iteration. The
arcs are then ``oblate.py``'s arc integrals. The rim meets the limb where
mu = 0, from the product form q = sin^2(alpha) - n_z^2 =
(sin(alpha) - |n_z|)(sin(alpha) + |n_z|): asin and acos near 1 would lose
half the digits of a small spot on the limb, whose rim and limb arcs
cancel to a sliver of order alpha^3.

By Green's theorem (``rings.py``), every outline -- the planet and each
rim -- is walked once, split at its limb points and at its corners with
the other outlines, each sub-arc weighted by the jump in w across it, plus
the limb's arcs weighted by w just inside. The planet-rim corners come
from the oblate crossing solver in the frame centred on the planet and
scaled by its radius (the planet the unit circle, the rim an ellipse);
rim-rim corners in closed form on the sphere. Membership is read at arc
midpoints, for a spot from the front surface point.

Gradients as in ``rings.py``: roots and arc endpoints are frozen; the
endpoints' motion is restored by a gradient-only corner term
kappa h(s) [C(P_lo) - C(P_hi)] per arc and column, with C taken in the sky
frame (a rim's own frame turns with its spot).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .dtypes import fp64_on_cpu
from .hybrid import _combine, combine_cols, get_law
from .oblate import (_C, _F_SW, _N_MAX, _alpha, _coeffs, _csqrt, _cwhere, _even_arc, _mod2pi,
                     _pole_arc, _q_coeffs, _to_dtype)
from .rings import (_BIG, _S1_CENTRED, _TINY, _arcs, _centred, _circle_roots, _crossings,
                    _h)
from .trig import sincos

__all__ = ["generator_cols_spots", "shape_cols_spots", "flux_dev_spots", "spot_vector"]

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_sg = mx.stop_gradient

# stand-in for a rim's pole roots on lanes where it is centred on the disc
# (|n_xy| ~ 0, mu_b ~ 0: the closed form divides by mu_b); those lanes take
# the centred-circle expansion instead. n = (sin 0.4, 0, cos 0.4).
_SAFE_N = (math.sin(0.4), 0.0, math.cos(0.4))


def spot_vector(lon, lat):
    """Unit centre vector of a spot at sky-frame longitude lon (from the
    line of sight towards +x) and latitude lat (towards +y), radians:
    (cos lat sin lon, sin lat, cos lat cos lon). mx or host values."""
    if isinstance(lon, mx.array) or isinstance(lat, mx.array):
        sl, cl = sincos(lon if isinstance(lon, mx.array) else mx.array(lon, dtype=lat.dtype))
        sb, cb = sincos(lat if isinstance(lat, mx.array) else mx.array(lat, dtype=lon.dtype))
        return mx.stack([cb * sl, sb, cb * cl], axis=-1)
    lon, lat = np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)


# --------------------------------------------------------------------------
# one rim
# --------------------------------------------------------------------------

class _Rim:
    """A spot's rim: its frame, ellipse, mu along it, limb points; live
    (differentiable) and frozen copies where they are needed."""

    def __init__(self, nx, ny, nz, alpha, dt):
        self.nx, self.ny, self.nz, self.alpha = nx, ny, nz, alpha
        nxy2 = nx * nx + ny * ny
        ok = nxy2 > _TINY[dt]
        nxy = mx.where(ok, mx.sqrt(mx.where(ok, nxy2, 1.0)), 0.0)
        nxs = mx.where(ok, nxy, 1.0)
        # frame: x along e1 = (n_y, -n_x)/|n_xy|, y along n_xy/|n_xy|;
        # a centred spot takes the sky frame
        self.ux, self.uy = mx.where(ok, ny / nxs, 1.0), mx.where(ok, -nx / nxs, 0.0)
        self.vx, self.vy = mx.where(ok, nx / nxs, 0.0), mx.where(ok, ny / nxs, 1.0)
        sa, ca = sincos(alpha)
        self.sa, self.ca, self.nxy = sa, ca, nxy
        self.x0 = 0.0 * alpha
        self.y0 = ca * nxy
        self.a = sa
        self.b = sa * nz
        self.mu_a = ca * nz
        self.mu_b = -sa * nxy
        # mu = 0 on the rim: q = sin^2(alpha) - n_z^2, product form
        q = (sa - mx.abs(nz)) * (sa + mx.abs(nz))
        self.limb = q > 0.0
        r = mx.where(self.limb, mx.sqrt(mx.where(self.limb, q, 1.0)), 0.0)
        den = mx.where(self.limb, sa * nxy, 1.0)
        mbs = mx.where(self.limb, self.mu_b, -1.0)
        self.psi_limb = mx.arctan2(-self.mu_a / mbs, r / den)        # and pi - it
        self.half_limb = mx.arctan2(r, ca)                            # limb arc / 2
        self.phi_n = mx.arctan2(ny, nx)
        # for the outline's live evaluation, smooth through the disc centre
        # (|n_xy| is not): y0 = cos(alpha) n . v and the frame's turn
        # -n . u = |n_xy| d(phi), both on the frame held fixed, and (a^2 -
        # b^2) / 4 on the sphere, where a ~ |b| near the centre cancels.
        # Only derivatives along the sphere are used.
        fu = (_sg(self.ux), _sg(self.uy))
        self.yl = ca * (nx * _sg(self.vx) + ny * _sg(self.vy))
        turn = -(nx * fu[0] + ny * fu[1])
        self.turn = turn - _sg(turn)
        self.s2 = 0.25 * (sa * sa) * (nx * nx + ny * ny)
        # a nearly centred rim: the pole roots' closed form divides by
        # |mu_b| = sin(alpha) |n_xy| (not |b| y0, which an edge-on rim makes
        # small, nor a y0, which a great circle does), and S1, S2 are then
        # O(mu_b), O(mu_b^2): the centred expansion holds
        self.cen = (self.sa * nxy < _S1_CENTRED[dt]) | ~ok

    def sky(self, psi):
        """Sky (x, y) and mu of the rim point at psi (live)."""
        s, c = sincos(psi)
        xr, yr = self.a * c, self.y0 + self.b * s
        return (xr * self.ux + yr * self.vx, xr * self.uy + yr * self.vy,
                self.mu_a + self.mu_b * s)

    def own(self, x, y, z):
        """Whether the 3D point (x, y, z) is in this cap (x may carry a
        trailing arc axis beyond the rim's shape)."""
        ex = (lambda v: v[..., None]) if x.ndim > self.nx.ndim else (lambda v: v)
        return x * ex(self.nx) + y * ex(self.ny) + z * ex(self.nz) >= ex(self.ca)

    def param(self, x, y, z):
        """The rim parameter of a 3D rim point."""
        # P - cos(alpha) n = sin(alpha) (cos psi e1 + sin psi e2); e2 =
        # n_z v - |n_xy| z_hat, with v the frame's y axis on the sky
        c = x * self.ux + y * self.uy
        s = self.nz * (x * self.vx + y * self.vy) - self.nxy * z
        return mx.arctan2(s, c)


def _rim_roots(rim, eps, dt):
    """The four roots of a rim's pole quartic, frozen: per sign the larger
    root of z^2 + 2 i kappa z - 1 without cancellation, its partner -1/z.
    Centred lanes take a stand-in rim (they use the centred expansion)."""
    sx, sy, sz = _SAFE_N
    mb = mx.where(rim.cen, -rim.sa * math.hypot(sx, sy), rim.mu_b)
    ma = mx.where(rim.cen, rim.ca * sz, rim.mu_a)
    se = math.sqrt(eps)
    out = []
    for sgn in (1.0, -1.0):
        kre, kim = ma / mb, sgn * se / mb
        mik = _C(kim, -kre)                          # -i kappa
        w = _csqrt(_C(1.0 - (kre * kre - kim * kim), -2.0 * kre * kim))
        z1, z2 = mik + w, mik - w
        big = _cwhere(z1.abs2() >= z2.abs2(), z1, z2)
        out += [big, -(1.0 / big)]
    return [z.sg() for z in out]


def _safe_rim_coeffs(rim):
    """The stand-in rim's (x0, y0, a, b) on centred lanes, else the rim's."""
    sx, sy, sz = _SAFE_N
    return (rim.x0, mx.where(rim.cen, rim.ca * math.hypot(sx, sy), rim.yl), rim.a,
            mx.where(rim.cen, rim.sa * sz, rim.b))


# --------------------------------------------------------------------------
# arcs of one outline
# --------------------------------------------------------------------------

def _hs(eps):
    return [(n, True) for n in range(3)] + [(1.0 + e, False) for e in eps]


def _outline_cols(coef, pole_coef, roots, cen, LO, HI, kappa, OK, sky_live, sky_frozen, eps,
                  s2=None, turn=None):
    """kappa-weighted arc integrals of one outline over the arcs [LO, HI]
    (trailing axis), with the corner terms: a list of columns.

    coef, pole_coef: (x0, y0, a, b) of the outline in its own frame (live;
    pole_coef with a stand-in on centred lanes); roots: per eps, four
    frozen _C; cen: the centred lanes (first-order expansion);
    sky_live(psi), sky_frozen(psi): the arc point on the sky -- or, with
    ``turn``, in a frame turning by d(phi), turn[0] = |n_xy| d(phi) a pure
    tangent, which adds [g(s_lo) - g(s_hi)] d(phi), g = h s, to each arc's
    corner term, with (s_lo - s_hi) / |n_xy| = turn[1](LO, HI) formed
    without cancellation. s2: (a^2 - b^2) / 4 given (live), in place of
    the coefficients'."""
    shp = LO.shape
    bx = lambda v: mx.broadcast_to(v[..., None], shp)
    X0, Y0, A, B = (bx(v) for v in coef)
    S0, S1, S2, K0, K1 = _coeffs(X0, Y0, A, B)
    if s2 is not None:
        S2 = bx(s2)
    cols = list(_even_arc(_q_coeffs(S0, S1, S2, K0, K1, _N_MAX), LO, HI))
    pS = _coeffs(*(bx(v) for v in pole_coef))
    if s2 is not None:
        pS = (pS[0], pS[1], mx.where(bx(cen), pS[2], S2), pS[3], pS[4])
    cenb = bx(cen)
    sl, cl = sincos(LO)
    sh, ch = sincos(HI)
    for e, z4 in zip(eps, roots):
        p = 1.0 + e
        z0 = [_C(bx(z.re), bx(z.im)) for z in z4]
        v = _pole_arc(*pS, p, z0, LO, HI)
        # centred circle: first order in the offset and in A - B (rings.py)
        R2 = A * B
        c0 = R2 / (2.0 * p * (p - R2))
        c1 = A * (p + R2) / (2.0 * p * (p - R2) ** 2)
        c2 = R2 / (2.0 * p * (p - R2) ** 2)
        vc = (c0 * (HI - LO) + c1 * (X0 * (sh - sl) + Y0 * (cl - ch))
              + c2 * (0.5 * (A * A - B * B)) * (sh * ch - sl * cl))
        cols.append(mx.where(cenb, vc, v))

    def C(psi):
        xf, yf = sky_frozen(psi)
        xl, yl = sky_live(psi)
        return xf * yl - yf * xl, xf * xf + yf * yf
    Cl, s_lo = C(LO)
    Ch, s_hi = C(HI)
    if turn is not None:
        dphi, ds = bx(turn[0]), _sg(turn[1](LO, HI))
    out = []
    for idx, (arg, even) in enumerate(_hs(eps)):
        corner = _h(arg, s_lo, even) * Cl - _h(arg, s_hi, even) * Ch
        corner = corner - _sg(corner)
        if turn is not None:
            corner = corner + _g_diff(arg, s_lo, s_hi, ds, even) * dphi
        out.append(mx.sum(kappa * mx.where(OK, cols[idx] + corner, 0.0), axis=-1))
    return out


def _g_diff(n_or_p, s_lo, s_hi, ds, even):
    """g(s_lo) - g(s_hi) for g = h s, given ds = s_lo - s_hi."""
    if even:
        al = _alpha(_N_MAX)[n_or_p]
        acc = 0.0 * ds
        for j, a in enumerate(al):
            pw = 0.0 * ds
            for i in range(j + 1):
                pw = pw + s_lo ** i * s_hi ** (j - i)
            acc = acc + float(a) * pw
        return ds * acc
    p = n_or_p
    return ds / (2.0 * (p - s_lo) * (p - s_hi))


# --------------------------------------------------------------------------
# the columns
# --------------------------------------------------------------------------

def _rim_rim(ri, rj):
    """The two points where rims i and j meet on the sphere (frozen), as
    3D components, and their validity."""
    nix, niy, niz = (_sg(v) for v in (ri.nx, ri.ny, ri.nz))
    njx, njy, njz = (_sg(v) for v in (rj.nx, rj.ny, rj.nz))
    ci, cj = _sg(ri.ca), _sg(rj.ca)
    d = nix * njx + niy * njy + niz * njz
    det = (1.0 - d) * (1.0 + d)
    okd = det > 1e-12
    dets = mx.where(okd, det, 1.0)
    a_, b_ = (ci - d * cj) / dets, (cj - d * ci) / dets
    bx_, by_, bz_ = a_ * nix + b_ * njx, a_ * niy + b_ * njy, a_ * niz + b_ * njz
    t2 = (1.0 - (bx_ * bx_ + by_ * by_ + bz_ * bz_)) / dets
    ok = okd & (t2 > 0.0)
    t = mx.sqrt(mx.where(ok, t2, 0.0))
    mx_, my_, mz_ = niy * njz - niz * njy, niz * njx - nix * njz, nix * njy - niy * njx
    return [((bx_ + s * t * mx_, by_ + s * t * my_, bz_ + s * t * mz_), ok) for s in (1.0, -1.0)]


def _region_cols(X, Y, rp, spots, c, law, dt):
    """Occulted generator columns (positive): the integral of w I over the
    disc. spots: list of (nx, ny, nz, alpha); c: list of contrasts."""
    eps = law.eps
    K = len(spots)
    ncol = 3 + len(eps)
    rims = [_Rim(*s, dt) for s in spots]
    gr = [_Rim(*(_sg(v) for v in s), dt) for s in spots]       # frozen copies
    gX, gY, grp = _sg(X), _sg(Y), _sg(rp)
    one = mx.ones(X.shape, dtype=mx.bool_)

    def weight(in_planet, in_spots):
        acc = 0.0 * X
        for k in range(K):
            acc = acc + mx.where(in_spots[k], 1.0 - c[k], 0.0)
        return mx.where(in_planet, 1.0, acc)

    def cap_flags(x, y, z):
        return [r.own(x, y, z) for r in gr]

    # ---- corners -----------------------------------------------------------
    # planet-rim, in the planet's unit frame: the rim as an ellipse with |b|
    # (a cap facing away is the mirror image, psi -> -psi)
    pr_psi, pr_ok = [], []
    for g in gr:
        px = gX * g.ux + gY * g.uy
        py = gX * g.vx + gY * g.vy
        ab = mx.abs(g.b)
        circ = (g.a - ab) <= _F_SW[dt] * g.a
        phis, valid = _crossings(-px / grp, (g.y0 - py) / grp, g.a / grp, ab / grp, circ, dt)
        flip = g.nz < 0
        pr_psi.append([_mod2pi(mx.where(flip, -ph, ph)) for ph in phis])
        pr_ok.append(valid)
    # rim-rim on the sphere
    rr = {}
    for i in range(K):
        for j in range(i + 1, K):
            rr[(i, j)] = _rim_rim(gr[i], gr[j])
    # the planet's limb crossings (a circle)
    pl_phi, pl_ok = _crossings(gX, gY, grp, grp, one, dt)

    occ = [0.0 * X for _ in range(ncol)]
    limb_th, limb_ok = [], []

    # ---- the planet's outline ---------------------------------------------
    bps, bv = list(pl_phi[:2]), list(pl_ok[:2])
    for k, g in enumerate(gr):
        for ph, v in zip(pr_psi[k], pr_ok[k]):
            x, y, _ = g.sky(ph)
            bps.append(_mod2pi(mx.arctan2(y - gY, x - gX)))
            bv.append(v)
    for ph, v in zip(pl_phi[:2], pl_ok[:2]):
        s_, c_ = sincos(ph)
        limb_th.append(_mod2pi(mx.arctan2(gY + grp * s_, gX + grp * c_)))
        limb_ok.append(v)
    los, his, oks = _arcs(bps, bv)
    LO, HI = _sg(mx.stack(los, -1)), _sg(mx.stack(his, -1))
    OK = mx.stack(oks, -1)
    sm, cm = sincos(0.5 * (LO + HI))
    xs, ys = gX[..., None] + grp[..., None] * cm, gY[..., None] + grp[..., None] * sm
    r2 = xs * xs + ys * ys
    in_star = r2 < 1.0
    zs = mx.sqrt(mx.where(in_star, 1.0 - r2, 0.0))
    fl = cap_flags(xs, ys, zs)
    acc = 0.0 * LO
    for k in range(K):
        acc = acc + mx.where(fl[k], 1.0 - c[k][..., None], 0.0)
    kappa = mx.where(OK & in_star, 1.0 - acc, 0.0)
    # roots: closed form for a circle (rings.py), centred lanes expanded
    cen = _centred(gX, gY, grp, dt)
    cx0 = mx.where(cen, 0.5 * grp, gX)
    roots = [_circle_roots(cx0, mx.where(cen, 0.0, gY), grp, 1.0 + e, dt) for e in eps]
    roots = [[z.sg() for z in r] for r in roots]
    pcoef = (mx.where(cen, _sg(0.5 * rp), X), mx.where(cen, 0.0, Y), rp, rp)

    def p_live(psi):
        s, c_ = sincos(psi)
        return X[..., None] + rp[..., None] * c_, Y[..., None] + rp[..., None] * s

    def p_frozen(psi):
        s, c_ = sincos(psi)
        return gX[..., None] + grp[..., None] * c_, gY[..., None] + grp[..., None] * s
    cols = _outline_cols((X, Y, rp, rp), pcoef, roots, cen, LO, HI, kappa, OK,
                         p_live, p_frozen, eps)
    occ = [o + v for o, v in zip(occ, cols)]

    # ---- each rim -------------------------------------------------------------
    for k in range(K):
        rim, g = rims[k], gr[k]
        p1 = g.psi_limb
        bps = [_mod2pi(p1), _mod2pi(_PI - p1)] + list(pr_psi[k])
        bv = [g.limb, g.limb] + list(pr_ok[k])
        for (i, j), pts in rr.items():
            if k in (i, j):
                for P, v in pts:
                    bps.append(_mod2pi(g.param(*P)))
                    bv.append(v)
        for sgn in (1.0, -1.0):
            limb_th.append(_mod2pi(g.phi_n + sgn * g.half_limb))
            limb_ok.append(g.limb)
        los, his, oks = _arcs(bps, bv)
        LO, HI = _sg(mx.stack(los, -1)), _sg(mx.stack(his, -1))
        OK = mx.stack(oks, -1)
        ex = lambda v: v[..., None]
        # the midpoint, frozen, in 3D
        mid = 0.5 * (LO + HI)
        s, c_ = sincos(mid)
        xr, yr = ex(g.a) * c_, ex(g.y0) + ex(g.b) * s
        xm = xr * ex(g.ux) + yr * ex(g.vx)
        ym = xr * ex(g.uy) + yr * ex(g.vy)
        mum = ex(g.mu_a) + ex(g.mu_b) * s
        front = mum > 0.0
        in_p = (xm - ex(gX)) ** 2 + (ym - ex(gY)) ** 2 < ex(grp) ** 2
        kappa = mx.where(OK & front & ~in_p, 1.0 - c[k][..., None], 0.0)
        # other spots: none of their flags changes across rim k
        roots = [_rim_roots(g, e, dt) for e in eps]

        # C in the rim's own frame; the frame's turn (1 / |n_xy| near the
        # disc centre) separately, as d(C) = s d(phi) at each end
        def r_live(psi, rim=rim):
            s_, c2 = sincos(psi)
            return rim.a[..., None] * c2, rim.yl[..., None] + rim.b[..., None] * s_

        def r_frozen(psi, g=g):
            s_, c2 = sincos(psi)
            return g.a[..., None] * c2, g.y0[..., None] + g.b[..., None] * s_

        def ds(lo, hi, g=g):
            sl, sh = sincos(lo)[0], sincos(hi)[0]
            return ((sh - sl) * (sh + sl) * ex(g.sa * g.sa * g.nxy)
                    + 2.0 * ex(g.ca * g.b) * (sl - sh))
        cols = _outline_cols((rim.x0, rim.yl, rim.a, rim.b), _safe_rim_coeffs(rim), roots,
                             g.cen, LO, HI, kappa, OK, r_live, r_frozen, eps,
                             s2=rim.s2, turn=(rim.turn, ds))
        occ = [o + v for o, v in zip(occ, cols)]

    # ---- the limb ---------------------------------------------------------------
    los, his, oks = _arcs(limb_th, limb_ok)
    h1 = [1.0 / (2.0 * (n + 1.0)) for n in range(3)] + [1.0 / (2.0 * (1.0 + e) * e) for e in eps]
    for lo, hi, ok in zip(los, his, oks):
        sm, cm = sincos(0.5 * (lo + hi))
        in_p = (cm - gX) ** 2 + (sm - gY) ** 2 < grp * grp
        zero = 0.0 * cm
        w = mx.where(ok, weight(in_p, cap_flags(cm, sm, zero)), 0.0)
        dth = _sg(hi - lo)
        for idx in range(ncol):
            occ[idx] = occ[idx] + w * h1[idx] * dth
    return occ


def _prep(X, Y, rp, n, alpha, contrast):
    """Cast to X's dtype and broadcast to a common shape: X, Y, rp of the
    point shape; spot fields as lists of K arrays of it."""
    dt = X.dtype
    Y, rp = _to_dtype(Y, dt), _to_dtype(rp, dt)
    n = n if isinstance(n, mx.array) else mx.array(np.asarray(n, dtype=np.float64), dtype=dt)
    n = n.astype(dt)
    alpha, contrast = _to_dtype(alpha, dt), _to_dtype(contrast, dt)
    if n.ndim < 2 or n.shape[-1] != 3:
        raise ValueError("n must be (..., K, 3): one unit vector per spot")
    K = n.shape[-2]
    alpha = mx.broadcast_to(alpha, n.shape[:-1]) if alpha.ndim == 0 else alpha
    contrast = mx.broadcast_to(contrast, n.shape[:-1]) if contrast.ndim == 0 else contrast
    point = np.broadcast_shapes(X.shape, Y.shape, rp.shape, n.shape[:-2],
                                alpha.shape[:-1], contrast.shape[:-1])
    zero = mx.zeros(point, dtype=dt)
    X, Y, rp = X + zero, Y + zero, rp + zero
    nb = mx.broadcast_to(n, point + (K, 3))
    ab = mx.broadcast_to(alpha, point + (K,))
    cb = mx.broadcast_to(contrast, point + (K,))
    spots = [(nb[..., k, 0], nb[..., k, 1], nb[..., k, 2], ab[..., k]) for k in range(K)]
    return X, Y, rp, spots, [cb[..., k] for k in range(K)]


def generator_cols_spots(X, Y, rp, n, alpha, contrast, law):
    """Generator deviation columns [E0, E1, E2, P_k...] (minus the integral
    of w I over the disc, for mu^0, mu^2, mu^4 and each pole of ``law``) of
    a spotted star crossed by a planet, elementwise over the point shape.
    (X, Y): the planet's sky position, rp its radius; n (..., K, 3) the
    spots' unit centre vectors, alpha (..., K) their angular radii
    (radians), contrast (..., K) their intensity ratios."""
    law = get_law(law)
    X, Y, rp, spots, c = _prep(X, Y, rp, n, alpha, contrast)
    occ = _region_cols(X, Y, rp, spots, c, law, X.dtype)
    return [-o for o in occ]


@fp64_on_cpu(any_arg=True)
def shape_cols_spots(X, Y, rp, n, alpha, contrast, law) -> mx.array:
    """Shape-basis columns [E0, S_1..S_n] stacked on a trailing axis -- the
    ``ld_basis`` contract (flux - 1 == (B @ c) / (N @ c), c = (1, -w),
    N = hybrid_norms(law), the unspotted star's) -- of a spotted star
    crossed by a planet, elementwise over the point shape. (X, Y): the
    planet's sky position, rp its radius (stellar radii); n (..., K, 3)
    the spots' unit centre vectors in the same sky frame, z towards the
    observer (``spot_vector``); alpha (..., K) their angular radii
    (radians); contrast (..., K) their intensities over the photosphere's.
    All cast to X's dtype; float64 on the CPU stream unless float32."""
    law = get_law(law)
    X, Y, rp, spots, c = _prep(X, Y, rp, n, alpha, contrast)
    args = [X, Y, rp] + [v for s in spots for v in s] + c
    from .metal import _gpu_stream_active
    if _gpu_stream_active():
        return _shape_cols(law, len(spots))(*args)
    return _shape_cols_compiled(law, len(spots))(*args)


def _shape_cols(law, K):
    S = law.generator_matrix()

    def cols(X, Y, rp, *rest):
        spots = [tuple(rest[4 * k:4 * k + 4]) for k in range(K)]
        occ = _region_cols(X, Y, rp, spots, list(rest[4 * K:]), law, X.dtype)
        gen = [-o for o in occ]
        return mx.stack([gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)], axis=-1)
    return cols


# On the CPU the columns' graph is compiled once per law and spot count, as
# rings.py's (only this piece: the orbit and the spots' rotation feeding it
# stay eager). Not on the GPU, where MLX's fused kernels for a graph this
# size abort.
_COMPILED = {}


def _shape_cols_compiled(law, K):
    key = (law.definition, K)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _COMPILED[key] = mx.compile(_shape_cols(law, K))
    return fn


@fp64_on_cpu(any_arg=True)
def flux_dev_spots(X, Y, rp, n, alpha, contrast, w, law) -> mx.array:
    """F - 1 of a spotted star crossed by a planet, normalised to the
    unspotted star, for a hybrid law with shape weights w; arguments as
    ``shape_cols_spots``."""
    B = shape_cols_spots(X, Y, rp, n, alpha, contrast, law)
    return combine_cols([B[..., k] for k in range(B.shape[-1])], w, law)
