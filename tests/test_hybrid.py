"""Hybrid limb-darkening laws (elome/hybrid.py): closed forms,
gradients, priors.

Ground truth is a 30-digit mpmath direct integration

    F - 1 = -[int I(x) alpha(x, z, r) x dx] / [2 pi int I(x) x dx]

(alpha the angle of the disc annulus at radius x inside the planet), which
shares no code with the closed forms.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import elome
from elome import hybrid as H
from elome import ld
from elome.hybrid import (HYBRID2, HYBRID4, HYBRID5, LAWS, HybridLaw, flux_dev_hybrid,
                                hybrid_norms, ladder, lens_geometry, pole_col,
                                shape_cols, shape_partials, w_to_c)
from elome.poly import sn_dev_poly
from elome.vjp import ld_basis_analytic

mp = pytest.importorskip("mpmath")
mp.mp.dps = 30

from hybrid_weights import phys_w as _phys_w, vertex_weight  # noqa: E402

NAMES = ["hybrid2", "hybrid4", "hybrid5"]
RS = [0.01, 0.1, 0.3, 0.8]


def f64(x):
    return mx.array(np.asarray(x, dtype=np.float64), dtype=mx.float64)


def intensity_np(law, w, mu):
    c, d = w_to_c(law, w)
    m2 = np.asarray(mu) ** 2
    out = c[0] + c[1] * m2 + c[2] * m2 ** 2
    for dk, ek in zip(d, law.eps):
        out = out + dk / (m2 + ek) ** 2
    return out


def law_shapes(law, mu):
    """The law's shapes on a mu grid, built from first principles: the
    terms T (even shapes, then Pi_eps per pole), mixed by ``law.mix``."""
    m2 = np.asarray(mu, dtype=np.float64) ** 2
    T = [a[0] + a[1] * m2 + a[2] * m2 ** 2 for a in law.shapes]
    for e in law.eps:
        N = e ** -2 - (1 + e) ** -2
        T.append(((m2 + e) ** -2 - (1 + e) ** -2) / N)
    if law.mix is None:
        return T
    return [sum(c * t for c, t in zip(row, T)) for row in law.mix]


def phys_w(law, seed):
    """A weight vector inside the law's prior region (hybrid2's triangle,
    hybrid4's curvature simplex, the simplex otherwise; hybrid_weights)."""
    return _phys_w(law, np.random.default_rng(seed))


def z_grid(r, law=None):
    """Every regime and switch: contacts, z ~ r, the Q = 0 lines."""
    pts = [0.0, 0.5 * r, r, 0.5 * (1.0 - r), 1.0 - r - 1e-7, 1.0 - r,
           1.0 - r + 1e-7, 1.0, 1.0 + r - 1e-7, 1.0 + r, 1.0 + r + 0.1, 1.5]
    pts += list(np.linspace(0.02, 1.0 + r - 0.02, 7))
    if law is not None:
        for e in law.eps:
            zq = math.sqrt(1.0 + e) - r
            pts += [zq + d for d in (-1e-4, -1e-7, 0.0, 1e-7, 1e-4)]
    return np.array(sorted({float(p) for p in pts if p >= 0.0}))


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------

def _alpha(x, z, r):
    if x + z <= r:
        return 2 * mp.pi
    if x >= z + r or x + r <= z:
        return mp.mpf(0)
    c = (x * x + z * z - r * r) / (2 * x * z)
    return 2 * mp.acos(max(mp.mpf(-1), min(mp.mpf(1), c)))


def _occulted(I, z, r):
    """int I(x) alpha x dx over the disc, split at every kink."""
    z, r = mp.mpf(z), mp.mpf(r)
    # kinks of alpha: |z - r| (the annulus that first clips the planet's
    # edge, on either side of z = r) and z + r
    pts = sorted({mp.mpf(0), abs(z - r), min(z + r, mp.mpf(1)), mp.mpf(1)})
    if z >= 1 + r:
        return mp.mpf(0)
    return mp.quad(lambda x: I(x) * _alpha(x, z, r) * x, pts)


def oracle_flux_dev(law, w, z, r):
    c, d = w_to_c(law, w)

    def I(x):
        m2 = 1 - x * x
        out = mp.mpf(c[0]) + mp.mpf(c[1]) * m2 + mp.mpf(c[2]) * m2 ** 2
        for dk, ek in zip(d, law.eps):
            out += mp.mpf(dk) / (m2 + mp.mpf(ek)) ** 2
        return out
    total = 2 * mp.pi * mp.quad(lambda x: I(x) * x, [0, 1])
    return float(-_occulted(I, z, r) / total)


def oracle_column(I, z, r):
    """Deviation column: minus the occulted flux of intensity I."""
    return float(-_occulted(I, z, r))


# ---------------------------------------------------------------------------
# constants and law definitions
# ---------------------------------------------------------------------------

class TestDefinitions:
    def test_ladders(self):
        for K in (1, 2, 3, 4):                 # the formula itself
            for alpha in (1.0, 25.0 / 16.0, 2.0):
                ref = [math.exp(2 * math.pi / math.sqrt(alpha)
                                * (math.sqrt(k + 0.5) - math.sqrt(K))) for k in range(K)]
                assert np.allclose(ladder(K, alpha), ref, rtol=1e-15, atol=0)
                assert np.allclose(np.log(ladder(K, alpha)),
                                   np.log(ladder(K)) / math.sqrt(alpha), rtol=1e-14)
        assert HYBRID2.eps == (0.36,)
        assert H.HYBRID4_ALPHA == 25.0 / 16.0
        assert HYBRID4.eps == ladder(2, 25.0 / 16.0) == H.HYBRID4_EPS
        assert np.allclose(HYBRID4.eps, (0.02860121, 0.38582418), rtol=1e-6)
        assert HYBRID5.eps == ladder(3)
        assert np.allclose(HYBRID5.eps, (0.00159655, 0.04127504, 0.38743472), rtol=1e-6)
        assert (HYBRID2.n_w, HYBRID4.n_w, HYBRID5.n_w) == (2, 4, 5)

    @pytest.mark.parametrize("name", NAMES)
    def test_shape_form_equals_expanded_form(self, name):
        """I/I(1) = 1 - sum w_j S_j(mu), each shape S_j from 1 at the limb
        to 0 at the centre; and the expanded c-form agrees on a mu grid."""
        law = LAWS[name]
        w = phys_w(law, 2)
        mu = np.linspace(0.0, 1.0, 101)
        T = law_shapes(law, mu)
        for t in T:
            assert abs(t[0] - 1.0) < 1e-12 and abs(t[-1]) < 1e-12
        shape = 1.0 - sum(wj * tj for wj, tj in zip(w, T))
        assert np.abs(intensity_np(law, w, mu) - shape).max() < 1e-12
        assert abs(intensity_np(law, w, 0.0) - (1.0 - w.sum())) < 1e-12

    @pytest.mark.parametrize("name", NAMES)
    def test_norms_are_full_disc_fluxes(self, name):
        law = LAWS[name]
        N = hybrid_norms(law)
        assert abs(N[0] - math.pi) < 1e-15
        for j in range(law.n_w):
            Tj = lambda x, j=j: float(law_shapes(law, float(x))[j])
            full = float(2 * mp.pi * mp.quad(lambda x: Tj(x) * x, [0, 1]))
            assert abs(N[j + 1] - full) < 1e-13, j

    def test_unknown_law_and_wrong_count(self):
        with pytest.raises(ValueError, match="unknown hybrid law"):
            H.get_law("hybrid3")
        with pytest.raises(ValueError, match="takes 5 weights"):
            w_to_c(HYBRID5, [0.1, 0.2])
        z = f64([0.5])
        with mx.stream(mx.cpu):
            with pytest.raises(ValueError, match="takes 4 weights"):
                flux_dev_hybrid(z, f64(0.1), [0.1, 0.2], "hybrid4")
            with pytest.raises(ValueError, match="takes 2 weights"):
                flux_dev_hybrid(z, f64(0.1), f64([0.1, 0.2, 0.3]), "hybrid2")


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------

class TestColumns:
    @pytest.mark.parametrize("r", RS)
    def test_even_columns_match_the_elliptic_recursion(self, r):
        """E0, E1, E2 from the even recursion equal the ALFM19 s_n route
        (which runs cel3) recombined; E0, E1 are ld_basis's B_0, B_2."""
        z = f64(z_grid(r))
        with mx.stream(mx.cpu):
            E0, E1, E2 = H.even_cols(z, f64(r))
            s = sn_dev_poly(z, f64(r), 4)
            B = ld_basis_analytic(z, f64(r))
            mx.eval(E0, E1, E2, *s, B)
        s0d, s1d, s2d, s3, s4 = [np.asarray(x) for x in s]
        assert np.abs(np.asarray(E0) - s0d).max() < 1e-15
        assert np.abs(np.asarray(E1) - (0.5 * s0d + 0.25 * s2d)).max() < 1e-15
        assert np.abs(np.asarray(E2) - (s0d / 3 + s2d / 6 + s4 / 6)).max() < 1e-14
        assert np.array_equal(np.asarray(E0), np.asarray(B)[:, 0])
        assert np.array_equal(np.asarray(E1), np.asarray(B)[:, 2])

    @pytest.mark.parametrize("r", RS)
    @pytest.mark.parametrize("eps", sorted({e for law in LAWS.values()
                                            for e in law.eps}))
    def test_pole_column_vs_oracle(self, r, eps):
        law = H.HybridLaw("one", (eps,), ())
        zs = z_grid(r, law)
        with mx.stream(mx.cpu):
            got = np.asarray(pole_col(f64(zs), f64(r), eps))
        I = lambda x, e=mp.mpf(eps): 1 / (1 - x * x + e) ** 2
        ref = np.array([oracle_column(I, z, r) for z in zs])
        scale = math.pi / (eps * (1 + eps))               # the column's norm
        err = np.abs(got - ref) / scale
        # within 1e-6 of a contact the lens depth d is itself resolved to
        # ~1e-16/d, and kite (the core's own geometry) carries that: 2e-13
        # measured at d = 1e-7 past the internal contact; 1e-13 elsewhere
        near = (np.abs(zs - (1 - r)) < 1e-6) | (np.abs(zs - (1 + r)) < 1e-6)
        assert err[~near].max() < 1e-13
        assert err[near].max() < 1e-12
        assert (got[zs >= 1 + r] == 0.0).all()
        # z = 0, planet fully inside: pi r^2 / (p (p - r^2))
        if r < 1:
            p = 1 + eps
            assert abs(got[zs == 0.0][0] + math.pi * r * r / (p * (p - r * r))) < 1e-14 * scale

    @pytest.mark.parametrize("name", NAMES)
    def test_shape_columns_contract(self, name):
        """flux_dev == (B @ c)/(N @ c) with c = (1, -w): the ld_basis
        contract, and the columns vanish out of transit."""
        law = LAWS[name]
        r = 0.1
        zs = z_grid(r, law)
        w = phys_w(law, 3)
        with mx.stream(mx.cpu):
            B = np.asarray(shape_cols(f64(zs), f64(r), law))
            F = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
        assert B.shape == (zs.size, law.n_col)
        c = np.concatenate([[1.0], -w])
        N = hybrid_norms(law)
        assert np.abs(B @ c / (N @ c) - F).max() < 1e-15
        assert (B[zs >= 1 + r] == 0.0).all()


# ---------------------------------------------------------------------------
# the laws against the oracle
# ---------------------------------------------------------------------------

    @pytest.mark.parametrize("r", [0.5, 0.8, 0.95])
    def test_pole_column_just_past_the_inner_contact(self, r):
        """Just outside z = 1 - r the vanishing factor is -Bp = 1 - (z + r)^2,
        whose product form carries the rounding of 1 - z - r; 0.12.x mixed
        it with the exact kite and kap1 / (p eps) amplified the mismatch:
        1.6e-7 absolute (8e-11 of the column norm) at r = 0.8, eps = 0.0016,
        1e-13 from the contact, 1.5e-4 of the norm in the fp32 kernel.
        Every pole, 1e-13 .. 1e-6 from the contact, against the oracle."""
        from elome.metal import metal_available
        from elome.metal_hybrid import flux_dev_metal_hybrid
        eps_all = sorted({e for law in LAWS.values() for e in law.eps})
        zs = [1.0 - r + d for d in (1e-13, 1e-11, 1e-9, 1e-6)]
        for e in eps_all:
            def I(x, e=e):
                return 1 / (1 - x * x + mp.mpf(e)) ** 2
            norm = math.pi / (e * (1 + e))
            with mx.stream(mx.cpu):
                got = np.asarray(H.pole_col(f64(zs), f64(r), e))
            for z, g in zip(zs, got):
                assert abs(g - oracle_column(I, z, r)) / norm < 1e-14, (e, z)
        if metal_available():
            z32 = (1.0 - r) + np.logspace(-7.5, -2, 40)
            z32 = z32.astype(np.float32)
            with mx.stream(mx.gpu):
                k = np.asarray(flux_dev_metal_hybrid(mx.array(z32), r, "hybrid5",
                                                     None, basis=True), np.float64)
            with mx.stream(mx.cpu):
                ref = np.asarray(H.shape_cols(f64(z32.astype(np.float64)),
                                              float(np.float32(r)), "hybrid5"))
            assert (np.abs(k - ref) / hybrid_norms("hybrid5")).max() < 2e-6


class TestFlux:
    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("r", [0.1, 0.3])
    def test_vs_direct_integration(self, name, r):
        law = LAWS[name]
        zs = z_grid(r, law)
        for seed in (4, 5):
            w = phys_w(law, seed)
            with mx.stream(mx.cpu):
                got = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
            ref = np.array([oracle_flux_dev(law, w, z, r) for z in zs])
            assert np.abs(got - ref).max() < 1e-13, (name, r, seed)

    @pytest.mark.parametrize("name", NAMES)
    def test_uniform_disc_at_zero_weights(self, name):
        law = LAWS[name]
        zs = z_grid(0.2)
        with mx.stream(mx.cpu):
            got = np.asarray(flux_dev_hybrid(f64(zs), f64(0.2),
                                             np.zeros(law.n_w), law))
            E0 = np.asarray(H.even_cols(f64(zs), f64(0.2))[0])
        assert np.array_equal(got, E0 / math.pi)

    @pytest.mark.parametrize("name", NAMES)
    def test_host_traced_and_batched_weights_agree(self, name):
        law = LAWS[name]
        r = 0.15
        zs = z_grid(r, law)
        W = np.stack([phys_w(law, s) for s in (9, 10, 11)])
        with mx.stream(mx.cpu):
            host = [np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
                    for w in W]
            traced = [np.asarray(flux_dev_hybrid(f64(zs), f64(r), f64(w), law))
                      for w in W]
            z2 = f64(np.broadcast_to(zs, (3, zs.size)))
            batch = np.asarray(flux_dev_hybrid(z2, f64(r), f64(W), law))
        for j in range(3):
            assert np.abs(host[j] - traced[j]).max() < 1e-15
            assert np.abs(host[j] - batch[j]).max() < 1e-15

    @pytest.mark.parametrize("name", NAMES)
    def test_fp32_vs_fp64(self, name):
        """At the simplex vertices (each weight alone, and all zero) the
        fp32 graph stays within 1e-6 of fp64 everywhere -- the measured
        worst is hybrid5's innermost pole at 0.33 ppm near the contacts."""
        law = LAWS[name]
        worst = 0.0
        for r in (0.05, 0.1, 0.2):
            zs = np.concatenate([np.linspace(0.0, 1 + r, 4001),
                                 z_grid(r, law)])
            for j in range(law.n_w + 1):
                w = np.zeros(law.n_w)
                if j:
                    w[j - 1] = vertex_weight(law)
                with mx.stream(mx.cpu):
                    a = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
                    b = np.asarray(flux_dev_hybrid(
                        mx.array(zs, dtype=mx.float32),
                        mx.array(r, dtype=mx.float32), w, law),
                        dtype=np.float64)
                assert np.isfinite(b).all()
                worst = max(worst, np.abs(a - b).max())
        assert worst < 1e-6, worst


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _switch_free(zs, r, law, h):
    """Points whose FD stencil of half-width h does not straddle a regime
    switch (contacts, z = r, the Q = 0 lines), and not z == 1 exactly:
    there the Kahan sort in _kite_sqarea ties (max(z, 1), min(z, 1)) and
    MLX's autodiff splits the gradient at the tie -- a measure-zero
    artefact of autodiff, not of the function (the analytic partials and
    both one-sided slopes agree there to 1e-8)."""
    edges = [1 - r, 1 + r, r]
    for e in law.eps:
        edges.append(math.sqrt(1 + e) - r)
    edges = np.array(edges)
    far = np.min(np.abs(zs[:, None] - edges[None, :]), axis=1) > 3 * h
    return np.logical_and(far, zs != 1.0)


class TestGradients:
    @pytest.mark.parametrize("name", NAMES)
    def test_autodiff_vs_finite_differences(self, name):
        law = LAWS[name]
        r = 0.12
        zs = z_grid(r, law)
        w = phys_w(law, 12)
        h = 1e-6
        # the cotangent is zeroed on the tie / switch points (see
        # _switch_free): perturbing r moves the switches, so the FD sums
        # for r and w would straddle them there just as the z stencil does
        ct = np.random.default_rng(0).normal(size=zs.shape)
        ct = ct * _switch_free(zs, r, law, h)
        with mx.stream(mx.cpu):
            c = f64(ct)

            def loss(z, rr, ww):
                return mx.sum(c * flux_dev_hybrid(z, rr, ww, law))

            gz, gr, gw = mx.grad(loss, argnums=(0, 1, 2))(f64(zs), f64(r), f64(w))
            mx.eval(gz, gr, gw)
            gz, gr, gw = np.asarray(gz), float(gr), np.asarray(gw)

            def val(z, rr, ww):
                return np.asarray(flux_dev_hybrid(f64(z), f64(rr), ww, law))

            fz = ct * (val(zs + h, r, w) - val(zs - h, r, w)) / (2 * h)
            fr = float(np.sum(ct * (val(zs, r + h, w) - val(zs, r - h, w)) / (2 * h)))
            # F is a ratio of linear functions of w: a wider step costs only
            # O(h^2) (1e-9 at 1e-4), while 1e-6 leaves rounding noise of
            # 3e-7 on hybrid5's innermost-pole weight, whose gradient is
            # ~1e-5 (autodiff equals the closed form to 3e-15 there)
            hw = 1e-4
            fw = []
            for j in range(law.n_w):
                wp, wm = w.copy(), w.copy()
                wp[j] += hw
                wm[j] -= hw
                fw.append(float(np.sum(ct * (val(zs, r, wp) - val(zs, r, wm)) / (2 * hw))))
        ok = _switch_free(zs, r, law, h)
        assert np.abs(gz - fz)[ok].max() / np.abs(fz).max() < 1e-7
        assert abs(gr - fr) / abs(fr) < 1e-6
        for j in range(law.n_w):
            assert abs(gw[j] - fw[j]) / max(abs(fw[j]), 1e-6) < 1e-7, j

    @pytest.mark.parametrize("name", NAMES)
    def test_analytic_partials_match_autodiff(self, name):
        """shape_partials is the reference for the kernel VJP: it must equal
        autodiff of shape_cols everywhere, the regime switches included.
        Everywhere except the two max/min ties z == r and z == 1 of the
        Kahan sort in _kite_sqarea, where MLX's autodiff splits the
        gradient at the tie (measure zero; the analytic partials there
        are the ones that match finite differences). 2e-11: hybrid4's curvature
        shapes mix the generators with coefficients up to 1.34 (measured
        1.1e-11; hybrid2, hybrid5 under 8e-12)."""
        law = LAWS[name]
        for r in (0.05, 0.3):
            zs = z_grid(r, law)
            zs = zs[(zs != r) & (zs != 1.0)]
            with mx.stream(mx.cpu):
                dz, dr = shape_partials(f64(zs), f64(r), law)
                mx.eval(dz, dr)
                for j in range(law.n_col):
                    ct = np.zeros((zs.size, law.n_col))
                    ct[:, j] = 1.0
                    c = f64(ct)
                    gz, gr_tot = mx.grad(
                        lambda z, rr: mx.sum(c * shape_cols(z, rr, law)),
                        argnums=(0, 1))(f64(zs), f64(r))
                    mx.eval(gz, gr_tot)
                    s = np.abs(np.asarray(dz)[:, j]).max() + 1e-12
                    assert np.abs(np.asarray(gz) - np.asarray(dz)[:, j]).max() / s < 2e-11, j
                    s = np.abs(np.asarray(dr)[:, j]).sum() + 1e-12
                    assert abs(float(gr_tot) - float(np.asarray(dr)[:, j].sum())) / s < 2e-11, j

    @pytest.mark.parametrize("name", NAMES)
    def test_partials_continuous_across_switches(self, name):
        """The Q = 0 lines are removable singularities of the pole forms:
        one-sided values of every partial agree to rounding. At the
        internal contact z = 1 - r the partials are continuous but with
        the square-root cusp every occulted-flux derivative has there
        (d occ/dz ~ kite ~ sqrt(depth)), so the one-sided values at a
        distance d agree only to O(sqrt d)."""
        law = LAWS[name]
        r = 0.2
        with mx.stream(mx.cpu):
            for e in law.eps:
                z0 = math.sqrt(1 + e) - r
                # the partials curve (B'' ~ 10 for the innermost pole), so
                # the one-sided gap at +-delta is ~2 delta B'' even when
                # continuous; extrapolate the gap to zero separation
                d1, _ = shape_partials(f64([z0 - 1e-9, z0 + 1e-9]), f64(r), law)
                d2, _ = shape_partials(f64([z0 - 2e-9, z0 + 2e-9]), f64(r), law)
                d1, d2 = np.asarray(d1), np.asarray(d2)
                gap1 = np.abs(d1[1] - d1[0])
                gap2 = np.abs(d2[1] - d2[0])
                jump = np.abs(2.0 * gap1 - gap2)
                scale = np.abs(d1).max(axis=0) + 1e-9
                assert (jump / scale < 1e-7).all(), (e, jump / scale)
            for d in (1e-6, 1e-8):
                z0 = 1 - r
                dz, dr = shape_partials(f64([z0 - d, z0 + d]), f64(r), law)
                dz, dr = np.asarray(dz), np.asarray(dr)
                gap = max(np.abs(dz[0] - dz[1]).max(), np.abs(dr[0] - dr[1]).max())
                assert gap < 5.0 * math.sqrt(d), (d, gap)

    @pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                             ids=["fp64", "fp32"])
    @pytest.mark.parametrize("name", NAMES)
    def test_no_nan_gradients_on_boundary_sweep(self, dtype, name):
        law = LAWS[name]
        rng = np.random.default_rng(3)
        r = 0.17
        base = np.array([0.0, r, 1 - r, 1.0, 1 + r]
                        + [math.sqrt(1 + e) - r for e in law.eps])
        n = 5000
        off = np.where(rng.random(n) < 0.3, 0.0,
                       10.0 ** rng.uniform(-9, -2, n) * rng.choice([-1, 1], n))
        zs = np.clip(rng.choice(base, n) + off, 0.0, None)
        w = phys_w(law, 13)
        with mx.stream(mx.cpu):
            c = mx.array(np.ones(n), dtype=dtype)

            def loss(z, rr, ww):
                return mx.sum(c * flux_dev_hybrid(z, rr, ww, law))

            g = mx.grad(loss, argnums=(0, 1, 2))(
                mx.array(zs, dtype=dtype), mx.array(r, dtype=dtype),
                mx.array(w, dtype=dtype))
            mx.eval(g)
        for x in g:
            assert np.isfinite(np.asarray(x)).all()


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------

class TestPriors:
    def test_hybrid2_vertices(self):
        Vc, Vl = ld.hybrid2_vertices()
        assert np.allclose(Vc, [-0.1246, 1.1246], atol=2e-4)
        assert np.allclose(Vl, [1.2010, -0.2010], atol=2e-4)
        assert all(np.array_equal(a, b) for a, b in zip(
            ld.hybrid2_vertices(law="hybrid2"), (Vc, Vl)))

    @pytest.mark.parametrize("which", ["hybrid2", "custom-pole"])
    def test_hybrid2_triangle(self, which):
        """Uniform samples of the exact triangle satisfy its (A)(B)(C) and
        give a non-negative, centre-brightening profile; the map inverts;
        MLX agrees with numpy -- for hybrid2, and for a custom hybrid2-type
        law whose pole reaches the prior only through ``law=``."""
        law = (HYBRID2 if which == "hybrid2" else
               HybridLaw("custom2", (0.15,), HYBRID2.shapes))
        kw = {} if which == "hybrid2" else {"law": law}
        e = law.eps[0]
        N = e ** -2 - (1 + e) ** -2
        g0, g1 = 2 / (N * e ** 3), 2 / (N * (1 + e) ** 3)
        rng = np.random.default_rng(0)
        q1, q2 = rng.random(5000), rng.random(5000)
        w1, w2 = ld.hybrid2_from_q_np(q1, q2, **kw)
        assert (w1 + w2 <= 1 + 1e-12).all()
        assert (w1 + g0 * w2 >= -1e-12).all()
        assert (w1 + g1 * w2 >= -1e-12).all()
        mu = np.linspace(0, 1, 201)
        for k in range(0, 5000, 250):
            I = intensity_np(law, [w1[k], w2[k]], mu)
            assert (I >= -1e-12).all() and (np.diff(I) >= -1e-12).all()
        b1, b2 = ld.hybrid2_to_q_np(w1, w2, **kw)
        assert np.allclose(b1, q1, atol=1e-12) and np.allclose(b2, q2, atol=1e-12)
        with mx.stream(mx.cpu):
            m1, m2 = ld.hybrid2_from_q(f64(q1[:50]), f64(q2[:50]), **kw)
        assert np.allclose(np.asarray(m1), w1[:50], atol=1e-15)
        assert np.allclose(np.asarray(m2), w2[:50], atol=1e-15)

    def test_vertices_are_cached_but_returned_fresh(self):
        a = ld.hybrid2_vertices()
        a[0][0] = 99.0                        # a caller mutating its copy
        assert ld.hybrid2_vertices()[0][0] != 99.0

    def test_prior_pole_arguments(self):
        """law= ties the triangle to a law; it refuses a non-hybrid2-type
        law and a conflicting eps=."""
        custom = HybridLaw("custom2", (0.15,), HYBRID2.shapes)
        assert all(np.array_equal(a, b) for a, b in zip(
            ld.hybrid2_vertices(law=custom), ld.hybrid2_vertices(eps=0.15)))
        with pytest.raises(ValueError, match="not a hybrid2-type law"):
            ld.hybrid2_vertices(law="hybrid4")
        with pytest.raises(ValueError, match="not both"):
            ld.hybrid2_from_q_np(0.5, 0.5, eps=0.2, law=custom)

    @pytest.mark.parametrize("n", [3, 5])
    def test_simplex(self, n):
        rng = np.random.default_rng(2)
        q = rng.random((20000, n))
        w = ld.simplex_from_q_np(q)
        assert (w >= -1e-12).all() and (w.sum(1) <= 1 + 1e-12).all()
        # flat Dirichlet(1,...,1) on n+1 parts: each mean 1/(n+1)
        assert np.allclose(w.mean(0), 1 / (n + 1), atol=0.01)
        assert np.allclose(ld.q_from_simplex_np(w), q, atol=1e-10)
        with mx.stream(mx.cpu):
            wm = np.asarray(ld.simplex_from_q(f64(q[:50])))
        assert np.allclose(wm, w[:50], atol=1e-15)

    def test_simplex_profiles_are_physical(self):
        """simplex_from_q is hybrid5's prior: every sample is a
        non-negative profile that darkens towards the limb."""
        w = ld.simplex_from_q_np(np.random.default_rng(2).random((20, 5)))
        mu = np.linspace(0, 1, 201)
        for wk in w:
            I = intensity_np(HYBRID5, wk, mu)
            assert (I >= -1e-12).all() and (np.diff(I) >= -1e-12).all()


class TestPriorInversesAtCorners:
    """The inverse maps divided by zero at valid corners (a uniform disc, a
    simplex vertex), so a sampler started there began at NaN."""

    def test_uniform_disc(self):
        from elome import hybrid2_to_q_np, u_to_q_np
        q1, q2 = u_to_q_np(0.0, 0.0)
        assert (q1, q2) == (0.0, 0.5)
        q1, q2 = hybrid2_to_q_np(0.0, 0.0)
        assert q1 == 0.0 and q2 == 0.5
        q1, q2 = u_to_q_np(np.array([0.0, 0.4]), np.array([0.0, 0.25]))
        assert np.isfinite(q2).all() and q2[0] == 0.5

    @pytest.mark.parametrize("w", [[1, 0, 0, 0], [0.5, 0.5, 0, 0],
                                   [0, 0, 0, 1], [0, 0, 0, 0, 0]])
    def test_simplex_vertices_round_trip(self, w):
        from elome import q_from_simplex_np, simplex_from_q_np
        q = q_from_simplex_np(w)
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        # the numpy map does not clamp (only the MLX one, for a gradient):
        # a vertex is reached exactly
        np.testing.assert_allclose(simplex_from_q_np(q), w, atol=1e-15)

    def test_interior_unchanged(self):
        from elome import q_from_simplex_np, simplex_from_q_np, u_to_q_np
        w = np.array([0.2, 0.2, 0.1, 0.1])
        np.testing.assert_allclose(simplex_from_q_np(q_from_simplex_np(w)), w,
                                   atol=1e-15)
        q1, q2 = u_to_q_np(0.4, 0.25)
        assert q1 == pytest.approx(0.4225) and q2 == pytest.approx(0.4 / 1.3)


class TestHybrid4CurvatureBasis:
    """hybrid4's weights refer to its curvature basis S = B T, in which the
    simplex w >= 0, sum w <= 1 is exactly the set of profiles in its span
    with I >= 0, g >= 0, g' <= 0, g'' >= 0 on x = mu^2 in [0, 1]."""

    B_REF = np.array([
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0006260842489149928, 0.001404013755411423, 1.020963273947444,
         -0.02299337195177007],
        [-0.08116019831892918, -0.2568535772254558, -0.0005906631447752816,
         1.33860443868916]])

    @staticmethod
    def conditions(w, x):
        """(I(0), min g, -max g', min g'') over x for weights w (..., 4):
        all >= 0 iff the profile is positive, monotonic, concave in mu^2
        and of convex slope in mu^2."""
        S = HYBRID4.generator_matrix()          # rows over [1, x, x^2, P0, P1]
        coef = -(np.atleast_2d(w) @ S)
        coef[:, 0] += 1.0
        c0, c1, c2 = coef[:, 0:1], coef[:, 1:2], coef[:, 2:3]
        d = coef[:, 3:]
        e = np.array(HYBRID4.eps)
        xe = x[None, :, None] + e
        I0 = (c0[:, 0] + (d / e ** 2).sum(-1))
        g = c1 + 2 * c2 * x - 2 * (d[:, None, :] * xe ** -3).sum(-1)
        g1 = 2 * c2 + 6 * (d[:, None, :] * xe ** -4).sum(-1)
        g2 = -24 * (d[:, None, :] * xe ** -5).sum(-1)
        return I0, g.min(1), -g1.max(1), g2.min(1)

    def test_basis_and_check_values(self):
        assert np.abs(H.HYBRID4_BASIS - self.B_REF).max() < 1e-15
        assert np.array_equal(H.hybrid4_basis(H.HYBRID4_EPS), H.HYBRID4_BASIS)
        assert np.array_equal(np.array(HYBRID4.mix), H.HYBRID4_BASIS)
        w = ld.hybrid4_from_q_np([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(
            w, [0.1734764882, 0.1954827707, 0.2359685782, 0.2359685782], atol=1e-10)
        np.testing.assert_allclose(
            w @ H.HYBRID4_BASIS,
            [0.0677347237, 0.2219429445, 0.2407758742, 0.3104428728], atol=1e-10)

    def test_vertices(self):
        """w = e_k gives I = 1 - S_k: S1 = 1 - mu^2, S2 = (1 - mu^2)^2
        exactly; each vertex is physical; S3, S4 have I(0) = 0 and
        g(1) = g'(1) = 0."""
        mu = np.linspace(0, 1, 101)
        np.testing.assert_allclose(intensity_np(HYBRID4, [1, 0, 0, 0], mu), mu ** 2, atol=1e-15)
        np.testing.assert_allclose(intensity_np(HYBRID4, [0, 1, 0, 0], mu),
                                   1 - (1 - mu ** 2) ** 2, atol=1e-15)
        x = np.linspace(0, 1, 2001)
        for k in range(4):
            w = np.eye(4)[k]
            assert all((c >= -1e-12).all() for c in self.conditions(w, x)), k
            assert abs(intensity_np(HYBRID4, w, 0.0)) < 1e-12
        for k in (2, 3):
            S = HYBRID4.generator_matrix()[k]
            e = np.array(HYBRID4.eps)
            g1 = S[1] + 2 * S[2] - 2 * (S[3:] * (1 + e) ** -3).sum()
            gp1 = 2 * S[2] + 6 * (S[3:] * (1 + e) ** -4).sum()
            assert abs(g1) < 1e-12 and abs(gp1) < 1e-12

    def test_simplex_is_exactly_the_curvature_region(self):
        """Precision and recall: over a box around the simplex, a weight
        vector meets the four conditions if and only if it is in the
        simplex (points within 0.01 of a facet excluded)."""
        rng = np.random.default_rng(7)
        w = rng.uniform(-0.3, 1.1, (40000, 4))
        slack = np.minimum(w.min(1), 1 - w.sum(1))
        keep = np.abs(slack) > 0.01
        w, inside = w[keep], slack[keep] > 0
        x = np.linspace(0, 1, 2001)
        ok = np.ones(len(w), bool)
        for chunk in range(0, len(w), 5000):
            conds = self.conditions(w[chunk:chunk + 5000], x)
            ok[chunk:chunk + 5000] = np.all([c >= 0 for c in conds], axis=0)
        assert inside.sum() > 300 and (~inside).sum() > 300
        assert np.array_equal(ok, inside)

    def test_same_light_curves_as_the_term_basis(self):
        """hybrid4 with curvature weights w is the term-basis law with
        weights w @ B: the same profile, the same light curve."""
        term = HybridLaw("hybrid4_terms", HYBRID4.eps, HYBRID4.shapes)
        assert term.definition != HYBRID4.definition
        rng = np.random.default_rng(3)
        for r in (0.05, 0.2):
            zs = z_grid(r, HYBRID4)
            for _ in range(3):
                w = ld.hybrid4_from_q_np(rng.random(4))
                with mx.stream(mx.cpu):
                    a = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, HYBRID4))
                    b = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w @ H.HYBRID4_BASIS, term))
                assert np.abs(a - b).max() < 1e-15

    def test_prior_map(self):
        rng = np.random.default_rng(4)
        q = rng.random((20000, 4))
        w = ld.hybrid4_from_q_np(q)
        assert (w >= 0).all() and (w.sum(1) <= 1 + 1e-12).all()
        assert np.allclose(w.mean(0), 0.2, atol=0.01)       # flat Dirichlet on 5 parts
        assert np.abs(ld.hybrid4_to_q_np(w) - q).max() < 1e-10
        with mx.stream(mx.cpu):
            wm = np.asarray(ld.hybrid4_from_q(f64(q[:50])))
            corners = f64([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]])
            g = mx.grad(lambda qq: mx.sum(ld.hybrid4_from_q(qq) * f64([1, 2, 3, 4])))(corners)
        assert np.abs(wm - w[:50]).max() < 1e-15
        assert np.isfinite(np.asarray(g)).all()

    @pytest.mark.parametrize("w", [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0.5, 0.5]])
    def test_prior_inverse_at_corners(self, w):
        q = ld.hybrid4_to_q_np(w)
        assert np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all()
        # the numpy map does not clamp: tabulated q on a face give their w
        np.testing.assert_allclose(ld.hybrid4_from_q_np(q), w, atol=1e-15)

    def test_prior_inverse_refuses_weights_outside(self):
        """Weights outside the curvature simplex have no q: raise rather
        than return q outside [0, 1] that the forward map clamps into
        another profile. One such is 1 - 0.5 (1 - mu^4), physical but
        convex in mu^2."""
        w_convex = np.array([0.5, 0, 0, 0]) @ np.linalg.inv(H.HYBRID4_BASIS)
        mu = np.linspace(0, 1, 201)
        I = intensity_np(HYBRID4, w_convex, mu)
        np.testing.assert_allclose(I, 1 - 0.5 * (1 - mu ** 4), atol=1e-12)
        for w in (w_convex, [0.6, 0.6, 0, 0], [-0.1, 0.2, 0.2, 0.2]):
            with pytest.raises(ValueError, match="curvature simplex"):
                ld.hybrid4_to_q_np(w)
        q = ld.hybrid4_to_q_np([1e-17, -1e-17, 0.3, 0.3])   # rounding: accepted
        assert ((q >= 0) & (q <= 1)).all()

    def test_mix_validation(self):
        with pytest.raises(ValueError, match="mix must be 4 x 4"):
            HybridLaw("bad", HYBRID4.eps, HYBRID4.shapes, mix=np.eye(3))
        with pytest.raises(ValueError, match="invertible"):
            HybridLaw("bad", HYBRID4.eps, HYBRID4.shapes, mix=np.zeros((4, 4)))
        with pytest.raises(ValueError, match="sum to 1"):
            HybridLaw("bad", HYBRID4.eps, HYBRID4.shapes, mix=2 * np.eye(4))
        plain = HybridLaw("plain", HYBRID4.eps, HYBRID4.shapes, mix=np.eye(4))
        assert plain.mix is None
        assert plain.definition == HybridLaw("p2", HYBRID4.eps, HYBRID4.shapes).definition


@pytest.mark.parametrize("z0,r,law", [(1.6155494421403511, 0.5, "hybrid2"),
                                      (1.6848148266204213, 0.9, "hybrid4")])
def test_no_nan_gradient_where_the_complete_form_divides_by_zero(z0, r, law):
    """Off the complete-transit mask the pole column's K + sqrt(Q) is
    K + 1, which passes through 0 at z^2 = 2 + eps + r^2 (out of transit for
    hybrid2's pole at r = 0.5, inside the partial regime for hybrid4's inner
    pole at r = 0.9). The masked lane's VJP was 0 * inf = NaN on the ulp
    where it rounds to 0."""
    zs = [z0]
    for step in (np.inf, 0.0):
        x = z0
        for _ in range(200):
            x = np.nextafter(x, step)
            zs.append(x)
    w = [0.3, 0.2] if law == "hybrid2" else [0.2, 0.2, 0.1, 0.1]
    with mx.stream(mx.cpu):
        Z = mx.array(np.array(zs), dtype=mx.float64)
        g = mx.grad(lambda z: mx.sum(flux_dev_hybrid(z, r, w, law)))(Z)
        assert np.isfinite(np.asarray(g)).all()
