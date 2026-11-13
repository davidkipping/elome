"""batman-frontend parity: same params in, same light curve out (to
batman's own ~2e-8 quadratic accuracy floor), across orbital
configurations, limb-darkening laws, supersampling, and secondary
eclipses."""

import batman as batman_pkg
import mlx.core as mx
import numpy as np
import pytest

import elome
from elome.metal import _gpu_stream_active, metal_available


def _params(**kw):
    p = elome.TransitParams()
    b = batman_pkg.TransitParams()
    defaults = dict(t0=0.31, per=2.7, rp=0.11, a=9.5, inc=87.6,
                    ecc=0.0, w=90.0, u=[0.35, 0.22], limb_dark="quadratic")
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(p, k, v)
        setattr(b, k, v)
    return p, b


T = np.linspace(-0.25, 3.0, 4000) + 0.31  # covers two transits


class TestPrimaryParity:
    def test_circular_quadratic(self):
        p, b = _params()
        got = elome.TransitModel(p, T).light_curve(p)
        want = batman_pkg.TransitModel(b, T).light_curve(b)
        assert got.shape == want.shape
        assert np.max(np.abs(got - want)) < 3e-8
        assert got.min() < 0.99  # the transit is actually there

    def test_inclination_sweep(self):
        for inc in (90.0, 89.0, 87.0, 85.5, 84.9):
            p, b = _params(inc=inc, a=8.0)
            got = elome.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 5e-8, f"inc={inc}"

    def test_non_transiting_is_flat(self):
        p, _ = _params(inc=70.0, a=6.0)  # b = a cos i ~ 2
        got = elome.TransitModel(p, T).light_curve(p)
        np.testing.assert_array_equal(got, np.ones_like(got))

    def test_linear_and_uniform_laws(self):
        for law, u in (("linear", [0.4]), ("uniform", [])):
            p, b = _params(limb_dark=law, u=u)
            got = elome.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 3e-8, law

    def test_eccentric_orbits(self):
        for ecc, w in ((0.3, 45.0), (0.5, 130.0), (0.2, 271.0),
                       (0.65, 90.0)):
            p, b = _params(ecc=ecc, w=w, a=14.0, inc=88.2)
            got = elome.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 1e-7, (ecc, w)
            assert got.min() < 0.999  # transiting configuration

    def test_supersampling(self):
        p, b = _params()
        got = elome.TransitModel(
            p, T, supersample_factor=7, exp_time=0.02).light_curve(p)
        want = batman_pkg.TransitModel(
            b, T, supersample_factor=7, exp_time=0.02).light_curve(b)
        assert np.max(np.abs(got - want)) < 3e-8

    def test_param_update_between_calls(self):
        """batman workflow: build once, vary params per call."""
        p, b = _params()
        m = elome.TransitModel(p, T)
        mb = batman_pkg.TransitModel(b, T)
        for rp in (0.05, 0.08, 0.13):
            p.rp = rp
            b.rp = rp
            assert np.max(np.abs(m.light_curve(p) - mb.light_curve(b))) < 3e-8


class TestSecondary:
    def test_secondary_eclipse_circular(self):
        p, b = _params(u=[], limb_dark="uniform")
        p.fp = b.fp = 1.5e-3
        p.t_secondary = b.t_secondary = p.t0 + 0.5 * p.per
        got = elome.TransitModel(
            p, T, transittype="secondary").light_curve(p)
        want = batman_pkg.TransitModel(
            b, T, transittype="secondary").light_curve(b)
        # both: 1 + fp out of eclipse, dipping to ~1 in eclipse
        assert abs(got.max() - (1 + p.fp)) < 1e-9
        assert np.max(np.abs(got - want)) < 1e-8


class TestValidation:
    def test_unsupported_law_raises(self):
        p, _ = _params(limb_dark="nonlinear", u=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="not supported"):
            elome.TransitModel(p, T)

    def test_wrong_coefficient_count_raises(self):
        p, _ = _params(limb_dark="quadratic", u=[0.1])
        with pytest.raises(ValueError):
            elome.TransitModel(p, T)

    def test_supersample_needs_exp_time(self):
        p, _ = _params()
        with pytest.raises(ValueError, match="exp_time"):
            elome.TransitModel(p, T, supersample_factor=5)


class TestLightCurveMx:
    """light_curve_mx used to build its graph on the default device, so a
    default (fp64) TransitModel raised 'float64 is not supported on the
    GPU' unless the caller had already switched to the CPU stream."""

    @pytest.mark.parametrize("dtype", [None, mx.float32],
                             ids=["default_fp64", "fp32"])
    @pytest.mark.parametrize("kw", [{}, dict(exp_time=0.02,
                                             integration="contact"),
                                    dict(exp_time=0.02,
                                         supersample_factor=5)],
                             ids=["plain", "contact", "supersample"])
    @pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 63.0)],
                             ids=["circular", "eccentric"])
    def test_works_on_the_default_stream(self, dtype, kw, ecc, w):
        p, _ = _params(ecc=ecc, w=w)
        m = elome.TransitModel(p, T, dtype=dtype, **kw)
        out = m.light_curve_mx(p)            # no stream context here
        mx.eval(out)                         # nor here
        assert out.dtype == m.dtype
        got = np.asarray(out, dtype=np.float64)
        if kw.get("supersample_factor", 1) > 1:   # raw grid: average it
            got = got.reshape(T.size, -1).mean(axis=1)
        tol = 1e-14 if m.dtype == mx.float64 else 5e-7
        assert np.abs(got - m.light_curve(p)).max() < tol


class TestEccentricKernelRouting:
    """The fp32 GPU primary path (circular and eccentric) is served by the
    fused model kernel;
    every other combination keeps the graph. Parity is adjudicated
    against float64, since both fp32 paths sit at the same rounding."""

    @staticmethod
    def _model(ecc, dtype, kernel=True, n=20001, **kw):
        """kernel=False is use_metal=False: the graph model. (An
        earlier version patched _kernel_usable around *construction* only,
        but the check runs when the graph is first compiled, so the
        'graph' model ran the kernel and the parity test compared the
        kernel with itself.)"""
        p, _ = _params(ecc=ecc, w=63.0)
        t = np.linspace(-0.35, 0.35, n)
        if not kernel:
            kw["use_metal"] = False
        return elome.TransitModel(p, t, dtype=dtype, **kw), p

    @pytest.mark.skipif(not (metal_available() and _gpu_stream_active()),
                        reason="the kernel needs Metal and the GPU stream")
    @pytest.mark.parametrize("ecc", [1e-4, 0.3, 0.7, 0.9])
    def test_kernel_matches_graph_and_fp64(self, ecc):
        mk, p = self._model(ecc, mx.float32, kernel=True)
        mg, _ = self._model(ecc, mx.float32, kernel=False)
        a, b = mk.light_curve(p), mg.light_curve(p)
        assert list(mk._compiled) == [(False, True)]       # the kernel
        assert list(mg._compiled) == [(False, False)]      # the graph
        with mx.stream(mx.cpu):
            m64, _ = self._model(ecc, mx.float64)
            ref = m64.light_curve(p)
        assert np.abs(a - b).max() < 5e-6
        # the kernel must be no worse than the graph against fp64
        assert np.abs(a - ref).max() < 3.0 * max(np.abs(b - ref).max(), 5e-7)

    def test_period_update_is_not_baked_in(self):
        """period_ref is 0 and the period rides the traced p_off input,
        so a batman-style parameter update must change the curve."""
        mk, p = self._model(0.3, mx.float32, n=4001)
        f1 = mk.light_curve(p)
        p.per = 3.5
        f2 = mk.light_curve(p)
        assert not np.allclose(f1, f2)
        p.per = 3.456
        with mx.stream(mx.cpu):
            m64, _ = self._model(0.3, mx.float64, n=4001)
            m64.t = mk.t
            ref = m64.light_curve(p)
        assert np.abs(mk.light_curve(p) - ref).max() < 5e-6

    def test_fp64_and_secondary_do_not_use_the_kernel(self):
        """The kernel decision is recorded in the compiled-graph cache
        key, (circular, kernel): fp64 cannot run it, and a secondary
        eclipse is not the model kernel's -- on an fp32 GPU model it takes
        the eclipse kernel instead (_sph_tau_fn)."""
        m64, p64 = self._model(0.3, mx.float64, n=101)
        assert not m64._kernel_usable()
        m64.light_curve(p64)
        assert list(m64._compiled) == [(False, False)]
        p, _ = _params(ecc=0.3, w=63.0)
        p.fp = 0.002
        msec = elome.TransitModel(
            p, np.linspace(-0.35, 0.35, 101), transittype="secondary",
            dtype=mx.float32)
        msec.light_curve(p)
        assert list(msec._compiled) == ([("sph-tau", "kh")] if msec._kernel_usable()
                                        else [(False, False)])

    def test_use_metal_false_disables_it(self):
        p, _ = _params(ecc=0.3, w=63.0)
        m = elome.TransitModel(p, np.linspace(-0.35, 0.35, 101),
                                     dtype=mx.float32, use_metal=False)
        assert not m._kernel_usable()


# ---------------------------------------------------------------------------
# production review: regressions
# ---------------------------------------------------------------------------

def _mp(**kw):
    return _params(**kw)[0]

@pytest.mark.parametrize("bad", [dict(rp=-0.1), dict(rp=float("nan")),
                                 dict(a=0.5), dict(per=-3.0)])
def test_invalid_geometry_raises(bad):
    t = np.linspace(-0.1, 0.1, 51)
    m = elome.TransitModel(_mp(), t)
    with pytest.raises(ValueError, match="must be"):
        m.light_curve(_mp(**bad))
    p = _mp()
    for k, v in bad.items():
        setattr(p, k, np.array([0.11 if k == "rp" else getattr(p, k), v]))
    with pytest.raises(ValueError, match="must be"):
        m.light_curves(p)


def test_invalid_traced_geometry_is_nan():
    t = np.linspace(-0.1, 0.1, 51)
    with mx.stream(mx.cpu):
        m = elome.TransitModel(_mp(), t)
        comp = mx.compile(lambda r: m.light_curve_mx(_mp(rp=r)))
        assert np.isnan(np.asarray(comp(mx.array(-0.1, dtype=mx.float64)))).all()
        assert np.isfinite(np.asarray(comp(mx.array(0.11, dtype=mx.float64)))).all()


@pytest.mark.parametrize("u", [[0.3, 0.2, 0.5], [0.3]])
def test_array_valued_light_curves_check_the_coefficient_count(u):
    m = elome.TransitModel(_mp(), np.linspace(-0.1, 0.1, 51))
    p = _mp()
    p.rp = np.array([0.1, 0.11])
    p.u = u
    with pytest.raises(ValueError, match="takes 2 coefficients"):
        m.light_curves(p)


def test_use_metal_false_gradients_finite_at_r1_z0():
    """The non-Metal fallback is the analytic core: flux_dev's reverse mode
    is NaN at r = 1, z = 0 (a white-dwarf occultor at mid-eclipse)."""
    p = _mp(t0=0.0, rp=1.0, inc=90.0, ecc=0.0)
    t = np.linspace(-0.01, 0.01, 21)
    for use_metal in (True, False):
        with mx.stream(mx.cpu):
            m = elome.TransitModel(p, t, use_metal=use_metal)

            def loss(r):
                q = _mp(t0=0.0, rp=r, inc=90.0, ecc=0.0)
                return mx.sum(m.light_curve_mx(q))
            g = mx.grad(loss)(mx.array(1.0, dtype=mx.float64))
        assert np.isfinite(float(g)), use_metal


def test_failed_metal_probe_warns_and_can_reprobe(monkeypatch):
    from elome import metal as M
    def boom():
        raise RuntimeError("simulated compile failure")
    monkeypatch.setattr(M, "_get_kernels", boom)
    with pytest.warns(RuntimeWarning, match="simulated compile failure"):
        assert M.metal_available(reprobe=True) is False
    monkeypatch.undo()
    assert M.metal_available(reprobe=True) == metal_available()


# ---------------------------------------------------------------------------
# package review: regressions
# ---------------------------------------------------------------------------

_T = np.linspace(-0.1, 0.1, 51)
_needs_gpu = pytest.mark.skipif(not metal_available(),
                                reason="needs a usable Metal device")


@pytest.mark.parametrize("field", ["t0", "per", "rp", "a", "inc"])
def test_a_missing_field_raises_on_every_path(field):
    """light_curves turned a None field into 0.0 -- rp=None gave a flat
    curve, t0=None silently meant t0 = 0 -- where light_curve raised."""
    m = elome.TransitModel(_mp(), _T)
    bad = _mp(**{field: None})
    with pytest.raises(ValueError, match=f"params.{field} is not set"):
        m.light_curve(bad)
    with pytest.raises(ValueError, match=f"params.{field} is not set"):
        m.light_curves([_mp(), bad])
    arr = _mp(rp=np.array([0.10, 0.11]))
    setattr(arr, field, None)
    with pytest.raises(ValueError, match=f"params.{field} is not set"):
        m.light_curves(arr)


def test_secondary_eclipse_needs_fp_at_every_call():
    """fp was checked at construction only; a later fp=None gave a flat
    eclipse (fp = 0) without an error."""
    m = elome.TransitModel(_mp(fp=1e-3), 1.35 + _T, transittype="secondary")
    with pytest.raises(ValueError, match="params.fp is not set"):
        m.light_curve(_mp(fp=None))
    with pytest.raises(ValueError, match="params.fp is not set"):
        m.light_curves([_mp(fp=1e-3), _mp(fp=None)])
    # a primary transit does not read fp: None stays fine there
    assert np.isfinite(elome.TransitModel(_mp(), _T).light_curve(_mp())).all()


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
def test_secondary_eclipse_of_a_point_planet(dtype):
    """rp = 0 divided 0 by 0 everywhere; a point planet is never hidden,
    so the flux is 1 + fp throughout, with finite gradients."""
    t2 = 1.35 + _T
    m = elome.TransitModel(_mp(fp=1e-3), t2, transittype="secondary", dtype=dtype)
    f = m.light_curve(_mp(rp=0.0, fp=1e-3))
    np.testing.assert_allclose(f, 1.0 + 1e-3, rtol=0, atol=1e-7)
    np.testing.assert_allclose(m.light_curves([_mp(rp=0.0, fp=1e-3)])[0], f,
                               rtol=0, atol=1e-7)
    if dtype == mx.float64:
        with mx.stream(mx.cpu):
            def total(rp):
                q = _mp(fp=1e-3)
                q.rp = rp
                return mx.sum(m.light_curve_mx(q))
            g = mx.grad(total)(mx.array(0.0, dtype=mx.float64))
        assert np.isfinite(float(g))


def test_contradictory_exposure_options_raise():
    with pytest.raises(ValueError, match="supersample_factor is for"):
        elome.TransitModel(_mp(), _T, integration="contact", exp_time=0.02,
                           supersample_factor=10)
    for n_gl in (0, -3):
        with pytest.raises(ValueError, match="n_gl must be >= 1"):
            elome.TransitModel(_mp(), _T, integration="contact",
                               exp_time=0.02, n_gl=n_gl)


@_needs_gpu
@pytest.mark.parametrize("law,u", [("quadratic", [0.4, 0.25]),
                                   ("hybrid2", [0.3, 0.2])])
def test_use_metal_takes_effect_after_first_use(law, u):
    """The choice was baked into graphs at their first trace, so turning it
    off later changed nothing for most laws. Now every graph is rebuilt."""
    p = _mp(limb_dark=law, u=u)
    m = elome.TransitModel(p, _T, dtype=mx.float32)
    m.light_curve(p)
    m.use_metal = False
    off = elome.TransitModel(p, _T, dtype=mx.float32, use_metal=False)
    np.testing.assert_array_equal(m.light_curve(p), off.light_curve(p))
    m.use_metal = True
    on = elome.TransitModel(p, _T, dtype=mx.float32)
    np.testing.assert_array_equal(m.light_curve(p), on.light_curve(p))


@_needs_gpu
def test_batched_contact_rule_uses_the_kernel(monkeypatch):
    """light_curves' contact rule once handed the core (n, m, nodes)
    separations, which flux_dev_metal sent to the graph fallback (about 4x
    slower). It now runs the tau kernel, which never falls back either."""
    import elome.metal as M

    def no_graph(*a, **k):
        raise AssertionError("fell back to the MLX graph")
    p = _mp()
    m = elome.TransitModel(p, _T, dtype=mx.float32, exp_time=0.02,
                           integration="contact", n_gl=5)
    ref = np.array([m.light_curve(_mp(rp=r)) for r in (0.09, 0.11)])
    monkeypatch.setattr(M, "flux_dev_analytic", no_graph)
    batch = _mp(rp=np.array([0.09, 0.11]))
    np.testing.assert_allclose(m.light_curves(batch), ref, rtol=0, atol=5e-7)


_CONTACT = dict(exp_time=0.02, integration="contact", n_gl=7)
_T_WIDE = 0.31 + np.linspace(-0.12, 0.12, 801)        # _mp's t0


@_needs_gpu
@pytest.mark.parametrize("law,u", [("quadratic", [0.35, 0.22]), ("linear", [0.4]),
                                   ("uniform", [])])
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circular", "e0.3"])
def test_contact_rule_takes_the_tau_kernel(law, u, ecc):
    """The quadratic family under the contact rule runs the quadratic tau
    kernel on an fp32 GPU model (orbit, contacts, quadrature and
    photometry in one pass): within fp32 noise of use_metal=False's graph
    and closer than it to the fp64 model."""
    p = _mp(limb_dark=law, u=u, ecc=ecc, w=63.0)
    m = elome.TransitModel(p, _T_WIDE, dtype=mx.float32, **_CONTACT)
    got = m.light_curve(p)
    assert list(m._compiled) == [("sph-tau", "circ" if ecc == 0.0 else "kh")]
    graph = elome.TransitModel(p, _T_WIDE, dtype=mx.float32, use_metal=False,
                               **_CONTACT).light_curve(p)
    f64 = elome.TransitModel(p, _T_WIDE, **_CONTACT).light_curve(p)
    assert got.min() < 0.99
    assert np.abs(got - graph).max() < 1e-6
    assert np.abs(got - f64).max() < 2e-7


@_needs_gpu
def test_polynomial_contact_rule_keeps_its_graph():
    """The polynomial law has no kernel: its contact rule stays the graph."""
    p = _mp(limb_dark="polynomial", u=[0.3, 0.2, 0.1])
    m = elome.TransitModel(p, _T_WIDE, dtype=mx.float32, **_CONTACT)
    m.light_curve(p)
    assert list(m._compiled) == [(True, False)]


@_needs_gpu
def test_contact_rule_batches_match_the_loop():
    """light_curves on the tau kernel, one row per set, against the loop:
    bitwise for circular sets, within a few ulps for eccentric ones (the
    batch folds b at conjunction on the host in fp64, light_curve in the
    fp32 graph) and when one eccentric set takes the whole batch down the
    (k, h) orbit."""
    rng = np.random.default_rng(8)
    m = elome.TransitModel(_mp(), _T_WIDE, dtype=mx.float32, **_CONTACT)
    ulp = float(np.finfo(np.float32).eps)
    for eccs, exact in (([0.0] * 3, True), ([0.1, 0.3, 0.2], False),
                        ([0.0, 0.3, 0.0, 0.5], False)):
        sets = [_mp(ecc=e, w=rng.uniform(0, 360), rp=rng.uniform(0.08, 0.12),
                    inc=rng.uniform(87.0, 88.5), u=[rng.uniform(0.2, 0.5), 0.2])
                for e in eccs]
        batched = m.light_curves(sets)
        looped = np.stack([m.light_curve(q) for q in sets])
        if exact:
            assert np.array_equal(batched, looped)
        else:
            assert np.abs(batched - looped).max() <= 4 * ulp


@_needs_gpu
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circular", "e0.3"])
def test_contact_rule_gradients_through_the_tau_kernel(ecc):
    """light_curve_mx's gradients in t0, rp, a, inc, u1 and u2 under the
    contact rule, through the tau kernel's VJP, against the fp64 model's,
    condition-scaled."""
    ct = np.random.default_rng(9).normal(size=_T_WIDE.size)

    def grads(dtype):
        m = elome.TransitModel(_mp(ecc=ecc, w=63.0), _T_WIDE, dtype=dtype, **_CONTACT)
        with mx.stream(mx.gpu if dtype == mx.float32 else mx.cpu):
            c = mx.array(ct, dtype=dtype)

            def loss(t0, rp, a, inc, u1, u2):
                q = _mp(ecc=ecc, w=63.0, t0=t0, rp=rp, a=a, inc=inc, u=[u1, u2])
                return mx.sum(c * m.light_curve_mx(q))
            g = mx.grad(loss, argnums=tuple(range(6)))(
                *[mx.array(v, dtype=dtype) for v in (0.31, 0.11, 9.5, 87.6, 0.35, 0.22)])
            mx.eval(g)
        assert any(isinstance(k, tuple) and k[0] == "sph-tau"
                   for k in m._compiled) == (dtype == mx.float32)
        return np.array([float(np.asarray(x, np.float64)) for x in g])
    g32, g64 = grads(mx.float32), grads(mx.float64)
    assert np.abs(g32 - g64).max() / np.abs(g64).max() < 2e-3
