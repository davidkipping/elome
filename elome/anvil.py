"""anvil integration: the batched transit model and
a synthetic transit-fitting target.

``make_quad_transit_flux`` returns the engine-contract model function
    model_fn(v: (n_chains, 8), x: (2, m)) -> (n_chains, m) flux deviation
and is pure elome (mlx + numpy only); ``make_hybrid_transit_flux`` is its
counterpart for the hybrid limb-darkening laws, optionally on a spotted
star. ``make_target`` assembles the
full sampling problem (chunked likelihood + transform + synthetic data);
the engine is imported lazily, so the core package stays standalone.

Model-space parameters v (all O(1), float32-safe offsets):
    0: t0_off  mid-transit offset from t0_ref [days]
    1: p_off   period offset from period_ref [days]
    2: r       radius ratio Rp/R*
    3: b       impact parameter
    4: a       scaled semi-major axis a/R*
    5: q1      Kipping (2013) triangular LD
    6: q2      Kipping (2013) triangular LD
    7: df0     baseline flux deviation from 1

``x`` rows: dt (per-orbit residual times, float64-preprocessed) and k
(orbit numbers). Predicts the flux *deviation* df0 + (F - 1), matching
data y - 1.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .flux import flux_dev
from .ld import q_to_u, q_to_u_np, u_to_q_np
from .orbit import epoch_center_times, separation_circular, tau_from_epochs

__all__ = ["make_quad_transit_flux", "make_target", "QuadTransitTarget",
           "import_engine", "make_ecc_transit_flux", "make_ecc_target",
           "EccTransitTarget", "ecc_constraint_penalty",
           "PenalizedLogLike", "PARAM_NAMES", "PARAM_NAMES_ECC",
           "make_transit_target", "TransitTarget", "DEFAULT_BOUNDS",
           "make_hybrid_transit_flux", "transit_param_names"]

PARAM_NAMES = ["t0_off", "p_off", "r", "b", "a", "q1", "q2", "df0"]


def import_engine():
    """Import the anvil sampling engine.

    Returns (engine_module, ChunkedGaussianLogLike). A missing engine
    raises ImportError with the install command; an ``anvil`` that is
    present but fails to import (a missing dependency of its own, or an
    unrelated package of the same name) re-raises that error.
    """
    import importlib.util
    if importlib.util.find_spec("anvil") is None:
        raise ImportError(
            "the anvil MCMC engine is not installed; install it with "
            "`pip install git+https://github.com/davidkipping/anvil`")
    try:
        import anvil as engine
        from anvil.precision import ChunkedGaussianLogLike
    except ImportError as err:
        raise ImportError(
            "an 'anvil' package is installed but is not the anvil MCMC "
            f"engine, or failed to import: {err}") from err
    return engine, ChunkedGaussianLogLike


def make_quad_transit_flux(period_ref: float, analytic_vjp: bool = True,
                           core: str = "metal"):
    """Engine-contract batched model. period_ref is baked in as a plain
    Python float (a numpy scalar here would silently force-evaluate the
    graph mid-compile).

    core selects the photometric implementation (identical forward math):
      "metal"    — fused Metal kernels on the fp32 GPU stream, silently
                   falling back to "analytic" for fp64/CPU/unsupported
                   machines (the default; anvil's fp64 verification path
                   and CPU data generation ride the fallback);
      "analytic" — MLX graph forward + ALFM19 closed-form backward
                   ("graph" is an alias, as for make_ecc_transit_flux);
      "autodiff" — pure reverse-mode autodiff (the reference).
    analytic_vjp=False is a back-compat alias for core="autodiff"."""
    period_ref = float(period_ref)
    if not analytic_vjp:
        core = "autodiff"
    if core == "metal":
        from .anchored import pack_orbit_constants
        from .metal import (_gpu_stream_active, flux_dev_metal,
                            make_model_core_metal, metal_available)
        graph_fn = _build_graph_model(period_ref, flux_dev_metal)
        if not metal_available():
            return graph_fn
        model_core = make_model_core_metal(period_ref)

        def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
            # Fused path: whole orbit + photometry in one kernel. The
            # circular MODE is kept (8 parameters, b sampled directly) but
            # runs on the same kernel as the eccentric model, with
            # k = h = 0 and cos i = b / a — exact, since the anchored orbit
            # degenerates to the circular one, and the kernel skips the
            # Kepler solve on e == 0 chains. Anything the kernel can't
            # serve (fp64, CPU stream — anvil's data generation and
            # verification paths) falls back to the graph model.
            if (v.dtype == mx.float32 and x.dtype == mx.float32
                    and _gpu_stream_active()):
                u1, u2 = q_to_u(v[:, 5], v[:, 6])
                zero = mx.zeros_like(v[:, 3])
                orb = pack_orbit_constants(zero, zero, v[:, 3] / v[:, 4])
                dev = model_core(x, v[:, 0], v[:, 1], v[:, 2], v[:, 4],
                                 orb, u1, u2)
                return v[:, 7:8] + dev
            return graph_fn(v, x)

        return flux_dev_fn
    elif core in ("analytic", "graph"):
        from .vjp import flux_dev_analytic as core_fn
    elif core == "autodiff":
        core_fn = flux_dev
    else:
        raise ValueError(f"unknown core {core!r}")
    return _build_graph_model(period_ref, core_fn)


def _build_graph_model(period_ref: float, core):
    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        dt, k = x[0], x[1]
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        r, b, a = v[:, 2:3], v[:, 3:4], v[:, 4:5]
        q1, q2 = v[:, 5:6], v[:, 6:7]
        df0 = v[:, 7:8]
        u1, u2 = q_to_u(q1, q2)
        period = period_ref + p_off
        tau = tau_from_epochs(dt[None, :], k[None, :], t0_off, p_off, period)
        z = separation_circular(tau, period, b, a)
        return df0 + core(z, r, u1, u2)

    return flux_dev_fn


# ---------------------------------------------------------------------------
# hybrid limb darkening, optionally on a spotted star
# ---------------------------------------------------------------------------

_HYBRID_LAWS = ("hybrid2", "hybrid4", "hybrid5")
_SPOT_FIELDS = ("lon", "sinlat", "rad", "contrast")
_DEG = math.pi / 180.0


def transit_param_names(limb_dark: str = "quadratic", eccentric: bool = False,
                        n_spots: int = 0) -> list:
    """The model-space parameter vector's names, in order, for
    ``make_transit_target``'s options:

        t0_off, p_off, r, b, a, the limb darkening's q1..qn,
        [secosw, sesinw], [per spot j = 1..K: lonj, sinlatj, radj,
        contrastj, then prot, cos_istar, lam], df0

    The quadratic law's are ``PARAM_NAMES`` / ``PARAM_NAMES_ECC``. A
    hybrid law's q are those of its prior (``ld.hybrid2_from_q``,
    ``ld.hybrid4_from_q``, ``ld.simplex_from_q``): n = 2, 4, 5. A spot's
    longitude (at ``spot_epoch``) and angular radius are in degrees, its
    latitude as its sine (uniform over the sphere), its contrast the
    spot's intensity over the photosphere's; the star's rotation period
    ``prot`` in days, its spin axis's inclination as cos(istar) (isotropic)
    and the sky-projected obliquity ``lam`` in degrees."""
    if limb_dark == "quadratic":
        n_q = 2
    elif limb_dark in _HYBRID_LAWS:
        from .hybrid import get_law
        n_q = get_law(limb_dark).n_w
    else:
        raise ValueError(f"unknown limb_dark {limb_dark!r}; expected "
                         f"'quadratic' or one of {_HYBRID_LAWS}")
    n_spots = int(n_spots)
    if n_spots < 0:
        raise ValueError("n_spots must be >= 0")
    if n_spots and limb_dark == "quadratic":
        raise ValueError("a spotted star needs a hybrid law (limb_dark="
                         "'hybrid2' | 'hybrid4' | 'hybrid5')")
    names = ["t0_off", "p_off", "r", "b", "a"]
    names += [f"q{j}" for j in range(1, n_q + 1)]
    if eccentric:
        names += ["secosw", "sesinw"]
    for j in range(1, n_spots + 1):
        names += [f"{f}{j}" for f in _SPOT_FIELDS]
    if n_spots:
        names += ["prot", "cos_istar", "lam"]
    return names + ["df0"]


def _w_from_q(limb_dark: str, q: mx.array) -> mx.array:
    """(n, n_w) q in the unit cube -> the law's shape weights, uniform on
    its physical region (``elome.ld``)."""
    from .ld import hybrid2_from_q, hybrid4_from_q, simplex_from_q
    if limb_dark == "hybrid2":
        return mx.stack(hybrid2_from_q(q[:, 0], q[:, 1]), axis=-1)
    if limb_dark == "hybrid4":
        return hybrid4_from_q(q)
    return simplex_from_q(q)


def make_hybrid_transit_flux(period_ref: float, limb_dark: str = "hybrid4",
                             eccentric: bool = False, n_spots: int = 0,
                             exp_time: float = 0.0,
                             integration: str = "contact", n_gl: int = 5,
                             n_sub: int = 1, relative: bool = True):
    """Engine-contract batched model for a hybrid limb-darkening law:
    v = (n_chains, d) with ``transit_param_names(limb_dark, eccentric,
    n_spots)`` and x = (2, m) epoch-centered times -- (3, m) with spots,
    the third row each point's time since ``spot_epoch`` -- returning
    df0 + (F - 1), (n_chains, m).

    The photometry is ``metal.flux_dev_from_tau``: float32 on the GPU
    runs the hybrid (and, with spots, the spot) Metal kernels, float64 the
    graph on the CPU stream, so anvil's two precisions share this one
    function. ``exp_time`` (days) > 0 integrates each exposure by
    ``integration`` ("contact", "supersample" or "none"; ``n_gl``,
    ``n_sub`` as there). A spotted model is evaluated eagerly: it raises
    under ``mx.compile`` (run anvil with ``compile_step=False``).

    ``relative`` (spots only): the flux relative to the spotted star's own
    level at each time, (1 + dF) / (1 + dF_star) - 1, with dF_star the
    same model with the planet half an orbit away -- what a light curve
    normalised to its out-of-transit level measures, at twice the cost.
    False: relative to the unspotted star (``flux_dev_from_tau``'s
    normalisation), which leaves ``df0`` to absorb the spots' visible
    deficit, a ridge HMC samples poorly.
    """
    from .metal import flux_dev_from_tau
    period_ref = float(period_ref)
    if limb_dark not in _HYBRID_LAWS:
        raise ValueError(f"limb_dark must be one of {_HYBRID_LAWS}; the "
                         "quadratic law is make_quad_transit_flux")
    names = transit_param_names(limb_dark, eccentric, n_spots)
    col = {n: i for i, n in enumerate(names)}
    iq, n_q = col["q1"], sum(n[0] == "q" and n[1:].isdigit() for n in names)
    K = int(n_spots)
    exp_time = float(exp_time)

    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        period = period_ref + v[:, 1]
        tau = tau_from_epochs(x[0][None, :], x[1][None, :], t0_off, p_off,
                              period[:, None])
        kw = {}
        if eccentric:
            k, h, _, _ = _ecc_orbit_from_v(v, ik=col["secosw"])
            kw.update(secosw=k, sesinw=h)
        if K:
            sp = mx.reshape(v[:, col["lon1"]:col["lon1"] + 4 * K], (-1, K, 4))
            # the sines kept off +-1, where arcsin's and arccos's slopes are
            # infinite (a float32 sigmoid reaches its box's edge exactly)
            lat = mx.arcsin(mx.clip(sp[..., 1], -1.0 + 1e-6, 1.0 - 1e-6))
            kw.update(
                spots=mx.stack([sp[..., 0] * _DEG, lat, sp[..., 2] * _DEG,
                                sp[..., 3]], axis=-1),
                spot_time=mx.broadcast_to(x[2][None, :], tau.shape),
                prot=v[:, col["prot"]],
                istar=mx.arccos(mx.clip(v[:, col["cos_istar"]], -1.0 + 1e-6,
                                        1.0 - 1e-6)),
                lam=v[:, col["lam"]] * _DEG)
        def dev(tt):
            return flux_dev_from_tau(
                tt, period, v[:, 4], v[:, 3], v[:, 2], limb_dark=limb_dark,
                u=_w_from_q(limb_dark, v[:, iq:iq + n_q]), exp_time=exp_time,
                integration=integration if exp_time > 0.0 else "none",
                n_gl=n_gl, n_sub=n_sub, **kw)
        if K and relative:
            # the star alone: the same times on the star's clock, the planet
            # at the opposite conjunction (behind the star, which hides it)
            half = 0.5 * period[:, None]
            star = dev(mx.where(tau < 0.0, tau + half, tau - half))
            return v[:, -1:] + (dev(tau) - star) / (1.0 + star)
        return v[:, -1:] + dev(tau)

    return flux_dev_fn


@dataclass
class QuadTransitTarget:
    """Synthetic quadratic-LD transit-fitting problem (anvil-ready)."""

    target: object                # anvil.TransformedLogDensity
    transform: object             # anvil.Transform
    loglike: object               # anvil.ChunkedGaussianLogLike
    truth_model: np.ndarray       # (8,) true params, model units
    t_ref: float
    t_model: np.ndarray
    y: np.ndarray


def make_target(
    n_data: int = 100_000,
    yerr: float = 5e-4,
    seed: int = 0,
    policy=None,
    baseline_days: float = 90.0,
    analytic_vjp: bool = True,
    core: str = "metal",
):
    """Build the synthetic problem in well-conditioned units.

    Absolute BJD-like times (~2.457e6 days) are generated and reduced in
    float64 on the CPU; only per-orbit residuals, orbit numbers, and
    baseline-subtracted fluxes ever reach float32.
    """
    engine, ChunkedGaussianLogLike = import_engine()

    rng = np.random.default_rng(seed)

    t_ref = 2_457_000.0
    t_abs = t_ref + np.sort(rng.uniform(0.0, baseline_days, size=n_data))
    t_model = t_abs - t_ref

    # truth: warm-Jupiter-ish, quadratic LD via Kipping q
    u1_true, u2_true = 0.40, 0.25
    q1_true, q2_true = u_to_q_np(u1_true, u2_true)
    truth = np.array([
        1.2345,       # t0 [days since t_ref]
        3.456,        # period [days]
        0.10,         # r
        0.30,         # b
        8.80,         # a/R*
        float(q1_true),
        float(q2_true),
        1.0,          # baseline flux
    ])

    # generate data through the model itself, in float64 on the CPU
    t0_ref = float(truth[0]) - 0.009
    period_ref = float(truth[1]) + 0.0005
    truth_model = np.array([
        truth[0] - t0_ref, truth[1] - period_ref,
        truth[2], truth[3], truth[4], truth[5], truth[6],
        truth[7] - 1.0,
    ])
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    model_fn = make_quad_transit_flux(period_ref=period_ref,
                                      analytic_vjp=analytic_vjp, core=core)
    with mx.stream(mx.cpu):
        dev_true = np.array(
            model_fn(mx.array(truth_model[None, :], dtype=mx.float64),
                     mx.array(x64, dtype=mx.float64))[0],
            dtype=np.float64,
        )
    y = 1.0 + dev_true + yerr * rng.standard_normal(n_data)
    y_fit = y - 1.0
    yerr_arr = np.full(n_data, yerr)

    transform = engine.Transform([
        engine.ParamSpec("t0_off", lo=-0.5, hi=0.5,
                            report_offset=t_ref + t0_ref),
        engine.ParamSpec("p_off", lo=-0.05, hi=0.05,
                            report_offset=period_ref),
        engine.ParamSpec("r", lo=0.01, hi=0.5),
        engine.ParamSpec("b", lo=0.0, hi=0.9),
        engine.ParamSpec("a", lo=2.0, hi=50.0),
        engine.ParamSpec("q1", lo=0.0, hi=1.0),
        engine.ParamSpec("q2", lo=0.0, hi=1.0),
        engine.ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
    ])
    loglike = ChunkedGaussianLogLike(model_fn, x64, y_fit, yerr_arr, policy)
    target = engine.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return QuadTransitTarget(
        target=target, transform=transform, loglike=loglike,
        truth_model=truth_model, t_ref=t_ref, t_model=t_model, y=y,
    )



# ---------------------------------------------------------------------------
# real-data entry point
# ---------------------------------------------------------------------------

#: default ParamSpec boxes, in model units. ``b`` reaches past 1 so
#: grazing geometries are inside the box; widen or narrow per target.
DEFAULT_BOUNDS = {
    "t0_off": (-0.5, 0.5), "p_off": (-0.05, 0.05),
    "r": (0.005, 0.5), "b": (0.0, 1.2), "a": (1.5, 200.0),
    "q1": (0.0, 1.0), "q2": (0.0, 1.0), "df0": (-0.01, 0.01),
    "secosw": (-0.95, 0.95), "sesinw": (-0.95, 0.95),
    "q3": (0.0, 1.0), "q4": (0.0, 1.0), "q5": (0.0, 1.0),
    # a spotted star: every spot's (lon, sinlat, rad, contrast) -- a key
    # here covers spots 1..K, a numbered one (``"lon2"``) a single spot --
    # and the star's (cos_istar, lam); prot has no default box
    "lon": (-180.0, 180.0), "sinlat": (-1.0, 1.0), "rad": (1.0, 30.0),
    "contrast": (0.0, 1.0), "cos_istar": (0.0, 1.0), "lam": (-180.0, 180.0),
}


@dataclass
class TransitTarget:
    """A fitting problem over REAL data, ready for anvil.

    ``target`` is the TransformedLogDensity to sample; ``x64`` and
    ``y_fit`` are the conditioned inputs (epoch-centered times, flux minus
    one) should a different likelihood -- an anvil-gp GP over the same
    mean function, say -- want to wrap the same model.
    """

    target: object                # anvil.TransformedLogDensity
    transform: object             # anvil.Transform
    loglike: object               # ChunkedGaussianLogLike (maybe penalized)
    model_fn: object              # the batched (v, x) -> flux deviation
    x64: np.ndarray               # (2, n) epoch-centered times
    y_fit: np.ndarray             # y - 1
    yerr: np.ndarray
    t_ref: float                  # float64 time origin subtracted first
    t0_ref: float                 # t0 reference (model units, from t_ref)
    period_ref: float
    param_names: list
    eccentric: bool
    limb_dark: str = "quadratic"
    n_spots: int = 0
    #: the time (input system) the spots' longitudes refer to; None unspotted
    spot_epoch: float | None = None
    #: pass to ``anvil.run(..., compile_step=)``: a spotted model is
    #: evaluated eagerly and raises under ``mx.compile``
    compile_step: bool = True

    def model_params(self, t0, period, **kw):
        """Physical values -> the model-space vector the sampler uses.

        Handles the two offset parameters so callers never have to
        remember which reference is subtracted from what.
        """
        vals = dict(kw)
        vals["t0_off"] = float(t0) - self.t_ref - self.t0_ref
        vals["p_off"] = float(period) - self.period_ref
        vals.setdefault("df0", 0.0)
        missing = [n for n in self.param_names if n not in vals]
        if missing:
            raise ValueError(f"missing parameters: {missing}")
        return np.array([float(vals[n]) for n in self.param_names])


def make_transit_target(t, y, yerr, t0_guess, period_guess,
                        eccentric: bool = False, bounds=None, policy=None,
                        core: str = "metal", strength: float = 1e6,
                        limb_dark: str = "quadratic", n_spots: int = 0,
                        exp_time: float = 0.0, integration: str = "contact",
                        n_gl: int = 5, spot_epoch=None,
                        relative: bool = True):
    """Build an anvil fitting problem from REAL photometry.

    ``core`` is "metal" (default: fused fp32 kernels, falling back to the
    analytic graph off the GPU), "analytic" (alias "graph") or "autodiff",
    for circular and eccentric models alike.

    This is the entry point for mission data (Kepler, TESS, ...). It owns
    the conditioning that the float32 sampling path depends on, so a
    caller never has to rediscover it:

    * absolute times are reduced by a float64 ``t_ref`` and then turned
      into (per-orbit residual, orbit number) by ``epoch_center_times``
      -- raw BJD cannot survive float32 (0.25 d ulp at 2.457e6);
    * flux is fit as ``y - 1`` so the graph carries O(depth) numbers;
    * ``t0_guess``/``period_guess`` become the references the sampler
      perturbs around, and ``report_offset`` puts the reported values back
      on the original time system;
    * for ``eccentric=True`` the joint-constraint barrier is attached.
      Elómë clamps every numerical hazard, so an unphysical geometry
      otherwise returns an ordinary finite log-likelihood and the chain
      samples an improper posterior in silence -- omitting the barrier is
      the single easiest way to get a wrong eccentric fit.

    Args:
        t, y, yerr: photometry, any consistent time system; y normalized
            to ~1 out of transit.
        t0_guess, period_guess: references in the SAME system as ``t``.
            They need only be good to the width of the ``t0_off``/``p_off``
            boxes (0.5 d and 0.05 d by default).
        eccentric: 10-parameter model with (sqrt(e) cos w, sqrt(e) sin w)
            instead of the 8-parameter circular one.
        limb_dark: "quadratic" (Kipping's q1, q2) or a hybrid law,
            "hybrid2" | "hybrid4" | "hybrid5", sampled through its q prior
            (q1..qn, uniform on the law's physical profiles); see
            ``transit_param_names`` for the vector's order.
        n_spots: K > 0 fits a spotted star (hybrid laws only): K spots of
            (lonj, sinlatj, radj, contrastj) and the star's (prot,
            cos_istar, lam). ``prot`` has no default box: give one in
            ``bounds``. Spots are interchangeable, so the posterior has K!
            relabelled copies; sort the draws afterwards (by longitude, say).
            Run anvil with ``compile_step=target.compile_step`` (False): a
            spotted model is evaluated eagerly.
        exp_time, integration, n_gl: an exposure time (in t's units) and
            its rule, for hybrid laws (``make_hybrid_transit_flux``).
        spot_epoch: when the spots' longitudes are measured (t's system);
            default the predicted mid-transit nearest the data's middle,
            which keeps the star's float32 phase accurate.
        relative: with spots, model the flux relative to the spotted
            star's own level (True, for data normalised to their
            out-of-transit level) or to the unspotted star (False); see
            ``make_hybrid_transit_flux``.
        bounds: per-parameter (lo, hi) overrides on DEFAULT_BOUNDS (a spot
            field, ``"rad"``, covers every spot; ``"rad2"`` one). The
            defaults cap r at 0.5, b at 1.2 and a at 200; an occultor
            larger than the star (a white-dwarf host) needs them widened,
            e.g. ``{"r": (1.0, 12.0), "b": (0.0, 13.0), "a": (1.5, 1000.0)}``.
            The model takes any r > 0 (fp32 error grows ~ r^2; see the
            docs on occultors larger than the star).
        strength: barrier weight for the eccentric constraints.
    """
    engine, ChunkedGaussianLogLike = import_engine()
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yerr = (np.full(t.size, float(yerr), dtype=np.float64)
            if np.ndim(yerr) == 0 else np.asarray(yerr, dtype=np.float64))
    if not (t.shape == y.shape == yerr.shape):
        raise ValueError(f"t, y, yerr must match in shape; got {t.shape}, "
                         f"{y.shape}, {yerr.shape}")
    if t.size == 0:
        raise ValueError("no data points")
    if not np.all(np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr)):
        raise ValueError("t, y and yerr must all be finite "
                         "(mask your data before fitting)")
    if np.any(yerr <= 0.0):
        raise ValueError("yerr must be positive")
    if np.any(np.diff(t) < 0.0):
        order = np.argsort(t)
        t, y, yerr = t[order], y[order], yerr[order]

    # float64 host conditioning, in the order that matters
    t_ref = float(np.floor(t.min()))
    t_model = t - t_ref
    t0_ref = float(t0_guess) - t_ref
    period_ref = float(period_guess)
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    y_fit = y - 1.0

    hybrid = limb_dark != "quadratic"
    names = transit_param_names(limb_dark, eccentric, n_spots)
    if hybrid and core != "metal":
        raise ValueError("core selects the quadratic law's implementation; "
                         "a hybrid law runs flux_dev_from_tau (float32 on the "
                         "GPU: the Metal kernels; float64: the graph)")
    if not hybrid and exp_time:
        raise ValueError("exp_time needs a hybrid law here; the quadratic "
                         "targets are instantaneous")
    lims = _param_bounds(names, bounds)
    if n_spots:
        # the star's clock: each point's time since spot_epoch, reduced in
        # float64 like the planet's
        if spot_epoch is None:
            mid = 0.5 * (t_model.min() + t_model.max())
            ep = t0_ref + round((mid - t0_ref) / period_ref) * period_ref
        else:
            ep = float(spot_epoch) - t_ref
        x64 = np.vstack([x64, (t_model - ep)[None, :]])
        spot_epoch = t_ref + ep
    report = {"t0_off": t_ref + t0_ref, "p_off": period_ref, "df0": 1.0}
    transform = engine.Transform([
        engine.ParamSpec(n, lo=lims[n][0], hi=lims[n][1],
                            **({"report_offset": report[n]}
                               if n in report else {}))
        for n in names])

    if hybrid:
        model_fn = make_hybrid_transit_flux(
            period_ref, limb_dark, eccentric=eccentric, n_spots=n_spots,
            exp_time=exp_time, integration=integration, n_gl=n_gl,
            relative=relative)
    elif eccentric:
        model_fn = make_ecc_transit_flux(period_ref=period_ref, core=core)
    else:
        model_fn = make_quad_transit_flux(period_ref=period_ref, core=core)
    loglike = ChunkedGaussianLogLike(model_fn, x64, y_fit, yerr, policy)
    if eccentric:
        ik = names.index("secosw")
        loglike = PenalizedLogLike(
            loglike, lambda v: ecc_constraint_penalty(v, strength=strength,
                                                      ik=ik))
    target = engine.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return TransitTarget(
        target=target, transform=transform, loglike=loglike,
        model_fn=model_fn, x64=x64, y_fit=y_fit, yerr=yerr, t_ref=t_ref,
        t0_ref=t0_ref, period_ref=period_ref, param_names=list(names),
        eccentric=eccentric, limb_dark=limb_dark, n_spots=int(n_spots),
        spot_epoch=spot_epoch if n_spots else None,
        compile_step=not n_spots)


def _param_bounds(names, bounds) -> dict:
    """(lo, hi) per name: ``bounds`` by name, then by spot field (``"rad"``
    for every ``"radj"``), then DEFAULT_BOUNDS likewise."""
    bounds = dict(bounds or {})
    family = {n: n.rstrip("0123456789") for n in names}
    known = set(DEFAULT_BOUNDS) | set(names) | set(family.values())
    for k in bounds:
        if k not in known:
            raise ValueError(f"unknown parameter {k!r}; expected "
                             f"{sorted(known)}")
    lims = {}
    for n in names:
        f = family[n] if family[n] in _SPOT_FIELDS else None
        for src in (bounds, DEFAULT_BOUNDS):
            if n in src or (f and f in src):
                lims[n] = tuple(src[n] if n in src else src[f])
                break
        else:
            raise ValueError(f"{n!r} has no default box: pass bounds="
                             f"{{{n!r}: (lo, hi)}}")
    return lims


# ---------------------------------------------------------------------------
# synthetic targets (injection-recovery)
# ---------------------------------------------------------------------------

PARAM_NAMES_ECC = ["t0_off", "p_off", "r", "b", "a", "q1", "q2",
                   "secosw", "sesinw", "df0"]

#: numerical validity ceiling of the solve (the Markley + one-refinement
#: scheme holds |dE| <= 4.1e-7 at e = 0.999).
E_MAX_NUMERICAL = 0.999


def _ecc_orbit_from_v(v, e_max: float = E_MAX_NUMERICAL, ik: int = 7):
    """Sampler coordinates -> (k, h, ci, e) for the anchored orbit.

    (secosw, sesinw) is shrunk radially onto the disc of radius
    sqrt(e_max) so the solve never sees an unsupported eccentricity; the
    *physical* limits (periastron clearance, cos i <= 1) are a prior
    concern and live in ``ecc_constraint_penalty``, not here. The shrink
    denominator is set to 1 where inactive rather than floored, per the
    masked-division rule. ``ik`` is secosw's column (sesinw follows it).
    """
    k_raw, h_raw = v[:, ik], v[:, ik + 1]
    e_raw = k_raw * k_raw + h_raw * h_raw
    big = e_raw > e_max
    den = mx.where(big, e_raw, mx.ones_like(e_raw))
    sc = mx.where(big, mx.sqrt(e_max / den), mx.ones_like(e_raw))
    k, h = k_raw * sc, h_raw * sc
    e = k * k + h * h
    # floor before the sqrt: d(sqrt)/de is infinite at e == 0, and
    # h * inf = NaN there. e = 0 is an interior point of the disc.
    sq = mx.sqrt(mx.maximum(e, 1e-30))
    esw = h * sq                                   # e sin w, division-free
    b, a = v[:, 3], v[:, 4]
    # b = (a cos i) (1 - e^2) / (1 + e sin w)  =>  cos i = b (1 + e sin w) / (a (1 - e^2))
    from .anchored import cos_inc_from_b
    ci = cos_inc_from_b(a, b, e, esw, e_max)
    return k, h, ci, e


def _huber(c, delta: float = 1.0):
    """One-sided Huber violation: 0 for c <= 0, c^2 near the boundary,
    linear beyond delta.

    A pure quadratic is the textbook HMC barrier, but here a far-outside
    proposal can drive a violation to O(1000) -- the projection onto the
    e_max disc sends 1 - e^2 to 0.002, which makes |cos i| explode -- and
    c^2 then contributes ~1e12 with ~1e11 gradients, collapsing the step
    size. Going linear past delta keeps the force bounded (2 delta) while
    never letting it vanish, so a chain thrown into the failure region is
    pushed back instead of either exploding or stalling.
    """
    x = mx.maximum(c, 0.0)
    return mx.where(x <= delta, x * x, delta * (2.0 * x - delta))


def ecc_constraint_penalty(v, strength: float = 1e6,
                           e_max: float = E_MAX_NUMERICAL,
                           ik: int = 7) -> mx.array:
    """Smooth barrier for the JOINT physical constraints that a box of
    ParamSpecs cannot express (returns <= 0, exactly 0 when feasible).

      * periastron clearance   a (1 - e) > 1 + r
      * a real inclination     |cos i| <= 1
      * the eccentricity disc  secosw^2 + sesinw^2 <= e_max

    Elómë clamps every numerical hazard, so an unphysical proposal
    otherwise returns an ordinary *finite* log-likelihood and the chain
    samples an improper posterior silently. A quadratic barrier is used
    rather than a -inf wall because hard walls make HMC diverge.

    The third term is what makes the barrier *escapable*. The model
    projects (secosw, sesinw) radially onto the e_max disc, so outside it
    the likelihood is exactly constant along the radial direction: a
    chain thrown into the corners of the +/-0.95 box (where e_raw reaches
    1.8) would sit on a plateau with no radial force pushing it back,
    only the ParamSpec sigmoid Jacobian. Penalising the UNPROJECTED
    e_raw restores a gradient that points inward.

    ``ik`` is secosw's column: 7 in the quadratic layout, later for a
    hybrid law's longer weight block (``transit_param_names``).
    """
    k, h, ci, e = _ecc_orbit_from_v(v, e_max, ik)
    r, a, b = v[:, 2], v[:, 4], v[:, 3]
    e_raw = v[:, ik] ** 2 + v[:, ik + 1] ** 2
    # e sin w of the PROJECTED orbit, as _ecc_orbit_from_v forms it (its
    # masked denominator is safe; re-deriving it here through
    # sqrt(e_max / max(e_raw, 1e-30)) gave NaN gradients in fp32 at e = 0,
    # the unused branch's backward squaring 1e-30 to 0)
    esw = h * mx.sqrt(mx.maximum(e, 1e-30))
    # each residual is DIMENSIONLESS and O(1): |cos i| - 1 would instead
    # explode, because projecting onto the e_max disc drives 1 - e^2 to
    # 0.002 and cos i with it, contributing ~1e12 with ~1e11 gradients.
    # Same zero set, same sign, bounded scale.
    c_peri = ((1.0 + r) - a * (1.0 - e)) / (1.0 + r + a)
    c_inc = (b * (1.0 + esw) - a * (1.0 - e * e)) / (1.0 + b + a)
    c_disc = (e_raw - e_max) / (1.0 + e_max)
    return -strength * (_huber(c_peri) + _huber(c_inc) + _huber(c_disc))


def make_ecc_transit_flux(period_ref: float, core: str = "metal"):
    """Engine-contract batched eccentric model, v = (n_chains, 10) with
    PARAM_NAMES_ECC and x = (2, m) epoch-centered times.

    Eccentricity is sampled as (sqrt(e) cos w, sqrt(e) sin w), whose
    e = 0 interior point a sampler genuinely visits; the transit-anchored
    orbit keeps float32 gradients accurate there (see anchored.py). Pair
    it with ``ecc_constraint_penalty`` — this function deliberately
    returns finite flux for unphysical geometries, exactly as the
    circular model does.
    """
    period_ref = float(period_ref)
    if core not in ("metal", "analytic", "graph", "autodiff"):
        raise ValueError(f"unknown core {core!r}; choose 'metal', "
                         "'analytic' (alias 'graph') or 'autodiff'")
    # the graph's photometric core: analytic backward (finite at every
    # contact, r = 1 included) unless the autodiff oracle is asked for
    graph_fn = _build_graph_ecc_model(period_ref,
                                      autodiff=core == "autodiff")
    if core != "metal":
        return graph_fn
    from .metal import _gpu_stream_active, make_model_core_metal, metal_available
    if not metal_available():
        return graph_fn
    from .anchored import pack_orbit_constants
    model_core = make_model_core_metal(period_ref)

    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        if (v.dtype == mx.float32 and x.dtype == mx.float32
                and _gpu_stream_active()):
            u1, u2 = q_to_u(v[:, 5], v[:, 6])
            k, h, ci, _ = _ecc_orbit_from_v(v)
            orb = pack_orbit_constants(k, h, ci)
            dev = model_core(x, v[:, 0], v[:, 1], v[:, 2], v[:, 4],
                             orb, u1, u2)
            return v[:, 9:10] + dev
        return graph_fn(v, x)

    return flux_dev_fn


def _build_graph_ecc_model(period_ref: float, autodiff: bool = False):
    """Same math in pure MLX graph ops: the fp64 / CPU verification path.
    The photometry is flux_dev_analytic (closed-form backward) unless
    ``autodiff``: flux_dev's reverse mode is NaN at r = 1, z = 0."""
    from .anchored import separation_anchored
    from .vjp import flux_dev_analytic
    phot = flux_dev if autodiff else flux_dev_analytic

    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        dt, kk = x[0], x[1]
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        r, a = v[:, 2:3], v[:, 4:5]
        u1, u2 = q_to_u(v[:, 5:6], v[:, 6:7])
        df0 = v[:, 9:10]
        k, h, ci, _ = _ecc_orbit_from_v(v)
        period = period_ref + p_off
        tau = tau_from_epochs(dt[None, :], kk[None, :], t0_off, p_off, period)
        phi = (2.0 * np.pi) * tau / period
        z, front = separation_anchored(phi, k[:, None], h[:, None], a,
                                       ci[:, None])
        f = phot(z, r, u1, u2)
        return df0 + mx.where(front & (z < 1.0 + r), f, 0.0)

    return flux_dev_fn


class PenalizedLogLike:
    """``ChunkedGaussianLogLike`` plus a joint-constraint barrier.

    The engine's ParamSpecs are per-parameter boxes, so constraints that
    couple parameters have nowhere else to live; without this the chain
    samples an improper posterior with no error message.
    """

    def __init__(self, base, penalty):
        self.base = base
        self.penalty = penalty

    def __call__(self, v: mx.array) -> mx.array:
        return self.base(v) + self.penalty(v)

    def hi(self, v: mx.array) -> mx.array:
        # fp64 lives on the CPU (MLX has no fp64 on Metal); the base's hi
        # switches stream itself, the penalty must too
        with mx.stream(mx.cpu):
            return self.base.hi(v) + self.penalty(v.astype(mx.float64))

    def __getattr__(self, name):
        # __getattr__ runs for ANY attribute missing from the instance,
        # including lookups that happen before __init__ has assigned
        # `base` -- copy/deepcopy/unpickle all probe __deepcopy__,
        # __getstate__, __setstate__ on a bare object first. Delegating
        # those blindly recurses on `base` itself and raises
        # RecursionError where AttributeError is required.
        if name.startswith("__") or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["base"], name)


@dataclass
class EccTransitTarget:
    """Synthetic eccentric transit-fitting problem (anvil-ready)."""

    target: object
    transform: object
    loglike: object
    truth_model: np.ndarray       # (10,) true params, model units
    t_ref: float
    t_model: np.ndarray
    y: np.ndarray


def make_ecc_target(n_data: int = 100_000, baseline_days: float = 90.0,
                    yerr: float = 5e-4, seed: int = 11,
                    core: str = "metal", ecc: float = 0.3,
                    omega_deg: float = 63.0) -> EccTransitTarget:
    """Injection-recovery problem for the 10-parameter eccentric model."""
    engine, ChunkedGaussianLogLike = import_engine()
    rng = np.random.default_rng(seed)

    t_ref = 2_457_000.0
    t_abs = t_ref + np.sort(rng.uniform(0.0, baseline_days, size=n_data))
    t_model = t_abs - t_ref

    u1_true, u2_true = 0.40, 0.25
    q1_true, q2_true = u_to_q_np(u1_true, u2_true)
    w = math.radians(omega_deg)
    truth = np.array([
        1.2345, 3.456, 0.10, 0.30, 8.80,
        float(q1_true), float(q2_true),
        math.sqrt(ecc) * math.cos(w), math.sqrt(ecc) * math.sin(w), 1.0,
    ])
    t0_ref = float(truth[0]) - 0.009
    period_ref = float(truth[1]) + 0.0005
    truth_model = np.array([
        truth[0] - t0_ref, truth[1] - period_ref, truth[2], truth[3],
        truth[4], truth[5], truth[6], truth[7], truth[8], truth[9] - 1.0,
    ])
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    model_fn = make_ecc_transit_flux(period_ref=period_ref, core=core)
    with mx.stream(mx.cpu):
        dev_true = np.array(
            model_fn(mx.array(truth_model[None, :], dtype=mx.float64),
                     mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)
    y = 1.0 + dev_true + yerr * rng.standard_normal(n_data)
    y_fit = y - 1.0

    policy = engine.PrecisionPolicy()
    transform = engine.Transform([
        engine.ParamSpec("t0_off", lo=-0.5, hi=0.5,
                            report_offset=t_ref + t0_ref),
        engine.ParamSpec("p_off", lo=-0.05, hi=0.05,
                            report_offset=period_ref),
        engine.ParamSpec("r", lo=0.01, hi=0.5),
        engine.ParamSpec("b", lo=0.0, hi=1.2),
        engine.ParamSpec("a", lo=2.0, hi=50.0),
        engine.ParamSpec("q1", lo=0.0, hi=1.0),
        engine.ParamSpec("q2", lo=0.0, hi=1.0),
        # the (sqrt(e) cos w, sqrt(e) sin w) disc inscribed in its box;
        # e <= e_max(a, r) and |cos i| <= 1 are joint, so they are the
        # barrier's job, not the box's.
        engine.ParamSpec("secosw", lo=-0.95, hi=0.95),
        engine.ParamSpec("sesinw", lo=-0.95, hi=0.95),
        engine.ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
    ])
    loglike = PenalizedLogLike(
        ChunkedGaussianLogLike(model_fn, x64, y_fit,
                               np.full(n_data, yerr), policy),
        ecc_constraint_penalty)
    target = engine.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return EccTransitTarget(
        target=target, transform=transform, loglike=loglike,
        truth_model=truth_model, t_ref=t_ref, t_model=t_model, y=y)
