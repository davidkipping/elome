"""batman-style frontend: TransitParams + TransitModel.

Mirrors the API and conventions of Kreidberg's ``batman`` package for
drop-in familiarity:

    import elome
    params = elome.TransitParams()
    params.t0 = 0.0            # time of inferior conjunction
    params.per = 1.0           # orbital period
    params.rp = 0.1            # planet radius [stellar radii]
    params.a = 15.0            # semi-major axis [stellar radii]
    params.inc = 87.0          # inclination [deg]
    params.ecc = 0.0           # eccentricity
    params.w = 90.0            # planet's argument of periastron [deg]
    params.u = [0.1, 0.3]      # limb-darkening coefficients
    params.limb_dark = "quadratic"

    m = elome.TransitModel(params, t)
    flux = m.light_curve(params)          # numpy array

Differences from batman, by design:

* the quadratic/linear/uniform models are evaluated with the analytic
  ALFM19 formulation (float64 accuracy ~1e-13; batman's quadratic path
  carries a ~2e-8 floor from its Hastings E/K approximations), and the
  whole computation is an MLX graph — differentiable and GPU-capable;
* supported limb_dark: "uniform", "linear", "quadratic", "polynomial"
  for I(mu)/I0 = 1 - sum_n u_n (1-mu)^n at ANY order (ALFM19's M_n
  recursion, elome/poly.py), and the hybrid laws "hybrid2",
  "hybrid4", "hybrid5" (even powers of mu plus double poles, elementary
  and more accurate than quadratic; elome/hybrid.py), whose ``u``
  holds the shape-basis weights w; batman's non-polynomial laws
  ("nonlinear", "squareroot", ...) are not covered by this formulation;
* no error-tolerance machinery (`max_err`, `fac`, `nthreads`): the model
  is closed-form, there is no integration error to budget.

For float32 GPU *sampling* with thousands of chains, use
``elome.anvil`` instead — absolute times of order 2.45e6 days
cannot survive float32, so this frontend computes in float64 on the CPU
by default (`dtype=mx.float32` opts into the GPU at your own risk for
well-conditioned times).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .flux import flux_dev
from .metal import _gpu_stream_active, flux_dev_metal, metal_available
from .metal_hybrid import flux_dev_metal_hybrid
from .anchored import anchor_constants_ew, separation_anchored
from .hybrid import LAWS as _HYBRID_LAWS, find_law as _hybrid_law
from .hybrid import flux_dev_hybrid
from .vjp import flux_dev_analytic
from .poly import flux_dev_poly
from .exposure import (contact_geometry, contact_offsets,
                       contact_offsets_anchored, exposure_nodes)
from .solution import sn_dev
from .trig import sincos

__all__ = ["TransitParams", "TransitModel"]

_BUILTIN_LD = ("uniform", "linear", "quadratic", "polynomial")


def _supported_ld() -> tuple:
    return _BUILTIN_LD + tuple(_HYBRID_LAWS)


def _is_vector_law(law) -> bool:
    """Coefficients enter the graph as one vector (uvec), not (u1, u2):
    the polynomial law at any order, and the hybrid laws at their fixed
    count of shape weights."""
    return law == "polynomial" or _hybrid_law(law) is not None


_RING_FIELDS = ("ring_in", "ring_out", "ring_f", "ring_tau")


def _is_ringed(params) -> bool:
    """params.ring_in given = a ringed planet; the other ring fields
    without it are an error."""
    ringed = getattr(params, "ring_in", None) is not None
    if not ringed and any(getattr(params, k, None) is not None
                          for k in _RING_FIELDS[1:]):
        raise ValueError("params.ring_out / ring_f / ring_tau describe a ring; "
                         "set params.ring_in too (or leave them all None)")
    return ringed


_SPOT_FIELDS = ("prot", "istar", "lam", "spot_epoch")


def _is_spotted(params) -> bool:
    """params.spots given = a spotted star; the star's rotation fields
    without it are an error."""
    spotted = getattr(params, "spots", None) is not None
    if not spotted and any(getattr(params, k, None) is not None
                           for k in _SPOT_FIELDS + ("spot_u",)):
        raise ValueError("params.prot / istar / lam / spot_epoch / spot_u "
                         "describe a spotted star; set params.spots too (or "
                         "leave them all None)")
    return spotted


def _spot_shape(spots):
    """The shape of params.spots: (K, 4), or (n_sets, K, 4) for batches."""
    shape = tuple(spots.shape) if isinstance(spots, mx.array) else np.shape(spots)
    if len(shape) not in (2, 3) or shape[-1] != 4 or shape[-2] < 1:
        raise ValueError("params.spots must be (K, 4) -- longitude, latitude, "
                         "angular radius [degrees], contrast per spot -- or "
                         "(n_sets, K, 4) in light_curves; got shape "
                         f"{shape}")
    return shape


# a batched field left None: its value (fp, theta, a ringed planet's f: 0;
# ring_tau: an opaque ring; istar: equator-on; spot_epoch: t0, filled in
# by _stack_params)
_BATCH_DEFAULTS = {"ring_tau": math.inf, "istar": 90.0}


def _is_oblate(params) -> bool:
    """params.f given = an oblate planet. theta alone is an error unless
    the planet has a ring (theta is the oblate planet's or the ring's sky
    angle); a TransitParams without the attributes has neither."""
    obl = getattr(params, "f", None) is not None
    if (not obl and getattr(params, "theta", None) is not None
            and not _is_ringed(params)):
        raise ValueError("params.theta is an oblate planet's or a ring's; "
                         "set params.f too, or params.ring_in (or leave theta "
                         "None)")
    return obl


class TransitParams:
    """Object to store the physical parameters of the transit
    (attribute-compatible with batman.TransitParams)."""

    def __init__(self):
        self.t0 = None            # time of inferior conjunction
        self.per = None           # orbital period
        self.rp = None            # planet radius [stellar radii]
        self.a = None             # semi-major axis [stellar radii]
        self.inc = None           # orbital inclination [degrees]
        self.ecc = 0.0            # eccentricity
        self.w = 90.0             # planet's argument of periastron [degrees]
        self.u = []               # limb-darkening coefficients
        self.limb_dark = None     # "uniform" | "linear" | "quadratic"
        self.fp = None            # planet/star flux ratio (secondary)
        self.t_secondary = None   # unused; secondary timing is computed
        # an oblate planet (hybrid laws only; elome.oblate): projected
        # flattening f = 1 - B/A, with rp then the area-equivalent radius
        # sqrt(A B), and theta the sky angle of the long axis from the
        # direction of motion at transit [degrees]. None = spherical.
        self.f = None
        self.theta = None
        # a ring in the planet's equatorial plane (hybrid laws only;
        # elome.rings): inner and outer radii in units of rp, the ring's
        # projected flattening 1 - |cos i_ring|, and its normal optical
        # depth (None: opaque). theta orients it (and an oblate planet,
        # which shares it). None = no ring.
        self.ring_in = None
        self.ring_out = None
        self.ring_f = None
        self.ring_tau = None
        # starspots (hybrid laws only; elome.spots): a (K, 4) array of
        # longitude, latitude and angular radius [degrees] and contrast
        # (the spot's intensity over the photosphere's; 1 = no spot) per
        # spot, on a star with rotation period prot, spin inclination
        # istar to the line of sight [degrees; None: 90] and sky-projected
        # obliquity lam [degrees; None: 0]. Longitudes are measured at
        # spot_epoch (None: t0) from the meridian facing the observer.
        # The flux is normalised to the unspotted star. None = no spots.
        self.spots = None
        self.prot = None
        self.istar = None
        self.lam = None
        self.spot_epoch = None
        self.spot_u = None        # reserved: a limb-darkening law for spots


def _u_vector(u, sets=False):
    """``params.u`` as ONE 1-D sequence of coefficients, whatever it
    arrived as -- or, with ``sets``, as a (n_sets, N) float64 array.

    A list or tuple of numbers, a numpy array and an mx.array all come
    through here, as does a list holding mx.array scalars (the way to
    differentiate one coefficient). The result is a float64 numpy vector
    for host input, the mx.array itself for a whole-array u, and for the
    mixed list the list of its validated scalars -- each converted to the
    model dtype later, on its own, so no entry borrows a sibling's dtype
    (an int mx scalar used to truncate its Python neighbours to 0).
    Anything that is not exactly one axis raises: a (n, 1) column -- a
    loadtxt slice, or a nested list -- used to pass the coefficient
    *count* and then run flux_dev_poly's batched branch as a wrong-order
    model, and a 0-d mx.array died in list() with an opaque IndexError.
    One container rule in one place is what makes "every container" true.

    ``sets`` is the array-valued ``light_curves`` form, whose u may be
    (N,) for all sets or (n_sets, N) per set; it is returned 2-D.
    """
    if u is None:
        v = np.zeros((1, 0) if sets else 0)
    elif isinstance(u, mx.array):
        v = u
    elif (isinstance(u, (list, tuple))
          and any(isinstance(x, mx.array) for x in u)):
        for x in u:
            if isinstance(x, mx.array):
                _need_scalar(x, "u")
            elif np.ndim(x) != 0:
                raise ValueError("u must be a 1-D vector; got a nested "
                                 f"entry of shape {np.shape(x)}")
        v = list(u)
    else:
        v = np.asarray(u, dtype=np.float64)
    if sets:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2:
            raise ValueError("u must be (N,) or (n_sets, N) here; got shape "
                             f"{tuple(v.shape)}")
        return v
    if np.ndim(v) != 1:
        raise ValueError(f"u must be a 1-D vector; got shape {tuple(np.shape(v))}")
    return v


def _ld_coeffs(params, conv=float, u=None) -> tuple[float, float]:
    """(u1, u2) for the quadratic core, validated. ``conv`` maps each
    value: float for the traced-scalar paths, a graph-preserving cast for
    light_curve_mx's differentiable one. ``u`` is the already-normalised
    vector when the caller has it."""
    law = params.limb_dark
    if u is None:
        u = _u_vector(params.u)
    if law == "uniform":
        if len(u) != 0:
            raise ValueError("uniform limb darkening takes no coefficients")
        return conv(0.0), conv(0.0)
    if law == "linear":
        if len(u) != 1:
            raise ValueError("linear limb darkening takes 1 coefficient")
        return conv(u[0]), conv(0.0)
    if law == "quadratic":
        if len(u) != 2:
            raise ValueError("quadratic limb darkening takes 2 coefficients")
        return conv(u[0]), conv(u[1])
    if law == "polynomial":
        if len(u) == 0:
            raise ValueError("polynomial limb darkening needs >= 1 "
                             "coefficient (use 'uniform' for none)")
        return None, None          # handled by the polynomial core
    hl = _hybrid_law(law)
    if hl is not None:
        if len(u) != hl.n_w:
            raise ValueError(f"{hl.name} takes {hl.n_w} weights; "
                             f"got {len(u)}")
        return None, None          # handled by the hybrid core
    raise ValueError(
        f"limb_dark {law!r} not supported; choose from {_supported_ld()} "
        "or pass a HybridLaw. "
        "Non-polynomial laws (nonlinear, squareroot, logarithmic) are "
        "outside the ALFM19 formulation.")


def _need_scalar(x, name):
    """Every per-set field is one number. A vector would run -- each time
    sample at its own value -- or die deep in a graph with an opaque
    reshape error; a (1,) or (1, 1) would change the output's shape."""
    if x.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {x.shape}")


def _scalar_value(x):
    """The Python float of a 0-d mx.array, or None if it is traced.

    Under mx.compile / mx.vmap an array cannot be read -- MLX raises on
    the attempt -- while eagerly, and under mx.grad (whose inputs are plain
    arrays), it can; the read is free for a leaf and otherwise evaluates
    the upstream graph. With the shape already checked, that refusal is
    the only ValueError float() can raise here. The MLX dependency is this
    one place (pinned by test_light_curve_mx.test_scalar_value_helper).
    """
    try:
        return float(x)
    except ValueError:
        return None


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


_ECC_MSG = "eccentricity must be in [0, 1); got {}"
# an oblate or ringed planet's orbit is rebuilt from the impact parameter,
# whose cos i (anchored.cos_inc_from_b) holds to e = 0.999 (metal._E_MAX)
_ECC_MAX_SHAPED = 0.999
_ECC_MSG_SHAPED = ("eccentricity must be in [0, 0.999] for an oblate or "
                   "ringed planet; got {}")


def _ecc_ok(ecc, shaped):
    return (ecc >= 0.0) & ((ecc <= _ECC_MAX_SHAPED) if shaped else (ecc < 1.0))


def _check_ecc(ecc, shaped=False):
    """Reject eccentricities the model cannot represent, before they reach
    a graph that would return a plausible flat curve (e >= 1) or an
    all-NaN row (e < 0) with nothing pointing at the parameter -- or, for
    an oblate or ringed planet (shaped), a wrong orbit (e > 0.999). A
    Python number is checked as a scalar; the batched path passes an
    array."""
    msg = _ECC_MSG_SHAPED if shaped else _ECC_MSG
    if not isinstance(ecc, np.ndarray):
        if not _ecc_ok(ecc, shaped):
            raise ValueError(msg.format(ecc))
        return
    bad = ~_ecc_ok(ecc, shaped)
    if bool(np.any(bad)):
        raise ValueError(msg.format(f"{ecc[bad][:4]} (and possibly more)"))


_GEOM_RULES = (("rp", "rp must be >= 0", lambda v: v >= 0.0),
               ("a", "a must be > 1 (stellar radii)", lambda v: v > 1.0),
               ("per", "per must be > 0", lambda v: v > 0.0))


def _check_geometry(params):
    """rp >= 0, a > 1, per > 0 (and finite) for the readable fields --
    Python numbers, numpy values, and mx.arrays outside a trace; returns
    the traced mx.array fields, which cannot be read (mx.compile / vmap)
    and are masked to NaN in the graph instead (_geometry_mask). A
    negative rp otherwise returns a finite, deeper, wrong transit."""
    traced = []
    for name, msg, ok in _GEOM_RULES:
        x = getattr(params, name)
        if isinstance(x, mx.array):
            v = _scalar_value(x)
            if v is None:
                traced.append(name)
                continue
        else:
            v = x
        vals = np.atleast_1d(np.asarray(v, dtype=np.float64))
        bad = ~(np.isfinite(vals) & ok(vals))
        if bool(np.any(bad)):
            raise ValueError(f"{msg}; got {vals[bad][:4]}")
    return traced


def _check_geometry_cols(cols):
    """_check_geometry's rules on light_curves' (n_sets,) columns."""
    for name, msg, ok in _GEOM_RULES:
        vals = cols[name]
        bad = ~(np.isfinite(vals) & ok(vals))
        if bool(np.any(bad)):
            raise ValueError(f"{msg}; got {vals[bad][:4]}")


def _geometry_mask(traced, values, dtype):
    """1.0, or NaN where a traced field breaks its rule (_check_geometry)."""
    ok = None
    rules = {name: rule for name, _, rule in _GEOM_RULES}
    for name in traced:
        v = values[name]
        c = mx.logical_and(rules[name](v), v == v)
        ok = c if ok is None else mx.logical_and(ok, c)
    return mx.where(ok, 1.0, float("nan")).astype(dtype)


def _ld_coeffs_batch(law, u):
    """(u1, u2) columns for the legacy laws, from a (n_sets, N) array."""
    if law == "uniform":
        z = np.zeros(u.shape[0])
        return z, z
    if law == "linear":
        return u[:, 0], np.zeros(u.shape[0])
    if law == "quadratic":
        return u[:, 0], u[:, 1]
    raise ValueError(f"unexpected law {law!r} in the batched path")


def _unpack_ld(vec, tail):
    """(u1, u2, uvec, fp) from the trailing limb-darkening + fp args."""
    if vec:
        uv, fp = tail
        return None, None, uv, fp
    u1, u2, fp = tail
    return u1, u2, None, fp


class TransitModel:
    """Precomputes the (super)sampled time grid; ``light_curve(params)``
    evaluates the model for (possibly updated) parameters, batman-style.

    Args:
        params: TransitParams (validated here; values may change between
            light_curve calls, but the limb-darkening *law* and the
            transit type are fixed per model).
        t: times, same units as params.t0/per (absolute values are fine —
            computation is float64 on the CPU by default).
        transittype: "primary" or "secondary".
        supersample_factor: average this many samples over exp_time.
        exp_time: exposure time (same units as t).
        dtype: mx.float64 (default; CPU, exact) or mx.float32 (the GPU
            kernels). In float32 the times are re-centred on the host in
            float64 first, so absolute mission times are fine; over
            hundreds of orbits the float32 phase loses precision.
        integration: "supersample" (batman's rule) or "contact"
            (Gauss-Legendre on windows split at the contact times).
        n_gl: Gauss-Legendre nodes per piece for "contact".
    """

    def __init__(self, params, t, transittype: str = "primary",
                 supersample_factor: int = 1, exp_time: float = 0.0,
                 dtype=None, use_metal: bool = True,
                 integration: str = "supersample", n_gl: int = 7):
        _ld_coeffs(params)  # validate law/coefficients early
        if transittype not in ("primary", "secondary"):
            raise ValueError("transittype must be 'primary' or 'secondary'")
        if transittype == "secondary" and params.fp is None:
            raise ValueError("secondary eclipse needs params.fp")
        if supersample_factor > 1 and exp_time <= 0.0:
            raise ValueError("supersampling needs exp_time > 0")
        if integration not in ("supersample", "contact"):
            raise ValueError('integration must be "supersample" or "contact"')
        if integration == "contact":
            if exp_time <= 0.0:
                raise ValueError("contact integration needs exp_time > 0")
            if transittype != "primary":
                raise ValueError("contact integration is primary-transit only")
            if supersample_factor > 1:
                raise ValueError("supersample_factor is for "
                                 "integration='supersample'; the contact rule "
                                 "sets its accuracy with n_gl")
            if int(n_gl) < 1:
                raise ValueError(f"n_gl must be >= 1; got {n_gl}")
        self.integration = integration
        self.n_gl = int(n_gl)
        self.transittype = transittype
        # oblate planet: fixed per model, like the law (its own graphs)
        self._oblate = _is_oblate(params)
        self._ringed = _is_ringed(params)
        self._spotted = _is_spotted(params)
        self._shaped = self._oblate or self._ringed
        self._batch_keys = self._BATCH_KEYS + (
            ("f", "theta") + _RING_FIELDS if self._ringed
            else ("f", "theta") if self._oblate
            else _SPOT_FIELDS if self._spotted else ())
        self._n_spots = 0
        if self._spotted:
            if self._shaped:
                raise ValueError("spots are for a spherical planet: leave "
                                 "params.f and params.ring_in None")
            if _hybrid_law(params.limb_dark) is None:
                raise ValueError("a spotted star (params.spots) takes a "
                                 "hybrid law: limb_dark='hybrid2' | "
                                 "'hybrid4' | 'hybrid5'")
            if transittype != "primary":
                raise ValueError("a spotted star is primary-transit only")
            if params.spot_u is not None:
                raise ValueError("params.spot_u is reserved; spots take the "
                                 "photosphere's law times their contrast")
            self._n_spots = _spot_shape(params.spots)[-2]
        if self._ringed:
            if _hybrid_law(params.limb_dark) is None:
                raise ValueError("a ringed planet (params.ring_in) takes a "
                                 "hybrid law: limb_dark='hybrid2' | "
                                 "'hybrid4' | 'hybrid5'")
            if transittype != "primary":
                raise ValueError("a ringed planet is primary-transit only")
        elif self._oblate:
            if _hybrid_law(params.limb_dark) is None:
                raise ValueError("an oblate planet (params.f) takes a hybrid "
                                 "law: limb_dark='hybrid2' | 'hybrid4' | "
                                 "'hybrid5'")
            if transittype != "primary":
                raise ValueError("an oblate planet is primary-transit only")
        self.limb_dark = params.limb_dark
        # a hybrid law is resolved ONCE, here: every graph this model traces
        # uses this definition, whatever the registry does afterwards
        self._hlaw = _hybrid_law(params.limb_dark)
        # vector laws (polynomial, hybrid): the coefficient count is fixed
        # per model, baked into the compiled graph
        self._n_vec = (len(_u_vector(params.u))
                       if _is_vector_law(params.limb_dark) else 0)
        self.dtype = mx.float64 if dtype is None else dtype
        self._stream = mx.cpu if self.dtype == mx.float64 else None
        # fused Metal kernel for fp32 GPU evaluation (falls back on its
        # own for fp64/CPU, so leaving this True is always safe)
        self._use_metal = bool(use_metal)

        t = np.asarray(t, dtype=np.float64)
        self.t = t
        self.supersample_factor = int(supersample_factor)
        self.exp_time = float(exp_time)
        if self.supersample_factor > 1 and integration == "supersample":
            n = self.supersample_factor
            # batman convention: endpoint-inclusive uniform samples
            offs = np.linspace(-0.5 * self.exp_time, 0.5 * self.exp_time, n)
            t_super = (t[:, None] + offs[None, :]).ravel()
        else:
            t_super = t
        self._t_super = t_super
        # (6) Absolute mission time stamps cannot survive float32: a TESS
        # BTJD of ~2500 d has an fp32 ulp of 2.4e-4 d (21 s), Kepler's
        # BJD-2454833 is similar and raw BJD (2.457e6 d) has a 0.25 d ulp.
        # So for any non-float64 dtype the grid is re-centred on a float64
        # reference HERE, on the host, and t0 is shifted by the same
        # amount at every call site -- exactly what epoch_center_times does
        # for the anvil path. The fp32 graph then only ever sees O(baseline)
        # numbers. float64 keeps a zero offset so its heavily-tested
        # round-off behaviour is untouched.
        self._t_ref = (0.0 if self.dtype == mx.float64
                       else float(0.5 * (t_super.min() + t_super.max())))
        self._t_mx = mx.array(t_super - self._t_ref, dtype=self.dtype)
        self._compiled = {}

    @property
    def use_metal(self) -> bool:
        """Whether float32 evaluation may use the fused Metal kernels."""
        return self._use_metal

    @use_metal.setter
    def use_metal(self, value):
        # the choice is baked into graphs as they are traced (a kernel or
        # the graph core, per law and orbit), so a change drops them all
        value = bool(value)
        if value != self._use_metal:
            self._compiled = {}
        self._use_metal = value

    def _require(self, get, names):
        """Raise for a parameter left as None: batched or not, a missing
        field must not become a plausible light curve."""
        for name in names:
            if get(name) is None:
                raise ValueError(f"params.{name} is not set")

    def _required_fields(self):
        names = ["t0", "per", "rp", "a", "inc", "ecc", "w"]
        if self.transittype == "secondary":
            names.append("fp")
        if self._oblate:
            names.append("f")
        if self._ringed:
            names += ["ring_in", "ring_out", "ring_f"]
        if self._spotted:
            names.append("prot")
        return names

    # -- internals ---------------------------------------------------------
    #
    # light_curve runs through mx.compile'd graphs (kernel fusion is
    # worth ~2x+ over eager MLX here). Parameters enter as *traced 0-d
    # arrays*, so batman-style parameter updates between calls reuse the
    # same compiled kernels — only the circular/eccentric branch (a
    # Python-level dispatch on ecc == 0) selects between two graphs,
    # built lazily and cached per model.

    def _photom(self, z, front, rp, u1, u2, fp, uvec=None):
        if self.transittype == "primary":
            # far side pushed past 1 + rp for any rp (2 + z cleared the
            # star only for rp <= 1)
            z_eff = mx.where(front, z, z + 2.0 * (1.0 + rp))
            if uvec is not None:          # a vector law: polynomial or hybrid
                if self.limb_dark == "polynomial":
                    return 1.0 + flux_dev_poly(z_eff, rp, uvec,
                                               n_max=self._n_vec)
                return 1.0 + self._hybrid_dev(z_eff, rp, uvec)
            # use_metal=False: the analytic core (flux_dev_metal's own
            # fallback), not flux_dev, whose reverse mode is NaN at r = 0
            # and at r = 1, z = 0
            core = flux_dev_metal if self.use_metal else flux_dev_analytic
            if core is flux_dev_metal and z_eff.ndim == 3:
                # light_curves' contact rule: (n_sets, m, nodes) with (n, 1, 1)
                # parameters. The kernel takes (n, m) with a value per row,
                # so fold the node axis into the points (as _hybrid_dev does)
                n = z_eff.shape[0]

                def row(c):              # (n, 1): the kernel and the graph
                    return (mx.reshape(c, (n, 1)) if isinstance(c, mx.array)
                            and c.size == n else c)
                dev = core(mx.reshape(z_eff, (n, -1)), row(rp), row(u1),
                           row(u2))
                return 1.0 + mx.reshape(dev, z_eff.shape)
            return 1.0 + core(z_eff, rp, u1, u2)
        z_eff = mx.where(front, z + 2.0 * (1.0 + rp), z)
        s0d, _, _ = sn_dev(z_eff, rp)
        # visible fraction of the (uniform) planet disk; rp = 0 is a point
        # planet, always fully visible (0/0 otherwise; the safe denominator
        # keeps the gradient finite there too)
        if isinstance(rp, (int, float)):
            if rp == 0.0:
                return 1.0 + fp + 0.0 * z
            return 1.0 + fp * (1.0 + s0d / (math.pi * rp * rp))
        pos = rp > 0.0
        den = mx.where(pos, math.pi * rp * rp, 1.0)
        return 1.0 + fp * mx.where(pos, 1.0 + s0d / den, 1.0)

    def _hybrid_dev(self, z, rp, uvec):
        """F - 1 for a hybrid law on separations ``z``: the fused z-input
        kernel (metal_hybrid) wherever it can run -- the role flux_dev_metal
        plays for the quadratic law -- and hybrid.py's graph otherwise.

        The decision is _kernel_usable's, taken here, inside the trace,
        from the active stream (mx.compile keeps one trace per stream, so
        a graph first traced on the CPU stream is retraced, kernel and
        all, on the GPU; no cache key is needed). An fp32 model on the CPU
        stream, a Metal-less machine and use_metal=False therefore run one
        and the same graph, bitwise.

        The kernel takes (n, m) points with r and the weights per row, so
        the shapes are flattened onto that: one row for a single parameter
        set (any node axes folded into m), n rows for light_curves' sets.
        r goes through as is: the entry point's canonicaliser takes a
        scalar or anything of size n (light_curves' contact path hands it
        (n, 1, 1)). One radius with one set of weights -- (n_w,) or a
        (1, n_w) row -- is one row of the grid, never a row per point.
        """
        law = self._hlaw
        if not self._kernel_usable():
            return flux_dev_hybrid(z, rp, uvec, law)
        n_w = law.n_w
        one_set = ((not isinstance(rp, mx.array) or rp.size == 1)
                   and uvec.size == n_w)
        if one_set:                        # (n_w,) or a (1, n_w) row
            out = flux_dev_metal_hybrid(mx.reshape(z, (-1,)), rp, law,
                                        mx.reshape(uvec, (-1,)))
        else:
            n = z.shape[0]
            w = (mx.reshape(uvec, (uvec.shape[0], -1)) if uvec.ndim >= 2
                 else uvec)
            out = flux_dev_metal_hybrid(mx.reshape(z, (n, -1)), rp, law, w)
        return mx.reshape(out, z.shape)

    def _kernel_usable(self) -> bool:
        """Can a fused kernel run *here*: fp32, a usable Metal device, the
        GPU stream active, and not switched off. Asked before tracing by
        _get_compiled for the fused *model* kernel (whose reach -- primary
        transits, quadratic limb darkening, no contact rule -- is stated
        beside the branches it governs, and whose answer is in the cache
        key), and inside the trace by _hybrid_dev for the hybrid z-kernel.
        """
        if not self.use_metal:
            return False
        if self.dtype != mx.float32:
            return False
        return metal_available() and _gpu_stream_active()

    def _get_compiled(self, circular: bool, ew: bool = False):
        """The compiled model graph. ``ew`` selects, for the eccentric
        graphs, (e, w [rad]) inputs in place of (k, h) -- light_curve_mx's
        differentiable route, exact at e = 0 (anchored.anchor_constants_ew).
        With ew=False every graph is exactly what it always was."""
        vec = bool(self._n_vec)
        # The kernel decision depends on the *active* stream, so it is part
        # of the key: a graph first built under the CPU stream (an fp64
        # gradient, say) must not be the one every later GPU call reuses.
        # It enters the key only where a kernel branch below is reachable,
        # so a graph the kernel never serves is compiled once, not once
        # per stream. This is THE statement of what the *model* kernel
        # serves: primary transits with quadratic limb darkening, on the
        # (k, h) or circular graph, without the contact rule. Every other
        # fp32 graph reaches a z-input kernel (quadratic or hybrid) through
        # _photom, decided inside the trace, which mx.compile keeps per
        # stream: no key needed.
        kernel_branch = (not ew and not vec and self.transittype == "primary"
                         and self.integration != "contact")
        kern = kernel_branch and self._kernel_usable()
        key = "ew" if ew else (circular, kern)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        t = self._t_mx
        if ew:
            circular = False

        def consts(k, h):
            """Keyword for the anchored helpers: nothing in (k, h) mode,
            so those graphs are unchanged; the (e, w) constants else."""
            return dict(consts=anchor_constants_ew(k, h)) if ew else {}

        if self.integration == "contact":
            # Node times depend on the parameters (the contacts move), so
            # the grid is rebuilt inside the compiled graph each call and
            # `t` stays the exposure mid-times.
            n_gl = self.n_gl
            ex = self.exp_time

            def _avg(z, front, rp, u1, u2, fp, uv, w):
                f = self._photom(z, front, rp, u1, u2, fp, uvec=uv)
                return mx.sum(f * w, axis=1)

            if circular:
                def raw(t0, per, a, b, rp, *ld_fp):
                    u1, u2, uv, fp = _unpack_ld(vec, ld_fp)
                    ci = b / a
                    cs = contact_offsets(rp, a, b)
                    T, W = exposure_nodes(t, t0, per, ex, cs, n_gl,
                                          dtype=self.dtype)
                    phase = (2.0 * math.pi) * (T - t0) / per
                    sphi, cphi = sincos(phase)
                    z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                           1e-24))
                    return _avg(z, cphi > 0.0, rp, u1, u2, fp, uv, W)
            else:
                def raw(t0, per, a, k, h, ci, rp, *ld_fp):
                    u1, u2, uv, fp = _unpack_ld(vec, ld_fp)
                    kw = consts(k, h)
                    if ew:
                        e, esw = kw["consts"][0], kw["consts"][2]
                    else:
                        e = k * k + h * h
                        esw = h * mx.sqrt(mx.maximum(e, 1e-30))  # e sin w
                    b_conj = contact_geometry(a, e, esw, ci)[1]
                    cs = contact_offsets_anchored(rp, a, b_conj, k, h, ci,
                                                  **kw)
                    T, W = exposure_nodes(t, t0, per, ex, cs, n_gl,
                                          dtype=self.dtype)
                    phi = (2.0 * math.pi) * (T - t0) / per
                    z, front = separation_anchored(phi, k, h, a, ci, **kw)
                    return _avg(z, front, rp, u1, u2, fp, uv, W)

        elif circular and vec:
            def raw(t0, per, a, b, rp, uv, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, None, None, fp,
                                    uvec=uv)
        elif vec:
            def raw(t0, per, a, k, h, ci, rp, uv, fp):
                phi = (2.0 * math.pi) * (t - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci,
                                               **consts(k, h))
                return self._photom(z, front, rp, None, None, fp, uvec=uv)
        elif circular and kern:
            # Circular orbit on the SAME fused kernel as the eccentric one:
            # k = h = 0 and cos i = b / a. Exact (the anchored orbit
            # degenerates to the circular one) and the kernel skips the
            # Kepler solve on e == 0 chains. See the eccentric branch below
            # for why period_ref is 0 and the period rides p_off.
            from .anchored import pack_orbit_constants
            from .metal import make_model_core_metal
            core = make_model_core_metal(0.0)
            m = t.shape[0]
            xdat = mx.stack([t, mx.zeros_like(t)])

            def raw(t0, per, a, b, rp, u1, u2, fp):
                def col(v):
                    return mx.reshape(v, (1,))
                zero = mx.zeros((1,), dtype=self.dtype)
                orb = pack_orbit_constants(zero, zero, col(b / a))
                dev = core(xdat, col(t0), col(per), col(rp), col(a), orb,
                           col(u1), col(u2))
                return 1.0 + mx.reshape(dev, (m,))
        elif circular:
            def raw(t0, per, a, b, rp, u1, u2, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, u1, u2, fp)
        elif kern and not ew:
            # Whole eccentric model in one kernel. (Not on the (e, w) route:
            # the kernel skips its seven eccentric-only gradient slots on
            # e == 0 chains, which is exact for (k, h) -- whose Jacobian
            # vanishes there -- but not for (e, w), where d(e cos w)/de =
            # cos w. The graph below carries every slot.) The anchored Kepler
            # solve as graph ops streams a lot of intermediates: measured
            # 0.23 Gpt/s against the kernel's 2.3, so this is ~10x at
            # large N.
            #
            # period_ref is baked into the kernel factory as a constant,
            # which would defeat batman-style parameter updates — so it
            # is set to zero and the period is carried by the *traced*
            # p_off input instead. With the epoch column k = 0 that is
            # exactly equivalent, gradients included: the kernel's
            # dphi/dp_off = 2 pi ((-k - n_w) P - tau_w) / P^2 reduces to
            # dphi/dP for k = 0.
            from .anchored import pack_orbit_constants
            from .metal import make_model_core_metal
            core = make_model_core_metal(0.0)
            m = t.shape[0]
            xdat = mx.stack([t, mx.zeros_like(t)])

            def raw(t0, per, a, k, h, ci, rp, u1, u2, fp):
                def col(v):
                    return mx.reshape(v, (1,))
                orb = pack_orbit_constants(col(k), col(h), col(ci))
                dev = core(xdat, col(t0), col(per), col(rp), col(a), orb,
                           col(u1), col(u2))
                return 1.0 + mx.reshape(dev, (m,))
        else:
            # transit-anchored: phi is measured straight from t0, so no
            # mean-anomaly-at-transit offset is needed, and float32 stays
            # accurate all the way to e = 0 (see anchored.py).
            def raw(t0, per, a, k, h, ci, rp, u1, u2, fp):
                phi = (2.0 * math.pi) * (t - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci,
                                               **consts(k, h))
                return self._photom(z, front, rp, u1, u2, fp)

        fn = mx.compile(raw)
        self._compiled[key] = fn
        return fn

    def _model_eval(self, params, uv) -> mx.array:
        """One parameter set -> flux, through the compiled graphs. The one
        path behind light_curve and light_curve_mx.

        A field holding an mx.array stays in the graph (gradients flow); a
        Python number is folded on the host in fp64 and cast once, so with
        all-Python fields the arguments -- b = a cos i, (k, h) =
        sqrt(e) (cos w, sin w) -- are formed exactly as they always were.

        Routing follows ``ecc``: a Python 0 takes the circular graph, a
        Python e > 0 the (k, h) graph (smooth in w for fixed e), an
        mx.array the (e, w) graph, which alone is exact in d/de at e = 0
        (anchored.anchor_constants_ew) and never uses the fused *model*
        kernel (its photometry still takes the z-input kernel on an fp32
        GPU model).
        """
        dt = self.dtype

        def is_arr(x):
            return isinstance(x, mx.array)

        # Shapes first, once, before any routing: every per-set field is
        # one number (``uv`` is u already normalised by _check_law). A
        # route that never reads a field (w on a circular orbit) still
        # rejects a bad one.
        for name in self._batch_keys:
            x = getattr(params, name)
            if is_arr(x):
                _need_scalar(x, name)

        def cast(x):
            if is_arr(x):
                if x.dtype == dt:
                    return x
                with mx.stream(mx.cpu):   # an fp64 value may not touch Metal
                    return x.astype(dt)
            if isinstance(x, np.ndarray):          # a host u vector
                return mx.array(x, dtype=dt)
            if isinstance(x, list):                # mixed u: entry by entry
                return mx.stack([cast(y) for y in x])
            return mx.array(float(x), dtype=dt)

        def rad(x):                       # degrees -> radians
            if is_arr(x):
                return cast(x) * (math.pi / 180.0)
            return cast(math.radians(float(x)))   # on the host, in fp64

        self._require(lambda k: getattr(params, k), self._required_fields())
        if self._spotted:
            return self._spot_eval(params, uv, cast, rad, is_arr)
        if self._ringed:
            return self._ring_eval(params, uv, cast, rad, is_arr)
        if self._oblate:
            return self._oblate_eval(params, uv, cast, rad, is_arr)

        t0_off, per, a, rp, ci, ci_py, gmask = self._prep(params, cast, rad,
                                                          is_arr)
        fp = cast(0.0 if params.fp is None else params.fp)
        if self._n_vec:
            ld = (cast(uv),)
        else:
            ld = _ld_coeffs(params, conv=cast, u=uv)

        route = self._ecc_route(params, cast, rad, is_arr)
        if route[0] == "ew":
            _, e, wr, emask = route
            f = self._get_compiled(False, ew=True)(
                t0_off, per, a, e, wr, ci, rp, *ld, fp)
        elif route[0] == "circ":                      # w is irrelevant
            emask = None
            b = (a * ci if is_arr(params.a) or is_arr(params.inc)
                 else cast(float(params.a) * ci_py))
            f = self._get_compiled(True)(t0_off, per, a, b, rp, *ld, fp)
        else:
            _, k, h, _, _ = route
            emask = None
            f = self._get_compiled(False)(t0_off, per, a, k, h, ci, rp,
                                          *ld, fp)
        return self._apply_masks(f, emask, gmask)

    # -- parameter preparation shared by the spherical and oblate paths ---

    def _prep(self, params, cast, rad, is_arr):
        """(t0_off, per, a, rp, cos i, cos i as a Python float or None,
        geometry mask or None). Python fields fold on the host in fp64 and
        cast once; mx.array fields stay in the graph. Readable geometry
        is validated (_check_geometry); traced fields get a NaN mask."""
        dt = self.dtype
        traced = _check_geometry(params)
        t0 = params.t0
        if is_arr(t0):
            # the reference-time subtraction in fp64, then the model dtype
            with mx.stream(mx.cpu):
                t0_off = (t0.astype(mx.float64) - self._t_ref).astype(dt)
        else:
            t0_off = cast(t0 - self._t_ref)
        inc = params.inc
        if is_arr(inc):
            ci, ci_py = sincos(rad(inc))[1], None   # fp64-accurate (MLX's cos is not)
        else:
            ci_py = math.cos(math.radians(float(inc)))
            ci = cast(ci_py)
        per, a, rp = cast(params.per), cast(params.a), cast(params.rp)
        gmask = (_geometry_mask(traced, {"rp": rp, "a": a, "per": per}, dt)
                 if traced else None)
        return t0_off, per, a, rp, ci, ci_py, gmask

    def _ecc_route(self, params, cast, rad, is_arr):
        """How the orbit enters the graph, validated:
          ("ew", e, w [rad], NaN mask or None)  an mx.array ecc: (e, w),
              exact in d/de at e = 0 (anchored.anchor_constants_ew);
          ("circ",)                              a Python e = 0;
          ("kh", k, h, e, sqrt(e))               a Python e > 0: (k, h) =
              sqrt(e) (cos w, sin w), smooth in w for fixed e.
        An array e is checked when readable; traced it cannot be, and an
        out-of-range value makes the output (and gradients) NaN instead --
        validated as the value the GRAPH sees (an fp64 e just below 1 can
        round to 1 in fp32)."""
        ecc, w = params.ecc, params.w
        if is_arr(ecc):
            e = cast(ecc)
            ev = _scalar_value(e)
            if ev is not None:
                _check_ecc(ev, self._shaped)
                return "ew", e, rad(w), None
            ok = _ecc_ok(e, self._shaped)
            return "ew", e, rad(w), mx.where(ok, 1.0, float("nan")).astype(
                self.dtype)
        ev = float(ecc)
        _check_ecc(ev, self._shaped)
        if ev == 0.0:
            return ("circ",)
        sq = math.sqrt(ev)
        if is_arr(w):
            sw, cw = sincos(rad(w))
            k, h = sq * cw, sq * sw
        else:
            wr = math.radians(float(w))
            k, h = cast(sq * math.cos(wr)), cast(sq * math.sin(wr))
        return "kh", k, h, ev, sq

    @staticmethod
    def _apply_masks(f, *masks):
        for m in masks:
            if m is not None:
                f = f * m
        return f

    # -- oblate planets ------------------------------------------------------
    #
    # One route for every surface: time since each point's own mid-transit,
    # then metal.flux_dev_from_tau(f=, theta=) -- the oblate Metal kernels
    # on an fp32 GPU model, the exact graph otherwise -- with this model's
    # exposure rule (supersampling stays this model's own grid and
    # average). The spherical graphs above are untouched.

    def _oblate_fn(self, orbit: str):
        """The compiled oblate model. ``orbit``: "circ"; "kh", (k, h) =
        sqrt(e) (cos w, sin w) from a Python ecc; "ew", (e, w [rad]) for an
        array ecc -- exact in d/de at e = 0, as _model_eval's (e, w) graph
        (oblate_tau._orbit_consts; its kernels keep every eccentric slot).
        The kernel/graph decision is flux_dev_from_tau_oblate's, inside the
        trace (mx.compile keeps one per stream), with this model's
        use_metal."""
        key = ("oblate", orbit)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        from . import oblate_tau
        from .metal import _INT_CONTACT, _INT_NONE
        from .oblate_tau import flux_dev_from_tau_oblate
        contact = self.integration == "contact"
        mode = _INT_CONTACT if contact else _INT_NONE
        exp_time = self.exp_time if contact else 0.0
        law, n_gl, use_metal = self._hlaw, self.n_gl, self.use_metal

        def flux(tt, t0, per, a, b, rp, f, th, uv, k=None, h=None,
                 blocks=False):
            tau = tt - t0
            tau = tau - per * mx.round(tau / per)
            return 1.0 + flux_dev_from_tau_oblate(
                tau, per, a, b, rp, f, th, law, uv, exp_time, mode, n_gl, 1,
                k=k, h=h, ew=orbit == "ew", use_metal=use_metal,
                compile_blocks=blocks)
        if orbit == "circ":
            def raw(tt, t0, per, a, b, rp, f, th, uv, blocks=False):
                return flux(tt, t0, per, a, b, rp, f, th, uv, blocks=blocks)
        else:
            def raw(tt, t0, per, a, b, rp, f, th, uv, k, h, blocks=False):
                return flux(tt, t0, per, a, b, rp, f, th, uv, k, h,
                            blocks=blocks)
        # An fp32 model with use_metal=False runs the oblate *graph* on the
        # GPU, and that graph is too large for one fused Metal primitive:
        # MLX's compile exhausts the kernel's argument buffers and aborts
        # the process. So that one configuration runs uncompiled (the
        # kernels, and every graph on the CPU stream, compile normally).
        if self.dtype == mx.float32 and not use_metal:
            fn = raw
        else:
            whole = mx.compile(raw)
            nodes = oblate_tau._nodes_per_time(exp_time, mode, n_gl, 1)

            def fn(tt, t0, *rest):
                # The graph (fp64) runs in blocks of _OBLATE_BLOCK points
                # (oblate_tau._in_blocks); compiling a blocked call whole
                # traces every block, so a call larger than one block runs
                # uncompiled around a compiled block instead.
                n = max([x.shape[0] for x in (tt, t0, *rest[:6])
                         if x.ndim == 2] or [1])
                if (self.dtype == mx.float64 and n * tt.shape[-1] * nodes
                        > oblate_tau._OBLATE_BLOCK):
                    return raw(tt, t0, *rest, blocks=True)
                return whole(tt, t0, *rest)
        self._compiled[key] = fn
        return fn

    def _oblate_eval(self, params, uv, cast, rad, is_arr):
        """light_curve / light_curve_mx for an oblate model: _model_eval's
        parameter preparation and ecc routing, b the impact parameter at
        conjunction from exposure.contact_geometry."""
        from .oblate import _check_domain
        _check_domain(params.rp, params.f)
        t0_off, per, a, rp, ci, ci_py, gmask = self._prep(params, cast, rad,
                                                          is_arr)
        f = cast(params.f)
        th = rad(0.0 if params.theta is None else params.theta)
        head = (self._t_mx, t0_off, per, a)
        route = self._ecc_route(params, cast, rad, is_arr)
        if route[0] == "ew":
            _, e, wr, emask = route
            b = contact_geometry(a, e, e * sincos(wr)[0], ci)[1]
            out = self._oblate_fn("ew")(*head, b, rp, f, th, cast(uv), e, wr)
        elif route[0] == "circ":
            emask = None
            b = (a * ci if is_arr(params.a) or is_arr(params.inc)
                 else cast(float(params.a) * ci_py))
            out = self._oblate_fn("circ")(*head, b, rp, f, th, cast(uv))
        else:
            _, k, h, ev, sq = route
            emask = None
            b = contact_geometry(a, cast(ev), h * sq, ci)[1]   # e sin w = h sqrt(e)
            out = self._oblate_fn("kh")(*head, b, rp, f, th, cast(uv), k, h)
        return self._apply_masks(out, emask, gmask)

    # -- ringed planets -------------------------------------------------------
    #
    # The oblate route's structure, with rings_tau's graph: time since each
    # point's own mid-transit, then metal.flux_dev_from_tau(ring_in=, ...).
    # An fp32 GPU model takes the ring Metal kernels (metal_rings); fp64 and
    # use_metal=False the graph -- not compiled whole: MLX 0.32's mx.compile
    # fails on it (an internal "unordered_map::at" error once the orbit and
    # the ring's columns are in one graph). On the CPU the columns alone
    # run compiled (rings._shape_cols_compiled), and the graph is built in
    # blocks of a bounded number of points (oblate_tau._in_blocks).

    def _ring_fn(self, orbit: str):
        """The ringed-planet model (not compiled whole: see above);
        ``orbit`` as _oblate_fn's."""
        key = ("ring", orbit)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        from .metal import _INT_CONTACT, _INT_NONE
        from .rings_tau import flux_dev_from_tau_ring
        contact = self.integration == "contact"
        mode = _INT_CONTACT if contact else _INT_NONE
        exp_time = self.exp_time if contact else 0.0
        law, n_gl, use_metal = self._hlaw, self.n_gl, self.use_metal

        def flux(tt, t0, per, a, b, rp, f, th, ri, ro, rf, rt, uv, k=None, h=None):
            tau = tt - t0
            tau = tau - per * mx.round(tau / per)
            return 1.0 + flux_dev_from_tau_ring(
                tau, per, a, b, rp, f, th, ri, ro, rf, rt, law, uv, exp_time,
                mode, n_gl, 1, k=k, h=h, ew=orbit == "ew", use_metal=use_metal)
        if orbit == "circ":
            def raw(tt, t0, per, a, b, rp, f, th, ri, ro, rf, rt, uv):
                return flux(tt, t0, per, a, b, rp, f, th, ri, ro, rf, rt, uv)
        else:
            def raw(tt, t0, per, a, b, rp, f, th, ri, ro, rf, rt, uv, k, h):
                return flux(tt, t0, per, a, b, rp, f, th, ri, ro, rf, rt, uv, k, h)
        self._compiled[key] = raw
        return raw

    def _ring_eval(self, params, uv, cast, rad, is_arr):
        """light_curve / light_curve_mx for a ringed model, as _oblate_eval."""
        from .rings import _check_ring
        f = 0.0 if params.f is None else params.f
        tau = math.inf if params.ring_tau is None else params.ring_tau
        _check_ring(f, params.ring_in, params.ring_out, params.ring_f, tau)
        t0_off, per, a, rp, ci, ci_py, gmask = self._prep(params, cast, rad,
                                                          is_arr)
        ring = (cast(f), rad(0.0 if params.theta is None else params.theta),
                cast(params.ring_in), cast(params.ring_out),
                cast(params.ring_f), cast(tau))
        head = (self._t_mx, t0_off, per, a)
        route = self._ecc_route(params, cast, rad, is_arr)
        if route[0] == "ew":
            _, e, wr, emask = route
            b = contact_geometry(a, e, e * sincos(wr)[0], ci)[1]
            out = self._ring_fn("ew")(*head, b, rp, *ring, cast(uv), e, wr)
        elif route[0] == "circ":
            emask = None
            b = (a * ci if is_arr(params.a) or is_arr(params.inc)
                 else cast(float(params.a) * ci_py))
            out = self._ring_fn("circ")(*head, b, rp, *ring, cast(uv))
        else:
            _, k, h, ev, sq = route
            emask = None
            b = contact_geometry(a, cast(ev), h * sq, ci)[1]
            out = self._ring_fn("kh")(*head, b, rp, *ring, cast(uv), k, h)
        return self._apply_masks(out, emask, gmask)

    # -- spotted stars ---------------------------------------------------------
    #
    # The ring route's structure, with spots_tau's graph, which takes the
    # absolute times as well (the star turns): the planet sees the time
    # since each point's own mid-transit, the spots the time since their
    # epoch. Not compiled whole (as the ring graph); the graph is built in
    # blocks of a bounded number of points (oblate_tau._in_blocks).

    def _spot_fn(self, orbit: str):
        """The spotted-star model; ``orbit`` as _oblate_fn's."""
        key = ("spots", orbit)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        from .spots_tau import flux_dev_spots_t
        law, n_gl = self._hlaw, self.n_gl
        exp_time = self.exp_time if self.integration == "contact" else 0.0

        def raw(tt, t0, ep, per, a, b, rp, prot, istar, lam, sp, uv, k=None, h=None):
            return 1.0 + flux_dev_spots_t(tt, t0, ep, per, a, b, rp, prot, istar, lam,
                                          sp, law, uv, k=k, h=h, ew=orbit == "ew",
                                          exp_time=exp_time, n_gl=n_gl)
        self._compiled[key] = raw
        return raw

    def _spot_array(self, spots, cast):
        """params.spots -> (..., K, 4) in the model dtype, angles in
        radians; host values validated."""
        from .spots_tau import _check_spots
        shape = _spot_shape(spots)
        if shape[-2] != self._n_spots:
            raise ValueError(f"the number of spots changed since model "
                             f"construction ({self._n_spots} -> {shape[-2]}); "
                             f"build a new TransitModel")
        scale = np.array([math.pi / 180.0] * 3 + [1.0])
        if isinstance(spots, mx.array):
            return cast(spots) * mx.array(scale, dtype=self.dtype)
        s = np.asarray(spots, dtype=np.float64) * scale
        _check_spots(s)
        return mx.array(s, dtype=self.dtype)

    def _spot_eval(self, params, uv, cast, rad, is_arr):
        """light_curve / light_curve_mx for a spotted star, as _ring_eval."""
        t0_off, per, a, rp, ci, ci_py, gmask = self._prep(params, cast, rad,
                                                          is_arr)
        ep = params.spot_epoch
        if ep is None:
            ep_off = t0_off
        elif is_arr(ep):
            with mx.stream(mx.cpu):
                ep_off = (ep.astype(mx.float64) - self._t_ref).astype(self.dtype)
        else:
            ep_off = cast(ep - self._t_ref)
        star = (cast(params.prot), rad(90.0 if params.istar is None else params.istar),
                rad(0.0 if params.lam is None else params.lam),
                self._spot_array(params.spots, cast))
        head = (self._t_mx, t0_off, ep_off, per, a)
        route = self._ecc_route(params, cast, rad, is_arr)
        if route[0] == "ew":
            _, e, wr, emask = route
            b = contact_geometry(a, e, e * sincos(wr)[0], ci)[1]
            out = self._spot_fn("ew")(*head, b, rp, *star, cast(uv), e, wr)
        elif route[0] == "circ":
            emask = None
            b = (a * ci if is_arr(params.a) or is_arr(params.inc)
                 else cast(float(params.a) * ci_py))
            out = self._spot_fn("circ")(*head, b, rp, *star, cast(uv))
        else:
            _, k, h, ev, sq = route
            emask = None
            b = contact_geometry(a, cast(ev), h * sq, ci)[1]
            out = self._spot_fn("kh")(*head, b, rp, *star, cast(uv), k, h)
        return self._apply_masks(out, emask, gmask)

    def _check_shape_kind(self, params):
        """Oblate or spherical, ringed or not, is fixed per model (separate
        graphs)."""
        if _is_oblate(params) != self._oblate:
            raise ValueError("params.f changed between None and a value since "
                             "model construction; build a new TransitModel")
        if _is_ringed(params) != self._ringed:
            raise ValueError("params.ring_in changed between None and a value "
                             "since model construction; build a new "
                             "TransitModel")
        if _is_spotted(params) != self._spotted:
            raise ValueError("params.spots changed between None and a value "
                             "since model construction; build a new "
                             "TransitModel")

    # -- batman-compatible surface ----------------------------------------

    def _check_law(self, params, sets=False):
        """The limb-darkening law AND its order are fixed per model: the
        order is baked into the compiled graph (and into the g_n affine
        map), so a changed count would be silently truncated by the zip in
        flux_dev_poly rather than raising. Returns the normalised u
        (_u_vector: shape judged before the count), so each entry point
        normalises once and threads it through."""
        if self._hlaw is not None:
            # a hybrid law is the same law if it computes the same thing:
            # the name "hybrid2" and elome.HYBRID2 are one law; a name
            # the registry has since re-bound to another definition is not
            new = _hybrid_law(params.limb_dark)
            changed = new is None or new.definition != self._hlaw.definition
        else:
            changed = params.limb_dark != self.limb_dark
        if changed:
            raise ValueError(
                "limb-darkening law changed since model construction; "
                "build a new TransitModel")
        self._check_shape_kind(params)
        u = _u_vector(params.u, sets=sets)
        n_u = u.shape[-1] if sets else len(u)
        if self._n_vec and n_u != self._n_vec:
            raise ValueError(
                f"limb-darkening coefficient count changed since model "
                f"construction ({self._n_vec} -> {n_u}); build a new "
                f"TransitModel")
        return u

    def light_curve(self, params) -> np.ndarray:
        """Model flux at the times given at construction (numpy array)."""
        uv = self._check_law(params)
        if self._stream is not None:
            with mx.stream(self._stream):
                f = self._model_eval(params, uv)
                out = np.array(f, dtype=np.float64)
        else:
            out = np.array(self._model_eval(params, uv), dtype=np.float64)
        if (self.supersample_factor > 1
                and self.integration == "supersample"):
            out = out.reshape(self.t.size,
                              self.supersample_factor).mean(axis=1)
        return out

    # -- batched surface (the sampler-friendly one) ------------------------

    _BATCH_KEYS = ("t0", "per", "rp", "a", "inc", "ecc", "w", "fp")

    def _stack_params(self, params_seq):
        """(n_sets, ) arrays for each scalar parameter, plus (n_sets, N)
        limb-darkening coefficients. Accepts a sequence of TransitParams
        or one TransitParams whose attributes are already arrays."""
        if isinstance(params_seq, (list, tuple)):
            seq = list(params_seq)
            if not seq:
                raise ValueError("no parameter sets given")
            cols = {}
            for p in seq:
                self._require(lambda k: getattr(p, k), self._required_fields())
            for k in self._batch_keys:
                vals = [getattr(p, k) for p in seq]
                # only optional fields are None here: fp of a primary
                # transit, theta, f of a ringed planet, ring_tau (opaque)
                dflt = _BATCH_DEFAULTS.get(k, 0.0)
                if k == "spot_epoch":
                    vals = [p.t0 if v is None else v for p, v in zip(seq, vals)]
                cols[k] = np.array([dflt if v is None else float(v)
                                    for v in vals], dtype=np.float64)
            if self._spotted:
                cols["spots"] = np.stack([np.asarray(p.spots, dtype=np.float64)
                                          for p in seq])
            rows = []
            for p in seq:
                # the same validation the single-set path performs: an
                # unchecked set returns a plausible but wrong curve, or a
                # silent NaN row that a sampler reads as -inf
                uv = self._check_law(p)
                _ld_coeffs(p, u=uv)
                rows.append([float(x) for x in uv])
            u = np.array(rows, dtype=np.float64).reshape(len(seq), -1)
            _check_ecc(cols["ecc"], self._shaped)
            _check_geometry_cols(cols)
            return cols, u
        p = params_seq                       # array-valued TransitParams
        self._require(lambda k: getattr(p, k), self._required_fields())
        u = self._check_law(p, sets=True)    # (1 or n_sets, N)
        # the coefficient count, as the sequence path's _ld_coeffs checks it
        # (a quadratic u of 3 entries was silently truncated, 1 entry gave
        # an opaque IndexError); vector laws are counted by _check_law
        _ld_coeffs(p, u=u[0])
        # any subset of the attributes may be arrays; the batch size is
        # the longest of them (scalars broadcast against it)
        sizes = set()
        for k in self._batch_keys:
            v = getattr(p, k)
            if v is None:
                continue
            sizes.add(np.atleast_1d(np.asarray(v, dtype=np.float64)).size)
        sizes.discard(1)
        if len(sizes) > 1:
            raise ValueError(f"inconsistent parameter-array lengths: "
                             f"{sorted(sizes)}")
        n = sizes.pop() if sizes else 1
        if self._spotted:
            sp = np.asarray(p.spots, dtype=np.float64)
            if sp.ndim == 3:
                if sp.shape[0] not in (1, n) and n != 1:
                    raise ValueError(f"inconsistent parameter-array lengths: "
                                     f"spots has {sp.shape[0]} sets, the "
                                     f"other fields {n}")
                n = max(n, sp.shape[0])
        cols = {}
        for k in self._batch_keys:
            v = getattr(p, k)
            if k == "spot_epoch" and v is None:
                v = p.t0
            v = _BATCH_DEFAULTS.get(k, 0.0) if v is None else v
            cols[k] = np.broadcast_to(
                np.atleast_1d(np.asarray(v, dtype=np.float64)),
                (n,)).astype(np.float64)
        if self._spotted:
            cols["spots"] = np.broadcast_to(sp if sp.ndim == 3 else sp[None],
                                            (n,) + sp.shape[-2:])
        if u.shape[0] not in (1, n):
            raise ValueError(f"inconsistent parameter-array lengths: u has "
                             f"{u.shape[0]} sets, the other fields {n}")
        u = np.broadcast_to(u, (n, u.shape[1]))
        _check_ecc(cols["ecc"], self._shaped)
        _check_geometry_cols(cols)
        return cols, np.ascontiguousarray(u)

    def light_curves(self, params_seq) -> np.ndarray:
        """(n_sets, n_times) for MANY parameter sets in ONE batched call.

        ``light_curve`` is deliberately one-parameter-set-at-a-time, for
        batman parity — and calling it in a loop is the single worst thing
        a sampler can do here (a GPU dispatch costs ~0.2-0.7 ms whatever
        its size, so looping is two to three orders of magnitude slower
        than batching; see the sampler guide in the docs). This is the
        batched form: the same times, many parameter sets, one dispatch.

        ``params_seq`` is either a sequence of TransitParams or a single
        TransitParams whose scalar attributes are arrays of equal length.
        Mixed circular and eccentric sets are fine: the transit-anchored
        orbit degenerates exactly to the circular one at e = 0, so one
        code path serves both.

        For fitting with thousands of chains prefer ``elome.anvil``,
        which owns the likelihood and the float32 conditioning as well.
        """
        cols, u_np = self._stack_params(params_seq)
        n = cols["t0"].size
        dt = self.dtype

        def col(a):
            return mx.array(np.asarray(a, np.float64).reshape(n, 1), dtype=dt)

        stream = self._stream
        ctx = mx.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            t0, per = col(cols["t0"] - self._t_ref), col(cols["per"])
            rp, a = col(cols["rp"]), col(cols["a"])
            ecc = cols["ecc"]
            inc = np.radians(cols["inc"])
            ci = col(np.cos(inc))
            w = np.radians(cols["w"])
            sq = np.sqrt(ecc)
            k, h = col(sq * np.cos(w)), col(sq * np.sin(w))
            fp = col(cols["fp"])
            if self._n_vec:
                uvec = mx.array(u_np, dtype=dt)
                u1 = u2 = None
            else:
                uvec = None
                u1, u2 = _ld_coeffs_batch(self.limb_dark, u_np)
                u1, u2 = col(u1), col(u2)

            if self._spotted:
                b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)[1]
                sp = self._spot_array(cols["spots"], None)
                args = (self._t_mx[None, :], t0, col(cols["spot_epoch"] - self._t_ref),
                        per, a, col(b_np), rp, col(cols["prot"]),
                        col(np.radians(cols["istar"])), col(np.radians(cols["lam"])),
                        sp, uvec)
                if bool(np.any(ecc > 0.0)):
                    out = self._spot_fn("kh")(*args, k, h)
                else:
                    out = self._spot_fn("circ")(*args)
            elif self._ringed:
                from .rings import _check_ring
                _check_ring(cols["f"], cols["ring_in"], cols["ring_out"],
                            cols["ring_f"], cols["ring_tau"])
                b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)[1]
                args = (self._t_mx[None, :], t0, per, a, col(b_np), rp,
                        col(cols["f"]), col(np.radians(cols["theta"])),
                        col(cols["ring_in"]), col(cols["ring_out"]),
                        col(cols["ring_f"]), col(cols["ring_tau"]), uvec)
                if bool(np.any(ecc > 0.0)):
                    out = self._ring_fn("kh")(*args, k, h)
                else:
                    out = self._ring_fn("circ")(*args)
            elif self._oblate:
                from .oblate import _check_domain
                _check_domain(cols["rp"], cols["f"])
                b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)[1]
                args = (self._t_mx[None, :], t0, per, a, col(b_np), rp,
                        col(cols["f"]), col(np.radians(cols["theta"])), uvec)
                if bool(np.any(ecc > 0.0)):
                    out = self._oblate_fn("kh")(*args, k, h)
                else:
                    out = self._oblate_fn("circ")(*args)
            elif self.integration == "contact":
                b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)[1]
                ld = (uvec,) if self._n_vec else (u1, u2)
                out = self._batch_graph_fn(True)(t0, per, rp, a, k, h, ci,
                                                 fp, col(b_np), *ld)
            elif (not self._n_vec and self.transittype == "primary"
                  and self._kernel_usable()):
                # the fused model kernel, as light_curve uses it (see
                # _get_compiled): orbit and photometry in one pass, one row
                # per set. A circular set's cos i is light_curve's b / a,
                # with b = a cos i folded on the host, so each row is
                # light_curve's bit for bit.
                b_circ = col(cols["a"] * np.cos(inc))
                ci_k = mx.where(col(ecc == 0.0) > 0, b_circ / a, ci)
                out = self._batch_kernel_fn()(t0, per, rp, a, k, h, ci_k,
                                              u1, u2)
            else:
                ld = (uvec,) if self._n_vec else (u1, u2)
                out = self._batch_graph_fn(False)(t0, per, rp, a, k, h, ci,
                                                  fp, *ld)
            res = np.array(out, dtype=np.float64)
        if (self.supersample_factor > 1
                and self.integration == "supersample"):
            res = res.reshape(n, self.t.size,
                              self.supersample_factor).mean(axis=2)
        return res

    def _batch_graph_fn(self, contact: bool):
        """light_curves' spherical graph, compiled as light_curve's graphs
        are; (n, 1) columns in, (n, len(t)) out. ``contact``: the contact
        rule, which takes b at conjunction after fp. The law's coefficients
        come last: (u1, u2), or the (n, n_w) vector of a vector law."""
        key = ("batch_graph", contact)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        vec = bool(self._n_vec)
        tt = self._t_mx[None, :]
        dt = self.dtype

        def body(t0, per, rp, a, k, h, ci, fp, ld, b=None):
            u1, u2, uvec = (None, None, ld[0]) if vec else (*ld, None)
            if not contact:
                phi = (2.0 * math.pi) * (tt - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci)
                return self._photom(z, front, rp, u1, u2, fp, uvec=uvec)
            # exact contacts (linearised ones returned as is at e = 0);
            # in contact mode self._t_mx is the exposure mid-times
            cs = contact_offsets_anchored(rp, a, b, k, h, ci)
            T, W = exposure_nodes(tt, t0, per, self.exp_time, cs, self.n_gl,
                                  dtype=dt)
            phi = (2.0 * math.pi) * (T - t0[..., None]) / per[..., None]
            z, front = separation_anchored(phi, k[..., None], h[..., None],
                                           a[..., None], ci[..., None])
            # every per-set column needs the trailing node axis
            nd = lambda c: None if c is None else c[..., None]  # noqa: E731
            f = self._photom(z, front, rp[..., None], nd(u1), nd(u2), nd(fp),
                             uvec=None if uvec is None else uvec[:, None, :])
            return mx.sum(f * W, axis=-1)

        if contact:
            def raw(t0, per, rp, a, k, h, ci, fp, b, *ld):
                return body(t0, per, rp, a, k, h, ci, fp, ld, b)
        else:
            def raw(t0, per, rp, a, k, h, ci, fp, *ld):
                return body(t0, per, rp, a, k, h, ci, fp, ld)
        fn = self._compiled[key] = mx.compile(raw)
        return fn

    def _batch_kernel_fn(self):
        """light_curves' fused-kernel graph, compiled as light_curve's is
        (the same ops fused the same way, so each row is light_curve's
        bit for bit); (n, 1) columns in, (n, len(t)) out."""
        fn = self._compiled.get("batch_kernel")
        if fn is None:
            from .anchored import pack_orbit_constants
            from .metal import make_model_core_metal
            core = make_model_core_metal(0.0)
            tt = self._t_mx
            xdat = mx.stack([tt, mx.zeros_like(tt)])

            def raw(t0, per, rp, a, k, h, ci, u1, u2):
                n = t0.shape[0]
                flat = lambda c: mx.reshape(c, (n,))       # noqa: E731
                orb = pack_orbit_constants(flat(k), flat(h), flat(ci))
                return 1.0 + core(xdat, flat(t0), flat(per), flat(rp),
                                  flat(a), orb, flat(u1), flat(u2))
            fn = self._compiled["batch_kernel"] = mx.compile(raw)
        return fn

    def light_curve_mx(self, params) -> mx.array:
        """Model flux as an MLX array that stays in the graph, for building
        on in MLX -- differentiable in the parameters.

        Any ``TransitParams`` field may be an ``mx.array`` scalar (t0, per,
        rp, a, inc, ecc, w, fp, and ``u`` or its entries), and gradients
        flow to every such field. Python numbers stay constants, folded on
        the host exactly as ``light_curve`` folds them. With ``ecc`` a
        Python number the result is ``light_curve``'s bit for bit when the
        arrays are among t0, per, rp, fp, u and -- on an eccentric orbit
        -- a, which enter the graph as they are. An array ``inc`` or
        ``w``, or ``a`` on a circular orbit, is combined in-graph in the
        model dtype (b = a cos i, (k, h) = sqrt(e) (cos w, sin w)), which
        on an fp32 model is within ~1 ulp of the host fold (see
        ``_model_eval``). Every field must be a scalar and ``u`` a flat
        vector -- list, tuple, numpy or mx.array, or a list holding
        mx.array scalars; a wrong shape raises before anything is built.

        Returns the flux at the times given at construction, exposure-
        averaged for ``integration="contact"``. With ``supersample_factor``
        it is the raw supersampled grid -- reshape to (n_times, factor) and
        average, as ``light_curve`` does.

        Differentiable details:

        * an array-valued ``ecc`` enters as (e, w) directly
          (``anchor_constants_ew``), so d/d(ecc) is finite and correct at
          e = 0 -- the one-sided derivative, since e >= 0 -- and d/dw is
          exactly 0 there. Out of [0, 1) it raises ValueError whenever the
          value can be read (eagerly, or under ``mx.grad``); when it is
          traced -- under ``mx.vmap`` or a caller's ``mx.compile`` -- it
          cannot raise, and instead the output and every gradient are NaN.
        * a Python ``ecc`` keeps ``light_curve``'s own graph -- circular at
          e = 0, (k, h) above it -- so a fit that differentiates, say, only
          rp and t0 runs exactly as fast as ``light_curve``, fused kernel
          included. Only an array-valued ``ecc`` takes the (e, w) graph.
          For the quadratic law that forgoes the fused *model* kernel (see
          _get_compiled): measured 3-4x slower on an fp32 GPU model at
          2e6 points (3.0 vs 0.75-0.91 ms). The hybrid laws' photometry
          runs in their z-input kernel on every graph, this one included.
        * ``u`` may be a list, a numpy array, or an mx.array vector.
        * the contact-rule nodes move with the parameters, so a gradient
          is that of the quadrature actually evaluated.
        * pass t0 as fp64 (a Python float or an fp64 array): an absolute
          BJD stored in fp32 has already lost its precision (0.25 d at
          2.45e6), before the model subtracts its reference time in fp64.
        * under ``mx.grad`` MLX needs the CPU stream for *any* float64
          input. With an fp32 (GPU) model, differentiate float32 fields on
          the GPU; for an fp64 field such as an absolute t0, take the
          gradient inside ``with mx.stream(mx.cpu):`` (or use an fp64
          model, the default, which lives there anyway).

        Precision: the graph is built on the model's own stream -- for the
        default ``dtype=mx.float64`` that is the CPU, as MLX has no float64
        on Metal. The result is a lazy fp64 array, so whatever you build on
        it must stay on the CPU too: wrap your own ops (and ``mx.grad``) in
        ``with mx.stream(mx.cpu):``.
        """
        uv = self._check_law(params)
        ctx = (mx.stream(self._stream) if self._stream is not None
               else _nullcontext())
        with ctx:
            return self._model_eval(params, uv)

