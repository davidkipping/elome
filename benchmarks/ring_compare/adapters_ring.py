"""Per-code adapters for the ringed-planet cross-code benchmark.

Each code provides, for the scenario in scenario_ring.py:
    <code>_prepare(n) -> state;  <code>_run(state) -> flux (n,)
and optionally <code>_batch_prepare() / _batch_run (BATCH_NPV x BATCH_NPT)
and <code>_grad_prepare(n) / _grad_run (value + gradient of sum(ct * flux)).
"""

import math
import os

import numpy as np

from scenario_ring import (A_RS, B_IMPACT, BATCH_NPT, INC, PER, RING_F,
                           RING_IN, RING_OUT, RING_TAU, RP, T0, THETA, U1, U2,
                           W_LD, batch_params, time_grid)

# what each code imports, loaded before the memory baseline (memwatch.py)
IMPORTS = {"elome": ["mlx.core", "elome"]}

# codes whose dependencies conflict with the others' run under their own
# interpreter (run_ring_compare._python)
PYTHON = {}


# ---------------------------------------------------------------------------
# Elómë (float64 on the CPU; float32 Metal kernels on the GPU)
# ---------------------------------------------------------------------------

def _mp_params():
    import elome
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, math.degrees(INC)
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "hybrid2", [W_LD, 0.0]
    p.ring_in, p.ring_out, p.ring_f = RING_IN, RING_OUT, RING_F
    p.ring_tau = None if math.isinf(RING_TAU) else RING_TAU
    p.theta = math.degrees(THETA)
    return p


def _mp_prepare(n, dtype):
    import elome
    p = _mp_params()
    m = elome.TransitModel(p, time_grid(n), dtype=dtype)
    m.light_curve(p)
    return m, p


def elome_fp64_prepare(n):
    import mlx.core as mx
    return _mp_prepare(n, mx.float64)


def elome_fp32_prepare(n):
    import mlx.core as mx
    return _mp_prepare(n, mx.float32)


def elome_fp64_run(state):
    m, p = state
    return m.light_curve(p)


elome_fp32_run = elome_fp64_run


def elome_fp32_batch_prepare():
    import mlx.core as mx
    import elome
    bp = batch_params()
    p = _mp_params()
    p.rp, p.theta = bp["rp"], np.degrees(bp["theta"])
    p.ring_in, p.ring_out, p.ring_f = bp["ring_in"], bp["ring_out"], bp["ring_f"]
    p.t0, p.per, p.a = bp["t0"], bp["per"], bp["a"]
    p.inc = np.degrees(np.arccos(bp["b"] / bp["a"]))
    m = elome.TransitModel(_mp_params(), time_grid(BATCH_NPT), dtype=mx.float32)
    m.light_curves(p)
    return m, p


def elome_fp32_batch_run(state):
    m, p = state
    return m.light_curves(p)


def _mp_grad_prepare(n, dtype):
    """value + gradient of sum(ct * flux) in rp, ring_in, ring_out, ring_f,
    theta, inc, t0, per, a and the limb-darkening weight, through
    light_curve_mx (and ring_tau for a translucent ring)."""
    import mlx.core as mx
    import elome
    p = _mp_params()
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    m = elome.TransitModel(p, time_grid(n), dtype=dtype)
    ct = mx.array(np.random.default_rng(0).normal(size=n), dtype=dtype)
    names = ["rp", "ring_in", "ring_out", "ring_f", "theta", "inc", "t0", "per", "a"]
    if p.ring_tau is not None:
        names.append("ring_tau")
    x0 = [mx.array(float(getattr(p, k)), dtype=dtype) for k in names] + \
         [mx.array(W_LD, dtype=dtype)]

    def loss(*v):
        q = _mp_params()
        for k, val in zip(names, v):
            setattr(q, k, val)
        q.u = [v[-1], 0.0]
        return mx.sum(ct * m.light_curve_mx(q))
    vg = mx.value_and_grad(loss, argnums=tuple(range(len(x0))))
    state = (vg, x0, stream)
    elome_grad_run(state)
    return state


def elome_grad_run(state):
    import mlx.core as mx
    vg, x0, stream = state
    with mx.stream(stream):
        val, g = vg(*x0)
        mx.eval(val, *g)
    return np.array([float(x) for x in g])


def elome_fp64_grad_prepare(n):
    import mlx.core as mx
    return _mp_grad_prepare(n, mx.float64)


def elome_fp32_grad_prepare(n):
    import mlx.core as mx
    return _mp_grad_prepare(n, mx.float32)


elome_fp64_grad_run = elome_grad_run
elome_fp32_grad_run = elome_grad_run


# ---------------------------------------------------------------------------
# pyPplusS (Rein & Ofir 2019; numpy, float64, CPU). Sky positions only, so
# the orbit is computed here, inside the timed call. Radii are absolute and
# per point; ring_inclination = arccos(|cos i_ring|), ring_rotation = theta
# (its rotation is Elómë's principal frame); opacity is the blocked
# fraction T. I = 1 - w (1 - mu^2) is its nonlinear law with c4 = w alone.
# n_center = n_gress = 5 is its default Gauss-Legendre order; 20 also runs.
# ---------------------------------------------------------------------------

def _pp_curve(t, n_q, rp=RP, ring_in=RING_IN, ring_out=RING_OUT,
              ring_f=RING_F, theta=THETA, t0=T0, per=PER, a=A_RS, b=B_IMPACT):
    from pyppluss.segment_models import LC_ringed
    from scenario_ring import blocked_fraction
    phi = 2.0 * np.pi * (t - t0) / per
    X, Y = a * np.sin(phi), -b * np.cos(phi)
    one = np.ones_like(t)
    return LC_ringed(rp * one, ring_in * rp * one, ring_out * rp * one, X, Y,
                     math.acos(1.0 - ring_f), theta, blocked_fraction(),
                     0.0, 0.0, 0.0, W_LD, n_center=n_q, n_gress=n_q)


def _np_nan_shim():
    if not hasattr(np, "NaN"):
        np.NaN = np.nan          # pyPplusS 0.1.5.3 predates NumPy 2


def pyppluss_prepare(n, n_q=5):
    _np_nan_shim()
    t = time_grid(n)
    _pp_curve(t, n_q)
    return t, n_q


def pyppluss_run(state):
    t, n_q = state
    return _pp_curve(t, n_q)


def pyppluss_n20_prepare(n):
    return pyppluss_prepare(n, 20)


pyppluss_n20_run = pyppluss_run


def pyppluss_batch_prepare():
    """No batch API (its ring parameters are scalars per call): a loop,
    timed over the first BATCH_SUBSET sets and scaled by the runner."""
    _np_nan_shim()
    bp = batch_params(BATCH_SUBSET["pyppluss"])
    t = time_grid(BATCH_NPT)
    sets = [dict(rp=bp["rp"][k], ring_in=bp["ring_in"][k], ring_out=bp["ring_out"][k],
                 ring_f=bp["ring_f"][k], theta=bp["theta"][k], t0=bp["t0"][k],
                 per=bp["per"][k], a=bp["a"][k], b=bp["b"][k])
            for k in range(len(bp["rp"]))]
    return t, sets


def pyppluss_batch_run(state):
    t, sets = state
    return np.stack([_pp_curve(t, 5, **s) for s in sets])


IMPORTS["pyppluss"] = ["pyppluss.segment_models"]

# slow batches: timed on this many sets, scaled to BATCH_NPV by the runner
BATCH_SUBSET = {"pyppluss": 8, "squishyplanet": 16, "jojo": 32, "pryngles": 16}


# ---------------------------------------------------------------------------
# squishyplanet RingedSystem (Cassese et al. 2024; JAX, float64, CPU). An
# opaque ring only (the translucent scenario is skipped). Orbit only; the
# ring's orientation is 3-D (ring_obliq, ring_prec), solved here so that its
# projection has 1 - ring_f as axis ratio and its long axis at theta from
# the motion (not timed). Runs in oblate_compare/.venv.
# ---------------------------------------------------------------------------

SQ_THETA_SIGN = 1.0


def _sq_orient(ring_f, theta, rp=RP, t0=T0, per=PER, a=A_RS, b=B_IMPACT):
    """(ring_obliq, ring_prec) projecting to (ring_f, theta)."""
    import jax.numpy as jnp
    from scipy.optimize import least_squares
    from squishyplanet import RingedSystem
    from squishyplanet.engine.rings import ring_para_coeffs
    t = np.linspace(t0 - 0.05, t0 + 0.05, 3)
    probe = RingedSystem(times=jnp.asarray(t), t0=t0, period=per, a=a,
                         i=math.acos(b / a), r=rp, tidally_locked=False,
                         ld_u_coeffs=jnp.array([U1, U2]), ring_inner_r=RING_IN * rp,
                         ring_outer_r=RING_OUT * rp, ring_obliq=0.3, ring_prec=0.3)
    st = probe.state
    orb = {k: jnp.atleast_1d(jnp.asarray(st[k])) for k in ("a", "e", "Omega", "i", "omega")}
    xc, yc = np.asarray(st["x_c"]).ravel(), np.asarray(st["y_c"]).ravel()
    motion = np.array([xc[-1] - xc[0], yc[-1] - yc[0]])
    motion /= np.hypot(*motion)
    f_mid = float(np.asarray(st["f"]).ravel()[1])

    def axes(obliq, prec):
        c = ring_para_coeffs(rRing=jnp.array([1.0]), f=jnp.array([f_mid]),
                             ring_obliq=jnp.array([obliq]), ring_prec=jnp.array([prec]), **orb)
        Mx = np.array([[float(c["c_x1"][0]), float(c["c_x2"][0])],
                       [float(c["c_y1"][0]), float(c["c_y2"][0])]])
        U, sv, _ = np.linalg.svd(Mx)
        return U[:, 0], sv[1] / sv[0]

    def resid(p):
        L, q = axes(*p)
        ang = np.arctan2(motion[0] * L[1] - motion[1] * L[0], motion @ L)
        return [q - (1.0 - ring_f), np.sin(2.0 * (ang - SQ_THETA_SIGN * theta))]
    sol = least_squares(resid, [0.5, 0.5], xtol=1e-15, ftol=1e-15, gtol=1e-15)
    assert np.max(np.abs(sol.fun)) < 1e-10, sol.fun
    return tuple(float(v) for v in sol.x)


def _sq_system(t, obliq, prec):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from squishyplanet import RingedSystem
    return RingedSystem(times=jnp.asarray(t), t0=T0, period=PER, a=A_RS, i=INC, r=RP,
                        tidally_locked=False, ld_u_coeffs=jnp.array([U1, U2]),
                        ring_inner_r=RING_IN * RP, ring_outer_r=RING_OUT * RP,
                        ring_obliq=obliq, ring_prec=prec)


def squishyplanet_prepare(n):
    import jax
    jax.config.update("jax_enable_x64", True)
    s = _sq_system(time_grid(n), *_sq_orient(RING_F, THETA))
    np.asarray(s.lightcurve())
    return s


def squishyplanet_run(s):
    return np.asarray(s.lightcurve())


def squishyplanet_batch_prepare():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    bp = batch_params(BATCH_SUBSET["squishyplanet"])
    orient = np.array([_sq_orient(bp["ring_f"][k], bp["theta"][k], bp["rp"][k], bp["t0"][k],
                                  bp["per"][k], bp["a"][k], bp["b"][k])
                       for k in range(len(bp["rp"]))])
    s = _sq_system(time_grid(BATCH_NPT), *_sq_orient(RING_F, THETA))
    col = lambda v: jnp.asarray(np.asarray(v, np.float64)[:, None])
    params = {"r": col(bp["rp"]), "ring_inner_r": col(bp["ring_in"] * bp["rp"]),
              "ring_outer_r": col(bp["ring_out"] * bp["rp"]),
              "ring_obliq": col(orient[:, 0]), "ring_prec": col(orient[:, 1]),
              "t0": col(bp["t0"]), "period": col(bp["per"]), "a": col(bp["a"]),
              "i": col(np.arccos(bp["b"] / bp["a"]))}
    f = jax.jit(jax.vmap(s.lightcurve))
    f(params).block_until_ready()
    return f, params


def squishyplanet_batch_run(state):
    f, params = state
    return np.asarray(f(params).block_until_ready())


def squishyplanet_grad_prepare(n):
    """value + gradient of sum(ct * flux) in r, ring radii, ring_obliq,
    ring_prec, i, t0, period, a and the limb-darkening coefficients."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    obliq, prec = _sq_orient(RING_F, THETA)
    s = _sq_system(time_grid(n), obliq, prec)
    ct = jnp.asarray(np.random.default_rng(0).normal(size=n))
    params = {"r": jnp.asarray(RP), "ring_inner_r": jnp.asarray(RING_IN * RP),
              "ring_outer_r": jnp.asarray(RING_OUT * RP), "ring_obliq": jnp.asarray(obliq),
              "ring_prec": jnp.asarray(prec), "i": jnp.asarray(INC), "t0": jnp.asarray(T0),
              "period": jnp.asarray(PER), "a": jnp.asarray(A_RS),
              "ld_u_coeffs": jnp.asarray([U1, U2])}
    vg = jax.jit(jax.value_and_grad(lambda prm: jnp.sum(ct * s.lightcurve(prm))))
    state = (vg, params)
    squishyplanet_grad_run(state)
    return state


def squishyplanet_grad_run(state):
    import jax
    vg, params = state
    val, g = vg(params)
    jax.block_until_ready(g)
    return g


# ---------------------------------------------------------------------------
# JoJo ring_lc (Liu & Zhu 2024; numpy, float64). Quadratic law only:
# (u1, u2) = (2w, -w), for which its numerically integrated mu^1 term
# vanishes. opacity is the blocked fraction T; a/R* comes from the stellar
# density, 3.753 (P^2 rho)^(1/3). Spherical planet. Runs in
# oblate_compare/.venv (vendored).
# ---------------------------------------------------------------------------

JOJO_THETA_SIGN = -1.0             # as its oblate model (adapters_oblate)


def _jojo_ring_args(tc=T0, b=B_IMPACT, per=PER, rp=RP, ring_in=RING_IN,
                    ring_out=RING_OUT, ring_f=RING_F, theta=THETA, a=A_RS):
    from scenario_ring import blocked_fraction
    rho = (a / 3.753) ** 3 / per ** 2
    return [tc, b, per, rp, ring_in * rp, ring_out * rp, ring_f,
            JOJO_THETA_SIGN * theta, 0.0, math.pi / 2, U1, U2, math.log10(rho),
            blocked_fraction()]


def jojo_prepare(n):
    from JoJo.jojo_ring import ring_lc
    t = time_grid(n)
    ring_lc(_jojo_ring_args(), t)
    return ring_lc, t


def jojo_run(state):
    ring_lc, t = state
    return np.asarray(ring_lc(_jojo_ring_args(), t))


def jojo_batch_prepare():
    from JoJo.jojo_ring import ring_lc
    return ring_lc, time_grid(BATCH_NPT), batch_params(BATCH_SUBSET["jojo"])


def jojo_batch_run(state):
    ring_lc, t, bp = state
    out = np.empty((bp["rp"].size, t.size))
    for j in range(bp["rp"].size):              # no native batch
        out[j] = ring_lc(_jojo_ring_args(bp["t0"][j], bp["b"][j], bp["per"][j],
                                         bp["rp"][j], bp["ring_in"][j], bp["ring_out"][j],
                                         bp["ring_f"][j], bp["theta"][j], bp["a"][j]), t)
    return out


# ---------------------------------------------------------------------------
# pryngles (Zuluaga et al. 2022; numpy, float64; "spangle" point sampling).
# Its ringed-planet transit (the RingedPlanet class used in its paper):
# Np = Nr = 5000 samples of planet and ring, as in the paper, ring tilt
# arcsin(1 - ring_f) to an orbit seen edge-on so |cos i_ring| = 1 - ring_f,
# taur the normal optical depth (as Elómë's ring_tau). One Python call per
# position: the planet's offset is placed in its observer frame along the
# ring's projected axes. Runs in .venv-pryngles (pryngles 1.0.2 needs
# Python >= 3.12).
# ---------------------------------------------------------------------------

PR_N = 5000


def _pr_planet(ring_f=RING_F, rp=RP, ring_in=RING_IN, ring_out=RING_OUT):
    import pryngles as pr
    taur = np.inf if math.isinf(RING_TAU) else RING_TAU
    P = pr.RingedPlanet(Rstar=0.01, Rplanet=0.01 * rp, Rint=ring_in, Rext=ring_out,
                        i=math.asin(1.0 - ring_f), a=0.2, e=0, Np=PR_N, Nr=PR_N, Nb=0,
                        eobs_ecl=np.array([math.pi / 2, 0.0]),
                        physics=dict(limb_cs=[0.0, 0.0, 0.0, W_LD], taur=taur))
    n = P.M_equ2obs @ np.array([0.0, 0.0, 1.0])          # ring normal, observer frame
    L = np.array([-n[1], n[0]]) / math.hypot(n[0], n[1])  # projected long axis
    M = np.array([-L[1], L[0]])
    return P, L, M


def _pr_curve(P, L, M, t, theta=THETA, t0=T0, per=PER, a=A_RS, b=B_IMPACT):
    phi = 2.0 * np.pi * (t - t0) / per
    X, Y = a * np.sin(phi), -b * np.cos(phi)
    c, s_ = math.cos(theta), math.sin(theta)
    x0, y0 = X * c + Y * s_, -X * s_ + Y * c             # the ring's principal frame
    out = np.empty(t.size)
    for k in range(t.size):
        off = x0[k] * L + y0[k] * M
        P.rstar_obs = np.array([-off[0], -off[1], -50.0])   # the star behind: a transit
        P._updateTransitability()
        P.updateTransit()
        out[k] = 1.0 - (P.Tip.sum() + P.Tir.sum())
    return out


def pryngles_prepare(n):
    P, L, M = _pr_planet()
    t = time_grid(n)
    return P, L, M, t


def pryngles_run(state):
    P, L, M, t = state
    return _pr_curve(P, L, M, t)


def pryngles_batch_prepare():
    """No batch API: a planet object per set (its geometry is fixed at
    construction), then the per-position loop; both timed."""
    bp = batch_params(BATCH_SUBSET["pryngles"])
    return time_grid(BATCH_NPT), bp


def pryngles_batch_run(state):
    t, bp = state
    out = np.empty((bp["rp"].size, t.size))
    for j in range(bp["rp"].size):
        P, L, M = _pr_planet(bp["ring_f"][j], bp["rp"][j], bp["ring_in"][j], bp["ring_out"][j])
        out[j] = _pr_curve(P, L, M, t, bp["theta"][j], bp["t0"][j], bp["per"][j],
                           bp["a"][j], bp["b"][j])
    return out


IMPORTS.update({"squishyplanet": ["jax", "squishyplanet"], "jojo": ["JoJo.jojo_ring"],
                "pryngles": ["pryngles"]})
_OBL_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "oblate_compare", ".venv", "bin", "python")
PYTHON.update({"squishyplanet": _OBL_PY, "jojo": _OBL_PY,
               "pryngles": os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        ".venv-pryngles", "bin", "python")})

CODES = ["elome_fp64", "elome_fp32", "pyppluss", "pyppluss_n20", "squishyplanet",
         "jojo", "pryngles"]
BATCH_CODES = ["elome_fp32", "pyppluss", "squishyplanet", "jojo", "pryngles"]
GRAD_CODES = ["elome_fp64", "elome_fp32", "squishyplanet"]
# opaque rings only
OPAQUE_ONLY = {"squishyplanet"}
