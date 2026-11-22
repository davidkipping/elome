"""Uniform per-code adapters for the cross-code benchmark.

Each adapter provides:
    prepare(n)        -> state (built once per process; jit/warmup here)
    run(state)        -> flux ndarray for the standard scenario, n points
    prepare_batch()/run_batch(state) (optional) -> (npv, npt) fluxes

Thread control happens via environment variables set BEFORE the worker
process imports the code (numba/OpenMP read them at import/first-call).

Capability notes (measured on this machine, macOS/arm64):
    batman        : nthreads > 1 needs OpenMP-built extension (probe)
    pytransit     : numba threading (NUMBA_NUM_THREADS)
    roadrunner    : PyTransit's RoadRunnerModel; its nthreads (numba)
    exoplanet-core: single-threaded C++ ops loop
    jaxoplanet    : XLA CPU thread pool (not user-partitionable cleanly)
    pizzetti      : single-threaded numba (ignores nthreads)
    elome   : MLX — one GPU (all cores; not partitionable) or CPU
    ellc          : needs a Fortran toolchain; unavailable without
                    gfortran (`brew install gcc` would enable it)
"""

import math

import numpy as np

# what each code imports, loaded before the memory baseline (memwatch.py)
IMPORTS = {"elome": ["mlx.core", "elome"], "batman": ["batman"],
           "pytransit": ["pytransit"], "roadrunner": ["pytransit"],
           "exoplanet": ["exoplanet_core"],
           "jaxoplanet": ["jax", "jaxoplanet.core.limb_dark"], "pizzetti": ["pizzetti"],
           "roadrunner_h2": ["pytransit", "numba", "elome.hybrid"]}

from scenario import (A_RS, B_IMPACT, ECC, INC_DEG, INC_ECC_DEG,
                      OMEGA_DEG, PER, RP, T0, U1, U2, W_H2, batch_params,
                      mean_anomaly_at_transit, time_grid, time_grid_ecc,
                      z_of_t, z_of_t_ecc)


# ---------------------------------------------------------------------------
# Elómë
# ---------------------------------------------------------------------------

def _mp_params(law="quadratic"):
    import elome
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.limb_dark = 0.0, 90.0, law
    p.u = list(W_H2) if law == "hybrid2" else [U1, U2]
    return p


def elome_fp64_prepare(n, law="quadratic"):
    import elome
    p = _mp_params(law)
    m = elome.TransitModel(p, time_grid(n))
    m.light_curve(p)  # warm
    return (m, p)


def elome_fp64_run(state):
    m, p = state
    return m.light_curve(p)


def elome_fp32_prepare(n, law="quadratic"):
    import mlx.core as mx
    import elome
    p = _mp_params(law)
    m = elome.TransitModel(p, time_grid(n), dtype=mx.float32)
    m.light_curve(p)
    return (m, p)


def elome_fp32_run(state):
    m, p = state
    return m.light_curve(p)


def elome_gpu_batch_prepare(_n=None, law="quadratic"):
    """Engine-contract batched model: (npv, 8) params x (2, npt) times,
    fp32 on the GPU — Elómë's design center. hybrid2: anvil's hybrid model,
    its shape weights as the unit square's (q1, q2)."""
    import mlx.core as mx
    from elome.anvil import make_hybrid_transit_flux, make_quad_transit_flux
    from elome.ld import hybrid2_to_q_np, u_to_q_np
    from scenario import BATCH_NPT, BATCH_NPV

    bp = batch_params()
    if law == "hybrid2":
        l1, l2 = hybrid2_to_q_np(bp["w1"], bp["w2"])
        flux = make_hybrid_transit_flux(PER, "hybrid2")
    else:
        l1, l2 = u_to_q_np(bp["u1"], bp["u2"])
        flux = make_quad_transit_flux(PER)
    v = np.stack([
        bp["t0"], bp["per"] - PER, bp["rp"], bp["b"], bp["a"], l1, l2,
        np.zeros(BATCH_NPV),
    ], axis=1).astype(np.float32)
    t = time_grid(BATCH_NPT)
    x = np.stack([t, np.zeros_like(t)]).astype(np.float32)
    model = mx.compile(flux)
    v_mx, x_mx = mx.array(v), mx.array(x)
    out = model(v_mx, x_mx)
    mx.eval(out)  # warm/compile
    return (model, v_mx, x_mx, mx)


def elome_gpu_batch_run(state):
    model, v, x, mx = state
    out = model(v, x)
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# batman
# ---------------------------------------------------------------------------

def batman_prepare(n, nthreads=1):
    import batman
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    m = batman.TransitModel(p, time_grid(n), nthreads=nthreads)
    m.light_curve(p)
    return (m, p)


def batman_run(state):
    m, p = state
    return m.light_curve(p)


def batman_batch_prepare(_n=None):
    import batman
    from scenario import BATCH_NPT
    bp = batch_params()
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    m = batman.TransitModel(p, time_grid(BATCH_NPT))
    m.light_curve(p)
    return (m, p, bp)


def _loop_batch_run(state):
    m, p, bp = state
    out = []
    for i in range(bp["rp"].size):
        p.t0 = bp["t0"][i]
        p.per = bp["per"][i]
        p.rp = bp["rp"][i]
        p.a = bp["a"][i]
        p.inc = math.degrees(math.acos(bp["b"][i] / bp["a"][i]))
        p.u = [bp["u1"][i], bp["u2"][i]]
        out.append(m.light_curve(p))
    return np.stack(out)


batman_batch_run = _loop_batch_run


# ---------------------------------------------------------------------------
# pizzetti: batman's interface; its interior series to its default bound
# (series_tol=1e-9 of the stellar flux), or None: the series off, the exact
# Mandel & Agol solution everywhere
# ---------------------------------------------------------------------------

def _pz_model(p, t, series_tol=1e-9):
    import pizzetti
    kw = {} if series_tol == 1e-9 else {"series_tol": 0.0}
    m = pizzetti.TransitModel(p, t, **kw)
    m.light_curve(p)
    return (m, p)


def pizzetti_prepare(n, series_tol=1e-9):
    import pizzetti
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    return _pz_model(p, time_grid(n), series_tol)


def pizzetti_run(state):
    m, p = state
    return m.light_curve(p)


def pizzetti_batch_prepare(_n=None):
    from scenario import BATCH_NPT
    m, p = pizzetti_prepare(BATCH_NPT)
    return (m, p, batch_params())


pizzetti_batch_run = _loop_batch_run    # batman's loop, on pizzetti's model


# ---------------------------------------------------------------------------
# PyTransit
# ---------------------------------------------------------------------------

def pytransit_prepare(n, interpolate=False):
    from pytransit import QuadraticModel
    tm = QuadraticModel(interpolate=interpolate)
    tm.set_data(time_grid(n))
    kw = dict(k=RP, ldc=[U1, U2], t0=T0, p=PER, a=A_RS,
              i=math.radians(INC_DEG))
    tm.evaluate(**kw)  # trigger numba compile
    return (tm, kw)


def pytransit_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


def pytransit_batch_prepare(_n=None):
    from pytransit import QuadraticModel
    from scenario import BATCH_NPT
    bp = batch_params()
    tm = QuadraticModel(interpolate=False)
    tm.set_data(time_grid(BATCH_NPT))
    inc = np.arccos(bp["b"] / bp["a"])
    ldc = np.stack([bp["u1"], bp["u2"]], axis=1)
    kw = dict(k=bp["rp"], ldc=ldc, t0=bp["t0"], p=bp["per"], a=bp["a"],
              i=inc)
    tm.evaluate(**kw)
    return (tm, kw)


def pytransit_batch_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


# ---------------------------------------------------------------------------
# RoadRunner (PyTransit's RoadRunnerModel, Parviainen 2020): its default
# grids, on BENCH_NTHREADS threads for one light curve and on every numba
# thread for a batch, as PyTransit's batch above
# ---------------------------------------------------------------------------

def _rr_model(ld="quadratic", nthreads=1):
    from pytransit import RoadRunnerModel
    return RoadRunnerModel(ld, nthreads=nthreads)


def roadrunner_prepare(n, ld="quadratic", ldc=(U1, U2), t=None):
    import os
    tm = _rr_model(ld, int(os.environ.get("BENCH_NTHREADS", "1")))
    tm.set_data(time_grid(n) if t is None else t)
    kw = dict(k=RP, ldc=np.asarray(ldc, dtype=np.float64), t0=T0, p=PER,
              a=A_RS, i=math.radians(INC_DEG))
    tm.evaluate(**kw)  # trigger numba compile
    return (tm, kw)


def roadrunner_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


def roadrunner_batch_prepare(_n=None):
    import os
    from scenario import BATCH_NPT
    bp = batch_params()
    tm = _rr_model(nthreads=int(os.environ.get("NUMBA_NUM_THREADS", "1")))
    tm.set_data(time_grid(BATCH_NPT))
    # k and t0 as (npv, 1): 1-D they are one radius per passband and one
    # mid-time per epoch, of ONE light curve
    kw = dict(k=bp["rp"][:, None], ldc=np.stack([bp["u1"], bp["u2"]], axis=1),
              t0=bp["t0"][:, None], p=bp["per"], a=bp["a"],
              i=np.arccos(bp["b"] / bp["a"]))
    tm.evaluate(**kw)
    return (tm, kw)


roadrunner_batch_run = roadrunner_run


# ---------------------------------------------------------------------------
# exoplanet-core (the compiled kernels inside `exoplanet`, FM+21)
# ---------------------------------------------------------------------------

def _xo_flux(b, r, u1, u2):
    from exoplanet_core.numpy import ops
    b = np.ascontiguousarray(b, dtype=np.float64)
    s = ops.quad_solution_vector(b, np.full_like(b, r))
    g = np.array([1.0 - u1 - 1.5 * u2, u1 + 2.0 * u2, -0.25 * u2])
    norm = np.pi * (g[0] + 2.0 * g[1] / 3.0)
    return s @ (g / norm)


def exoplanet_prepare(n):
    t = time_grid(n)
    _xo_flux(z_of_t(t[:16]), RP, U1, U2)
    return t


def exoplanet_run(t):
    # orbit -> z inside the timed region: every code is measured from
    # orbital elements, not a precomputed separation
    return _xo_flux(z_of_t(t), RP, U1, U2)


def exoplanet_batch_prepare(_n=None):
    from scenario import BATCH_NPT
    bp = batch_params()
    t = time_grid(BATCH_NPT)
    return (t, bp)


def exoplanet_batch_run(state):
    t, bp = state
    out = []
    for i in range(bp["rp"].size):
        phi = 2 * np.pi * (t - bp["t0"][i]) / bp["per"][i]
        cosi = bp["b"][i] / bp["a"][i]
        z = bp["a"][i] * np.sqrt(np.sin(phi) ** 2
                                 + cosi ** 2 * np.cos(phi) ** 2)
        z = np.where(np.cos(phi) > 0, z, 2.0 + z)
        out.append(_xo_flux(z, bp["rp"][i], bp["u1"][i], bp["u2"][i]))
    return np.stack(out)


# ---------------------------------------------------------------------------
# jaxoplanet
# ---------------------------------------------------------------------------

def jaxoplanet_prepare(n, order=10):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core.limb_dark import light_curve
    jax.config.update("jax_enable_x64", True)
    u = jnp.array([U1, U2])
    t = jnp.asarray(time_grid(n))
    cosi = B_IMPACT / A_RS

    @jax.jit
    def f(t_):
        # orbit inside the jit: measured from orbital elements like
        # every other code (XLA fuses it, so the penalty is small)
        phi = 2.0 * jnp.pi * (t_ - T0) / PER
        z = A_RS * jnp.sqrt(jnp.sin(phi) ** 2
                            + cosi ** 2 * jnp.cos(phi) ** 2)
        z = jnp.where(jnp.cos(phi) > 0, z, 2.0 + z)
        return 1.0 + light_curve(u, z, RP, order=order)

    f(t).block_until_ready()
    return (f, t)


def jaxoplanet_run(state):
    f, t = state
    return np.asarray(f(t).block_until_ready())


def jaxoplanet_batch_prepare(_n=None):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core.limb_dark import light_curve
    from scenario import BATCH_NPT
    jax.config.update("jax_enable_x64", True)
    bp = batch_params()
    t = jnp.asarray(time_grid(BATCH_NPT))

    def one(rp, t0, per, a, b_imp, u1, u2):
        phi = 2 * jnp.pi * (t - t0) / per
        cosi = b_imp / a
        z = a * jnp.sqrt(jnp.sin(phi) ** 2 + cosi ** 2 * jnp.cos(phi) ** 2)
        z = jnp.where(jnp.cos(phi) > 0, z, 2.0 + z)
        return 1.0 + light_curve(jnp.stack([u1, u2]), z, rp, order=10)

    f = jax.jit(jax.vmap(one))
    args = tuple(jnp.asarray(bp[k]) for k in
                 ("rp", "t0", "per", "a", "b", "u1", "u2"))
    f(*args).block_until_ready()
    return (f, args)


def jaxoplanet_batch_run(state):
    f, args = state
    return np.asarray(f(*args).block_until_ready())


# ---------------------------------------------------------------------------
# value + gradient of sum(ct * flux) in (rp, t0, per, a, b, u1, u2): the
# differentiable codes only (Elómë, jaxoplanet)
# ---------------------------------------------------------------------------

def _elome_grad_prepare(n, dtype, law="quadratic"):
    import mlx.core as mx
    import elome
    p = _mp_params(law)
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    m = elome.TransitModel(p, time_grid(n), dtype=dtype)
    ct = mx.array(np.random.default_rng(0).normal(size=n), dtype=dtype)
    x0 = [mx.array(v, dtype=dtype) for v in (RP, T0, PER, A_RS, B_IMPACT, *p.u)]

    def loss(rp, t0, per, a, b, u1, u2):
        q = _mp_params(law)
        q.rp, q.t0, q.per, q.a = rp, t0, per, a
        q.inc = mx.arccos(b / a) * (180.0 / math.pi)
        q.u = [u1, u2]
        return mx.sum(ct * m.light_curve_mx(q))
    vg = mx.value_and_grad(loss, argnums=tuple(range(7)))
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


def elome_fp32_grad_prepare(n):
    import mlx.core as mx
    return _elome_grad_prepare(n, mx.float32)


def elome_fp64_grad_prepare(n):
    import mlx.core as mx
    return _elome_grad_prepare(n, mx.float64)


elome_fp32_grad_run = elome_grad_run
elome_fp64_grad_run = elome_grad_run


def jaxoplanet_grad_prepare(n, order=10):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core.limb_dark import light_curve
    jax.config.update("jax_enable_x64", True)
    t = jnp.asarray(time_grid(n))
    ct = jnp.asarray(np.random.default_rng(0).normal(size=n))

    def loss(rp, t0, per, a, b, u1, u2):
        phi = 2.0 * jnp.pi * (t - t0) / per
        cosi = b / a
        z = a * jnp.sqrt(jnp.sin(phi) ** 2 + cosi ** 2 * jnp.cos(phi) ** 2)
        z = jnp.where(jnp.cos(phi) > 0, z, 2.0 + z)
        return jnp.sum(ct * (1.0 + light_curve(jnp.stack([u1, u2]), z, rp,
                                               order=order)))
    vg = jax.jit(jax.value_and_grad(loss, argnums=tuple(range(7))))
    x0 = tuple(jnp.asarray(v) for v in (RP, T0, PER, A_RS, B_IMPACT, U1, U2))
    state = (vg, x0)
    jaxoplanet_grad_run(state)
    return state


def jaxoplanet_grad_run(state):
    import jax
    vg, x0 = state
    val, g = vg(*x0)
    jax.block_until_ready(g)
    return np.array([float(x) for x in g])


# ---------------------------------------------------------------------------
# Eccentric variants. Every code solves Kepler's equation itself here, so
# these measure the orbit solver as well as the photometry — exoplanet-core
# and jaxoplanet through their own kepler() (which return (sin f, cos f)),
# batman / PyTransit / Elómë through their normal orbital-element
# interfaces.
# ---------------------------------------------------------------------------

_W_RAD = math.radians(OMEGA_DEG)
_COSI_ECC = math.cos(math.radians(INC_ECC_DEG))


def _mp_params_ecc(law="quadratic"):
    import elome
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_ECC_DEG
    p.ecc, p.w, p.limb_dark = ECC, OMEGA_DEG, law
    p.u = list(W_H2) if law == "hybrid2" else [U1, U2]
    return p


def elome_fp64_ecc_prepare(n):
    import elome
    p = _mp_params_ecc()
    m = elome.TransitModel(p, time_grid_ecc(n))
    m.light_curve(p)
    return (m, p)


def elome_fp64_ecc_run(state):
    m, p = state
    return m.light_curve(p)


def elome_fp32_ecc_prepare(n):
    import mlx.core as mx
    import elome
    p = _mp_params_ecc()
    m = elome.TransitModel(p, time_grid_ecc(n), dtype=mx.float32)
    m.light_curve(p)
    return (m, p)


def elome_fp32_ecc_run(state):
    m, p = state
    return m.light_curve(p)


def batman_ecc_prepare(n, nthreads=1):
    import batman
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_ECC_DEG
    p.ecc, p.w, p.u, p.limb_dark = ECC, OMEGA_DEG, [U1, U2], "quadratic"
    m = batman.TransitModel(p, time_grid_ecc(n), nthreads=nthreads)
    m.light_curve(p)
    return (m, p)


def batman_ecc_run(state):
    m, p = state
    return m.light_curve(p)


def pizzetti_ecc_prepare(n, series_tol=1e-9):
    import pizzetti
    p = pizzetti.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_ECC_DEG
    p.ecc, p.w, p.u, p.limb_dark = ECC, OMEGA_DEG, [U1, U2], "quadratic"
    return _pz_model(p, time_grid_ecc(n), series_tol)


def pizzetti_ecc_run(state):
    m, p = state
    return m.light_curve(p)


def pytransit_ecc_prepare(n, interpolate=False):
    from pytransit import QuadraticModel
    tm = QuadraticModel(interpolate=interpolate)
    tm.set_data(time_grid_ecc(n))
    kw = dict(k=RP, ldc=[U1, U2], t0=T0, p=PER, a=A_RS,
              i=math.radians(INC_ECC_DEG), e=ECC, w=_W_RAD)
    tm.evaluate(**kw)
    return (tm, kw)


def pytransit_ecc_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


def roadrunner_ecc_prepare(n):
    import os
    tm = _rr_model(nthreads=int(os.environ.get("BENCH_NTHREADS", "1")))
    tm.set_data(time_grid_ecc(n))
    kw = dict(k=RP, ldc=np.array([U1, U2]), t0=T0, p=PER, a=A_RS,
              i=math.radians(INC_ECC_DEG), e=ECC, w=_W_RAD)
    tm.evaluate(**kw)
    return (tm, kw)


roadrunner_ecc_run = roadrunner_run


def _xo_z_ecc(t):
    """Separation from exoplanet-core's OWN Kepler solver."""
    from exoplanet_core.numpy import ops
    M = 2.0 * np.pi * (t - T0) / PER + mean_anomaly_at_transit()
    sinf, cosf = ops.kepler(np.ascontiguousarray(M),
                            np.full_like(M, ECC))
    r_orb = A_RS * (1.0 - ECC ** 2) / (1.0 + ECC * cosf)
    swf = math.sin(_W_RAD) * cosf + math.cos(_W_RAD) * sinf
    z = r_orb * np.sqrt(np.maximum(
        1.0 - swf ** 2 * (1.0 - _COSI_ECC ** 2), 0.0))
    return np.where(swf > 0.0, z, 2.0 + z)


def exoplanet_ecc_prepare(n):
    t = time_grid_ecc(n)
    _xo_flux(_xo_z_ecc(t[:16]), RP, U1, U2)
    return t


def exoplanet_ecc_run(t):
    return _xo_flux(_xo_z_ecc(t), RP, U1, U2)


def jaxoplanet_ecc_prepare(n, order=10):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core import kepler
    from jaxoplanet.core.limb_dark import light_curve
    jax.config.update("jax_enable_x64", True)
    u = jnp.array([U1, U2])
    t = jnp.asarray(time_grid_ecc(n))
    m_tra = mean_anomaly_at_transit()
    sw, cw = math.sin(_W_RAD), math.cos(_W_RAD)

    @jax.jit
    def f(t_):
        M = 2.0 * jnp.pi * (t_ - T0) / PER + m_tra
        sinf, cosf = kepler(M, ECC)
        r_orb = A_RS * (1.0 - ECC ** 2) / (1.0 + ECC * cosf)
        swf = sw * cosf + cw * sinf
        z = r_orb * jnp.sqrt(jnp.maximum(
            1.0 - swf ** 2 * (1.0 - _COSI_ECC ** 2), 0.0))
        z = jnp.where(swf > 0.0, z, 2.0 + z)
        return 1.0 + light_curve(u, z, RP, order=order)

    f(t).block_until_ready()
    return (f, t)


def jaxoplanet_ecc_run(state):
    f, t = state
    return np.asarray(f(t).block_until_ready())


# ---------------------------------------------------------------------------
# the hybrid2 law (W_H2): Elómë in closed form; RoadRunner tabulating it as
# a function of mu, with its exact disc integral. The others cannot take it
# (batman's custom law needs its C rebuilt; pizzetti and jaxoplanet take
# sums of powers of mu, exoplanet-core the quadratic law).
# ---------------------------------------------------------------------------

def elome_fp64_h2_prepare(n):
    return elome_fp64_prepare(n, "hybrid2")


def elome_fp32_h2_prepare(n):
    return elome_fp32_prepare(n, "hybrid2")


def elome_fp64_h2_ecc_prepare(n):
    import elome
    p = _mp_params_ecc("hybrid2")
    m = elome.TransitModel(p, time_grid_ecc(n))
    m.light_curve(p)
    return (m, p)


def elome_fp32_h2_ecc_prepare(n):
    import mlx.core as mx
    import elome
    p = _mp_params_ecc("hybrid2")
    m = elome.TransitModel(p, time_grid_ecc(n), dtype=mx.float32)
    m.light_curve(p)
    return (m, p)


def elome_gpu_h2_batch_prepare(_n=None):
    return elome_gpu_batch_prepare(law="hybrid2")


def elome_fp32_h2_grad_prepare(n):
    import mlx.core as mx
    return _elome_grad_prepare(n, mx.float32, "hybrid2")


def elome_fp64_h2_grad_prepare(n):
    import mlx.core as mx
    return _elome_grad_prepare(n, mx.float64, "hybrid2")


elome_fp64_h2_run = elome_fp32_h2_run = elome_fp64_run
elome_fp64_h2_ecc_run = elome_fp32_h2_ecc_run = elome_fp64_run
elome_gpu_h2_batch_run = elome_gpu_batch_run
elome_fp32_h2_grad_run = elome_fp64_h2_grad_run = elome_grad_run


def _h2_ld():
    """(I(mu, w), its disc integral(w)) of hybrid2 for RoadRunner, numba
    functions of the shape weights w: I = c0 + c1 mu^2 + c2 mu^4 + sum_k
    d_k (mu^2 + eps_k)^-2, (c, d) affine in w (elome.hybrid.w_to_c), and
    2 pi int I mu dmu = 2 pi [c0/2 + c1/4 + c2/6 + sum_k d_k / (2 eps_k (1 + eps_k))]."""
    from numba import njit
    from elome.hybrid import get_law, w_to_c
    law = get_law("hybrid2")
    eps = np.array(law.eps, dtype=np.float64)
    flat = lambda w: np.concatenate([np.asarray(x, np.float64) for x in w_to_c(law, w)])  # noqa: E731
    base = flat([0.0, 0.0])
    A = np.stack([flat(e) - base for e in np.eye(2)], axis=1)
    w_test = np.array([0.37, -0.21])
    assert np.allclose(flat(w_test), base + A @ w_test, rtol=0, atol=1e-13), "w_to_c not affine"
    K = eps.size

    @njit
    def coef(pv):
        return base + A[:, 0] * pv[0] + A[:, 1] * pv[1]

    @njit
    def ld(mu, pv):
        c = coef(pv)
        mu2 = mu * mu
        out = c[0] + c[1] * mu2 + c[2] * mu2 * mu2
        for k in range(K):
            out = out + c[3 + k] / (mu2 + eps[k]) ** 2
        return out

    @njit
    def ldi(pv):
        c = coef(pv)
        s = c[0] / 2.0 + c[1] / 4.0 + c[2] / 6.0
        for k in range(K):
            s += c[3 + k] / (2.0 * eps[k] * (1.0 + eps[k]))
        return 2.0 * math.pi * s
    return ld, ldi


def roadrunner_h2_prepare(n, t=None):
    return roadrunner_prepare(n, ld=_h2_ld(), ldc=W_H2, t=t)


roadrunner_h2_run = roadrunner_run


def roadrunner_h2_ecc_prepare(n):
    import os
    tm = _rr_model(_h2_ld(), int(os.environ.get("BENCH_NTHREADS", "1")))
    tm.set_data(time_grid_ecc(n))
    kw = dict(k=RP, ldc=np.asarray(W_H2, dtype=np.float64), t0=T0, p=PER, a=A_RS,
              i=math.radians(INC_ECC_DEG), e=ECC, w=_W_RAD)
    tm.evaluate(**kw)
    return (tm, kw)


roadrunner_h2_ecc_run = roadrunner_run


def roadrunner_h2_batch_prepare(_n=None):
    import os
    from scenario import BATCH_NPT
    bp = batch_params()
    tm = _rr_model(_h2_ld(), nthreads=int(os.environ.get("NUMBA_NUM_THREADS", "1")))
    tm.set_data(time_grid(BATCH_NPT))
    kw = dict(k=bp["rp"][:, None], ldc=np.stack([bp["w1"], bp["w2"]], axis=1),
              t0=bp["t0"][:, None], p=bp["per"], a=bp["a"],
              i=np.arccos(bp["b"] / bp["a"]))
    tm.evaluate(**kw)
    return (tm, kw)


roadrunner_h2_batch_run = roadrunner_run

