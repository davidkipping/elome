# Fitting a spherical planet

This page is about sampling a posterior: which entry point to use, how to
call it from each kind of sampler, and the mistakes that produce a wrong
answer without any error. Before writing a sampler loop, read
{doc}`../guides/batching`. It is the one rule that decides whether
Elómë is the fastest code you have or the slowest.

## The model a sampler sees

`elome.anvil` provides two ready-made models in the contract samplers
want: `model_fn(v, x)` maps a `(n_chains, n_params)` array of parameters
and the data's times to a `(n_chains, m)` array of flux deviations, in
one fused Metal kernel, with analytic gradients.

| builder | parameters, in order |
|---|---|
| `make_quad_transit_flux(period_ref)` | `t0_off, p_off, r, b, a, q1, q2, df0` (8, circular) |
| `make_ecc_transit_flux(period_ref)` | `t0_off, p_off, r, b, a, q1, q2, secosw, sesinw, df0` (10, eccentric) |

The order is `elome.anvil.PARAM_NAMES` and `PARAM_NAMES_ECC`: in the
eccentric model `secosw` and `sesinw` come *before* `df0`.

- `t0_off` and `p_off` are offsets from a reference epoch and period, so
  that float32 carries small numbers.
- `q1` and `q2` are the Kipping (2013) limb-darkening parameters (uniform
  on $[0, 1]^2$ for physical quadratic profiles).
- `df0` is a flux offset.
- The eccentric model samples $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$,
  and `b` is the impact parameter at inferior conjunction,
  $\cos i = b\,(1 + e\sin\omega) / [a\,(1 - e^2)]$.

Both models run on the same kernel. A circular orbit is the case $e = 0$
of the eccentric one, and chains at $e = 0$ skip the Kepler solve.

## Fitting real mission data with anvil

`make_transit_target` builds the whole fitting problem from a light
curve: model, likelihood, parameter boxes and reporting:

```python
from elome.anvil import make_transit_target, import_engine
engine, _ = import_engine()

tt = make_transit_target(t, y, yerr,          # BJD/BTJD/BKJD; flux near 1
                         t0_guess=..., period_guess=...)
u0 = ...                                       # (n_chains, 8) starting points
res = engine.run(engine.ChEESHMC(tt.target, max_leapfrog=384),
                 tt.target, u0, n_warmup=300, n_samples=200)
```

It owns the conditioning that the float32 path depends on, so that you
never have to rediscover it:

- the float64 reduction and epoch centring of the times (a raw BJD has a
  0.25-day float32 resolution and cannot survive otherwise);
- fitting $y - 1$, so the model carries numbers of the size of the
  transit depth;
- the parameter boxes (`elome.anvil.DEFAULT_BOUNDS`, overridable with
  `bounds=`);
- reporting values back on the input time system;
- with `eccentric=True`, the joint-constraint barrier described below.

`tt.model_params(t0=..., period=..., r=..., ...)` converts physical values
into the model's parameter vector. `tt.x64`, `tt.y_fit` and `tt.model_fn`
are exposed so that a different likelihood, such as a Gaussian process
over the same mean model, can wrap them. `examples/fit_mission_data.py`
in the repository runs the whole thing end to end, and the first
tutorial does it step by step.

### Hybrid limb darkening and spotted stars

`limb_dark="hybrid2" | "hybrid4" | "hybrid5"` fits a hybrid law instead
of the quadratic one. Its weights are sampled through the law's own q
prior (`q1` to `qn`, uniform on the cube and mapped onto exactly the
physical profiles; see {doc}`../reference/hybrid-laws`), and `exp_time=`
integrates each exposure, by the contact rule by default:

```python
tt = make_transit_target(t, y, yerr, t0_guess=..., period_guess=...,
                         limb_dark="hybrid4", exp_time=29.4 / 1440)
```

`n_spots=K` (a hybrid law only) adds a spotted star: four parameters per
spot and three for the star's rotation (see {doc}`../spots/accuracy-and-fitting`).
`elome.anvil.transit_param_names` gives the vector's order for any
combination of options, and `tt.param_names` the target's own. A spotted
model is evaluated eagerly, so pass `compile_step=tt.compile_step` (False)
to `engine.run`:

```python
tt = make_transit_target(t, y, yerr, t0_guess=..., period_guess=...,
                         limb_dark="hybrid4", n_spots=2,
                         bounds={"prot": (8.0, 14.0)})
res = engine.run(engine.ChEESHMC(tt.target, max_leapfrog=128), tt.target,
                 u0, n_warmup=300, n_samples=200,
                 compile_step=tt.compile_step)
```

### The eccentric model needs a barrier

Two physical constraints couple parameters: the orbit must clear the
star at periastron, $a(1-e) > 1 + r$, and the inclination must be real,
$|\cos i| \le 1$. No box on single parameters can express them. And
because Elómë keeps every computation finite, an unphysical geometry
returns an ordinary finite log-likelihood, so a chain can sample an
improper posterior without any error.

`make_transit_target(..., eccentric=True)` and `make_ecc_target` attach
a smooth quadratic barrier for you. A hard `-inf` wall would make HMC
diverge. With your own likelihood, add `ecc_constraint_penalty` or wrap
the likelihood in `PenalizedLogLike`. **Omitting the barrier is the
easiest way to get a wrong eccentric fit.**

## anvil's emcee-style sampler

For an emcee-style workflow on the GPU, `anvil.EnsembleSampler` has the
same `(nwalkers, ndim, log_prob_fn)`, `run_mcmc` and `get_chain` surface
as emcee. The stretch move runs entirely on the GPU, with no upload or
synchronisation inside the loop. Its `log_prob_fn` must take and
return MLX arrays, `(nwalkers, ndim) -> (nwalkers,)`. Pair it with
`ChunkedGaussianLogLike` as `elome.anvil.make_target` does.

## emcee on the CPU

emcee can batch through `vectorize=True`, but know what it batches. The
default stretch move updates the ensemble in two halves, so your
log-probability receives `(n_walkers/2, ndim)`. That is two dispatches
per move, each carrying `n_walkers/2 × m` points. Size the ensemble so
that a *half*-ensemble dispatch carries at least $3\times10^4$ points.

A correct and fast recipe, priors included:

```python
import numpy as np, mlx.core as mx, emcee
from anvil.precision import ChunkedGaussianLogLike
from elome.anvil import make_quad_transit_flux
from elome.orbit import epoch_center_times

model = make_quad_transit_flux(period_ref)        # fused Metal kernel inside
x64 = epoch_center_times(t, t0_ref, period_ref)   # float64 host preprocessing
loglike = ChunkedGaussianLogLike(model, x64, y - 1.0, yerr)  # fp32-safe sums
loglike_c = mx.compile(lambda v: loglike(v))      # ~2x: fuses the reduction

lo = np.array([-0.5, -0.05, 0.01, 0.0, 2.0, 0.0, 0.0, -0.01])
hi = np.array([ 0.5,  0.05, 0.50, 0.9, 50., 1.0, 1.0,  0.01])

def log_prob_batch(theta):                        # (n_walkers/2, 8) from emcee
    lp = np.array(loglike_c(mx.array(theta.astype(np.float32))),
                  dtype=np.float64)
    bad = np.any((theta < lo) | (theta > hi), axis=1)
    lp[bad] = -np.inf                             # emcee treats -inf as reject
    return lp

sampler = emcee.EnsembleSampler(n_walkers, 8, log_prob_batch, vectorize=True)
```

Three things this recipe gets right that a minimal one gets wrong without
any error:

1. **Priors are mandatory.** Unphysical proposals (negative `r`, `q1`
   outside $[0, 1]$, $|b| > a$) return ordinary finite log-likelihoods.
   Without the bounds the chain wanders into flat, unbounded directions
   and mirror modes, and samples an improper posterior. anvil's parameter
   transforms impose such bounds for you; emcee does not.
2. **Compile the likelihood.** `mx.compile` fuses the reduction. The same
   likelihood run eagerly is about 1.5 times as slow at 1024 × 65,536
   points (29.3 against 19.7 ms, `benchmarks/bench_vjp.py`).
3. **Use `ChunkedGaussianLogLike`, not a bare `mx.sum`.** A single
   float32 sum over a long light curve accumulates rounding towards the
   scale of a Metropolis decision. The chunked tree (16,384-point chunks
   by default) keeps it small. Casting the finished float32 sum to float64
   recovers nothing.

## Your own sampler

The principle is the same: propose for all chains, stack the proposals
into `(n_chains, ndim)`, make one likelihood call, and accept or reject
in a vectorised step. Extra chains are nearly free while the GPU is
unsaturated, below about $10^6$ points per dispatch. Beyond that, the
time grows linearly with the number of chains.

## Tuning HMC

### Measure the trajectory length

ChEES-HMC adapts its step size but is given a ceiling on the trajectory
length, `max_leapfrog`. That ceiling matters a great deal, and in
opposite directions on the two models. With 256 chains, 300 warmup and
200 samples on 20,000 points (`benchmarks/bench_leapfrog_cap.py`):

| `max_leapfrog` | circular, 8 parameters | eccentric, 10 parameters |
|---:|---|---|
| 16 | 7.7 ESS/s, $\hat R$ 2.20 | **5.3 ESS/s**, $\hat R$ 3.91 |
| 24 | 8.2 ESS/s, $\hat R$ 1.83 | 4.9 ESS/s, $\hat R$ 3.74 |
| 96 | 10.9 ESS/s, $\hat R$ 1.13 | 2.0 ESS/s, $\hat R$ 3.78, 172 divergences |
| 384 | **71.2 ESS/s**, $\hat R$ **1.00** | — |

On the well-conditioned circular posterior, going from 24 to 384 buys
8.7 times the effective samples per second and takes $\hat R$ from 1.83
to 1.00, with no divergences. On the eccentric posterior the same change
is harmful: the sampling rate falls steadily and long trajectories
eventually break the integrator. Its duration ridge in $(a, b, e, \omega)$
is not traversed by going further in a straight line. The robust claim
is the direction and size of the effect, not the number 384. Re-measure
for your problem: it is two lines of configuration.

### What to expect from the eccentric model

From the reference run (`examples/chees_ecc.py`, measured by
`benchmarks/ecc_sampling_geometry.py`):

1. **Give it a long warmup.** ChEES produced about 1% divergences after
   200 warmup iterations and none after 400–600. They come from the
   step-size adaptation, not from the constraint barrier: removing the
   barrier leaves the divergence count unchanged, and it is active in
   none of the samples.
2. **Budget for slow mixing.** Transit photometry constrains a
   *combination* of $(a, b, e, \omega)$ through the transit duration, so
   the posterior is a curved, strongly correlated ridge that a diagonal
   mass matrix crawls along. At 256 chains × (400 warmup + 400 samples)
   the circular problem reaches $\hat R = 1.46$ and the eccentric one
   3.53, improving from 5.54 at 100 samples. That is slow mixing, not a
   trap: the truth is recovered within 1σ on all ten parameters.

## Gradients cost nothing until you use them

MLX computes a gradient only when a sampler asks for one, with `mx.grad`
or `mx.vjp`, and Elómë's gradient kernels recompute what they need rather
than storing a tape. So a forward-only sampler (Metropolis, the stretch
move, emcee) pays only for the forward kernel: 19.6 ms at 1024 chains ×
65,536 points on a circular orbit. A gradient sampler such as ChEES-HMC
pays 54.4 ms for the value and gradient together. That is under three
times a forward evaluation, and about 570 times cheaper than reverse-mode
automatic differentiation of the same model (see
{doc}`../reference/performance`).
