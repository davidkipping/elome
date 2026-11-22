# Key concepts

Five ideas run through the rest of the documentation.

## Two precisions, chosen by the data's dtype

| | float32 | float64 |
|---|---|---|
| where it runs | the Apple-Silicon GPU | the CPU |
| how | hand-fused Metal kernels | an MLX graph of the closed-form solution |
| accuracy | about $10^{-7}$ in flux | rounding, about $10^{-16}$ |
| use it for | everything: sampling, batches, long light curves | reference values and checks; machines without the GPU kernels |

The two paths compute the same function. The dtype of the *data* decides
which one runs. For the batman-style frontend that is
`TransitModel(params, t, dtype=...)`, which defaults to float32 where the
GPU kernels run (a Mac with Apple Silicon) and to float64 elsewhere. For the
array functions, float32 data (an `mx.array` or a numpy `float32`
array) takes the GPU kernels, and anything else is float64 on the CPU.
{doc}`../guides/gradients-and-precision` has the full rule.

## Two levels of API

- **The batman-style frontend.** `TransitParams` holds the parameters,
  `TransitModel` is built once for a set of times, and
  `light_curve(params)` returns the flux. `light_curves` evaluates many
  parameter sets in one call, and `light_curve_mx` is the differentiable
  version. Use it for light curves, plots, and moderate batches.
- **The array API for samplers.** `flux_dev_from_tau` and the model
  builders in `elome.anvil` take per-chain parameter arrays and return
  per-chain light curves with analytic gradients, in one GPU launch per
  sampler step. See {doc}`../guides/sampler-api`.

## Deviations, not fluxes

Functions whose names contain `flux_dev` return $F - 1$, which is
exactly 0 out of transit and negative in it, rather than the flux $F$.
In float32 this keeps the full precision for the transit signal, which
would otherwise be a small difference from 1.

## Batch everything

A GPU dispatch costs a fixed 0.2-0.7 ms however little work it carries.
A loop that evaluates one parameter set at a time pays that cost every
time and is two to three orders of magnitude slower than one call that
evaluates them all. See {doc}`../guides/batching`.

## Units and conventions

- Times and periods are in days; `t0` is the time of inferior
  conjunction (mid-transit).
- `rp` and `a` are in units of the stellar radius.
- Angles are **degrees** in `TransitParams` (`inc`, `w`, `theta`) and
  **radians** in the array API (`theta` in `flux_dev_from_tau`).
- The array API describes the orbit by the impact parameter `b` rather
  than the inclination, $b = a\cos i\,(1 - e^2)/(1 + e\sin\omega)$
  (just $a\cos i$ on a circular orbit). For eccentric orbits it uses
  $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$, which samplers handle well.
