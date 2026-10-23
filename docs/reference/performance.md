# Performance and accuracy

All measurements are on an Apple M2 Max (12 CPU cores, 30-core GPU), each
configuration in its own process on a quiet machine. The scripts are in
the repository's `benchmarks/` directory, and each table names the one
that produced it.

## Against other codes

Each planet type is compared with the other public codes for it -- what
each covers, accuracy, speed, batches and gradients -- in its own
section:

- spherical planets, against batman, PyTransit, jaxoplanet and
  exoplanet-core: {doc}`../spherical/accuracy-and-cost`;
- oblate planets, against squishyplanet, JoJo, GreenLantern and
  eclipsoid: {doc}`../oblate/accuracy-and-cost`;
- ringed planets, against pyPplusS, squishyplanet, JoJo and pryngles:
  {doc}`../rings/accuracy-and-fitting`;
- planets crossing starspots, against jaxoplanet and spotter:
  {doc}`../spots/accuracy-and-fitting`.

In short, Elómë's float32 GPU kernels are the fastest code for oblate
and ringed planets at every size measured, and for spherical planets
past about $10^4$ points, below which the GPU's fixed dispatch cost
dominates and the fastest CPU codes win. For a planet crossing
starspots they are faster than jaxoplanet and spotter at every
resolution measured, and Elómë is the one of the three whose accuracy
is not set by a resolution. For batches of parameter sets the margin is
from about 20 times (spherical, against PyTransit) to about 1,700 times
(ringed). A float32 light curve of $10^7$ points holds
under 300 MB of memory for each planet type. The float64 path is exact
to rounding.

## Gradients

Of the spherical codes only Elómë and jaxoplanet are differentiable,
and Elómë's gradients are analytic, fused into Metal kernels. For the gradient of a chunked
log-likelihood at 1024 chains × 65,536 points, compiled, in float32
(`benchmarks/bench_vjp.py`, which records them in `benchmarks/vjp.json`):

| | forward | value + gradient |
|---|---:|---:|
| reverse-mode automatic differentiation | 492 ms | 31,264 ms |
| analytic gradient (MLX graph) | 496 ms | 627 ms |
| **fused Metal kernel**, circular orbit | **19.6 ms** | **54.4 ms** |
| **fused Metal kernel**, eccentric ($e = 0.3$) | **25.7 ms** | **68.6 ms** |

The fused kernel keeps the whole model (orbit, Kepler solve and
photometry) in registers, with about 12 bytes of memory traffic per
point. That is 25 times faster than the same model as a compiled MLX graph,
and about 570 times faster for the gradient than reverse-mode
differentiation. End to end, a 1024-chain ChEES-HMC
injection-recovery fit runs at about 13 effective samples per second
with no divergences.

## Exposure integration, limb darkening, oblate and ringed planets

- Exposure rules: {doc}`../guides/exposures` and
  {doc}`../guides/sampler-api`.
- The hybrid laws against the quadratic law: {doc}`../guides/sampler-api`.
- Oblate and ringed planets' own cost: {doc}`../oblate/accuracy-and-cost`
  and {doc}`../rings/accuracy-and-fitting`.
