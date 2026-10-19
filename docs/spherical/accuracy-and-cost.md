# Accuracy and cost

Elómë against the other public transit codes for spherical planets:
[batman](https://github.com/lkreidberg/batman),
[PyTransit](https://github.com/hpparvi/PyTransit),
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet) and
[exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core).
All measurements are on an Apple M2 Max (12 CPU cores, 30-core GPU),
each configuration in its own process on a quiet machine
(`benchmarks/`, results in `benchmarks/RESULTS.md`).

## What each code covers

| | Elómë | batman | PyTransit | jaxoplanet | exoplanet-core |
|---|---|---|---|---|---|
| limb darkening | uniform, linear, quadratic, polynomial of any order, hybrid2/4/5 | uniform, linear, quadratic, square-root, logarithmic, exponential, power-2, nonlinear, custom | uniform, linear, quadratic, nonlinear, square-root, logarithmic, exponential, power-2, general, any function of $\mu$ | polynomial of any order | quadratic |
| the hybrid laws | closed form | numerically, as a custom law | numerically, as a function of $\mu$ | no | no |
| orbit | Keplerian, circular and eccentric | Keplerian, circular and eccentric | eccentric by a series about mid-transit | straight line at constant speed, or Keplerian | none (takes $b$ and $r$) |
| finite exposures | contact rule, supersampling | supersampling | supersampling | numerical integration over the exposure | none |
| gradients | analytic, every parameter | none | none | automatic (JAX) | analytic in $b$ and $r$ |
| hardware, precision | Apple GPU (float32), CPU (float64) | CPU, float64 | CPU (float64), OpenCL GPU (float32) | CPU or GPU (JAX), float32 or float64 | CPU, float64 |
| many parameter sets | native batch | loop | native batch | `jax.vmap` | loop |
| also | secondary eclipses, oblate and ringed planets | secondary eclipses | eclipses, phase curves, timing variations, gravity darkening | several planets, timing variations, radial velocities, stellar maps | the backend of exoplanet |
| platforms | macOS on Apple silicon | any | any | any | any |

The hybrid laws, which match real stars' limb darkening as well as the
four-parameter law or better (see {doc}`limb-darkening`), can be added to
batman and PyTransit as numerically integrated laws; Elómë evaluates
them in closed form, with gradients. In the other direction, Elómë has
no square-root, logarithmic, exponential, power-2 or nonlinear laws.

## The benchmark

A quadratic limb-darkened transit with $P = 3.456$ d,
$a/R_\star = 8.8$, $b = 0.45$, $R_p/R_\star = 0.1$ and
$u = (0.40, 0.25)$. Each code runs its own exact (or most accurate)
mode, with its JIT compilation and warm-up outside the timed region.
batman and PyTransit run on one thread (PyTransit is no faster on up to
12); JAX and Elómë's float64 path use what they choose.

**Accuracy**, as the maximum error against a 30-digit direct integral
over 241 points:

| code | circular | eccentric ($e = 0.3$) |
|---|---:|---:|
| Elómë, float64 | $2.2\times10^{-16}$ | $4.4\times10^{-16}$ |
| exoplanet-core | $3.3\times10^{-16}$ | $5.6\times10^{-16}$ |
| jaxoplanet | $1.8\times10^{-12}$ (order 50) | $4.6\times10^{-9}$ (order 10) |
| batman | $6.0\times10^{-9}$ | $5.4\times10^{-8}$ |
| PyTransit (exact) | $4.0\times10^{-8}$ | $2.9\times10^{-7}$ |
| Elómë, float32 (GPU) | $1.8\times10^{-7}$ | $1.3\times10^{-7}$ |

**One light curve**, median wall time:

| code | $N = 10^3$ | $10^5$ | $10^6$ | $10^7$ |
|---|---:|---:|---:|---:|
| Elómë, float32 (GPU) | 0.32 ms | 0.38 ms | 1.13 ms | 9.65 ms |
| Elómë, float64 (CPU) | 0.64 ms | 14.5 ms | 131 ms | 1.32 s |
| batman | 0.02 ms | 1.86 ms | 19.1 ms | 190 ms |
| jaxoplanet | 0.09 ms | 2.05 ms | 19.6 ms | 179 ms |
| exoplanet-core | 0.04 ms | 2.79 ms | 29.2 ms | 310 ms |
| PyTransit | 0.04 ms | 3.55 ms | 35.8 ms | 360 ms |

Eccentric orbits, with each code's own Kepler solver inside the timed
region, take Elómë's GPU path to 10.2 ms at $10^7$ points, against
190 ms for the fastest CPU code. Below about $10^4$ points the GPU's
fixed dispatch cost dominates and the CPU codes are faster.

**Many light curves**, 512 parameter sets × $10^5$ points each:

| code | wall time | light curves per second | memory |
|---|---:|---:|---:|
| Elómë, float32 (GPU), one call | 0.010 s | 49,092 | 208 MB |
| PyTransit (native arrays) | 0.235 s | 2,183 | 75 MB |
| jaxoplanet (`jax.vmap`, CPU) | 0.879 s | 582 | 5.9 GB |
| exoplanet-core (loop) | 1.53 s | 336 | 27 MB |
| batman (loop) | 1.77 s | 289 | 255 MB |

Elómë's throughput stays at about 55,000 light curves per second from
256 to 4,096 sets.

**Value and gradient** of a scalar loss on one light curve, in every
parameter each code exposes (only Elómë and jaxoplanet are
differentiable):

| code | $N = 10^4$ | $N = 10^5$ |
|---|---:|---:|
| Elómë, float32 (GPU) | 0.87 ms | 0.91 ms |
| jaxoplanet (CPU) | 1.90 ms | 12.3 ms |
| Elómë, float64 (CPU) | 7.34 ms | 45.6 ms |

**Memory**, the peak of the process's physical footprint (what Activity
Monitor shows) during one light curve, above its level after importing
the code: the call's working arrays and whatever the code keeps between
calls, such as compiled code and memory pools. It counts CPU and GPU
memory alike.

| code | $N = 10^5$ | $10^7$ |
|---|---:|---:|
| Elómë, float32 (GPU) | 14 MB | 279 MB |
| Elómë, float64 (CPU) | 84 MB | 7.9 GB |
| batman | 3.3 MB | 154 MB |
| jaxoplanet | 57 MB | 1.3 GB |
| exoplanet-core | 2.4 MB | 88 MB |
| PyTransit | 49 MB | 203 MB |

Elómë's float64 path holds the intermediates of its whole graph, about
0.8 KB per point.

For a sampler's batched gradients, where the fused kernels matter most,
see {doc}`../reference/performance`.
