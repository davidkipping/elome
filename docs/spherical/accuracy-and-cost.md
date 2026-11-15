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
| platforms | macOS on Apple silicon; float64 also Linux | any | any | any | any |

The hybrid laws, which match real stars' limb darkening as well as the
four-parameter law or better (see {doc}`limb-darkening`), can be added to
batman and PyTransit as numerically integrated laws; Elómë evaluates
them in closed form, with gradients. In the other direction, Elómë has
no square-root, logarithmic, exponential, power-2 or nonlinear laws.

PyTransit appears twice below. Its quadratic model evaluates the
quadratic law's closed form. RoadRunner (Parviainen 2020), its model for
any limb-darkening law, tabulates the law and interpolates the light
curve from precomputed grids, which trades a few parts per million of
accuracy for speed.

## The benchmark

A quadratic limb-darkened transit with $P = 3.456$ d,
$a/R_\star = 8.8$, $b = 0.45$, $R_p/R_\star = 0.1$ and
$u = (0.40, 0.25)$. Each code runs its own exact (or most accurate)
mode, with its JIT compilation and warm-up outside the timed region.
batman and PyTransit run on one thread (PyTransit's quadratic model is no
faster on up to 12), and RoadRunner also on 12; JAX and Elómë's float64
path use what they choose.

**Accuracy**, as the maximum error against a 30-digit direct integral
over 241 points:

| code | circular | eccentric ($e = 0.3$) |
|---|---:|---:|
| Elómë, float64 | $2.2\times10^{-16}$ | $4.4\times10^{-16}$ |
| exoplanet-core | $3.3\times10^{-16}$ | $5.6\times10^{-16}$ |
| jaxoplanet | $1.7\times10^{-12}$ (order 50) | $4.6\times10^{-9}$ (order 10) |
| batman | $6.0\times10^{-9}$ | $5.4\times10^{-8}$ |
| PyTransit (quadratic model) | $4.0\times10^{-8}$ | $2.9\times10^{-7}$ |
| Elómë, float32 (GPU) | $1.2\times10^{-7}$ | $1.3\times10^{-7}$ |
| PyTransit (RoadRunner) | $4.6\times10^{-6}$ | $6.5\times10^{-6}$ |

**One light curve**, median wall time:

| code | $N = 10^3$ | $10^5$ | $10^6$ | $10^7$ |
|---|---:|---:|---:|---:|
| Elómë, float32 (GPU) | 0.35 ms | 0.39 ms | 1.20 ms | 9.60 ms |
| Elómë, float64 (CPU) | 0.48 ms | 14.7 ms | 134 ms | 1.35 s |
| PyTransit (RoadRunner), 12 threads | 0.21 ms | 0.43 ms | 2.70 ms | 25.0 ms |
| PyTransit (RoadRunner) | 0.03 ms | 0.99 ms | 10.2 ms | 103 ms |
| batman | 0.02 ms | 1.91 ms | 19.3 ms | 193 ms |
| jaxoplanet | 0.12 ms | 2.08 ms | 20.5 ms | 190 ms |
| exoplanet-core | 0.04 ms | 2.79 ms | 29.2 ms | 312 ms |
| PyTransit (quadratic model) | 0.04 ms | 3.61 ms | 36.0 ms | 368 ms |

RoadRunner is the fastest CPU code from $10^4$ points up, about twice as
fast as batman on one thread. Eccentric orbits, with each code's own
Kepler solver inside the timed region, take Elómë's GPU path to 10.1 ms
at $10^7$ points, against 103 ms for the fastest CPU code on one thread.
Below about $10^4$ points the GPU's fixed dispatch cost dominates and
the CPU codes are faster.

**Many light curves**, 512 parameter sets × $10^5$ points each:

| code | wall time | light curves per second | memory |
|---|---:|---:|---:|
| Elómë, float32 (GPU), one call | 0.010 s | 52,259 | 210 MB |
| PyTransit (RoadRunner), native arrays | 0.110 s | 4,667 | 122 MB |
| PyTransit (quadratic model), native arrays | 0.245 s | 2,089 | 79 MB |
| jaxoplanet (`jax.vmap`, CPU) | 0.915 s | 560 | 5.9 GB |
| exoplanet-core (loop) | 1.54 s | 333 | 87 MB |
| batman (loop) | 1.78 s | 288 | 247 MB |

PyTransit's two models run their batches on all 12 threads.

Elómë's throughput stays at about 50,000 light curves per second from
256 to 4,096 sets.

**Value and gradient** of a scalar loss on one light curve, in every
parameter each code exposes (only Elómë and jaxoplanet are
differentiable):

| code | $N = 10^4$ | $N = 10^5$ |
|---|---:|---:|
| Elómë, float32 (GPU) | 0.91 ms | 0.95 ms |
| jaxoplanet (CPU) | 1.95 ms | 12.3 ms |
| Elómë, float64 (CPU) | 7.35 ms | 45.3 ms |

**Memory**, the peak of the process's physical footprint (what Activity
Monitor shows) during one light curve, above its level after importing
the code: the call's working arrays and whatever the code keeps between
calls, such as compiled code and memory pools. It counts CPU and GPU
memory alike.

| code | $N = 10^5$ | $10^7$ |
|---|---:|---:|
| Elómë, float32 (GPU) | 16 MB | 280 MB |
| Elómë, float64 (CPU) | 84 MB | 7.9 GB |
| PyTransit (RoadRunner), 12 threads | 100 MB | 245 MB |
| PyTransit (RoadRunner) | 88 MB | 241 MB |
| batman | 3.7 MB | 154 MB |
| jaxoplanet | 57 MB | 1.3 GB |
| exoplanet-core | 2.4 MB | 88 MB |
| PyTransit (quadratic model) | 49 MB | 199 MB |

Elómë's float64 path holds the intermediates of its whole graph, about
0.8 KB per point.

For a sampler's batched gradients, where the fused kernels matter most,
see {doc}`../reference/performance`.

(law-cost)=
## The cost of the limb-darkening law

The benchmark above uses the quadratic law, which every code has. What
a law itself costs is a separate question, and comparing Elómë's hybrid
laws with another code's quadratic law would mostly compare
implementations: Elómë's float64 path evaluates an MLX array graph on
the CPU, where batman runs hand-written C point by point. Across codes,
Elómë's float64 hybrid2 takes 0.79 s at $10^7$ points, against batman's
quadratic law at 0.19 s.

So each code runs each of its laws as released, with no change to its
code, and its time is read against its own quadratic law, which cancels
the implementation. batman would need its C rebuilt to run a hybrid law,
so it runs its built-in laws. RoadRunner is left out: it tabulates any
law before computing the light curve, so its cost does not depend on the
law, and its raw speed is in the tables above. One light curve,
$N = 10^7$, each code's time against its own quadratic law
(`benchmarks/bench_law_cost.py`):

| | power-2 | hybrid2 | four-parameter | hybrid4 | hybrid5 |
|---|---:|---:|---:|---:|---:|
| Elómë, float64 (CPU) | | 0.60× | | 0.66× | 0.99× |
| Elómë, float32 (GPU) | | 0.88× | | 0.91× | 0.92× |
| batman | 14.7× | | 30.4× | | |

- **Elómë**: hybrid2 and hybrid4 cost less than the quadratic law,
  because none of their terms needs an elliptic integral, and hybrid5,
  with three poles, about the same. On the GPU a light curve's fixed
  cost dilutes the difference; at a sampler's batch size the kernel
  table in {doc}`../guides/sampler-api` shows hybrid2 at 0.59 times the
  quadratic law.
- **batman** integrates every law but the quadratic numerically, to its
  default tolerance of 1 ppm, at 15 to 30 times the cost.

The four-parameter law and hybrid4 describe real stars about equally
well (median light-curve errors of 2.0 and 1.8 ppm; see
{doc}`limb-darkening`): batman's four-parameter law costs 30 times its
quadratic law, Elómë's hybrid4 two thirds of it.

The error of each, against a 30-digit integral of the same law over
241 points:

| | quadratic | power-2 | hybrid2 | four-parameter | hybrid4 | hybrid5 |
|---|---:|---:|---:|---:|---:|---:|
| Elómë, float64 (CPU) | $2.2\times10^{-16}$ | | $1.1\times10^{-16}$ | | $1.1\times10^{-16}$ | $1.1\times10^{-16}$ |
| Elómë, float32 (GPU) | $1.2\times10^{-7}$ | | $3.1\times10^{-8}$ | | $3.1\times10^{-8}$ | $3.1\times10^{-8}$ |
| batman | $6.0\times10^{-9}$ | $1.0\times10^{-6}$ | | $1.0\times10^{-6}$ | | |
