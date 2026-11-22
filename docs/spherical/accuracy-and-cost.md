# Accuracy and cost

Elómë against the other public transit codes for spherical planets:
[batman](https://github.com/lkreidberg/batman),
[PyTransit](https://github.com/hpparvi/PyTransit),
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet),
[exoplanet-core](https://github.com/exoplanet-dev/exoplanet-core) and
[pizzetti](https://github.com/hippke/pizzetti).

```{admonition} The machine, and how to reproduce these numbers
:class: note

Every number on this page was measured on an **Apple M2 Max** (12 CPU
cores: 8 performance and 4 efficiency; 30-core GPU; 64 GB of memory)
under macOS, each configuration in its own process on a quiet machine.
The scripts are
[`bench_precision.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_precision.py),
[`bench_speed.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_speed.py)
and
[`bench_law_cost.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_law_cost.py)
in the repository's
[`benchmarks/`](https://github.com/davidkipping/elome/tree/main/benchmarks)
directory, with every result in
[`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/RESULTS.md).
To run them on your own machine, see {doc}`../reference/benchmarks`.
```

## What each code covers

| | Elómë | batman | PyTransit | jaxoplanet | exoplanet-core | pizzetti |
|---|---|---|---|---|---|---|
| limb darkening | uniform, linear, quadratic, polynomial of any order, hybrid2/4/5 | uniform, linear, quadratic, square-root, logarithmic, exponential, power-2, nonlinear, custom | uniform, linear, quadratic, nonlinear, square-root, logarithmic, exponential, power-2, general, any function of $\mu$ | polynomial of any order | quadratic | uniform, linear, quadratic, square-root, nonlinear, power-2 |
| the hybrid laws | closed form | numerically, as a custom law (its C rebuilt) | RoadRunner, tabulated from a function of $\mu$ | no | no | no (sums of powers of $\mu$ only) |
| orbit | Keplerian, circular and eccentric | Keplerian, circular and eccentric | eccentric by a series about mid-transit | straight line at constant speed, or Keplerian | none (takes $b$ and $r$) | Keplerian, circular and eccentric |
| finite exposures | contact rule, supersampling | supersampling | supersampling | numerical integration over the exposure | none | supersampling |
| gradients | analytic, every parameter | none | none | automatic (JAX) | analytic in $b$ and $r$ | none (exact ones in its Julia package) |
| hardware, precision | Apple GPU (float32); CPU (float64) as a fallback | CPU, float64 | CPU (float64), OpenCL GPU (float32) | CPU or GPU (JAX), float32 or float64 | CPU, float64 | CPU, float64 |
| many parameter sets | native batch | loop | native batch | `jax.vmap` | loop | loop |
| also | secondary eclipses, oblate, tidally locked and ringed planets, starspots, moons | secondary eclipses | eclipses, phase curves, timing variations, gravity darkening | several planets, timing variations, radial velocities, stellar maps | the backend of exoplanet | secondary eclipses, a proven error bound, limb darkening and baseline integrated out, Gaussian-process noise |
| platforms | macOS on Apple silicon; Linux on the CPU, without the kernels | any | any | any | any | any |

The hybrid laws, which match real stars' limb darkening as well as the
four-parameter law or better (see {doc}`limb-darkening`), can be added to
batman as a numerically integrated law and to PyTransit's RoadRunner as
a tabulated one; Elómë evaluates them in closed form, with gradients. In
the other direction, Elómë has no square-root, logarithmic, exponential,
power-2 or nonlinear laws.

PyTransit appears twice below. Its closed-form model evaluates the
quadratic law's closed form. RoadRunner (Parviainen 2020), its model for
any limb-darkening law, tabulates the law and interpolates the light
curve from precomputed grids, which trades a few parts per million of
accuracy for speed. pizzetti takes the flux between second and third
contact from a series in $1/(1 - z^2)$ whose truncation it bounds, by
default to $10^{-9}$ of the stellar flux, and the limb from the exact
solution.

## The benchmark

A transit with $P = 3.456$ d, $a/R_\star = 8.8$, $b = 0.45$ and
$R_p/R_\star = 0.1$, under a two-parameter limb-darkening law: the
quadratic law with $u = (0.40, 0.25)$, which every code has, and hybrid2
with $w = (0.5, 0.2)$ for the codes that can take it. Elómë evaluates
hybrid2 in closed form; RoadRunner takes it as a function of $\mu$ with
its exact disc integral and tabulates it, as it does any law. The others
cannot: batman integrates a custom law only once its C is rebuilt,
pizzetti and jaxoplanet take sums of powers of $\mu$, and exoplanet-core
the quadratic law. Each row names its law.

A light curve of $N$ points is $N$ times spaced evenly over 0.24 d
centred on mid-transit: the transit and its out-of-transit shoulders,
with 52.5% of the points in transit (between first and fourth contact,
0.126 d).

Each code runs its own exact (or most accurate) mode, pizzetti its
default, with its JIT compilation and half a second of warm-up calls
outside the timed region (the warm-up brings the GPU to its working
clock: macOS slows it after a second idle). batman, pizzetti and
PyTransit run on one thread (PyTransit's closed-form model is no faster
on up to 12), and RoadRunner also on 12; JAX and Elómë's float64 path
use what they choose.

Elómë's float64 path comes last in each table, set back in grey: it is
the CPU fallback, and the exact reference that the GPU kernels are
checked against, not the way Elómë is meant to be run. Its numbers are
here for completeness.

**Accuracy**, as the maximum error against a 30-digit direct integral
of each row's own law, over 241 points:

| code | circular | eccentric ($e = 0.3$) |
|---|---:|---:|
| exoplanet-core, quadratic | $3.3\times10^{-16}$ | $5.6\times10^{-16}$ |
| pizzetti (series off), quadratic | $4.4\times10^{-16}$ | $4.4\times10^{-16}$ |
| jaxoplanet, quadratic | $1.7\times10^{-12}$ (order 50) | $4.6\times10^{-9}$ (order 10) |
| pizzetti, quadratic | $7.1\times10^{-10}$ | $5.4\times10^{-10}$ |
| batman, quadratic | $6.0\times10^{-9}$ | $5.4\times10^{-8}$ |
| Elómë, float32 (GPU), hybrid2 | $3.1\times10^{-8}$ | $3.1\times10^{-8}$ |
| PyTransit (closed form), quadratic | $4.0\times10^{-8}$ | $2.9\times10^{-7}$ |
| Elómë, float32 (GPU), quadratic | $1.2\times10^{-7}$ | $1.3\times10^{-7}$ |
| PyTransit (RoadRunner), quadratic | $4.6\times10^{-6}$ | $6.5\times10^{-6}$ |
| PyTransit (RoadRunner), hybrid2 | $5.4\times10^{-6}$ | $7.6\times10^{-6}$ |
| [Elómë, float64 (CPU), quadratic]{.fallback} | $2.2\times10^{-16}$ | $3.3\times10^{-16}$ |
| [Elómë, float64 (CPU), hybrid2]{.fallback} | $1.1\times10^{-16}$ | $2.2\times10^{-16}$ |

pizzetti with its series turned off is the exact solution everywhere.

**One light curve** of $N$ points, median wall time:

| code | $N = 10^3$ | $10^5$ | $10^6$ | $10^7$ |
|---|---:|---:|---:|---:|
| Elómë, float32 (GPU), hybrid2 | 0.25 ms | 0.26 ms | 0.87 ms | 8.21 ms |
| Elómë, float32 (GPU), quadratic | 0.33 ms | 0.38 ms | 1.02 ms | 9.49 ms |
| PyTransit (RoadRunner), 12 threads, quadratic | 0.21 ms | 0.43 ms | 2.70 ms | 25.0 ms |
| PyTransit (RoadRunner), 12 threads, hybrid2 | 0.24 ms | 0.42 ms | 2.72 ms | 25.3 ms |
| pizzetti, quadratic | 0.01 ms | 0.40 ms | 5.11 ms | 45.6 ms |
| PyTransit (RoadRunner), quadratic | 0.03 ms | 0.99 ms | 10.2 ms | 103 ms |
| PyTransit (RoadRunner), hybrid2 | 0.03 ms | 1.04 ms | 10.3 ms | 105 ms |
| jaxoplanet, quadratic | 0.12 ms | 2.08 ms | 20.5 ms | 190 ms |
| batman, quadratic | 0.02 ms | 1.91 ms | 19.3 ms | 193 ms |
| exoplanet-core, quadratic | 0.04 ms | 2.79 ms | 29.2 ms | 312 ms |
| PyTransit (closed form), quadratic | 0.04 ms | 3.61 ms | 36.0 ms | 368 ms |
| [Elómë, float64 (CPU), quadratic]{.fallback} | 0.48 ms | 14.7 ms | 134 ms | 1.35 s |
| [Elómë, float64 (CPU), hybrid2]{.fallback} | 0.30 ms | 6.81 ms | 79.5 ms | 666 ms |

Elómë's GPU path is the fastest code from about $10^5$ points up. At
$10^7$ points it takes 9.5 ms, and 8.2 ms with hybrid2: five times
faster than the fastest CPU code, pizzetti on one thread, and 10 to 20
times faster than RoadRunner and batman on one thread. Eccentric orbits,
with each code's own Kepler solver inside the timed region, take Elómë's
GPU path to 10.4 ms at $10^7$ points, against 46.2 ms for the fastest CPU
code on one thread. hybrid2 costs Elómë less than the quadratic law,
since none of its terms needs an elliptic integral, and RoadRunner the
same, since it tabulates any law.

Below $10^5$ points the GPU's fixed cost of about 0.3 ms per call
dominates, and the CPU codes are faster there; of those, pizzetti is the
fastest at every size.

**Many light curves**, 512 parameter sets × $10^5$ points each, over the
same window:

| code | wall time | light curves per second | memory |
|---|---:|---:|---:|
| Elómë, float32 (GPU), one call, hybrid2 | 0.006 s | 80,676 | 602 MB |
| Elómë, float32 (GPU), one call, quadratic | 0.010 s | 52,470 | 212 MB |
| PyTransit (RoadRunner), native arrays, hybrid2 | 0.107 s | 4,799 | 126 MB |
| PyTransit (RoadRunner), native arrays, quadratic | 0.110 s | 4,667 | 122 MB |
| PyTransit (closed form), native arrays, quadratic | 0.245 s | 2,089 | 79 MB |
| pizzetti (loop), quadratic | 0.441 s | 1,161 | 119 MB |
| jaxoplanet (`jax.vmap`, CPU), quadratic | 0.915 s | 560 | 5.9 GB |
| exoplanet-core (loop), quadratic | 1.54 s | 333 | 87 MB |
| batman (loop), quadratic | 1.78 s | 288 | 247 MB |

Elómë evaluates the 512 light curves in one call, in 6 to 10 ms: 52,000
light curves per second under the quadratic law and 81,000 under
hybrid2, 11 to 17 times the fastest CPU code (RoadRunner on 12 threads)
and 45 to 70 times pizzetti. PyTransit's two models run their batches on
all 12 threads.

Elómë's throughput stays at about 50,000 light curves per second from
256 to 4,096 sets, under the quadratic law.

**Value and gradient** of a scalar loss on one light curve of $N$ points,
in every parameter each code exposes. Only Elómë and jaxoplanet are
differentiable in every parameter, and Elómë's GPU path takes about 1 ms
for both laws, two to twelve times faster than jaxoplanet:

| code | $N = 10^4$ | $N = 10^5$ |
|---|---:|---:|
| Elómë, float32 (GPU), quadratic | 0.95 ms | 1.00 ms |
| Elómë, float32 (GPU), hybrid2 | 0.94 ms | 0.99 ms |
| jaxoplanet (CPU), quadratic | 1.95 ms | 12.3 ms |
| [Elómë, float64 (CPU), quadratic]{.fallback} | 7.35 ms | 45.3 ms |
| [Elómë, float64 (CPU), hybrid2]{.fallback} | 3.98 ms | 22.8 ms |

**Memory**, the peak of the process's physical footprint (what Activity
Monitor shows) during one light curve of $N$ points, above its level after importing
the code: the call's working arrays and whatever the code keeps between
calls, such as compiled code and memory pools. It counts CPU and GPU
memory alike.

| code | $N = 10^5$ | $10^7$ |
|---|---:|---:|
| Elómë, float32 (GPU), hybrid2 | 21 MB | 317 MB |
| Elómë, float32 (GPU), quadratic | 21 MB | 357 MB |
| PyTransit (RoadRunner), 12 threads, quadratic | 96 MB | 245 MB |
| PyTransit (RoadRunner), 12 threads, hybrid2 | 107 MB | 249 MB |
| pizzetti, quadratic | 4.2 MB | 209 MB |
| PyTransit (RoadRunner), quadratic | 88 MB | 241 MB |
| PyTransit (RoadRunner), hybrid2 | 95 MB | 243 MB |
| jaxoplanet, quadratic | 57 MB | 1.3 GB |
| batman, quadratic | 3.7 MB | 154 MB |
| exoplanet-core, quadratic | 2.4 MB | 88 MB |
| PyTransit (closed form), quadratic | 49 MB | 199 MB |
| [Elómë, float64 (CPU), quadratic]{.fallback} | 84 MB | 7.9 GB |
| [Elómë, float64 (CPU), hybrid2]{.fallback} | 27 MB | 2.3 GB |

Elómë's float64 path holds the intermediates of its whole graph, about
0.8 KB per point.

For a sampler's batched gradients, where the fused kernels matter most,
see {doc}`../reference/performance`.

(law-cost)=
## The cost of the limb-darkening law

The benchmark above uses the quadratic law, which every code has, and
hybrid2 where a code can take it. What a law itself costs, within a code,
is a separate question, and comparing Elómë's hybrid laws with another
code's quadratic law would mostly compare implementations: Elómë's float64 path evaluates an MLX array graph on
the CPU, where batman runs hand-written C point by point. Across codes,
Elómë's float64 hybrid2 takes 0.79 s at $10^7$ points, against batman's
quadratic law at 0.19 s.

So each code runs each of its laws as released, with no change to its
code, and its time is read against its own quadratic law, which cancels
the implementation. batman would need its C rebuilt to run a hybrid law,
and pizzetti takes only sums of powers of $\mu$, so both run their
built-in laws. RoadRunner is left out: it tabulates any
law before computing the light curve, so its cost does not depend on the
law, and its raw speed is in the tables above. One light curve,
$N = 10^7$, each code's time against its own quadratic law
([`benchmarks/bench_law_cost.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_law_cost.py)):

| | power-2 | hybrid2 | four-parameter | hybrid4 | hybrid5 |
|---|---:|---:|---:|---:|---:|
| Elómë, float32 (GPU) | | 0.88× | | 0.91× | 0.92× |
| batman | 14.7× | | 30.4× | | |
| pizzetti | 2.09× | | 2.63× | | |
| [Elómë, float64 (CPU)]{.fallback} | | 0.60× | | 0.66× | 0.99× |

- **Elómë**: hybrid2 and hybrid4 cost less than the quadratic law,
  because none of their terms needs an elliptic integral, and hybrid5,
  with three poles, about the same. On the GPU a light curve's fixed
  cost dilutes the difference; at a sampler's batch size the kernel
  table in {doc}`../guides/sampler-api` shows hybrid2 at 0.59 times the
  quadratic law.
- **batman** integrates every law but the quadratic numerically, to its
  default tolerance of 1 ppm, at 15 to 30 times the cost.
- **pizzetti** keeps its interior series for every law and takes the
  limb of power-2 and the four-parameter law by a fixed-node quadrature
  instead of the quadratic law's closed form, at two to two and a half
  times the cost, and to $3\times10^{-10}$.

The four-parameter law and hybrid4 describe real stars about equally
well (median light-curve errors of 2.0 and 1.8 ppm; see
{doc}`limb-darkening`): batman's four-parameter law costs 30 times its
quadratic law, pizzetti's two and a half times, Elómë's hybrid4 two
thirds of it.

The error of each, against a 30-digit integral of the same law over
241 points:

| | quadratic | power-2 | hybrid2 | four-parameter | hybrid4 | hybrid5 |
|---|---:|---:|---:|---:|---:|---:|
| Elómë, float32 (GPU) | $1.2\times10^{-7}$ | | $3.1\times10^{-8}$ | | $3.1\times10^{-8}$ | $3.1\times10^{-8}$ |
| batman | $6.0\times10^{-9}$ | $1.0\times10^{-6}$ | | $1.0\times10^{-6}$ | | |
| pizzetti | $7.1\times10^{-10}$ | $3.1\times10^{-10}$ | | $3.1\times10^{-10}$ | | |
| [Elómë, float64 (CPU)]{.fallback} | $2.2\times10^{-16}$ | | $1.1\times10^{-16}$ | | $1.1\times10^{-16}$ | $1.1\times10^{-16}$ |
