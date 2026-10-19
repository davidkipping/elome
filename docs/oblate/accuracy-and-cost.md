# Accuracy and cost

## Accuracy

**float64** is exact to rounding. In the cross-code benchmark below, light
curves agree with a 30-digit direct integral of the occulted intensity
to $1\times10^{-16}$. The test suite checks every limb-darkening term
against an independent high-precision quadrature over every way an
ellipse can overlap the disc, including the limb and pole tangencies
where closed forms are most fragile. It also checks light curves on
circular and eccentric orbits, and that $f = 0$ reproduces the spherical
result (to $2\times10^{-17}$).

**float32** (the GPU kernels) is within $2\times10^{-7}$ of float64 in
flux. Its gradients in every input are within $10^{-3}$ of float64
automatic differentiation, and typically within $10^{-5}$. The inputs
are `tau`, `period`, `a`, `b`, `r`, `f`, `theta`, every weight, and
`secosw` and `sesinw`.

### Where float32 has limits

- **Very small flattening.** Below $f = 10^{-5}$ a float32 evaluation
  uses the spherical forms, because the flattening moves the flux by less
  than $4\times10^{-8}$ there. So $\partial F/\partial f = 0$ in that
  band, and just above it the derivative is good to about 1%, improving
  with $f$. A sampler started at exactly $f = 0$ leaves it on the
  likelihood, not on the gradient; start at a small positive $f$
  instead.
- **Within about $10^{-6}$ of a limb contact.** float32 cannot resolve
  the sliver of overlap there. The flux lost is about $10^{-13}$ and the
  gradient error about $10^{-4}$, in a window of about $10^{-6}$ in
  orbital phase.

## Cost

A spherical planet's light curve is a closed form at each point. An
oblate planet's also needs the roots of a quartic per point and per
limb-darkening pole (and one more where the planet crosses the limb),
which the GPU kernel seeds from the closed-form (Ferrari) solution and
polishes. Fused float32 kernels, 512 chains × 5,000 points around a
transit (`benchmarks/bench_oblate.py`, M2 Max):

| | forward | vs spherical | value + gradient | vs spherical |
|---|---:|---:|---:|---:|
| hybrid2, one evaluation per point | 9.9 ms | 15× | 31.5 ms | 22× |
| hybrid4, one evaluation per point | 15.7 ms | 20× | 46.4 ms | 29× |
| hybrid5, one evaluation per point | 20.0 ms | 22× | 62.3 ms | 32× |
| hybrid2, contact rule (`n_gl = 5`) | 108 ms | 19× | 323 ms | 21× |
| hybrid4, contact rule (`n_gl = 5`) | 157 ms | 19× | 467 ms | 20× |
| hybrid5, contact rule (`n_gl = 5`) | 205 ms | 20× | 621 ms | 21× |

Under the contact rule, an exposure that no contact falls inside costs
`n_gl` evaluations; one containing a contact costs `n_gl` more per
contact. In absolute terms, an oblate light curve of a few thousand
points takes milliseconds.

## Against other oblate-planet codes

The other public codes for oblate planets are
[squishyplanet](https://github.com/ben-cassese/squishyplanet),
[JoJo](https://github.com/Flippedx/JoJo),
[GreenLantern](https://github.com/emprice/greenlantern) and
[eclipsoid](https://pypi.org/project/eclipsoid/).

| | Elómë | squishyplanet | JoJo | GreenLantern | eclipsoid |
|---|---|---|---|---|---|
| limb darkening | hybrid2/4/5 | polynomial of any order | quadratic | quadratic | polynomial of any order |
| method | Green's theorem, closed forms | Green's theorem; the planet's arc by quadrature | Green's theorem; the $\mu$ term by a 100-step sum | Green's theorem, Simpson's rule (2,048 steps) | Green's theorem; the planet's arc by quadrature |
| precision | float64 (CPU), float32 (GPU) | float64 | float64 | float32 | float64 |
| gradients | analytic, every parameter | automatic (JAX) | none | forward mode, circular orbits | automatic (JAX, reverse mode) |
| planet | projected ellipse | triaxial, or projected ellipse | projected ellipse | triaxial | projected ellipse, or ellipsoid |
| orbit | circular and eccentric | circular and eccentric | circular and eccentric | circular (eccentric: values only) | jaxoplanet's orbits |
| finite exposures | contact rule, supersampling | oversampling | supersampling | binning | jaxoplanet's integration |
| many parameter sets | native batch | `jax.vmap` | loop | native batch | `jax.vmap` |
| hardware | Apple GPU, CPU | CPU | CPU | OpenCL GPU | CPU |
| also | rings | rings | rings | tidally locked planets | stellar surface maps |

The benchmark is the spherical one's transit with $f = 0.2$ at 30°, and
the one limb-darkening profile all five codes express exactly,
$I(\mu) = 1 - 0.5\,(1 - \mu^2)$ (`benchmarks/oblate_compare/`):

| | Elómë float32 (GPU) | GreenLantern (GPU) | JoJo | squishyplanet | eclipsoid | Elómë float64 (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| error vs 30-digit integral | $7\times10^{-8}$ | $4.5\times10^{-6}$ | $2\times10^{-15}$ | $1.3\times10^{-15}$ | $1.6\times10^{-9}$ | $1\times10^{-16}$ |
| one light curve, $N = 10^3$ | 0.44 ms | 0.79 ms | 0.64 ms | 3.9 ms | 8.8 ms | 13 ms |
| one light curve, $N = 10^5$ | 0.83 ms | 24 ms | 45 ms | 373 ms | 889 ms | 471 ms |
| one light curve, $N = 10^7$ | 35 ms | 2.3 s | 6.2 s | 40 s | | 47 s |
| 512 sets × $10^4$ points | 0.020 s | 1.2 s | 2.6 s | 29 s | 46 s | |
| value + gradient, $N = 10^5$ | 2.7 ms | 54 ms | no gradients | 2.3 s | 1.5 s | 1.1 s |
| memory, $N = 10^5$ | 13 MB | 14 MB | 31 MB | 127 MB | 37 MB | 288 MB |
| memory, $N = 10^7$ | 239 MB | 89 MB | 9.4 GB | 1.7 GB | | 21 GB |
| memory, 512 sets × $10^4$ | 90 MB | 14 MB | 50 MB | 691 MB | 4.3 GB | |

eclipsoid was timed up to $10^6$ points (9.0 s, 853 MB). Its largest
error is near second and third contact; elsewhere it is exact to
rounding.

Memory is the peak of the process's physical footprint (what Activity
Monitor shows) during a call, above its level after importing the code:
the call's working arrays and whatever the code keeps between calls,
such as compiled code and memory pools. It counts CPU and Apple GPU
memory alike; GreenLantern's OpenCL buffers count only as far as macOS
attributes them to the process. Elómë's float64 path holds its whole
graph's intermediates, about 2 KB per point.

Elómë's float32 kernel is the fastest at every size measured; below
about $10^4$ points its time is mostly the GPU's fixed dispatch cost.
The other codes cover what Elómë does not: the standard quadratic law
(JoJo, GreenLantern, and squishyplanet and eclipsoid through their
polynomial laws), polynomial laws of any order (squishyplanet,
eclipsoid), triaxial planets (squishyplanet, GreenLantern), and stellar
surface maps (eclipsoid).
