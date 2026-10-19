# Accuracy, cost and fitting

## Accuracy

**float64** is exact to rounding. The test suite compares every
limb-darkening term against two unrelated brute-force integrations of the
occulted intensity, which agree with each other to $4\times10^{-14}$:
rays about the star's centre, and rays about the ring's centre with
closed-form radial integrals. Against them, Elómë is within
$5\times10^{-16}$ of each term's full-disc flux in every geometry tested:

- a planet inside the ring's gap, and overlapping one or both edges;
- on the stellar limb, off the star with the ring on it, and centred on
  it;
- a ring wider than its distance to the limb, crossing it four times;
- a ring larger than the star;
- face-on and nearly edge-on rings, down to $|\cos i| = 10^{-10}$;
- exact and near tangencies between the planet and a ring edge,
  including where they meet on the limb;
- spherical and oblate planets, opaque and translucent rings.

A transparent ring (`ring_tau = 0`) reproduces the spherical and oblate
light curves to $10^{-14}$, and the light curve is exactly linear in the
ring's blocked fraction.

**float32** (the GPU kernels) is within $10^{-7}$ of float64 in flux,
on every exposure rule and orbit. Its gradients in every input are
within $10^{-6}$ of float64 automatic differentiation, relative to their
scale, and typically within $10^{-7}$. The inputs are `tau`, the orbit,
`r`, `f`, `theta`, `ring_in`, `ring_out`, `ring_f`, `ring_tau` and every
weight.

As for an oblate planet, a flattening below $10^{-5}$ is evaluated as a
circle in float32 -- here also `ring_f` below $10^{-5}$, a face-on ring
-- and the derivative with respect to it is 0 in that band.

## Cost

Every outline -- the planet and both ring edges -- is integrated, and
each splits into arcs where it crosses the others, so a ringed planet
costs about 10–20 times an oblate one. Fused float32 kernels, 512 chains
× 5,000 points around a transit, a spherical planet with a ring
(`benchmarks/bench_rings.py`, M2 Max):

| | forward | vs oblate | value + gradient | vs oblate |
|---|---:|---:|---:|---:|
| hybrid2, one evaluation per point | 108 ms | 11× | 420 ms | 13× |
| hybrid4, one evaluation per point | 175 ms | 11× | 707 ms | 15× |
| hybrid5, one evaluation per point | 229 ms | 11× | 980 ms | 16× |
| hybrid2, contact rule (`n_gl = 5`) | 1.59 s | 15× | 5.57 s | 17× |
| hybrid4, contact rule (`n_gl = 5`) | 2.52 s | 16× | 9.05 s | 19× |
| hybrid5, contact rule (`n_gl = 5`) | 3.27 s | 16× | 12.2 s | 20× |

An oblate planet with the same ring costs 10–20% more. The gradients
are in every input, the ring's included; the contact-rule times include
finding each chain's kinks.

## Against other ringed-planet codes

The other public codes for ringed planets are
[pyPplusS](https://github.com/edanrein/pyPplusS),
[squishyplanet](https://github.com/ben-cassese/squishyplanet)'s
`RingedSystem`, [JoJo](https://github.com/Flippedx/JoJo)'s `ring_lc` and
[pryngles](https://github.com/seap-udea/pryngles-public)' `RingedPlanet`.

| | Elómë | pyPplusS | squishyplanet | JoJo | pryngles |
|---|---|---|---|---|---|
| limb darkening | hybrid2/4/5 | nonlinear (four coefficients), quadratic | polynomial of any order | quadratic | linear, quadratic, nonlinear |
| ring | optical depth | blocked fraction | opaque only | blocked fraction | optical depth |
| planet | spherical or oblate | spherical | triaxial | spherical | spherical |
| method | Green's theorem, closed forms | areas of polygons and segments, quadrature over the limb darkening | Green's theorem; the planet's arc by quadrature | Green's theorem; the $\mu$ term by a 300-step sum | sampling by points ("spangles") |
| precision | float64 (CPU), float32 (GPU) | float64 | float64 | float64 | float64, sampling-limited |
| input | orbit (circular, eccentric), or sky positions | sky positions | orbit | orbit | orbit |
| finite exposures | contact rule, supersampling | none | oversampling | none | none |
| gradients | analytic, every parameter | none | automatic (JAX) | none | none |
| many parameter sets | native batch | loop | `jax.vmap` | loop | loop |
| also | | any convex occultor (in its paper) | | contact times | reflected light, shadows, polarisation, moons |

The benchmark is the spherical transit of {doc}`../spherical/accuracy-and-cost`
with a ring from 1.4 to 2.6 planet radii, `ring_f = 0.7` and its long
axis at 25° to the motion, so the ring crosses the planet's disc; and
the one limb-darkening profile every code expresses exactly,
$I(\mu) = 1 - 0.5\,(1 - \mu^2)$. The reference is an independent float64
ray integral (`benchmarks/ring_compare/`):

| | Elómë float32 (GPU) | pyPplusS | squishyplanet | JoJo | pryngles | Elómë float64 (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| error, opaque ring | $1.0\times10^{-7}$ | $5.6\times10^{-7}$ | $5.6\times10^{-16}$ | $1.1\times10^{-14}$ | $4.7\times10^{-3}$ | $2.2\times10^{-16}$ |
| error, translucent ($\tau = 0.6$) | $1.1\times10^{-7}$ | $4.8\times10^{-7}$ | opaque only | $9.7\times10^{-15}$ | $4.1\times10^{-3}$ | $1.1\times10^{-16}$ |
| error, nearly edge-on (`ring_f = 0.95`) | $1.0\times10^{-7}$ | $4.5\times10^{-6}$ | $7.8\times10^{-16}$ | $8.6\times10^{-15}$ | $9.6\times10^{-4}$ | $1.1\times10^{-16}$ |
| one light curve, $N = 10^3$ | 1.6 ms | 0.46 s | 43 ms | 69 ms | 0.12 s | 73 ms |
| one light curve, $N = 10^5$ | 4.0 ms | 38 s | 4.1 s | 7.2 s | 12 s | 4.1 s |
| one light curve, $N = 10^6$ | 34 ms | | 40 s | 86 s | | 41 s |
| 512 sets × $10^4$ points | 0.21 s | about 2,000 s | about 510 s | about 350 s | about 2,700 s | |
| value + gradient, $N = 10^5$ | 17 ms | no gradients | 20 s | no gradients | no gradients | 11 s |
| memory, $N = 10^5$ | 13 MB | 111 MB | 255 MB | 125 MB | 75 MB | 1.1 GB |
| memory, $N = 10^6$ | 34 MB | | 548 MB | 26 GB | | 1.4 GB |

The batches of the slower codes were timed on 8 to 32 sets and scaled to
512. pyPplusS is shown at its default order; at order 20 its error is
$4\times10^{-10}$ and it takes about 3.5 times as long. pryngles samples
5,000 points on the planet and 5,000 on the ring. Memory is measured as
for oblate planets (see {doc}`../oblate/accuracy-and-cost`); Elómë's
float64 path evaluates long light curves in blocks of 32,768 points,
which caps it near 1.4 GB.

Two of these codes have limitations worth knowing, in the versions
benchmarked (pryngles 1.0.2, JoJo 0.1.0):

- **pryngles** adds the ring's blocked light to the planet's even where
  the ring lies over the planet's disc, so that area counts twice; that
  is most of its error above. Against an exact model with the same
  double count it agrees to its sampling noise ($3\times10^{-5}$), and
  where the ring does not cross the planet's disc it is accurate to
  that noise.
- **JoJo** is exact when the planet's disc crosses the ring band, as
  here. When the planet sits inside the ring's gap, `ring_lc` makes the
  ring $1/\sqrt{1 - f_r}$ too large, about 1% of the flux for
  $f_r = 0.2$.

## Fitting a ringed planet

Everything in {doc}`../oblate/fitting` about oblate planets applies:
batch every chain into one call with `flux_dev_from_tau`, wrap `theta`
onto $[0, \pi)$, and fold it onto $[0, \pi/2]$ for a near-central
transit. What is specific to a ring:

- **Sample `ring_f` uniformly** on $[0, 1)$ for an isotropic prior on the
  ring's orientation.
- **Optical depth and tilt trade off.** On a nearly edge-on ring the
  blocked fraction $1 - e^{-\tau/(1-\mathtt{ring\_f})}$ saturates, and
  the ring's area shrinks; a thin opaque ring and a wider translucent one
  can look alike. A prior on `ring_tau` (or sampling the blocked fraction
  directly and deriving `ring_tau`) helps.
- **Keep chains in the domain.** `ring_in` must reach past the planet's
  equator, `ring_in` $\ge 1/\sqrt{1-f}$, and `ring_out` must exceed
  `ring_in`; outside, an MLX array gives NaN. Sampling, for example,
  `ring_in` and the ratio `ring_out / ring_in` with bounded priors keeps
  them there by construction.
- **Expect correlations with the planet.** A ring's ingress and egress
  overlap the planet's, so `rp`, `ring_in` and the impact parameter
  correlate, more strongly the closer the ring is to face-on.
