# Finite exposures

A long exposure records the average flux over its duration, not the
flux at its midpoint. For Kepler's 29.4-minute long cadence, or TESS's
30-minute full-frame images, ignoring this distorts the ingress and
egress and biases the fitted parameters. Elómë has two ways to integrate
over the exposure, for spherical and oblate planets alike.

## Supersampling

The rule of Kipping (2010), as batman implements it: average
`supersample_factor` evenly spaced samples across each exposure,
endpoints included.

```python
m = elome.TransitModel(params, t, supersample_factor=15, exp_time=29.4 / 60 / 24)
```

It is simple and it is the default, for compatibility with batman. But
it converges slowly. The light curve's derivative jumps at each contact,
where the planet's limb crosses the star's, so uniform samples converge
only as $O(1/N)$.

## The contact rule

`integration="contact"` splits each exposure at the contact times that
fall inside it and integrates each smooth piece with Gauss-Legendre
quadrature. With the kinks at the ends of the pieces, the error falls
as about the fifth power of `n_gl`, against supersampling's first power
of N. (It is not geometric convergence: near a contact the flux still
changes as a fractional power of the time past it.)

```python
m = elome.TransitModel(params, t, exp_time=29.4 / 60 / 24,
                       integration="contact", n_gl=5)
```

Splitting at the contacts follows Agol, Luger & Foreman-Mackey (2020),
who integrate each piece by adaptive Simpson quadrature. Elómë uses a
fixed Gauss-Legendre order instead: an adaptive depth depends on the
data, which would break batching and compilation, while a fixed rule is
branchless and keeps the nodes' positions differentiable.

A 29-minute exposure of a 0.126-day transit (`benchmarks/bench_exposure.py`):

| method | evaluations per exposure | maximum error |
|---|---:|---:|
| supersampling, $N = 101$ | 101 | $1.7\times10^{-5}$ |
| supersampling, $N = 10{,}001$ | 10,001 | $1.7\times10^{-7}$ |
| **contact rule, `n_gl = 5`** | **5–25** | $\mathbf{8.9\times10^{-8}}$ |
| contact rule, `n_gl = 11` | 11–55 | $2.0\times10^{-9}$ |

Each piece takes `n_gl` evaluations: an exposure that no contact falls
inside takes `n_gl`, and only one containing all four contacts takes
5 `n_gl`. On the GPU, the empty pieces cost nothing. Even against the
most, 25 evaluations, supersampling needs about 19,500 to match, 780
times the work.

- **Spherical planets, eccentric orbits.** The contacts are found
  exactly, by Newton steps on the orbit itself rather than by the usual
  linear approximation, which can be off by minutes on an eccentric
  orbit. With them, `n_gl = 5` is accurate to $7\times10^{-7}$ or better on
  every orbit tested.
- **Oblate planets.** The contacts are the ellipse's own, found
  separately for ingress and egress. See {doc}`../oblate/light-curves`.
- **Gradients.** The quadrature nodes move with the parameters, so the
  gradient is exactly the gradient of the integral that was computed.
- **Secondary eclipses.** The eclipse on an orbit is the transit of the
  same orbit turned by 180°, at the impact parameter of the occultation:
  the planet sits opposite its place on that transit, and its outline is
  symmetric about its centre, so the eclipse's contacts and its split are
  that transit's, for spherical and oblate planets alike.

In float32 on the GPU, `TransitModel` integrates the contact rule inside
its fused kernel, as the sampler API does below.

## In the sampler API

`flux_dev_from_tau` integrates inside the GPU kernel, in registers, so
the sub-exposure axis never becomes an array in memory:

```python
dev = flux_dev_from_tau(tau, period, a, b, r, u1, u2,
                        exp_time=29.4 / 60 / 24, integration="contact", n_gl=5)
```

`integration` is `"contact"` (the default), `"supersample"` (with `n_sub`
samples), or `"none"` for instantaneous fluxes. See
{doc}`sampler-api` for what this saves.
