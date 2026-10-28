# Oblate light curves

## With the frontend

Set `f` and `theta` on `TransitParams` and choose a hybrid law. Everything
else is as for a spherical planet:

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 87.0, 0.0, 90.0
p.limb_dark, p.u = "hybrid4", [0.3, 0.2, 0.1, 0.1]
p.f, p.theta = 0.3, 35.0                      # theta in degrees

t = np.linspace(-0.12, 0.12, 2000)
m = elome.TransitModel(p, t, exp_time=29.4 / 60 / 24, integration="contact")
flux = m.light_curve(p)
```

A model is built for an oblate or a spherical planet: `f` is fixed per
model, like the limb-darkening law, while its *value* and `theta` can
change between calls.

- `light_curves(sets)` takes per-set `f` and `theta`, like any other
  field. See {doc}`../guides/batching`.
- `light_curve_mx(params)` is differentiable in `f` and `theta` as well as
  in every orbital and limb-darkening parameter. See
  {doc}`../guides/gradients-and-precision`.
- `dtype=mx.float32` runs on the GPU kernels.

## Secondary eclipses

`transittype="secondary"` gives the eclipse of the planet by the star,
with the planet a uniform disc of flux ratio `fp`, as for a spherical
planet: the flux is $1 + f_p$ out of eclipse, 1 in totality, and
$1 + f_p$ times the visible fraction of the planet's outline in between.
The planet carries no limb darkening of its own, so the model takes any
law and ignores it. `theta` is the same sky angle as at transit: the
outline's projection does not change along the orbit.

```python
p.fp = 1e-3
t2 = p.t0 + 0.5 * p.per + np.linspace(-0.12, 0.12, 2000)   # circular orbit
m2 = elome.TransitModel(p, t2, transittype="secondary")
eclipse = m2.light_curve(p)
```

The orbit places the eclipse, as for a spherical planet (see
{doc}`../spherical/light-curves`). Use `supersample_factor` for finite
exposures; the contact rule is for primary transits. The eclipse runs as
an MLX graph in float32 as well as float64, not on the GPU kernels.

## Choosing the limb-darkening law

An oblate planet takes `hybrid2`, `hybrid4` or `hybrid5`, or a custom
`HybridLaw`. Asking for the quadratic law with `f` set raises an error.

Prefer **`hybrid4`** or **`hybrid5`**. Limb darkening and oblateness both
shape the ingress and egress, so a law that cannot follow the star's
limb (the quadratic law, or `hybrid2`) can leave residuals that a fit
then attributes to oblateness: a confident oblateness signal from a
spherical planet. The four- and five-weight laws reproduce real stars to
1-2 ppm in the light curve. See {doc}`../spherical/limb-darkening` for the laws and
their priors, and {doc}`../reference/hybrid-laws` for their definitions.

## With the sampler API

`flux_dev_from_tau` takes `f` and `theta` (in **radians** here), as
scalars or as one value per chain:

```python
import mlx.core as mx
from elome import flux_dev_from_tau

n, m = 512, 2000                                  # chains, points per chain
tau = mx.array(np.tile(np.linspace(-0.12, 0.12, m), (n, 1)), dtype=mx.float32)
period, a, b, r = 3.45, 8.8, 0.46, 0.1
w = mx.array([0.3, 0.2, 0.1, 0.1], dtype=mx.float32)

dev = flux_dev_from_tau(tau, period, a, b, r, limb_dark="hybrid4", u=w,
                        f=0.3, theta=0.61,            # theta in radians
                        exp_time=29.4 / 60 / 24, integration="contact")
flux = 1.0 + dev                                      # (n, m)
```

Gradients flow in `tau`, the orbit, `r`, `f`, `theta` and every weight.
Eccentric orbits take `secosw` and `sesinw`, and `ld_basis=True` returns
the limb-darkening basis, as for a spherical planet. See
{doc}`../guides/sampler-api`.

## Exposures and contact points

The contact rule ({doc}`../guides/exposures`) splits each exposure at the
moments the planet's outline touches the stellar limb. For an oblate
planet those are its own contacts, which Elómë solves for, separately
for ingress and egress: a tilted ellipse enters and leaves the disc
differently. On 30-minute exposures that is 4 to 40 times more accurate
than splitting at the contacts of the equivalent spherical planet.

The contacts are available directly, as offsets in orbital phase
$\phi = 2\pi(t - t_0)/P$ from mid-transit:

```python
from elome import contact_offsets_oblate

phi = contact_offsets_oblate(a=8.8, b=0.3, r=0.1, f=0.3, theta=0.6)
phi1, phi2, phi3, phi4 = (float(x) for x in phi)     # float64 MLX scalars
t_14 = (phi4 - phi1) * 3.45 / (2 * np.pi)            # the total duration, days
```

The result is computed in float64 on the CPU unless every array argument
is float32; convert it with `float` or `np.asarray` before using it
elsewhere. The arguments may be numbers or arrays (one geometry per
element); pass
`k` and `h` ($\sqrt{e}\cos\omega$, $\sqrt{e}\sin\omega$) for an eccentric
orbit. On a grazing transit the two inner contacts collapse onto one
point inside the transit, and without a transit all four do.
