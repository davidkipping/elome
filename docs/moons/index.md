# Exomoons: overview

A planet with a moon transits as two bodies: the moon adds its own dip,
before, during or after the planet's, and the pair circle their common
centre of mass, so the planet's transits shift in time and duration from
one epoch to the next. Elómë models both, after Kipping (2011): the
barycentre follows the orbit of the usual transit parameters, the moon
orbits the planet on a Keplerian orbit of its own, and the light either
body blocks, including where they overlap on the star, is computed at
every time.

A planet with a moon is a `TransitParams` with the moon's fields set:

- `moon_r`, the moon's radius, and `moon_a`, the semi-major axis of its
  orbit about the planet, both in units of the planet's radius `rp`;
- `moon_per`, its orbital period, in the units of `t`;
- `moon_q`, its mass over the planet's (default 0);
- `moon_inc` and `moon_node`, the inclination and node of its orbit on
  the sky, in degrees (default `inc` and 0: coplanar with the planet's
  orbit);
- `moon_phase`, where it is on its orbit at `t0`, in degrees (default 0:
  in front of the planet);
- `moon_ecc` and `moon_w`, its eccentricity and argument of periastron
  (default 0).

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 30.0, 0.08, 60.0, 89.7, 0.0, 90.0
p.limb_dark, p.u = "hybrid4", [0.2, 0.2, 0.1, 0.1]
p.moon_r, p.moon_a, p.moon_per = 0.35, 6.0, 0.9     # radius and orbit in planet radii
p.moon_q, p.moon_phase = 0.02, 40.0

t = np.linspace(-0.4, 0.4, 2000)
flux = elome.TransitModel(p, t).light_curve(p)
```

`t0`, `per`, `a`, `inc`, `ecc` and `w` describe the orbit of the
**barycentre** of the planet and its moon; `t0` is the barycentre's
mid-transit.

## What is the same, and what differs

The same as for a spherical planet:

- the frontend (`light_curve`, `light_curves`, `light_curve_mx`);
- circular and eccentric orbits, for the planet and for the moon;
- exposure integration, including the contact rule;
- float64 and float32, with analytic gradients in every parameter, the
  moon's included;
- a fitting target for `anvil`.

What differs:

- **The hybrid laws are exact, and the quadratic law nearly so.** Each
  body's own disc is exact for every law the model takes. Where the planet
  and the moon overlap on the star, the hybrid laws are exact too. The
  uniform, linear and quadratic laws there take a closure for one term:
  better than a part per million for a moon of a few hundredths of the
  star's radius, less good for a larger one, and exact where one disc
  lies inside the other (see {doc}`accuracy-and-fitting`). The polynomial laws are not available.
- **A spherical planet on an unspotted star, and primary transits only.**
- **One moon per planet.**
- **float32 kernels for the photometry only.** Each disc and the overlap
  run in fused Metal kernels; the orbits and the exposure quadrature stay
  MLX operations around them. float32 keeps its accuracy far from `t0`
  (see {doc}`accuracy-and-fitting`).
- **Not under `mx.compile` or `mx.vmap`.** A model with a moon is called
  eagerly: it chooses on the host which points and overlaps to evaluate.
- **Two orbits in time.** The moon's phase advances between transits,
  so every transit differs: each point's orbit number matters as well as
  its time from mid-transit (see {doc}`light-curves`).
- **Cost.** Only points from which either body can reach the star are
  evaluated, and the overlap only where the two discs overlap on it, but
  a model with a moon costs more than a bare transit. Spherical, oblate,
  ringed and spotted models never load or run any of the moon code.

## Reading order

1. {doc}`geometry`: the barycentre and the moon's orbit, the sky-frame
   elements and their symmetries, the domain.
2. {doc}`light-curves`: computing light curves with a moon, at sky
   positions as well as on orbits, and how exposures are integrated.
3. {doc}`accuracy-and-fitting`: what to expect in each precision, and
   what is specific to fitting a moon.
4. The shared guides: {doc}`../guides/exposures`,
   {doc}`../guides/batching`, {doc}`../guides/gradients-and-precision`.
