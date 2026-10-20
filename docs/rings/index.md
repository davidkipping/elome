# Ringed planets: overview

A planet can carry a ring in its equatorial plane, like Saturn. The ring
is a flat, circular annulus; seen in transit it projects to an elliptical
annulus around the planet, and it can block far more starlight than the
planet does, with a light curve of a different shape.

Elómë models a ringed planet exactly, with hybrid limb darkening, on top
of the oblate-planet machinery. Four parameters describe the ring, on
top of the planet's own:

- `ring_in`, `ring_out`, the ring's inner and outer radii, in units of
  `rp`;
- `ring_f`, the ring's projected flattening $1 - |\cos i_\mathrm{ring}|$:
  0 face-on, approaching 1 edge-on;
- `ring_tau`, its normal optical depth (`None`, the default, is an opaque
  ring).

`theta` orients the ring on the sky, and an oblate planet with it: a ring
in the equatorial plane shares the planet's projected long axis.

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.08, 8.8, 88.5, 0.0, 90.0
p.limb_dark, p.u = "hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1]
p.ring_in, p.ring_out = 1.4, 2.6        # radii in units of rp
p.ring_f, p.ring_tau = 0.7, 0.6         # projected flattening; normal optical depth
p.theta = 25.0                          # degrees

t = np.linspace(-0.15, 0.15, 2000)
flux = elome.TransitModel(p, t).light_curve(p)
```

## What is the same, and what differs

The same as for a spherical or oblate planet:

- the frontend (`light_curve`, `light_curves`, `light_curve_mx`) and the
  sampler API (`flux_dev_from_tau`);
- circular and eccentric orbits;
- exposure integration, including the contact rule;
- float64 and float32, with analytic gradients in every parameter, the
  ring's included.

What differs:

- **The hybrid limb-darkening laws only**, as for an oblate planet.
- **Primary transits only.**
- **Eccentricity up to 0.999**, as for an oblate planet.
- **The planet may be spherical or oblate.** Leave `f` unset for a
  spherical planet; set it for an oblate one, whose ring shares its
  `theta`.
- **Cost.** A ringed planet costs about 10–20 times an oblate one: every
  outline -- the planet and both ring edges -- is integrated. Spherical
  and oblate models never load or run any of the ring code.

## Reading order

1. {doc}`geometry`: what the ring parameters mean, the opacity, the
   symmetries and the domain.
2. {doc}`light-curves`: computing ringed light curves from the frontend
   and the sampler API, and how exposures are integrated.
3. {doc}`accuracy-and-fitting`: what to expect in each precision, what
   it costs, and what is specific to fitting a ring.
4. The shared guides: {doc}`../guides/exposures`,
   {doc}`../guides/batching`, {doc}`../guides/gradients-and-precision`,
   {doc}`../guides/sampler-api`.
