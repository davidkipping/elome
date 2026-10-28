# Oblate planets: overview

A rotating planet is flattened at its poles. Seen in transit, it
projects onto the sky as an ellipse rather than a circle, and the shape
of its ingress and egress changes in a way a spherical model cannot
reproduce. Saturn's projected flattening is about 0.1; a young or
rapidly rotating planet can be flatter.

Elómë models the projected ellipse exactly. Two parameters describe it,
on top of the spherical ones:

- `f`, the projected flattening $1 - B/A$;
- `theta`, the angle of the long axis on the sky, measured from the
  direction of motion.

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 87.0, 0.0, 90.0
p.limb_dark, p.u = "hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1]
p.f, p.theta = 0.3, 35.0               # flattening; long-axis angle in degrees

t = np.linspace(-0.12, 0.12, 2000)
flux = elome.TransitModel(p, t).light_curve(p)
```

## What is the same, and what differs

The same as for a spherical planet:

- the frontend (`light_curve`, `light_curves`, `light_curve_mx`) and the
  sampler API (`flux_dev_from_tau`);
- circular and eccentric orbits;
- exposure integration, including the contact rule;
- secondary eclipses, with the planet a uniform disc, in `TransitModel`;
- float64 and float32, with analytic gradients in every parameter,
  `f` and `theta` included.

What differs:

- **The hybrid limb-darkening laws only.** The quadratic law's linear term
  in $\mu$ has no elementary form along an ellipse; the hybrid laws were
  built so that every term does. They are also more accurate than the
  quadratic law for real stars (see {doc}`../spherical/limb-darkening`).
- **A domain:** $r_p \le (1-f)^{3/2}$, so that the planet crosses the
  stellar limb at most twice. That is 0.59 at $f = 0.3$ and 0.35 at
  $f = 0.5$, which covers planets.
- **Eccentricity up to 0.999**, against any $e < 1$ for a spherical
  planet.
- **Cost.** An evaluation costs tens of times a spherical one. It is still
  milliseconds for a light curve on the GPU.

## Reading order

1. {doc}`geometry`: what `rp`, `f` and `theta` mean, and their symmetries.
2. {doc}`light-curves`: computing oblate light curves, with exposures and
   eccentric orbits, from the frontend and from the sampler API.
3. {doc}`accuracy-and-cost`: what to expect in each precision, and how
   long it takes.
4. {doc}`fitting`: what is different about sampling an oblate planet.
5. The shared guides: {doc}`../guides/exposures`,
   {doc}`../guides/batching`, {doc}`../guides/gradients-and-precision`,
   {doc}`../guides/sampler-api`.

The tutorial {doc}`../tutorials/03_spherical_and_oblate_planets` takes the
same steps for a spherical and an oblate planet side by side.
