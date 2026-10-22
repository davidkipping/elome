# Starspots: overview

A planet crossing a spotted star blocks less light where it covers a dark
spot, and the transit shows a bump; out of transit, the spots turning
with the star modulate the flux. Elómë models both at once, exactly:
each spot is a circular cap on the stellar surface, which projects to an
ellipse on the sky, darker (or, for a facula, brighter) than the
photosphere by a constant factor, and limb-darkened by the same hybrid
law.

A spotted star is a `TransitParams` with a spot table and the star's
rotation:

- `spots`, one row per spot: longitude, latitude and angular radius (in
  degrees) and contrast, the spot's intensity over the photosphere's;
- `prot`, the star's rotation period, in the units of `t`;
- `istar`, the inclination of its spin axis to the line of sight, in
  degrees (default 90: the equator seen edge-on);
- `lam`, the sky-projected obliquity in degrees, in the convention of the
  Rossiter–McLaughlin literature (default 0: aligned);
- `spot_epoch`, the time at which the longitudes are measured (default
  `t0`).

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.0, 0.1, 9.0, 88.5, 0.0, 90.0
p.limb_dark, p.u = "hybrid4", [0.2, 0.2, 0.1, 0.1]
p.spots = [[10.0, -10.0, 8.0, 0.4],      # longitude, latitude, radius [deg], contrast
           [60.0, 20.0, 12.0, 0.6]]
p.prot, p.istar, p.lam = 12.0, 80.0, 10.0

t = np.linspace(-0.15, 0.15, 2000)
flux = elome.TransitModel(p, t).light_curve(p)
```

The flux is normalised to the **unspotted** star: with dark spots on the
visible hemisphere it is below 1 even out of transit.

## What is the same, and what differs

The same as for a spherical planet:

- the frontend (`light_curve`, `light_curves`, `light_curve_mx`);
- circular and eccentric orbits;
- exposure integration, including the contact rule;
- float64 and float32, with analytic gradients in every parameter, every
  spot's included.

What differs:

- **The hybrid limb-darkening laws only**, as for oblate and ringed
  planets.
- **A spherical planet, and primary transits only.**
- **The number of spots is fixed per model.** Their values can change
  between calls, but a model built for two spots takes two.
- **The flux is not 1 out of transit.** It is the spots' rotational
  modulation, relative to the unspotted star; see
  {doc}`accuracy-and-fitting` for what that means for a fit.
- **The frontend only.** The sampler API (`flux_dev_from_tau`) takes the
  time since mid-transit, but spots need absolute time, since the star
  turns between transits. Batch parameter sets through `light_curves`
  instead.
- **Cost.** Every visible spot's outline is integrated at every time, in
  transit or not, so a spotted star costs more than a bare transit, in
  proportion to the number of spots. Spherical, oblate and ringed models
  never load or run any of the spot code.

## Reading order

1. {doc}`geometry`: the spot and the star, the frames and conventions,
   overlapping spots, the domain.
2. {doc}`light-curves`: computing spotted light curves, at sky positions
   as well as on an orbit, and how exposures are integrated.
3. {doc}`accuracy-and-fitting`: what to expect in each precision, and
   what is specific to fitting spots.
4. The shared guides: {doc}`../guides/exposures`,
   {doc}`../guides/batching`, {doc}`../guides/gradients-and-precision`.
