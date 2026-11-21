# Light curves with a moon

## With the frontend

Set the moon's fields on `TransitParams`. Everything else is as for a
spherical planet:

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 30.0, 0.08, 60.0, 89.7, 0.1, 70.0
p.limb_dark, p.u = "hybrid4", [0.2, 0.2, 0.1, 0.1]
p.moon_r, p.moon_a, p.moon_per, p.moon_q = 0.35, 6.0, 0.9, 0.02
p.moon_inc, p.moon_node, p.moon_phase = 89.0, 10.0, 40.0
p.moon_ecc, p.moon_w = 0.05, 30.0

# two transits: the moon is somewhere else on its orbit at each
t = np.concatenate([np.arange(-0.4, 0.4, 2 / 1440), 90.0 + np.arange(-0.4, 0.4, 2 / 1440)])
m = elome.TransitModel(p, t, exp_time=30.0 / 60 / 24, integration="contact")
flux = m.light_curve(p)
```

A model is built with a moon or without one, which is fixed per model
like the limb-darkening law. Every value can change between calls.

- **The references.** `t0`, `per` and `moon_per` must be numbers when the
  model is built. Each time's distance from its reference transit, its
  orbit number and the moon's phase at that transit are then reduced
  once, in float64, against these values. Later calls take any values,
  and the light curve is the same whatever references were used, to
  rounding.
- **`light_curves(sets)`** evaluates many parameter sets in one pass,
  with per-set moon fields like any other field (see
  {doc}`../guides/batching`). A `moon_inc` left `None` is each set's own
  `inc`.
- **`light_curve_mx(params)`** is differentiable in every moon field and
  in every orbital and limb-darkening parameter (see
  {doc}`../guides/gradients-and-precision`).
- **`dtype=mx.float32`** runs the same graph on the GPU.

```python
import mlx.core as mx

def loss(moon_a, moon_phase, moon_q):
    p.moon_a, p.moon_phase, p.moon_q = moon_a, moon_phase, moon_q
    return mx.sum(m.light_curve_mx(p))

with mx.stream(mx.cpu):
    g = mx.grad(loss, argnums=[0, 1, 2])(*(mx.array(v, dtype=mx.float64)
                                            for v in (6.0, 40.0, 0.02)))
```

Prefer a **hybrid law**: it is exact where the planet and the moon
overlap on the star. The quadratic law is exact for each disc and takes a
closure for the overlap (see {doc}`accuracy-and-fitting`).

## For a sampler

`elome.anvil.make_moon_transit_flux` is the batched, per-chain form: one
parameter vector per chain, every chain evaluated in one pass, with
gradients. `make_transit_target(..., moon=True)` builds a whole fitting
problem around it (see {doc}`accuracy-and-fitting`). Its times are three
float64-reduced rows: each point's time from its reference transit, its
orbit number, and the moon's phase at that transit in turns
(`elome.moons.moon_turns_ref`).

## At sky positions

The photometry alone is available at given sky positions, without
orbits: the planet's centre `(xp, yp)` and the moon's `(xm, ym)` in
stellar radii, and their radii. `front_p` and `front_m` say whether each
body is on the observer's side of the star (default: both are).

```python
import mlx.core as mx
from elome import flux_dev_moon, flux_dev_moon_quad, shape_cols_moon

f = lambda v: mx.array(v, dtype=mx.float64)
x = np.linspace(-1.2, 1.2, 300)
xp, yp, xm, ym = f(x), f(0.1), f(x + 0.25), f(0.05)                # the moon leads
dev = flux_dev_moon(xp, yp, xm, ym, 0.08, 0.03,
                    [0.2, 0.2, 0.1, 0.1], "hybrid4")              # F - 1, (300,)
B = shape_cols_moon(xp, yp, xm, ym, 0.08, 0.03, "hybrid4")          # (300, 5)
dq = flux_dev_moon_quad(xp, yp, xm, ym, 0.08, 0.03, 0.4, 0.25)      # quadratic
```

`shape_cols_moon` returns the limb-darkening basis, so that
$F - 1 = (B\,c)/(N\,c)$ with $c = (1, -w)$ and $N$ the star's norms
(`elome.hybrid_norms`), as for every other planet.

## Exposures and kinks

A light curve with a moon has kinks wherever either body touches the
stellar limb, as for any planet, and where the bodies' outlines change
their contact with each other on the star:

- where the planet, or the moon, touches the limb from outside or from
  inside;
- where the two discs touch, from outside or one inside the other;
- where a point at which their outlines cross reaches the limb.

The moon moves between transits, so these differ from transit to
transit. The contact rule ({doc}`../guides/exposures`) finds them in
each exposure. It samples eight smooth event functions, one per kind of
contact, across the exposure, and bisects every change of sign. It then
splits the exposure at the earliest six events and integrates each piece
by Gauss–Legendre.

The sampling step is a quarter of the smaller body's radius at the
fastest sky speed: the barycentre's at periastron plus the moon's about
the planet at its periastron. That is between 4 and 256 samples per
exposure. A pair of events closer than a step, such as a brief graze,
can be missed; its effect on the exposure is equally brief. The split
points carry their derivatives in every parameter, so the gradient of
`light_curve_mx` is that of the quadrature it evaluates.

For 30-minute exposures through two transits of a moon at 4 planet
radii with a 0.6-day period, the contact rule agrees with the exposure
average to $2\times10^{-7}$ at `n_gl = 3`, $2\times10^{-8}$ at
`n_gl = 5` and $4\times10^{-9}$ at the default `n_gl = 7`, for
`hybrid5` and the quadratic law alike. Supersampling the same exposures
by 7 points is off by $1.5\times10^{-4}$.

A point is evaluated only if either body can reach the star from it
during its exposure. Every other point is exactly 1. The overlap is
computed only where the two discs overlap with both on the star.
