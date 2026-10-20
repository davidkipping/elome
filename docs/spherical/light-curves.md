# Light curves

## The parameters

`TransitParams` holds one set of parameters. The fields are batman's:

| field | meaning | units |
|---|---|---|
| `t0` | time of inferior conjunction (mid-transit) | days |
| `per` | orbital period | days |
| `rp` | planet radius | stellar radii |
| `a` | semi-major axis | stellar radii |
| `inc` | orbital inclination | degrees |
| `ecc` | eccentricity, $0 \le e < 1$ | |
| `w` | the planet's argument of periastron | degrees |
| `limb_dark` | the limb-darkening law | see {doc}`limb-darkening` |
| `u` | the law's coefficients | a list |
| `fp` | planet-to-star flux ratio (secondary eclipses only) | |

`w` is the planet's argument of periastron, as in batman. A radial-velocity
solution usually gives the star's, $\omega_\star$; then
`w` $= \omega_\star + 180°$.

Two more fields, `f` and `theta`, make the planet oblate; leave them as
`None` here. See {doc}`../oblate/index`.

## The model

`TransitModel(params, t, ...)` prepares a model for the times `t`. It
fixes the limb-darkening law and the transit type, and any parameter
may change between evaluations:

```python
m = elome.TransitModel(params, t)
flux = m.light_curve(params)          # numpy array

params.rp = 0.12                      # change anything...
flux = m.light_curve(params)          # ...and evaluate again, nothing rebuilt
```

| keyword | default | meaning |
|---|---|---|
| `transittype` | `"primary"` | `"primary"` or `"secondary"` |
| `exp_time` | `0.0` | exposure length, in the units of `t` |
| `supersample_factor` | `1` | samples per exposure, batman's rule |
| `integration` | `"supersample"` | or `"contact"`, the more accurate rule |
| `n_gl` | `7` | Gauss-Legendre nodes per piece for `"contact"` |
| `dtype` | `mx.float64` | `mx.float32` runs on the GPU |

Exposure integration has its own page: {doc}`../guides/exposures`.

Three ways to evaluate the model:

- `light_curve(params)` returns one light curve as a numpy array.
- `light_curves(sets)` takes many parameter sets and returns a
  `(n_sets, len(t))` array from one dispatch. See
  {doc}`../guides/batching`.
- `light_curve_mx(params)` returns an MLX array that is differentiable in
  every field. See {doc}`../guides/gradients-and-precision`.

## Precision and speed

The default model is float64 on the CPU. It is the exact reference:
light curves agree with a 30-digit direct integral of the occulted
intensity to $2\times10^{-16}$ on a circular orbit and $4\times10^{-16}$
on an eccentric one. Against batman,
the agreement is at batman's own accuracy, about $10^{-8}$.

`dtype=mx.float32` runs on the GPU, in a single fused Metal kernel that
computes the orbit and the photometry together. It is accurate to about
$2\times10^{-7}$ in flux and computes $10^7$ points in about 10 ms.
Absolute mission times (BJD, BTJD) are fine in both precisions: in
float32 the model re-centres the times on the host, in float64, before
anything reaches the GPU. Over a baseline of hundreds of orbits the
float32 orbital phase itself loses precision; see {ref}`long-baselines`.

## Eccentric orbits

Set `ecc` and `w`. Kepler's equation is solved without iteration
(Markley's method, refined once), so the solve is exact to rounding,
cheap, and differentiable. The eccentric orbit is solved about the
transit rather than about periastron, which keeps it well conditioned in
float32 down to $e = 0$:

```python
params.ecc, params.w = 0.3, 63.0
flux = elome.TransitModel(params, t).light_curve(params)
```

`t0` stays the time of inferior conjunction, as in batman.

## Secondary eclipses

`transittype="secondary"` gives the eclipse of the planet by the star,
with the planet a uniform disc of flux ratio `fp`. The flux is $1 + f_p$
out of eclipse and 1 in totality. The time of secondary eclipse follows
from the orbit:

```python
params.fp = 1e-3
t2 = params.t0 + 0.5 * params.per + np.linspace(-0.1, 0.1, 1000)  # circular orbit
m2 = elome.TransitModel(params, t2, transittype="secondary")
eclipse = m2.light_curve(params)
```

The eclipse is placed at the orbit's geometric superior conjunction:
the light-travel time across the orbit, $2a/c$ (about 40 s for
$a = 8\,R_\odot$, 15 minutes for $a = 200\,R_\odot$), is not included, as
in batman. Add it to the times if your timing precision needs it.

`integration="contact"` is for primary transits. Use
`supersample_factor` for a secondary eclipse.

## Differences from batman

- Supported as in batman: parameter updates between calls,
  `supersample_factor` and `exp_time` (the same endpoint-inclusive
  rule), eccentric orbits, secondary eclipses with `fp`.
- Elómë adds the contact rule for exposures, many parameter sets per call
  (`light_curves`), gradients (`light_curve_mx`), float32 on the GPU,
  the hybrid limb-darkening laws, and oblate planets.
- Not available: the nonlinear, square-root, logarithmic, exponential
  and power-2 laws, which have no closed form in this formulation; the
  polynomial and hybrid laws cover the same ground (see
  {doc}`limb-darkening`). batman's `max_err` and `fac` have no
  counterpart because there is no numerical integration to tune.
