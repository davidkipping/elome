# Quickstart

This page takes ten minutes. It computes a light curve the way batman
does, changes parameters, takes a gradient, evaluates many parameter sets
at once, and finally makes the planet oblate. Coming from batman, start
with {ref}`coming-from-batman`.

## A light curve

```python
import numpy as np
import elome

params = elome.TransitParams()
params.t0 = 0.0                  # time of inferior conjunction [days]
params.per = 3.0                 # orbital period [days]
params.rp = 0.1                  # planet radius [stellar radii]
params.a = 10.0                  # semi-major axis [stellar radii]
params.inc = 88.0                # inclination [degrees]
params.ecc = 0.0                 # eccentricity
params.w = 90.0                  # argument of periastron [degrees]
params.limb_dark = "quadratic"   # the limb-darkening law
params.u = [0.4, 0.25]           # its coefficients

t = np.linspace(-0.1, 0.1, 1000)
m = elome.TransitModel(params, t)
flux = m.light_curve(params)     # a numpy array, 1 out of transit
```

On a Mac with Apple Silicon the model runs on the GPU in float32, through
fused Metal kernels, to about $10^{-7}$ of the flux. For the exact float64
path on the CPU, the reference, build it with `dtype=mx.float64` (with
`import mlx.core as mx`); elsewhere, without the GPU kernels, float64 is
the default. See {doc}`concepts`.

As in batman, the model is built once for a set of times and then
evaluated for any parameters:

```python
for rp in (0.05, 0.10, 0.15):
    params.rp = rp
    print(rp, m.light_curve(params).min())
params.rp = 0.1
```

Eccentric orbits and finite exposures are fields and keywords, as in
batman:

```python
params.ecc, params.w = 0.3, 45.0
m_long = elome.TransitModel(params, t, exp_time=29.4 / 60 / 24,
                            integration="contact")   # a 29.4-minute exposure
flux_long = m_long.light_curve(params)
params.ecc, params.w = 0.0, 90.0
```

`integration="contact"` is a more accurate alternative to batman's
`supersample_factor` (which is also supported); see
{doc}`../guides/exposures`.

## A gradient

`light_curve_mx` returns the flux as an MLX array, differentiable in every
field you set to an `mx.array`. Here is the gradient of a chi-squared in
the planet's radius and the two limb-darkening coefficients, on the GPU:

```python
import mlx.core as mx

y = flux + 1e-4 * np.random.default_rng(1).standard_normal(t.size)   # fake data
y32 = mx.array(y)                # float32, as the model

def chi2(rp, u1, u2):
    params.rp, params.u = rp, [u1, u2]
    return mx.sum(((m.light_curve_mx(params) - y32) / 1e-4) ** 2)

g = mx.grad(chi2, argnums=(0, 1, 2))(mx.array(0.1), mx.array(0.4), mx.array(0.25))
params.rp, params.u = 0.1, [0.4, 0.25]
```

In float32 the gradients agree with float64's to about $10^{-3}$ of their
size, plenty for a sampler. For the exact float64 gradient, build the
model with `dtype=mx.float64`, use float64 arrays and run `mx.grad` inside
`with mx.stream(mx.cpu):`; see {doc}`../guides/gradients-and-precision`.

## Many parameter sets at once

A sampler never wants one light curve. It wants hundreds, and it should
ask for them in one call:

```python
batch = elome.TransitParams()
for name in ("t0", "per", "a", "inc", "ecc", "w", "limb_dark", "u"):
    setattr(batch, name, getattr(params, name))
batch.rp = np.linspace(0.08, 0.12, 500)       # 500 different planets
fluxes = m.light_curves(batch)                # shape (500, 1000), one GPU dispatch
```

One batched call is two to three orders of magnitude faster than a
Python loop over `light_curve`. {doc}`../guides/batching` explains why.

## An oblate planet

Give the planet a projected flattening `f` and a sky angle `theta` (the
long axis's angle from the direction of motion, in degrees), and choose
a hybrid limb-darkening law:

```python
params.f, params.theta = 0.2, 30.0
params.limb_dark = "hybrid4"
params.u = [0.3, 0.2, 0.1, 0.1]    # shape weights; see the limb-darkening pages
flux_oblate = elome.TransitModel(params, t).light_curve(params)
```

`rp` is then the radius of the circle with the same area as the
ellipse. Leave `f` as `None` for a spherical planet.

## Where next

- **Spherical planets:** {doc}`../spherical/index`.
- **Oblate and tidally locked planets:** {doc}`../oblate/index`.
- **Ringed planets:** {doc}`../rings/index`.
- **Starspots:** {doc}`../spots/index`.
- **Exomoons:** {doc}`../moons/index`.
- **Long exposures** (Kepler long cadence, full-frame images):
  {doc}`../guides/exposures`.
- **Fitting with a sampler:** {doc}`../spherical/fitting` and the
  {doc}`../tutorials/index`.
