# Spotted light curves

## With the frontend

Set the spot table and the star's rotation on `TransitParams`, and
choose a hybrid law. Everything else is as for a spherical planet:

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.0, 0.1, 9.0, 88.5, 0.0, 90.0
p.limb_dark, p.u = "hybrid4", [0.2, 0.2, 0.1, 0.1]
p.spots = [[8.0, -9.0, 5.0, 0.3],              # longitude, latitude, radius [deg], contrast
           [40.0, 15.0, 10.0, 0.6]]
p.prot = 4.0                                   # in the units of t
p.istar, p.lam = None, None                    # 90 and 0 degrees
p.spot_epoch = None                            # longitudes measured at t0

t = np.linspace(-0.15, 0.15, 500)
m = elome.TransitModel(p, t, exp_time=30.0 / 60 / 24, integration="contact")
flux = m.light_curve(p)
```

A model is built for a spotted star or not, and for a number of spots:
setting `spots` and its number of rows are fixed per model, like the
limb-darkening law, while every value can change between calls.

- `light_curves(sets)` takes per-set spot tables and star fields like
  any other field; as arrays, `spots` is (n_sets, K, 4). See
  {doc}`../guides/batching`.
- `light_curve_mx(params)` is differentiable in every entry of the spot
  table (pass it as an MLX array, in degrees) and in `prot`, `istar`,
  `lam` and `spot_epoch`, as well as in every orbital and limb-darkening
  parameter. See {doc}`../guides/gradients-and-precision`.
- `dtype=mx.float32` runs on the GPU kernels; `use_metal=False` takes the
  graph instead.

```python
import mlx.core as mx

def loss(spots, prot):
    p.spots, p.prot = spots, prot
    return mx.sum(m.light_curve_mx(p))

with mx.stream(mx.cpu):
    g_spots, g_prot = mx.grad(loss, argnums=[0, 1])(
        mx.array([[8.0, -9.0, 5.0, 0.3], [40.0, 15.0, 10.0, 0.6]], dtype=mx.float64),
        mx.array(4.0, dtype=mx.float64))
```

As for oblate and ringed planets, prefer **`hybrid4`** or **`hybrid5`**: a
spot near the limb and the planet's ingress both probe the limb, and a
law that cannot follow it leaves residuals a fit can attribute to spots.

## With the sampler API

`flux_dev_from_tau` takes a spotted star as keywords, for samplers that
build their own likelihood and batch their chains: `spots` (longitude,
latitude and radius in **radians**, and contrast; `(K, 4)` or one table
per chain), the star's clock `spot_time` (each point's time since the
spots' epoch, beside `tau`, the time since its own mid-transit), `prot`,
`istar` and `lam`:

```python
import mlx.core as mx
from elome import flux_dev_from_tau

tau = mx.array(np.linspace(-0.15, 0.15, 500), dtype=mx.float64)
spots = np.concatenate([np.radians([[8.0, -9.0, 5.0], [40.0, 15.0, 10.0]]),
                        [[0.3], [0.6]]], axis=1)
dev = flux_dev_from_tau(tau, 3.0, 9.0, 0.24, 0.1, limb_dark="hybrid4",
                        u=[0.2, 0.2, 0.1, 0.1], spots=spots,
                        spot_time=tau, prot=4.0)      # the spots' epoch at t0
```

Gradients flow in both clocks and every parameter; transit-timing
variations change `tau` and leave `spot_time` alone. See
{doc}`../guides/sampler-api` for the rest, including why a spotted model
is called eagerly rather than under `mx.compile`.

## At sky positions

The light curve is also available at given sky positions, without an
orbit or a rotating star: `elome.flux_dev_spots` and
`elome.shape_cols_spots` take the planet's centre `(X, Y)` and radius,
and each spot's unit centre vector $\hat n$ (with $\hat n_z$ towards the
observer), angular radius in **radians** and contrast. Any sky frame
will do, as long as the planet and the spots share it.
`elome.spot_vector(lon, lat)` builds $\hat n$ from a longitude measured
from the line of sight towards $+x$ and a latitude towards $+y$:

```python
import mlx.core as mx
from elome import flux_dev_spots, shape_cols_spots, spot_vector

n = spot_vector(np.radians([20.0, 35.0]), np.radians([10.0, 12.0]))   # (K, 3)
alpha = np.radians([8.0, 5.0])
contrast = np.array([0.4, 0.7])
X = mx.array(np.linspace(-0.2, 0.8, 200), dtype=mx.float64)
dev = flux_dev_spots(X, 0.2, 0.1, n, alpha, contrast,
                     [0.2, 0.2, 0.1, 0.1], "hybrid4")                # F - 1, (200,)
B = shape_cols_spots(X, 0.2, 0.1, n, alpha, contrast, "hybrid4")     # (200, 5)
```

`n`, `alpha` and `contrast` may also carry the point axes, one set of
spots per point. `shape_cols_spots` returns the limb-darkening basis, so
that $F - 1 = (B\,c)/(N\,c)$ with $c = (1, -w)$ and $N$ the unspotted
star's norms (`elome.hybrid_norms`), as for every other planet.

## Exposures and kinks

A spotted light curve has kinks where the planet touches the stellar
limb, as for any planet, and where it touches a spot's visible
boundary:

- where the planet first and last touches a spot's rim, and, for a spot
  smaller than the planet, where the spot first lies wholly inside the
  planet's disc and where it starts to leave it;
- for a spot on the limb, where the planet's edge passes one of the two
  points at which the spot's rim meets the limb.

The star turns between transits, so these differ from transit to
transit. The contact rule ({doc}`../guides/exposures`) finds them
exposure by exposure: it samples how many corners the planet makes with
each spot's visible rim across the exposure, bisects every change, and
splits the exposure at these events and the planet's limb contacts, up
to six split points in all, integrating each piece by Gauss–Legendre. The sampling step is a
quarter of the time the planet takes to move by its own radius, between
8 and 256 samples per exposure, so a pair of events closer than that,
such as a brief graze, can be missed; its effect on the exposure is
equally brief. The split points carry their derivatives in every
parameter, so the gradient of `light_curve_mx` is that of the quadrature
it evaluates.

Two slow kinks are not split: a spot's rim reaching the limb (a spot
rotating onto or off the disc) and two spots starting to overlap. Both
happen on the rotation time scale and are kinks of curvature, not of
slope; on a star rotating in 0.8 days they cost about $5\times10^{-9}$
of the flux in a 30-minute exposure.

For 30-minute exposures across the crossing of a spot smaller than the
planet, the contact rule agrees with the exposure average to
$5\times10^{-7}$ at `n_gl = 5`, $8\times10^{-8}$ at the default
`n_gl = 7`, $2\times10^{-8}$ at `n_gl = 9` and $6\times10^{-9}$ at
`n_gl = 12`. Supersampling the same exposures by 45 points is off by
$7\times10^{-5}$, and by 101 points by $3\times10^{-5}$.
