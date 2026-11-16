# Ringed light curves

## With the frontend

Set the ring fields on `TransitParams` and choose a hybrid law. Everything
else is as for a spherical or oblate planet:

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.08, 8.8, 88.5, 0.0, 90.0
p.limb_dark, p.u = "hybrid4", [0.3, 0.2, 0.1, 0.1]
p.ring_in, p.ring_out, p.ring_f = 1.4, 2.6, 0.7
p.ring_tau = None                             # opaque
p.theta = 25.0                                # degrees
p.f = None                                    # a spherical planet; set it for an oblate one

t = np.linspace(-0.15, 0.15, 2000)
m = elome.TransitModel(p, t, exp_time=29.4 / 60 / 24, integration="contact")
flux = m.light_curve(p)
```

A model is built for a ringed planet or not: setting `ring_in` is fixed
per model, like the limb-darkening law, while the ring's values and
`theta` can change between calls.

- `light_curves(sets)` takes per-set ring fields, like any other field.
  See {doc}`../guides/batching`.
- `light_curve_mx(params)` is differentiable in every ring field and in
  `theta`, as well as in every orbital and limb-darkening parameter. See
  {doc}`../guides/gradients-and-precision`.
- `dtype=mx.float32` runs on the GPU kernels; `use_metal=False` takes the
  graph instead.

As for an oblate planet, prefer **`hybrid4`** or **`hybrid5`**: a law that
cannot follow the stellar limb leaves residuals that a fit can attribute
to the ring. See {doc}`../oblate/light-curves`.

## With the sampler API

`flux_dev_from_tau` takes the ring as keywords (`theta` in **radians**
here), as scalars or one value per chain:

```python
import mlx.core as mx
from elome import flux_dev_from_tau

n, m = 512, 2000                                  # chains, points per chain
tau = mx.array(np.tile(np.linspace(-0.15, 0.15, m), (n, 1)), dtype=mx.float32)
w = mx.array([0.3, 0.2, 0.1, 0.1], dtype=mx.float32)

dev = flux_dev_from_tau(tau, 3.45, 8.8, 0.23, 0.08, limb_dark="hybrid4", u=w,
                        ring_in=1.4, ring_out=2.6, ring_f=0.7, ring_tau=0.6,
                        theta=0.44,                   # radians
                        exp_time=29.4 / 60 / 24, integration="contact")
flux = 1.0 + dev                                      # (n, m)
```

Gradients flow in `tau`, the orbit, `r`, `f` (if given), `theta`, every
ring parameter and every weight. Eccentric orbits take `secosw` and
`sesinw`, and `ld_basis=True` returns the limb-darkening basis, so that
$F - 1 = (B\,c)/(N\,c)$ as for every other planet. See
{doc}`../guides/sampler-api`.

The light curves themselves are also available at sky positions:
`elome.flux_dev_ring` and `elome.shape_cols_ring` take the planet's
centre in the principal frame (`elome.oblate.principal_frame`), as
`flux_dev_oblate` does.

## Exposures and kinks

A ringed light curve is smooth except where an outline -- the planet or a
ring edge -- touches the stellar limb: first and last contact, the planet
fully on the disc, and for a nearly edge-on ring the moments its outline
goes from crossing the limb twice to four times and back. The contact
rule ({doc}`../guides/exposures`) splits each exposure at the kinks that
fall inside it (at most four), so it costs the same number of
evaluations as for any other planet. On a nearly edge-on ring that
crosses the limb four times, it agrees with supersampling extrapolated to
infinitely many points to $2\times10^{-7}$ at `n_gl = 5` (the sampler API's default), and
to $2\times10^{-9}$ at `n_gl = 20`. Uniform supersampling, by contrast,
converges only as $1/N$ across a kink.

The kinks are available directly, as offsets in orbital phase
$\phi = 2\pi(t - t_0)/P$ from mid-transit:

```python
from elome import ring_kinks

phi = ring_kinks(a=8.8, b=0.23, r=0.08, f=0.0, theta=0.44,
                 ring_in=1.4, ring_out=2.6, ring_f=0.7)   # (n, 24), sorted
phi = np.asarray(phi)[0]
phi = phi[np.isfinite(phi)]                               # unused slots are inf
```

Pass `k` and `h` ($\sqrt{e}\cos\omega$, $\sqrt{e}\sin\omega$) for an
eccentric orbit.
