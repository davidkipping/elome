# The sampler API: `flux_dev_from_tau`

`flux_dev_from_tau` is the entry point for samplers that build their own
likelihood. It takes per-chain parameters and per-chain times and
returns per-chain flux deviations $F - 1$, from one fused GPU kernel per
call, with analytic gradients in every input. The same call covers
spherical, oblate and ringed planets and spotted stars, the quadratic
and hybrid limb-darkening laws, circular and eccentric orbits, and every
exposure rule.

```python
from elome import flux_dev_from_tau

# tau: (n_chains, m) time since each point's OWN mid-transit [days]
dev = flux_dev_from_tau(tau, period, a, b, r, u1, u2,
                        exp_time=29.4 / 60 / 24,   # Kepler long cadence
                        integration="contact", n_gl=5)
flux = 1.0 + dev
```

| argument | shape | meaning |
|---|---|---|
| `tau` | `(n, m)` or `(m,)` | time since mid-transit for each point [days] |
| `period`, `a`, `b`, `r` | scalar or `(n,)` | orbit and planet; `b` is the impact parameter |
| `u1`, `u2` | scalar or `(n,)` | quadratic limb darkening |
| `limb_dark`, `u` | name; `(n_w,)` or `(n, n_w)` | a hybrid law and its weights, instead of `u1`, `u2` |
| `f`, `theta` | scalar or `(n,)` | an oblate planet; `theta` in radians |
| `spots`, `spot_time`, `prot`, `istar`, `lam` | `(K, 4)` or `(n, K, 4)`; like `tau`; scalar or `(n,)` | a spotted star (below); angles in radians |
| `secosw`, `sesinw` | scalar or `(n,)` | an eccentric orbit, $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$ |
| `exp_time`, `integration`, `n_gl`, `n_sub` | | the exposure rule; see {doc}`exposures` |
| `ld_basis` | bool | return the limb-darkening basis instead (below) |

float32 `tau` runs on the GPU; float64 runs the exact graph on the CPU
and computes the *same* function. Even the quadrature's split points are
treated identically, so a gradient checked in float64 is the gradient
that runs in float32.

## Why `tau`

The ready-made model in `elome.anvil` derives each point's phase from a
linear ephemeris, so the times are data there. A sampler that fits
**individual transit times** (transit-timing variations, with tens of
mid-times sampled jointly with the shape) cannot express its model that
way. With `flux_dev_from_tau` the caller builds `tau` however its
parameterisation requires, MLX chains the gradient back through it, and
gradients flow in `tau` as well as in every parameter.

## Exposures inside the kernel

The exposure rule runs per output point in registers, so the
sub-exposure axis never becomes an array. Expanding it instead multiplies
every intermediate and every gradient array by the number of
sub-exposures. At 512 chains × 5,000 points with 15 sub-exposures
(`benchmarks/bench_tau_kernel.py`, M2 Max), against computing the
separations, evaluating the kernel and averaging outside it:

| | forward | value + gradient | peak memory |
|---|---:|---:|---:|
| sub-exposure axis expanded | 19.9 ms | 58.2 ms | 2,776 MB |
| the same rule, in the kernel (`n_sub = 15`) | 5.7 ms | 12.8 ms | 51 MB |
| contact rule, in the kernel (`n_gl = 5`) | 3.8 ms | 8.4 ms | 51 MB |

The last row is also about 1,300 times more accurate than the first
($1.2\times10^{-7}$ against $1.6\times10^{-4}$). Reaching $10^{-6}$ by
supersampling would need about 2,271 sub-exposures, which at 512 chains
would ask for about 368 GB.

(ld-basis)=
## Limb darkening as a linear block: `ld_basis=True`

A light curve is linear in the limb-darkening intensity coefficients.
For the quadratic law, write $I(\mu) = c_0 + c_1\mu + c_2\mu^2$, so
$c = (1 - u_1 - u_2,\; u_1 + 2u_2,\; -u_2)$. Then

$$
F - 1 = \frac{B\,c}{N\,c}, \qquad N = \left(\pi, \tfrac{2\pi}{3}, \tfrac{\pi}{2}\right),
$$

where column $j$ of $B$ is the (negative, unnormalised) deficit of a star
of intensity $\mu^j$. $B$ depends on the geometry alone. A sampler that
integrates the limb darkening out analytically, or samples it in a Gibbs
step, needs $B$ itself, and `ld_basis=True` returns it from one launch:

```python
import math
import mlx.core as mx

B = flux_dev_from_tau(tau, period, a, b, r, exp_time=29.4 / 60 / 24,
                      integration="contact", n_gl=5, ld_basis=True)
# (n, m, 3); u1 and u2 are not needed
c = mx.array([1 - u1 - u2, u1 + 2 * u2, -u2])
dev = (B @ c) / (math.pi * (1 - u1 / 3 - u2 / 6))   # equals the ordinary call
```

For a hybrid law the basis is its shape-basis columns,
$[E_0, S_1, \ldots, S_n]$ (for `hybrid4`, its curvature basis), with
$c = (1, -w)$ and $N$ from
`elome.hybrid_norms(law)`:

```python
B = flux_dev_from_tau(tau, period, a, b, r, limb_dark="hybrid5",
                      ld_basis=True, exp_time=29.4 / 60 / 24)
# (n, m, 6); dev == (B @ c) / (N @ c), c = (1, -w), N = elome.hybrid_norms("hybrid5")
```

This works for oblate planets and eccentric orbits too. The gradient
takes an `(n, m, k)` cotangent and returns gradients in every geometric
input. The cost is that of one ordinary call to within a few per cent,
and three times less than forming three separate light curves
(`benchmarks/bench_ld_basis.py`).

## Eccentric orbits: `secosw`, `sesinw`

Pass $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$ and the orbit becomes
eccentric. Omit both and it stays circular:

```python
dev = flux_dev_from_tau(tau, period, a, b, r, u1, u2,
                        secosw=k, sesinw=h,           # (n,) or scalars
                        exp_time=29.4 / 60 / 24, integration="contact")
```

- `tau` is the time since inferior conjunction, and `b` the impact
  parameter there, $b = a\cos i\,(1 - e^2)/(1 + e\sin\omega)$. This
  reduces to the circular `b` at $e = 0$.
- At exactly $e = 0$, $\partial/\partial k = \partial/\partial h = 0$ and
  every other gradient equals the circular one.
- One batch can mix circular ($k = h = 0$) and eccentric chains.
- The contact rule uses contacts found exactly on the eccentric orbit
  (see {doc}`exposures`).

The eccentric orbit costs a Kepler solve at each quadrature node: at
512 × 5,000 under the contact rule, 2.3–3.3 times the circular cost
(`benchmarks/bench_tau_ecc.py`). Chains at $e = 0$ skip the solve but
still run in the larger kernel, so leave `secosw` and `sesinw` unset for
a purely circular fit.

## Starspots: `spots`, `spot_time`

A spotted star (see {doc}`../spots/index`) needs a second clock. The
planet's position follows `tau`, the time since each point's own
mid-transit; the spots' positions follow `spot_time`, each point's time
since the spots' reference epoch, because the star turns between
transits. Both are ordinary differentiable inputs that the caller builds:

```python
import mlx.core as mx
import numpy as np
from elome import flux_dev_from_tau

n, m = 64, 2000                                   # chains, points
t = np.linspace(-0.14, 0.14, m)                   # times [days], here near one transit
tau = mx.array(np.tile(t, (n, 1)), dtype=mx.float32)
spot_time = tau + 0.4                             # the spots' epoch 0.4 d before t0
spots = mx.array(np.radians([[10.0, -17.0, 8.0], [50.0, 15.0, 10.0]]), dtype=mx.float32)
spots = mx.concatenate([spots, mx.array([[0.5], [0.6]], dtype=mx.float32)], axis=1)

dev = flux_dev_from_tau(tau, 3.45, 8.84, 0.3, 0.115, limb_dark="hybrid4",
                        u=[0.2, 0.2, 0.1, 0.1], spots=spots, spot_time=spot_time,
                        prot=12.0, istar=1.4, lam=0.2,
                        exp_time=29.4 / 60 / 24, integration="contact")
```

- `spots` holds one row per spot: longitude, latitude and angular radius
  in **radians** (the frontend's are in degrees), and contrast. Shared
  as `(K, 4)` or per chain as `(n, K, 4)`. Every chain in a call has
  the same number of spots; a chain with fewer takes rows of contrast 1,
  which are no spot.
- `prot` is the rotation period in `tau`'s units, `istar` and `lam` the
  spin axis's inclination and sky-projected obliquity in radians
  (defaults $\pi/2$ and 0).
- Gradients flow in `tau`, `spot_time`, every spot entry, `prot`,
  `istar`, `lam` and the rest. A fitted spot epoch is
  `spot_time = t - epoch` with `epoch` a sampled parameter; transit
  timing variations change `tau` and leave `spot_time` alone.
- The flux is normalised to the unspotted star, so out of transit it is
  the spots' rotational modulation, not 0.
- `integration="contact"` also splits each exposure at the moments the
  planet meets a spot's edge (see {doc}`../spots/light-curves`);
  `"supersample"` and `"none"` work as for any planet. `ld_basis=True`
  returns the basis, under the usual contract.
- **Call it eagerly, not under `mx.compile` or `mx.vmap`**, which raise.
  The spotted model reads values back to the host (to find the exposures'
  split points and skip the empty ones), which a transform cannot, and
  its float32 kernels are already fused, so compiling would gain nothing.
  Compile the rest of a likelihood separately if it helps.
- In float32 the star's phase $2\pi\,$`spot_time`$/$`prot` carries a
  relative error of about $6	imes10^{-8}$. Over a 1,000-day baseline
  on a star turning in a day that is about $4	imes10^{-4}$ rad. Keep
  `|spot_time|` small by putting the spots' epoch within the data, or
  use float64 for long baselines on fast rotators.

## Separations instead of times: `flux_dev_metal`

`flux_dev_metal(z, r, ...)` takes sky separations $z$ (in stellar radii)
instead of times, for callers that compute the orbit themselves. It has
the same limb-darkening keywords and `ld_basis`, but no exposure
integration.

## Hybrid-law speed

With the hybrid laws, the GPU kernels are faster than with the
quadratic law, because no term needs an elliptic integral; only
hybrid5's gradient, with three poles, costs more. At 512 × 5,000
points under the contact rule (`benchmarks/bench_hybrid.py`):

| law | forward | vs quadratic | value + gradient | vs quadratic |
|---|---:|---:|---:|---:|
| quadratic | 3.7 ms | 1.00× | 8.4 ms | 1.00× |
| hybrid2 | 2.2 ms | 0.59× | 5.1 ms | 0.61× |
| hybrid4 | 2.9 ms | 0.78× | 7.7 ms | 0.91× |
| hybrid5 | 3.5 ms | 0.93× | 10.0 ms | 1.19× |
| hybrid5, `ld_basis=True` | 3.5 ms | 0.94× | 10.0 ms | 1.20× |
