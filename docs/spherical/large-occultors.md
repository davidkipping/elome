# Companions larger than their star

A planet transiting a white dwarf can be several times larger than its
star (WD 1856+534 b has $R_p/R_\star = 7.28$). Every entry point
accepts `rp > 1`, with every limb-darkening law, in both precisions, with
gradients, for a spherical planet, a ringed one and a planet crossing
starspots alike. A ring may also be wider than the star around a planet
that is not.

A fourth regime then appears beside no overlap, partial overlap and a
planet fully inside the disc: **total occultation**, $z \le r_p - 1$
(with $z$ the sky separation in stellar radii), where the star is fully
covered, $F = 0$ exactly and every gradient is zero.

```python
import math
import numpy as np
import mlx.core as mx
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a = 0.0, 1.4079, 7.28, 336.0     # WD 1856+534 b-like
p.inc = math.degrees(math.acos(7.79 / p.a))           # b = 7.79: grazing
p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, "quadratic", [0.4, 0.25]
t = np.linspace(-0.01, 0.01, 2001)                    # a 5-minute, 16%-deep graze
flux = elome.TransitModel(p, t, dtype=mx.float64).light_curve(p)   # exact
```

## Accuracy

The largest error in the flux, by law ([`benchmarks/bench_large_occultor.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_large_occultor.py)).

float64, against a 30-digit direct integral of the occulted intensity,
at 41 separations for each radius, in total occultation, partial overlap
and none:

| `rp` | quadratic | polynomial | hybrid laws |
|---|---|---|---|
| 1.5 | $2.2\times10^{-16}$ | $2.2\times10^{-16}$ | $4.4\times10^{-16}$ |
| 7.28 | $1.6\times10^{-14}$ | $1.5\times10^{-13}$ | $2.2\times10^{-16}$ |
| 20 | $2.0\times10^{-13}$ | $2.9\times10^{-11}$ | $3.3\times10^{-16}$ |
| 50 | $5.0\times10^{-12}$ | $3.7\times10^{-9}$ | $3.3\times10^{-16}$ |

float32, partial overlap, at 2,001 separations for each radius, against
float64 on the same inputs (the Metal kernels; the polynomial law, which
has none, on the MLX graph):

| `rp` | quadratic | polynomial | hybrid laws |
|---|---|---|---|
| 1.5 | $2.6\times10^{-7}$ | $1.9\times10^{-7}$ | $3.1\times10^{-7}$ |
| 7.28 | $1.5\times10^{-5}$ | $1.8\times10^{-4}$ | $2.8\times10^{-7}$ |
| 20 | $1.9\times10^{-4}$ | $2.3\times10^{-2}$ | $3.1\times10^{-7}$ |
| 50 | $3.5\times10^{-3}$ | 2.4 | $2.6\times10^{-7}$ |

The hybrid laws (hybrid5 with weights that give it a $\mu^4$ term) stay
at float32's and float64's rounding at every size: their terms remain of
order 1 for any `rp`. Total occultation is exact in both precisions. In
partial overlap the other two laws lose accuracy as `rp` grows, because
terms of size $\sim r_p$ cancel to order 1. The quadratic law's float32
error grows as $r_p^2$, and the polynomial law's grows much faster.

**For a large occultor in float32, use a hybrid law.** With the quadratic
law, use float64 above $r_p \approx 10$, or budget for the error in the
table. With the polynomial law, use float64.

## Things to know

- **Exposure integration.** The contact rule converges as it does for a
  planet, but its absolute error scales with the eclipse depth. For a
  *total* eclipse of the same system ($b = 3$: 100% deep for 10.6
  minutes, 14.8 minutes from first to last contact; not the grazing
  example above), 2-minute exposures are accurate to about $10^{-5}$ at
  `n_gl = 5` and to below $10^{-6}$ at `n_gl = 9`.
- **Sampler bounds.** The ready-made anvil targets bound `r` to
  $[0.005, 0.5]$, `b` to $[0, 1.2]$ and `a` to $[1.5, 200]$ by default
  (`elome.anvil.DEFAULT_BOUNDS`). For a
  WD 1856+534 b-like system pass wider ones, for example
  `bounds={"r": (1.0, 12.0), "b": (0.0, 13.0), "a": (1.5, 1000.0)}` to
  `make_transit_target`. An upper bound on `r` above 1 also switches on
  the hybrid laws' large-occultor form (below).
- **The hybrid laws' large-occultor form** is built into the Metal
  kernels and the float64 graph only where `r` may exceed 1, because
  planets smaller than their star would pay for it (up to 6% in some
  gradient kernels). `TransitModel` decides from `rp`,
  `make_transit_target` from its bounds on `r`, and a call with `r` as a
  Python number from its value. A direct call with `r` as an array needs
  `large_occultor=True`: `flux_dev_from_tau`, `flux_dev_metal`,
  `flux_dev_hybrid`, `shape_cols`, `make_hybrid_transit_flux` and
  `make_eclipse_flux` take it. Without it, the error beyond `rp = 1`
  grows as for the polynomial law.
- **Ringed planets and spotted stars** have the same form for any
  outline larger than the star: the planet when `rp > 1`, and a ring
  edge wider than the star even around a planet smaller than it. Their
  float32 error stays near $5\times10^{-7}$ (at `rp = 7.28`; a few
  $10^{-6}$ for a ring edge 36 stellar radii across, the rounding of its
  float32 inputs). `TransitModel` decides from `rp`, `ring_out` and `f`;
  a direct call with them as arrays declares `large_occultor=True` as
  above.
- **`rp = 1` exactly** is a degenerate point, where the contacts meet at
  $z = 0$. The result there is good to $5\times10^{-9}$ in float64 (to
  rounding for the hybrid laws) and $10^{-4}$ in float32, and finite
  everywhere.
