# Many parameter sets at once

## The rule

**Never evaluate parameter sets one at a time. Hand the model as many of
them as you can in a single call.**

A GPU dispatch costs 0.2–0.7 ms however much work it carries, and the
GPU needs about $10^5$ threads in flight before all its cores are busy.
The cost of one call is roughly

$$
t \approx n_\text{dispatches}\, t_\text{floor} + \frac{\text{total points}}{R},
$$

with $R \approx$ 3.5–5.5 billion points per second for the fused kernels,
and "total points" = (parameter sets per call) × (points per light
curve). The formula is good for orders of magnitude; the measured
throughput varies by about 35% with the shape of the batch.

Measured, for 10,000 parameter sets of a 1,000-point light curve
(`benchmarks/verify_doc_claims.py`):

| strategy | wall time | per light curve |
|---|---:|---:|
| 10,000 separate calls (a Python loop) | about 2,400 ms | about 240 µs |
| one batched call | about 2.9 ms | about 290 ns |

The ratio, about 800 here, moves by a few tens of per cent with the
machine's state; the order of magnitude is the point. A per-walker loop makes Elómë the
slowest code in any benchmark; a batched call makes it the fastest. If a
call carries fewer than about $3\times10^4$ points in total, a CPU code
is genuinely faster: batch harder, or use the float64 path.

## With the frontend: `light_curves`

`TransitModel.light_curves` evaluates many parameter sets on the model's
times in one dispatch, and returns a `(n_sets, len(t))` array. Pass a
sequence of `TransitParams`, or one `TransitParams` whose fields are
arrays:

```python
import mlx.core as mx

m = elome.TransitModel(params, t, dtype=mx.float32)
flux = m.light_curves([p1, p2, p3])        # (3, len(t))

p.rp = np.array([0.09, 0.10, 0.11])        # array-valued fields...
p.inc = np.array([88.0, 88.5, 89.0])
flux = m.light_curves(p)                   # ...(3, len(t))
```

For 2,000 sets of 301 points in float32 this is about 100 times faster
than a Python loop over `light_curve` with a list of sets, and several
hundred times faster with array-valued fields, which skip unpacking 2,000
Python objects. Each set's light curve is `light_curve`'s, bit for bit.
Circular and eccentric sets can be mixed in one
batch, and every exposure rule and limb-darkening law works, including
per-set coefficients and, for oblate planets, per-set `f` and `theta`.

## With the sampler API

`flux_dev_from_tau` and the `elome.anvil` models take one row of
parameters per chain and evaluate every chain in one launch. That is how
a sampler should call them: once per step, for every chain. See
{doc}`sampler-api` and {doc}`../spherical/fitting`.

## Memory

- **Forward.** The fused kernels keep every intermediate in registers;
  a forward call stores only its inputs and output, about 12 bytes per
  point.
- **Gradients.** The gradient kernels write several full
  `(n_chains, m)` per-point gradient arrays before reducing them, about
  32 bytes per point. At $10^9$ points in total that is 32 GB; split the
  data axis well before that. anvil's `ChunkedGaussianLogLike` does so
  automatically, in 16,384-point chunks by default.
