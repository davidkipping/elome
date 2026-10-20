# Gradients and precision

## Gradients through the frontend: `light_curve_mx`

`light_curve_mx(params)` returns the flux as an MLX array. It is
differentiable in every `TransitParams` field you set to an `mx.array`
(`t0`, `per`, `rp`, `a`, `inc`, `ecc`, `w`, `fp`, the entries of `u`, and
`f` and `theta` for an oblate planet). Python numbers stay constants:

```python
import mlx.core as mx

m = elome.TransitModel(params, t)            # float64, on the CPU
y64 = mx.array(y, dtype=mx.float64)
with mx.stream(mx.cpu):
    def chi2(rp, ecc):
        params.rp, params.ecc = rp, ecc
        return mx.sum(((m.light_curve_mx(params) - y64) / yerr) ** 2)
    g = mx.grad(chi2, argnums=(0, 1))(mx.array(0.1, dtype=mx.float64),
                                      mx.array(0.0, dtype=mx.float64))
```

- **It is the same model.** With all-Python parameters the result is
  `light_curve`'s, bit for bit, through the same compiled graphs.
  Differentiating any field other than `ecc` costs nothing extra.
- **Eccentricity is safe at zero.** An array-valued `ecc` enters through
  $(e, \omega)$ in a form whose derivative in $e$ is finite and correct at
  $e = 0$ (one-sided, since $e \ge 0$), with $\partial/\partial\omega = 0$
  there. That form is 3–4 times slower than the default, so it is used
  only when `ecc` is an array.
- **Exposures are differentiated exactly.** The contact rule's nodes move
  with the parameters, so the gradient is that of the quadrature actually
  evaluated.
- **Invalid values.** An `ecc` outside $[0, 1)$ raises `ValueError`
  whenever its value can be read: a Python number, or an array, eagerly
  or under `mx.grad`. Inside `mx.vmap` or your own `mx.compile` it is
  traced and cannot raise; the output and every gradient are then NaN.
- **Everything is checked** against float64 finite differences, for every
  field, exposure rule, limb-darkening law and transit type.

Gradients never become NaN at the model's internal boundaries (contacts,
a planet fully inside the disc, $z = r$, and so on): every branch is
computed in a form that is finite on both sides.

## Which precision runs

**The dtype of the data is the dtype of the computation**, for every
function that takes separations or times. float32 data, in any container
(an `mx.array`, a numpy `float32` array), runs the fused GPU kernels.
Anything else (float64, a numpy `float64` array, a Python list,
integers, float16) is computed in float64 by the exact graph on the CPU
stream, and Elómë puts it there for you.

- MLX's own `mx.array(x)` turns a numpy float64 array into float32
  silently. Elómë's functions do not: hand them numpy float64 and they
  stay in float64. To use the GPU kernels, pass
  `mx.array(x, dtype=mx.float32)`.
- The data may be passed positionally or by name.
- **Parameters.** The kernel entry points (`flux_dev_from_tau`,
  `flux_dev_metal`) cast parameters to the data's dtype and take one
  value per row of `(n, m)` data. The graph functions (`flux_dev`,
  `flux_dev_hybrid`, `shape_cols`, ...) follow MLX's rules instead: they
  broadcast, so `(n, m)` data takes `r` as `(n, 1)`, and they promote, so
  a float64 parameter gives a float64 result.
- **A gradient with respect to a float64 parameter** needs the `mx.grad`
  call itself under `with mx.stream(mx.cpu):`, even when the data are
  float32. MLX runs a transformation's float64 operations on the default
  stream, which the GPU cannot execute, and no library function can
  redirect that. With a float32 GPU model, differentiate float32 fields;
  for an absolute `t0`, which must be float64, take the gradient on the
  CPU stream.

## MLX's float64 is not float64 for every operation

In MLX 0.32, `mx.sin`, `mx.cos`, `mx.exp` and `mx.erf` are only as
accurate as float32 even on float64 arrays (relative errors of
$10^{-7}$ to $10^{-5}$), while `log`, `sqrt`, `arctan2`, `arccos` and
arithmetic are true float64. Elómë computes its own trigonometry with
`elome.trig.sincos`, which is accurate to $10^{-16}$, so its float64
results are unaffected.

Geometry you compute in MLX *before* calling Elómë is affected, however.
For example, $a/R_\star$ from the transit duration needs
$\sin(\pi T_{14}/P)$, and computed with `mx.sin` it carries a $10^{-7}$
error into a "float64" light curve. Do such preprocessing in numpy or
`math`, or with `elome.trig.sincos`, which works on MLX arrays of either
dtype, is differentiable, and compiles.

(long-baselines)=
## Long baselines in float32

In float32, the orbital phase loses precision when the data span many
orbits. The frontend re-centres the times on their midpoint in float64,
so absolute times and a distant `t0` cost nothing, but float32 still
sees times up to half the data's span. Measured through the frontend
for $P = 10$ days and $r_p = 0.1$ on a circular orbit, with transits at
both ends of the span (`benchmarks/verify_doc_claims.py`), the maximum
flux error is $2.7\times10^{-7}$ for a span of one orbit,
$1.6\times10^{-5}$ for 100, and $2.6\times10^{-4}$ for 1,000. For
long baselines in float32, use the models below, or float64.

The fix is to split each time, in float64 on the host, into an orbit
number and a residual within the orbit, which is what
`elome.orbit.epoch_center_times` does. The anvil models and
`make_transit_target` take times in that form, and are then good to
about $10^{-7}$ in flux for circular orbits and $10^{-6}$ for eccentric
ones up to $e = 0.999$. For per-transit models, `flux_dev_from_tau` takes
the time since each point's own mid-transit, which is small by
construction.

## Eccentricity near zero

Sample $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$, not $(e, \omega)$.
Then $e = 0$ is an interior point, and a sampler genuinely visits it.
Elómë's eccentric orbit is solved about the transit rather than about
periastron, so that float32 gradients survive there. Its relative
gradient error stays at about $10^{-7}$ from $e = 0.1$ down to
$e = 10^{-8}$, where the textbook formulation's float32 gradients are
already entirely wrong (`benchmarks/kh_grad_conditioning.py`).

## A planet exactly as large as the star

At exactly $r_p = 1$, with the planet's centre within about $10^{-12}$
stellar radii of the star's, the planet's and the star's outlines
coincide and the closed forms are degenerate. There the flux is off by
up to $5\times10^{-9}$ and its gradients are wrong: 0, very large, or
NaN, depending on the law and the path. Anywhere else, including
$r_p = 0.999$ or a separation of $10^{-3}$, they are accurate. A sampler
whose prior on $r_p$ reaches 1 is very unlikely to land on exactly 1.0,
but avoid fixing $r_p = 1$ with $b = 0$.
