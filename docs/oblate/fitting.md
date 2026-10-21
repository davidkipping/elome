# Fitting an oblate planet

Everything in {doc}`../spherical/fitting` about samplers applies:
batch every chain into one call, bound the parameters with priors, and
compile the likelihood. The ready-made anvil targets are for spherical
planets with the quadratic law. For an oblate planet, build the
likelihood on `flux_dev_from_tau` with `f` and `theta`, which runs every
chain in one GPU launch with analytic gradients in every input:

```python
dev = flux_dev_from_tau(tau, period, a, b, r, limb_dark="hybrid5", u=w,
                        f=f, theta=theta,          # (n,) per chain; theta in radians
                        exp_time=EXP, integration="contact")
```

What is specific to an oblate planet:

## Parameterisation

- **Wrap `theta`.** It has period $\pi$, so sample it on $[0, \pi)$ and
  wrap, or the posterior has two identical modes.
- **Fold `theta` for central transits.** At $b = 0$, `theta` and
  $\pi -$ `theta` give the same light curve (see {doc}`geometry`), so a
  wrapped posterior is still bimodal there. Fold `theta` onto
  $[0, \pi/2]$ if the transit is near-central.
- **Expect `f` and `theta` to correlate with `b` and `rp`** on
  near-central transits, where an aligned oblate planet and a round one
  at a slightly different impact parameter can look alike. That is a
  property of the data. Choose priors and parameters with it in mind.

## The domain

The model requires $r_p \le (1-f)^{3/2}$. Outside it a numpy input
raises `ValueError` and an MLX array gives NaN; neither is `-inf`. Keep
chains inside with the prior (a box on `f` must respect the limit at the
largest `rp` the chains reach), or map a non-finite log-likelihood to
`-inf` in your wrapper.

## Near f = 0

In float32 the model's value uses the spherical forms below
$f = 10^{-5}$, but its gradient in `f` is the oblate one, so chains can
start at exactly $f = 0$, the edge of a uniform prior on $[0, 1)$.

## Limb darkening

Use `hybrid4` or `hybrid5`. A law that cannot follow the stellar limb
can turn its residuals into a spurious oblateness signal (see
{doc}`light-curves`). The light curve is linear in the limb-darkening
weights, so they can be integrated out analytically rather than sampled:
`ld_basis=True` returns the basis that needs (see {ref}`ld-basis`), and
removes four or five dimensions from the sampled space.

## Cost

One oblate evaluation costs 16–31 times a spherical hybrid one, and about
53–63 times under the contact rule. See {doc}`accuracy-and-cost`. To find
the mode quickly, `integration="none"` (no exposure integration) or a
smaller `n_gl` is much cheaper; sample the posterior with the full rule.
