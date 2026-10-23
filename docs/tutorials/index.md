# Tutorials

Four notebooks, each a complete worked example. They are rendered here
with their outputs, and the notebooks themselves are in the repository's
`docs/tutorials/` directory.

| tutorial | what it covers | needs |
|---|---|---|
| {doc}`01_elome_with_anvil` | A tour of the forward model (orbits, limb darkening, exposures, batching), then a fit of Kepler-like photometry with anvil's gradient-based HMC. | `elome`, `anvil` |
| {doc}`02_joint_transit_and_gp` | A transit and correlated stellar variability fitted together: Elómë as the mean model and a Gaussian process over the residual, all twelve parameters sampled at once. | `elome`, `anvil`, `anvilgp` (not yet public) |
| {doc}`03_spherical_and_oblate_planets` | Spherical, oblate and ringed planets side by side: light curves, hybrid limb darkening, exposures, gradients, batching and the sampler API, then what is new for an oblate planet and for a ring. | `elome` |
| {doc}`04_starspots_on_hat_p_11` | Real Kepler data: two starspot crossings in a transit of HAT-P-11b, located with a batched grid search, refined with `anvil.optimize` and sampled with ChEES-HMC; the spots on the stellar disc against the literature. | `elome`, `anvil`, `astropy`, network access once |

`matplotlib` is optional: every cell also prints its numbers.

Measured on an M2 Max, the fits are healthy at the settings used:
tutorial 1 reaches no divergences and $\hat R = 1.002$ in about 25 s,
tutorial 2's twelve-parameter joint fit reaches no divergences,
$\hat R = 1.001$, and every parameter within 1.2σ of the truth in about
two minutes, and tutorial 4's 21-parameter spot fit to real data reaches
no divergences and $\hat R \le 1.10$ (the impact parameter, the slowest
direction) in about ten minutes.

## Running them yourself

Each notebook is generated from a build script beside it, so the prose
and code live in reviewable source rather than in notebook JSON:

```bash
python docs/tutorials/build_03.py docs/tutorials/03_spherical_and_oblate_planets.ipynb
jupyter nbconvert --to notebook --execute --inplace docs/tutorials/03_spherical_and_oblate_planets.ipynb
```

The committed notebooks are executed on Apple Silicon before each
release, which is why the documentation can show their outputs without
running them.
