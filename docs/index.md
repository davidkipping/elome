# Elómë

**Analytic, differentiable transit light curves on Apple Silicon.**

Elómë computes the light curve of a planet crossing its star exactly and
in closed form, with analytic gradients in every parameter. A
float32 path fuses each model into hand-written Metal kernels on the
Apple-Silicon GPU, and a float64 path runs the same mathematics on the
CPU. Both run through [MLX](https://github.com/ml-explore/mlx).

- **Familiar.** The frontend follows [batman](https://github.com/lkreidberg/batman):
  `TransitParams`, `TransitModel`, `light_curve`.
- **Exact.** In float64 the light curves agree with a 30-digit direct
  integral to $2\times10^{-16}$.
- **Fast.** A light curve of $10^7$ points in about 10 ms on the GPU, and
  about 50,000 light curves of $10^5$ points each per second when
  parameter sets are batched.
- **Differentiable.** Every parameter, including the limb darkening,
  the eccentricity and the planet's shape, has an analytic gradient.
- **Spherical, oblate and ringed planets, and spotted stars.** A
  flattened planet's light curve, a ringed one's and a spotted star's
  stay elementary with the *hybrid* limb-darkening laws.

```python
import numpy as np
import elome

params = elome.TransitParams()
params.t0, params.per, params.rp, params.a = 0.0, 3.0, 0.1, 10.0
params.inc, params.ecc, params.w = 88.0, 0.0, 90.0
params.limb_dark, params.u = "quadratic", [0.4, 0.25]

t = np.linspace(-0.1, 0.1, 1000)
flux = elome.TransitModel(params, t).light_curve(params)
```

## Choose your path

Most of what follows applies to every planet. The paths differ in
their geometry, in the limb-darkening laws available, and in what to
watch for when fitting.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Spherical planets
:link: spherical/index
:link-type: doc

The standard transit: uniform, linear, quadratic, polynomial or hybrid
limb darkening; circular and eccentric orbits; secondary eclipses;
companions larger than their star.
:::

:::{grid-item-card} Oblate planets
:link: oblate/index
:link-type: doc

A planet flattened by rotation projects to an ellipse. Two more
parameters (flattening and sky angle), the hybrid limb-darkening laws,
and the same fast, differentiable machinery.
:::

:::{grid-item-card} Ringed planets
:link: rings/index
:link-type: doc

A spherical or oblate planet with a flat ring in its equatorial plane,
opaque or translucent: four more parameters, exact limb darkening, every
gradient.
:::

:::{grid-item-card} Starspots
:link: spots/index
:link-type: doc

Circular spots and faculae on a rotating star, crossed by the planet:
the crossings and the rotational modulation, exact limb darkening, every
gradient.
:::
::::

New to Elómë? Start with {doc}`getting-started/installation` and the
{doc}`getting-started/quickstart`.

## Citing Elómë

If Elómë contributes to your research, please cite the Elómë paper
(*Journal of Open Source Software*, in preparation). The hybrid
limb-darkening laws and their coefficients have papers of their own;
see {doc}`about/citing`.

```{toctree}
:hidden:
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/concepts
```

```{toctree}
:hidden:
:caption: Spherical planets

spherical/index
spherical/light-curves
spherical/limb-darkening
spherical/large-occultors
spherical/accuracy-and-cost
spherical/fitting
```

```{toctree}
:hidden:
:caption: Oblate planets

oblate/index
oblate/geometry
oblate/light-curves
oblate/accuracy-and-cost
oblate/fitting
```

```{toctree}
:hidden:
:caption: Ringed planets

rings/index
rings/geometry
rings/light-curves
rings/accuracy-and-fitting
```

```{toctree}
:hidden:
:caption: Starspots

spots/index
spots/geometry
spots/light-curves
spots/accuracy-and-fitting
```

```{toctree}
:hidden:
:caption: Guides for both

guides/exposures
guides/batching
guides/gradients-and-precision
guides/sampler-api
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
tutorials/01_elome_with_anvil
tutorials/02_joint_transit_and_gp
tutorials/03_spherical_and_oblate_planets
tutorials/04_starspots_on_hat_p_11
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/hybrid-laws
reference/performance
reference/how-it-works
```

```{toctree}
:hidden:
:caption: About

about/citing
about/name
about/changelog
```
