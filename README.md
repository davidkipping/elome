# Elómë &nbsp;<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/davidkipping/elome/main/docs/assets/elome_tengwar_dark.svg"><img src="https://raw.githubusercontent.com/davidkipping/elome/main/docs/assets/elome_tengwar_light.svg" alt="Elómë in Tengwar" height="44" align="top"></picture>

**Analytic, differentiable transit light curves on Apple Silicon.**

[![Documentation](https://readthedocs.org/projects/elome/badge/?version=latest)](https://elome.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/davidkipping/elome/blob/main/LICENSE)

Elómë computes exoplanet transit light curves exactly and in closed form,
with analytic gradients in every parameter. It runs on
[MLX](https://github.com/ml-explore/mlx), in float32 on the Apple-Silicon
GPU through fused Metal kernels, or in float64 on the CPU. It models
spherical planets with uniform, linear, quadratic, polynomial and
*hybrid* limb darkening, oblate and ringed planets and spotted stars
with the hybrid laws, which keep their light curves elementary, and
planets with moons. The frontend follows
[batman](https://github.com/lkreidberg/batman).

*Elómë* (eh-LOH-meh) is Quenya for "star-dimness".

## Installation

```bash
pip install elome
```

Elómë needs Python 3.10 or later, MLX and NumPy. Its GPU kernels need a
Mac with Apple Silicon; the float64 path runs wherever MLX runs. On
Linux, `pip install "elome[cpu]"` adds MLX's CPU backend.

## Example

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

## Coming from batman

Elómë has batman's interface, so most batman scripts run after one line:

```python
import elome as batman
```

`TransitParams`, `TransitModel` and `light_curve` work as in batman, with
the uniform, linear and quadratic laws, eccentric orbits, supersampling
and secondary eclipses, and so do batman's orbit helpers
(`get_t_secondary`, `get_t_periastron`, `get_t_conjunction`,
`get_true_anomaly`); its `max_err`, `nthreads` and `fac` are accepted and
ignored, since every law here is a closed form. On a Mac with Apple
Silicon the light curve is computed on the GPU, in float32, to about
10⁻⁷ of the flux; `dtype=mx.float64` gives the exact float64 path on
the CPU. On an Apple M2 Max, under the quadratic law, against batman on one
thread:

| | batman | Elómë (GPU) |
|---|---:|---:|
| one light curve, 10⁵ points | 1.9 ms | 0.38 ms |
| one light curve, 10⁷ points | 193 ms | 9.5 ms |
| 512 light curves, 10⁵ points each | 1.8 s | 0.010 s |

batman's square-root, logarithmic, exponential, power-2 and four-parameter
laws are not here. The hybrid laws take their place: they follow real
stars' limb darkening as well as the four-parameter law or better, in
closed form and with gradients, and `elome.hybrid_weights("nonlinear", u)`
turns batman coefficients into hybrid4 weights (a light curve within a few
ppm of batman's).

Points are times spread evenly over a transit and its shoulders, about
half of them in transit; see the
[spherical comparison](https://elome.readthedocs.io/en/latest/spherical/accuracy-and-cost.html).

## Documentation

**[elome.readthedocs.io](https://elome.readthedocs.io)**: installation,
guides for spherical, oblate and ringed planets, spotted stars and
exomoons, fitting
with samplers, tutorials
and the API reference.

## Citation

If Elómë contributes to your research, please cite the Elómë paper
(Kipping, in preparation). If you use
the hybrid limb-darkening laws, oblate or ringed planets or starspots, please
also cite the
hybrid-law papers, also in preparation; see
[Citing Elómë](https://elome.readthedocs.io/en/latest/about/citing.html).

## License

MIT; see [LICENSE](https://github.com/davidkipping/elome/blob/main/LICENSE).
