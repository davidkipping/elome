# Elómë &nbsp;<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/davidkipping/elome/main/docs/assets/elome_tengwar_dark.svg"><img src="https://raw.githubusercontent.com/davidkipping/elome/main/docs/assets/elome_tengwar_light.svg" alt="Elómë in Tengwar" height="44" align="top"></picture>

**Analytic, differentiable transit light curves on Apple Silicon.**

[![Documentation](https://readthedocs.org/projects/elome/badge/?version=latest)](https://elome.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/davidkipping/elome/blob/main/LICENSE)

Elómë computes exoplanet transit light curves exactly and in closed form,
with analytic gradients in every parameter. It runs on
[MLX](https://github.com/ml-explore/mlx), in float32 on the Apple-Silicon
GPU through fused Metal kernels, or in float64 on the CPU. It models
spherical planets with uniform, linear, quadratic, polynomial and
*hybrid* limb darkening, and oblate and ringed planets and spotted stars
with the hybrid laws, which keep their light curves elementary. The frontend follows
[batman](https://github.com/lkreidberg/batman).

*Elómë* (eh-LOH-meh) is Quenya for "star-dimness".

## Installation

```bash
pip install elome
```

Elómë needs a Mac with Apple Silicon for its GPU kernels, Python 3.10 or
later, MLX and NumPy.

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

## Documentation

**[elome.readthedocs.io](https://elome.readthedocs.io)**: installation,
guides for spherical, oblate and ringed planets and spotted stars, fitting
with samplers, tutorials
and the API reference.

## Citation

If Elómë contributes to your research, please cite the Elómë paper
(Kipping, in preparation, *Journal of Open Source Software*). If you use
the hybrid limb-darkening laws, oblate or ringed planets or starspots, please
also cite the
hybrid-law papers, also in preparation; see
[Citing Elómë](https://elome.readthedocs.io/en/latest/about/citing.html).

## License

MIT; see [LICENSE](https://github.com/davidkipping/elome/blob/main/LICENSE).
