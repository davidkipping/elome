# Installation

## Requirements

- **For the float32 GPU kernels, a Mac with Apple Silicon** (M1 or
  later).
- **For the float64 path, anywhere MLX runs.** It runs on MLX's CPU
  stream on every platform, so it needs no GPU. It is tested on macOS and
  on Linux (ARM, with MLX's CPU backend). MLX also publishes CUDA
  backends for Linux and wheels for Windows, on which Elómë is untested.
- **Python 3.10 or later**, with [MLX](https://github.com/ml-explore/mlx)
  0.30 or later and NumPy. pip installs both for you.

Without a usable Metal device (on Linux, say), `metal_available()`
returns `False` with a warning, and float32 calls run an MLX graph of the
same model instead of the fused kernels. The results agree to float32
rounding, but on a Mac a forward evaluation is about 25 times slower and
a value with its gradient about 12 times slower.

## Install

From PyPI:

```bash
pip install elome
```

On Linux, MLX needs a backend installed with it; `pip install "elome[cpu]"`
adds MLX's CPU backend.

Or from a clone, for development:

```bash
git clone https://github.com/davidkipping/elome
cd elome
pip install -e .
```

To check that the GPU kernels are available:

```python
>>> import elome
>>> elome.metal_available()
True
```

## Optional packages

| package | what it adds | install |
|---|---|---|
| [anvil](https://github.com/davidkipping/anvil) | the GPU MCMC engine used in {doc}`../spherical/fitting` and the tutorials | `pip install git+https://github.com/davidkipping/anvil` |
| matplotlib | the figures in the tutorials (every cell also prints its numbers) | `pip install matplotlib` |
| emcee | the CPU ensemble sampler recipe in {doc}`../spherical/fitting` | `pip install emcee` |

Elómë itself depends only on MLX and NumPy. It is a forward model:
parameters and times in, flux and its derivatives out. Samplers and
likelihoods sit on the other side of a one-way dependency, so nothing
in Elómë needs anvil except the ready-made sampler targets in
`elome.anvil`. Those fail with an `ImportError` that gives the install
command if it is missing.

## Running the tests

```bash
pip install -e ".[test]"               # pytest, scipy, batman, mpmath
                                       # (".[test,cpu]" on Linux)
python -m pytest tests -m "not slow"   # the fast suite
python -m pytest tests                 # everything
```

The suite checks every path against direct numerical integration
(30-40 digits, with [mpmath](https://mpmath.org)), against batman, and
the float32 kernels against the float64 reference. Without Metal, the
tests of the kernels skip.
