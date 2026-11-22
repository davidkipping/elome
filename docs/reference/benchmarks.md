# Benchmarks

Every speed and accuracy figure in these pages comes from a script in the
repository's
[`benchmarks/`](https://github.com/davidkipping/elome/tree/main/benchmarks)
directory, and each can be run again on your own machine.

## The machine

An **Apple M2 Max**: 12 CPU cores (8 performance and 4 efficiency), a
30-core GPU and 64 GB of memory, under macOS. Each configuration runs in
its own process, so that no thread pool, compiled code or thermal state
carries over from one to the next, on a quiet machine (no indexing or
other load). Timings are medians over repeated calls, with compilation
and warm-up outside the timed region.

Your numbers will differ with your machine. The ratios between codes on
the same machine are what carry over: a Mac with more GPU cores (a Max or
an Ultra) runs Elómë's float32 kernels faster, and the CPU codes scale
with the speed of one core, or of several where a code uses them.

## The scripts

| comparison | page | scripts | results |
|---|---|---|---|
| spherical planets: batman, PyTransit, jaxoplanet, exoplanet-core, pizzetti | {doc}`../spherical/accuracy-and-cost` | [`bench_precision.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_precision.py), [`bench_speed.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_speed.py), [`bench_law_cost.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_law_cost.py) | [`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/RESULTS.md) |
| oblate planets: squishyplanet, JoJo, GreenLantern, eclipsoid | {doc}`../oblate/accuracy-and-cost` | [`bench_oblate.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_oblate.py), [`oblate_compare/`](https://github.com/davidkipping/elome/tree/main/benchmarks/oblate_compare) | [`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/oblate_compare/RESULTS.md) |
| ringed planets | {doc}`../rings/accuracy-and-fitting` | [`bench_rings.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_rings.py), [`ring_compare/`](https://github.com/davidkipping/elome/tree/main/benchmarks/ring_compare) | [`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/ring_compare/RESULTS.md) |
| starspots | {doc}`../spots/accuracy-and-fitting` | [`bench_spots.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_spots.py), [`spot_compare/`](https://github.com/davidkipping/elome/tree/main/benchmarks/spot_compare) | [`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/spot_compare/RESULTS.md) |
| the sampler's batched kernels | {doc}`performance` | [`bench_vjp.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_vjp.py), [`bench_tau_kernel.py`](https://github.com/davidkipping/elome/blob/main/benchmarks/bench_tau_kernel.py) | [`vjp.json`](https://github.com/davidkipping/elome/blob/main/benchmarks/vjp.json), [`tau_kernel.json`](https://github.com/davidkipping/elome/blob/main/benchmarks/tau_kernel.json) |

Each other page names the script behind each of its tables. The oblate,
ring and spot comparisons each have a README in their directory with
their own environment: some of those codes need building.

## Running the spherical comparison

```bash
git clone https://github.com/davidkipping/elome
cd elome
python -m venv .venv && source .venv/bin/activate
pip install -e . -r benchmarks/requirements.txt

python benchmarks/bench_precision.py                # against a 30-digit integral
python benchmarks/bench_speed.py --out my_speed.json
python benchmarks/bench_law_cost.py
```

- **`bench_precision.py`** compares each code with a 30-digit direct
  integral of the light curve. The integrals are cached in the
  repository, so it takes seconds.
- **`bench_speed.py`** times one light curve from $10^3$ to $10^7$
  points, eccentric orbits, a batch of 512 parameter sets and a value
  with its gradient, each in its own process. Give it a new `--out` file:
  it skips measurements already in its results file, and the default is
  this repository's. `--codes elome_fp32 pizzetti`, say, times only those
  codes. A full run takes about half an hour.
- **`bench_law_cost.py`** times each code's own limb-darkening laws
  against its quadratic law.

`python benchmarks/make_report.py --speed my_speed.json --out my_results.md`
renders them into a report like
[`RESULTS.md`](https://github.com/davidkipping/elome/blob/main/benchmarks/RESULTS.md). Elómë's float32 kernels need a Mac with Apple silicon;
elsewhere its float32 path runs as an MLX graph on the CPU, and its
numbers there are not those of the kernels.
