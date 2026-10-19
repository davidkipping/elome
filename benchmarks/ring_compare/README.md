# Ringed planets: cross-code benchmark

Elómë's ringed-planet model against the other public ringed-planet transit
codes: [pyPplusS](https://github.com/edanrein/pyPplusS) (Rein & Ofir 2019),
squishyplanet's `RingedSystem` (Cassese et al. 2024),
[JoJo](https://github.com/Flippedx/JoJo)'s `ring_lc` (Liu & Zhu 2024) and
[pryngles](https://github.com/seap-udea/pryngles-public)' `RingedPlanet`
(Zuluaga et al. 2022). Results: [RESULTS.md](RESULTS.md) (generated).

## Files

| file | role |
|---|---|
| `scenario_ring.py` | the transit; `BENCH_SCENARIO=translucent` (tau = 0.6) or `edge` (ring_f = 0.95) |
| `adapters_ring.py` | each code's parameter mapping; single curve, batch, value + gradient |
| `worker_ring.py` | one timed measurement (and its peak memory) per subprocess |
| `run_ring_compare.py` | precision against `tests/ring_oracle.py`, then the timings |
| `make_report_ring.py` | `results_ring.json` -> `RESULTS.md` |
| `oracle_<scenario>.npy` | the cached reference light curves (minutes to compute) |

```bash
cd benchmarks/ring_compare
for s in translucent edge; do BENCH_SCENARIO=$s .venv/bin/python run_ring_compare.py --precision; done
.venv/bin/python run_ring_compare.py              # opaque precision + timings, quiet machine
.venv/bin/python make_report_ring.py
```

Each code runs under its own interpreter where its dependencies conflict
(`adapters_ring.PYTHON`): squishyplanet and JoJo in `../oblate_compare/.venv`,
pryngles in `.venv-pryngles` (pryngles 1.0.2 needs Python >= 3.12).

## Parameter mappings (checked against the reference before timing)

| code | mapping | agreement |
|---|---|---|
| pyPplusS | sky positions; absolute radii per point; `ring_inclination` = arccos(1 - ring_f), `ring_rotation` = theta; `opacity` = the blocked fraction T; nonlinear law with `c4` = w | 6e-7 (order 5), 4e-10 (order 20) |
| squishyplanet | `ring_obliq`, `ring_prec` solved so the ring projects to (ring_f, theta); radii in R*; quadratic (2w, -w); opaque only | 7e-16 |
| JoJo | `f_r` = ring_f, obliquity = -theta (as its oblate model); `opacity` = T; a/R* from the stellar density; quadratic (2w, -w) | 1e-14 |
| pryngles | `RingedPlanet`, ring tilt arcsin(1 - ring_f) to an orbit seen edge-on; the planet's offset placed along the ring's projected axes; `taur` = tau; nonlinear law with `c4` = w; 5000 + 5000 samples | see below |

**Known issues in the codes compared** (versions as above):

- pryngles' `RingedPlanet.updateTransit` adds the ring's blocked light over
  the planet's disc to the planet's, so wherever the ring crosses the planet
  that area counts twice: 4.7e-3 in this scenario (depth 0.024). An exact
  model with the same double count reproduces pryngles to its sampling
  noise (3e-5).
- JoJo's `ring_lc` sizes the ring 1/sqrt(1 - f_r) too large when the planet
  doesn't cross the ring band (r_p <= r_in (1 - f_r)); not the case in this
  scenario, where it agrees to 1e-14.

## Environment (local only; the venvs are git-ignored)

```bash
cd benchmarks/ring_compare
python3.11 -m venv .venv
.venv/bin/pip install numpy scipy mpmath matplotlib pyppluss
.venv/bin/pip install -e ../..                     # Elómë
conda create -p ./.venv-pryngles python=3.13       # any Python >= 3.12
.venv-pryngles/bin/python -m pip install pryngles==1.0.2 numpy scipy
```

pyPplusS 0.1.5.3 predates NumPy 2: the adapter sets `np.NaN = np.nan`.
