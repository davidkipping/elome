# Spot crossings: cross-code benchmark

Elómë's starspot model against the two modern differentiable codes that
model a planet crossing a spotted star:
[jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet)'s `starry`
module (the spot as a spherical-harmonic expansion of the stellar
surface) and [spotter](https://github.com/lgrcia/spotter) (Garcia,
Rackham & Panwar 2025; the surface as a HEALPix pixel map). Both are JAX
codes. Results: [RESULTS.md](RESULTS.md) (generated).

Elómë integrates a sharp-edged circular spot exactly; the other two
approximate the surface, and their accuracy is set by a resolution -- the
spherical-harmonic degree `ydeg`, the HEALPix `nside` -- which also sets
their cost. The comparison is therefore error against cost as each
resolution rises.

## Files

| file | role |
|---|---|
| `scenario_spot.py` | the crossing, the spots (three cases) and the resolutions |
| `adapters_spot.py` | each code's model in the scenario's terms; a curve and its value + gradient |
| `worker_spot.py` | one measurement per subprocess |
| `run_spot_compare.py` | `--precision` against the brute-force spot integral; `--speed` the timings |
| `make_report_spot.py` | `results_spot.json` -> `RESULTS.md` |

```bash
cd benchmarks/spot_compare
.venv/bin/python run_spot_compare.py --precision     # ~30 min (JAX compilation)
.venv/bin/python run_spot_compare.py --speed         # ~1 h; quiet machine, GPU idle
.venv/bin/python make_report_spot.py
```

## The scenario

A planet of radius 0.1 crosses the disc along the chord y = 0.2, at sky
positions given directly (no orbit, so no orbital conventions enter). The
star is seen equator-on, aligned, and does not turn during the crossing.
Limb darkening I(mu) = 1 - w (1 - mu^2), w = 0.4: Elómë's `hybrid2` with
its pole weight at 0, the quadratic law (2w, -w) for the others.

| case | spots (longitude, latitude, radius in degrees; contrast) |
|---|---|
| `chord` | one spot on the chord, larger than the planet: (10, 11.5, 8; 0.5) |
| `overlap` | two overlapping spots on the chord, deficits adding: (5, 11.5, 8; 0.5), (14, 14.5, 5; 0.6) |
| `limb` | a foreshortened spot near the limb, on the chord: (55, 11.5, 8; 0.5) |

Compared: the relative flux, each code's flux over its own unocculted
flux, which removes the codes' different normalisations. The reference is
`tests/spot_oracle.py`, a brute-force integral that shares no code with
any code under test; Elómë's float64 graph agrees with it to 3e-16.

## Mappings (checked before measuring)

| code | mapping | check |
|---|---|---|
| jaxoplanet | `Surface(y=sum of ylm_spot(ydeg)(1 - c, radius, lat, lon), u=(2w, -w), inc=pi/2, obl=0)`, `surface_light_curve(surface, r, x, y, z=1, theta=0)` | its spot-free transit (ydeg 10) matches Elómë's to 3e-10 ppm |
| spotter | `Star(y=1 - sum (1 - c) spot(nside, lat, lon, radius), u=(2w, -w), inc=pi/2, obl=None)`, `transit_light_curve(star, x, y, z=1, r, time=0)` | spot positions by an occultor scan |

With these conventions a spot at (lon, lat) sits at sky
(cos(lat) sin(lon), sin(lat)) in all three codes: a small occultor
scanned over the disc finds each code's spot where Elómë puts it.

**spotter with `obl=0.0`.** spotter 0.0.11's `transit_design_matrix`
swaps an obliquity of exactly 0 for 1 (to keep its gradient finite) and
then tests the swapped value, not the original, when deciding whether to
rotate; a star given `obl=0.0` has its occultation turned by 1 radian
(57 degrees) about the line of sight, while its rotational modulation is
not. The adapter passes `obl=None`, which skips the rotation.

## What each code computes

- **Elómë** integrates the occulted intensity exactly over the planet's
  disc and each spot's visible cap (Green's theorem along every outline).
- **jaxoplanet** expands the spot's profile in spherical harmonics to
  degree `ydeg`, smoothed by exp(-l(l+1) s^2 / 2) with s = 2 / ydeg (so
  that the expansion does not ring); at `ydeg` 30 that blurs the spot's
  edge over about 4 degrees. The occultation of the smooth map is exact up
  to a quadrature of order 20. Its compilation time grows steeply with
  `ydeg`: at 30, 5 minutes for the value and 35 for the gradient. The
  sweep stops there; at `ydeg` 40 the expansion was numerically unstable
  (errors of several percent).
- **spotter** samples the surface at HEALPix pixel centres and counts a
  pixel as occulted when its centre is inside the planet (a steep
  sigmoid); the error falls as the pixel count rises, and the transit
  design matrix is (times x pixels) in memory.

## Environment (local only; `.venv/` is git-ignored)

A separate environment, Python 3.12 (JAX 0.12 needs it), with Elómë
installed from this checkout:

```bash
conda create -p benchmarks/spot_compare/.venv python=3.12
benchmarks/spot_compare/.venv/bin/python -m pip install spotter jaxoplanet healpy mpmath matplotlib
benchmarks/spot_compare/.venv/bin/python -m pip install -e .
```

Measured with spotter 0.0.11, jaxoplanet 0.2.0, JAX 0.12.0.1 (CPU), MLX
0.32.4.
