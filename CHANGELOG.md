# Changelog

All notable changes to Elómë. Versioning: semantic-ish (MAJOR.MINOR.PATCH);
every release is tagged `vX.Y.Z` in git.

## [Unreleased]

### Added
- **anvil targets for the hybrid laws and spotted stars.**
  `make_transit_target(..., limb_dark="hybrid2" | "hybrid4" | "hybrid5")`
  samples the law's weights through its q prior (uniform on its physical
  profiles), circular or eccentric, with `exp_time=` integrating each
  exposure; `n_spots=K` adds a spotted star (per spot `lonj`, `sinlatj`,
  `radj`, `contrastj`; the star's `prot`, `cos_istar`, `lam`), its clock
  measured from `spot_epoch` (default: the predicted mid-transit nearest
  the data's middle). Bounds given by spot field (`"rad"`) cover every
  spot. `transit_param_names` gives the parameter order for any options;
  `make_hybrid_transit_flux` is the model function on its own. A spotted
  target sets `compile_step=False`, to pass to `anvil.run`.
  `ecc_constraint_penalty` takes the column of `secosw` (`ik=`).

## [0.21.0] — 2026-10-22

### Added
- **Starspots** (`params.spots`, hybrid laws only): circular spots on a
  rotating star crossed by a spherical planet. Each spot is a row of
  longitude, latitude, angular radius (degrees) and contrast (its
  intensity over the photosphere's; above 1, a facula); overlapping spots
  add, so an umbra in a penumbra is two concentric spots. The star turns
  with period `prot` about a spin axis set by `istar` and the
  sky-projected obliquity `lam` (Fabrycky & Winn's lambda), from
  longitudes measured at `spot_epoch` (default `t0`). The flux is
  normalised to the unspotted star, so out of transit it is the spots'
  rotational modulation. Exact by Green's theorem in float64 and float32,
  differentiable in every parameter through `light_curve_mx`, batched in
  `light_curves` (`spots` of shape (n_sets, K, 4)). `shape_cols_spots`
  and `flux_dev_spots` give the columns at a sky position directly.
  Primary transits. `integration="contact"` splits each exposure at the
  planet's limb contacts and at every planet-spot tangency inside it,
  found exposure by exposure (the star turns between transits); its
  split points carry their implicit derivatives, so `light_curve_mx`'s
  gradient is that of the quadrature evaluated. A spot rim reaching the
  limb, and two spots starting to overlap, are not split (slow,
  curvature-only kinks). A float32 model on the GPU runs fused Metal
  kernels (`use_metal=False` for the graph): the columns and their
  gradient per point, and the contact rule's event search per exposure;
  the orbit, the star's rotation and the quadrature stay graph operations
  around them. Samplers take a spotted star through `flux_dev_from_tau`
  (`spots=`, `spot_time=`, `prot=`, `istar=`, `lam=`; angles in radians):
  the planet on `tau`, the star on `spot_time`, each point's time since
  the spots' epoch, both differentiable, with per-chain spot tables and
  every exposure rule. A spotted model is called eagerly: under
  `mx.compile` or `mx.vmap` it raises.

### Documentation
- A **Starspots** section: overview, geometry and conventions, light
  curves (from the frontend and at sky positions; the contact rule's
  split points), and accuracy, cost and fitting; a "How it works" section
  on spots, and `flux_dev_spots`, `shape_cols_spots` and `spot_vector` in
  the API reference. The sampler guide gains a section on spotted stars
  (the two clocks, per-chain spot tables).
- The sampler guide's exposure-kernel and hybrid-law tables and the oblate
  kernel table, re-measured (the spherical contact-rule kernels are about
  15% faster than those tables said; an oblate planet under the contact
  rule costs 53–63 times a spherical one, not 19–21), and now checked by
  `benchmarks/verify_doc_claims.py` against the results files
  `bench_tau_kernel.py`, `bench_hybrid.py` and `bench_oblate.py` write.

## [0.20.0] — 2026-10-21

### Changed
- **`TransitModel.light_curves` in float32 on the GPU** (uniform, linear
  and quadratic laws, primary transits, without the contact rule) runs the
  fused model kernel that `light_curve` uses, one row per set: 2.3-3.4
  times faster (256 sets x 4,096 points: 3.9 to 1.2 ms), and each row is
  `light_curve`'s bit for bit.
- **Ringed planets' contact rule in float64**: the kinks of the planet and
  both ring edges are found in one stacked solve instead of three, with
  the same result bit for bit. A 200-point contact-rule `light_curve` is
  2.7 times faster (3.9 to 1.5 s; the first call, 8.8 to 4.7 s).
- **The quadratic law's float32 Metal kernels** run 8 iterations of the
  elliptic-integral recursion instead of 10 (it converges within 6 in
  float32 over the whole argument range): 9% off the z kernel's forward
  pass, 7% off its value and gradient, 4-5% off the contact-rule tau
  kernels. Accuracy against float64 is unchanged (the same maximum error
  over 1.8 million points, a slightly smaller RMS), and the outputs differ
  from 0.19.0's at float32 rounding.
- **Eccentric contacts in float32 on the GPU** are solved by one Metal
  kernel, one thread per set, instead of a graph of ~1,000 small
  operations: 7.1 to 0.7 ms whatever the number of sets. An eccentric
  contact-rule `flux_dev_from_tau` call is 6.6 times faster at 64 sets
  (7.6 to 1.2 ms) and 3.1 times at 512; the oblate and ring kernels'
  contact windows use it too (an eccentric oblate call at 64 sets, 26 to
  20 ms). The same steps as the graph's solve, as accurate (over 20,000
  orbits up to e = 0.95, grazing ones included); circular sets are
  unchanged bit for bit, eccentric ones change at float32 rounding.
- **`TransitModel.light_curves` on the graph** (float64, and the float32
  cases the fused kernel does not take) runs compiled, as `light_curve`
  does: 1.2-1.8 times faster (64 sets x 2,000 points, quadratic: 32 to
  23 ms; contact rule: 1.09 to 0.76 s; 256 x 4,000: 252 to 172 ms).
  Values move by a few float64 ulp (the fused arithmetic).
- **Oblate planets' float64 graph runs in blocks** of at most 262,144
  evaluation points, as the ringed planets' does, so its memory no longer
  grows with the light curve: a 200,000-point contact-rule light curve
  (5 million evaluation points) peaks at 0.6 GB, where it needed ~15 GB.
  The contact rule's contacts are solved once per set, not per block. A
  `TransitModel` call larger than one block compiles one block rather than
  the whole call (compiling a blocked call whole traces every block:
  minutes at 30 blocks). Values are unchanged bit for bit, but for a
  `TransitModel` call over one block, which moves by an ulp.
- **Oblate planets' contacts on the graph** (float64, and float32 off the
  GPU): each pair of contacts (ingress and egress, outer then inner) is
  one stacked solve rather than two, which halves the graph: the solve is
  about 2 times faster for a few sets (1.5 at 1,024), a `TransitModel`
  first call 6-14% faster and a steady one 2-5%. Bit for bit unchanged.

### Documentation
- The batching guide's `light_curves` figure, now measured by
  `benchmarks/verify_doc_claims.py`: about 100 times a Python loop with a
  list of sets (not 160), several hundred with array-valued fields.
- Correction to 0.19.0's notes: re-measured on a quiet machine, the
  eccentric contact-rule kernel is about 2.6 times faster, not 3; the
  other figures stand.
- The oblate cross-code comparison's float64 memory, re-measured with the
  blocked graph: 1.3 GB at 10^7 points (was 21 GB), capped at any length.
- The fused Metal kernel's timings re-measured with the 8-iteration core
  (19.6 / 54.4 ms circular, 25.7 / 68.6 ms eccentric, at 1024 chains ×
  65,536 points), and the batching guide's batched call (about 2.7 ms,
  was 2.9).

## [0.19.0] — 2026-10-20

### Fixed
- **Eccentric contact times on grazing transits.** On an eccentric orbit
  the planet's closest approach to the star's centre is not at
  conjunction. The contact solver assumed it was, and on a grazing transit
  it could return the ingress at conjunction, or miss a transit that
  happens only off conjunction (b > 1 + r there). The contact rule then
  split each exposure in the wrong place: at e = 0.88 the flux was off by
  3.5e-5, 1% of the depth, against ~1e-7 normally. The solver
  (`exposure.contact_offsets_anchored`) now finds the closest approach,
  brackets each contact, and solves the four together; on 2,600 random
  orbits up to e = 0.95 its float64 contacts are within 1e-12 of the
  transit's duration of a bisected root. Other eccentric contact-rule
  results move by up to a few parts in 1e6 of their gradients' and values'
  scale; circular orbits are unchanged.

### Changed
- **Faster float32 GPU kernels** (`flux_dev_from_tau`, every law). An
  exposure that ends before the first contact or starts after the last is
  skipped rather than evaluated node by node. Circular orbits: about 55%
  off a sampler's uniform-phase grid under the contact rule, 12-15% on a
  grid around the transit, 6-25% with no exposure integration, with
  results bit for bit the same. Eccentric orbits: the contact-rule kernel
  is 3 times faster on a uniform-phase grid, and a whole call 31% faster
  (19% with no exposure integration).

## [0.18.8] — 2026-10-20

### Fixed
- **Eccentricities above 0.999 for oblate and ringed planets now raise**
  ("eccentricity must be in [0, 0.999] for an oblate or ringed planet"),
  or give NaN for a traced value, instead of a wrong light curve. These
  routes rebuild the orbit from the impact parameter, whose inclination
  holds to e = 0.999; above it the transit was wrong by up to 2% of its
  depth (e = 0.9995). Spherical planets are exact there and still take
  any e < 1.
- **`flux_dev_metal` on the graph route** (float64, the CPU stream, or no
  Metal) takes per-chain parameters as the kernel does: separations of
  shape (n, m) with r, u1, u2 of shape (n,). It raised before. Calls that
  already worked give the same results bit for bit.
- **NaN gradient in the hybrid laws' pole terms** (float64 and the float32
  graph; the kernels were not affected) at a single separation per pole,
  where an unused branch divided by zero: z^2 = 2 + eps + r^2, outside
  the transit for most radii, inside it for hybrid4's inner pole at
  r = 0.9.

### Documentation
- `w` is the planet's argument of periastron (an RV solution's
  omega_star + 180°), as in batman.
- Secondary eclipses are placed at geometric superior conjunction; the
  light-travel time across the orbit is not included.
- `theta` is measured from the direction of motion at mid-transit, fixed
  on the sky; and a transit at -b with -theta is the mirror image of one
  at b with theta.
- Known limitation, now documented: at exactly rp = 1 with the planet
  centred on the star (within about 1e-12), the flux is off by up to 5e-9
  and its gradients are wrong on every path.

## [0.18.7] — 2026-10-19

Documentation only; the package is unchanged from 0.18.6.

### Fixed
- **The oblate and ring pages** no longer say the gradient in `f` or
  `ring_f` is 0 below 1e-5 and that fits should start at a small positive
  value. Since 0.18.6 the gradient there is the oblate (or elliptical)
  one, and a fit can start at exactly 0.
- **Comparison tables.** Four rounded figures corrected (jaxoplanet's
  accuracy and memory; pyPplusS's and JoJo's errors for a nearly edge-on
  ring), and pyPplusS's order-20 error given over all three ring cases.
  The batched throughput is about 50,000 light curves per second, not
  55,000.
- `benchmarks/verify_doc_claims.py --check` now checks every number in
  the comparison tables, and the figures quoted from them, against the
  results files.

## [0.18.6] — 2026-10-19

### Fixed
- **Gradients at a round planet and a face-on ring.** `f = 0` and
  `ring_f = 0` are the edges of the uniform priors and where fits start,
  and the gradients there were wrong:
  - oblate planets: d/df was 0 below the spherical switch (f < 1e-5 in
    float32, f = 0 in float64) on every path, so a fit started at a
    sphere never moved in f. The value there is still the sphere's; d/df
    is now the oblate slope, taken at the switch (a bias of about 1e-5
    relative). Tutorial 3's advice to start f at 1e-4 in float32 is gone.
  - ringed planets, float32 GPU kernel: within 1e-5 of face-on, d/d ring_f
    lost the ring's geometric term (the wrong sign in the test case), and
    d/df was 0 for a round planet.
  - ringed planets, float32 graph (the CPU stream, or `use_metal=False`):
    NaN gradients in ring_f and f within 1e-5 of face-on or round.
  - float64: d/d ring_f and d/df off by 0.3% and 3% at exactly 0, and
    wrong for a circular outline centred on the star.

  The graph now takes a nearly circular outline's far root (about
  1/(r^2 f)) through its partner inside the disc, as the kernels already
  did; that also fixes float32-graph errors in d/df of up to a factor 2.4
  just above the switch. Cost: about 5% on float64 oblate and ring
  gradients.

## [0.18.5] — 2026-10-19

### Fixed
- **A ringed planet with a flattening outside [0, 1) now raises**, as an
  oblate one does ("flattening f must be in [0, 1)"), instead of returning
  NaN fluxes.
- **Ring opacity in float64.** A Python `ring_tau` was rounded through
  float32 before the opacity was formed, an error of about 1e-8 in the
  ring's blocked fraction (about 1e-10 in flux) on a path otherwise exact
  to rounding.
- `rings.generator_cols_ring` accepts Python numbers for the planet and
  ring arguments, as `shape_cols_ring` does, instead of raising
  `TypeError`.

### Changed
- **Ringed planets in float64: memory bounded on every route.** The
  blocking of 0.18.4 counted times only, and only in `light_curve` and
  `light_curves`. The ring graph is now built in blocks of at most 32,768
  evaluation points -- times x parameter sets x exposure nodes -- inside
  the graph itself, so `light_curves` with many sets, the contact rule and
  `flux_dev_from_tau(..., ring_in=...)` are bounded too (about 1.6 GB at
  10^6 points; about 2.5 GB under the contact rule), bit for bit the same
  light curves, at the same speed. The contact rule finds each set's kinks
  once rather than per block. A gradient needs less as well: 6.8 GB at
  10^5 points, down from 11.8 GB, for about 10% more time.

## [0.18.4] — 2026-10-18

### Changed
- **Ringed planets in float64: memory capped at about 1 GB.** The ring
  graph holds about 20 KB per time point until it is evaluated, so a
  10^6-point light curve needed about 34 GB. `light_curve` and
  `light_curves` now evaluate it in blocks of 32,768 times: about 1 GB at
  any length, bit for bit the same light curve, no slower (10^6 points:
  43 s -> 41 s). `light_curve_mx`, whose graph the caller evaluates, is
  unchanged. The float32 GPU ring kernels were not affected.

## [0.18.3] — 2026-10-18

### Changed
- **Ringed planets in float64 (and the float32 graph on the CPU): about
  2x faster.** The ring's limb-darkening columns now run as one compiled
  MLX graph per law (10^5 points: 7.6 s -> 4.1 s for one light curve).
  The orbit around them stays uncompiled -- MLX 0.32's `mx.compile` still
  fails when both are in one graph (0.32.4 included) -- and so does the
  float32 graph on the GPU (`use_metal=False`), where MLX's fused kernels
  for it abort. The GPU kernels and spherical and oblate models are
  unchanged.

## [0.18.2] — 2026-10-17

### Changed
- **Spherical planets, float32 on the GPU, contact rule: about 3x faster**
  (512 x 5,000, quadratic: 15.0 ms -> 4.4 ms forward, 30.2 ms -> 9.8 ms
  with gradients; the hybrid laws alike). As for oblate planets in
  0.18.1, the kernels skip the quadrature nodes of zero weight; results
  are bitwise unchanged. Benchmark tables in the docs and
  `benchmarks/RESULTS.md` re-measured.

## [0.18.1] — 2026-10-17

### Fixed
- **Ringed planets, float32 on the GPU, with an exposure rule:** the
  ring kernels evaluated every quadrature node of an exposure in one GPU
  thread (25 ring evaluations under the contact rule), long enough that
  macOS aborted the command buffer ("Impacting Interactivity") whenever
  the display needed the GPU -- at any batch size, often surfacing only
  at the next evaluation, after a call that returned incomplete flux.
  Each exposure's nodes are now shared out over several threads (up to 8
  forward, 25 for gradients), skipping the empty sub-intervals the contact
  split leaves, and summed. Results agree with the float64 graph as
  before; spherical and oblate kernels are unchanged.

### Changed
- **Oblate planets, float32 on the GPU, contact rule: about 3x faster**
  (hybrid5, 512 x 5,000: 636 ms -> 205 ms forward, 1.95 s -> 621 ms with
  gradients). The kernels skip the quadrature nodes of zero weight -- the
  empty sub-intervals of the contact split -- which added exactly nothing;
  results are bitwise unchanged.

### Added
- `benchmarks/bench_rings.py`: ringed against spherical and oblate cost;
  the costs are in the docs (Ringed planets: accuracy, cost and fitting).

## [0.18.0] — 2026-10-16

### Added
- **Ringed planets** in `TransitModel` (`TransitParams.ring_in`,
  `ring_out`, `ring_f`, `ring_tau`, with `theta` orienting the ring and an
  oblate planet alike) and `flux_dev_from_tau(ring_in=, ring_out=,
  ring_f=, ring_tau=)`, on every surface: `light_curve`, `light_curves`
  (sequence and array-valued), `light_curve_mx` (differentiable in every
  ring field), the contact and supersample rules, eccentric orbits,
  `ld_basis`. The contact rule splits each exposure at the kinks inside
  it -- every outline's limb tangencies, found per parameter set
  (`elome.ring_kinks`) -- and matches converged supersampling to
  2e-9 on a nearly edge-on ring crossing the limb four times. Float32
  on the GPU runs fused Metal kernels (`metal_rings`, built on the oblate
  kernels' device functions, which are unchanged): forward, the VJP in
  every parameter, `ld_basis`, and a one-thread-per-chain kink solver.
  Against the float64 graph: light curves <= 1e-7 on every exposure rule
  and orbit, gradients <= 1e-6 of their scale. Float64 runs the MLX graph
  on the CPU, uncompiled (MLX 0.32's `mx.compile` fails on it). Oblate
  and spherical models never import or run any ring code.
- `elome.flux_dev_ring`, `elome.shape_cols_ring`: the reference graph
  behind them, at sky positions (exported lazily: importing `elome`
  loads none of the ring code). A flat circular ring in the planet's
  equatorial plane, given by projected parameters like an oblate planet: `ring_in`, `ring_out`
  (radii in units of `rp`), `ring_f` (projected flattening,
  1 - |cos i|; uniform in it is the isotropic prior), the planet's
  `theta`, and `ring_tau` (normal optical depth; inf is opaque). Exact
  hybrid limb darkening: the occultor is a weighted region (1 in the
  planet, 1 - exp(-tau / |cos i|) in the ring) whose flux follows from
  Green's theorem as each outline's arcs weighted by the jump across
  them, with the oblate solution's per-arc integrals. Up to four limb
  crossings per outline (nearly edge-on rings), planet-ring corners in
  closed form, analytic gradients in every parameter. Against two
  independent brute-force integrations: <= 5e-16 (float64) and <= 5e-8
  (float32) of each column's norm, including tangencies, a planet
  centred on the star and rings to |cos i| = 1e-10.

### Documentation
- A "Ringed planets" section: overview, geometry, light curves and
  exposures, accuracy and fitting; the method in "How it works".

### Fixed
- Oblate planets: an unused near-double-root slot in the pole integral
  could, for some root configurations, give a NaN gradient in float32
  (its stand-in factor took a live Newton step). No existing output
  changes (all 1,432 regression arrays bitwise identical).

## [0.17.2] — 2026-10-15

### Documentation
- The hybrid4 pages, docstrings and changelog describe only its
  curvature basis and its prior map.

## [0.17.1] — 2026-10-15

### Fixed
- `hybrid4_to_q_np` raises for weights outside hybrid4's curvature
  simplex, which have no q. It returned q outside [0, 1], which the
  forward map clamped into a different profile. Weights outside describe
  a profile convex in mu^2 somewhere.
- The numpy prior maps (`hybrid2_from_q_np`, `hybrid4_from_q_np`,
  `simplex_from_q_np`) no longer clamp q >= 1e-12, so q on a face of the
  cube gives its weights exactly; the clamp moved depth and vertices by
  up to 1e-3. The MLX maps keep it, for a finite gradient.
- `HybridLaw(mix=)` requires each row to sum to 1, so every shape is 1
  at the limb and I(0)/I(1) = 1 - sum w holds.

### Documentation
- The hybrid-law pages and docstrings write the law's shapes as S_j (for
  hybrid4 its curvature basis), keeping T_j for its terms, and say that
  hybrid4's curvature simplex is a subset of its physical profiles.

## [0.17.0] — 2026-10-15

### Changed
- **hybrid4's weights refer to its curvature basis**: shapes S1..S4
  (`HYBRID4_BASIS`, rows over its terms 1 - mu^4, (1 - mu^2)^2, Pi_eps0,
  Pi_eps1), with S1 = 1 - mu^2 and S2 = (1 - mu^2)^2. In this basis the
  simplex w >= 0, sum w <= 1 is *exactly* the set of hybrid4 profiles
  that are positive, monotonic, concave in mu^2 and of convex slope in
  mu^2, the basis the hybrid-law coefficients paper tabulates. The
  `ld_basis` columns for hybrid4 are [E0, S1..S4], and
  `hybrid_norms("hybrid4")` follows.

### Added
- `hybrid4_from_q` (MLX), `hybrid4_from_q_np`, `hybrid4_to_q_np`: the
  uniform prior map from the unit cube onto hybrid4's curvature simplex,
  the analogue of Kipping's q1, q2 (q1 = (1 - I(0))^4).
- `HybridLaw(..., mix=M)`: a law whose shapes are combinations M T of
  its terms; `hybrid4_basis(eps)` gives the curvature basis for any two
  poles.

## [0.16.4] — 2026-10-15

### Fixed
- **NaN gradients for small, nearly circular oblate planets in float32.**
  For flattening just above the spherical switch (f = 1e-5) and a small
  planet (r f below about 1.5e-7: an Earth-sized planet with f < 2e-5,
  any planet with r < 0.013 at f = 1e-5), d/dr and d/df were NaN at every
  partial-transit point. One root of each pole's quartic sits at
  |z| ~ 2/(A f), and its derivative step overflowed float32. Each root
  outside the unit circle (|z| > 2) is now evaluated through its partner
  1/conj(z) inside it, which is exact and better conditioned: gradients
  are finite everywhere down to r = 1e-5, and d/dr and d/df are closer to
  the float64 reference across the oblate domain.

## [0.16.3] — 2026-10-15

### Changed
- **Oblate planets are 1.2-1.6x faster in float32.** Each quartic's roots
  (the limb crossings and one per limb-darkening pole) are now seeded by
  Ferrari's closed form and polished (Aberth, then six Newton steps), with
  the balanced QR eigensolver only as the fallback for a quartic that fails
  the check. That replaces one QR per point plus level-to-level warm
  starts, which failed the check on most partial-overlap points and then
  paid a second QR. Nearly circular planets (|S2| < 3e-3 |S1|, where the
  closed form cancels in float32) keep the previous method, bitwise.
  Accuracy is unchanged: against the float64 reference, on 60,000 random
  geometries, the overlap and tangency stress sets and 24 light-curve
  configurations, the new and old kernels are at parity; spherical and
  float64 outputs are bitwise unchanged. Measured at 512 x 5,000: hybrid4
  forward 22.8 -> 16.0 ms without exposure integration; hybrid2 forward with
  the contact rule 481 -> 314 ms.

### Documentation
- Oblate timings re-measured on a quiet machine (the cost table, the
  ratios to spherical planets, the comparison with other oblate-planet
  codes, where Elómë's float32 kernel is now the fastest at every size
  measured); the tutorials re-executed. The cross-code report states its
  small-N result from the data.

## [0.16.2] — 2026-10-13

Fixes from a review of the package.

### Fixed
- **A missing parameter in `light_curves` became 0.** With `t0`, `per`,
  `rp`, `a`, `inc`, `ecc` or `w` left as None, `light_curves` used 0
  (a flat curve for `rp`, `t0 = 0` for `t0`) where `light_curve` raised.
  Every path now raises `ValueError: params.<name> is not set`.
- **Secondary eclipses** take `fp` at every call: a later `fp = None` gave a
  flat eclipse; it now raises. At `rp = 0` the eclipse was NaN everywhere
  (0/0); a point planet is now never hidden (flux 1 + fp, finite
  gradients).
- **Setting `use_metal` after the first evaluation** now takes effect for
  every law, orbit and planet shape; it had reached only the quadratic
  law's fused model kernel.
- **`light_curves` with the contact rule in float32** fell back to the MLX
  graph for the quadratic law instead of the fused kernel: now 3.8x
  faster (256 sets x 2,000 points: 265 -> 70 ms); results agree to float32
  rounding.
- **Exposure options:** `supersample_factor > 1` with `integration="contact"`
  (silently ignored before) and `n_gl < 1` now raise when the model is
  built.
- **Prior inverses at valid corners:** `u_to_q_np(0, 0)` and
  `hybrid2_to_q_np` at the uniform disc, and `q_from_simplex_np` at a
  simplex vertex, returned NaN; an undetermined coordinate is now 0.5.

### Documentation
- Corrections from a review of the docs: the eccentric anvil model's
  parameter order (`secosw`, `sesinw` come before `df0`; `PARAM_NAMES`,
  `PARAM_NAMES_ECC` and both target classes are now in the API
  reference); current kernel timings, re-measured (19.5 / 55.0 ms, about
  570x reverse mode; `benchmarks/bench_vjp.py` now records them in
  `benchmarks/vjp.json`); anvil's default chunk size (16,384); the gradient-memory
  arithmetic; the white-dwarf example (a 16%-deep grazing eclipse) and the
  total-eclipse exposure figures; what runs without a usable Metal device;
  the eccentric float64 accuracy (4e-16).
- Tutorial 2 says that anvil-gp is not yet public. The citation page shows
  the forthcoming papers without placeholder links.
- The documentation's build dependencies are pinned in one place,
  `docs/requirements.txt` (the `docs` extra is gone), with every indirect
  dependency pinned through `docs/constraints.txt`.
- A second review: the compiled-likelihood timing re-measured (1.5x, not
  2x); the no-Metal fallback's slowdown (about 25x forward, 11x value and
  gradient; results equal to float32 rounding); the total-eclipse exposure
  case described as measured (b = 3, 14.8 minutes); tutorial 2 credits the
  cross-sampler check to the full-size fit; the float32 long-baseline
  error is stated against the span of the data (the frontend re-centres
  times), and re-measured that way; all three tutorials re-executed.
- `benchmarks/bench_vjp.py` (moved from `examples/`) records its timings,
  with the eccentric kernel and the eager-versus-compiled likelihood, in
  `benchmarks/vjp.json`; `make_report.py` reads them, and now generates
  every section of `benchmarks/RESULTS.md`.
  `python benchmarks/verify_doc_claims.py --check` fails if the docs quote
  a number that differs from the stored measurements.

## [0.16.1] — 2026-10-13

### Added
- **On PyPI: `pip install elome`.** Releases are published by a GitHub
  Actions workflow (`.github/workflows/publish.yml`) through PyPI trusted
  publishing when a GitHub release is published.

### Changed
- Packaging metadata uses an SPDX license expression (`license = "MIT"`,
  `license-files`), and the README's images and links are absolute so that
  they render on PyPI.

## [0.16.0] — 2026-10-12

The first public-facing release.

### Documentation
- **A documentation site** (Sphinx, for Read the Docs): getting started,
  separate paths for spherical and oblate planets, guides shared by both
  (exposures, batching, gradients and precision, the sampler API),
  the three tutorials rendered with their outputs, the hybrid-law
  definitions, performance, how it works, an API reference, citing and the
  name. The README is now a short landing page that points to it. The
  development plans and design notes are retired from the repository.
- anvil is public: the docs link it and give its install command, and so
  does the ImportError raised when it is missing.
- MLX's float64 `sin`, `cos`, `exp` and `erf` are only float32-accurate.
  Elómë's own results are unaffected, since it uses `elome.trig.sincos`,
  but a caller's MLX preprocessing can carry the error into a float64
  light curve; the docs say so.

### Removed
- The deprecated import alias added in 0.14.0.
- `import_engine` no longer looks for the sampling engine under a former
  package name.

## [0.15.0] — 2026-10-11

### Changed
- **hybrid4's poles move to the tuned ladder** of the hybrid-law
  coefficients study: the analytic ladder with alpha_eff = 25/16, eps =
  0.0286, 0.3858. Over 8,100 model-passband combinations outside the pole
  study the median light-curve error is 1.8 ppm (Claret four-parameter
  2.0, hybrid5 1.1), as good as the free two-pole optimum. hybrid4 light
  curves change, and weights fitted before this release do not carry
  over. hybrid2 and hybrid5 are unchanged. New constants `HYBRID4_ALPHA`,
  `HYBRID4_EPS` (also exported from `elome`).

### Added
- The name in the docs: the README title also shows Elómë in Tengwar
  (Quenya mode; SVG outlines in light and dark versions, rendered with the
  OFL font Alcarin Tengwar -- `docs/assets/`, with the script that made
  them), followed by the etymology and the pronunciation, /ɛˈloː.mɛ/,
  "eh-LOH-meh"; the package docstring carries both too.

## [0.14.0] — 2026-10-11

### Changed
- **The package is Elómë** (Quenya: él "star" + lómë "dimness"):
  `import elome`, distribution `elome`. Results are unchanged (outputs and
  kernel sources bitwise v0.13.2). A deprecated import alias kept older
  installations working until it was removed.

## [0.13.2] — 2026-10-11

A production-readiness review of the package, and the oblate tutorial.

### Fixed
- **NaN gradients at e = 0 in fp32 eccentric anvil targets.**
  `ecc_constraint_penalty` re-derived e sin w through a masked branch whose
  backward pass underflowed (0 * inf); d/d(secosw, sesinw) was NaN at
  e ~ 0 -- where chains start -- in every `make_transit_target(eccentric=
  True)` / `make_ecc_target` likelihood. It now uses `_ecc_orbit_from_v`'s
  projected h (the same value, a safe denominator).
- **Unvalidated geometry.** `TransitModel` raises for rp < 0, a <= 1 or
  per <= 0 (or non-finite) -- a negative rp returned a finite, deeper,
  wrong transit -- in `light_curve`, `light_curve_mx` (readable arrays)
  and `light_curves`; a traced out-of-range field makes the output NaN.
- **`light_curves` with array-valued fields** now checks the coefficient
  count of the uniform, linear and quadratic laws (3 quadratic
  coefficients were silently truncated; 1 gave an opaque IndexError).
- **`use_metal=False` used the autodiff core** for the quadratic law,
  whose gradients are NaN at r = 0 and at r = 1, z = 0; it now uses the
  analytic core, as the default route's fallback does. anvil's eccentric
  graph model likewise.
- **`PenalizedLogLike.hi`** raised "float64 is not supported on the GPU"
  outside the CPU stream (eccentric targets only).
- **`make_transit_target(core=...)`** accepted different names for
  circular and eccentric models; both now take "metal", "analytic"
  (alias "graph") and "autodiff".
- **`import_engine`** turned any ImportError from an installed `anvil`
  into a misleading "No module named ..." error; it now reports the
  real error.
- **`metal_available`** failing was silent (every fp32 call then ran the
  ~10x slower graph); it now warns once with the exception, and
  `metal_available(reprobe=True)` probes again.
- **`contact_offsets_oblate`** (now exported at the top level) follows the
  package's input conventions: Python numbers, numpy and mx arrays,
  fp64 on the CPU stream, cos i derived when omitted.

### Changed
- `TransitModel`'s spherical and oblate paths share one parameter
  preparation and one eccentricity routing; cos i from b at conjunction
  has one copy (`anchored.cos_inc_from_b`). Spherical outputs and kernel
  sources are bitwise v0.13.0.
- Docs review of the above: out-of-domain numpy inputs raise (only
  mx.array inputs give NaN, and neither is -inf); on a central transit
  theta and pi - theta give identical light curves, so wrapping theta to
  [0, pi) still leaves two modes; notebook 03's eccentric example holds b
  at conjunction fixed (it held inc, which moved b), its f -> 0 section
  states the float32 flat band, and its cost figures agree with the
  benchmark. The notebook build scripts share one writer
  (`notebooks/_nbwriter.py`), which adds nbformat cell ids; notebooks 01
  and 02 change only by those ids.

### Added
- `notebooks/03_spherical_and_oblate_planets.ipynb` (generated by
  `notebooks/build_03.py`): spherical and oblate planets side by side --
  the same steps for each (light curves, hybrid limb darkening, exposures,
  gradients, batching, `flux_dev_from_tau` and `ld_basis`), then the
  oblate geometry, the oblateness signal against an equal-area sphere,
  per-side contacts, eccentric orbits, cost and limits. Needs only
  elome; runs in about a minute.
- The sampler guide: "Sampling an oblate planet".
- `contact_offsets_oblate` exported at the top level (it was public in
  `elome.oblate_tau`); notebook 03 uses only public API.

## [0.13.1] — 2026-10-11

Code-review round on 0.13.0 (oblate frontend).

### Fixed
- **d/d(ecc) at e = 0 for an oblate `TransitModel`.** An array `ecc` went
  through (k, h) = sqrt(e) (cos w, sin w), so `light_curve_mx`'s gradient
  in ecc at e = 0 was wrong (-0.06 against the true one-sided +1.02 in
  the reported case). It now enters as (e, w), as the spherical graph
  does; the oblate kernels gained an (e, w) variant that keeps all nine
  eccentric gradient slots (the (k, h) kernels drop seven on a circular
  chain, exact only there).
- **Out-of-range array eccentricities on an oblate model** (`mx.array(1.5)`,
  `mx.array(-0.2)`) returned a plausible curve; they now raise when
  readable and give NaN under `mx.compile`, as for a spherical model.
- **`use_metal=False` was ignored for oblate models**: an fp32 model still
  ran the Metal kernel. It now runs the graph (uncompiled on the GPU,
  where the fp32 oblate graph exceeds MLX's fused-kernel argument limit).
- **The fp32 oblate kernel route solved the contacts on every call**, also
  for rules that never read them: 1.94 -> 0.70 ms at 4 x 500 points with
  no exposure.
- `oblate.axes` raised for a numpy array of f.

### Changed
- The impact parameter at conjunction on the oblate route comes from
  `exposure.contact_geometry` (one copy, with its 1 - e^2 floor) in both
  the single-set and batched paths; the oblate and hybrid branches of
  `flux_dev_from_tau` share one set of argument checks; one helper states
  the f / theta rule for `TransitModel`.

## [0.13.0] — 2026-10-10

Oblate planets: an elliptical planet (area-equivalent radius, projected
flattening f, sky angle theta) for the hybrid laws, on every surface --
the exact MLX graph, `flux_dev_from_tau(f=, theta=)` with fp32 Metal
kernels and analytic gradients, and `TransitParams.f` / `.theta`. Spherical
planets are untouched (f = None). Cross-code comparison:
`benchmarks/oblate_compare/`.

### Added
- **Oblate planets: the reference graph** (`elome/oblate.py`). `flux_dev_oblate(x0, y0, r, f, w, law)`
  and `shape_cols_oblate(x0, y0, r, f, law)` give the hybrid laws' transit
  of an ellipse with area-equivalent radius r and flattening f, the centre
  in the planet's principal frame (`oblate.principal_frame(X, Y, theta)`).
  An MLX graph in fp64 and fp32, differentiable in every input, under the
  usual data contract. Agrees with an independent analytic reference to
  3e-15 (fp64) and 4e-7 (fp32) of each column's unocculted flux over its
  topology classes and limb- and pole-tangency stress sets. Below
  f = 1e-10 (fp64) / 1e-5 (fp32) it returns the spherical columns. The
  domain is r <= (1 - f)^1.5. This is the reference the Metal kernels
  are validated against.
- **Oblate planets on an orbit.** `flux_dev_from_tau(...,
  limb_dark=<hybrid law>, u=w, f=f, theta=theta)`, with f and theta
  scalars or per chain, circular or eccentric orbits, every exposure rule,
  and the `ld_basis=True` form (light curves agree with an independent
  reference to 5e-16). The contact rule splits at the oblate planet's own contacts,
  solved per side (a tilted planet's ingress and egress differ, and a
  grazing one can have both outer contacts on one side of conjunction):
  exact to 2e-15 rad in fp64, and on 30-minute exposures 4-40x more
  accurate than splitting at the spherical contacts. With f = 0 the result
  equals the spherical path's to 2e-17. Without f and theta, nothing on
  the spherical paths changes (bitwise). Out-of-domain array arguments
  give NaN rather than raising, so the call stays compilable.
- **Oblate planets: fp32 Metal kernels** (`elome/metal_oblate.py`). fp32 data on the GPU now takes fused kernels for every
  exposure rule, circular and eccentric orbits, the scalar and `ld_basis`
  forms, forward and VJP. Within 2e-7 of the fp64 graph in flux; gradients
  in every input (tau, period, a, b, r, f, theta, the weights,
  secosw/sesinw) within 1e-3 of fp64 autodiff, typically 1e-5. The VJP is
  the derivative of exactly the forward's function: one templated
  evaluation, instantiated on float and on a dual number carrying
  d/d(centre, axes). The quartic roots come from the same hand-written
  QR eigensolver as the graph (once per point, the other levels
  warm-started from it); contacts are solved one thread per chain. Below
  f = 1e-5 a chain takes the spherical forms, with d/df = 0 there.
  Cost on the GPU: 9-25x the spherical hybrid kernels with one evaluation
  per point, ~60-75x under the contact rule (the spherical hybrid kernel
  is ~0.2 ns per node there; the oblate one ~14).

- **Oblate planets in the frontend: `TransitParams.f` and `.theta`.** Optional,
  None by default (spherical, bitwise unchanged); `theta` in degrees like
  `inc` and `w`. An oblate model needs a hybrid law and a primary transit.
  `light_curve`, `light_curves` (per-set f and theta) and
  `light_curve_mx` (differentiable in f and theta) all route through
  `flux_dev_from_tau(f=, theta=)` -- the oblate kernels on an fp32 GPU
  model -- with the model's own exposure rule. Light curves agree with an
  independent reference to 8e-16 in absolute times;
  f = 0 equals the spherical model to 2e-15. Switching a model between
  spherical and oblate raises, as a changed law does.
- `benchmarks/bench_oblate.py`: oblate vs spherical, per law, forward and
  value+grad, subprocess-isolated.
- `benchmarks/oblate_compare/`: a cross-code benchmark against
  squishyplanet, JoJo and GreenLantern -- precision against a 30-digit
  integral, single-curve wall time to 10^7 points, a 512-set batch and
  value + gradient -- with the macOS OpenCL 1.2 build notes for
  GreenLantern. At 10^7 points the fp32 kernel is 43x faster than
  GreenLantern, 128x than JoJo and 795x than squishyplanet; the fp64 codes
  all reach rounding.

### Fixed
- **Hybrid pole columns just past the inner contact, large planets.** The
  partial-regime pole form mixed the exact lens term (kite) with
  1 - (z + r)^2 in its product form, which near the inner contact carries
  the rounding of 1 - z - r; kap1 / (p eps) amplified the mismatch. At
  r = 0.8, eps = 0.0016, 1e-13 past the contact the fp64 column was off by
  1.6e-7 (8e-11 of its norm; growing ~1/sqrt(distance)), the fp32 kernel
  by 1.5e-4 of its norm, and the fp32 graph's d/dz by 2.5e-3 at r = 0.95.
  The smaller of 1 - (z - r)^2 and (z + r)^2 - 1 is now always derived
  from kite^2 = their product. Now 1.4e-15 (fp64) and 8e-7 (fp32) of the
  norm there; r <= 0.5 was already at rounding. Found in a cross-check
  against a transit research tool. Hybrid outputs move at
  rounding elsewhere; the quadratic paths are bitwise unchanged.

### Changed
- The oblate graph's quartic roots are seeded by Elómë's own 4x4
  complex eigensolver (balanced, Wilkinson-shifted Hessenberg QR, written
  in elementwise operations) instead of `mx.linalg.eigvals`, which runs
  only on the CPU in complex64 and has no Metal counterpart. Same accuracy
  on every stress set; the fp32 graph no longer leaves the GPU; and it is
  the algorithm the oblate kernels carry.
- The oblate graph's arc choice no longer rests on a point next to a limb
  crossing (a graze's sliver made that test a coin toss in fp32), and the
  fully-inside regime sums each pole's two inner residues as one quadratic
  factor, which keeps fp32 derivatives when those roots are tiny (a nearly
  centred, nearly round planet). Values unchanged at fp64 rounding.

## [0.12.2] — 2026-10-10

Code-review round on 0.12.1.

### Fixed
- **A registered law whose name is not a Metal identifier aborted the
  interpreter** (e.g. `hybrid.LAWS["my-law"]`): 0.12.1 sanitised kernel
  names only for unregistered laws. Every kernel-name fragment is now a
  valid identifier -- a registered name that already is one (hybrid2/4/5)
  is kept, so their kernels are unchanged; anything else is sanitised and
  given a digest of the definition.
- **A law name and its equal law object counted as different laws**:
  a model built with `"hybrid2"` raised "law changed" when called with
  `elome.HYBRID2`, and the reverse. `_check_law` compares the
  resolved laws' definitions.
- **A model re-read the registry on every new graph trace**, so re-binding
  or deleting a registered name after construction could make one model
  trace two laws, or fail. `TransitModel` resolves its hybrid law once, at
  construction; a name the registry has since re-bound to another
  definition is "law changed", and the model's own law keeps working.

### Changed
- `hybrid.find_law` is the one law resolver (a HybridLaw as is, a
  registry name read at call time, or None); `get_law` raises on None and
  the frontend uses it directly -- its private copy is gone.

### Tests
- The fp64 custom-law `TransitModel` test checks against `flux_dev_hybrid`
  on the orbit's separations (it compared a model with an identical one).
- The renamed-law kernel-sharing test counts the tau cores as well as the
  z kernels, and its always-true clause is gone.
- New: a name and its equal law object, both ways; a model keeping its law
  through registry re-binding and deletion (including a graph traced only
  afterwards); a registered non-identifier name, in a subprocess. All three
  fail on 0.12.1.

## [0.12.1] — 2026-10-10

Code-review round on 0.12.0.

### Fixed
- **`TransitModel` refused a `HybridLaw` object**, though the README and
  the registry promised one is accepted wherever a law name is (only
  `flux_dev_from_tau` and `flux_dev_metal` took it). The frontend now
  resolves `limb_dark` through one helper -- a `HybridLaw`, or a name in
  the registry read at call time, so a law registered after import is
  known to every entry point alike -- on `light_curve`, `light_curves`
  and `light_curve_mx`, fp64 and fp32. Equal law objects are the same law
  (`_check_law` compares by value).
- **The kernel cache identity included the law's name**, so a registered
  law re-made under another name compiled duplicate kernels. It is now
  `HybridLaw.definition` (poles and shapes): a renamed copy reuses the
  registered kernels, bitwise.

### Changed
- `HybridLaw` normalises its fields on construction (lists, numpy scalars
  -> tuples of floats), so it hashes and compares by value; `definition`
  and `is_hybrid2_type` are properties -- the one rule for cache identity
  and for which prior a law gets (`ld`'s `law=` check and the tests' weight
  sampler both read it). The cache-key helper is gone.
- `hybrid2_vertices` is computed once per pole (cached; fresh arrays
  returned), so a sampler's log-prior no longer repeats host work every
  step.
- README: kernels are kept per definition; each new one is one compile.

### Tests
- `TransitModel` with a hybrid2-type and a two-pole custom law on every
  path and precision; equal law objects; a law registered after import;
  a renamed registered law sharing its kernels; vertices returned fresh.
- A cross-check against a transit research tool fails if either
  package moves hybrid2's pole; the prior cross-check compares both
  packages' default priors.

## [0.12.0] — 2026-10-10

### Changed
- **hybrid2's pole is eps = 0.36** (`HYBRID2_EPS`), the hybrid-law
  coefficients study's tuning over 198 PHOENIX v3.0 and MPS-ATLAS models x
  37 passbands: the 90th/99th-percentile light-curve errors are minimised at
  0.35-0.36, where hybrid2 beats quadratic in 73% of model-bands. hybrid2
  light curves and its prior triangle (vertices (-0.1246, 1.1246) and
  (1.2010, -0.2010)) follow; hybrid4/5 are unchanged.

### Fixed
- **Compiled hybrid kernels were cached by law name alone**, so a
  user-built `HybridLaw` sharing a registered name but not its poles or
  shapes silently ran the registered law's kernel (pre-existing since
  0.10.0). The four kernel caches now key on the whole definition, and a
  non-registered law's kernels get a sanitised name plus a digest -- any
  law name works, where one that was not a valid Metal identifier aborted
  the process. Registered laws keep their kernel names and sources.

### Added
- `hybrid2_vertices`, `hybrid2_from_q`, `hybrid2_from_q_np`,
  `hybrid2_to_q_np` take `law=` (a name or a hybrid2-type `HybridLaw`), so
  a prior is tied to the law it serves, or `eps=`; `law=` refuses a non-hybrid2-type law and a conflicting `eps=`.
- `HYBRID2_EPS` is exported; the supported limb-darkening names derive
  from the hybrid law registry (`hybrid.LAWS`).

### Tests
- One weight sampler for every hybrid test file (`tests/hybrid_weights.py`),
  choosing the physical region by the law's structure, not its name; the
  four drifting copies are gone. New: a same-named law with another pole
  gets its own kernel (z and tau paths) and a non-identifier law name
  compiles -- both fail on the old keying; the prior's `law=` path at a
  non-default pole (constraints, monotone profile, round trip, MLX =
  numpy) and its argument checks.

## [0.11.0] — 2026-10-10

### Added
- **Occultors larger than the star: `rp > 1`** (white-dwarf hosts; WD
  1856+534 b has Rp/R* = 7.28). Every entry point and law, fp64 and fp32,
  with gradients. A fourth regime, **total occultation** (z <= rp - 1):
  F = 0 exactly and every gradient is zero, on the graph and in every
  kernel family. Measured against a direct integral of the occulted
  intensity (`tests/test_large_occultor.py`, 101 tests):

  | rp | fp64 graph (quadratic / polynomial / hybrid) | fp32 kernels (partial overlap) |
  |---|---|---|
  | 1.5 | 2e-16 | 8e-8 |
  | 7.28 | 6e-15 / 7e-14 / 1e-14 | 3-9e-6 |
  | 20 | 8e-14 / 2e-11 / 1e-13 | 1e-4 |
  | 50 | 3e-12 / 1e-9 / 4e-12 | 2e-3 |

  The fp32 partial-overlap error grows ~ rp^2 (terms of size ~rp cancel
  to order 1): use fp64 above rp ~ 10. README section "rp > 1"; anvil's
  `make_transit_target` docs give the bounds to widen.

### Fixed
- **The far-side push** (`orbit.separation_circular`, `api._photom`
  primary and secondary) moved points behind the star to 2 + z, which
  clears it only for rp <= 1: an occultor larger than the star with
  b < rp - 1 got a full eclipse at the far conjunction (fp64 graph; the
  kernels test the far side explicitly and were right). Now z + 2a and
  z + 2 (1 + rp).
- **No total-occultation branch in the fp32 kernels.** The quadratic
  core returned NaN at z = 0, errors up to 17 for 0 < z < rp - 1, and
  inf at rp = 1.001 -- in the z, tau, model and basis kernels alike. The
  core now assigns the full-disk values (s0d, s1d, s2d) = (-pi, -2 pi/3,
  0) and zero partials on those lanes, after the general path; the
  hybrid kernel zeroes the lens (kite = 0) so its general formulas land
  there exactly, branch-free.
- **fp32 hybrid laws returned NaN at the internal contact** (pre-existing
  since 0.10.0, found by this release's r < 1 gate): where z + r rounds
  to exactly 1, the lens depth -Bp was -0.0, the pole's ratio flipped to
  -inf and the lane took sqrt(-Q) -- NaN in every column (generator
  mixing), graph and kernel alike, about 1 point in 10 within two ulps
  of the contact. The depth there is now kite^2 / A, which kite already
  carries.
- The contact Taylor switch (|z + r - 1| < sqrt(eps)) is guarded to
  r <= 1, where its sqrt(r (1 - r)) is defined.

### Changed
- **Total occultation is handled upstream on the graph**: the lens kite
  is zeroed on those lanes, so kap0 = 0 and kap1 = pi exactly and the
  existing formulas give the full-disk values with zero gradients
  (finite at z = 0, where two masked divisions were made 1/1). Selecting
  constants at the outputs instead changed MLX's fusion of the compiled
  polynomial frontend graph and moved r < 1 results by 1-2 ulp.
- Contact docstrings and `exposure.contact_offsets[_anchored]` name the
  inner pair |1 - r|. (The solver only ever used Z^2, so the contacts were
  always right; the deep-eclipse contact-rule error, 1e-5 at n_gl = 5,
  is ordinary quadrature resolution and falls below 1e-6 at n_gl = 9.)

### Gates
- r < 1 untouched: the 296-array release corpus bitwise; every fp32
  kernel family's outputs and VJPs (146 arrays) and the hybrid graph
  (120 arrays) bitwise except the internal-contact NaNs fixed above. The
  kernel *sources* change by design (the 18-source byte gate retired for
  this release, replaced by the output gates).
- `tests/test_entry_contract.py` gains r > 1 rows for every kernel
  family and two graph functions; its radius-varying tests now scale
  from each entry's own radius and its fp32 tolerances with it.

## [0.10.7] — 2026-10-08

Code-review round on 0.10.6, plus a sweep of every public entry point
under every calling form, which the review rounds had been finding one
form at a time.

### Corrected
- **0.10.6 overclaimed the fp64-parameter fix.** It fixed the forward
  pass only. `mx.grad` with respect to an fp64 parameter still raises
  from the GPU default stream, and no library change can fix that: MLX's
  transform machinery runs float64 ops on the default stream even when
  the differentiated function runs wholly on the CPU stream (verified;
  a cast with its own CPU-stream backward pass, the review's proposed
  root fix, raises too). The supported form -- the `mx.grad` call under
  `mx.stream(mx.cpu)` -- is now stated in `elome.dtypes`, the entry
  points' docstrings and the README, and pinned by a test.

### Fixed
- **Calling with the data by keyword raised** (`flux_dev_metal(z=...)`,
  `flux_dev_metal_hybrid(z=...)`): 0.10.6's decorator took the data
  positionally only. It now accepts the first parameter by its own name;
  a missing one is a clear TypeError.
- **The quadratic z entry's graph route skipped the parameter cast**:
  an fp64 r with fp32 z raised on the GPU (3-D z, or no Metal) and gave
  an fp64 result on the CPU stream, against the rule. Parameters are cast
  before the route is chosen.
- **float16 / integer data**: numpy float16 was fp64 but an mx float16
  or int array raised. One rule for every container: float32 stays
  float32, anything else is float64 (cast on the CPU stream).
- **The public graph functions** -- `flux_dev`, `light_curve`,
  `flux_dev_poly`, `sn_dev_poly`, `flux_dev_hybrid`, `shape_cols`,
  `sn_dev`, `sn_dev_with_aux`, `flux_dev_analytic`, `ld_basis_analytic`
  -- raised opaque errors ("'list' object has no attribute 'dtype'",
  "unsupported dtype for cel") on numpy or list data, and "float64 is not
  supported on the GPU" on fp64 data or parameters on the default stream.
  They now follow the same data rule and go to the CPU stream for fp64
  themselves, keeping MLX broadcasting and promotion for parameters. Only
  calls that raised before change: everything that worked runs as it did
  (corpus bitwise).
- **`flux_dev_analytic`'s backward pass crashed with Python-float
  coefficients** (`.shape` on a float) -- reached by `flux_dev_metal`'s
  fp64 route under `mx.grad`. Pre-existing; the VJPs take a float
  primal's shape as ().
- `combine_cols` checks the column count against the law (a quadratic
  basis with hybrid2 weights silently gave a wrong answer).
- The `tau` docstring 0.10.6 split mid-sentence is whole again; dead
  `.astype` calls after `_canon_param` removed; a non-fp64 dtype mismatch
  is cast on the active stream, the CPU round trip kept for fp64 only.

### Changed
- The contract lives in one module, `elome/dtypes.py`
  (`as_data`, `fp64_on_cpu`, still re-exported from `metal`), shared by
  the kernel entry points and the graph functions.

### Tests
- New `tests/test_entry_contract.py`: 20 entry points (12 kernel forms,
  8 graph functions) x call forms, eight data containers, parameter
  dtypes on both streams, (m,) and (n, m) layouts, `mx.grad` in fp32 and
  fp64, `mx.compile`, with spies asserting the kernel ran; plus
  `TransitModel` across six `t` containers x three dtypes and keyword
  calls. 375 cases. 1571 green (1 slow green); corpus 296/296 bitwise;
  18 kernel sources byte-identical; timings unchanged.

## [0.10.6] — 2026-10-08

Code-review round on 0.10.5. The two substantive findings were
package-wide contracts, not entry-point bugs; both are now stated and
enforced in one place each.

### Changed
- **The dtype of the data is the dtype of the computation, and non-MLX
  data keeps its precision.** `metal.as_data` converts a non-mx.array
  z or tau honouring its dtype: float32 numpy stays fp32, float64 numpy
  and Python sequences are fp64. Before, `mx.array()` took a float64
  numpy array to fp32 silently on the hybrid z and both tau entry
  points (kernel-precision numbers for a caller who believed they had
  the fp64 graph -- the trap in the project's own numerics notes), while
  the quadratic z entry raised MLX's "unsupported dtype for cel". A
  numpy float64 caller of a kernel entry point now gets the exact CPU
  graph; pass `mx.array(x, dtype=mx.float32)` for the kernel. The
  README states the rule.
- **An fp64 parameter alongside fp32 data is cast, not an exception.**
  MLX refuses any fp64 array on the GPU, so an fp64 radius, coefficient
  or weight vector with fp32 z or tau raised in every kernel entry
  point. `_canon_param` casts an off-dtype mx.array parameter to the
  data's dtype first, on the CPU stream; `_canon_w` did so already but
  after a GPU broadcast (so it raised before it got there) and now casts
  first. One place each, all four entry points fixed.
- **One fp64 re-route.** `metal.fp64_on_cpu` decorates the four kernel
  entry points (`flux_dev_metal`, `_flux_dev_from_tau_impl`,
  `flux_dev_from_tau_hybrid`, `flux_dev_metal_hybrid`): converts the
  data by `as_data` and runs the call on the CPU stream when it is fp64
  on the GPU stream. The four hand-copied re-routes (each keyed on its
  own argument) are gone.
- `hybrid.combine_cols` takes the stacked (..., 1 + n_w) columns as well
  as the list, so the tau graph (and the test) pass `shape_cols`' /
  `ld_basis`' output straight in. Two dead function-local `.hybrid`
  imports in metal_hybrid.py removed (the module-level import made the
  cycle guard moot). `tests/test_api.py`'s parity test skips unless
  Metal *and* the GPU stream are there, since it asserts the kernel's
  cache key.

### Tests
- New: fp64 parameters with fp32 data on all five z/tau entry forms
  (kernel runs, fp32 out, within an ulp of the all-fp32 call); numpy
  float64, numpy float32 and Python-list data on the same five (bitwise
  the explicit-dtype mx.array call). 1196 green; corpus 296/296 bitwise;
  18 kernel sources byte-identical.

## [0.10.5] — 2026-10-08

Code-review round on 0.10.4.

### Fixed
- **fp64 on the default stream fell through to an exception in every
  z-input entry point, quadratic included.** `flux_dev_metal` (quadratic,
  `ld_basis`, hybrid) and `flux_dev_metal_hybrid` promised to fall back
  to the graph for fp64, but MLX has no fp64 on Metal at all, so unless
  the caller already sat in `mx.stream(mx.cpu)` the fallback raised
  "float64 is not supported on the GPU". Pre-existing (0.10.3 raised the
  same way; the tau entry points have re-routed fp64 since 0.8.2). Both
  z entry points now put fp64 on the CPU stream themselves, as the tau
  ones do. (0.10.4's bitwise fallback test passed only because it
  supplied that stream.)
- **One off-kernel expression, everywhere.** `hybrid.combine_cols` is
  the sequential contraction `flux_dev_hybrid` always used, now called
  by the z-kernel entry point's fallback *and* the tau graph, whose own
  dot-product `_combine` (2.4e-7 from it in fp32 on the CPU stream) is
  deleted. The fp64 tau graph moves by at most 7e-18 (hybrid-only; no
  corpus array covers it).
- A scalar radius with one set of weights given as a (1, n_w) row on
  (m,) separations went down `_hybrid_dev`'s batched branch as m
  single-point kernel rows (right answer, m-row launch); it is one
  (1, m) row now, as the (n_w,) form always was.
- `tests/test_api.py`'s repaired parity test is Metal-gated (it asserts
  the kernel's cache key).

### Tests
- New: fp64 on the default stream for all five z-entry forms; the tau
  graph's scalar call equals `combine_cols` over its own `ld_basis`
  columns bitwise (fp64 and fp32 CPU stream, every law); the one-row
  launch shape via a spy on the kernel core. 1176 green; corpus 296/296
  bitwise; 18 kernel sources byte-identical.

## [0.10.4] — 2026-10-08

Code-review round on 0.10.3.

### Fixed
- **A quadratic parity test compared the fused model kernel with
  itself.** `tests/test_api.py::TestEccentricKernelRouting` built its
  "graph" model by patching `_kernel_usable` around *construction*, but
  the check runs when the graph is first compiled, on the first call --
  after the patch was restored. Both models ran the kernel, so
  `test_kernel_matches_graph_and_fp64` had been vacuous since it was
  written. The graph model is now `use_metal=False`, and the test pins
  both cache keys. With a real graph on the other side it passes: kernel
  vs graph 2.4-3.0e-7 at e = 1e-4..0.9 (20001 points); each 2.0-3.1e-7
  from fp64. No kernel regression was hiding behind it.
- **`flux_dev_metal_hybrid`'s own fallback is `flux_dev_hybrid`.** Off
  the kernel (fp64, the CPU stream, a Metal-less machine) the public
  entry point contracted the columns with a dot product, a different fp32
  expression from hybrid.py's graph (2.4e-7 apart) -- the inconsistency
  0.10.3 fixed for `TransitModel` only. Every caller now gets one
  expression, bitwise; the frontend guard is an optimisation, not a
  correctness patch. (`_combine` stays for the fp64 tau graph.)
- A (1, n_w) weight row with several rows of z raised on `_hybrid_dev`'s
  kernel route (reshaped by the z row count) while the graph broadcast
  it; the row count now comes from the weights and the entry point
  broadcasts.
- `_hybrid_dev`'s docstring states the radius contract the code relies
  on: a scalar or anything of size n (`light_curves`' contact path hands
  it (n, 1, 1)), not the narrower list 0.10.3 gave.
- The float-radius test's attribution: the unguarded reshape was
  0.10.1's (0.10.2 added the guard), and the test guards the direct-call
  contract, since every frontend route casts rp to an array first.

### Tests
- New: the entry point's fallback equals `flux_dev_hybrid` bitwise (fp64
  and fp32 on the CPU stream, every law); a (1, n_w) row on the kernel
  route. 1163 green; corpus 296/296 bitwise; 18 kernel sources
  byte-identical.

## [0.10.3] — 2026-10-08

Code-review round on 0.10.2.

### Changed
- **`_hybrid_dev` falls back to hybrid.py's graph, not the entry point's
  own.** The guard is now `_kernel_usable()` whole -- stream check
  included, taken inside the trace -- so an fp32 model with Metal on,
  run on the CPU stream, evaluates the same graph as `use_metal=False`
  and a Metal-less machine, bitwise (0.10.2 let the guard pass and ran
  `flux_dev_metal_hybrid`'s internal fallback: a different fp32
  expression, 2.4e-7 apart). One predicate for both kernels; the
  duplicated static checks and `_kernel_usable`'s local import are gone.
- The radius goes to the kernel entry point as is: its canonicaliser
  takes a float, 0-d, (n,) or (n, 1) (checked), so 0.10.2's two
  `isinstance` reshapes -- where the float-radius bug lived -- are
  deleted rather than patched.
- The per-stream retrace that the design rests on is verified on the
  dependency floor too: MLX 0.30.0 and 0.32.2 both retrace a graph
  first traced on the CPU stream when it runs on the GPU, and three
  streams (two GPU, one CPU) give three traces, so "one trace per
  stream" is the right statement.

### Tests
- `test_fp32_light_curve_mx_gradients_through_the_kernel` asserts the
  kernel route again (via the spy; 0.10.2 dropped the key check without
  a replacement). The float-radius test is Metal-gated and spies the
  route, so it cannot pass vacuously. New: the CPU-stream fallback is
  the `use_metal=False` graph bitwise, both sides on the CPU stream
  (the same graph rounds differently on the two devices). The
  CPU-stream-first test sees the GPU trace alone now (the CPU trace
  never calls the entry point). 1156 green; corpus 296/296 bitwise;
  18 kernel sources byte-identical; 2e6-point fp32 timings unchanged.

## [0.10.2] — 2026-10-08

Code-review round on 0.10.1. Its headline finding -- that the quadratic
law's contact-rule and (e, w) graphs carry the v0.9.2 stream-pinning trap
-- turned out to be false, and showed 0.10.1's own guard against it to be
unnecessary.

### Changed
- **The hybrid z-kernel decision moved into the trace; the 0.10.1 cache
  key is gone.** A z-input kernel entry point (`flux_dev_metal`,
  `flux_dev_metal_hybrid`) decides on the active stream *inside* the
  compiled graph, and `mx.compile` keeps one trace per stream (verified
  on MLX 0.32.2: a function traced on the CPU stream is retraced, kernel
  and all, when first called on the GPU). So no graph reached through
  `_photom` can be pinned to the graph photometry, quadratic or hybrid,
  and the 0.9.2 key is needed only for the fused *model* kernel, chosen
  in Python before tracing. `_hybrid_dev` now settles only what cannot
  vary per call (dtype, `use_metal`, a Metal device) and calls the entry
  point, which falls back on its own; the `hyb_kernel` argument, the
  per-graph `photom` closure, the `("ew", True)` key and the duplicated
  device check are removed. Cache keys are 0.10.0's again. Same speed:
  2e6 points fp32 GPU, hybrid2/4/5 1.3/1.1/1.5 ms vs quadratic 1.6 ms.

### Fixed
- `_hybrid_dev` reshaped a Python-float radius in its per-set-weights
  branch (`mx.reshape` rejects floats); no frontend caller passed one,
  but the kernel accepts a scalar r, and now so does the branch.
- Three docstrings (`_kernel_usable`, `_model_eval`, `light_curve_mx`)
  still said the hybrid laws, the (e, w) route and the contact rule never
  use a fused kernel; they now distinguish the fused *model* kernel
  (quadratic, plain graphs) from the z-input kernels every other fp32
  graph takes.

### Tests
- The hybrid kernel-route tests assert the route through a spy on the
  entry point (the stream each trace saw) rather than through cache keys;
  the CPU-stream-first test now shows two traces under one key, on the
  plain and contact graphs. New: a float radius with per-set weights.
- `test_fp32_light_curves_match_the_loop` gate 2 -> 4 fp32 ulps, with the
  reason: the two sides form z from different fp32 orbit expressions
  (anchored with k = h = 0 vs the circular graph); measured 0.5-1.0 ulp
  over seeds, so 2 was a margin, not a bound. 1150 green; corpus 296/296
  bitwise; 18 kernel sources byte-identical.

## [0.10.1] — 2026-10-08

### Changed
- **`TransitModel` runs the hybrid laws' photometry in the fused fp32
  kernel.** On an fp32 GPU model, `_photom` now sends a hybrid law to
  metal_hybrid's z-input kernel -- the role `flux_dev_metal` plays for the
  quadratic law -- with the orbit still on the MLX graph. Every frontend
  path takes it: `light_curve`, `light_curves` (one kernel row per set),
  the contact and supersample rules, eccentric orbits and the (e, w)
  route of `light_curve_mx`; the contact rule's nodes still move with the
  parameters. At 2e6 points: hybrid2 4.2 -> 1.7 ms, hybrid4 5.2 -> 1.5 ms,
  hybrid5 7.4 -> 2.3 ms, against the quadratic law's fused model kernel at
  1.7 ms. Chosen over a new fused *model* kernel per law (an estimated
  further ~0.5 ms per 2e6-point call) and over routing through the tau
  kernels (measured 1.4-2.4 ms): one call site, no new Metal code.

  The kernel decision was made at graph build time from the active stream
  and put in the compiled-graph cache key (`("ew", True)` for the (e, w)
  route) -- needlessly, as 0.10.2 found; the key is gone again there. A
  CPU-stream first call cannot pin a later GPU call to the graph. Agrees
  with the fp32
  graph to 1e-6 and the fp64 model to 2e-6; `light_curve_mx` gradients
  through the kernel's VJP match fp64 to 2e-3 of their scale. Quadratic
  and polynomial paths unchanged: same keys, same ops, 296-array corpus
  bitwise identical, 18 kernel sources byte-identical. 29 tests; 1148
  green.

## [0.10.0] — 2026-10-08

### Added
- **Hybrid limb-darkening laws: closed forms and the fp64
  reference** (`elome/hybrid.py`). The hybrid2, hybrid4 and hybrid5
  laws -- even powers of mu plus double-pole terms
  1/(mu^2 + eps)^2, written in a shape basis whose weights w are each the
  share of the centre-to-limb drop a term carries -- as
  `flux_dev_hybrid(z, r, w, law)`, the shape-basis columns `shape_cols`
  (the `ld_basis` contract `flux - 1 == (B @ c)/(N @ c)`, `c = (1, -w)`,
  `N = hybrid_norms`), their analytic partials `shape_partials`, and the
  uniform priors on the physical regions (`ld.hybrid2_from_q` on the
  exact triangle, `ld.simplex_from_q` by stick-breaking, with inverses).

  For a spherical planet the mathematics simplifies: every column is
  elementary. The even powers come from s_0, s_2 and the ALFM19 even
  recursion M_0 -> M_2 -> M_4 (never the elliptic M_1, M_3); a pole term's
  occulted flux is kap1/(p eps) plus an atan or log of rational functions
  of (z, r) (Green's theorem with h = 1/(2p(p - s))), with a series
  bridging the removable singularity on z + r = sqrt(1 + eps). Unlike the
  quadratic law's mu^1 term, **no column needs an elliptic integral**.
  Every sqrt(depth) dependence is routed through the core's `kite`, so
  the three cancelling terms of the pole form see one rounded lens near
  the contacts (without that, fp32 lost 1e-6 there).

  Measured: against a 30-digit direct integration, the pole columns agree
  to 1e-13 of their norm (1e-12 within 1e-6 of a contact, the geometry's
  own conditioning), the laws to 1e-13; against an independent ellipse
  code at a = b, 1e-12; the even columns match the cel3 route to 1e-14.
  Autodiff matches finite differences to 1e-7 in z, r and every weight;
  the analytic partials match autodiff to 1e-11 (except at MLX's max/min
  ties z == r and z == 1, where autodiff splits the gradient and the
  analytic partials are the ones that match FD). fp32 stays within 1e-6
  of fp64 at every simplex vertex. 90 tests (`tests/test_hybrid.py`).
  Existing paths bitwise unchanged (296 arrays).
- **Hybrid limb-darkening laws in `TransitModel`.**
  `limb_dark="hybrid2" | "hybrid4" | "hybrid5"` with `params.u` the shape
  weights (2, 4 or 5 of them), on every frontend path: `light_curve`,
  `light_curves` (sequence and array-valued, per-set weights), the
  contact and supersample rules, eccentric orbits, and `light_curve_mx`
  differentiable in each weight (FD to 1e-6; the every-field gradient
  test now runs over the three laws). Secondary eclipses ignore limb
  darkening as before. The laws use the polynomial law's vector-law
  plumbing: `_n_poly` is `_n_vec`, the graph branches read `vec`, and the
  one law-specific line is the dispatch in `_photom`. They run on the
  MLX graph (fp32 within 2e-6 of fp64), never the fused kernel, as the
  polynomial law does. 34 tests (`tests/test_hybrid_api.py`); existing
  paths bitwise unchanged (296 arrays).
- **Hybrid limb-darkening laws: fused fp32 kernels**
  (`elome/metal_hybrid.py`). `flux_dev_from_tau(..., limb_dark=,
  u=)` and `flux_dev_metal(..., limb_dark=, u=)` for all three laws:
  circular and eccentric orbits, every exposure rule, analytic VJPs in
  tau, the period, a, b, r, every weight (shared `(n_w,)` or per chain
  `(n, n_w)`) and secosw/sesinw; `ld_basis=True` returns the
  `(n, m, 1 + n_w)` shape-basis columns with the contract
  `flux - 1 == (B @ c)/(N @ c)`, `c = (1, -w)`. The tau kernels are
  derived from metal.py's orbit-generic templates by asserted
  substitution (the quadrature, its tau derivative, the contact split and
  the simd_sum reduction stay one copy); the device functions put
  hybrid.py's closed forms in registers, per law by substituting its
  poles, shape matrix and norms into one template. The fp64 graph path
  computes the same function.

  Measured: fp32 kernel within 1.2e-8 of the fp64 graph on every law,
  rule and orbit (gates 5e-7, 2e-6 for hybrid5); the fp64 graph path
  equals `TransitModel` to 1e-12; kernel VJPs match fp64 autodiff at the
  quadratic kernel's gates; the basis VJP equals the scalar VJP through
  the identity to 1e-5. At 512 x 5,000 with the contact rule: hybrid2 is
  0.50x quadratic's forward time and 0.60x its value+grad, hybrid4
  0.62x / 0.82x, hybrid5 0.81x / 1.02x (`benchmarks/bench_hybrid.py`) --
  no elliptic integral. 183 tests (`tests/test_hybrid_metal.py`).

  hybrid.py's series bridge across the Q = 0 line now has six terms and,
  in fp32, a 0.05 half-width (1e-3 in fp64), so the closed-form partials'
  near-cancellation there never costs fp32 more than ~1.3 digits; the
  kernels use the fp32 rule. Quadratic kernels: all 18 sources
  byte-identical, outputs bitwise unchanged (296 arrays); 1119 green.

## [0.9.7] — 2026-10-04

Fixes from a seventh code review (of 0.9.6).

### Fixed
- **0.9.6 broke the array-valued `light_curves` form with a per-set `u`
  (regression).** `light_curves(params)` with array attributes takes a
  per-set `u` of shape (n_sets, N) (README), and 0.9.6 routed it through
  the single-set normaliser, which demands 1-D: it raised
  `u must be a 1-D vector; got shape (3, 2)`. On 0.9.5 it ran and matched
  the loop to 4e-16. The bitwise corpus never held that case, which is
  why it passed. That branch now normalises `u` to (n_sets, N) itself
  (`_u_vector(..., sets=True)`) and compares N -- not the number of sets,
  which 0.9.5 had counted, wrongly, for a polynomial law -- against the
  model order; a mismatched set count or a 3-D `u` is a clear error.
- **A mixed `u` list no longer borrows a sibling's dtype.** 0.9.6
  converted the Python entries of a list holding an `mx.array` scalar to
  that scalar's dtype: `[mx.array(1), 0.4]` turned 0.4 into int 0 (flux
  2e-3 off, no error), and an fp32 entry rounded its neighbours before
  the model cast. Each entry is now cast on its own, to the model dtype,
  as 0.9.5 did. A nested non-mx entry beside an mx one gets the shape
  error rather than an opaque `TypeError`.
- **`u` is normalised once per entry point.** `_check_law` returns the
  normalised vector and `light_curve`, `light_curve_mx` and the
  sequence form of `light_curves` thread it through; 0.9.6 normalised two
  or three times per call on the sampler-facing path.
- `TransitModel.__init__` derives the polynomial order through the same
  normaliser, so a wrong-shaped `u` is rejected at construction too.
- A test compared `f is m.light_curves`, which is always False (a bound
  method is a fresh object per access), so the `light_curves` branch it
  meant to cover was never exercised. It compares by name.

4 new tests, including the array-valued per-set `u` case for quadratic,
polynomial and linear laws against the looped result; 809 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.6] — 2026-10-04

Fixes from a sixth code review (of 0.9.5).

### Fixed
- **0.9.5's 2-D `u` fix covered only an `mx.array`.** A numpy (3, 1)
  column (a `loadtxt` slice, say) or a nested list `[[.4], [.25], [.05]]`
  reproduced the identical bug on 0.9.5: a (3, 601) curve of a wrong-order
  model, 8.8e-4 off, with no error, on a release whose notes said the
  case now raises. Every container -- list, tuple, numpy, `mx.array`, a
  list holding `mx.array` scalars -- now goes through one normaliser,
  `_u_vector`, which returns one 1-D vector or raises
  `u must be a 1-D vector`. That was the root cause: each container type
  had been validated by its own code path.
- **The shape is judged before the count.** `_check_law` runs first in
  both entry points and in `light_curves`, and it now normalises `u`
  before counting coefficients. So a 0-d `mx.array` `u`, which died in
  `list()` with `IndexError: SmallVector out of range`, and a (1, n) `u`,
  which got a misleading "order changed" message (a message 0.9.5's test
  had codified), both get the shape error.
- A quadratic `u` with size-1 numpy entries (a (2, 1) array) ran through
  numpy's deprecated array-to-float conversion, which a future numpy
  turns into an opaque `TypeError`; it is now rejected with the same
  shape error.
- The up-front scalar loop uses `_BATCH_KEYS` rather than restating it.
- The scalar-field test matched the first letter of a field name, so it
  could not tell `per`, `rp` and `fp` apart; it anchors the full name.

The `u` shape test now runs over all three containers, on
`light_curve_mx`, `light_curve` and `light_curves`. 6 new tests;
805 green. `light_curve`, `light_curves` and `flux_dev_from_tau` remain
bitwise identical to 0.8.2 (296 arrays).

## [0.9.5] — 2026-10-04

Fixes from a fifth code review (of 0.9.4).

### Fixed
- **A 2-D polynomial `u` ran a wrong-order model.** A whole-array `u` was
  cast without a shape check (0.9.4's guard was per entry), so a (3, 1)
  `u` passed `_check_law`'s count of three, took `flux_dev_poly`'s
  batched branch and returned a (3, 601) curve of a one-coefficient
  model, 8.8e-4 off, with no error. A whole-array `u` must now be a flat
  vector.
- **Shape validation is one pass, before any routing.** 0.9.4's notes
  said "one chokepoint"; it was three sites (`cast`, a manual check for
  t0, another for w), and `fp` and the whole-array `u` had none. All
  nine fields and `u` are now checked in one loop at the top of
  `_model_eval`, so a route that never reads a field still rejects a bad
  one and the `w`-on-a-circular-orbit special case is gone.
- **What the kernel serves is stated once.** `_kernel_usable()` now
  answers only the device question (fp32, Metal, GPU stream, switched
  on); which graphs the kernel serves (primary transits, quadratic limb
  darkening, no contact rule) is decided in `_get_compiled`, beside the
  branches it governs, and nowhere else. The three tests that pinned the
  old answer pin the cache key instead.
- A test docstring still claimed Python and array `w` give the identical
  result, which 0.9.4 retracted; it now states the 1-ulp invariant its
  body checks.

### Not changed
- `_scalar_value` treats any `ValueError` from reading a 0-d array as
  "traced". The review notes an upstream evaluation failure under
  `mx.grad` would then skip the range check and surface at the caller's
  later `mx.eval` with a less specific traceback. That error still
  surfaces; the previous message-matching guard (removed in 0.9.4 on the
  prior review's advice) traded that traceback for a dependency on MLX's
  exact wording. Left as is.

The scalar-field test now runs on fp32 as well as fp64 and includes
`fp`; 33 new tests; 799 green. `light_curve`, `light_curves` and
`flux_dev_from_tau` remain bitwise identical to 0.8.2 (296 arrays).

## [0.9.4] — 2026-10-04

Fixes from a fourth code review (of 0.9.3).

### Fixed
- **Every field of `light_curve_mx` must be a scalar, not just `ecc`.**
  0.9.3 guarded `ecc` alone: a vector `rp` (or a, inc, w, t0, per, fp, an
  entry of u) still ran, one value per time sample, and on the fp32
  fused-kernel route died with an opaque reshape error. The check is now
  at the one chokepoint every field passes through (`cast`), including
  `w` on a circular orbit, where it is otherwise unread.
- **"Scalar" means shape (), not size 1.** A (1, 1) `ecc` passed 0.9.3's
  check and changed the output's shape to (1, 601).
- **The readable-vs-traced helper no longer matches MLX's error text.**
  With the shape checked first, the traced-eval refusal is the only
  `ValueError` `float()` can raise, so a plain `except` is both simpler
  and robust to MLX rewording its message.
- **The cache key follows `_get_compiled`'s branch structure.** The
  kernel decision enters the key only where a kernel branch is reachable
  (not (e, w), not polynomial, not the contact rule), computed beside
  those branches rather than restated in `_kernel_usable`. A polynomial
  fp32 model, which 0.9.3 still compiled once per stream, now compiles
  once.
- **Docstring: an array `a` on an eccentric orbit is bitwise with
  `light_curve`** (it enters the (k, h) graph as it is); only `inc`,
  `w`, and `a` on a circular orbit combine in-graph.
- **A claim in 0.9.3's notes was too strong.** A Python `w` and an array
  `w` do not give the identical (e, w) result in general: fp32(radians(w))
  differs from fp32(w) * fp32(pi/180) for ~9% of w values (13.5, 27, ...),
  so the fixture's w = 63 agreed by luck. They agree to 1 fp32 ulp, and
  the test now asserts that over w = 63, 13.5 and 27.

24 new tests (every field x three bad shapes x both routes); 766 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.3] — 2026-10-04

Fixes from a third code review (of 0.9.2).

### Fixed
- **A vector-valued `ecc` ran unvalidated.** The readable-vs-traced test
  caught `ValueError`, and MLX raises that same type for "cannot convert
  a multi-element array", so `ecc = mx.full((601,), 0.3)` returned a
  (601,) curve with each time sample at its own eccentricity. `ecc` must
  now be a scalar (a clear `ValueError` otherwise). The readable-vs-traced
  decision lives in one named helper, `_scalar_value`, which checks the
  shape first and matches MLX's traced-eval error explicitly; a test pins
  that dependency.
- **An fp64 `ecc` just below 1 reached an fp32 graph as exactly 1.**
  The range check read the fp64 host value; the graph received the fp32
  cast, so `ecc = 1 - 1e-9` passed and ran to a flat curve. The check now
  sees the value the graph sees.
- **The kernel decision is no longer in the cache key where it cannot
  matter.** `_kernel_usable()` is False for `integration="contact"` (the
  kernel never served that path), so an fp32 contact model called on
  both streams compiles its graph once, not twice.
- **Docstring overclaim.** `light_curve_mx` matches `light_curve` bit for
  bit when the array fields are among t0, per, rp, fp and u, which enter
  the graph as they are; an array a, inc or w is combined in-graph in
  the model dtype, ~1 ulp from the host fold on fp32. The docstring now
  says exactly that.
- A Python `w` on the (e, w) route is folded to radians on the host in
  fp64 again (0.9.2 did it in-graph), matching the other Python fields.
  Note the review that flagged it attributed the (e, w) route's 2.4e-7
  fp32 difference from `light_curve` to this; it is not. The two routes
  are the same function (4e-16 apart in fp64) computed along different
  fp32 paths, each ~1.8e-7 from fp64 truth, and a Python and an array
  `w` give the identical (e, w) result. A test now pins that.

Not changed: an array `ecc` is read (and so evaluated) under `mx.grad`
to validate it, as 0.9.2 documents. 5 new tests; 741 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.2] — 2026-10-04

Fixes from two code reviews of 0.9.1.

### Fixed
- **A CPU-stream call could permanently slow `light_curve` on an fp32
  model (0.9.1 regression).** The compiled-graph cache was keyed on
  `circular` alone, while the fused-kernel decision is made at build
  time from the *active* stream. With 0.9.1's shared cache, the README's
  own recipe (an fp64 t0 gradient through `light_curve_mx` under the CPU
  stream) as a model's first call cached a kernel-less graph that every
  later GPU `light_curve` reused: 3.4 ms instead of 1.4 ms at 2e6 points.
  The kernel decision is now part of the cache key.
- **An out-of-range array `ecc` gave NaN with a zero gradient.** 0.9.1
  masked both the input and the output, so a fitter holding e as an
  array saw a NaN loss and a zero gradient, with nothing pointing at the
  parameter, and an eager call with a bad array `ecc` returned NaN where
  0.9.0 raised. Now: wherever the value can be read (eagerly, or under
  `mx.grad`, whose inputs are plain arrays) it is validated and raises
  `ValueError`. Only when traced, under `mx.vmap` or a caller's
  `mx.compile`, is raising impossible, and there the output and every
  gradient are NaN, through a NaN factor rather than a mask. A valid
  point's gradient beside an invalid one is unchanged, which the test
  now checks over a loss summed across all rows.
- **One array field changed the result on an fp32 model.** 0.9.1's
  fixed-e routes formed (k, h) and b in-graph in the model dtype, while
  `light_curve` forms them on the host in fp64 and casts once -- a 1-ulp
  difference, so adding an array `rp` moved the output at 1e-7. Python
  fields are now folded on the host exactly as `light_curve` folds them,
  and the output is `light_curve`'s bit for bit whichever other fields
  are arrays (only `ecc` as an array changes the route).

### Changed
- **One evaluation path.** `_eval_compiled`, `_eval_graph`, `_has_arrays`
  and `_uvec` are replaced by a single `_model_eval` behind both
  `light_curve` and `light_curve_mx`: a Python field is folded on the
  host, an array field stays in the graph. The routing and the
  eccentricity validation (`_check_ecc`, now scalar-aware, with one
  message) live in one place. An `mx.array` vector `u` is cast whole
  rather than sliced and restacked.

### Tests
- fp32 gradients through the fused kernel's custom VJP (Python `ecc`,
  plain mode) and through the fp32 contact / supersample graphs, against
  the fp64 model -- the routes 0.9.1 enabled but never differentiated.
- The kernel route is asserted from the cache key, not inferred from the
  absence of the (e, w) key; the cache-poisoning scenario is a regression
  test. 9 new tests; 736 green.

`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.1] — 2026-10-04

Fixes from a code review of 0.9.0.

### Fixed
- **`light_curve_mx` crashed on vector limb darkening.** It failed with a
  numpy-array `u` (the usual batman style; "truth value of an array is
  ambiguous") and with an `mx.array` vector `u`, the natural way to
  differentiate it whole. `light_curve` and 0.8.2 accepted both. The
  cause was `params.u or []`. Both forms now work, and an `mx.array`
  vector `u` is differentiable.
- **An out-of-range eccentricity could return a plausible curve.**
  - 0.9.0 checked an array-valued `ecc` with `float()` and skipped the
    check whenever that failed, so under `mx.vmap` or a caller's
    `mx.compile`, e = 1.5 returned a flat curve. Such an `ecc` is now
    checked *in the graph*: out of [0, 1) the output is NaN (it cannot
    raise when traced), and the graph sees a safe e, so a valid point's
    gradient stays clean beside an invalid one. This also removes a
    forced evaluation on every call.
  - Separately, and since long before 0.9.0, single-set
    `light_curve(params)` with a Python e >= 1 silently returned a flat
    curve, while `light_curves` raised. Both now raise `ValueError`.
- **One array field moved a fit off `light_curve`'s graph.** In 0.9.0,
  any array-valued field sent `light_curve_mx` through the (e, w)
  eccentric graph, which never uses the fused kernel. Making only `rp`
  an array on a circular fp32 model then lost the kernel. Routing now
  follows `ecc`:
  - a Python 0 takes the circular graph;
  - a Python e > 0 takes the (k, h) graph, with w in-graph (smooth
    there, since sqrt(e) is a constant);
  - only an array-valued `ecc` takes the (e, w) graph.

  At 2e6 points (fp32) the fixed-e routes run at `light_curve`'s speed
  (0.71 / 0.91 ms against 0.75 / 0.91), against 3.0 ms on the (e, w)
  route.
- Dead code: an unreachable `return self._eval(params)` calling a
  deleted method, and a no-op `consts=` in the fused-kernel branch,
  which wrongly suggested that branch serves the (e, w) route.
- `docs/frontend-circular-kernel-plan.md` still called `light_curve_mx`
  the eager path.

`light_curve`, `light_curves` and `flux_dev_from_tau` are bitwise
unchanged for every valid input (296 arrays). 15 new tests; 727 green.

## [0.9.0] — 2026-10-04

### Added
- **`TransitModel.light_curve_mx` is differentiable in every
  `TransitParams` field.** Up to 0.8.2 it read each field through
  `float()`, so a gradient taken through it was silently zero; 0.8.2
  documented that. It now runs the *compiled* model graphs that
  `light_curve` uses, with array-valued fields (t0, per, rp, a, inc, ecc,
  w, fp, `u`) kept in the graph. Python numbers stay constants, and with
  all-Python parameters the output is `light_curve`'s, bit for bit.
  Details:
  - Eccentricity enters as (e, w) through the new
    `anchored.anchor_constants_ew`. (k, h) is the right pair for a
    sampler, but it is singular as a map from (e, w): dk/de is infinite
    at e = 0, so d/d(ecc) taken through it is inf * 0 = NaN, although F
    is differentiable in e there (one-sided). With w given, the anchored
    constants need no sqrt(e) and no division. d/d(ecc) at e = 0 now
    matches a one-sided finite difference, and d/dw is exactly 0 there.
  - `anchor_constants` is now a front end on a shared core;
    `separation_anchored`, `contact_offsets_anchored` and
    `pack_orbit_constants` accept precomputed `consts=`. With the default
    (k, h) inputs every graph is unchanged: `light_curve`, `light_curves`
    and the eccentric `flux_dev_from_tau` are bitwise identical to 0.8.2
    (296 arrays).
  - The (e, w) route uses the compiled graph rather than the fused model
    kernel. The kernel skips its eccentric-only gradient slots on e == 0
    chains, which is exact for (k, h), whose Jacobian vanishes there, but
    not for (e, w).
  - Under `mx.grad`, MLX needs the CPU stream for any float64 input. That
    is MLX's rule, now documented: an fp32 GPU model differentiates fp32
    fields on the GPU, and an absolute t0, which must be fp64, is
    differentiated under `mx.stream(mx.cpu)`.

  Measured: every gradient matches fp64 finite differences to ~1e-7
  (2e-5 gate; 1e-4 for the one-sided e = 0 case under the contact rule),
  across plain, contact and supersampled modes, all four laws, primary
  and secondary eclipses, circular, e = 0.3 and e = 0.7. fp32 gradients
  are within 2e-3 of fp64. A caller can wrap `light_curve_mx` in
  `mx.compile`. Speed is 1.4-3.3x faster than the old eager path at
  100,000 points. The eager internals (`_eval`, `_contact_nodes`,
  `_separation`) are removed, and with them the misleading
  `test_light_curve_mx_gradient`, which differentiated a hand-built graph
  rather than the method. 52 tests in `tests/test_light_curve_mx.py`;
  712 tests green.

## [0.8.2] — 2026-10-04

### Fixed
- **`TransitModel.light_curve_mx` failed on a default model.**
  `TransitModel` defaults to `dtype=mx.float64`, and MLX has no float64 on
  Metal. `light_curve` and `light_curves` build on the model's CPU stream;
  `light_curve_mx` built on the default (GPU) device and raised
  `float64 is not supported on the GPU` unless the caller had already
  switched streams. It now builds on the model's stream too. The result
  is a lazy fp64 array, so the caller's own downstream ops still have to
  run on `mx.cpu`; the docstring now says so. 12 tests cover the default
  fp64 and fp32 models, circular and eccentric, plain, contact and
  supersampled, with no stream context.
- **`light_curve_mx`'s docstring claimed differentiability it does not
  have.** It reads the `TransitParams` fields as Python floats, so a
  gradient taken through it in a parameter is silently **zero**. The
  docstring now says so and points to `flux_dev_from_tau` and
  `elome.anvil`, and a test pins the behaviour so that a future fix
  must update the docs. (`TestDifferentiability.test_light_curve_mx_gradient`
  never exercised it: it differentiates a hand-built graph instead.)

## [0.8.1] — 2026-10-04

### Fixed
- **`TransitModel(integration="contact")` on eccentric orbits now splits
  each exposure at the exact contact times.** It used the linearised
  contacts (`contact_geometry` + `contact_offsets`). Those can miss a kink
  by minutes, which degrades the quadrature: at the default n_gl = 7 the
  worst orbit tested (e = 0.5, grazing-adjacent) was at 1.8e-6 against
  the exact integral, and is now at 3.1e-8. All three contact sites (the
  eager path, the compiled eccentric graph and the batched
  `light_curves`) now call `exposure.contact_offsets_anchored`, as
  `flux_dev_from_tau` does, and the two now agree to 1e-12 in contact
  mode as well. Eccentric outputs change by up to 3.1e-6. Circular output
  is bit-identical: `contact_offsets_anchored` returns the linearisation
  unchanged at e = 0, where it is exact (40 arrays: all three sites, fp32
  and fp64, quadratic and polynomial limb darkening).
- **Grazing transits gave NaN gradients through the contact times**,
  circular included. Where the clip collapses the inner pair, `sqrt` and
  `arcsin` sit at an infinite derivative, and the clip's zero cotangent
  makes 0 * inf = NaN. Only the differentiable frontend graph saw this
  (`flux_dev_from_tau` detaches its contacts, and the public `TransitModel`
  methods return plain arrays), so it never reached a user. The active
  branch now sees a sanitised argument and the collapsed value is
  detached. Values are bit-identical, and the gradients match finite
  differences.

## [0.8.0] — 2026-10-04

### Added
- **Eccentric orbits on `flux_dev_from_tau`** (`secosw=`, `sesinw=`). Both
  are omitted by default, and then nothing changes. Given, the orbit is
  the transit-anchored one (`anchored.py`) and the Kepler solve is the
  model kernel's own, lifted into a device function. So there is one copy
  of the eccentric numerics, and it is exact at e = 0 with fp32-safe
  gradients as e -> 0. `tau` is the time since inferior conjunction and
  `b` the impact parameter there (anvil's algebra), so e = 0 reduces to
  the circular call. Gradients flow in all nine inputs. `ld_basis=True`
  works too, and a batch can mix circular and eccentric chains.

  Measured:
  - The fp64 graph path agrees with `TransitModel` to <= 6e-16 for the
    instantaneous and supersampled rules, e = 0 to 0.7.
  - The fp32 kernel is within 2.5e-7 of that graph.
  - e = 0 through the eccentric path equals the circular path (1.6e-16
    fp64).
  - Cost at 512 x 5,000, contact, n_gl = 5: 2.27x forward and 2.12x
    value+grad relative to circular for e = 0.3. e = 0 chains in the
    eccentric kernel cost 1.68x / 1.65x.
  - 102 tests (`tests/test_tau_ecc.py`); benchmark
    `benchmarks/bench_tau_ecc.py`.

- **`exposure.contact_offsets_anchored`**: exact contact times for the
  anchored orbit, refined from the linearised ones by Newton steps
  through the anchored solve. They agree with bisected roots to 1e-11 d
  and handle grazing geometry. The linearisation `TransitModel` uses can
  misplace a contact by 2.3e-3 d on a grazing e = 0.5 orbit. That makes
  the contact rule 20x less accurate there at n_gl = 5 (2.9e-6 vs
  1.5e-7). It also puts the jump of dF/dtheta *inside* a quadrature
  piece, which costs the frozen-split gradient ~1% on d/dperiod. The
  eccentric tau path uses the exact contacts. `TransitModel` still uses
  the linearised ones (unchanged).

### Changed
- **The tau kernels are one orbit-generic template.** The exposure
  quadrature, its exact tau derivative and the per-chain simd_sum
  reduction are now written once, with the orbit (circular or anchored
  eccentric) as a plug-in that supplies z, dz/dphi and dz/dtheta. The
  circular instantiation is **bitwise identical** to v0.7.0: outputs and
  every gradient, scalar and `ld_basis`, all three rules, fp32 and fp64
  (94 arrays). Its speed is unchanged too (0.998x to 1.001x in an
  interleaved A/B). The four hand-written circular sources it replaces
  are gone.

### Fixed
- **README `ld_basis` timings** were measured on a contended machine.
  Quiet: 14.0 / 13.6 ms forward, 29.5 / 28.3 ms value+grad (scalar /
  basis). The ratios stand (0.97x / 0.96x). The instantaneous-rule
  overhead is 1.12x, not 1.30x.

## [0.7.0] — 2026-10-03

### Fixed
- **`flux_dev_from_tau`'s graph path could not batch chains.** Reported by
  a downstream sampler, which hit it on the fp64 reference path and worked
  around it by looping chain-by-chain (that workaround can go). With `n_chains > 1` and
  `integration="contact"` or `"supersample"`, any parameter spelling --
  including scalars -- raised `[broadcast_shapes] Shapes (n,1) and
  (n,m,k) cannot be broadcast`. Each exposure rule appends a node axis to
  the times, but the parameters were shaped `(n, 1)` regardless; that
  broadcasts against anything when `n == 1` and against nothing when it is
  not. How many trailing axes the parameters need is a property of the
  rule, so `_tau_graph` now derives it there.

  The fp32 kernel was unaffected -- it indexes parameters by chain rather
  than broadcasting -- which is why it never showed. The test matrix had
  the same blind spot: every fp64 test used a 1-D `tau`, so `n` was always
  1, and the `n > 1` cases were all behind the Metal skip. 60 tests added
  that run the graph at `n` in {1, 2, 4, 33} across all three rules, both
  precisions and both parameter spellings, and check that row `j` is what
  chain `j`'s parameters produce alone -- on chains deliberately made
  distinct, since shapes that broadcast are not automatically shapes that
  broadcast correctly. 455 tests green.

### Added
- **`ld_basis=True` on `flux_dev_from_tau` (and `flux_dev_metal`)**: off by
  default. Requested for collapsed limb darkening, which treats (u1, u2)
  as a linear block.
  It returns the (n, m, 3) basis `B`, where `B[..., j]` is the
  exposure-integrated, unnormalised deficit for intensity mu^j. For any
  quadratic law, with c = (1 - u1 - u2, u1 + 2 u2, -u2) and
  N = (pi, 2 pi/3, pi/2), the scalar call equals `(B @ c) / (N @ c)`.
  The kernel already held the Green's-basis deficits s0d, s1d, s2d, so
  the basis is B = (s0d, s1d, s0d/2 + s2d/4) stored where one collapsed
  number used to be. The VJP takes an (n, m, 3) cotangent and returns
  gradients in tau, period, a, b and r. It contracts the cotangent into
  the core first (ct . B is a scalar function of z, r), so the
  quadrature-derivative logic is shared with the scalar VJP rather than
  copied. The fp64 graph path (`vjp.ld_basis_analytic`, which has an
  analytic VJP) takes the same keyword.

  Measured: identity to <= 1.2e-8 (fp32 kernel) and ~1e-17 (fp64 graph),
  including grazing geometry and r = 0.3, over the whole q-box. Cost at
  512 x 5,000, contact, n_gl = 5: 0.95x forward and 0.97x value+grad
  relative to one scalar call, against 3x for the three calls it
  replaces. `ld_basis=False` is bit-identical to 0.6.1. The scalar
  kernel sources are byte-for-byte unchanged (the basis kernels are
  separate), and outputs and every gradient were compared bitwise against
  the previous commit on both paths. `u1`/`u2` are now optional keywords,
  required only when `ld_basis=False`. 84 tests
  (`tests/test_ld_basis.py`), benchmark `benchmarks/bench_ld_basis.py`.
  538 tests green.

- **`flux_dev_from_tau`** — a `tau`-input entry point with the exposure
  integration *inside* the kernel, requested for Kepler/TESS fitting with
  per-transit mid-times. Two gaps it closes. First, the
  fused model kernel treats times as data and returns a zero gradient for
  them, so a sampler fitting TTVs — a mid-time per epoch, sampled jointly
  with the shape parameters — could not use it; here `tau` is an ordinary
  differentiable input and the VJP returns `d(F)/d(tau)` alongside all six
  parameter gradients. Second, exposure integration existed only on the
  batman-style frontend, never on the path a sampler takes.

  The rule runs per output point in registers, so the sub-exposure axis
  never becomes an MLX array. At 512 chains x 5,000 points x 15
  sub-exposures: 2.8x forward and 3.6x value+grad on the *same*
  arithmetic, with peak memory 2776 MB -> 51 MB (55x), and the contact
  rule (25 evaluations) beats the 15-node expanded route on speed *and* is
  ~1,200x more accurate. Matched to 1e-6 the ratio is 30-57x, because
  supersampling converges as O(1/N) and needs n_sub ~ 2,271 — ~368 GB at
  512 chains. `benchmarks/bench_tau_kernel.py`, `tests/test_tau_kernel.py`.

  The fp64 / CPU / no-Metal graph path computes the same *function*, not
  merely the same value: both freeze the quadrature's split points, so a
  gradient certified in fp64 is the gradient that runs in fp32. (Freezing
  is exact, not an approximation — a split point is interior to a
  continuous integrand, so moving it adds +f(c)dc and -f(c)dc, which
  cancel. The window *ends* are genuine Leibniz boundary terms and the
  kernel carries them.) Getting there needed one fix: `exposure_nodes`
  converts contact *phases* to times with `period / 2 pi`, which leaked a
  period dependence into the split points that the kernel did not have —
  worth 9e-4 relative on `d/d(period)`, 2,000x the fp32 noise floor.
- **`notebooks/02_joint_transit_and_gp.ipynb`** — fitting a transit and
  correlated stellar variability *together*: Elómë as anvil-gp's mean
  model, a SHO Gaussian process over the residual, and anvil sampling all
  twelve parameters at once. The two packages compose with no adapter, because
  `make_quad_transit_flux` already returns exactly the `(v, x) -> (n_chains, m)`
  contract anvil-gp's `mean_fn` wants — which is what exposing
  `x64`/`y_fit`/`model_fn` on `TransitTarget` anticipated. Measured at the
  committed settings: 0 divergences, R-hat 1.001, every parameter within
  1.11 sigma (including the GP amplitude and timescale), ~2 minutes. It also
  records two findings from building it: `max_leapfrog=384`, best for the
  transit-only posterior, is wrong here (128 is right), and *shrinking* the
  light curve to save time made convergence worse, because the GP
  hyperparameters lose their constraint.
- **`notebooks/01_elome_with_anvil.ipynb`** — a tutorial covering
  forward modelling (orbits, limb-darkening laws, finite exposures, batched
  evaluation) and fitting real Kepler/TESS-style photometry end to end with
  anvil. Generated from `notebooks/build_01.py` so the prose and code stay
  reviewable in a diff, and every code cell is executed before commit; the
  committed copy carries no outputs. The sampling cell is sized to finish in
  ~25 s while producing a genuinely healthy fit (0 divergences, R-hat 1.002,
  ~1500 effective samples/s), and it teaches the coupling that produced the
  first draft's failure: 200 warmup iterations at `max_leapfrog=384` gave
  3,120 divergences and R-hat 1.22, where 300 gave zero and 1.002.
- **`make_transit_target(t, y, yerr, t0_guess, period_guess, ...)`** — the
  entry point for real Kepler/TESS photometry, and the piece that was
  missing for production use. It owns the float64 conditioning the
  float32 sampler depends on (time reduction, epoch centering, flux
  offset), the ParamSpec boxes (with `b` reaching past 1 so grazing
  geometries are inside the box, unlike the synthetic targets' 0.9),
  `report_offset` so results come back on the input time system, and the
  eccentric barrier. `TransitTarget.model_params()` builds the
  model-space vector from physical values; `x64`/`y_fit`/`model_fn` are
  exposed for anvil-gp to wrap. `examples/fit_mission_data.py` fits a
  TESS-shaped sector end to end: 40,000 points, zero divergences,
  R-hat 1.002, all eight parameters within 1.7 sigma.

### Fixed
- **float32 models silently corrupted mission time stamps.** The frontend
  uploaded absolute times raw, so a TESS BTJD (~2500 d, fp32 ulp 21 s)
  gave 1.65e-4 flux error — larger than many planet depths — Kepler BKJD
  5.1e-5, and raw BJD 1.2e-2 (unusable). The grid is now re-centred on a
  float64 reference at construction with the offset carried into t0
  everywhere, so the error is **9.9e-7 regardless of the time system**.
  float64 models keep a zero offset and are bit-unchanged.
- **`PenalizedLogLike` raised `RecursionError` under `copy`, `deepcopy`
  and unpickling**: `__getattr__` delegated unconditionally, so lookups
  that precede `__init__` recursed on `base` itself. Checkpointing or
  forking an eccentric target crashed.
- **The eccentric barrier had a plateau outside the e_max disc.** The
  model projects (secosw, sesinw) radially onto the disc, so beyond it the
  likelihood is exactly constant in the radial direction and the previous
  penalty's gradient there was purely tangential — a chain thrown into the
  box corners (e_raw up to 1.8) had no force pushing it back. The barrier
  now also penalises the *unprojected* eccentricity, and every residual is
  dimensionless: `|cos i| - 1` used to explode to ~1e12 with ~1e11
  gradients once the projection drove 1 - e^2 to 0.002, which would
  collapse HMC's step size. A one-sided Huber shape keeps the force
  bounded but never zero.
- **Changing the polynomial limb-darkening order between calls returned
  wrong flux silently** (2.6e-4 off; the extra terms were dropped by a
  `zip`). The order is now validated alongside the law.
- **The batched `light_curves` skipped the single path's validation**:
  `ecc < 0` produced an all-NaN row with only a numpy warning, and a
  quadratic set carrying three coefficients silently used the first two.
  Both now raise, as they already did for `light_curve`.

### Changed
- `light_curves` reuses the device-resident time grid instead of
  re-uploading it per call, the contact geometry
  (`a_sky`, `b_conj`) is computed by one `contact_geometry` helper rather
  than three copies that had already drifted apart, and `_unbroadcast` is
  shared from `vjp.py` instead of duplicated in `kepler.py`.

- `benchmarks/RESULTS.md` Elómë single-curve rows re-measured on a
  quiet machine after the 0.6.0 unification (the circular frontend now
  runs on the fused kernel): fp32 0.74 → 0.42 ms at N = 1e5 and
  1.43 → 1.13 ms at 1e6; unchanged at 1e7 (host-copy-bound); 0.24 →
  0.34 ms at N ≤ 1e4, where the kernel's dispatch floor is slightly above
  the graph path's — a 0.1 ms regression at sizes no GPU should be used
  for. The crossover against batman (between 1e4 and 1e5) is unchanged.
  fp64 rows, an unchanged path, moved < 3%.

## [0.6.1] — 2026-09-28

Quiet-machine re-measurement of 0.6.0, and a correction to how its
headline number was defined.

### Fixed
- **The cost of retiring the circular kernel is 1.06x forward / 1.10x
  value+grad**, not the 1.07x / 1.27x stated in 0.6.0. Two things were
  wrong with the earlier figure: it was measured under GPU contention,
  and its "value+grad" evaluated only the gradients. Our custom VJP
  recomputes from the primals and ignores the forward output, so when
  only gradients are evaluated MLX's lazy graph never runs the forward
  kernel at all and the number silently becomes VJP-only (1.21x on a
  quiet machine). A sampler needs energy *and* force — forward plus VJP —
  which is what every earlier table meant. `bench_ecc_kernel.py`,
  `profile_vjp_reduction.py` and the new `ab_retired_kernel.py` now
  evaluate both. Without the fast paths the same measurement reads
  1.43x / 1.40x; eccentric chains pay 1.01x.
- README kernel figures are now measured, not derived: circular
  21.8 / 60.2 ms, eccentric 29.4 / 76.2 ms at 1024 x 65,536 (3.08 and
  2.28 Gpt/s); eccentric value+grad peak memory 0.39 GB.
- The batched frontend measures **163x** a loop on a quiet machine
  (2,000 sets x 301 points, fp32); the 112x in 0.5.0 was taken under
  load. The batching claim in the sampler guide re-verified at 782x,
  inside its documented range.

### Added
- `benchmarks/ab_retired_kernel.py`: reproduces the unified-vs-retired
  comparison by extracting the pre-0.6.0 `metal.py` from git, fixing its
  imports and renaming its kernels (MLX caches JIT kernels by name, so
  two sources with one name would collide silently).

### Noted
- The frontend's fp32 `light_curve()` gains **nothing** from the kernel
  route at N >= 1e6 (0.99x circular, 1.01x eccentric at 1e7) because it
  is dominated by the float64 host copy — 80 MB at 1e7 — and 2.06x
  (circular) at 1e5 where dispatch overhead matters. The superseded
  `frontend-circular-kernel-plan.md` had a "revert below 1.3x at 1e7"
  gate for a *separate* route; this route is the same kernel and costs
  nothing to keep, so it stays, and the earlier claim of 2.1x for the
  eccentric frontend routing (measured under load) should be read as
  "2x at 1e5, host-copy-bound above".

## [0.6.0] — 2026-09-28

One model kernel. The dedicated circular kernel is retired; circular
stays a first-class *mode* (the 8-parameter anvil target, `ecc = 0.0` in
the frontend) and runs as e = 0 on the transit-anchored eccentric kernel,
which is exact there (7e-16 against the circular closed form).

### Changed
- **Circular orbits run on the eccentric kernel, with per-chain fast
  paths.** A simdgroup-uniform `if (e == 0)` skips the Markley starter
  and refinement in the forward pass, and skips computing, reducing and
  storing the seven eccentric-only gradient slots in the VJP. Measured
  against the retired kernel at 1024 x 65,536, same run: forward
  **1.07x**, value+grad **1.27x** (without the fast paths it would have
  been 1.46x / 1.41x); genuinely eccentric chains pay 0.98x — nothing.
  *Superseded by 0.6.1: measured under contention and with a VJP-only
  definition of value+grad; the corrected figure is 1.06x / 1.10x.*
  Flux parity 4e-7 at e = 0 (fp32 operation order) and bit-identical at
  e = 0.3. That cost (10% on value+grad once measured properly, see
  0.6.1) is the deliberate price of one kernel: half the surface for
  every future photometric fix, and mixed circular/eccentric batches
  (`TransitModel.light_curves`) for free.
- `make_model_core_metal(period_ref)` is now the unified factory taking
  the packed orbit constants; `make_ecc_core_metal` remains as an alias.
  The `reduce="grid"|"simd"` switch is gone with the kernel it belonged
  to — the in-kernel `simd_sum` reduction is the only path (its A/B
  numbers are preserved in `benchmarks/profile_vjp_reduction.py`'s
  docstring).
- The frontend's fp32 GPU *circular* path now routes through the kernel
  too, which is what `docs/frontend-circular-kernel-plan.md` proposed —
  achieved by unification rather than by a second route. That plan is
  marked superseded and kept as the record of the road not taken.
- A "set e = 1e-4" fudge is neither needed nor harmless: measured, it is
  a real model error scaling linearly with e — 3e-6 at e = 1e-4, above
  batman's floor. Use exactly zero.

### Removed
- `_ORBIT` (circular), `_MODEL_VJP_TAIL` (7-slot) and the v2 kernels. The
  retired kernel is recoverable from git history (the commit before this
  release) if a fixed-circular workload ever needs the last 27% back.

## [0.5.0] — 2026-09-27

### Added
- **`TransitModel.light_curves(params_seq)`** — the batched frontend.
  Same time grid, many parameter sets, one dispatch: **112x** a Python
  loop over `light_curve` (2,000 sets x 301 points, float32 GPU). Accepts
  a sequence of `TransitParams` or one whose scalar attributes are
  arrays. Works in every mode (uniform/linear/quadratic/polynomial,
  supersampled or contact-integrated) and handles **mixed circular and
  eccentric sets in one batch**, since the transit-anchored orbit
  degenerates exactly to the circular one at e = 0. This closes the gap
  between what `docs/sampler-integration.md` prescribes — never loop,
  always batch — and what the batman-style API made easy.
- `docs/frontend-circular-kernel-plan.md`: a full plan for routing the
  frontend's *circular* path through the v2 kernel, deliberately **not
  implemented**. Measured reward is only ~1.4x (2.31 vs 3.26 Gpt/s)
  against the highest risk on the backlog, so the design is recorded for
  whoever revisits it rather than executed.

### Changed
- `exposure_nodes` and `flux_dev_poly` accept a leading batch axis, so
  contact integration and polynomial limb darkening both work under the
  batched frontend.
- **ChEES `max_leapfrog` re-measured** (`benchmarks/bench_leapfrog_cap.py`).
  The examples' cap of 24 dated from when gradients cost ~1.2 s; at
  55–76 ms it was the binding constraint on the circular target, where
  384 gives **8.7× the ESS/s (8.2 → 71.2)** and takes R̂ from 1.83 to
  1.00 with no divergences. On the *eccentric* target the same change is
  harmful — ESS/s falls monotonically and 96 steps produces 172
  divergences — so `examples/chees_ecc.py` keeps 24 and says why. The
  guidance, and the caution that ESS is ceiling-limited by
  `n_chains × n_samples`, is in `docs/sampler-integration.md`.

## [0.4.1] — 2026-09-27

Two defects in 0.4.0's new exposure code, found by a closing audit.

### Fixed
- `light_curve_mx` returned the *instantaneous* flux at the exposure
  mid-times under `integration="contact"` while `light_curve` returned
  the averaged one — a silent ~1e-3 discrepancy on the entry point
  documented for differentiable pipelines. The eager path now performs
  the same contact-split average (they agree to 2e-16).
  `supersample_factor` still returns the raw supersampled grid, which is
  its documented behaviour.
- Contact-split quadrature weights are renormalised. The clamped
  sub-interval widths summed to the exposure only to ~1e-14, so
  out-of-transit flux came back as 1 + 2e-14 rather than 1 to round-off,
  breaking the exact-unity contract kept everywhere else. The weights are
  a partition of unity by construction, so this removes round-off and
  nothing else: now 1 ulp, convergence figures unchanged.

## [0.4.0] — 2026-09-27

Closes the two remaining optional items from the original M6 list
(arbitrary order, exposure-time integration) and gives the cross-code
benchmark an eccentric half.

### Added
- **Arbitrary-order polynomial limb darkening**, `limb_dark="polynomial"`
  with `u = [u_1 ... u_N]` for any N (`elome/poly.py`). ALFM19's
  three-term M_n recursion seeded by closed forms for M_0..M_3, then
  s_n = -(2 r² M_n - n/(n+2)[(1-r²-z²) M_n + sqarea M_{n-2}]). Validated
  against a 40-digit mpmath direct integration sharing no code with it:
  1e-15 at N = 8, 2e-13 at N = 16, 3e-9 even at N = 30 (the upward
  recursion loses ~1 digit per 2 orders; Limbdark.jl's downward
  series-seeded variant is documented but unnecessary at any order a
  physical law uses). Non-polynomial laws remain unsupported — they are
  outside the ALFM19 formulation, and the error now says so.
- `greens_affine()` writes the u → g map as the affine map it is, so the
  coefficients stay **traced**: they can change between calls on a built
  model (the batman workflow) and the limb darkening is differentiable.
- **Contact-split exposure integration**, `integration="contact"`
  (`elome/exposure.py`). The light curve's derivative jumps at each
  contact, so uniform supersampling converges only as O(1/N). Splitting
  the exposure window at the contacts and applying a fixed-order
  Gauss-Legendre rule to each smooth piece converges geometrically:
  25 evaluations per exposure reach 8.9e-8 where supersampling needs
  N ~ 19,500 — ~780x fewer model evaluations
  (`benchmarks/bench_exposure.py`). Fixed order rather than Limbdark's
  adaptive Simpson, because adaptive depth is data-dependent control
  flow and would break batching and `mx.compile`; the five-interval split
  is branchless, so grazing and full transits share one code path and the
  node positions stay differentiable.
- **Eccentric cross-code benchmarks**: `RESULTS.md` now compares all five
  codes at e = 0.3 on precision and orbit-inclusive speed, each solving
  Kepler's equation itself. Elómë fp64 is the most accurate
  (4.4e-16, tied with exoplanet-core) and its fp32 GPU path the fastest
  past N ~ 1e5 (15.3 ms at 10^7 points vs 207 ms for the next code).

## [0.3.0] — 2026-09-27

The eccentric release: eccentric orbits now run on the fused Metal
kernel with analytic gradients, and have their own anvil sampling
target. Plan, gates and review log: `docs/v3eccentrickernel_plan.md`.

### Added
- **Transit-anchored eccentric orbit** (`elome/anchored.py`).
  Solves for δ = E − E₀ about inferior conjunction rather than for E
  about periastron. Algebraically identical to the direct formulation
  (verified to 2.5e-15 in flux over an (e, w) grid including
  near-apastron and near-periastron transits at e = 0.999), but it never
  forms ω as an intermediate — only e·cosω and e·sinω, taken directly
  from the (√e cosω, √e sinω) sampling pair. Since ∂ω/∂k = −h/e
  diverges, the direct form computes a finite gradient as the difference
  of two O(a/e) terms: measured fp32 gradient error 5e-2 at e = 1e-5 and
  1e+1 at e = 1e-8, versus ~1e-7 *flat* for the anchored form
  (`benchmarks/v3_kh_grad_conditioning.py`).
- **v3 eccentric Metal kernel** (`make_ecc_core_metal`): anchored solve,
  Cartesian tail and ALFM19 photometry in one register-resident pass,
  plus its analytic VJP (14 per-point partials). At 1024 × 65,536:
  forward 29.0 ms, value+grad 76.0 ms — 1.41x / 1.38x the circular
  kernel, and 20x / 902x the same model as a compiled MLX graph.
  Parity 3.7e-7 vs the graph over e ∈ [0, 0.999].
- **In-kernel gradient reduction** for the circular v2 VJP
  (`make_model_core_metal(reduce="simd"|"grid")`, simd default).
  `metal::simd_sum` reduces over the *active* lanes, so early-returned
  lanes drop out by themselves — no predication, threadgroup memory,
  barrier or grid padding. Backward 35.8 → 30.8 ms (1.16x), peak memory
  4.56 → 2.74 GB, transients 1.88 GB → 58.7 MB. `"grid"` is retained as
  the parity oracle.
- **Eccentric anvil target**: `make_ecc_transit_flux` / `make_ecc_target`
  (10 parameters), `ecc_constraint_penalty` and `PenalizedLogLike` for
  the two joint constraints a box of ParamSpecs cannot express
  (periastron clearance, |cos i| ≤ 1) — without them an unphysical
  geometry returns a finite log-likelihood and the chain silently
  samples an improper posterior.
- `examples/chees_ecc.py`: eccentric ChEES-HMC injection-recovery at a
  deliberately low truth (e = 0.02), where the parameterization matters.
  1024 chains, 600 warmup + 200 samples: **zero divergences**, truth
  recovered within 1σ on all ten parameters. The 10-parameter posterior
  mixes far more slowly than the circular one (R-hat 3.5 vs 1.5 for the
  circular control at matched 256 chains × 400+400) — the well-known
  (a, b, e, ω) transit-duration degeneracy, not a model defect.
  `benchmarks/v3_ecc_sampling_geometry.py` reproduces the diagnosis,
  including the A/B that eliminates the constraint barrier as the cause
  of the divergences seen at shorter warmups.
- The batman-style frontend routes its fp32 GPU *eccentric* primary
  transits through the v3 kernel: 2.1x at N = 1e7, 1.5x at 1e6, 1.2x at
  1e5 on `light_curve()`. The period is carried by the kernel's traced
  `p_off` input (with `period_ref = 0`) so batman-style parameter
  updates never see a baked-in constant.
- Benchmarks: `bench_ecc_kernel.py`, `v3_reduction_spike.py`,
  `v3_kh_grad_conditioning.py`, `v3_ecc_graph_baseline.py`; the VJP
  profiler now A/Bs both reduction strategies.

### Changed
- Markley starter cbrt: `metal::precise::powr` (~220x a multiply, ~20%
  of orbit time) replaced by the inverse-cbrt bit trick with the
  principled (4/3)·0x3f800000 seed and three division-free Newton steps;
  1.7e-6 max relative error in-kernel against a ~1e-4 requirement.
- The batman-style frontend uses the anchored formulation for eccentric
  orbits, so φ is measured straight from t0 and the
  mean-anomaly-at-transit offset no longer appears in the compiled graph.
- Kernel sources share one `_PHOT_PARTIALS` fragment and expand through
  `_subst`, which asserts no marker survives — an unexpanded marker is a
  Metal compile failure, which aborts the process rather than raising.

### Fixed
- **NaN gradients at δ ≈ π** (ordinary apastron-side geometry):
  `_one_minus_cos` floored its denominator instead of setting it to 1 on
  the inactive branch, so the VJP hit 0 × inf for every parameter routed
  through the solve.
- **NaN gradients at exactly e = 0** — an interior point of the sampling
  disc — from `sqrt(max(e, 0))`, whose derivative is infinite there, and
  from `arctan2(0, 0)`, whose gradient is undefined.
- `test_loglike_at_truth_is_sane` now accounts for upstream anvil's
  recentring policy (the unnormalized logL is `value + log_offset_const`).

## [0.2.0] — 2026-09-06

### Added
- **Fused Metal kernels** (`elome/metal.py`): the fp32 GPU path
  runs the whole model in registers. v1: photometric core
  (z, r, u1, u2) → flux + analytic-VJP kernel; v2: model-level kernel
  consuming the anvil (v, x) contract with the orbit folded in.
  Measured (M2 Max, 1024×65,536): forward 17.5 ms, value+gradient
  52.5 ms (519× reverse-mode autodiff); ~55,000 light curves/s in
  population batches. `core="metal"` is the default for the anvil
  target and the fp32 frontend, with silent graph fallback for
  fp64/CPU.
- **Markley Kepler solver** (`kepler`, `kepler_E_sincos`):
  non-iterative starter + one fifth-order refinement, a single sincos
  per point, implicit-function-theorem custom VJPs at both the true-
  anomaly and eccentric-anomaly level.
- Benchmark suite (`benchmarks/`): precision vs a 30-digit mpmath
  oracle, subprocess-isolated speed sweeps vs batman / PyTransit /
  exoplanet-core / jaxoplanet, batch scaling, and
  `verify_doc_claims.py` making every documented number reproducible.
- Docs: `docs/sampler-integration.md` (the batching rule, sampler
  recipes, verified against a spy test), `docs/eccentric-kernel-notes.md`
  (verified backward-pass formulas and kernel design findings).
- Engine-side (anvil repo): `run(progress=N)` flushed progress ticks
  with rate/acceptance/divergences/ETA.

### Changed
- **Eccentric separation is now Cartesian-from-E** — the true anomaly
  is never computed. Exactly equivalent to the previous conic tail
  (proven symbolically and to 4.5e-25 at 40 digits) and strictly
  better conditioned: removes a ~1e-6 fp32 flux-error floor near every
  transit and a ≥1e-3-flux failure mode at near-apastron transits of
  high-e orbits; fp32-trustworthy at the 1e-6 flux level to e ≤ 0.999.
- batman-style frontend evaluates through `mx.compile`d graphs with
  parameters as traced scalars (no retrace on parameter updates).

### Fixed
- Three latent dtype bugs where MLX ops on pure-Python scalar operands
  minted float32 constants inside fp64 paths (`beta`, `fac`, `ome2`).
- Circular orbit no longer fabricates a mirror transit at the far
  conjunction (front-side masking).
- 15 code-review findings in the sampler guide, including an incorrect
  emcee batching claim (half-ensembles), a priors-free example that
  sampled an improper posterior, and unreproducible measured numbers.

## [0.1.0] — 2026-09-05

Initial release: ALFM19 quadratic limb-darkened transits in MLX.
Batched fixed-iteration Bulirsch `cel`; deviation-form solution vector
with masked regimes and Taylor stability switches; Green's-basis flux
assembly; analytic custom VJP; Kipping (2013) (q1,q2) limb darkening;
epoch-centered circular orbit; batman-parity frontend
(TransitParams/TransitModel incl. eccentric orbits, supersampling,
secondary eclipses); anvil integration target; accurate fp64
sincos workaround for MLX's fp32-accurate fp64 trig; validation against
batman, a Limbdark.jl port, and direct mpmath integration.
