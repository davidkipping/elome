# Accuracy, cost and fitting

## Accuracy

**float64** is exact to rounding. The test suite compares every
limb-darkening term against two unrelated brute-force integrations of
the occulted intensity, which agree with each other to
$7\times10^{-16}$: rays about the star's centre, cutting each spot's cap
along every ray; and the planet's disc plus each spot's cap integrated
over $\mu$, less the lens the planet cuts from it by rays about the
planet's centre. Against them, Elómë is within $2\times10^{-15}$ of each
term's full-disc flux in every geometry tested:

- a spot alone, on the disc, behind it and straddling the limb, down to
  a thin crescent on the limb;
- the planet across a spot's rim, inside a spot, on a spot inside it,
  and on the limb across a rim;
- overlapping spots, a facula over a spot, an umbra in a penumbra, and
  three spots at once;
- a spot covering the disc's centre, and at random across the visible
  hemisphere and just behind the limb.

Light curves on circular and eccentric orbits, at any obliquity, agree
to $10^{-13}$ with an independent reference: the planet's sky position
and the spots' from plain numpy orbits and rotations, and the
brute-force integral at each time. A spot of contrast 1 reproduces the
unspotted spherical light curve to $2\times10^{-15}$.

**float32** (the GPU kernels) is within $2\times10^{-7}$ of float64 in
flux, on both exposure rules and on circular and eccentric orbits, the
same as the float32 graph. Its gradients in every parameter are within
$10^{-5}$ of float64 automatic differentiation, relative to their scale,
and typically within $10^{-6}$; `t0` is the least precise, as the
float32 times are.

### Where float32 has limits

- **`hybrid5` at the limb.** Its innermost pole has $\epsilon = 0.0016$,
  so its term $(\mu^2 + \epsilon)^{-2}$ spans a factor of $4\times10^5$
  across the disc. For a thin spot on the limb, or a spot's rim just
  grazing it, a float32 column can be off by up to $1.5\times10^{-6}$ of
  its full-disc flux, against $3\times10^{-7}$ for `hybrid4` and
  $10^{-7}$ for `hybrid2`. In a light curve the spot's small area scales
  this down.
- **A spot near the disc's centre.** A spot's rim is described in a frame
  that turns with the spot, and that turns fastest, as $1/|n_{xy}|$, at
  the centre. The kernels and the graph both take the turn in a form
  that stays finite there, so a spot crossing the centre has gradients
  accurate to $2\times10^{-5}$, at the centre itself included.

## Cost

Every spot's visible outline is integrated at every time, in transit or
not, and each splits into arcs where it meets the limb, the planet and
the other spots. The cost therefore grows with the number of spots. Under
the contact rule an exposure also pays for its event search, and up to
seven pieces of `n_gl` nodes each; most exposures need only one or two,
and only the nodes of non-empty pieces are evaluated.

Fused float32 kernels, 512 parameter sets × 5,000 points around a
transit, `hybrid4`, spots of 6–12°, two of them on the transit chord, on
a star turning in 12 days, against the same transit with no spots
(`benchmarks/bench_spots.py`, M2 Max):

| | forward | vs no spots | value + grad | vs no spots |
|---|---:|---:|---:|---:|
| one spot, one evaluation per point | 47 ms | 59× | 232 ms | 137× |
| two spots, one evaluation per point | 76 ms | 96× | 374 ms | 220× |
| four spots, one evaluation per point | 151 ms | 190× | 697 ms | 410× |
| one spot, contact rule (`n_gl = 5`) | 0.95 s | 270× | 3.22 s | 419× |
| two spots, contact rule (`n_gl = 5`) | 1.43 s | 408× | 4.81 s | 626× |
| four spots, contact rule (`n_gl = 5`) | 4.04 s | 1,154× | 12.3 s | 1,599× |

The gradients are in every input: the orbit, the planet, the star, every
entry of the spot table and the weights. Two spots under the contact rule
cost less than a ringed planet (see {doc}`../rings/accuracy-and-fitting`).

float64 on the CPU is the reference, not a route for sampling: one light
curve of 1,000 points with two spots takes 0.088 s for its value and
0.37 s with its gradient, and under the contact rule 3.1 s and 3.7 s.

## Other codes

Among codes that model a planet crossing spots:

- **KSint** (Montalto et al. 2014) also integrates circular spots exactly,
  with the quadratic law, in Fortran, without gradients.
- **spotrod** (Béky, Kipping & Holman 2014) is exact in angle and
  integrates in radius by quadrature.
- **spotter** (Garcia, Rackham & Panwar 2025) maps the star on a
  HEALPix grid on the GPU, in JAX.
- **starry** (Luger et al. 2019) expands the surface in spherical
  harmonics, so a spot is smoothed to the resolution of the expansion.
- **fleck** (Morris 2020) models the rotational modulation of many
  spots, and their crossings by sampling.

Elómë is exact like KSint, with the hybrid laws, gradients in every
parameter, and batched float32 kernels on the GPU.

## Fitting spots

- **Fit a normalisation.** The model is normalised to the unspotted
  star, so out of transit it sits below 1 by the spots' coverage. Data
  normalised to their own out-of-transit level need a free factor (or
  offset) on the model; it is nearly degenerate with the spots' overall
  coverage when the modulation is not observed.
- **Spots are interchangeable.** Swapping two spots' rows gives the same
  light curve, so a posterior over K spots has K! copies of each mode.
  Order them, for example by longitude, or give them distinct priors.
- **Sample geometry isotropically.** A spot placed uniformly on the star
  has longitude uniform on $[0°, 360°)$ and $\sin$(latitude) uniform on
  $[-1, 1]$. For the size, $\cos\alpha$ uniform is uniform in the
  fraction of the surface covered. Wrap the longitude; it is periodic.
- **Expect size, contrast and position to trade off.** A single
  crossing measures how much light the planet failed to block, and for
  how long: a larger, fainter spot and a smaller, darker one can look
  alike, and the spot's distance from the transit chord trades against
  its size. The rotational modulation, repeated crossings and several
  wavelengths break these.
- **Fit wavelengths jointly.** A spot's position and size are the same in
  every passband, and its contrast and the star's limb darkening are not.
  Build one model per passband and share the spots' longitudes,
  latitudes and radii between them, with a contrast and limb-darkening
  weights per passband.
- **Keep chains in the domain.** The angular radius must lie in
  $(0°, 90°]$, the latitude in $[-90°, 90°]$ and the contrast be
  non-negative; outside, an MLX array gives NaN. Overlapping dark spots
  are allowed, and their deficits add, so two can reach a negative
  intensity (see {doc}`geometry`); keep spots apart in the prior, or
  nest them deliberately.
- **Fix the number of spots per model.** Compare models with different
  numbers of spots by their evidence, not by letting a contrast run to
  1.
- **Batch the chains.** `flux_dev_from_tau` with `spots=` evaluates every
  chain in one call, with gradients (`spots` of shape (n, K, 4)); in
  float32 on the GPU it runs the kernels. Call it eagerly, not under
  `mx.compile`. See {doc}`../guides/sampler-api`.
- **With anvil**, `make_transit_target(..., limb_dark="hybrid4",
  n_spots=K)` builds the whole problem. Each spot is sampled as `lonj`
  (degrees, at `tt.spot_epoch`), `sinlatj`, `radj` (degrees) and
  `contrastj`, then the star as `prot` (days), `cos_istar` and `lam`
  (degrees). The boxes are `elome.anvil.DEFAULT_BOUNDS`; a field's key
  (`"rad"`) sets every spot's box and a numbered one (`"rad2"`) a single
  spot's. `prot` has no default, so pass `bounds={"prot": (lo, hi)}`.
  The longitude is a box, not a wrapped angle: if a spot's posterior
  reaches ±180°, move `spot_epoch` or the box. The spots are left
  unordered; relabel the draws afterwards. Run the sampler with
  `compile_step=tt.compile_step`. By default (`relative=True`) the model
  is the flux relative to the spotted star's own level at each time, which
  is what a light curve normalised to its out-of-transit level measures;
  it costs a second evaluation, without the planet. Tutorial 4 fits two
  crossings in Kepler data this way, from finding the spots to sampling.
- **Keep the spots' epoch near the data.** The star's phase grows with
  the time since the epoch, and float32 carries it to about
  $6\times10^{-8}$ of itself: an epoch in the middle of the data keeps
  that error smallest.
