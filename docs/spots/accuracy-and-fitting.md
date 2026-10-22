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
the other spots. The cost therefore grows with the number of spots, and
an exposure under the contact rule also pays for its event search and
about fifty evaluations. float32 on the GPU runs fused kernels for both,
in milliseconds for hundreds of exposures, and is the route for
sampling. float64 on the CPU is the reference: under the contact rule it
takes seconds for a few thousand exposures.

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
- **Batch the chains.** `light_curves` evaluates every parameter set in
  one call (`spots` of shape (n_sets, K, 4)); in float32 on the GPU it
  runs the kernels. See {doc}`../guides/batching`.
