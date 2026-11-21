# Accuracy, cost and fitting

## Accuracy

**float64** is exact to rounding with a hybrid law. The test suite
compares every limb-darkening term against an unrelated integration of
the occulted intensity: in polar coordinates about the star's centre,
with the angle each body covers on every circle in closed form, split
at every kink. Against it, Elómë is within $4\times10^{-15}$ of each
term's full-disc flux in every geometry tested:

- the two bodies apart, overlapping and one inside the other, on the
  disc and across the limb;
- the moon off the star while it overlaps the planet on the limb, and
  the planet off the star with the moon on the limb;
- a tiny moon on the planet's edge, and a moon nearly as large as the
  planet;
- either body at the disc's centre, and both off the star;
- exact tangencies: the discs touching from outside and inside, and the
  moon touching the limb while it overlaps the planet.

The sky positions agree with rotation matrices built independently to
$5\times10^{-14}$ of the orbit's size. They agree with LUNA's equations
(Kipping 2011), after converting the elements, to $10^{-11}$. Light
curves on circular and eccentric orbits agree to $10^{-13}$ with the
photometry evaluated at those independent positions, and a moon of zero
size and mass leaves the planet's own light curve to $2\times10^{-14}$.

**The quadratic law** (and the uniform and linear laws, its special
cases) is exact for each body's disc. Where the two discs overlap on the
star, its $\mu^1$ term would need incomplete elliptic integrals. Elómë
instead takes the overlap's exact $\mu^0$, $\mu^2$ and $\mu^4$ moments
and closes the $\mu^1$ term with a distribution of $\mu^2$ that matches
their mean and variance. For a moon of radius 0.03 on a planet of 0.1,
that is within $1.3\times10^{-9}$ while the overlap is on the disc, and
$7.6\times10^{-7}$ at worst while it straddles the limb. Treating the
moon's whole disc at one intensity would be off by about $10^{-5}$
there. The error grows with the overlap's area.

**float32** keeps the moon's phase to its float32 rounding however far
the data lie from `t0`. The barycentre's orbit numbers and residual
times, and the moon's phase at each transit, are reduced in float64 on
the host, as on the anvil path. 120 orbits from a BJD-sized `t0`, a
light curve with a hybrid law is within $6\times10^{-8}$ of float64. With
the quadratic law it is within $1.3\times10^{-6}$, for the reason below.

### Where float32 has limits

- **The quadratic law where the discs overlap across the limb.** The
  $\mu^1$ term goes as the square root of $\mu^2$ near the limb, so the
  closure magnifies the overlap's float32 rounding, about $2\times10^{-7}$
  in its moments. While the overlap straddles the limb, a float32
  quadratic light curve can be off by up to $7\times10^{-6}$ (the median
  is $5\times10^{-8}$). A hybrid law stays within $1.3\times10^{-7}$
  there. For mutual events near the limb, use a hybrid law or float64.
- **`t0`** is the least precise gradient, as for every model, since the
  float32 times are.

Gradients in every parameter, the moon's included, agree with finite
differences, under the contact rule as well (see {doc}`light-curves`).

## Cost

A model with a moon is evaluated eagerly, not compiled. The host chooses
which points either body can reach the star from and where the two discs
overlap, so each call pays a fixed cost for building the graph, and then
a few operations per point that is evaluated. Points far from transit
cost nothing. The overlap is computed only where the discs overlap on
the star, and mutual events are brief. Under the contact rule each
exposure also pays for its event search, and up to seven pieces of
`n_gl` nodes each, though most need one or two. Batch parameter sets
with `light_curves` or the anvil model rather than calling
`light_curve` in a loop. Spherical, oblate, ringed and spotted models
never load or run any of the moon code.

## Fitting a moon

- **Restrict the node to 180°.** The mirror image (`moon_node`,
  `moon_w` and `moon_phase` each plus 180°) gives the same light curve,
  so every posterior has two copies of each mode. A range such as
  $(-90°, 90°)$ keeps a coplanar moon (node 0) in the middle and leaves
  out its mirror. For an edge-on planet the mirror of a prograde moon is
  a retrograde one: the transit cannot tell them apart.
- **Sample the orientation isotropically.** An orbit normal uniform on
  the sphere has $\cos$(`moon_inc`) uniform on $[-1, 1]$ and the node
  uniform.
- **Expect the period to alias.** The light curve samples the moon's
  phase only around each transit, so periods that bring the moon back to
  nearly the same phases at the observed transits fit nearly as well.
  Bound the period, or start chains across its range.
- **The mass and the radius are separate signals.** `moon_q` moves the
  planet about the barycentre, which shifts its transits in time and
  duration; `moon_r` adds the moon's own dip. A moon of zero mass makes
  no timing signal, and one of zero radius makes no dip.
- **Keep chains physical.** The model returns a finite light curve for a
  moon that dips inside the planet's disc (an eccentric orbit with
  `moon_a` $(1 - e_m) \le 1 +$ `moon_r`), and it knows nothing of
  stability. A moon must orbit well inside the planet's Hill radius,
  $a\,(M_p / 3M_\star)^{1/3}$ in stellar radii; on a prograde orbit,
  within about half of it (Domingos, Winter & Yokoyama 2006). Bound
  `moon_a` or add these as priors.
- **Prefer a hybrid law** for float32 fits with mutual events near the
  limb (see above). Elsewhere the quadratic law is as accurate as the
  data allow.
- **Batch the chains.** `elome.anvil.make_moon_transit_flux` evaluates
  every chain in one call, with gradients. Call it eagerly, not under
  `mx.compile`.
- **With anvil**, `make_transit_target(..., moon=True,
  moon_period_guess=P_m)` builds the whole problem, for the quadratic law
  or a hybrid law. The moon is sampled as `moon_r` and `moon_a` (planet
  radii), `moon_p_off` (days, the period's offset from
  `moon_period_guess`), `moon_q`, `cos_moon_inc`, `moon_node` and
  `moon_phase` (degrees). With `moon_eccentric=True` it adds
  `moon_secosw` and `moon_sesinw`. The boxes are
  `elome.anvil.DEFAULT_BOUNDS`. Its node box is $(-90°, 90°)$, and
  `moon_p_off` has no default, so pass `bounds={"moon_p_off": (lo,
  hi)}`. `tt.model_params(t0, period, moon_per=...)` converts a period
  to its offset. Run the sampler with `compile_step=tt.compile_step`.
