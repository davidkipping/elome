# Geometry

## The barycentre and the two bodies

The sky frame is the one every Elómë model uses: $X$ along the
barycentre's motion at mid-transit, $Y$ the sky projection of its orbit
normal, and $Z$ towards the observer; the barycentre transits at
$Y = -b$. Its position $\mathbf B(t)$ is that of a spherical planet on
the orbit (`t0`, `per`, `a`, `inc`, `ecc`, `w`).

The moon's position relative to the planet, $\mathbf d$, follows a
Keplerian orbit of semi-major axis $a_{ps}$ = `moon_a` × `rp` (in stellar
radii), eccentricity $e_m$ and argument of periastron $\omega_m$. With
$f_m$ its true anomaly and $r_m = a_{ps}(1 - e_m^2)/(1 + e_m\cos f_m)$,

$$
(u, v) = r_m\,\big(\cos(\omega_m + f_m),\ \sin(\omega_m + f_m)\big),
\qquad
\mathbf d = R_Z(\Omega_m)\,\big(-u,\ -v\cos i_m,\ v\sin i_m\big),
$$

where $i_m$ = `moon_inc`, $\Omega_m$ = `moon_node` and $R_Z$ turns the
sky counterclockwise, from $+X$ towards $+Y$. The barycentre's own orbit
has this form with $\Omega = 0$. The planet and the moon sit on either
side of the barycentre, in proportion to their masses, $q$ = `moon_q`:

$$
\mathbf r_p = \mathbf B - \frac{q}{1 + q}\,\mathbf d,
\qquad
\mathbf r_m = \mathbf B + \frac{1}{1 + q}\,\mathbf d.
$$

So the planet wobbles about the barycentre by $q/(1 + q)$ of the moon's
distance. That shifts its transits in time and duration, which a massless
moon ($q = 0$) does not do.

Each body is an opaque disc, of radius `rp` and `rp` × `moon_r`. A body
behind the star ($Z < 0$) blocks nothing. Where the two discs overlap on
the star, the light behind both is counted once.

## The moon's orbit on the sky

The moon's elements are in the **sky frame**, as in gefera, Pandora and
planetplanet. Its orientation does not depend on the planet's orbit, nor
on the planet's argument of periastron.

- **Coplanar** is `moon_inc = inc`, `moon_node = 0`, the default: the
  moon orbits in the barycentre's orbital plane, in the same sense.
- **`moon_node`** turns the moon's orbit about the line of sight. The
  orbit normal is
  $(-\sin\Omega_m \sin i_m,\ \cos\Omega_m \sin i_m,\ \cos i_m)$, against
  the barycentre's $(0, \sin i, \cos i)$.
- **The mutual inclination** $\iota$, the angle between the two orbits,
  follows from
  $\cos\iota = \cos i \cos i_m + \sin i \sin i_m \cos\Omega_m$.
  `elome.moons.mutual_inclination(inc, moon_inc, moon_node)` returns it
  in degrees, together with the mirror image's (below). A retrograde
  coplanar moon is `moon_inc = 180 - inc`, `moon_node = 180`.

**`moon_phase`** is the moon's mean argument of latitude at `t0`, less
90°: $\varphi_m = M_m(t_0) + \omega_m - 90°$, with $M_m$ its mean
anomaly. Its mean phase at time $t$ is
$\varphi_m + 360°\,(t - t_0)/P_m$. For a circular moon it is the angle
along the orbit from conjunction in front of the planet. For a coplanar
moon:

- at 0° the moon is in front of the planet;
- at 90° it leads, by its full distance, in the direction of motion;
- at 180° it is behind the planet;
- at 270° it trails.

The period `moon_per` is free. To tie it to the planet's density,
`elome.moons.moon_period(moon_a, rho_p, q)` gives Kepler's third law:
days from the semi-major axis in planet radii and the planet's mean
density in kg m⁻³.

## The parameters

| parameter | definition | notes |
|---|---|---|
| `moon_r` | the moon's radius | in units of `rp`; setting it makes a model with a moon |
| `moon_a` | the semi-major axis of the moon's orbit about the planet | in units of `rp`; required |
| `moon_per` | the moon's orbital period | required; in the units of `t` |
| `moon_q` | the moon's mass over the planet's | `None`: 0 |
| `moon_inc` | the inclination of the moon's orbit on the sky (degrees) | `None`: `inc` |
| `moon_node` | the node of the moon's orbit on the sky (degrees) | `None`: 0 |
| `moon_phase` | the moon's mean argument of latitude at `t0`, less 90° (degrees) | `None`: 0 |
| `moon_ecc` | the moon's eccentricity | `None`: 0 |
| `moon_w` | the moon's argument of periastron (degrees) | `None`: 0 |

`t0`, `per`, `a`, `inc`, `ecc` and `w` are the barycentre's. `rp` is
the planet's radius, so the moon's radius and orbit scale with it.

## Symmetries

- **The mirror image.** Adding 180° to `moon_node`, `moon_w` and
  `moon_phase` together reflects the moon's orbit through the sky plane.
  Its sky positions, and so the light curve, are the same at every time.
  The transit cannot tell the two apart. For an edge-on planet the mirror
  of a prograde coplanar moon is a retrograde one. In a fit, restrict
  `moon_node` to a range of 180° (see {doc}`accuracy-and-fitting`).
- **The moon's periastron on a circular orbit.** At `moon_ecc = 0`,
  `moon_w` has no effect: `moon_phase` already places the moon from
  conjunction.
- **A massless, invisible moon.** At `moon_r = 0` and `moon_q = 0` the
  light curve is the planet's alone.

## The domain

- `moon_r` $\ge 0$ and `moon_q` $\ge 0$;
- $0 \le$ `moon_ecc` $< 1$;
- the moon clears the planet at periastron:
  `moon_a` $(1 -$ `moon_ecc`$) > 1 +$ `moon_r`;
- every value finite.

The planet's orbit and the limb darkening follow a spherical planet's
rules. Outside these a Python number or numpy array raises `ValueError`.
An MLX array is not checked, so keep a sampler inside the domain itself.
The moon's orbit is not checked for stability: a moon must orbit well
inside the planet's Hill sphere, which the model does not know.

The two bodies may be anywhere: overlapping on the sky (a mutual
event), one in transit and the other not, or one behind the star.
