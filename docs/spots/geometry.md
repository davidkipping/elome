# Geometry

## A spot on the sky

A spot is a circular cap on the stellar surface: every point within an
angle $\alpha$ of its centre, the unit vector $\hat n$. Its boundary, the
rim, is a circle on the sphere, and a circle on a sphere projects to an
ellipse on the sky:

- **centre** at $\cos\alpha\,(n_x, n_y)$, the projected centre of the
  rim's circle;
- **semi-major axis $\sin\alpha$**, perpendicular to the direction of the
  spot from the disc's centre;
- **semi-minor axis $\sin\alpha\,|n_z|$**, along it: a spot near the limb
  is foreshortened.

The visible part of a spot is bounded by the near side of its rim and,
for a spot on the limb, by the limb itself. Along the rim the cosine of
the emergent angle, $\mu$, varies linearly in the sine of the rim angle;
that is what keeps the light curve elementary with the hybrid laws (see
{doc}`../reference/how-it-works`).

## The star

The sky frame is the planet's: $X$ along its motion at mid-transit, $Y$
the sky projection of the orbit normal, and $Z$ towards the observer;
the planet transits at $Y = -b$. The star's spin axis is

$$
\hat s = (\sin i_\star \sin\lambda,\; \sin i_\star \cos\lambda,\; \cos i_\star),
$$

with $i_\star$ = `istar` its inclination to the line of sight and
$\lambda$ = `lam` the sky-projected obliquity, measured clockwise on the
sky from the projected orbit normal to the projected spin axis. This is
the convention of Fabrycky & Winn (2009), so a published
Rossiter–McLaughlin $\lambda$ can be used as it is.

The star turns about $\hat s$ with period `prot`, prograde: on an aligned
star ($\lambda = 0$, $i_\star = 90°$) its visible face moves along $+X$,
with the planet. A spot at stellar latitude $\varphi$ and longitude $L$
is at

$$
\hat n = \cos\varphi\,\big[\cos(L + \Omega t)\,\hat e_\mathrm{obs}
       + \sin(L + \Omega t)\,\hat e_\mathrm{east}\big] + \sin\varphi\,\hat s,
\qquad \Omega = \frac{2\pi}{P_\mathrm{rot}},
$$

where $t$ is the time since `spot_epoch`, $\hat e_\mathrm{obs}$ is the
meridian facing the observer and $\hat e_\mathrm{east} = \hat s \times
\hat e_\mathrm{obs}$. Longitude 0 faces the observer at `spot_epoch`, and
longitudes increase in the sense of rotation. Latitudes are positive
towards the pole $\hat s$ points to.

## The parameters

| parameter | definition | notes |
|---|---|---|
| `spots` | one row per spot: longitude, latitude, angular radius $\alpha$ (degrees), contrast | shape (K, 4); (n_sets, K, 4) in `light_curves` |
| `prot` | the rotation period | required; in the units of `t` |
| `istar` | the spin axis's inclination to the line of sight (degrees) | `None`: 90 |
| `lam` | the sky-projected obliquity (degrees) | `None`: 0 |
| `spot_epoch` | the time at which the longitudes are measured | `None`: `t0`; an absolute time, like `t0` |

$\alpha$ is the cap's angular radius on the star, not its radius on the
sky: a spot of angular radius $\alpha$ seen face-on has projected radius
$\sin\alpha$ in stellar radii, and covers a fraction $\sin^2\alpha$ of
the disc's area.

## Contrast and overlapping spots

A spot of contrast $c$ has intensity $c\,I(\mu)$, the photosphere's
times $c$, at every point: $c < 1$ is a dark spot, $c > 1$ a bright
facula and $c = 1$ no spot at all. Where spots overlap, their deficits
**add**: a point inside spots $j$ and $k$ has intensity
$[1 - (1 - c_j) - (1 - c_k)]\,I(\mu)$. Behind the planet the intensity
is 0, whatever the spots.

That makes a spot with an umbra two concentric spots. For a penumbra of
contrast $c_p$ around an umbra of contrast $c_u$, give the outer spot
contrast $c_p$ and the inner spot contrast $1 + c_u - c_p$. The rule
does not stop overlapping dark spots from summing to a negative
intensity (two spots of contrast 0.4 overlap at $-0.2$); keep spots
apart, or nest them as above.

## Normalisation

The flux is the spotted star's, occulted or not, divided by the
**unspotted** star's. So:

- with no planet on the disc it is the spots' rotational modulation,
  below 1 for dark spots;
- in transit the planet blocks the light of a star dimmed by its spots,
  and where it crosses a dark spot it blocks less than it would of the
  bare photosphere, so the flux rises.

To compare with data normalised to their out-of-transit level, fit a
normalisation factor; see {doc}`accuracy-and-fitting`.

## The domain

- $0 < \alpha \le 90°$ (a spot of 90° covers a hemisphere);
- $|\text{latitude}| \le 90°$;
- contrast $\ge 0$;
- every value finite.

`prot` must be positive, and the eccentricity at most 0.999, as for an
oblate planet. Outside these, a Python number or numpy array raises
`ValueError`, and an MLX array gives NaN. The number of
spots, K, is fixed when the model is built. A spot may be anywhere: on
the far side, rotating onto or off the disc, straddling the limb,
overlapping others, or covering the planet's whole chord.

## Symmetries

- **The modulation repeats every `prot`**, and the whole light curve,
  crossings included, whenever the elapsed time is a common multiple of
  `prot` and the orbital period.
- **Longitude and `spot_epoch` trade exactly**: moving `spot_epoch` later
  by $\Delta t$ is the same as subtracting $360°\,\Delta t/P_\mathrm{rot}$
  from every longitude. Fix one of them in a fit.
