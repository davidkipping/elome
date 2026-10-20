# Geometry

```{image} ../assets/oblate_geometry_light.svg
:alt: An elliptical planet crossing a limb-darkened star, with its semi-axes A and B, the long axis at angle theta from the direction of motion, and the impact parameter b.
:class: only-light
:width: 560px
:align: center
```

```{image} ../assets/oblate_geometry_dark.svg
:alt: An elliptical planet crossing a limb-darkened star, with its semi-axes A and B, the long axis at angle theta from the direction of motion, and the impact parameter b.
:class: only-dark
:width: 560px
:align: center
```

## The parameters

The planet's projection is an ellipse with semi-axes $A \ge B$ (in
stellar radii). Elómë describes it by:

| parameter | definition | notes |
|---|---|---|
| `rp` | $\sqrt{AB}$ | the **area-equivalent radius**: a circle of radius `rp` has the ellipse's area |
| `f` | $1 - B/A$ | the projected flattening, $0 \le f < 1$ |
| `theta` | the angle from the direction of motion at mid-transit to the long axis | degrees in `TransitParams`, radians in the array API |

so that

$$
A = \frac{r_p}{\sqrt{1-f}}, \qquad B = A\,(1-f) = r_p\sqrt{1-f}.
$$

Because `rp` is the area-equivalent radius, the transit depth stays close
to $r_p^2$ whatever the flattening, and `rp` keeps the meaning it has for
a spherical planet.

`theta` is measured from the direction in which the planet moves across
the star at mid-transit, positive towards the star's centre:
anticlockwise in the figure, where the planet crosses below the centre
($b > 0$). The axis is fixed on the sky for the whole transit, as a
planet's spin axis is, even though a curved orbit's direction of motion
turns slightly. The
orbit is the same as for a spherical planet: `t0`, `per`, `a`, `inc` (or
`b`), and `ecc`, `w` if eccentric, with $e \le 0.999$ (a spherical
planet takes any $e < 1$).

## Symmetries

- **`theta` has period $\pi$.** An ellipse is unchanged by a half turn,
  so `theta` and `theta + 180°` give identical light curves.
- **On a central transit, `theta` and $\pi -$ `theta` agree.** At $b = 0$
  the chord is a line of symmetry of the star, and reflecting across it
  maps `theta` to $\pi -$ `theta`. The two light curves agree to rounding
  at $b = 0$ and differ by $2\times10^{-5}$ at $b = 0.05$.
- **`b` and $-$`b` mirror `theta`.** A transit at $-b$ (`inc` above 90°)
  with $-$`theta` is the mirror image of one at $b$ with `theta`, and
  the light curves are identical. Which side of the star the planet
  crosses cannot be seen, so the data fix the sign of `theta` only
  relative to the sign of `b`.
- **`f` and `theta` trade against `b` and `rp`.** On near-central
  transits an aligned oblate planet and a round one at slightly different
  $b$ can look alike. That is a property of the data, and matters for
  sampling; see {doc}`fitting`.

## The domain

Elómë requires

$$
r_p \le (1-f)^{3/2},
$$

which keeps the planet small enough that its outline crosses the stellar
limb at most twice. The limit is 0.59 at $f = 0.3$ and 0.35 at $f = 0.5$.
Outside it, a Python number or numpy array raises `ValueError`, and an
MLX array (inside a traced or compiled function, where nothing can be
raised) gives NaN.

## Spherical as a special case

- `f = None`, the default, is a spherical planet. It takes the spherical
  code paths, unchanged.
- `f = 0` is a spherical planet through the oblate machinery. It agrees
  with the spherical result to $2\times10^{-17}$.
- In float32, a flattening below $10^{-5}$ is evaluated with the spherical
  closed forms, since it moves the flux by less than $4\times10^{-8}$.
  The derivative with respect to `f` is still the oblate one; see
  {doc}`accuracy-and-cost`.
