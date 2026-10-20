# Geometry

## The ring on the sky

A ring is a flat circle of radius $R$ in the planet's equatorial plane,
seen at an inclination $i_\mathrm{ring}$ to the sky plane. On the sky it
is an ellipse:

- **semi-major axis $R$**, along the line where the ring plane meets the
  sky plane, which projection does not shorten;
- **semi-minor axis $R\,|\cos i_\mathrm{ring}|$**;
- **long axis at the projected angle `theta`**.

The ring's two true orientation angles, its inclination and its azimuth,
become two projected numbers: `theta` and the axis ratio. Both ring edges
share them, since they lie in one plane, and they are concentric with the
planet.

## The parameters

| parameter | definition | notes |
|---|---|---|
| `ring_in`, `ring_out` | the ring's inner and outer radii, in units of `rp` | the projected semi-major axes of its edges; `ring_in` must reach past the planet's equator |
| `ring_f` | $1 - \lvert\cos i_\mathrm{ring}\rvert$, the projected flattening of both edges | 0 face-on, $\to 1$ edge-on; $0 \le$ `ring_f` $< 1$ |
| `ring_tau` | the normal optical depth | `None` or $\infty$ is opaque; $\ge 0$ |
| `theta` | the angle from the direction of motion at mid-transit to the ring's long axis | degrees in `TransitParams`, radians in the array API; shared with an oblate planet |

In stellar radii the edges are ellipses with semi-axes

$$
A_k = r_k\, r_p, \qquad B_k = A_k\,(1 - \mathtt{ring\_f}), \qquad
r_k \in \{\mathtt{ring\_in}, \mathtt{ring\_out}\}.
$$

The planet is the circle of radius `rp`, or, if `f` is set, the oblate
planet's ellipse (see {doc}`../oblate/geometry`), with its long axis
along the ring's.

**A uniform prior on `ring_f` is the isotropic prior** on the ring's
orientation, since $|\cos i_\mathrm{ring}|$ is uniform for random
orientations.

## Opacity

Light crossing the ring travels a slant path $1/|\cos i_\mathrm{ring}|$
times the normal one, so the ring blocks the fraction

$$
T = 1 - \exp\!\left(-\frac{\tau}{1 - \mathtt{ring\_f}}\right)
$$

of the starlight behind it. The planet blocks all of it, so whether the
ring passes in front of the planet or behind it makes no difference. The
light curve is linear in $T$.

## Symmetries

- **`theta` has period $\pi$**, as for an oblate planet: the ring is
  unchanged by a half turn.
- **On a central transit, `theta` and $\pi -$ `theta` agree**, by
  reflection across the chord.
- **`ring_tau` and `ring_f` trade against each other** through $T$ on a
  nearly edge-on ring, where the ring covers little area; see
  {doc}`accuracy-and-fitting`.

## The domain

- $0 \le$ `ring_f` $< 1$, `ring_tau` $\ge 0$, `ring_out` $>$ `ring_in`;
- `ring_in` $\ge 1/\sqrt{1-f}$: the ring's inner edge lies outside the
  planet's equator (`ring_in` $\ge 1$ for a spherical planet).

Outside it a Python number or numpy array raises `ValueError`, and an MLX
array gives NaN. Unlike an oblate planet alone, a ringed planet has no
size limit: an outline may cross the stellar limb four times (a nearly
edge-on ring wider than its distance to the limb), or cover the whole
star.

An equatorial ring also implies $f \le$ `ring_f` for the planet's own
projected flattening (a planet seen pole-on looks round). This is not
enforced.
