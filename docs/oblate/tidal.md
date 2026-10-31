# Tidally locked planets

The oblate mode holds one ellipse on the sky for the whole transit. That
suits a planet flattened by its rotation. A close-in, tidally locked planet
is different: it is stretched towards the star into a triaxial ellipsoid
whose long axis follows the star around the orbit. Its outline is smallest
at mid-transit, when that axis points at you, and grows towards the limb.
The triaxial mode models this, with the outline recomputed at every instant
and at every exposure node.

## The model

One shape parameter $q$ sets the three semi-axes (Correia 2014): $r_1$, the
nose, which points at the star at every instant on any orbit; $r_2$ along
the orbit; and $r_3$ along the orbit normal,

$$
r_1 = r_2(1 + 3q), \qquad r_3 = r_2(1 - q),
\qquad R_V^3 = r_1 r_2 r_3 .
$$

This is the `love` shape of `ellc` (Maxted 2016), built to agree with it term
for term; see {ref}`tidal-ellc`.

The radius you pass is the **volumetric** radius $R_V$, not the
area-equivalent radius of the oblate mode. At $q = 0.05$ the two differ by
about 5%, so the same number means a different planet in the two modes.

The nose fixes one axis, so one freedom remains: `theta` turns $r_2$ and
$r_3$ about the nose by a fixed angle $\vartheta$, default 0. At conjunction
on an edge-on orbit it is the sky angle of the outline's long axis, the
oblate mode's `theta`. The twist stays fixed relative to the orbit: it
follows the star, it does not hold a direction on the sky.

An ellipsoid always projects to an ellipse, so each instant is an oblate
problem with its own $r$, $f$ and sky angle. With $s, c = \sin\nu, \cos\nu$
and $\nu$ the angle along the orbit from conjunction, which is the orbital
phase on a circular orbit, the outline's shape matrix on the sky is

$$
S = r_1^2\, e_1 e_1^T + r_2^2\, e_2' e_2'^T + r_3^2\, e_3' e_3'^T,
$$

$$
e_1 = (-s,\ c\cos i), \quad e_2 = (c,\ s\cos i), \quad e_3 = (0,\ \sin i),
$$

$$
e_2' = e_2\cos\vartheta + e_3\sin\vartheta, \quad
e_3' = -e_2\sin\vartheta + e_3\cos\vartheta,
$$

using the sky components of the axes. Its eigenvalues are the squared
semi-axes of the outline.

## With the frontend

Set `tidal_q` in place of `f`:

```python
import numpy as np
import elome

p = elome.TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 0.925, 0.1215, 2.99, 88.5, 0.0, 90.0
p.limb_dark, p.u = "hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1]
p.tidal_q = 0.05                       # p.rp is now the volumetric radius
p.theta = 10.0                         # twist about the nose in degrees; optional

t = np.linspace(-0.075, 0.075, 2000)
flux = elome.TransitModel(p, t, exp_time=60.0 / 86400.0,
                          integration="contact").light_curve(p)
```

## With the sampler API

`flux_dev_from_tau` takes the same parameters, scalar or per chain, with
`theta` in radians here as for an oblate planet, and returns gradients in
them and in everything else:

```python
import mlx.core as mx
import elome

tau = mx.array(np.linspace(-0.075, 0.075, 2000), dtype=mx.float32)
q = mx.array([0.05, 0.0, -0.02], dtype=mx.float32)     # three chains
dev = elome.flux_dev_from_tau(tau, 0.925, 2.99, 0.08, 0.1215,
                              limb_dark="hybrid5",
                              u=[0.2, 0.2, 0.1, 0.1, 0.1],
                              exp_time=60.0 / 86400.0, integration="contact",
                              tidal_q=q, theta=0.2,
                              secosw=0.1, sesinw=0.2)
```

`elome.contact_offsets_tidal(a, b, rv, q, theta)` returns the four contact
phases, and takes `k`, `h` and `mass_ratio`.

## Shape, mass ratio and Love number

`tidal_q` always sets the shape. Two other quantities are separate from it:

- **`mass_ratio`**, $M_p/M_\star$, is an optional input to the model, and only
  adds the small offset described below.
- **$h_f$** is not an input to the model. It is a conversion between $q$ and
  $h_f$ that always needs a mass ratio, whether or not the model used one.

So there are three ways to set up a call:

| Inputs | Model | $h_f$ |
|---|---|---|
| `tidal_q` | shape only | `tidal_hf_from_q`, with $\mu$ from elsewhere |
| `tidal_q`, `mass_ratio` | shape and offset | `tidal_hf_from_q` |
| $h_f$ and $\mu$, through `tidal_q_from_hf` | shape and offset | given |

The last, for MLX arrays, is differentiable in $h_f$ and $\mu$:

```python
hf, mu = mx.array(1.5, dtype=mx.float32), mx.array(1.2e-3, dtype=mx.float32)
a, rv = mx.array(2.99, dtype=mx.float32), mx.array(0.1215, dtype=mx.float32)
dev = elome.flux_dev_from_tau(tau, 0.925, a, 0.08, rv, limb_dark="hybrid5",
                              u=[0.2, 0.2, 0.1, 0.1, 0.1],
                              tidal_q=elome.tidal_q_from_hf(hf, mu, a, rv),
                              mass_ratio=mu)
```

## What $q$ can be

$-1/3 < q < 1$. **A negative $q$ is allowed**, so the sphere is an ordinary
point of the domain and not its edge. The light curve and its gradients are
smooth through $q = 0$, where the mode is the spherical path.

The longest axis squared may not exceed the shortest, the oblate domain at
every phase. A value outside the domain raises an error for host numbers and
gives NaN for arrays, as in the oblate mode.

## Symmetry

On a circular orbit without a twist, the outline tilts by an angle that is
antisymmetric about mid-transit and vanishes at $b = 0$. The phase $-\phi$
is then the mirror image of $+\phi$, so the light curve is exactly
symmetric in time and the tilt does not make ingress and egress differ. A
twist or an eccentric orbit breaks that symmetry. A half-turn of the twist
gives the same ellipsoid.

(tidal-ellc)=
## Agreement with `ellc`

`tidal_q` is $q$ at the separation $a$, and the same model as `ellc`'s `love`
shape follows from it:

- **The shape follows the separation at fixed volume.** With $r$ the
  star-planet distance and $\mu = M_p/M_\star$,

  $$
  q(1 + 3q)(1 - q) = \frac{h_f}{2\mu}\frac{R_V^3}{r^3},
  $$

  which is `ellc`'s $q = h_f r_2^3 / (2\mu r^3)$ with $r_2$ taken from the
  volume. On a circular orbit $q$ is constant, and on an eccentric one it is
  solved at every instant.
- **`elome.tidal_q_from_hf(hf, mu, a, rv)`** gives $q$ from $h_f$ at $r = a$,
  and `elome.tidal_hf_from_q` the reverse. The commonly printed first-order
  relation $h_f = 2q\,\mu\,(a/R_V)^3$ uses $R_V$ where this one uses $r_2$. The
  two differ by $(1+3q)(1-q)$.
- **`mass_ratio` is optional**, and is needed only for the third-order tide:
  `ellc` moves the ellipsoid's centre towards the star by
  $D = r_2^5/(\mu a^4)$, with $\Delta_3 = 1$ and no dependence on $r$. That moves
  the planet's barycentric position, so the sky position scales by
  $1 - D/((1+\mu)r)$. On a circular orbit this is exactly the same model at
  $a \to a - D/(1+\mu)$ with the same inclination, so `mass_ratio=None` fits
  the same light curves with a slightly different $a$.

Three conventions differ but do not change the model:

- `ellc`'s `t_zero` on an inclined eccentric orbit is the time of least sky
  separation; `t0` here is inferior conjunction.
- Its exposure rule is a trapezium over `n_int` points. The contact rule
  converges to the same integral.
- Its volume is converged to 1e-6, where here it is exact, and its flux comes
  from a numerical integration over a finite grid.

`ellc` takes power-2 limb darkening among others, and this mode takes the hybrid
laws; `ellc` also accepts a tabulated intensity profile, which can carry a
hybrid law. `ellc` has no twist, so `theta` is 0 in any comparison.

## Limits

- Primary transits with a hybrid law, as for any oblate planet in transit.
- One shape parameter: the 3 : −1 split between the axes is the model's
  assumption, not something it fits.
- On an eccentric orbit the nose stays on the star however fast the planet
  would have to turn to keep it there, and the shape responds at once to the
  separation. Both are taken as given.
- $q(1+3q)(1-q)$ scaled to periastron must stay between −0.09 and 0.6.

## Accuracy and cost

The light curve agrees with an independent quadrature of the projected
ellipse at each phase, built from the 3-D ellipsoid, and fp32 stays within
0.15 ppm of fp64. A call costs about what an oblate one of the same
flattening costs. See {doc}`accuracy-and-cost` for the oblate figures, which
apply here.
