# Limb darkening

A star is darker towards its limb. Write $\mu = \cos\gamma$ for the
cosine of the angle between the line of sight and the stellar surface
normal ($\mu = 1$ at the centre of the disc, $\mu = 0$ at the limb). A
limb-darkening law gives the intensity $I(\mu)$. Set the law with
`params.limb_dark` and its coefficients with `params.u`.

## The laws

| `limb_dark` | `u` | intensity $I(\mu)/I(1)$ |
|---|---|---|
| `"uniform"` | `[]` | $1$ |
| `"linear"` | `[u1]` | $1 - u_1(1-\mu)$ |
| `"quadratic"` | `[u1, u2]` | $1 - u_1(1-\mu) - u_2(1-\mu)^2$ |
| `"polynomial"` | `[u1, ..., uN]`, any $N$ | $1 - \sum_n u_n (1-\mu)^n$ |
| `"hybrid2"` | 2 weights | $1 - \sum_j w_j S_j(\mu)$ |
| `"hybrid4"` | 4 weights | $1 - \sum_j w_j S_j(\mu)$ |
| `"hybrid5"` | 5 weights | $1 - \sum_j w_j S_j(\mu)$ |

Every law has an exact, closed-form light curve and analytic gradients in
its coefficients. The coefficients may change between calls on a built
model, and they can be sampled.

## Which law to choose

- **Quadratic** is the standard choice and has well-known prior
  machinery (below). Most published coefficient tables are for it.
- **Polynomial** extends the quadratic law to any order. It is exact to
  $10^{-15}$ at $N = 8$ and $2\times10^{-13}$ at $N = 16$ against a
  40-digit direct integral.
- **The hybrid laws** follow a real star's profile more closely. They
  combine even powers of $\mu$ with *pole* terms
  $(\mu^2 + \epsilon)^{-2}$ that steepen at the limb, as real profiles do.
  Over 8,100 combinations of stellar model and passband (PHOENIX and
  MPS-ATLAS), the median light-curve error is 1.8 ppm for `hybrid4` and
  1.1 ppm for `hybrid5`, against 2.0 ppm for the four-parameter Claret
  law.[^coeffs] For a spherical
  planet `hybrid2` and `hybrid4` are also *cheaper* than the quadratic
  law, and `hybrid5` costs about the same: no term needs an elliptic
  integral (see {ref}`law-cost`). They are also the only laws available for an
  {doc}`oblate planet <../oblate/index>`.

The definitions of the hybrid laws are in {doc}`../reference/hybrid-laws`.

[^coeffs]: Hybrid-law limb-darkening coefficients paper (in
    preparation); see {doc}`../about/citing`.

### Shape weights

The hybrid laws are written in a *shape basis*. Each shape $S_j(\mu)$
falls from 1 at the limb to 0 at the centre, so its weight $w_j$ is
simply the share of the centre-to-limb drop that the shape carries:

$$
\frac{I(\mu)}{I(1)} = 1 - \sum_j w_j S_j(\mu), \qquad
\frac{I(0)}{I(1)} = 1 - \sum_j w_j .
$$

For `hybrid2` and `hybrid5` the shapes are the law's terms (powers of
$\mu^2$ and normalised poles). For `hybrid4` they are its *curvature
basis*: $S_1 = 1 - \mu^2$, $S_2 = (1 - \mu^2)^2$, and two pole-dominated
shapes, each a fixed combination of its terms. In that basis the simplex
$w_j \ge 0$, $\sum_j w_j \le 1$ is *exactly* the set of `hybrid4`
profiles that are non-negative, darker towards the limb, concave in
$\mu^2$ and of convex slope in $\mu^2$: every point of it is such a
profile, and every such profile is in it. A physical profile that is
convex in $\mu^2$ lies outside it. For `hybrid5` the same simplex
guarantees a non-negative profile that darkens towards the limb. For
`hybrid2` the physical region is an exact triangle.

## Priors on the coefficients

Sampling the coefficients directly lets chains wander into unphysical
profiles. Each law has a map from the unit cube, where a uniform prior
is easy, onto a region of physical profiles, uniformly:

| law | map | inverse |
|---|---|---|
| quadratic | `q_to_u(q1, q2)` (Kipping 2013) | `u_to_q_np` |
| hybrid2 | `hybrid2_from_q(q1, q2)`, the exact triangle | `hybrid2_to_q_np` |
| hybrid4 | `hybrid4_from_q(q)`, the exact curvature simplex | `hybrid4_to_q_np` |
| hybrid5 | `simplex_from_q(q)`, stick-breaking onto the simplex | `q_from_simplex_np` |

For `hybrid4`, $q_1$–$q_4$ are the direct analogue of Kipping's $q_1$,
$q_2$: $q_1 = (1 - I(0))^4$ sets the centre-to-limb drop, and the others
divide it among the four shapes.

Each map has an MLX version, which is differentiable and suited to
sampling, and a numpy version with the `_np` suffix:

```python
from elome import TransitParams, TransitModel, simplex_from_q_np

p = TransitParams()
p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 87.0, 0.0, 90.0
p.limb_dark = "hybrid5"
p.u = simplex_from_q_np([0.3, 0.5, 0.2, 0.6, 0.4])   # 5 weights from the unit cube
flux = TransitModel(p, t).light_curve(p)
```

## Limb darkening as a linear block

Every law here is linear in its intensity coefficients, so a light curve
can be written as a ratio of two linear forms in the coefficients:
$F - 1 = (B\,c)/(N\,c)$, where the basis $B$ depends on the geometry
alone. A sampler can then integrate the limb darkening out analytically,
or sample it in a Gibbs step, with one kernel launch per geometry.
`ld_basis=True` returns $B$; see {ref}`ld-basis`.
