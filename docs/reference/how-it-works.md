# How it works

A short tour of the numerical methods, for readers who want to know why
the results can be trusted.

## The spherical light curve

The occulted flux of a limb-darkened star is computed with the
closed-form solution of Agol, Luger & Foreman-Mackey (2020): the
intensity is expanded in Green's basis, and each basis term's occulted
flux is a closed form in the planet's radius $r$ and its sky separation
$z$ from the star's centre. The quadratic law's linear term needs
complete elliptic integrals, which Elómë evaluates with Bulirsch's
algorithm at a fixed iteration count. The polynomial law of any order
uses the same solution's recursion.

- **Regimes.** No overlap, partial overlap, a planet fully inside the
  disc, and (for `rp > 1`) total occultation are separate formulas.
  Elómë computes them all and selects with masks, with every branch's
  arguments made safe on the inactive side. A clamped-but-tiny
  denominator would give a finite value and an infinite derivative,
  which turns into NaN in the backward pass.
- **Degenerate lines.** Two narrow Taylor switches, $|z - r| < 10\epsilon$
  and $|z + r - 1| < \sqrt{\epsilon}$, handle the only places where the
  closed forms lose precision. The analytic gradients need no switches.
- **Fixed iteration counts everywhere**, with no data-dependent control
  flow, so every function can be compiled and batched.

## Gradients

The photometric core has analytic derivatives with respect to $r$, $z$
and the limb-darkening coefficients, written as MLX custom functions.
They avoid reverse-mode differentiation's memory-bound backward pass,
and are 50–570 times faster depending on the path.

## Orbits

- **Circular orbits** use the phase measured from mid-transit, computed in
  float64 on the host as an orbit number and a residual
  (`epoch_center_times`), so float32 never sees a large number.
- **Eccentric orbits.** Kepler's equation is solved by Markley's
  non-iterative method with one fifth-order refinement: exact to
  rounding, branch-free, and differentiable through an implicit
  derivative. The separation comes from the orbital-plane Cartesian
  coordinates $X = a(\cos E - e)$, $Y = a\sqrt{1-e^2}\sin E$, never from
  the true anomaly, which avoids cancellations near transit and at
  apastron.
- **Transit-anchored.** The solve is made for the eccentric anomaly's
  offset from inferior conjunction rather than for the anomaly itself,
  and it uses only $e\cos\omega$ and $e\sin\omega$, taken from the sampled
  $(\sqrt{e}\cos\omega, \sqrt{e}\sin\omega)$. The textbook form divides
  by $e$ in its derivatives; in float32 its gradients are 5% wrong at
  $e = 10^{-5}$ and entirely wrong at $10^{-8}$. The anchored form is
  algebraically identical (to $2.5\times10^{-15}$ in flux) and accurate to
  about $10^{-7}$ throughout.

## The GPU kernels

The float32 path fuses the whole model, orbit and photometry, into one
hand-written Metal kernel per call, keeping every intermediate in
registers. The gradient kernel evaluates the analytic derivatives in the
same pass and reduces them per chain. A circular orbit is the $e = 0$
case of the eccentric kernel, with the Kepler solve skipped per chain.
Exposure integration runs inside the kernel, per output point.

The float64 path computes the same function as an MLX graph on the CPU.
It shares the kernels' choices, such as where exposures are split, so
that it serves as their exact reference.

## The hybrid laws and oblate planets

For a spherical planet, the hybrid laws' terms have short closed forms
in the same quantities as the quadratic law, without elliptic integrals;
see {doc}`hybrid-laws`.

For an oblate planet, the occulted flux is a line integral around the
boundary of the overlap, which consists of arcs of the ellipse and of
the stellar limb. Along the ellipse, the even terms integrate to finite
Fourier sums, and each pole term to a sum over the roots of a quartic,
one quartic per pole and point. Where the ellipse crosses the limb is
another quartic. The GPU kernel seeds each quartic's roots from the
closed-form (Ferrari) solution, polishes them with Aberth and Newton
iterations, and checks them against the residual and Vieta's relations.
For a nearly circular planet, where the closed form loses precision in
float32 (its leading coefficient is proportional to the flattening),
and for any quartic that fails the check, the seeds come instead from
its own small eigenvalue solver (Metal has no linear algebra). The
roots come in pairs $(z, 1/\bar z)$, and the term of a root well outside
the unit circle is evaluated through its partner inside it. That is
exact and better conditioned; for a nearly circular planet one root is
of order $2/(A f)$, and its derivatives taken directly would overflow
float32. The contact times for exposure integration are solved per
side with a monotone Newton iteration on convex distance functions, and
a separate kernel computes them.

## Ringed planets

A ring makes the occultor a region of piecewise-constant opacity: 1
inside the planet, the ring's blocked fraction $T$ between its edges, 0
elsewhere. Green's theorem then gives the occulted flux as a sum over
every outline -- the planet's and the two ring edges' -- of its arcs on
the star, each weighted by the jump in opacity across it, plus the limb
arcs weighted by the opacity there. Each outline is walked once. Each
arc is the same integral as an oblate planet's, so the even terms are
Fourier sums and each pole term is a partial-fraction sum over the roots
of a quartic. The arcs end where an outline crosses the limb (up to four
times) or meets another outline. Because the outlines are concentric and
coaxial, those corners have a closed form, and which side of another
outline an arc lies on follows from it rather than from a point test,
which would be fragile at a tangency. A circular outline (a spherical
planet, a face-on ring) has a cubic for a quartic: its roots come from
the quadratic formula, and a circle centred on the star uses a
first-order expansion in its offset. Gradients come from the same
frozen-endpoint scheme as for an oblate planet, with a corner term at
every vertex weighted by the potential there. The float32 kernels
evaluate each outline with its own dual numbers and chain the results to
the ring parameters by hand. A separate kernel finds the moments each
outline touches the limb, which are the exposure rule's split points.

## Starspots

A spotted star is the same weighted-outline problem with the weights on
the star rather than on the occultor: the blocked fraction is 1 inside
the planet and $\sum_k (1 - c_k)$ over the visible spots $k$ elsewhere.
Each spot's rim is a circle on the sphere, which projects to an ellipse
with its own frame, so its arcs are again the oblate integrals. Along
the rim $\mu$ is linear in the sine of the rim angle, because the rim
lies on the sphere. A pole's quartic then splits into two quadratics with
root product $-1$, whose roots have a closed form, and the rim meets the
limb where $\mu = 0$, also in closed form. The latter is written as a
product so that a small spot on the limb, whose rim and limb arcs cancel
to a sliver, keeps its digits. The planet's corners with a rim come from
the ring code's crossing solver, in a frame centred on the planet and
scaled by its radius. Where two rims cross, nothing needs splitting:
the spots' deficits add, so the jump in the blocked fraction across a
rim is its own spot's $1 - c_k$, whichever other spots cover it. Which
spots a point belongs to is read at each arc's midpoint, from the point
on the star's front surface.

A rim's frame turns with its spot, as $1/|n_{xy}|$ near the disc's
centre. The gradients therefore take the frame's turn separately, as a
corner term multiplying a difference formed without cancellation. They
take the ellipse's $(a^2 - b^2)/4$ in its form on the sphere,
$\sin^2\!\alpha\,|n_{xy}|^2/4$, and stay finite at the centre itself.
The float32 kernels evaluate the columns per point, with dual numbers
seeded in the planet's position and radius and in each spot's centre
and radius; the orbit and the star's rotation stay graph operations, so
their gradients chain automatically. The contact rule's split points
depend on the rotating star and so differ between transits. A separate
kernel finds them exposure by exposure, and graph operations give them
their implicit derivatives.

## MLX caveats

- `mx.sin`, `mx.cos`, `mx.exp` and `mx.erf` are only float32-accurate on
  float64 arrays in MLX 0.32, so Elómë uses its own float64 `sincos`
  (Cody–Waite reduction, accurate to $10^{-16}$).
- MLX operations on Python numbers create float32 scalars, so where a
  parameter may be a Python number Elómë does the arithmetic in Python
  first.
