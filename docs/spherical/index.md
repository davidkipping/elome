# Spherical planets: overview

A spherical planet is the standard transit model: a dark disc crossing a
limb-darkened star. Everything Elómë offers is available for it:

| | |
|---|---|
| limb darkening | uniform, linear, quadratic, polynomial of any order, and the hybrid laws |
| orbits | circular and eccentric |
| transit types | primary transits and secondary eclipses |
| finite exposures | supersampling, or the more accurate contact rule |
| sizes | any `rp`, including companions larger than their star |
| precision | float64 (exact, CPU) and float32 (GPU kernels) |
| gradients | analytic, in every parameter |

## Reading order

1. {doc}`light-curves`: the batman-style frontend, orbits and secondary
   eclipses.
2. {doc}`limb-darkening`: which law to choose, and priors on its
   coefficients.
3. {doc}`../guides/exposures`: long-cadence data needs exposure
   integration.
4. {doc}`fitting`: sampling the posterior with anvil, emcee or your
   own sampler.
5. {doc}`accuracy-and-cost`: against batman, PyTransit, jaxoplanet and
   exoplanet-core.
6. If you write your own likelihood:
   {doc}`../guides/batching`, {doc}`../guides/gradients-and-precision`
   and {doc}`../guides/sampler-api`.

Is your companion bigger than its star, such as a planet around a white
dwarf? Read {doc}`large-occultors` too.

The first tutorial, {doc}`../tutorials/01_elome_with_anvil`, walks
through all of this on a Kepler-like light curve.
