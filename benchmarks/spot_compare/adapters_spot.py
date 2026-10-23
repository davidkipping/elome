"""Each code's spot-crossing model in the scenario's terms.

``make(code, case, res)`` returns (curve, value_and_grad):

* curve(x, y) -> relative flux at the planet's sky positions (numpy);
* value_and_grad(x, y) -> a scalar (the curve against fixed weights) and
  its gradient in the planet's radius and every spot's longitude,
  latitude, radius and contrast -- the parameters a fit of a crossing
  samples.

``res``: jaxoplanet's spherical-harmonic degree, spotter's HEALPix nside,
None for Elómë. JAX runs on the CPU (no Metal backend in this
environment), in float64. Elómë: "elome64" its float64 graph on the CPU,
"elome32" its float32 Metal kernels on the GPU.
"""
import math

import numpy as np

from scenario_spot import OOT, R_P, U_QUAD, W_LD, spots_rad

CODES = ("elome64", "elome32", "jaxoplanet", "spotter")


def _weights(n):
    return np.cos(np.linspace(0.0, 7.0, n))


# ---------------------------------------------------------------------------
# Elómë
# ---------------------------------------------------------------------------

def _elome(case, dt, kernel):
    import mlx.core as mx
    from elome.hybrid import combine_cols
    from elome.spots import shape_cols_spots
    from elome.spots_tau import spot_vectors
    stream = mx.gpu if kernel else mx.cpu
    sp = np.array(spots_rad(case))
    K = len(sp)

    def model(X, Y, r, lon, lat, alpha, c):
        n = spot_vectors(mx.array(0.0, dtype=dt), mx.array(10.0, dtype=dt),
                         mx.array(math.pi / 2, dtype=dt), mx.array(0.0, dtype=dt), lon, lat)
        shp = X.shape
        args = (X, Y, mx.broadcast_to(r, shp), mx.broadcast_to(n, shp + (K, 3)),
                mx.broadcast_to(alpha, shp + (K,)), mx.broadcast_to(c, shp + (K,)))
        if kernel:
            from elome.metal_spots import spot_cols_kernel
            B = spot_cols_kernel(*args, "hybrid2")
        else:
            B = shape_cols_spots(*args, "hybrid2")
        return 1.0 + combine_cols([B[..., k] for k in range(B.shape[-1])], [W_LD, 0.0], "hybrid2")

    def rel(X, Y, r, lon, lat, alpha, c):
        ox = mx.array([OOT[0]], dtype=dt)
        oy = mx.array([OOT[1]], dtype=dt)
        return model(X, Y, r, lon, lat, alpha, c) / model(ox, oy, r, lon, lat, alpha, c)

    p0 = [mx.array(v, dtype=dt) for v in (R_P, sp[:, 0], sp[:, 1], sp[:, 2], sp[:, 3])]

    def curve(x, y):
        with mx.stream(stream):
            out = rel(mx.array(x, dtype=dt), mx.array(y, dtype=dt), *p0)
            mx.eval(out)
            return np.array(out).astype(np.float64)

    def value_and_grad(x, y):
        with mx.stream(stream):
            X, Y = mx.array(x, dtype=dt), mx.array(y, dtype=dt)
            wt = mx.array(_weights(len(x)), dtype=dt)
            f = mx.value_and_grad(lambda *p: mx.sum(rel(X, Y, *p) * wt), argnums=(0, 1, 2, 3, 4))
            v, g = f(*p0)
            mx.eval(v, g)
            return float(v), np.concatenate([np.array(gi, dtype=np.float64).ravel() for gi in g])
    return curve, value_and_grad


# ---------------------------------------------------------------------------
# jaxoplanet (starry)
# ---------------------------------------------------------------------------

def _jax():
    import jax
    jax.config.update("jax_enable_x64", True)
    return jax


def _jaxoplanet(case, ydeg):
    jax = _jax()
    import jax.numpy as jnp
    from jaxoplanet.starry.light_curves import surface_light_curve
    from jaxoplanet.starry.surface import Surface
    from jaxoplanet.starry.ylm import Ylm, ylm_spot
    spot = ylm_spot(ydeg)
    sp = np.array(spots_rad(case))

    def surface(lon, lat, alpha, c):
        # each expansion is 1 plus the spot's (negative) profile; the
        # deficits add, as in the other codes
        y = sum(spot(1.0 - c[k], alpha[k], lat[k], lon[k]).todense() for k in range(len(sp)))
        y = y.at[0].add(-(len(sp) - 1.0))
        return Surface(y=Ylm.from_dense(y, normalize=False), u=U_QUAD,
                       inc=math.pi / 2, obl=0.0)

    def rel(x, y, r, lon, lat, alpha, c):
        s = surface(lon, lat, alpha, c)
        f = jax.vmap(lambda xi, yi: surface_light_curve(s, r, xi, yi, 1.0, 0.0))(x, y)
        return f / surface_light_curve(s, r, OOT[0], OOT[1], 1.0, 0.0)

    p0 = (jnp.asarray(R_P), *(jnp.asarray(sp[:, i]) for i in range(4)))
    f_curve = jax.jit(lambda x, y: rel(x, y, *p0))

    def curve(x, y):
        return np.asarray(jax.block_until_ready(f_curve(jnp.asarray(x), jnp.asarray(y))))

    wt = None
    f_vg = jax.jit(jax.value_and_grad(
        lambda p, x, y, w: jnp.sum(rel(x, y, *p) * w)))

    def value_and_grad(x, y):
        nonlocal wt
        if wt is None or wt.shape[0] != len(x):
            wt = jnp.asarray(_weights(len(x)))
        v, g = jax.block_until_ready(f_vg(p0, jnp.asarray(x), jnp.asarray(y), wt))
        return float(v), np.concatenate([np.ravel(np.asarray(gi)) for gi in g])
    return curve, value_and_grad


# ---------------------------------------------------------------------------
# spotter
# ---------------------------------------------------------------------------

# spotter's transit design matrix is (times x pixels): evaluate the times
# in chunks so nside 256 (786,432 pixels) stays near 1 GB
_SPOTTER_CHUNK = 128


def _spotter(case, nside):
    jax = _jax()
    import jax.numpy as jnp
    from spotter import core
    from spotter.light_curves import transit_light_curve
    from spotter.star import Star
    sp = np.array(spots_rad(case))

    def star(lon, lat, alpha, c):
        y = 1.0 - sum((1.0 - c[k]) * core.spot(nside, lat[k], lon[k], alpha[k])
                      for k in range(len(sp)))
        # obl=None, not 0.0: spotter 0.0.11 turns the occultation by 1 rad
        # when the obliquity is exactly 0 (see README)
        return Star(y=y, u=jnp.asarray(U_QUAD), inc=math.pi / 2, obl=None, period=None)

    def rel(x, y, r, lon, lat, alpha, c):
        s = star(lon, lat, alpha, c)
        f = transit_light_curve(s, x, y, jnp.ones_like(x), r, 0.0).ravel()
        o = transit_light_curve(s, jnp.asarray([OOT[0]]), jnp.asarray([OOT[1]]),
                                jnp.ones(1), r, 0.0).ravel()
        return f / o[0]

    p0 = (jnp.asarray(R_P), *(jnp.asarray(sp[:, i]) for i in range(4)))
    f_curve = jax.jit(lambda x, y: rel(x, y, *p0))
    f_vg = jax.jit(jax.value_and_grad(lambda p, x, y, w: jnp.sum(rel(x, y, *p) * w)))

    def chunks(n):
        return [(i, min(i + _SPOTTER_CHUNK, n)) for i in range(0, n, _SPOTTER_CHUNK)]

    # every chunk padded to one shape (unocculted positions, weight 0): one
    # compilation
    def curve(x, y):
        out = []
        for i, j in chunks(len(x)):
            xc = np.full(_SPOTTER_CHUNK, OOT[0])
            yc = np.full(_SPOTTER_CHUNK, OOT[1])
            xc[:j - i], yc[:j - i] = x[i:j], y[i:j]
            out.append(np.asarray(jax.block_until_ready(
                f_curve(jnp.asarray(xc), jnp.asarray(yc))))[:j - i])
        return np.concatenate(out)

    def value_and_grad(x, y):
        w = _weights(len(x))
        v, g = 0.0, None
        for i, j in chunks(len(x)):
            xc = np.full(_SPOTTER_CHUNK, OOT[0])
            yc = np.full(_SPOTTER_CHUNK, OOT[1])
            wc = np.zeros(_SPOTTER_CHUNK)
            xc[:j - i], yc[:j - i], wc[:j - i] = x[i:j], y[i:j], w[i:j]
            vi, gi = jax.block_until_ready(
                f_vg(p0, jnp.asarray(xc), jnp.asarray(yc), jnp.asarray(wc)))
            gi = np.concatenate([np.ravel(np.asarray(t)) for t in gi])
            v, g = v + float(vi), (gi if g is None else g + gi)
        return v, g
    return curve, value_and_grad


def make(code, case, res=None):
    if code == "elome64":
        import mlx.core as mx
        return _elome(case, mx.float64, kernel=False)
    if code == "elome32":
        import mlx.core as mx
        return _elome(case, mx.float32, kernel=True)
    if code == "jaxoplanet":
        return _jaxoplanet(case, int(res))
    if code == "spotter":
        return _spotter(case, int(res))
    raise ValueError(code)
