"""A planet with a moon through TransitModel (params.moon_r): the light curve
against the photometry at independently computed positions, the planet
alone as the moon vanishes, exposures, gradients, float32 far from t0, and
the parameter checks."""

import copy
import math

import mlx.core as mx
import numpy as np
import pytest

import elome
import test_moons as TM
from elome import moons

HYB = ("hybrid5", [0.2, 0.15, 0.1, 0.05, 0.1])
QUAD = ("quadratic", [0.4, 0.25])
EXP = 30.0 / 1440.0


def system(law=HYB, ecc=0.1, moon_ecc=0.05, **over):
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.2, 30.0, 0.08, 60.0, 89.7, ecc, 70.0
    p.limb_dark, p.u = law
    p.moon_r, p.moon_a, p.moon_per, p.moon_q = 0.35, 6.0, 0.9, 0.02
    p.moon_inc, p.moon_node, p.moon_phase = 89.0, 10.0, 40.0
    p.moon_ecc, p.moon_w = moon_ecc, 30.0
    for k, v in over.items():
        setattr(p, k, v)
    return p


def windows(p, epochs=(0, 3), half=0.4, step=2.0 / 1440.0):
    return np.concatenate([p.t0 + k * p.per + np.arange(-half, half, step) for k in epochs])


def reference(p, t):
    """The photometry at moons.py-independent positions (rotation matrices)."""
    d = dict(t0=p.t0, per=p.per, a=p.a, inc=math.radians(p.inc), ecc=p.ecc,
             w=math.radians(p.w), a_ps=p.moon_a * p.rp, q=p.moon_q, moon_per=p.moon_per,
             moon_inc=math.radians(p.moon_inc), node=math.radians(p.moon_node),
             moon_ecc=p.moon_ecc, moon_w=math.radians(p.moon_w),
             phase=math.radians(p.moon_phase))
    P, M, _ = TM.ref_positions(t, d)
    with mx.stream(mx.cpu):
        f = lambda v: mx.array(v, dtype=mx.float64)
        args = (f(P[0]), f(P[1]), f(M[0]), f(M[1]), f(p.rp), f(p.rp * p.moon_r))
        fronts = dict(front_p=f(P[2]) > 0, front_m=f(M[2]) > 0)
        if p.limb_dark == "quadratic":
            out = moons.flux_dev_moon_quad(*args, *p.u, **fronts)
        else:
            out = moons.flux_dev_moon(*args, p.u, p.limb_dark, **fronts)
        return 1.0 + np.array(out)


@pytest.mark.parametrize("law", [HYB, QUAD])
@pytest.mark.parametrize("ecc,moon_ecc", [(0.0, 0.0), (0.1, 0.05)])
def test_light_curve_matches_the_photometry_at_independent_positions(law, ecc, moon_ecc):
    p = system(law, ecc, moon_ecc)
    t = windows(p)
    F = elome.TransitModel(p, t).light_curve(p)
    assert np.abs(F - reference(p, t)).max() < 1e-13
    assert F.min() < 1.0 - 1e-3                     # a transit, with the moon's dip


@pytest.mark.parametrize("law", [HYB, QUAD])
def test_a_vanishing_moon_leaves_the_planet(law):
    p = system(law, moon_r=0.0, moon_q=0.0)
    t = windows(p)
    q = elome.TransitParams()
    for k in ("t0", "per", "rp", "a", "inc", "ecc", "w", "limb_dark", "u"):
        setattr(q, k, getattr(p, k))
    F = elome.TransitModel(p, t).light_curve(p)
    G = elome.TransitModel(q, t).light_curve(q)
    assert np.abs(F - G).max() < 2e-14       # the orbit by the moon route's own split


def test_defaults_are_a_coplanar_circular_moon_in_front_at_t0():
    p = system(moon_ecc=0.0, moon_inc=None, moon_node=None, moon_phase=None, moon_w=None)
    q = system(moon_ecc=0.0, moon_inc=p.inc, moon_node=0.0, moon_phase=0.0, moon_w=0.0)
    q.moon_q = p.moon_q = None
    t = windows(p)
    m = elome.TransitModel(p, t)
    q.moon_q = 0.0
    assert np.array_equal(m.light_curve(p), elome.TransitModel(q, t).light_curve(q))


def test_the_far_side_of_the_orbit_is_flat():
    p = system()
    t = p.t0 + 0.5 * p.per + np.linspace(-0.5, 0.5, 201)
    assert np.all(elome.TransitModel(p, t).light_curve(p) == 1.0)


def midpoint(p, t, n=1500):
    off = (np.arange(n) + 0.5) / n * EXP - 0.5 * EXP
    tt = (t[:, None] + off[None, :]).ravel()
    return elome.TransitModel(p, tt).light_curve(p).reshape(t.size, n).mean(axis=1)


@pytest.mark.parametrize("law", [HYB, QUAD])
def test_the_contact_rule_against_a_converged_quadrature(law):
    p = system(law, ecc=0.0, moon_ecc=0.0, moon_a=4.0, moon_per=0.6, moon_inc=89.5)
    t = windows(p, epochs=(0, 1), step=EXP)
    R = midpoint(p, t)
    errs = []
    for n_gl in (5, 7):
        m = elome.TransitModel(p, t, exp_time=EXP, integration="contact", n_gl=n_gl)
        errs.append(np.abs(m.light_curve(p) - R).max())
    assert errs[1] < 3e-8 and errs[0] < 1.5e-7, errs
    # an unsplit rule is far worse there
    m7 = elome.TransitModel(p, t, exp_time=EXP, supersample_factor=7)
    assert np.abs(m7.light_curve(p) - R).max() > 100 * errs[1]


def test_supersampling_averages_the_instantaneous_curve():
    p = system()
    t = windows(p, step=EXP)
    n = 5
    F = elome.TransitModel(p, t, exp_time=EXP, supersample_factor=n).light_curve(p)
    off = np.linspace(-0.5 * EXP, 0.5 * EXP, n)
    G = elome.TransitModel(p, (t[:, None] + off).ravel()).light_curve(p).reshape(t.size, n)
    assert np.abs(F - G.mean(axis=1)).max() < 1e-15


NAMES = ["t0", "per", "rp", "a", "inc", "moon_r", "moon_a", "moon_per", "moon_q",
         "moon_inc", "moon_node", "moon_phase", "moon_ecc", "moon_w"]


@pytest.mark.parametrize("contact", [False, True])
def test_gradients_through_light_curve_mx(contact):
    """Every moon field differentiable; under the contact rule the kinks
    move with the parameters, so the gradient is the evaluated quadrature's."""
    p = system()
    t = windows(p, epochs=(0, 4), step=EXP if contact else 4.0 / 1440.0)
    kw = dict(exp_time=EXP, integration="contact", n_gl=5) if contact else {}
    m = elome.TransitModel(p, t, **kw)
    wts = np.random.default_rng(0).normal(size=t.size)
    base = {n: float(getattr(p, n)) for n in NAMES}

    def loss(vals):
        q = copy.copy(p)
        for n, v in zip(NAMES, vals):
            setattr(q, n, v)
        return mx.sum(m.light_curve_mx(q) * mx.array(wts, dtype=mx.float64))

    with mx.stream(mx.cpu):
        x0 = [mx.array(base[n], dtype=mx.float64) for n in NAMES]
        g = mx.grad(lambda *v: loss(v), argnums=tuple(range(len(NAMES))))(*x0)
        for i, n in enumerate(NAMES):
            h = 1e-8 * max(1.0, abs(base[n])) if n in ("per", "moon_per", "t0") else \
                1e-6 * max(1.0, abs(base[n]))
            up = [mx.array(base[k] + (h if k == n else 0.0), dtype=mx.float64) for k in NAMES]
            dn = [mx.array(base[k] - (h if k == n else 0.0), dtype=mx.float64) for k in NAMES]
            fd = (loss(up).item() - loss(dn).item()) / (2 * h)
            gi = g[i].item()
            assert np.isfinite(gi) and abs(gi - fd) <= 2e-4 * max(abs(fd), 1e-3), (n, gi, fd)


@pytest.mark.parametrize("law,tol", [(HYB, 2e-7), (QUAD, 1.5e-6)])
def test_float32_holds_far_from_t0(law, tol):
    """120 orbits from t0 at a BJD-sized epoch: the moon's split phase keeps
    float32 at its photometry's own level (the quadratic disc's ~3e-7)."""
    p = system(law, ecc=0.0, moon_ecc=0.0, t0=2459000.2)
    t = windows(p, epochs=(0, 40, 120))
    F64 = elome.TransitModel(p, t).light_curve(p)
    F32 = elome.TransitModel(p, t, dtype=mx.float32).light_curve(p)
    assert np.abs(F32 - F64).max() < tol


@pytest.mark.parametrize("change,msg", [
    (dict(moon_r=None), "set params.moon_r"),
    (dict(limb_dark="polynomial", u=[0.3, 0.1, 0.05]), "hybrid law"),
    (dict(spots=np.array([[0.0, 0.0, 10.0, 0.5]]), prot=10.0), "spherical planet"),
    (dict(ring_in=1.5, ring_out=2.0, ring_f=0.5), "spherical planet"),
    (dict(f=0.1), "spherical planet"),
])
def test_construction_checks(change, msg):
    p = system(**change)
    with pytest.raises(ValueError, match=msg):
        elome.TransitModel(p, windows(p))


def test_evaluation_checks():
    p = system()
    t = windows(p)
    with pytest.raises(ValueError, match="primary-transit"):
        elome.TransitModel(system(fp=1e-4), t, transittype="secondary")
    m = elome.TransitModel(p, t)
    with pytest.raises(ValueError, match="outside the planet"):
        m.light_curve(system(moon_a=1.2))
    with pytest.raises(ValueError, match="moon_r changed"):
        q = system()
        for k in ("moon_r", "moon_a", "moon_per", "moon_q", "moon_inc", "moon_node",
                  "moon_phase", "moon_ecc", "moon_w"):
            setattr(q, k, None)
        m.light_curve(q)
    with pytest.raises(ValueError, match="outside the planet"):
        m.light_curves([p, system(moon_a=1.2)])


def sets(law):
    """Four parameter sets around system(law): eccentric and circular,
    a default moon_inc, a massless moon, a moon behind the planet."""
    return [system(law),
            system(law, ecc=0.0, moon_ecc=0.0, moon_inc=None, moon_q=0.0),
            system(law, rp=0.1, moon_r=0.2, moon_a=4.5, moon_per=0.7, moon_phase=200.0,
                   moon_node=-30.0, moon_w=120.0, per=30.001),
            system(law, t0=0.21, inc=89.9, moon_inc=92.0, moon_ecc=0.2)]


@pytest.mark.parametrize("law", [HYB, QUAD])
@pytest.mark.parametrize("kw", [{}, dict(exp_time=EXP, integration="contact", n_gl=5),
                                dict(exp_time=EXP, supersample_factor=3)])
def test_light_curves_matches_light_curve_per_set(law, kw):
    ps = sets(law)
    t = windows(ps[0], step=EXP)
    m = elome.TransitModel(ps[0], t, **kw)
    F = m.light_curves(ps)
    G = np.stack([m.light_curve(p) for p in ps])
    assert F.shape == (len(ps), t.size)
    assert np.abs(F - G).max() < 1e-14
    assert np.all(F.min(axis=1) < 1.0 - 1e-3)


def test_light_curves_takes_array_valued_params():
    ps = sets(HYB)
    t = windows(ps[0])
    m = elome.TransitModel(ps[0], t)
    q = copy.copy(ps[0])
    for name in ("t0", "per", "rp", "inc", "ecc", "moon_r", "moon_a", "moon_per",
                 "moon_q", "moon_node", "moon_phase", "moon_ecc", "moon_w"):
        setattr(q, name, np.array([getattr(p, name) for p in ps], dtype=np.float64))
    q.moon_inc = None                      # each set's own inc
    ref = [copy.copy(p) for p in ps]
    for r in ref:
        r.moon_inc = r.inc
    assert np.abs(m.light_curves(q) - m.light_curves(ref)).max() < 1e-15


def test_light_curves_in_float32():
    ps = [system(HYB, ecc=0.0, moon_ecc=0.0, t0=2459000.2),
          system(HYB, ecc=0.0, moon_ecc=0.0, t0=2459000.2, moon_phase=120.0)]
    t = windows(ps[0], epochs=(0, 40))
    F64 = elome.TransitModel(ps[0], t).light_curves(ps)
    F32 = elome.TransitModel(ps[0], t, dtype=mx.float32).light_curves(ps)
    assert np.abs(F32 - F64).max() < 2e-7
