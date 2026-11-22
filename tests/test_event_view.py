"""The float64 graphs evaluate only the times an event can touch
(TransitModel._view): the same light curve as evaluating every time, the
same gradients, and never a dropped point that is in the event."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome as el
from elome.api import _view_size

RNG = np.random.default_rng(3)
T = 2500.0 + np.sort(RNG.uniform(0.0, 27.0, 4000))
H4 = dict(limb_dark="hybrid4", u=[0.2, 0.2, 0.1, 0.1])
EXP = 0.02


def P(**kw):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 2501.3, 3.45, 0.1, 9.0, 89.0, 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [0.4, 0.25]
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def pair(p, **mk):
    """A model, and the same model evaluating every time."""
    full = el.TransitModel(p, T, **{"dtype": mx.float64, **mk})
    full._view = lambda *a, **k: None
    return el.TransitModel(p, T, **{"dtype": mx.float64, **mk}), full


CASES = {
    "quadratic": ({}, {}),
    "eccentric": (dict(ecc=0.4, w=-50.0), {}),
    "hybrid4": (H4, {}),
    "polynomial": (dict(limb_dark="polynomial", u=[0.3, 0.2, -0.1]), {}),
    "supersample": ({}, dict(supersample_factor=7, exp_time=EXP)),
    "contact": ({}, dict(integration="contact", exp_time=EXP)),
    "eccentric contact": (dict(ecc=0.3, w=40.0, **H4),
                          dict(integration="contact", exp_time=EXP)),
    "secondary": (dict(fp=1e-3, ecc=0.2, w=10.0), dict(transittype="secondary")),
    "oblate": (dict(f=0.2, theta=20.0, **H4), {}),
    "oblate contact": (dict(f=0.2, theta=20.0, **H4),
                       dict(integration="contact", exp_time=EXP)),
    "ringed": (dict(ring_in=1.3, ring_out=2.2, ring_f=0.6, theta=20.0, **H4), {}),
    "large occultor": (dict(rp=7.28, a=336.0, per=1.4079,
                            inc=math.degrees(math.acos(7.79 / 336.0)), **H4), {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_subset_is_the_full_light_curve(name):
    """Bit for bit, but for the contact rule's rounding; light_curve_mx
    takes the same subset, so it stays light_curve's bit for bit."""
    pk, mk = CASES[name]
    p = P(**pk)
    m, full = pair(p, **mk)
    view = m._event_view(p)
    assert view is not None and view[0].shape[0] < 0.2 * m._t_np.size
    got, ref = m.light_curve(p), full.light_curve(p)
    assert 1.0 - ref.min() > 1e-4 or "secondary" in name
    tol = 1e-15 if mk.get("integration") == "contact" else 0.0
    assert np.abs(got - ref).max() <= tol
    lc_mx = np.asarray(m.light_curve_mx(p))
    if m.supersample_factor > 1:
        lc_mx = lc_mx.reshape(T.size, -1).mean(axis=1)
    assert np.array_equal(lc_mx, got)


@pytest.mark.parametrize("name", ["quadratic", "eccentric contact", "oblate",
                                  "ringed"])
def test_light_curves_take_the_union_of_the_sets(name):
    pk, mk = CASES[name]
    n = 8
    pa = P(**pk)
    pa.rp = np.linspace(0.08, 0.12, n)
    pa.t0 = 2501.3 + np.linspace(-0.02, 0.02, n)
    m, full = pair(P(**pk), **mk)
    got, ref = m.light_curves(pa), full.light_curves(pa)
    tol = 1e-15 if mk.get("integration") == "contact" else 0.0
    assert np.abs(got - ref).max() <= tol


def test_gradients_are_the_full_graphs():
    """Under mx.grad the orbit is readable, so the subset is taken; the
    gradient in every field is the full graph's."""
    p = P(**H4, ecc=0.2, w=30.0)
    m, full = pair(p, integration="contact", exp_time=EXP)
    y = mx.array(full.light_curve(p) + 1e-4, dtype=mx.float64)

    def loss(model):
        def f(rp, a, inc, t0):
            q = P(**H4, ecc=0.2, w=30.0)
            q.rp, q.a, q.inc, q.t0 = rp, a, inc, t0
            return mx.sum((model.light_curve_mx(q) - y) ** 2)
        return f
    args = [mx.array(v, dtype=mx.float64) for v in (0.1, 9.0, 89.0, 2501.3)]
    with mx.stream(mx.cpu):
        g = mx.grad(loss(m), argnums=(0, 1, 2, 3))(*args)
        g_ref = mx.grad(loss(full), argnums=(0, 1, 2, 3))(*args)
        mx.eval(g, g_ref)
    for a, b in zip(g, g_ref):
        assert abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))


def test_inside_a_compile_every_time_is_evaluated():
    """A traced orbit cannot be read: the full graph, the same values."""
    p = P(**H4)
    m, full = pair(p)
    ref = full.light_curve(p)

    def f(rp):
        q = P(**H4)
        q.rp = rp
        return m.light_curve_mx(q)
    with mx.stream(mx.cpu):
        got = np.asarray(mx.compile(f)(mx.array(0.1, dtype=mx.float64)))
    assert np.array_equal(got, ref)


def test_float32_and_spots_and_tidal_planets_evaluate_every_time():
    assert el.TransitModel(P(), T, dtype=mx.float32)._event_view(P()) is None
    sp = P(**H4, spots=[[10.0, 5.0, 8.0, 0.6]], prot=12.0)
    assert el.TransitModel(sp, T, dtype=mx.float64)._event_view(sp) is None
    td = P(**H4, tidal_q=0.05, a=4.0)
    assert el.TransitModel(td, T, dtype=mx.float64)._event_view(td) is None


def test_an_orbit_that_misses_the_star():
    """No event at all: every point takes the out-of-event flux."""
    p = P(inc=60.0)
    m, full = pair(p)
    assert m._event_view(p)[0].shape[0] == _view_size(1)
    assert np.array_equal(m.light_curve(p), full.light_curve(p))
    assert np.all(m.light_curve(p) == 1.0)


def _kepler(M, e):
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def test_no_point_in_the_event_is_dropped():
    """On random orbits, transits and eclipses, with and without
    exposures: every time whose exposure touches the event (exact Kepler
    orbit, 41 samples across the exposure) is kept."""
    rng = np.random.default_rng(42)
    checked = 0
    for _ in range(400):
        e = 0.0 if rng.random() < 0.3 else rng.uniform(0.0, 0.95)
        w, a, rp = rng.uniform(-180, 180), rng.uniform(1.5, 60), rng.uniform(0.005, 0.6)
        if a * (1 - e) < 1.3 + rp:
            continue
        inc = math.degrees(math.acos(rng.uniform(0, min(1.0, 1.5 * (1 + rp) / a))))
        per, t0 = rng.uniform(0.5, 30), rng.uniform(0, 30)
        exp = 0.0 if rng.random() < 0.5 else rng.uniform(0.001, 0.05) * per
        sec = bool(rng.integers(2))
        p = P(t0=t0, per=per, rp=rp, a=a, inc=inc, ecc=e, w=w, fp=1e-3)
        t = np.sort(rng.uniform(0, 3 * per, 1500))
        kw = dict(integration="contact", exp_time=exp) if exp > 0 else {}
        m = el.TransitModel(p, t, transittype="secondary" if sec else "primary", **{"dtype": mx.float64, **kw})
        view = m._event_view(p)
        if view is None:
            continue
        pos = np.asarray(view[1])
        keep = pos < pos.max()
        tt = t[:, None] + (np.linspace(-exp / 2, exp / 2, 41) if exp > 0 else np.zeros(1))
        wr = math.radians(w)
        nu_c = math.pi / 2 - wr
        Ec = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu_c / 2),
                            math.sqrt(1 + e) * math.cos(nu_c / 2))
        M = Ec - e * math.sin(Ec) + 2 * math.pi * (tt - t0) / per
        E = _kepler(np.mod(M + math.pi, 2 * math.pi) - math.pi, e)
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
        r = a * (1 - e * np.cos(E))
        X, Y = -r * np.cos(wr + nu), -r * np.sin(wr + nu) * math.cos(math.radians(inc))
        front = np.sin(wr + nu) > 0
        inev = ((np.hypot(X, Y) < 1 + rp) & (~front if sec else front)).any(axis=1)
        assert not np.any(inev & ~keep)
        checked += 1
    assert checked > 300


def test_view_sizes_are_few():
    sizes = {_view_size(n) for n in range(1, 10 ** 6, 97)}
    assert len(sizes) <= 30
    assert all(_view_size(n) >= n for n in range(1, 5000))
    assert all(_view_size(n) <= max(64, 1.42 * n) for n in range(1, 5000))
