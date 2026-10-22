"""Spotted stars through TransitModel (params.spots): light curves against
an independent reference (the spot oracle at sky positions from a plain
numpy orbit and a plain numpy rotating star), the conventions (Fabrycky &
Winn's lambda, prograde rotation, longitudes at the spots' epoch), the
batched and differentiable surfaces, and the parameter checks."""

import math

import mlx.core as mx
import numpy as np
import pytest

import elome as el
import oblate_oracle as OO
import spot_oracle as SO
from elome.hybrid import get_law, hybrid_norms

D = math.radians
LAW, W = "hybrid4", [0.2, 0.2, 0.1, 0.1]


def P(**kw):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.0, 0.1, 9.0, 88.5, 0.0, 90.0
    p.limb_dark, p.u = LAW, list(W)
    p.spots = [[10.0, -10.0, 8.0, 0.4], [60.0, 20.0, 12.0, 0.6]]
    p.prot, p.istar, p.lam, p.spot_epoch = 12.0, None, None, None
    for k, v in kw.items():
        setattr(p, k, v)
    return p


T = np.linspace(-0.12, 0.12, 49)


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------

def star_vectors(t, prot, istar, lam, spots, epoch):
    """Spot centres from the definitions: spin axis s = (sin i sin lam,
    sin i cos lam, cos i) (Fabrycky & Winn 2009), longitude 0 at the epoch
    on the meridian facing the observer, prograde about s."""
    i, l = D(istar), D(lam)
    s = np.array([math.sin(i) * math.sin(l), math.sin(i) * math.cos(l), math.cos(i)])
    z = np.array([0.0, 0.0, 1.0])
    e_obs = z - (z @ s) * s
    e_obs = (e_obs / np.linalg.norm(e_obs) if np.linalg.norm(e_obs) > 1e-12
             else np.array([-math.sin(l), -math.cos(l), 0.0]))
    e_east = np.cross(s, e_obs)
    out = []
    for lon, lat, al, c in spots:
        ph = D(lon) + 2 * math.pi * (t - epoch) / prot
        n = math.cos(D(lat)) * (math.cos(ph) * e_obs + math.sin(ph) * e_east) + math.sin(D(lat)) * s
        out.append((n, D(al), c))
    return out


def reference(p, t):
    law = get_law(p.limb_dark)
    epoch = p.t0 if p.spot_epoch is None else p.spot_epoch
    istar = 90.0 if p.istar is None else p.istar
    lam = 0.0 if p.lam is None else p.lam
    X, Y, front = OO.sky_position(np.asarray(t) - p.t0, p.per, p.a, D(p.inc), p.ecc, D(p.w))
    from elome.hybrid import w_to_c
    c, d = w_to_c(law, np.asarray(p.u, dtype=np.float64))
    coef = np.concatenate([np.pad(c, (0, 3 - len(c))), d])
    norm = math.pi * (coef[0] + coef[1] / 2 + coef[2] / 3
                      + sum(dk / (e * (1 + e)) for dk, e in zip(d, law.eps)))
    out = []
    for ti, x, y, fr in zip(np.atleast_1d(t), X, Y, front):
        planet = (x, y, p.rp) if fr and math.hypot(x, y) < 1 + p.rp else None
        sp = star_vectors(ti, p.prot, istar, lam, p.spots, epoch)
        out.append(1.0 + SO.columns(planet, sp, law.eps) @ coef / norm)
    return np.array(out)


@pytest.mark.parametrize("kw", [
    {},
    dict(lam=35.0, istar=60.0),
    dict(lam=-120.0, istar=150.0, spot_epoch=-0.4),
    dict(ecc=0.3, w=40.0),
    dict(inc=90.0, spots=[[0.0, 0.0, 10.0, 0.3], [0.0, 0.0, 5.0, 0.5]]),   # umbra on the chord
    dict(spots=[[-30.0, -8.6, 15.0, 1.4]]),                                # a facula
], ids=["aligned", "oblique", "retrograde, epoch", "eccentric", "umbra", "facula"])
def test_against_the_reference(kw):
    p = P(**kw)
    got = el.TransitModel(p, T).light_curve(p)
    assert np.max(np.abs(got - reference(p, T))) < 1e-13


def test_out_of_transit_is_the_rotational_modulation():
    """Over a whole rotation, away from transit, the model is the spots'
    own modulation: the planet's columns are exactly zero."""
    p = P()
    t = np.linspace(0.6, 0.6 + p.prot, 40) % p.per + 0.4       # never in transit
    got = el.TransitModel(p, t).light_curve(p)
    assert np.max(np.abs(got - reference(p, t))) < 1e-13
    assert np.ptp(got) > 1e-3


def test_unspotted_is_the_plain_model():
    """Contrast 1 everywhere: the spherical hybrid model, to rounding."""
    p = P(spots=[[10.0, -10.0, 8.0, 1.0], [60.0, 20.0, 12.0, 1.0]])
    q = P(spots=None, prot=None)
    a = el.TransitModel(p, T).light_curve(p)
    b = el.TransitModel(q, T).light_curve(q)
    assert np.max(np.abs(a - b)) < 2e-15


# ---------------------------------------------------------------------------
# conventions
# ---------------------------------------------------------------------------

def test_spin_axis_is_fabrycky_winn():
    from elome.spots_tau import spin_frame
    for i, l in ((90.0, 0.0), (60.0, 35.0), (150.0, -120.0), (0.0, 20.0)):
        with mx.stream(mx.cpu):
            s, e_obs, e_east = (np.array([c.item() for c in v]) for v in spin_frame(
                mx.array(D(i), dtype=mx.float64), mx.array(D(l), dtype=mx.float64)))
        ref = [math.sin(D(i)) * math.sin(D(l)), math.sin(D(i)) * math.cos(D(l)), math.cos(D(i))]
        assert np.allclose(s, ref, atol=1e-15)
        assert np.allclose(np.cross(s, e_obs), e_east, atol=1e-15)
        assert e_obs[2] >= 0.0 and abs(s @ e_obs) < 1e-15


def bump_time(p):
    """The time of the largest spot-crossing bump: spotted minus unspotted."""
    t = np.linspace(-0.08, 0.08, 321)
    q = P(**{k: getattr(p, k) for k in ("inc", "lam", "istar", "prot")})
    q.spots = [[s[0], s[1], s[2], 1.0] for s in p.spots]
    d = el.TransitModel(p, t).light_curve(p) - el.TransitModel(q, t).light_curve(q)
    return t[np.argmax(d)]


def test_prograde_spot_moves_with_the_planet():
    """A dark spot on the chord, ahead of the planet (X = +0.25 at t0): an
    aligned star (lam = 0) carries it along +X with the planet, so the
    planet reaches it later than if the star did not turn; a retrograde
    star (lam = 180, its spin axis along -Y, so the spot's latitude and
    longitude mirror to stay at the same place at t0) carries it towards
    the planet, earlier."""
    lat = -math.degrees(math.asin(9.0 * math.cos(D(88.5))))   # the chord, Y = -b
    pro = bump_time(P(lam=0.0, spots=[[15.0, lat, 4.0, 0.2]], prot=2.0))
    still = bump_time(P(lam=0.0, spots=[[15.0, lat, 4.0, 0.2]], prot=1e9))
    retro = bump_time(P(lam=180.0, spots=[[-15.0, -lat, 4.0, 0.2]], prot=2.0))
    assert pro > still + 0.001 and still > retro + 0.001       # 2 grid steps


def test_lambda_sign():
    """Positive lambda turns the projected spin axis clockwise from the
    projected orbit normal (+Y) as the observer sees the sky (X right, Y
    up): towards +X. A spot at the visible pole of a star tilted towards
    the observer (i* = 60) sits on the projected spin axis."""
    for lam, sx in ((30.0, 1.0), (-30.0, -1.0)):
        p = P(lam=lam, istar=60.0, spots=[[0.0, 90.0, 5.0, 0.5]])
        n = star_vectors(0.0, p.prot, 60.0, lam, p.spots, 0.0)[0][0]
        assert np.sign(n[0]) == sx and n[1] > 0
        got = el.TransitModel(p, T[:3]).light_curve(p)
        assert np.max(np.abs(got - reference(p, T[:3]))) < 1e-13


def test_recurrence_on_an_aligned_star():
    """Sanchis-Ojeda & Winn's geometry: with lam = 0, i* = 90 and P_rot =
    2 P_orb, a spot crossed at one transit is crossed again two transits
    later at the same phase, and is on the far side of the star in the
    transit between."""
    lat = -math.degrees(math.asin(9.0 * math.cos(D(88.5))))
    p = P(spots=[[0.0, lat, 6.0, 0.3]], prot=6.0)
    q = P(spots=[[0.0, lat, 6.0, 1.0]], prot=6.0)
    t = np.linspace(-0.08, 0.08, 81)
    bumps = []
    for k in range(3):
        tk = t + k * p.per
        bumps.append(el.TransitModel(p, tk).light_curve(p) - el.TransitModel(q, tk).light_curve(q))
    assert np.ptp(bumps[0]) > 1e-3                            # the crossing
    assert np.max(np.abs(bumps[2] - bumps[0])) < 1e-12
    assert np.max(np.abs(bumps[1])) < 1e-15                   # the spot on the far side


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------

def test_supersampling_averages_the_instantaneous_curve():
    p = P()
    n, e = 7, 0.01
    a = el.TransitModel(p, T, supersample_factor=n, exp_time=e).light_curve(p)
    offs = np.linspace(-e / 2, e / 2, n)
    b = el.TransitModel(p, (T[:, None] + offs).ravel()).light_curve(p).reshape(-1, n).mean(1)
    assert np.max(np.abs(a - b)) < 1e-15


def test_float32_matches_float64():
    p = P(lam=20.0, istar=70.0, ecc=0.2, w=60.0)
    a = el.TransitModel(p, T).light_curve(p)
    b = el.TransitModel(p, T, dtype=mx.float32).light_curve(p)
    assert np.max(np.abs(a - b)) < 2e-6


def test_absolute_times_in_float32():
    """A mission epoch: t0 and the spots' epoch at BJD-like values."""
    t0 = 2459000.123
    p, q = P(t0=t0, spot_epoch=t0 - 5.0), P(spot_epoch=-5.0)
    a = el.TransitModel(p, T + t0, dtype=mx.float32).light_curve(p)
    b = el.TransitModel(q, T).light_curve(q)
    assert np.max(np.abs(a - b)) < 2e-6


def test_light_curves_equals_the_loop():
    rng = np.random.default_rng(3)
    seq = []
    for _ in range(4):
        p = P(rp=rng.uniform(0.05, 0.12), inc=rng.uniform(87.5, 90.0), lam=rng.uniform(-60, 60),
              istar=rng.uniform(50, 130), prot=rng.uniform(5, 20),
              spot_epoch=rng.uniform(-1, 1) if rng.uniform() < 0.5 else None,
              spots=[[rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(3, 15),
                      rng.uniform(0.2, 1.2)] for _ in range(2)])
        seq.append(p)
    m = el.TransitModel(seq[0], T)
    got = m.light_curves(seq)
    for i, p in enumerate(seq):
        assert np.max(np.abs(got[i] - m.light_curve(p))) < 1e-15
    # the same sets as arrays, spots (n, K, 4)
    arr = P()
    for k in ("rp", "inc", "lam", "istar", "prot"):
        setattr(arr, k, np.array([getattr(p, k) for p in seq]))
    arr.spot_epoch = np.array([p.t0 if p.spot_epoch is None else p.spot_epoch for p in seq])
    arr.spots = np.array([p.spots for p in seq])
    assert np.max(np.abs(m.light_curves(arr) - got)) < 1e-15


def test_light_curves_shared_spots_and_eccentric_sets():
    seq = [P(ecc=0.0), P(ecc=0.2, w=30.0), P(ecc=0.4, w=200.0)]
    m = el.TransitModel(seq[0], T)
    got = m.light_curves(seq)
    for i, p in enumerate(seq):
        assert np.max(np.abs(got[i] - m.light_curve(p))) < 1e-14


def test_light_curve_mx_gradients():
    """Gradients in the spot table, the star's rotation and the orbit,
    against central differences of light_curve."""
    p0 = P(lam=20.0, istar=70.0)
    t = np.linspace(-0.06, 0.06, 13)
    m = el.TransitModel(p0, t)
    names = ["rp", "inc", "prot", "istar", "lam"]
    base = {k: float(getattr(p0, k)) for k in names}
    sp0 = np.array(p0.spots, dtype=np.float64)
    wts = mx.array(np.linspace(1.0, 2.0, t.size), dtype=mx.float64)

    def F(spots, *vals):
        p = P(lam=20.0, istar=70.0)
        for k, v in zip(names, vals):
            setattr(p, k, v)
        p.spots = spots
        return mx.sum(m.light_curve_mx(p) * wts)

    with mx.stream(mx.cpu):
        args = [mx.array(sp0, dtype=mx.float64)] + [mx.array(base[k], dtype=mx.float64) for k in names]
        grads = mx.grad(F, argnums=list(range(1 + len(names))))(*args)

    def Fh(spots, vals):
        p = P(lam=20.0, istar=70.0)
        for k, v in zip(names, vals):
            setattr(p, k, v)
        p.spots = spots.tolist()
        return float(np.sum(m.light_curve(p) * np.array(wts)))
    for j, k in enumerate(names):
        h = 1e-6 * max(abs(base[k]), 1.0)
        up = [base[n] + (h if n == k else 0.0) for n in names]
        dn = [base[n] - (h if n == k else 0.0) for n in names]
        fd = (Fh(sp0, up) - Fh(sp0, dn)) / (2 * h)
        assert abs(grads[1 + j].item() - fd) < 1e-6 * max(abs(fd), 1e-3), k
    G = np.array(grads[0])
    vals = [base[n] for n in names]
    for i in range(sp0.shape[0]):
        for c in range(4):
            h = 1e-6 * max(abs(sp0[i, c]), 1.0)
            up, dn = sp0.copy(), sp0.copy()
            up[i, c] += h
            dn[i, c] -= h
            fd = (Fh(up, vals) - Fh(dn, vals)) / (2 * h)
            assert abs(G[i, c] - fd) < 1e-6 * max(abs(fd), 1e-3), (i, c)


def test_ld_basis_contract_through_the_model():
    """The flux is the shape columns' ratio with the unspotted star's norms."""
    from elome.spots_tau import flux_dev_spots_t
    p = P()
    law = get_law(LAW)
    with mx.stream(mx.cpu):
        args = [mx.array(v, dtype=mx.float64) for v in (0.0, 0.0, 3.0, 9.0, 9.0 * math.cos(D(88.5)),
                                                       0.1, 12.0, D(90.0), 0.0)]
        sp = np.array(p.spots, dtype=np.float64) * np.array([D(1)] * 3 + [1.0])
        tt = mx.array(T, dtype=mx.float64)
        B = np.array(flux_dev_spots_t(tt, *args, sp, law, W, basis=True))
        dev = np.array(flux_dev_spots_t(tt, *args, sp, law, W))
    c = np.concatenate([[1.0], -np.asarray(W)])
    assert np.max(np.abs(dev - (B @ c) / (hybrid_norms(law) @ c))) < 1e-16


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw, msg", [
    (dict(limb_dark="quadratic", u=[0.3, 0.2]), "hybrid law"),
    (dict(spot_u=[0.1]), "reserved"),
    (dict(f=0.1), "spherical planet"),
    (dict(spots=[[0.0, 0.0, 5.0]]), r"\(K, 4\)"),
    (dict(spots=[]), r"\(K, 4\)"),
])
def test_construction_checks(kw, msg):
    with pytest.raises(ValueError, match=msg):
        el.TransitModel(P(**kw), T)


def test_secondary_is_refused():
    p = P()
    p.fp = 1e-3
    with pytest.raises(ValueError, match="primary-transit only"):
        el.TransitModel(p, T, transittype="secondary")


@pytest.mark.parametrize("spots, msg", [
    ([[0.0, 0.0, 0.0, 0.5]], "angular radius"), ([[0.0, 0.0, 95.0, 0.5]], "angular radius"),
    ([[0.0, 91.0, 5.0, 0.5]], "latitude"), ([[0.0, 0.0, 5.0, -0.1]], "contrast"),
    ([[0.0, np.nan, 5.0, 0.5]], "finite")])
def test_spot_domain(spots, msg):
    p = P(spots=spots)
    m = el.TransitModel(P(spots=[[0.0, 0.0, 5.0, 0.5]]), T)
    with pytest.raises(ValueError, match=msg):
        m.light_curve(p)


def test_fixed_per_model():
    m = el.TransitModel(P(), T)
    with pytest.raises(ValueError, match="number of spots"):
        m.light_curve(P(spots=[[0.0, 0.0, 5.0, 0.5]]))
    with pytest.raises(ValueError, match="params.spots changed"):
        m.light_curve(P(spots=None, prot=None))


def test_star_fields_need_spots():
    p = P(spots=None)
    with pytest.raises(ValueError, match="set params.spots"):
        el.TransitModel(p, T)


def test_prot_required():
    with pytest.raises(ValueError, match="prot"):
        el.TransitModel(P(prot=None), T).light_curve(P(prot=None))


# ---------------------------------------------------------------------------
# the contact rule
# ---------------------------------------------------------------------------

LAT_CHORD = -math.degrees(math.asin(9.0 * math.cos(D(88.5))))
EXP = 30.0 / 1440                                        # long cadence


def exposure_mean(p, ti, e=EXP, n=40001):
    """The exposure average of the instantaneous curve, by the trapezoid
    rule on a fine grid (its kinks cost O(h^2))."""
    tt = np.linspace(ti - e / 2, ti + e / 2, n)
    return np.trapezoid(el.TransitModel(p, tt).light_curve(p), tt) / e


@pytest.mark.parametrize("kw", [
    dict(spots=[[8.0, LAT_CHORD, 5.0, 0.3], [40.0, 15.0, 10.0, 0.6]], prot=4.0),
    dict(spots=[[8.0, LAT_CHORD, 12.0, 0.4]], prot=1.0, lam=30.0, istar=70.0),
    dict(spots=[[-5.0, LAT_CHORD, 4.0, 0.2]], prot=3.0, ecc=0.3, w=50.0),
], ids=["spot smaller than the planet", "fast rotator, oblique", "eccentric"])
def test_contact_rule_against_the_exposure_average(kw):
    """Long-cadence exposures across a spot crossing: the rule's kinks
    include every planet-rim tangency, so it converges with n_gl as the
    spherical rule does (a spot smaller than the planet is wholly covered
    mid-crossing: four events in one exposure)."""
    p = P(**kw)
    ts = np.array([-0.01, -0.005, 0.0, 0.005, 0.01, 0.015])
    ref = np.array([exposure_mean(p, ti) for ti in ts])
    err = {}
    for g in (5, 9):
        got = el.TransitModel(p, ts, integration="contact", exp_time=EXP, n_gl=g).light_curve(p)
        err[g] = np.max(np.abs(got - ref))
    assert err[9] < 1e-7
    assert err[9] < 0.2 * err[5]


def test_contact_rule_beats_supersampling():
    p = P(spots=[[8.0, LAT_CHORD, 5.0, 0.3]], prot=4.0)
    ts = np.array([0.0, 0.005, 0.01])
    ref = np.array([exposure_mean(p, ti) for ti in ts])
    c = el.TransitModel(p, ts, integration="contact", exp_time=EXP, n_gl=5).light_curve(p)
    s = el.TransitModel(p, ts, supersample_factor=35, exp_time=EXP).light_curve(p)
    assert np.max(np.abs(c - ref)) < 0.02 * np.max(np.abs(s - ref))


def test_contact_rule_out_of_transit_is_the_modulation():
    """Away from transit the rule integrates the modulation; on a fast
    rotator spots cross the limb within an exposure, a kink it does not
    split (a curvature discontinuity: about 5e-9 here)."""
    p = P(prot=0.8)
    ts = np.array([0.5, 1.0, 1.4])
    got = el.TransitModel(p, ts, integration="contact", exp_time=EXP, n_gl=7).light_curve(p)
    ref = np.array([exposure_mean(p, ti, n=4001) for ti in ts])
    assert np.max(np.abs(got - ref)) < 2e-8


def test_contact_rule_batched_and_float32():
    seq = [P(spots=[[8.0, LAT_CHORD, 5.0, 0.3]], prot=4.0),
           P(spots=[[20.0, LAT_CHORD + 3.0, 8.0, 0.5]], prot=2.0, lam=20.0, ecc=0.1, w=80.0)]
    ts = np.linspace(-0.05, 0.05, 11)
    m = el.TransitModel(seq[0], ts, integration="contact", exp_time=EXP)
    got = m.light_curves(seq)
    for i, p in enumerate(seq):
        assert np.max(np.abs(got[i] - m.light_curve(p))) < 1e-15
    m32 = el.TransitModel(seq[0], ts, integration="contact", exp_time=EXP, dtype=mx.float32)
    assert np.max(np.abs(m32.light_curves(seq) - got)) < 2e-6


def test_contact_rule_gradients():
    """light_curve_mx differentiates the quadrature it evaluates: the
    split points carry their events' implicit derivatives, so the gradient
    matches central differences of light_curve."""
    p0 = P(spots=[[8.0, LAT_CHORD, 5.0, 0.3]], prot=4.0)
    ts = np.array([-0.005, 0.0, 0.005, 0.01])
    m = el.TransitModel(p0, ts, integration="contact", exp_time=EXP)
    names = ["rp", "inc", "prot", "lam"]
    base = {"rp": 0.1, "inc": 88.5, "prot": 4.0, "lam": 0.0}

    def F(*vals):
        p = P(spots=[[8.0, LAT_CHORD, 5.0, 0.3]])
        for k, v in zip(names, vals):
            setattr(p, k, v)
        return mx.sum(m.light_curve_mx(p))

    with mx.stream(mx.cpu):
        g = mx.grad(F, argnums=list(range(len(names))))(
            *[mx.array(base[k], dtype=mx.float64) for k in names])
    for j, k in enumerate(names):
        h = 1e-6 * max(abs(base[k]), 1.0)
        up = [base[n] + (h if n == k else 0.0) for n in names]
        dn = [base[n] - (h if n == k else 0.0) for n in names]
        with mx.stream(mx.cpu):
            fd = (F(*[mx.array(v, dtype=mx.float64) for v in up]).item()
                  - F(*[mx.array(v, dtype=mx.float64) for v in dn]).item()) / (2 * h)
        assert np.isfinite(g[j].item())
        assert abs(g[j].item() - fd) < 1e-6 * max(abs(fd), 1e-3), k


def test_contact_rule_gradient_across_blocks(monkeypatch):
    """A light curve long enough to be evaluated in blocks: the gradient is
    the one-block gradient (the event search runs once, outside the
    blocks, and nothing in it is re-traced by the backward pass)."""
    from elome import spots_tau as ST
    p0 = P(spots=[[8.0, LAT_CHORD, 5.0, 0.3], [40.0, 15.0, 10.0, 0.6]], prot=4.0)
    ts = np.linspace(-0.06, 0.06, 40)

    def grad():
        m = el.TransitModel(p0, ts, integration="contact", exp_time=EXP)

        def F(prot, rp):
            p = P(spots=p0.spots)
            p.prot, p.rp = prot, rp
            return mx.sum(m.light_curve_mx(p))
        with mx.stream(mx.cpu):
            g = mx.grad(F, argnums=[0, 1])(mx.array(4.0, dtype=mx.float64),
                                            mx.array(0.1, dtype=mx.float64))
        return np.array([v.item() for v in g])
    one = grad()
    monkeypatch.setattr(ST, "_SPOT_BLOCK", 400)            # 40 times x 49 nodes: 5 blocks
    monkeypatch.setattr(ST, "_EVENT_SAMPLES", 300)         # and a chunked search
    many = grad()
    assert np.all(np.isfinite(many))
    assert np.max(np.abs(many - one)) < 1e-12 * np.max(np.abs(one))


def test_sample_count():
    from elome.spots_tau import _S_MAX, _S_MIN, sample_count
    assert sample_count(1e-4, 9.0, 3.0, 0.1) == _S_MIN               # short cadence
    assert sample_count(30 / 1440, 9.0, 3.0, 0.1) == 16
    assert sample_count(30 / 1440, 9.0, 3.0, 1e-4) == _S_MAX
