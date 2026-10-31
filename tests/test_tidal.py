"""A tidally locked triaxial planet: flux_dev_from_tau(..., tidal_q=, theta=,
mass_ratio=), ellc's "love" model.

The outline pinned to a 3-D ellipsoid with its nose on the star, and the
shape, the offset and the sky positions to a port of ellc's equations
(``ellc_axes``, ``instant``: f_src/stellar.f90 shape_love and starshape,
f_src/ellc.f90 lines 1004-1105, which were read but not run), on an
independent numpy orbit; light curves, exposures included, to the
independent quadrature of tests/oblate_oracle.py at each instant; contacts
to a dense reference; q = 0 against the spherical path and each instant
against the oblate mode; the kernels against fp64; gradients through q = 0
against finite differences; the frontend; the keyword checks.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import elome as el
from elome import metal as M
from elome import oblate_tau as OT

import oblate_oracle as OR

P, A_RS, RV = 0.925, 2.99, 0.1215
LAWS = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
        "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
W = LAWS["hybrid5"]
TAU = np.linspace(-0.075, 0.075, 241)
EXP = 60.0 / 86400.0
GPU = M.metal_available()


def F64(v):
    return mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)


def axes(rv, q):
    """Independently of the package: the ellipsoid's (r1, r2, r3)."""
    r2 = rv / ((1.0 + 3.0 * q) * (1.0 - q)) ** (1.0 / 3.0)
    return r2 * (1.0 + 3.0 * q), r2, r2 * (1.0 - q)


def g3(q):
    return q * (1.0 + 3.0 * q) * (1.0 - q)


def ellc_axes(rv_a, rsep, hf, qmass, rtol=1e-14):
    """ellc's shape_love inside starshape's loop, in units of a: (A, B, C, D)
    for the volumetric radius rv_a at separation rsep, with hf the Love
    number and qmass = M* / Mp. ellc's own rtol is 1e-6."""
    rt = rv_a
    for _ in range(100):
        b = rt
        qr = 0.5 * hf * qmass * (b / rsep) ** 3
        A, C = b * (1.0 + 3.0 * qr), b * (1.0 - qr)
        D = rt * qmass * rt ** 4
        t = (A * b * C) ** (1.0 / 3.0) / rv_a
        rt = rt / t
        if abs(t - 1.0) < rtol:
            break
    return A, b, C, D


def silhouette(nose, ci, abc, tw=0.0):
    """The 3-D ellipsoid's outline: its 2 x 2 shape matrix on the sky. Axes
    in (X, Y, towards the observer): abc[0] along ``nose``, the unit vector
    from the planet to the star, then abc[1] and abc[2] about it from the
    orbit's normal, turned by tw. Rotation matrices only, the support
    function of M(unit sphere)."""
    si = math.sqrt(1.0 - ci * ci)
    normal = np.array([0.0, si, ci])
    along = np.cross(nose, normal)
    e2 = math.cos(tw) * along + math.sin(tw) * normal
    e3 = -math.sin(tw) * along + math.cos(tw) * normal
    r1, r2, r3 = abc
    M3 = np.stack([r1 * nose, r2 * e2, r3 * e3], axis=1)
    return (M3 @ M3.T)[:2, :2]


def cos_i(a, b, e, w):
    return b * (1.0 + e * math.sin(w)) / (a * (1.0 - e * e))


def instant(t, a, ci, per, e, om, rv, q, tw, mu):
    """(S, X, Y, front) at time t since conjunction, in stellar radii. The
    oracle's Kepler orbit gives the angle th = nu + omega; the rest is
    ellc's: the shape at this separation, and uapp2 - uapp1, vapp2 - vapp1
    with the offset on the planet's barycentric position alone."""
    Xp, Yp, front = OR.sky_position(t, per, a, 0.0, e, om)
    rad = math.hypot(Xp[0], Yp[0])
    ct, st = -Xp[0] / rad, -Yp[0] / rad      # cos, sin (nu + omega)
    rsep = rad / a
    # any (hf, qmass) with this product gives q at rsep = 1; mu sets D
    qmass = 1.0 / mu if mu else 1.0
    hf = 2.0 * g3(q) / qmass * (a / rv) ** 3
    A, B, C, D = ellc_axes(rv / a, rsep, hf, qmass)
    D = D if mu else 0.0
    qmass1 = mu if mu else 0.0
    qfac2 = 1.0 / (1.0 + qmass1)
    qfac1 = qmass1 * qfac2
    u1, v1 = rsep * ct * qfac1, rsep * ci * st * qfac1
    u2, v2 = -rsep * ct * qfac2, -rsep * ci * st * qfac2
    X = a * (u2 * (rsep - D) / rsep - u1)
    Y = a * (v2 * (rsep - D) / rsep - v1)
    s, c = -ct, st                           # the angle from conjunction
    si = math.sqrt(1.0 - ci * ci)
    nose = -np.array([s, -ci * c, si * c])
    return silhouette(nose, ci, (a * A, a * B, a * C), tw), X, Y, bool(front[0])


def reference_flux(tau, rv, q, b, law, w, a=A_RS, per=P, tw=0.0, e=0.0,
                   om=math.pi / 2, mu=None):
    """F - 1 by the oracle's quadrature of the ellipse at each time. The
    orbit and the outline come from the numpy above, not from the package."""
    ci = cos_i(a, b, e, om)
    out = np.zeros(np.size(tau))
    for i, t in enumerate(np.ravel(tau)):
        S, X, Y, front = instant(t, a, ci, per, e, om, rv, q, tw, mu)
        if not front:
            continue
        val, vec = np.linalg.eigh(S)
        A, B = math.sqrt(val[1]), math.sqrt(val[0])
        major = vec[:, 1]
        x0 = X * major[0] + Y * major[1]
        y0 = -X * major[1] + Y * major[0]
        out[i] = OR.law_flux_dev(law, w, x0, y0, A, B)[0]
    return out


def flux(tau, b, q, law="hybrid5", rv=RV, a=A_RS, per=P, **kw):
    tau = mx.array(np.asarray(tau, dtype=np.float64), dtype=kw.pop("dtype", mx.float64))
    with mx.stream(mx.cpu if tau.dtype == mx.float64 else mx.gpu):
        out = el.flux_dev_from_tau(tau, per, a, b, rv, limb_dark=law,
                                   u=kw.pop("u", LAWS[law]), tidal_q=q, **kw)
        mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def kh(e, w):
    return dict(secosw=math.sqrt(e) * math.cos(w), sesinw=math.sqrt(e) * math.sin(w))


# -- the outline ---------------------------------------------------------

def outline(s, c, ci, rv, q, tw=0.0):
    with mx.stream(mx.cpu):
        return tuple(float(v) for v in OT.tidal_outline(
            F64(s), F64(c), F64(ci), F64(rv), F64(q), F64(tw)))


def test_the_outline_is_the_ellipsoids_silhouette():
    rng = np.random.default_rng(2)
    for _ in range(300):
        phi = rng.uniform(-1.4, 1.4)
        ci = rng.uniform(0.0, 0.6)
        q = rng.uniform(-0.25, 0.4)
        tw = rng.uniform(-3.2, 3.2)
        s, c = math.sin(phi), math.cos(phi)
        si = math.sqrt(1.0 - ci * ci)
        nose = -np.array([s, -ci * c, si * c])
        r, f, th = outline(s, c, ci, 0.1, q, tw)
        A = r / math.sqrt(1.0 - f)
        B = A * (1.0 - f)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        S = R @ np.diag([A * A, B * B]) @ R.T
        assert np.abs(S - silhouette(nose, ci, axes(0.1, q), tw)).max() < 1e-14


def test_mid_transit_axes():
    """At conjunction on an edge-on orbit the outline is (r2, r3), the long
    axis along the orbit for q > 0 and across it for q < 0, and the twist
    is its sky angle."""
    for q in (0.05, -0.05):
        r1, r2, r3 = axes(RV, q)
        for tw in (0.0, 0.4):
            r, f, th = outline(0.0, 1.0, 0.0, RV, q, tw)
            assert r == pytest.approx(math.sqrt(r2 * r3), rel=1e-14)
            assert f == pytest.approx(1.0 - min(r2, r3) / max(r2, r3), rel=1e-13)
            want = tw if q > 0 else tw - math.pi / 2
            assert (th - want + 0.5 * math.pi) % math.pi == pytest.approx(0.5 * math.pi, abs=1e-14)


def test_the_nose_shows_at_quadrature():
    """A quarter orbit on from conjunction (edge-on, untwisted), the nose
    lies on the sky along the orbit: the outline is (r1, r3), whatever the
    twist does to the other two axes at conjunction."""
    r1, r2, r3 = axes(RV, 0.05)
    r, f, th = outline(1.0, 0.0, 0.0, RV, 0.05)
    assert r / math.sqrt(1.0 - f) == pytest.approx(r1, rel=1e-14)
    assert r * math.sqrt(1.0 - f) == pytest.approx(r3, rel=1e-14)
    r, f, th = outline(1.0, 0.0, 0.0, RV, 0.05, 0.7)
    assert r / math.sqrt(1.0 - f) == pytest.approx(r1, rel=1e-14)


def test_the_volume_is_rv():
    r1, r2, r3 = axes(RV, 0.07)
    assert (r1 * r2 * r3) ** (1.0 / 3.0) == pytest.approx(RV, rel=1e-14)
    assert r1 / r2 - 1.0 == pytest.approx(3 * 0.07) and r3 / r2 == pytest.approx(0.93)


def test_a_round_outline_is_finite():
    """cos i ~ 1/2: the outline at mid-transit is a circle at q != 0, where
    its angle is undefined; nothing may be NaN, in value or gradient."""
    q = 0.02
    al, be = 6.0 + 9.0 * q, 2.0 - q
    ci = math.sqrt(be / (al + be))
    r, f, th = outline(0.0, 1.0, ci, 0.1, q)
    assert f < 1e-7 and math.isfinite(th) and math.isfinite(r)

    def L(p):
        return sum(OT.tidal_outline(F64(0.0), F64(1.0), p, F64(0.1), F64(q), F64(0.0)))
    with mx.stream(mx.cpu):
        assert math.isfinite(mx.grad(L)(F64(ci)).item())


# -- light curves --------------------------------------------------------

MU = 1.16e-3                       # a hot Jupiter's scale
MU_BIG = 0.05                      # large enough to see the offset's terms


@pytest.mark.parametrize("law,q,b,tw,e,om,mu", [
    ("hybrid5", 0.05, 0.08, 0.0, 0.0, 1.57, None),
    ("hybrid5", 0.1, 0.5, 0.0, 0.0, 1.57, None),
    ("hybrid4", 0.05, 0.3, 0.0, 0.0, 1.57, None),
    ("hybrid2", 0.03, 0.5, 0.0, 0.0, 1.57, None),
    ("hybrid5", -0.05, 0.3, 0.0, 0.0, 1.57, None),
    ("hybrid5", 0.05, 0.3, 0.6, 0.0, 1.57, None),
    ("hybrid5", -0.05, 0.3, -1.1, 0.0, 1.57, None),
    ("hybrid5", 0.05, 0.3, 0.0, 0.2, 0.7, None),
    ("hybrid5", 0.08, 0.2, 0.9, 0.4, 2.5, None),
    ("hybrid4", -0.02, 0.4, -0.5, 0.3, 4.4, None),
    ("hybrid5", 0.05, 0.08, 0.0, 0.0, 1.57, MU),
    ("hybrid5", 0.05, 0.3, 0.0, 0.0, 1.57, MU_BIG),
    ("hybrid5", 0.05, 0.3, 0.0, 0.2, 0.7, MU),
    ("hybrid5", 0.08, 0.2, 0.0, 0.4, 2.5, MU_BIG),
    ("hybrid4", 0.04, 0.2, 0.5, 0.3, 4.4, MU_BIG),
])
def test_light_curve_matches_the_reference(law, q, b, tw, e, om, mu):
    ref = reference_flux(TAU, RV, q, b, law, LAWS[law], tw=tw, e=e, om=om, mu=mu)
    got = flux(TAU, b, q, law, theta=tw, mass_ratio=mu, **(kh(e, om) if e else {}))
    assert np.abs(got - ref).max() < 3e-11
    assert ref.min() < -0.005


def test_q_follows_the_separation_as_ellc():
    """The cubic is the fixed point of ellc's loop, at any separation."""
    hf, mu, rv_a = 1.59, MU, RV / A_RS
    for rsep in (0.6, 1.0, 1.4):
        A, B, C, D = ellc_axes(rv_a, rsep, hf, 1.0 / mu)
        with mx.stream(mx.cpu):
            q = OT._tidal_q_solve(F64(0.5 * hf / mu * rv_a ** 3 / rsep ** 3)).item()
        assert A / B - 1.0 == pytest.approx(3.0 * q, rel=1e-12)
        assert 1.0 - C / B == pytest.approx(q, rel=1e-12)
        assert (A * B * C) ** (1.0 / 3.0) == pytest.approx(rv_a, rel=1e-13)
        assert D == pytest.approx(B ** 5 / mu, rel=1e-14)


def test_q_from_hf():
    hf, a, rv = 1.59, A_RS, RV
    A, B, C, D = ellc_axes(rv / a, 1.0, hf, 1.0 / MU)
    q = el.tidal_q_from_hf(hf, MU, a, rv)
    assert q == pytest.approx(1.0 - C / B, rel=1e-12)
    assert el.tidal_hf_from_q(q, MU, a, rv) == pytest.approx(hf, rel=1e-12)
    # the first-order relation sets q a little high, as ellc's r2 < R_V
    assert 0.5 * hf / MU * (rv / a) ** 3 == pytest.approx(g3(q), rel=1e-12)
    with mx.stream(mx.cpu):
        g = mx.grad(lambda h: el.tidal_q_from_hf(h, F64(MU), F64(a), F64(rv)))(F64(hf))
        mx.eval(g)
    d = 1e-6
    fd = (el.tidal_q_from_hf(hf + d, MU, a, rv) - el.tidal_q_from_hf(hf - d, MU, a, rv)) / (2 * d)
    assert g.item() == pytest.approx(fd, rel=1e-8)


def test_mass_ratio_is_optional_and_only_relabels_a_circular_orbit():
    """Without it there is no offset. With it, on a circular orbit, it is
    exactly the same model at a - D / (1 + mu), at the same inclination, so
    no transit fit can tell the two apart."""
    r2 = axes(RV, 0.05)[1]
    D = r2 ** 5 / (MU * A_RS ** 4) / (1.0 + MU)
    a1 = A_RS - D
    got = flux(TAU, 0.3, 0.05, mass_ratio=MU)
    relabelled = flux(TAU, 0.3 * a1 / A_RS, 0.05, a=a1)
    assert np.abs(got - relabelled).max() < 1e-14
    assert np.abs(got - flux(TAU, 0.3, 0.05)).max() > 1e-6


def _exposure_reference(tau, q, b, n=161, **orb):
    """60 s averaged by a fine trapezium on the oracle: good to ~2e-9 at
    the contacts, the limit of the check below."""
    x = np.linspace(-0.5 * EXP, 0.5 * EXP, n)
    w = np.full(x.size, 1.0)
    w[0] = w[-1] = 0.5
    w /= w.sum()
    out = np.zeros(tau.size)
    for i, t in enumerate(tau):
        out[i] = w @ reference_flux(t + x, RV, q, b, "hybrid5", W, **orb)
    return out


@pytest.mark.parametrize("q,b,tw,e,om,mu", [(0.05, 0.08, 0.0, 0.0, 1.57, None),
                                            (0.1, 0.5, 0.0, 0.0, 1.57, None),
                                            (0.05, 0.3, 0.7, 0.3, 2.0, None),
                                            (0.05, 0.3, 0.0, 0.3, 2.0, MU_BIG)])
def test_exposures_match_the_reference(q, b, tw, e, om, mu):
    """Around the contacts, where the signal is."""
    orb = kh(e, om) if e else {}
    c = np.array([float(v) for v in el.contact_offsets_tidal(
        A_RS, b, RV, q, tw, orb.get("secosw"), orb.get("sesinw"), mass_ratio=mu)])
    # a phase offset on an eccentric orbit: the time is what the test needs
    scale = P / (2.0 * math.pi)
    tau = (c * scale)[:, None] + np.array([-EXP, -0.2 * EXP, 0.2 * EXP, EXP])
    tau = tau.ravel()
    ref = _exposure_reference(tau, q, b, tw=tw, e=e, om=om, mu=mu)
    kw = dict(exp_time=EXP, theta=tw, mass_ratio=mu, **orb)
    assert np.abs(flux(tau, b, q, integration="contact", n_gl=5, **kw) - ref).max() < 5e-9
    # equal weights on uniform nodes: first order, and still under 1 ppm
    assert np.abs(flux(tau, b, q, integration="supersample", n_sub=41, **kw) - ref).max() < 1e-6


def test_contact_rule_converges():
    kw = dict(exp_time=EXP, integration="contact")
    fine = flux(TAU, 0.3, 0.08, n_gl=40, **kw)
    assert np.abs(flux(TAU, 0.3, 0.08, n_gl=5, **kw) - fine).max() < 2e-9


# -- contacts ------------------------------------------------------------

def _ref_contacts(a, b, rv, q, tw=0.0, e=0.0, om=math.pi / 2, per=2.0 * math.pi,
                  mu=None):
    """Dense scan and bisection on the sampled outline: distance from the
    star's centre to the disc = 1, and largest boundary distance = 1."""
    psi = np.linspace(0.0, 2.0 * math.pi, 7201)
    ci = cos_i(a, b, e, om)

    def gs(phi):
        S, X, Y, _ = instant(phi, a, ci, per, e, om, rv, q, tw, mu)
        L = np.linalg.cholesky(S)
        P_ = L @ np.stack([np.cos(psi), np.sin(psi)])
        d2 = (X + P_[0]) ** 2 + (Y + P_[1]) ** 2
        return d2.min(), d2.max()

    def root(lo, hi, j):
        g_lo = gs(lo)[j] - 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            g = gs(mid)[j] - 1.0
            if g * g_lo > 0.0:
                lo, g_lo = mid, g
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return (root(-0.9, 0.0, 0), root(-0.9, 0.0, 1), root(0.9, 0.0, 1),
            root(0.9, 0.0, 0))


@pytest.mark.parametrize("q,b,a,tw,e,om,mu", [
    (0.05, 0.08, A_RS, 0.0, 0.0, 1.57, None), (0.05, 0.5, A_RS, 0.0, 0.0, 1.57, None),
    (-0.05, 0.3, A_RS, 0.0, 0.0, 1.57, None), (0.2, 0.5, 2.2, 0.0, 0.0, 1.57, None),
    (0.2, 0.5, 2.2, 0.7, 0.0, 1.57, None), (0.1, 0.3, A_RS, -0.9, 0.3, 2.0, None),
    (0.05, 0.3, A_RS, 0.0, 0.0, 1.57, MU_BIG), (0.1, 0.3, A_RS, 0.0, 0.3, 2.0, MU_BIG)])
def test_contacts_match_a_dense_reference(q, b, a, tw, e, om, mu):
    orb = kh(e, om) if e else {}
    ref = _ref_contacts(a, b, RV, q, tw, e, om, mu=mu)
    got = [float(v) for v in el.contact_offsets_tidal(
        a, b, RV, q, tw, orb.get("secosw"), orb.get("sesinw"), mass_ratio=mu)]
    assert got == pytest.approx(ref, abs=2e-6)


@pytest.mark.skipif(not GPU, reason="needs Metal")
def test_contact_kernel_matches_the_graph():
    from elome.metal_oblate import contact_offsets_oblate_kernel
    a = mx.array([A_RS, A_RS, 2.2, A_RS])
    b = mx.array([0.08, 0.5, 0.5, 1.05])
    r = mx.array([RV, RV, RV, RV])
    q = mx.array([0.05, -0.05, 0.2, 0.05])
    tw = mx.array([0.0, 0.4, -0.7, 0.2])
    with mx.stream(mx.gpu):
        r2 = OT._tidal_r2(r, q)
        k = np.asarray(contact_offsets_oblate_kernel(a, b, r2, q, tw, tidal=True))
    g = np.stack([np.asarray(v) for v in el.contact_offsets_tidal(a, b, r, q, tw)], axis=1)
    assert np.abs(k - g).max() < 2e-4


# -- limits --------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, dict(exp_time=EXP, integration="contact"),
                                dict(exp_time=EXP, integration="supersample", n_sub=5)])
@pytest.mark.parametrize("orb", [{}, kh(0.3, 2.0)], ids=["circ", "ecc"])
def test_q_zero_is_the_spherical_path(kw, orb):
    tau = F64(TAU)
    with mx.stream(mx.cpu):
        sph = el.flux_dev_from_tau(tau, P, A_RS, 0.3, RV, limb_dark="hybrid5",
                                   u=W, **kw, **orb)
        mx.eval(sph)
    # a sphere has no twist
    got = flux(TAU, 0.3, 0.0, theta=0.8, **kw, **orb)
    assert np.abs(got - np.asarray(sph)).max() < 1e-13


def test_zero_eccentricity_is_the_circular_path():
    a = flux(TAU, 0.3, 0.05, theta=0.4)
    assert np.abs(flux(TAU, 0.3, 0.05, theta=0.4, secosw=0.0, sesinw=0.0) - a).max() < 1e-13


def test_time_symmetry():
    """-phi is +phi mirrored in X with the angle reversed: on a circular
    orbit without a twist, the tilt does not make ingress and egress differ."""
    for kw in ({}, dict(exp_time=EXP, integration="contact")):
        a = flux(TAU, 0.5, 0.1, **kw)
        assert np.abs(a - a[::-1]).max() < 1e-11


def test_mass_ratio_needs_tidal_q():
    with pytest.raises(ValueError, match="mass_ratio"):
        el.flux_dev_from_tau(mx.array(TAU), P, A_RS, 0.3, RV, limb_dark="hybrid5",
                             u=W, mass_ratio=MU)


def test_a_twist_breaks_the_symmetry():
    a = flux(TAU, 0.5, 0.1, theta=0.6)
    assert np.abs(a - a[::-1]).max() > 1e-6
    c = [float(v) for v in el.contact_offsets_tidal(A_RS, 0.5, RV, 0.1, 0.6)]
    assert abs(c[0] + c[3]) > 1e-5
    # a half-turn about the nose is the same ellipsoid
    assert np.abs(flux(TAU, 0.5, 0.1, theta=0.6 + math.pi) - a).max() < 1e-12


def test_the_light_curve_is_smooth_through_q_zero():
    """The slopes either side of q = 0 part as h^2 (curvature), not as h (a
    kink), and the central slope is the wider one's."""
    for tw in (0.0, 0.7):
        def gap(h):
            d = [flux(TAU, 0.3, q, theta=tw) for q in (-2 * h, -h, h, 2 * h)]
            return np.abs((d[3] - d[2]) - (d[1] - d[0])).max(), d
        wide, d = gap(1e-3)
        fine = gap(2.5e-4)[0]
        assert wide / fine == pytest.approx(16.0, rel=0.02)
        assert np.abs((d[2] - d[1]) / 2e-3).max() > 1e-3


def test_each_instant_is_the_oblate_mode():
    """Mid-transit: the same (r, f, theta) in f= gives the same flux."""
    r, f, th = outline(0.0, 1.0, 0.3 / A_RS, RV, 0.05, 0.4)
    tau = F64([0.0])
    with mx.stream(mx.cpu):
        obl = el.flux_dev_from_tau(tau, P, A_RS, 0.3, r, limb_dark="hybrid5",
                                   u=W, f=f, theta=th)
        mx.eval(obl)
    assert flux([0.0], 0.3, 0.05, theta=0.4)[0] == pytest.approx(np.asarray(obl)[0], abs=1e-15)


def test_the_outline_grows_towards_the_limb():
    """What separates it from the oblate mode: a fixed shape cannot do this."""
    q, b = 0.05, 0.08
    r0, f0, th0 = outline(0.0, 1.0, b / A_RS, RV, q)
    with mx.stream(mx.cpu):
        obl = el.flux_dev_from_tau(F64(TAU), P, A_RS, b, r0, limb_dark="hybrid5",
                                   u=W, f=f0, theta=th0)
        mx.eval(obl)
    d = flux(TAU, b, q) - np.asarray(obl)
    assert np.abs(d)[np.abs(TAU) < 1e-4].max() < 1e-12
    assert np.abs(d).max() > 1e-5


@pytest.mark.parametrize("orb", [{}, kh(0.3, 2.0)], ids=["circ", "ecc"])
def test_out_of_transit_and_far_side_are_exactly_zero(orb):
    tau = np.array([-0.45, -0.3, 0.3, 0.45]) * P
    assert np.all(flux(tau, 0.3, 0.05, theta=0.4, **orb) == 0.0)


def test_per_chain_q_and_batches():
    q = mx.array(np.array([0.05, 0.0, -0.03]), dtype=mx.float64)
    tw = mx.array(np.array([0.0, 0.3, -0.4]), dtype=mx.float64)
    with mx.stream(mx.cpu):
        out = el.flux_dev_from_tau(F64(TAU), P, A_RS, 0.3, RV,
                                   limb_dark="hybrid5", u=W, tidal_q=q, theta=tw)
        mx.eval(out)
    for j, (qj, tj) in enumerate(((0.05, 0.0), (0.0, 0.3), (-0.03, -0.4))):
        assert np.abs(np.asarray(out)[j] - flux(TAU, 0.3, qj, theta=tj)).max() < 1e-15


@pytest.mark.parametrize("orb", [{}, kh(0.3, 2.0)], ids=["circ", "ecc"])
def test_ld_basis_identity(orb):
    from elome.hybrid import hybrid_norms
    B = flux(TAU, 0.3, 0.05, ld_basis=True, u=None, theta=0.3, **orb)
    c = np.concatenate([[1.0], -np.array(W)])
    N = hybrid_norms("hybrid5")
    assert np.abs(B @ c / (N @ c) - flux(TAU, 0.3, 0.05, theta=0.3, **orb)).max() < 1e-15


# -- precision -----------------------------------------------------------

CASES = [dict(), dict(exp_time=EXP, integration="contact"),
         dict(exp_time=EXP, integration="supersample", n_sub=7)]


@pytest.mark.skipif(not GPU, reason="needs Metal")
@pytest.mark.parametrize("kw", CASES, ids=["none", "contact", "super"])
@pytest.mark.parametrize("law", list(LAWS))
@pytest.mark.parametrize("q,b,tw,orb", [
    (0.05, 0.08, 0.0, {}), (0.1, 0.5, 0.0, {}), (-0.05, 0.3, 0.0, {}),
    (0.0, 0.3, 0.0, {}), (1e-4, 0.3, 0.0, {}), (0.05, 0.3, 0.7, {}),
    (0.05, 0.3, 0.0, kh(0.13, 0.6)), (0.05, 0.2, 0.7, kh(0.25, 2.5))])
@pytest.mark.parametrize("mu", [None, MU_BIG], ids=["no-offset", "offset"])
def test_kernel_matches_fp64(kw, law, q, b, tw, orb, mu):
    """The target is 1 ppm; the kernels stay ten times inside it."""
    args = dict(theta=tw, mass_ratio=mu, **kw, **orb)
    d = np.abs(flux(TAU, b, q, law, dtype=mx.float32, **args)
               - flux(TAU, b, q, law, **args)).max()
    assert d < 1.5e-7


@pytest.mark.skipif(not GPU, reason="needs Metal")
def test_fp32_graph_matches_fp64():
    tau = mx.array(TAU, dtype=mx.float32)
    with mx.stream(mx.cpu):
        out = el.flux_dev_from_tau(tau, P, A_RS, 0.3, RV, limb_dark="hybrid5",
                                   u=W, tidal_q=0.05)
        mx.eval(out)
    assert np.abs(np.asarray(out, np.float64) - flux(TAU, 0.3, 0.05)).max() < 3e-7


# -- gradients -----------------------------------------------------------

def _value_and_grad(dt, stream, per, a, b, rv, q, tw, k, h, mu, w, ecc=False, **kw):
    rng = np.random.default_rng(7)
    ct = rng.normal(size=TAU.size)
    with mx.stream(stream):
        t = mx.array(TAU, dtype=dt)
        c = mx.array(ct, dtype=dt)
        ps = [mx.array([v], dtype=dt) for v in (per, a, b, rv, q, tw, k, h, mu)]

        def L(tt, per_, a_, b_, r_, q_, tw_, k_, h_, mu_, w_):
            orb = dict(secosw=k_, sesinw=h_) if ecc else {}
            return mx.sum(c * el.flux_dev_from_tau(
                tt, per_, a_, b_, r_, limb_dark="hybrid5", u=w_, tidal_q=q_,
                theta=tw_, mass_ratio=mu_, **orb, **kw))
        v, g = mx.value_and_grad(L, argnums=list(range(11)))(
            t, *ps, mx.array(w, dtype=dt))
        mx.eval(v, *g)
    return (float(v.item()), [np.asarray(x, dtype=np.float64).ravel() for x in g])


GRAD_CASES = [(0.05, 0.08, 0.0, 0.0, 0.0), (-0.05, 0.3, 0.0, 0.0, 0.0),
              (0.0, 0.3, 0.0, 0.0, 0.0), (1e-6, 0.3, 0.0, 0.0, 0.0),
              (-1e-6, 0.3, 0.0, 0.0, 0.0), (0.05, 0.3, 0.6, 0.0, 0.0),
              (0.05, 0.2, 0.7, -0.4, 0.3), (0.0, 0.3, 0.4, 0.3, -0.2)]
GRAD_MU = 0.05


@pytest.mark.parametrize("kw", CASES[:2], ids=["none", "contact"])
@pytest.mark.parametrize("q,b,tw,k,h", GRAD_CASES)
def test_gradients_match_finite_differences(kw, q, b, tw, k, h):
    ecc = (k, h) != (0.0, 0.0)
    v0 = [P, A_RS, b, RV, q, tw, k, h, GRAD_MU]
    call = lambda v: _value_and_grad(mx.float64, mx.cpu, *v, W, ecc=ecc, **kw)
    _, g = call(v0)
    for j in (range(9) if ecc else (0, 1, 2, 3, 4, 5, 8)):
        d = 1e-6 * max(abs(v0[j]), 0.1)
        up, dn = list(v0), list(v0)
        up[j] += d
        dn[j] -= d
        fd = (call(up)[0] - call(dn)[0]) / (2 * d)
        # contacts are detached split points: FD moves them, autodiff does
        # not (test_oblate_tau's note), so the rule's own error shows here:
        # measured 1e-6 at 5 nodes, 1e-7 at 12, 3e-9 at 40
        rel, tol = (1e-4, 2e-6) if kw else (3e-5, 1e-7)
        assert g[j + 1][0] == pytest.approx(fd, rel=rel, abs=tol), j


@pytest.mark.skipif(not GPU, reason="needs Metal")
@pytest.mark.parametrize("kw", CASES, ids=["none", "contact", "super"])
@pytest.mark.parametrize("q,b,a,tw,k,h", [
    (0.05, 0.08, A_RS, 0.0, 0.0, 0.0), (0.1, 0.5, A_RS, 0.0, 0.0, 0.0),
    (-0.05, 0.3, A_RS, 0.0, 0.0, 0.0), (0.0, 0.3, A_RS, 0.0, 0.0, 0.0),
    (1e-4, 0.3, A_RS, 0.0, 0.0, 0.0), (-1e-4, 0.3, A_RS, 0.0, 0.0, 0.0),
    (0.1, 0.8, 1.6, 0.0, 0.0, 0.0), (0.05, 0.3, A_RS, 0.5, 0.0, 0.0),
    (-0.05, 0.3, A_RS, -0.8, 0.0, 0.0), (0.05, 0.3, A_RS, 0.0, 0.3, 0.2),
    (0.05, 0.2, A_RS, 0.7, -0.4, 0.3), (-0.03, 0.1, A_RS, 2.0, 0.1, 0.5)])
def test_kernel_gradients_match_fp64(kw, q, b, a, tw, k, h):
    """Every parameter, and tau: the kernel's analytic chain against
    autodiff of the fp64 graph. d/dtwist vanishes with q, so its scale is
    q's slope times q."""
    ecc = (k, h) != (0.0, 0.0)
    v0 = [P, a, b, RV, q, tw, k, h, GRAD_MU]
    g32 = _value_and_grad(mx.float32, mx.gpu, *v0, W, ecc=ecc, **kw)[1]
    g64 = _value_and_grad(mx.float64, mx.cpu, *v0, W, ecc=ecc, **kw)[1]
    for j, (x, y) in enumerate(zip(g32, g64)):
        if not ecc and j in (7, 8):
            continue
        s = np.abs(y).max()
        if j == 6:
            s = max(s, abs(q) * np.abs(g64[5]).max())
            if q == 0.0:
                assert np.abs(x).max() < 1e-4 * np.abs(g64[5]).max()
                continue
            assert np.abs(x - y).max() < 2e-2 * s
            continue
        assert np.abs(x - y).max() < 1e-3 * s


@pytest.mark.skipif(not GPU, reason="needs Metal")
def test_kernel_gradient_at_a_round_outline_is_finite():
    q = 0.02
    ci = math.sqrt((2.0 - q) / (8.0 + 8.0 * q))
    a = 1.6
    g = _value_and_grad(mx.float32, mx.gpu, P, a, a * ci, 0.1, q, 0.0, 0.0, 0.0, 0.0, W)[1]
    assert all(np.all(np.isfinite(x)) for x in g)


def test_gradients_are_continuous_through_q_zero():
    gs = [_value_and_grad(mx.float64, mx.cpu, P, A_RS, 0.3, RV, q, 0.5, 0.0, 0.0, 0.0, W)[1]
          for q in (-1e-7, 0.0, 1e-7)]
    for j in range(1, 6):
        s = np.abs(gs[1][j]).max()
        assert np.abs(gs[0][j] - gs[2][j]).max() < 3e-4 * s
    # d/dtwist goes with q: nothing at the sphere, and its sign at either side
    assert abs(gs[1][6][0]) < 1e-15
    assert gs[0][6][0] == pytest.approx(-gs[2][6][0], rel=1e-5)


# -- the frontend --------------------------------------------------------

def params(q=0.05, law="hybrid5", theta=None):
    p = el.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w = 0.0, P, RV, A_RS, 0.0, 90.0
    p.inc = math.degrees(math.acos(0.3 / A_RS))
    p.limb_dark, p.u = law, LAWS[law]
    p.tidal_q, p.theta = q, theta
    return p


def test_frontend_matches_the_sampler_api():
    p = params(theta=20.0)
    m = el.TransitModel(p, TAU, dtype=mx.float64)
    ref = flux(TAU, 0.3, 0.05, theta=math.radians(20.0))
    assert np.abs(m.light_curve(p) - 1.0 - ref).max() < 1e-13


def test_frontend_eccentric():
    p = params(theta=20.0)
    e, om = 0.2, 60.0
    p.ecc, p.w = e, om
    p.inc = math.degrees(math.acos(cos_i(A_RS, 0.3, e, math.radians(om))))
    m = el.TransitModel(p, TAU, dtype=mx.float64)
    ref = flux(TAU, 0.3, 0.05, theta=math.radians(20.0), **kh(e, math.radians(om)))
    assert np.abs(m.light_curve(p) - 1.0 - ref).max() < 1e-11
    with mx.stream(mx.cpu):
        p.ecc = F64(e)                     # the (e, w) route
        out = m.light_curve_mx(p)
        mx.eval(out)
    assert np.abs(np.asarray(out) - 1.0 - ref).max() < 1e-11


@pytest.mark.skipif(not GPU, reason="needs Metal")
def test_frontend_fp32_ew_route():
    """An array ecc takes the (e, w) kernels."""
    p = params(theta=20.0)
    p.ecc, p.w = 0.2, 60.0
    m = el.TransitModel(p, TAU, dtype=mx.float32)
    want = m.light_curve(p)
    p.ecc = mx.array(0.2, dtype=mx.float32)
    out = m.light_curve_mx(p)
    mx.eval(out)
    assert np.abs(np.asarray(out, dtype=np.float64) - want).max() < 3e-7


def test_frontend_mass_ratio():
    p = params()
    p.mass_ratio = MU_BIG
    m = el.TransitModel(p, TAU, dtype=mx.float64)
    assert np.abs(m.light_curve(p) - 1.0 - flux(TAU, 0.3, 0.05, mass_ratio=MU_BIG)).max() < 1e-13
    ps = [params(0.05), p]
    out = m.light_curves(ps)
    assert np.abs(out[0] - 1.0 - flux(TAU, 0.3, 0.05)).max() < 1e-13
    assert np.abs(out[1] - m.light_curve(p)).max() < 1e-13


def test_frontend_exposure_and_update():
    p = params()
    m = el.TransitModel(p, TAU, dtype=mx.float64, exp_time=EXP,
                        integration="contact")
    a = m.light_curve(p)
    p.tidal_q = 0.02
    assert np.abs(m.light_curve(p) - a).max() > 1e-6
    ref = flux(TAU, 0.3, 0.02, exp_time=EXP, integration="contact", n_gl=m.n_gl)
    assert np.abs(m.light_curve(p) - 1.0 - ref).max() < 1e-13


def test_frontend_light_curves_batch():
    ps = [params(0.05), params(0.0, theta=10.0), params(-0.03, theta=-40.0)]
    m = el.TransitModel(ps[0], TAU, dtype=mx.float64)
    out = m.light_curves(ps)
    for j, p in enumerate(ps):
        assert np.abs(out[j] - m.light_curve(p)).max() < 1e-13


def test_frontend_light_curve_mx_is_differentiable():
    p = params()
    m = el.TransitModel(p, TAU, dtype=mx.float64)
    with mx.stream(mx.cpu):
        def L(q):
            p.tidal_q = q
            return mx.sum(m.light_curve_mx(p))
        g = mx.grad(L)(F64(0.05)).item()
    d = 1e-6
    p.tidal_q = 0.05 + d
    up = m.light_curve(p).sum()
    p.tidal_q = 0.05 - d
    dn = m.light_curve(p).sum()
    assert g == pytest.approx((up - dn) / (2 * d), rel=1e-5)


def test_frontend_refusals():
    p = params()
    p.f = 0.1
    with pytest.raises(ValueError, match="planet's shape"):
        el.TransitModel(p, TAU)
    p = params()
    p.fp = 1e-3
    with pytest.raises(ValueError, match="primary"):
        el.TransitModel(p, TAU, transittype="secondary")
    p = params()
    p.limb_dark, p.u = "quadratic", [0.3, 0.2]
    with pytest.raises(ValueError, match="hybrid"):
        el.TransitModel(p, TAU)
    p = params()
    m = el.TransitModel(p, TAU)
    p.tidal_q = None
    with pytest.raises(ValueError, match="tidal_q changed"):
        m.light_curve(p)


# -- the keywords --------------------------------------------------------

@pytest.mark.parametrize("kw,msg", [
    (dict(f=0.1), "planet's shape"),
    (dict(ring_in=1.2, ring_out=2.0, ring_f=0.5), "planet's shape"),
    (dict(transittype="secondary", fp=1e-3), "primary"),
    (dict(limb_dark="quadratic", u=None, u1=0.3, u2=0.2), "hybrid"),
])
def test_keyword_errors(kw, msg):
    args = dict(limb_dark="hybrid5", u=W, tidal_q=0.05)
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        el.flux_dev_from_tau(mx.array(TAU), P, A_RS, 0.3, RV, **args)


@pytest.mark.parametrize("q", [-1.0 / 3.0, -0.5, 1.0])
def test_domain_host(q):
    with pytest.raises(ValueError, match="tidal_q must be"):
        flux(TAU, 0.3, q)


def test_domain_at_periastron():
    """A shape that is legal at a can pass the cubic's range at periastron."""
    q = mx.array([0.05, 0.3], dtype=mx.float64)
    with mx.stream(mx.cpu):
        out = el.flux_dev_from_tau(F64(TAU), P, A_RS, 0.3, RV, limb_dark="hybrid5",
                                   u=W, tidal_q=q, secosw=0.7, sesinw=0.0)
        mx.eval(out)
    assert np.isnan(np.asarray(out)).all(axis=1).tolist() == [False, True]


def test_domain_large_planet():
    with pytest.raises(ValueError, match="long axis"):
        flux(TAU, 0.3, 0.2, rv=0.6)


@pytest.mark.parametrize("dtype", [mx.float64] + ([mx.float32] if GPU else []),
                         ids=["fp64", "fp32"][:2 if GPU else 1])
def test_domain_arrays_are_nan(dtype):
    q = mx.array([0.05, -0.5, 1.2], dtype=dtype)
    tau = mx.array(TAU, dtype=dtype)
    with mx.stream(mx.cpu if dtype == mx.float64 else mx.gpu):
        out = el.flux_dev_from_tau(tau, P, A_RS, 0.3, RV, limb_dark="hybrid5",
                                   u=W, tidal_q=q)
        mx.eval(out)
    nan = np.isnan(np.asarray(out, dtype=np.float64)).all(axis=1)
    assert nan.tolist() == [False, True, True]
