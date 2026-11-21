"""A planet with a moon (elome/moons.py): the sky positions against an
independent rotation-matrix construction and against LUNA's equations
(Kipping 2011) after converting the elements, the symmetries the
convention promises, float32 phases far from t0, and gradients."""

import math

import mlx.core as mx
import numpy as np
import pytest

from elome import moons
from elome.orbit import epoch_center_times, tau_from_epochs

PI = math.pi


# --------------------------------------------------------------------------
# independent references (numpy, float64)
# --------------------------------------------------------------------------

def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def true_anomaly(M, e):
    E = np.array(M, np.float64)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def m_conj(e, w):
    """Mean anomaly at conjunction, f = pi/2 - w."""
    f = PI / 2 - w
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(f / 2), math.sqrt(1 + e) * math.cos(f / 2))
    return E - e * math.sin(E)


def orbit_vec(M, a, e):
    f = true_anomaly(M, e)
    r = a * (1 - e * e) / (1 + e * np.cos(f))
    return np.stack([r * np.cos(f), r * np.sin(f), 0 * f])


def ref_positions(t, p):
    """Planet and moon (3, n) in the sky frame: the barycentre's orbit is
    R_z(pi) R_x(inc) R_z(w) r, the moon's offset R_z(node + pi) R_x(moon_inc)
    R_z(moon_w) r_m, both Murray & Dermott with the observer on +Z."""
    MB = 2 * PI * (t - p["t0"]) / p["per"] + m_conj(p["ecc"], p["w"])
    B = Rz(PI) @ Rx(p["inc"]) @ Rz(p["w"]) @ orbit_vec(MB, p["a"], p["ecc"])
    Mm = 2 * PI * (t - p["t0"]) / p["moon_per"] + p["phase"] + PI / 2 - p["moon_w"]
    d = (Rz(p["node"] + PI) @ Rx(p["moon_inc"]) @ Rz(p["moon_w"])
         @ orbit_vec(Mm, p["a_ps"], p["moon_ecc"]))
    q = p["q"]
    return B - q / (1 + q) * d, B + d / (1 + q), d


def elome_positions(t, p, dt=mx.float64, per_ref=None, moon_per_ref=None, t0_ref=None):
    """moons.py's positions, with the period / t0 references at the true
    values unless given (the split the float32 graph uses)."""
    per_ref = p["per"] if per_ref is None else per_ref
    moon_per_ref = p["moon_per"] if moon_per_ref is None else moon_per_ref
    t0_ref = p["t0"] if t0_ref is None else t0_ref
    with mx.stream(mx.cpu):
        ep = epoch_center_times(t, t0_ref, per_ref)
        k_np = ep[1]
        dtt, k = mx.array(ep[0], dtype=dt), mx.array(k_np, dtype=dt)
        f = lambda v: mx.array(v, dtype=dt)
        t0_off, p_off, m_off = p["t0"] - t0_ref, p["per"] - per_ref, p["moon_per"] - moon_per_ref
        tau = tau_from_epochs(dtt, k, f(t0_off), f(p_off), f(p["per"]))
        phi = (2 * PI / f(p["per"])) * tau
        XB, YB, ZB = moons.barycentre_sky(phi, f(p["a"]), f(p["inc"]),
                                          moons.anchored_constants(p["ecc"], p["w"], dt))
        turns = f(moons.moon_turns_ref(k_np, per_ref, moon_per_ref))
        phi_m = moons.moon_phase(tau, k, turns, per_ref, moon_per_ref, f(p_off), f(m_off),
                                 f(p["phase"]))
        d = moons.moon_offset(phi_m, f(p["a_ps"]), f(p["moon_inc"]), f(p["node"]),
                              moons.moon_orbit(p["moon_ecc"], p["moon_w"], dt))
        P, M = moons.body_positions(XB, YB, ZB, *d, f(p["q"]))
        out = [np.stack([np.array(c) for c in v]) for v in (P, M, d)]
    return out


def random_system(rng, ecc=True, moon_ecc=True):
    return dict(t0=rng.uniform(-1, 1), per=rng.uniform(20, 80), a=rng.uniform(40, 120),
                inc=math.radians(rng.uniform(87, 90)), ecc=rng.uniform(0, 0.4) if ecc else 0.0,
                w=rng.uniform(-PI, PI), a_ps=rng.uniform(0.02, 0.2), q=rng.uniform(0, 0.3),
                moon_per=rng.uniform(0.7, 6), moon_inc=rng.uniform(0, PI),
                node=rng.uniform(-PI, PI), moon_ecc=rng.uniform(0, 0.3) if moon_ecc else 0.0,
                moon_w=rng.uniform(-PI, PI), phase=rng.uniform(-PI, PI))


# a window round each of several transits, the far side excluded
def transit_times(p, epochs=(-3, 0, 2, 7)):
    return np.concatenate([p["t0"] + k * p["per"] + np.linspace(-0.6, 0.6, 301) for k in epochs])


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ecc,moon_ecc", [(False, False), (True, False), (False, True),
                                          (True, True)])
def test_positions_match_the_rotation_matrices(ecc, moon_ecc):
    rng = np.random.default_rng(1 + 2 * ecc + moon_ecc)
    for _ in range(25):
        p = random_system(rng, ecc, moon_ecc)
        t = transit_times(p)
        ref = ref_positions(t, p)
        got = elome_positions(t, p)
        for r, g in zip(ref, got):              # Z ~ a: rounding scales with a
            assert np.abs(r - g).max() < 5e-14 * p["a"]


def luna_separations(t, L):
    """LUNA's S_P*, S_S*, S_PS (Kipping 2011, appendix), with its moon
    inclination taken as the mutual inclination (0: coplanar prograde)."""
    ib, wb, es, ws, isb, Os = L["ib"], L["wb"], L["es"], L["ws"], L["is"], L["Os"]
    a_sb = L["a_ps"] / (1 + L["q"])
    fb = true_anomaly(2 * PI * (t - L["Tb"]) / L["Pb"] + m_conj(L["eb"], wb), L["eb"])
    ts = t - L["Tb"] - L["phi_s"] * L["Ps"] / (2 * PI)
    fs = true_anomaly(2 * PI * ts / L["Ps"] + m_conj(es, ws), es)
    vb = L["ab"] * (1 - L["eb"] ** 2) / (1 + L["eb"] * np.cos(fb))
    vs = a_sb * (1 - es ** 2) / (1 + es * np.cos(fs))
    cO, sO = math.cos(wb + Os), math.sin(wb + Os)
    b1, b2 = -math.cos(isb) * sO, cO
    g1 = math.cos(ib) * math.cos(isb) * cO - math.sin(ib) * math.sin(isb)
    g2 = math.cos(ib) * sO
    us, ub = fs + ws, fb + wb
    beta = vs * (b1 * np.sin(us) + b2 * np.cos(us))
    gamma = vs * (g1 * np.sin(us) + g2 * np.cos(us))
    X0, Y0 = vb * np.cos(ub), vb * math.cos(ib) * np.sin(ub)
    q = L["q"]
    return (np.hypot(X0 - q * beta, Y0 - q * gamma), np.hypot(X0 + beta, Y0 + gamma),
            (1 + q) * np.hypot(beta, gamma))


def from_luna(L):
    """LUNA's elements -> moons.py's. LUNA's sky frame is moons.py's turned by
    pi about the line of sight, and its moon orbit is
    R_x(ib) R_z(wb + Os) R_x(is) R_z(ws) in it, so (node, moon_inc, moon_w)
    are that matrix's z-x-z angles; the mean anomaly at t0, M + w - pi/2,
    is the phase."""
    R = Rx(L["ib"]) @ Rz(L["wb"] + L["Os"]) @ Rx(L["is"]) @ Rz(L["ws"])
    node = math.atan2(R[0, 2], -R[1, 2])
    moon_inc = math.acos(max(-1.0, min(1.0, R[2, 2])))
    moon_w = math.atan2(R[2, 0], R[2, 1])
    M_t0 = -L["phi_s"] + m_conj(L["es"], L["ws"])
    return dict(t0=L["Tb"], per=L["Pb"], a=L["ab"], inc=L["ib"], ecc=L["eb"], w=L["wb"],
                a_ps=L["a_ps"], q=L["q"], moon_per=L["Ps"], moon_inc=moon_inc, node=node,
                moon_ecc=L["es"], moon_w=moon_w, phase=M_t0 + moon_w - PI / 2)


def test_agrees_with_luna_after_converting_the_elements():
    rng = np.random.default_rng(7)
    for _ in range(40):
        L = dict(Tb=rng.uniform(-1, 1), Pb=rng.uniform(20, 80), ab=rng.uniform(40, 120),
                 ib=math.radians(rng.uniform(87, 90)), eb=rng.uniform(0, 0.3),
                 wb=rng.uniform(0, 2 * PI), a_ps=rng.uniform(0.02, 0.2), q=rng.uniform(0, 0.3),
                 Ps=rng.uniform(0.7, 6), phi_s=rng.uniform(0, 2 * PI), es=rng.uniform(0, 0.3),
                 ws=rng.uniform(0, 2 * PI), Os=rng.uniform(-PI, PI), **{"is": rng.uniform(0, PI)})
        p = from_luna(L)
        t = transit_times(p)
        SP, SS, SPS = luna_separations(t, L)
        P, M, d = elome_positions(t, p)
        assert np.abs(np.hypot(P[0], P[1]) - SP).max() < 1e-11
        assert np.abs(np.hypot(M[0], M[1]) - SS).max() < 1e-11
        assert np.abs(np.hypot(d[0], d[1]) - SPS).max() < 1e-11


def test_a_coplanar_moon_has_node_zero_and_the_planets_inclination():
    """moon_inc = inc, node = 0: the two orbit normals coincide, and at its
    conjunction the moon moves along +X relative to the planet, as the
    barycentre does at transit."""
    rng = np.random.default_rng(3)
    p = random_system(rng, ecc=True, moon_ecc=False)
    p.update(moon_inc=p["inc"], node=0.0, phase=0.0)
    t = p["t0"] + np.array([-1e-5, 0.0, 1e-5])
    P, M, d = elome_positions(t, p)
    _, _, d_far = elome_positions(t + 0.25 * p["moon_per"], p)
    n_moon = np.cross(d[:, 1], d_far[:, 1])
    n_planet = np.array([0.0, math.sin(p["inc"]), math.cos(p["inc"])])
    assert np.allclose(n_moon / np.linalg.norm(n_moon), n_planet, atol=1e-12)
    assert d[2, 1] > 0 and abs(d[0, 1]) < 1e-12 and d[0, 2] > d[0, 0]
    assert np.allclose(moons.mutual_inclination(math.degrees(p["inc"]),
                                                math.degrees(p["inc"]), 0.0)[0], 0.0, atol=1e-6)


def test_the_moons_orbit_does_not_move_with_the_planets_periastron():
    rng = np.random.default_rng(4)
    p = random_system(rng, ecc=True, moon_ecc=True)
    t = transit_times(p)
    d1 = elome_positions(t, p)[2]
    p2 = dict(p, ecc=0.1, w=p["w"] + 1.3)
    d2 = elome_positions(t, p2)[2]
    assert np.abs(d1 - d2).max() < 1e-14


def test_the_mirror_orbit_gives_the_same_sky_positions():
    """(node, moon_w, phase) + pi: X and Y identical for both bodies, Z
    reversed, and the mutual inclination is mutual_inclination's mirror."""
    rng = np.random.default_rng(5)
    for moon_ecc in (False, True):
        p = random_system(rng, moon_ecc=moon_ecc)
        t = transit_times(p)
        P, M, d = elome_positions(t, p)
        p2 = dict(p, node=p["node"] + PI, moon_w=p["moon_w"] + PI, phase=p["phase"] + PI)
        P2, M2, d2 = elome_positions(t, p2)
        assert np.abs(P[:2] - P2[:2]).max() < 1e-13 and np.abs(M[:2] - M2[:2]).max() < 1e-13
        assert np.abs(d[2] + d2[2]).max() < 1e-13

        def normal(pp):
            _, _, a = elome_positions(np.array([0.0]), pp)
            _, _, b = elome_positions(np.array([0.01 * pp["moon_per"]]), pp)
            n = np.cross(a[:, 0], b[:, 0])
            return n / np.linalg.norm(n)
        n_planet = np.array([0.0, math.sin(p["inc"]), math.cos(p["inc"])])
        iota = [math.degrees(math.acos(np.dot(normal(pp), n_planet))) for pp in (p, p2)]
        mi = moons.mutual_inclination(*(math.degrees(p[k]) for k in ("inc", "moon_inc", "node")))
        assert np.allclose(iota, mi, atol=1e-9)


def test_the_barycentre_is_the_centre_of_mass():
    rng = np.random.default_rng(6)
    p = random_system(rng)
    t = transit_times(p)
    P, M, _ = elome_positions(t, p)
    B, _, _ = elome_positions(t, dict(p, a_ps=0.0))
    assert np.abs((P + p["q"] * M) / (1 + p["q"]) - B).max() < 1e-13
    P0, _, _ = elome_positions(t, dict(p, q=0.0))
    assert np.abs(P0 - B).max() < 1e-13


# --------------------------------------------------------------------------
# float32
# --------------------------------------------------------------------------

def test_float32_moon_phase_holds_thousands_of_orbits_from_t0():
    """References off the true periods (as a sampler's are): the moon's
    position at transits 3000 orbits from t0 stays within float32's
    resolution of a_ps (1.2e-7 stellar radii), where a phase formed directly
    from the traced periods is off by 0.1 rad."""
    p = dict(t0=0.3, per=37.31, a=90.0, inc=math.radians(89.6), ecc=0.0, w=0.0, a_ps=0.08,
             q=0.05, moon_per=1.377, moon_inc=math.radians(80.0), node=0.4, moon_ecc=0.0,
             moon_w=0.0, phase=1.1)
    t = transit_times(p, epochs=(-3000, -1, 0, 1500, 3000))
    refs = dict(per_ref=37.31 - 2e-4, moon_per_ref=1.377 + 3e-5, t0_ref=0.3 + 1e-3)
    d64 = elome_positions(t, p, mx.float64, **refs)[2]
    d32 = elome_positions(t, p, mx.float32, **refs)[2]
    assert np.abs(d64 - d32).max() < 5e-7          # measured 1.2e-7; unsplit: 8.5e-3


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------

GRAD_KEYS = ("a_ps", "q", "moon_inc", "node", "moon_ecc", "moon_w", "phase", "moon_per")


@pytest.mark.parametrize("moon_ecc", [0.0, 0.2])
def test_gradients_match_finite_differences(moon_ecc):
    rng = np.random.default_rng(8)
    p = random_system(rng, ecc=True)
    p["moon_ecc"] = moon_ecc
    t = transit_times(p, epochs=(0, 3))
    wts = np.random.default_rng(9).normal(size=(3, t.size))
    ep = epoch_center_times(t, p["t0"], p["per"])

    def scalar(vals):
        q = dict(p, **{k: v for k, v in zip(GRAD_KEYS, vals)})
        f = lambda v: v if isinstance(v, mx.array) else mx.array(v, dtype=mx.float64)
        dtt, k = mx.array(ep[0], dtype=mx.float64), mx.array(ep[1], dtype=mx.float64)
        tau = tau_from_epochs(dtt, k, f(0.0), f(0.0), f(q["per"]))
        phi = (2 * PI / q["per"]) * tau
        B = moons.barycentre_sky(phi, f(q["a"]), f(q["inc"]),
                                 moons.anchored_constants(q["ecc"], q["w"], mx.float64))
        turns = f(moons.moon_turns_ref(ep[1], p["per"], p["moon_per"]))
        phi_m = moons.moon_phase(tau, k, turns, p["per"], p["moon_per"], f(0.0),
                                 q["moon_per"] - p["moon_per"], q["phase"])
        # moon_ecc stays an array so the eccentric branch is taken at e = 0 too
        d = moons.moon_offset(phi_m, q["a_ps"], q["moon_inc"], q["node"],
                              moons.moon_orbit(q["moon_ecc"], q["moon_w"], mx.float64))
        Pl, Mo = moons.body_positions(*B, *d, q["q"])
        W = mx.array(wts, dtype=mx.float64)
        return sum(mx.sum(W[i] * (Pl[i] + 2.0 * Mo[i])) for i in range(3))

    with mx.stream(mx.cpu):
        x0 = [mx.array(p[k], dtype=mx.float64) for k in GRAD_KEYS]
        g = mx.grad(lambda *v: scalar(v), argnums=tuple(range(len(x0))))(*x0)
        for i, key in enumerate(GRAD_KEYS):
            h = 1e-6 * max(1.0, abs(p[key]))
            up = [mx.array(p[k] + (h if j == i else 0.0), dtype=mx.float64)
                  for j, k in enumerate(GRAD_KEYS)]
            if key == "moon_ecc" and moon_ecc == 0.0:          # one-sided at e = 0
                fd = (scalar(up).item() - scalar(x0).item()) / h
                tol = 1e-4
            else:
                dn = [mx.array(p[k] - (h if j == i else 0.0), dtype=mx.float64)
                      for j, k in enumerate(GRAD_KEYS)]
                fd = (scalar(up).item() - scalar(dn).item()) / (2 * h)
                tol = 1e-6
            gi = g[i].item()
            assert np.isfinite(gi), key
            assert abs(gi - fd) <= tol * max(1.0, abs(fd)), (key, gi, fd)


# --------------------------------------------------------------------------
# host helpers
# --------------------------------------------------------------------------

def test_moon_period_is_keplers_third_law():
    """Against P = 2 pi sqrt(a^3 / (G M_p (1 + q))) for the Earth and the
    Moon (two-body: 27.28 days against the sidereal month's 27.32)."""
    G, M, R, a, q = 6.67430e-11, 5.9722e24, 6.371e6, 3.844e8, 0.0123
    rho = M / (4.0 / 3.0 * PI * R ** 3)
    P = 2 * PI * math.sqrt(a ** 3 / (G * M * (1 + q))) / 86400.0
    assert abs(moons.moon_period(a / R, rho, q) / P - 1.0) < 1e-12
    assert abs(P - 27.32) < 0.06


@pytest.mark.parametrize("args", [(-0.1, 10.0, 0.0), (0.1, 10.0, -0.1),
                                  (0.1, 10.0, 0.0, 1.0), (0.3, 1.2, 0.0),
                                  (0.1, 5.0, 0.0, 0.8), (np.nan, 10.0, 0.0)])
def test_check_moon_rejects(args):
    with pytest.raises(ValueError):
        moons.check_moon(*args)
    moons.check_moon(0.1, 10.0, 0.01, 0.2)


# ==========================================================================
# photometry
# ==========================================================================

import functools  # noqa: E402
import warnings  # noqa: E402

import moon_oracle as MO  # noqa: E402
from elome.flux import flux_dev  # noqa: E402
from elome.hybrid import (HYBRID2, HYBRID4, HYBRID5, HybridLaw, flux_dev_hybrid,  # noqa: E402
                          hybrid_norms)

EPS = tuple(HYBRID5.eps) + tuple(HYBRID4.eps) + tuple(HYBRID2.eps)
ALL = HybridLaw("all_poles", EPS, ((1.0, -1.0, 0.0),))     # every pole in one call
NORM = np.array([PI, PI / 2, PI / 3] + [PI / (e * (1 + e)) for e in EPS])
TOL = {mx.float64: 1e-14, mx.float32: 2e-6}               # error / column norm

# planet (x, y, rp), moon (x, y, rs)
SCENES = {
    "apart, both on the disc": ((0.3, 0.1, 0.1), (0.5, -0.2, 0.03)),
    "overlapping on the disc": ((0.3, 0.1, 0.1), (0.38, 0.13, 0.03)),
    "moon inside the planet": ((0.3, 0.1, 0.1), (0.32, 0.09, 0.03)),
    "moon concentric with the planet": ((-0.4, 0.2, 0.1), (-0.4, 0.2, 0.03)),
    "overlap across the limb": ((0.95, 0.1, 0.1), (1.0, 0.16, 0.04)),
    "moon off the star, overlapping the planet on the limb": ((0.95, 0.1, 0.1), (1.03, 0.16, 0.04)),
    "lens on the limb, planet on the disc": ((0.85, 0.0, 0.1), (0.95, 0.02, 0.03)),
    "planet off the star, moon on the limb": ((1.08, 0.0, 0.1), (0.99, 0.05, 0.03)),
    "planet on the limb, moon on the disc across it": ((0.0, 0.97, 0.08), (0.05, 0.88, 0.04)),
    "tiny moon on the planet's edge": ((0.1, -0.4, 0.12), (0.1, -0.28, 0.005)),
    "large moon": ((-0.2, 0.3, 0.1), (-0.15, 0.36, 0.08)),
    "planet centred, moon across it": ((0.0, 0.0, 0.1), (0.1, 0.0, 0.03)),
    "moon centred, inside the planet": ((0.02, 0.0, 0.1), (0.0, 0.0, 0.03)),
    "both off the star": ((1.3, 0.2, 0.1), (1.25, 0.3, 0.03)),
    "planet larger than the star": ((0.5, 0.0, 1.3), (1.75, 0.2, 0.1)),
}
# exact tangencies: membership must not hinge on a coin toss at the touch
TANGENT = {
    "moon touching the planet outside": ((0.3, 0.1, 0.1), (0.3 + 0.13, 0.1, 0.03)),
    "moon touching the planet inside": ((0.3, 0.1, 0.1), (0.3 + 0.07, 0.1, 0.03)),
    "moon touching the limb inside, overlapping the planet": ((0.88, 0.0, 0.1), (0.97, 0.0, 0.03)),
    "moon touching the limb outside, overlapping the planet": ((0.94, 0.0, 0.1), (1.03, 0.0, 0.03)),
}


@functools.lru_cache(maxsize=None)
def oracle_cols(scene):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MO.generator_cols(list(scene), EPS)


def moon_cols(scene, dt=mx.float64, law=ALL, **kw):
    (xp, yp, rp), (xm, ym, rs) = scene
    with mx.stream(mx.cpu):
        a = lambda v: mx.array([v], dtype=dt)
        g = moons.generator_cols_moon(a(xp), a(yp), a(xm), a(ym), a(rp), a(rs), law,
                                      large_occultor=rp > 1, **kw)
        return np.array([c.item() for c in g])


@pytest.mark.parametrize("name", list(SCENES) + list(TANGENT))
@pytest.mark.parametrize("dt", [mx.float64, mx.float32])
def test_columns_match_the_oracle(name, dt):
    scene = SCENES.get(name) or TANGENT[name]
    err = np.abs(moon_cols(scene, dt) - oracle_cols(scene)) / NORM
    assert err.max() < TOL[dt] * (10 if scene[0][2] > 1 else 1), (name, err)


def single(z, r, dt=mx.float64):
    from elome.hybrid import _generator_terms, lens_geometry
    with mx.stream(mx.cpu):
        g = _generator_terms(lens_geometry(mx.array([z], dtype=dt), mx.array([r], dtype=dt)),
                             ALL)[0]
        return np.array([c.item() for c in g])


def test_without_an_overlap_the_discs_add():
    (xp, yp, rp), (xm, ym, rs) = SCENES["apart, both on the disc"]
    both = moon_cols(SCENES["apart, both on the disc"])
    tol = 1e-16 * NORM
    assert np.all(np.abs(both - single(math.hypot(xp, yp), rp) - single(math.hypot(xm, ym), rs))
                  <= tol)
    assert np.all(np.abs(moon_cols(((xp, yp, rp), (xm, ym, 0.0)))
                         - single(math.hypot(xp, yp), rp)) <= tol)


def test_a_moon_inside_the_planet_blocks_nothing_more():
    (xp, yp, rp), _ = SCENES["moon inside the planet"]
    for scene in ("moon inside the planet", "moon concentric with the planet"):
        got = moon_cols(SCENES[scene])
        assert np.abs(got - single(math.hypot(*SCENES[scene][0][:2]), SCENES[scene][0][2])).max() \
            < 1e-15 * NORM.max()


def test_a_body_behind_the_star_blocks_nothing():
    scene = SCENES["overlapping on the disc"]
    (xp, yp, rp), (xm, ym, rs) = scene
    with mx.stream(mx.cpu):
        t, f = mx.array([True]), mx.array([False])
        got = moon_cols(scene, front_p=t, front_m=f)
    assert np.abs(got - single(math.hypot(xp, yp), rp)).max() < 1e-16 * NORM.max()


def test_flux_and_shape_columns_follow_the_ld_basis_contract():
    w = [0.2, 0.15, 0.1, 0.05, 0.1]
    for scene in SCENES.values():
        (xp, yp, rp), (xm, ym, rs) = scene
        with mx.stream(mx.cpu):
            a = lambda v: mx.array([v], dtype=mx.float64)
            args = (a(xp), a(yp), a(xm), a(ym), a(rp), a(rs))
            F = moons.flux_dev_moon(*args, w, HYBRID5, large_occultor=rp > 1).item()
            B = np.array(moons.shape_cols_moon(*args, HYBRID5, large_occultor=rp > 1))[0]
        c = np.concatenate([[1.0], -np.array(w)])
        assert abs(F - (B @ c) / (hybrid_norms(HYBRID5) @ c)) < 1e-15 * max(1.0, abs(F))
        # and the moon at infinity is the spherical planet
        with mx.stream(mx.cpu):
            far = moons.flux_dev_moon(a(xp), a(yp), a(9.0), a(9.0), a(rp), a(rs), w, HYBRID5,
                                      large_occultor=rp > 1).item()
            sph = flux_dev_hybrid(a(math.hypot(xp, yp)), a(rp), w, HYBRID5,
                                  large_occultor=rp > 1).item()
        assert abs(far - sph) < 1e-16


QUAD = (0.4, 0.25)


@pytest.mark.parametrize("name", list(SCENES))
def test_quadratic_discs_are_exact_and_the_lens_close(name):
    """Each disc exact; the lens's mu^1 term by the closure: within 2e-3 ppm
    for a lens on the disc, and 1 ppm for one across the limb."""
    scene = SCENES[name]
    (xp, yp, rp), (xm, ym, rs) = scene
    if rp > 1:
        pytest.skip("the quadratic spherical solution is for r <= 1 here")
    with mx.stream(mx.cpu):
        a = lambda v: mx.array([v], dtype=mx.float64)
        got = moons.flux_dev_moon_quad(a(xp), a(yp), a(xm), a(ym), a(rp), a(rs), *QUAD).item()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = MO.quad_flux_dev(list(scene), *QUAD)
        lens = MO.occulted(list(scene), lambda mu: 1.0, "lens")
    on_limb = lens > 0 and any(abs(math.hypot(x, y) - 1) < r for x, y, r in scene)
    tol = 1e-6 if on_limb else (2e-9 if lens > 0 else 1e-15)
    assert abs(got - ref) < tol, (name, got - ref)


def test_quadratic_closure_on_random_limb_lenses():
    """Worst case over lenses straddling the limb, a 0.03 moon on a 0.1
    planet: below 1 ppm (the small-moon approximation of the whole moon
    costs ~10 ppm there)."""
    rng = np.random.default_rng(12)
    worst, n = 0.0, 0
    while n < 60:
        p, s = 0.1, 0.03
        ang, dP = rng.uniform(0, 2 * PI), rng.uniform(1 - p, 1 + p)
        P = (dP * math.cos(ang), dP * math.sin(ang), p)
        a2, D = rng.uniform(0, 2 * PI), rng.uniform(p - s, p + s)
        M = (P[0] + D * math.cos(a2), P[1] + D * math.sin(a2), s)
        if abs(math.hypot(M[0], M[1]) - 1) > s:
            continue
        n += 1
        with mx.stream(mx.cpu):
            a = lambda v: mx.array([v], dtype=mx.float64)
            got = moons.flux_dev_moon_quad(a(P[0]), a(P[1]), a(M[0]), a(M[1]), a(p), a(s),
                                           *QUAD).item()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            worst = max(worst, abs(got - MO.quad_flux_dev([P, M], *QUAD)))
    assert worst < 1e-6


@pytest.mark.parametrize("law", ["quadratic", "hybrid5"])
def test_float32_lens_across_the_limb(law):
    """Random lenses straddling the limb, float32 against float64: Green's
    forms about a point near the lens keep each moment's rounding at the
    lens's scale, which the quadratic closure would otherwise magnify there
    (measured 5.6e-7 and 1.2e-8 over 20,000; before, 6.9e-6 and 1.3e-7)."""
    rng = np.random.default_rng(3)
    n, rp, rs = 4000, 0.1, 0.03
    ang, zp = rng.uniform(0, 2 * PI, n), rng.uniform(1 - rp, 1 + rp, n)
    xp, yp = zp * np.cos(ang), zp * np.sin(ang)
    a2, D = rng.uniform(0, 2 * PI, n), rng.uniform(0, rp + rs, n)
    xm, ym = xp + D * np.cos(a2), yp + D * np.sin(a2)

    def flux(dt):
        f = lambda v: mx.array(v, dtype=dt)
        with mx.stream(mx.cpu):
            if law == "quadratic":
                out = moons.flux_dev_moon_quad(f(xp), f(yp), f(xm), f(ym), f(rp), f(rs),
                                               0.4, 0.25)
            else:
                out = moons.flux_dev_moon(f(xp), f(yp), f(xm), f(ym), f(rp), f(rs),
                                          mx.array([0.2, 0.15, 0.1, 0.05, 0.1], dtype=dt),
                                          law)
            return np.array(out, dtype=np.float64)
    err = np.abs(flux(mx.float32) - flux(mx.float64)).max()
    assert err < (1e-6 if law == "quadratic" else 3e-8), err


@pytest.mark.parametrize("name", ["overlapping on the disc", "overlap across the limb",
                                  "moon off the star, overlapping the planet on the limb",
                                  "planet on the limb, moon on the disc across it",
                                  "lens on the limb, planet on the disc"])
@pytest.mark.parametrize("kind", ["hybrid5", "quadratic"])
def test_photometry_gradients_match_finite_differences(name, kind):
    scene = SCENES[name]
    x0 = [scene[0][0], scene[0][1], scene[1][0], scene[1][1], scene[0][2], scene[1][2]]
    w = [0.2, 0.15, 0.1, 0.05, 0.1]

    def F(*v):
        if kind == "quadratic":
            return moons.flux_dev_moon_quad(*v, *QUAD)
        return moons.flux_dev_moon(*v, w, HYBRID5)

    with mx.stream(mx.cpu):
        args = [mx.array([v], dtype=mx.float64) for v in x0]
        g = mx.grad(lambda *v: mx.sum(F(*v)), argnums=tuple(range(6)))(*args)
        for i in range(6):
            h = 1e-6         # the lens's moments are small differences of its arcs'
            up = [mx.array([v + (h if j == i else 0.0)], dtype=mx.float64) for j, v in enumerate(x0)]
            dn = [mx.array([v - (h if j == i else 0.0)], dtype=mx.float64) for j, v in enumerate(x0)]
            fd = (F(*up).item() - F(*dn).item()) / (2 * h)
            gi = g[i].item()
            # the difference's rounding floor: ~1e-9 on a flux of ~1e-2
            assert np.isfinite(gi) and abs(gi - fd) <= 1e-6 * max(1e-2, abs(fd)), (i, gi, fd)
