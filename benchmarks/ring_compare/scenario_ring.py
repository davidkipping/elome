"""The ringed-planet cross-code scenario: the spherical benchmark's
transit (benchmarks/scenario.py) with a Saturn-like ring.

Limb darkening is the one profile every code here can express exactly:
I(mu) = 1 - w (1 - mu^2), w = 0.5 -- Elómë's hybrid2 law with its pole
weight 0 (ringed planets take hybrid laws only), and the quadratic law
(u1, u2) = (2w, -w) elsewhere. Circular orbit, instantaneous fluxes (no
exposure integration), so every code computes the same function.

BENCH_SCENARIO=translucent: the same ring with normal optical depth 0.6;
=edge: a nearly edge-on ring (ring_f = 0.95), whose outline can cross the
stellar limb four times.
"""

import math
import os

import numpy as np

SCEN = os.environ.get("BENCH_SCENARIO", "opaque")

T0 = 0.0
PER = 3.456                       # d
A_RS = 8.8
B_IMPACT = 0.45
INC = math.acos(B_IMPACT / A_RS)  # rad
INC_DEG = math.degrees(INC)
RP = 0.1                          # planet radius (spherical planet)
RING_IN, RING_OUT = 1.4, 2.6      # in units of RP
RING_F = 0.95 if SCEN == "edge" else 0.7   # 1 - |cos i_ring|
RING_TAU = 0.6 if SCEN == "translucent" else math.inf
THETA_DEG = 25.0                  # sky angle of the ring's long axis from the motion
THETA = math.radians(THETA_DEG)
W_LD = 0.5
U1, U2 = 2.0 * W_LD, -W_LD        # quadratic equivalent


def blocked_fraction():
    """The ring's blocked fraction T = 1 - exp(-tau / |cos i_ring|)."""
    if math.isinf(RING_TAU):
        return 1.0
    return -math.expm1(-RING_TAU / (1.0 - RING_F))


T_HALF = 0.12                     # as the spherical benchmark
N_PREC = 241
N_SWEEP = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
BATCH_NPV = int(os.environ.get("BENCH_NPV", "512"))
BATCH_NPT = 10_000
GRAD_N = [10_000, 100_000]


def time_grid(n):
    return np.linspace(-T_HALF, T_HALF, int(n))


def sky(t):
    """(X, Y) in stellar radii, X along the motion, Y = -b cos(phi); all
    front-side here."""
    phi = 2.0 * np.pi * (np.asarray(t, np.float64) - T0) / PER
    return A_RS * np.sin(phi), -B_IMPACT * np.cos(phi)


def principal(t):
    """The planet's centre in the ring's principal frame (long axis x)."""
    X, Y = sky(t)
    c, s = math.cos(THETA), math.sin(THETA)
    return X * c + Y * s, -X * s + Y * c


def batch_params(npv=BATCH_NPV, seed=0):
    """npv parameter draws around the scenario."""
    rng = np.random.default_rng(seed)
    return {
        "rp": RP * (1 + 0.1 * rng.standard_normal(npv)),
        "ring_in": RING_IN + 0.05 * rng.standard_normal(npv),
        "ring_out": RING_OUT + 0.1 * rng.standard_normal(npv),
        "ring_f": np.clip(RING_F + 0.03 * rng.standard_normal(npv), 0.0, 0.98),
        "theta": THETA + 0.2 * rng.standard_normal(npv),
        "t0": 0.002 * rng.standard_normal(npv),
        "per": PER + 1e-4 * rng.standard_normal(npv),
        "a": A_RS * (1 + 0.02 * rng.standard_normal(npv)),
        "b": np.clip(B_IMPACT + 0.05 * rng.standard_normal(npv), 0.0, 0.8),
    }
