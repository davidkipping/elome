"""Exposure rules for a secondary eclipse: the contact rule against
supersampling.

TransitModel(transittype="secondary") over a 29.4-minute exposure, 2,001
times across the eclipse, for three planets: spherical on a circular
orbit, spherical on an eccentric one (e = 0.3, w = 63 deg), and oblate
(f = 0.2, theta = 30 deg) on a circular one. The error is the largest
|F - F_ref| as a fraction of the eclipse depth fp, against the float64
contact rule with n_gl = 40 (its agreement with n_gl = 80 is reported).
Float64 runs the MLX graph on the CPU. The float32 row runs the fused GPU
kernels through flux_dev_from_tau, which returns F - 1: TransitModel's
float32 F itself is rounded to float32 near 1, half an ulp being 6e-8,
or 6e-5 of this 1e-3 depth.

    python benchmarks/bench_eclipse_exposure.py

Writes benchmarks/eclipse_exposure.json, which the docs check
(benchmarks/verify_doc_claims.py --check) reads.
"""

import json
import os

import mlx.core as mx
import numpy as np

import elome

HERE = os.path.dirname(os.path.abspath(__file__))
P, A, INC, RP, FP = 3.4525, 8.84, 88.06, 0.1153, 1e-3
EXP = 29.4 / 60.0 / 24.0
W4 = [0.2, 0.2, 0.1, 0.1]
PLANETS = {
    "sphere": dict(),
    "sphere_ecc": dict(ecc=0.3, w=63.0),
    "oblate": dict(f=0.2, theta=30.0),
}
# (method, n, float32): supersampling with n samples or the contact rule
# with n_gl = n
METHODS = [("supersample", 15, False), ("supersample", 101, False),
           ("contact", 5, False), ("contact", 7, False), ("contact", 7, True)]


def params(planet):
    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, P, RP, A, INC
    p.limb_dark, p.u, p.fp = "hybrid4", W4, FP
    for k, v in PLANETS[planet].items():
        setattr(p, k, v)
    return p


def times(p):
    s = np.sqrt(p.ecc)
    k, h = s * np.cos(np.radians(p.w)), s * np.sin(np.radians(p.w))
    mid = float(elome.occultation_offset(k, h)) * P
    return mid + np.linspace(-0.15, 0.15, 2001)


def flux(p, t, method, n):
    """F - 1 from TransitModel in float64."""
    kw = (dict(supersample_factor=n) if method == "supersample"
          else dict(integration="contact", n_gl=n))
    m = elome.TransitModel(p, t, transittype="secondary", exp_time=EXP,
                           dtype=mx.float64, **kw)
    return np.asarray(m.light_curve(p), dtype=np.float64) - 1.0


def dev_sampler(p, t, n, dtype):
    """F - 1 from flux_dev_from_tau (the contact rule, n_gl = n): float32
    returns it as computed, without rounding 1 + (F - 1) to float32."""
    from elome.metal import flux_dev_from_tau
    s = np.sqrt(p.ecc)
    k, h = s * np.cos(np.radians(p.w)), s * np.sin(np.radians(p.w))
    b = p.a * np.cos(np.radians(p.inc)) * (1 - p.ecc ** 2) / (1 + p.ecc * np.sin(np.radians(p.w)))
    tau = t - P * np.round(t / P)
    one = lambda v: mx.array([v], dtype=dtype)  # noqa: E731
    kw = dict(secosw=one(k), sesinw=one(h)) if p.ecc else {}
    if p.f is not None:
        kw.update(f=one(p.f), theta=one(np.radians(p.theta)))
    d = flux_dev_from_tau(mx.array(tau[None], dtype=dtype), one(P), one(p.a),
                          one(b), one(p.rp), transittype="secondary",
                          fp=one(FP), exp_time=EXP, integration="contact",
                          n_gl=n, **kw)
    return np.asarray(d[0], dtype=np.float64)


def main():
    out = {"exp_time_min": EXP * 1440.0, "points": 2001, "planets": {}}
    for planet in PLANETS:
        p = params(planet)
        t = times(p)
        ref = flux(p, t, "contact", 40)
        agree = float(np.abs(flux(p, t, "contact", 80) - ref).max() / FP)
        with mx.stream(mx.cpu):
            same = np.abs(dev_sampler(p, t, 40, mx.float64) - ref).max() / FP
        assert same < 1e-10, (planet, same)    # one model, both routes
        rows = []
        for method, n, fp32 in METHODS:
            got = (dev_sampler(p, t, n, mx.float32) if fp32
                   else flux(p, t, method, n))
            err = float(np.abs(got - ref).max() / FP)
            rows.append({"method": method, "n": n, "float32": fp32,
                         "max_err_depth": err})
        out["planets"][planet] = {"reference_agreement": agree, "rows": rows}
        print(f"\n{planet}: reference n_gl 40 vs 80: {agree:.1e} of the depth")
        for r in rows:
            prec = "float32" if r["float32"] else "float64"
            print(f"  {r['method']:<12}{r['n']:>4}  {prec}  "
                  f"{r['max_err_depth']:.2e}")
    path = os.path.join(HERE, "eclipse_exposure.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
