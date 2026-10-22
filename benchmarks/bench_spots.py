"""Spotted stars against a bare transit: the fused fp32 kernels on the GPU,
and the float64 graph on the CPU.

GPU, at the production shape (512 parameter sets x 5,000 phase-ordered
points around a transit, Kepler long cadence), hybrid4:

  spherical    a bare transit (flux_dev_from_tau, the spherical kernel)
  spots<K>     the same transit on a star with K spots (flux_dev_spots_t,
               the batched route behind TransitModel.light_curves),
               K = 1, 2, 4: spots of 6-12 degrees, two of them on the
               transit chord, a star turning in 12 days

CPU, float64, one light curve of 1,000 points with two spots
(TransitModel.light_curve and, for value+grad, light_curve_mx in every
field of the spot table and the star).

value+grad evaluates value AND gradients in every input (the times, the
orbit, the planet, the star, the spot table and the weights). Each
configuration runs in its own subprocess, interleaved over rounds;
reported is the median over rounds of each round's median.

    python benchmarks/bench_spots.py [--rounds 3]

Writes benchmarks/spot_kernels.json, which the docs check
(benchmarks/verify_doc_claims.py --check) reads. Run on a quiet machine:
GPU timings on a loaded one mean little.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

P, A, B, R = 3.4525, 8.84, 0.30, 0.1153
EXP = 29.4 / 60.0 / 24.0
W4 = [0.2, 0.2, 0.1, 0.1]
LAT = -math.degrees(math.asin(B))                       # on the transit chord
SPOTS = [[10.0, LAT, 8.0, 0.5], [50.0, 15.0, 10.0, 0.6],
         [-40.0, LAT + 4.0, 6.0, 0.7], [80.0, -5.0, 12.0, 0.5]]
PROT, ISTAR, LAM = 12.0, 90.0, 0.0
GPU_ROUTES = ("spherical", "spots1", "spots2", "spots4")
RULES = ("none", "contact")


def gpu_worker(route, rule, grad, n, m, reps):
    import numpy as np
    import mlx.core as mx
    from elome.metal import flux_dev_from_tau
    from elome.spots_tau import flux_dev_spots_t

    rng = np.random.default_rng(11)
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    t = mx.array(np.sort(tau, axis=1).astype(np.float32))

    def full(v):
        return mx.array(np.full(n, v, np.float32))
    exp_time = EXP if rule == "contact" else 0.0
    w = mx.array(np.tile(W4, (n, 1)).astype(np.float32))
    if route == "spherical":
        args = [t] + [full(v) for v in (P, A, B, R)] + [w]

        def model(tt, p, a, b, r, u):
            return flux_dev_from_tau(tt, p, a, b, r, limb_dark="hybrid4", u=u,
                                     exp_time=exp_time, integration=rule, n_gl=5)
    else:
        K = int(route[len("spots"):])
        sp = np.radians(np.array(SPOTS[:K])) * [1, 1, 1, 0] + [0, 0, 0, 1] * np.array(SPOTS[:K])
        args = ([t] + [full(v) for v in (0.0, 0.0, P, A, B, R, PROT,
                                         math.radians(ISTAR), math.radians(LAM))]
                + [mx.array(np.tile(sp, (n, 1, 1)).astype(np.float32)), w])

        def model(tt, *v):
            return flux_dev_spots_t(tt, *v[:9], v[9], "hybrid4", v[10],
                                    exp_time=exp_time, n_gl=5)

    ct = mx.array(rng.normal(size=(n, m)).astype(np.float32))
    vg = mx.value_and_grad(lambda *v: mx.sum(ct * model(*v)),
                           argnums=tuple(range(len(args))))

    def once():
        if grad:
            val, gs = vg(*args)
            mx.eval(val, *gs)
        else:
            mx.eval(model(*args))
    return _timed(once, reps, mx)


def cpu_worker(rule, grad, reps):
    import numpy as np
    import mlx.core as mx
    import elome

    p = elome.TransitParams()
    p.t0, p.per, p.rp, p.a, p.ecc, p.w = 0.0, P, R, A, 0.0, 90.0
    p.inc = math.degrees(math.acos(B / A))
    p.limb_dark, p.u = "hybrid4", list(W4)
    p.spots, p.prot = [list(s) for s in SPOTS[:2]], PROT
    t = np.linspace(-0.14, 0.14, 1000)
    kw = dict(exp_time=EXP, integration="contact", n_gl=5) if rule == "contact" else {}
    m = elome.TransitModel(p, t, **kw)
    ct = mx.array(np.random.default_rng(3).normal(size=t.size), dtype=mx.float64)

    def loss(spots, prot, istar, lam):
        p.spots, p.prot, p.istar, p.lam = spots, prot, istar, lam
        return mx.sum(ct * m.light_curve_mx(p))
    vg = mx.value_and_grad(loss, argnums=(0, 1, 2, 3))
    vals = [mx.array(np.array(SPOTS[:2]), dtype=mx.float64)] + [
        mx.array(v, dtype=mx.float64) for v in (PROT, ISTAR, LAM)]

    def once():
        with mx.stream(mx.cpu):
            if grad:
                val, gs = vg(*vals)
                mx.eval(val, *gs)
            else:
                m.light_curve(p)
    return _timed(once, reps, mx)


def _timed(once, reps, mx):
    for _ in range(2):
        once()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--_worker", nargs=4, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        dev, route, rule, g = a._worker
        if dev == "gpu":
            dt = gpu_worker(route, rule, g == "1", a.chains, a.points, a.reps)
        else:
            dt = cpu_worker(rule, g == "1", max(1, a.reps // 2))
        print(json.dumps({"s": dt}))
        return

    here = os.path.abspath(__file__)
    jobs = ([("gpu", r, rule, g) for rule in RULES for r in GPU_ROUTES for g in (False, True)]
            + [("cpu", "spots2", rule, g) for rule in RULES for g in (False, True)])
    res = {j: [] for j in jobs}
    for _ in range(a.rounds):
        for j in jobs:
            dev, route, rule, g = j
            out = subprocess.run(
                [sys.executable, here, "--chains", str(a.chains), "--points",
                 str(a.points), "--reps", str(a.reps),
                 "--_worker", dev, route, rule, "1" if g else "0"],
                check=True, capture_output=True, text=True).stdout
            res[j].append(json.loads(out.splitlines()[-1])["s"])
    med = {j: statistics.median(v) for j, v in res.items()}

    print(f"\nGPU, float32: {a.chains} sets x {a.points} points, hybrid4, n_gl=5 "
          f"(median of {a.rounds} isolated rounds)\n")
    print("| rule | route | forward | vs spherical | value+grad | vs spherical |")
    print("|---|---|---:|---:|---:|---:|")
    for rule in RULES:
        f0, g0 = med[("gpu", "spherical", rule, False)], med[("gpu", "spherical", rule, True)]
        for r in GPU_ROUTES:
            f, g = med[("gpu", r, rule, False)], med[("gpu", r, rule, True)]
            print(f"| {rule} | {r} | {f * 1e3:.1f} ms | {f / f0:.0f}x | "
                  f"{g * 1e3:.1f} ms | {g / g0:.0f}x |")
    print("\nCPU, float64: one light curve of 1,000 points, two spots\n")
    for rule in RULES:
        print(f"| {rule} | {med[('cpu', 'spots2', rule, False)]:.2f} s | "
              f"{med[('cpu', 'spots2', rule, True)]:.2f} s |")

    path = os.path.join(os.path.dirname(here), "spot_kernels.json")
    out = {"chains": a.chains, "points": a.points, "rounds": a.rounds,
           "gpu": {rule: {r: {"forward_ms": med[("gpu", r, rule, False)] * 1e3,
                              "value_grad_ms": med[("gpu", r, rule, True)] * 1e3}
                          for r in GPU_ROUTES} for rule in RULES},
           "cpu_1000_points_two_spots": {
               rule: {"forward_s": med[("cpu", "spots2", rule, False)],
                      "value_grad_s": med[("cpu", "spots2", rule, True)]}
               for rule in RULES}}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
