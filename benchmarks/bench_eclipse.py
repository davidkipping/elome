"""flux_dev_from_tau for a secondary eclipse against the transit.

The fused fp32 kernels of both (metal_eclipse and metal_oblate for the
eclipse, metal_hybrid and metal_oblate for the transit), at the
production shape (512 chains x 5,000 phase-ordered points around the
event, circular orbit, Kepler long cadence):

  sphere_transit   spherical planet, hybrid4
  sphere_eclipse   spherical planet, transittype="secondary"
  oblate_transit   f = 0.2, theta = 0.5 rad, hybrid4
  oblate_eclipse   f = 0.2, theta = 0.5 rad, transittype="secondary"

Both events take the same exposure rule: one evaluation per point, or
supersampling with n_sub = 15 (an eclipse has no contact rule). Each
configuration runs in its own subprocess, interleaved over rounds;
reported is the median over rounds of each round's median. value+grad
evaluates value AND gradients in every array input.

    python benchmarks/bench_eclipse.py [--chains 512] [--points 5000]

Writes benchmarks/eclipse_kernels.json, which the docs check
(benchmarks/verify_doc_claims.py --check) reads. Run on a quiet machine:
GPU timings on a loaded one mean little.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

P, A, B, R = 3.4525, 8.84, 0.30, 0.1153
F, TH, FP = 0.2, 0.5, 1e-3
EXP = 29.4 / 60.0 / 24.0
N_SUB = 15
W4 = [0.2, 0.2, 0.1, 0.1]
ROUTES = ("sphere_transit", "sphere_eclipse", "oblate_transit", "oblate_eclipse")
RULES = ("none", "supersample")


def worker(route, grad, n, m, rule, reps):
    import numpy as np
    import mlx.core as mx
    from elome.metal import flux_dev_from_tau

    shape, event = route.split("_")
    rng = np.random.default_rng(11)
    centre = 0.5 * P if event == "eclipse" else 0.0
    tau = np.linspace(centre - 0.14, centre + 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    tau = mx.array(np.sort(tau, axis=1).astype(np.float32))

    def full(v):
        return mx.array(np.full(n, v, np.float32))

    kw = {}
    if rule == "supersample":
        kw = dict(exp_time=EXP, integration="supersample", n_sub=N_SUB)
    args = [tau] + [full(v) for v in (P, A, B, R)]
    if event == "eclipse":
        kw["transittype"] = "secondary"
        args.append(full(FP))
        extra = lambda x: dict(fp=x)  # noqa: E731
    else:
        kw["limb_dark"] = "hybrid4"
        args.append(mx.array(np.tile(W4, (n, 1)).astype(np.float32)))
        extra = lambda x: dict(u=x)  # noqa: E731
    if shape == "oblate":
        args += [full(F), full(TH)]

    def model(t, p, a, b, r, x, *ft):
        geo = dict(f=ft[0], theta=ft[1]) if ft else {}
        return flux_dev_from_tau(t, p, a, b, r, **extra(x), **geo, **kw)

    ct = mx.array(rng.normal(size=(n, m)).astype(np.float32))
    vg = mx.value_and_grad(lambda *v: mx.sum(ct * model(*v)),
                           argnums=tuple(range(len(args))))

    def once():
        if grad:
            val, gs = vg(*args)
            mx.eval(val, *gs)
        else:
            mx.eval(model(*args))

    for _ in range(3):
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
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--_worker", nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        route, rule, g = a._worker
        dt = worker(route, g == "1", a.chains, a.points, rule, a.reps)
        print(json.dumps({"s": dt}))
        return

    here = os.path.abspath(__file__)
    keys = [(r, rule, g) for rule in RULES for g in (False, True) for r in ROUTES]
    res = {k: [] for k in keys}
    for _ in range(a.rounds):
        for r, rule, g in keys:
            out = subprocess.run(
                [sys.executable, here, "--chains", str(a.chains),
                 "--points", str(a.points), "--reps", str(a.reps),
                 "--_worker", r, rule, "1" if g else "0"],
                check=True, capture_output=True, text=True).stdout
            res[(r, rule, g)].append(json.loads(out.splitlines()[-1])["s"])

    med = {k: statistics.median(v) * 1e3 for k, v in res.items()}
    print(f"\n{a.chains} chains x {a.points} points, n_sub={N_SUB} "
          f"(median of {a.rounds} isolated rounds)\n")
    print("| route | rule | forward | vs transit | value+grad | vs transit |")
    print("|---|---|---:|---:|---:|---:|")
    for rule in RULES:
        for shape in ("sphere", "oblate"):
            e = [med[(f"{shape}_eclipse", rule, g)] for g in (False, True)]
            t = [med[(f"{shape}_transit", rule, g)] for g in (False, True)]
            print(f"| {shape} eclipse | {rule} | {e[0]:.2f} ms | {e[0] / t[0]:.2f}x | "
                  f"{e[1]:.2f} ms | {e[1] / t[1]:.2f}x |")

    path = os.path.join(os.path.dirname(here), "eclipse_kernels.json")
    with open(path, "w") as f:
        json.dump({"chains": a.chains, "points": a.points, "n_sub": N_SUB,
                   "rules": {rule: {r: {"forward_ms": med[(r, rule, False)],
                                        "value_grad_ms": med[(r, rule, True)]}
                                    for r in ROUTES} for rule in RULES}},
                  f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
