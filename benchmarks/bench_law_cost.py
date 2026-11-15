"""The cost of the limb-darkening law, within each code.

One light curve of the spherical benchmark's transit (scenario.py) under
each law a code runs as released, with no change to the code:

  Elómë       float64 (CPU) and float32 (GPU): quadratic, hybrid2,
              hybrid4, hybrid5
  batman      its built-in quadratic, power-2 and nonlinear (four-
              parameter) laws; the last two it integrates numerically,
              to its default max_err of 1 ppm

A law's cost is read against the same code's quadratic law, which
cancels the implementation: what is left is the law. (PyTransit's
RoadRunner tabulates any law before computing the light curve, so its
cost does not depend on the law; it is in the cross-code benchmark
only.) batman runs on one thread, Elómë's float64 on the cores MLX
chooses, as in the cross-code benchmark. Each (code, law, N) runs in its own
process, interleaved over rounds; reported is the median over rounds of
each round's median.

Accuracy: each (code, law) on 241 points against a 30-digit direct
integral of that law's intensity (oracle.flux_direct_law).

    python benchmarks/bench_law_cost.py [--rounds 3]

Writes benchmarks/law_cost.json, which the docs check
(benchmarks/verify_doc_claims.py --check) reads. Run on a quiet machine.
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from scenario import A_RS, INC_DEG, PER, RP, T0, time_grid, z_of_t  # noqa: E402

# physical profiles (positive, darkening to the limb) for each law
LAWS = {
    "quadratic": [0.40, 0.25],
    "power2": [0.65, 0.60],                    # 1 - c (1 - mu^alpha)
    "nonlinear": [0.55, -0.10, 0.60, -0.30],   # 1 - sum c_n (1 - mu^(n/2))
    "hybrid2": [0.5, 0.2],
    "hybrid4": [0.2, 0.2, 0.1, 0.1],
    "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1],
}
HYBRID = ("hybrid2", "hybrid4", "hybrid5")
RUNS = {
    "elome_fp64": ["quadratic", *HYBRID],
    "elome_fp32": ["quadratic", *HYBRID],
    "batman": ["quadratic", "power2", "nonlinear"],
}
N_SWEEP = [100_000, 1_000_000, 10_000_000]
N_PREC = 241


def _expanded(law):
    """A hybrid law's (c (3,), d (K,), eps (K,)):
    I = c0 + c1 mu^2 + c2 mu^4 + sum_k d_k (mu^2 + eps_k)^-2."""
    from elome.hybrid import get_law, w_to_c
    c, d = w_to_c(law, LAWS[law])
    return [float(x) for x in c], [float(x) for x in d], get_law(law).eps


def intensity(law):
    """I(mu, mp) for the oracle, in mpmath."""
    u = LAWS[law]
    if law == "quadratic":
        return lambda mu, mp: 1 - u[0] * (1 - mu) - u[1] * (1 - mu) ** 2
    if law == "power2":
        return lambda mu, mp: 1 - u[0] * (1 - mu ** mp.mpf(u[1]))
    if law == "nonlinear":
        return lambda mu, mp: 1 - sum(
            c * (1 - mu ** (mp.mpf(n + 1) / 2)) for n, c in enumerate(u))
    c, d, eps = _expanded(law)
    return lambda mu, mp: (c[0] + c[1] * mu ** 2 + c[2] * mu ** 4 + sum(
        dk * (mu ** 2 + mp.mpf(e)) ** -2 for dk, e in zip(d, eps)))


def prepare(code, law, t):
    """A zero-argument call that returns the light curve at times t."""
    if code.startswith("elome"):
        import mlx.core as mx
        import elome
        p = elome.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
        p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, law, list(LAWS[law])
        dt = mx.float32 if code == "elome_fp32" else mx.float64
        m = elome.TransitModel(p, t, dtype=dt)
        run = lambda: np.asarray(m.light_curve(p))  # noqa: E731
    elif code == "batman":
        import batman
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
        p.ecc, p.w, p.limb_dark, p.u = 0.0, 90.0, law, list(LAWS[law])
        m = batman.TransitModel(p, t, nthreads=1)
        run = lambda: m.light_curve(p)  # noqa: E731
    else:
        raise ValueError(code)
    run()    # compile and warm
    return run


def worker(code, law, n):
    run = prepare(code, law, time_grid(n))
    ts, total = [], 0.0
    while len(ts) < 15 and (len(ts) < 3 or total < 1.5):
        t0 = time.perf_counter()
        run()
        dt = time.perf_counter() - t0
        ts.append(dt)
        total += dt
        if dt > 30.0:
            break
    return statistics.median(ts)


def accuracy():
    from oracle import oracle_lightcurve_law
    t = time_grid(N_PREC)
    out = {}
    for law in LAWS:
        tag = f"law-{law}-{N_PREC}-" + ",".join(map(str, LAWS[law]))
        ref = oracle_lightcurve_law(t, RP, intensity(law), z_of_t, tag)
        for code, laws in RUNS.items():
            if law in laws:
                f = np.asarray(prepare(code, law, t)(), dtype=np.float64)
                out.setdefault(code, {})[law] = float(np.abs(f - ref).max())
                print(f"  accuracy {code:12s} {law:10s} {out[code][law]:.1e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--_worker", nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._worker:
        code, law, n = a._worker
        print(json.dumps({"s": worker(code, law, int(n))}))
        return

    print("accuracy against a 30-digit integral of each law:")
    acc = accuracy()
    keys = [(c, law, n) for n in N_SWEEP for c, laws in RUNS.items() for law in laws]
    res = {k: [] for k in keys}
    for _ in range(a.rounds):
        for code, law, n in keys:
            # threads as bench_speed.run_one: one for batman, all for Elómë
            t = "12" if code.startswith("elome") else "1"
            env = dict(os.environ, NUMBA_NUM_THREADS=t, OMP_NUM_THREADS=t,
                       VECLIB_MAXIMUM_THREADS=t)
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--_worker",
                 code, law, str(n)],
                check=True, capture_output=True, text=True, env=env).stdout
            res[(code, law, n)].append(json.loads(out.splitlines()[-1])["s"])
    speed = {}
    for (code, law, n), v in res.items():
        speed.setdefault(code, {}).setdefault(law, {})[str(n)] = statistics.median(v)

    print(f"\none light curve (median of {a.rounds} isolated rounds), and "
          "relative to the code's own quadratic law:")
    for n in N_SWEEP:
        print(f"\nN = {n:,}")
        for code, laws in RUNS.items():
            q = speed[code]["quadratic"][str(n)]
            cells = [f"{law} {speed[code][law][str(n)] / q:.2f}x"
                     for law in laws[1:]]
            print(f"  {code:12s} quadratic {q * 1e3:9.2f} ms   " + "   ".join(cells))

    path = os.path.join(HERE, "law_cost.json")
    with open(path, "w") as f:
        json.dump({"n_sweep": N_SWEEP, "rounds": a.rounds, "n_prec": N_PREC,
                   "laws": LAWS, "speed": speed, "accuracy": acc}, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
