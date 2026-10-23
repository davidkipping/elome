"""Spot-crossing cross-code benchmark: precision and speed.

    benchmarks/spot_compare/.venv/bin/python run_spot_compare.py --precision
    benchmarks/spot_compare/.venv/bin/python run_spot_compare.py --speed

Precision: each code, at each resolution, on the scenario's 241-point
grid for every case, against the brute-force spot integral of
tests/spot_oracle.py (shares no code with any code under test). Speed:
one light curve of N_SPEED points, value and value + gradient, each
(code, resolution, mode) in its own subprocess (worker_spot.py). Both
merge into results_spot.json; make_report_spot.py renders RESULTS.md.
Run --speed on a quiet machine.
"""
import json
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))   # spot_oracle

import numpy as np  # noqa: E402

from scenario_spot import (CASES, JAX_YDEG, N_PREC, OOT, R_P, SPOTTER_NSIDE,  # noqa: E402
                           W_LD, grid, spots_rad)

RESULTS = os.path.join(HERE, "results_spot.json")
RUNS = [("elome64", None), ("elome32", None)] + \
       [("jaxoplanet", d) for d in JAX_YDEG] + [("spotter", n) for n in SPOTTER_NSIDE]


def oracle(case):
    """Relative flux by the brute-force integral, I = (1 - w) mu^0 + w mu^2."""
    import spot_oracle as SO
    spots = [(SO.spot_vector(lon, lat), al, c) for lon, lat, al, c in spots_rad(case)]
    coef = np.array([1.0 - W_LD, W_LD, 0.0])

    def flux(x, y):
        return np.pi * (1.0 - W_LD) + W_LD * np.pi / 2 + SO.columns((x, y, R_P), spots, ()) @ coef
    oot = flux(*OOT)
    return np.array([flux(x, y) for x, y in zip(*grid(N_PREC))]) / oot


def bare_bump(case):
    """The spots' own signal in the crossing: the oracle with and without
    them, for the precision's scale."""
    import spot_oracle as SO
    coef = np.array([1.0 - W_LD, W_LD, 0.0])
    tot = np.pi * (1.0 - W_LD) + W_LD * np.pi / 2

    def bare(x, y):
        return (tot + SO.columns((x, y, R_P), [], ()) @ coef) / tot
    return np.array([bare(x, y) for x, y in zip(*grid(N_PREC))])


def run_worker(code, res, mode, case, out=None):
    args = [sys.executable, os.path.join(HERE, "worker_spot.py"), code,
            "-" if res is None else str(res), mode, case] + ([out] if out else [])
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        return None
    return r.stdout


def load():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save(res):
    res["machine"] = {"platform": platform.platform(), "processor": platform.processor(),
                      "python": platform.python_version()}
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=1)


def precision(res):
    out = res.setdefault("precision", {})
    for case in CASES:
        ref = oracle(case)
        bump = float(np.max(np.abs(ref - bare_bump(case))))
        out.setdefault(case, {})["spot_signal"] = bump
        print(f"{case}: the spots' signal {bump * 1e6:.0f} ppm")
        for code, r in RUNS:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "f.npy")
                if run_worker(code, r, "curve", case, path) is None:
                    continue
                f = np.load(path)
            err = np.abs(f - ref)
            key = code if r is None else f"{code}:{r}"
            out[case][key] = {"max": float(err.max()), "median": float(np.median(err))}
            print(f"  {key:16s} max {err.max() * 1e6:10.4f} ppm   median {np.median(err) * 1e6:10.4f} ppm")
            save(res)


def speed(res):
    out = res.setdefault("speed", {})
    case = "chord"
    for code, r in RUNS:
        key = code if r is None else f"{code}:{r}"
        for mode in ("time", "time_grad"):
            s = run_worker(code, r, mode, case)
            if s is None:
                continue
            m = json.loads(s.strip().splitlines()[-1])
            out.setdefault(key, {})[mode] = m
            print(f"  {key:16s} {mode:9s} first {m['first_s']:8.3f} s   warm {m['warm_s'] * 1e3:9.2f} ms")
            save(res)


def main():
    res = load()
    if "--precision" in sys.argv:
        precision(res)
    if "--speed" in sys.argv:
        speed(res)
    save(res)


if __name__ == "__main__":
    main()
