"""Ringed-planet cross-code benchmark: precision, speed and memory.

    benchmarks/ring_compare/.venv/bin/python run_ring_compare.py [--quick]

Precision: every code on the scenario's 241-point grid against the
float64 ray integral of tests/ring_oracle.py (adaptive to 1e-15, shares no
code with any code under test; cached per scenario in oracle_<scen>.npy).
Speed and memory: each (code, mode, N) in its own subprocess
(worker_ring.py). Writes results_ring.json; make_report_ring.py renders
RESULTS.md. Run on a quiet machine.

    --precision   precision only, for BENCH_SCENARIO (opaque, translucent,
                  edge), into results_ring.json as precision_<scen>
    --codes a,b   time only these codes, replacing their entries

Every run merges into results_ring.json (other entries are kept).
"""
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))   # ring_oracle

import numpy as np  # noqa: E402

import adapters_ring as A  # noqa: E402
from scenario_ring import (BATCH_NPT, BATCH_NPV, GRAD_N, N_PREC, N_SWEEP,  # noqa: E402
                           RING_F, RING_IN, RING_OUT, RP, SCEN, W_LD,
                           blocked_fraction, principal, time_grid)


def oracle_flux():
    """I(mu) = 1 - w (1 - mu^2) = (1 - w) mu^0 + w mu^2, integrated by rays
    about the star's centre; cached (it takes minutes)."""
    path = os.path.join(HERE, f"oracle_{SCEN}.npy")
    if os.path.exists(path):
        return np.load(path)
    import ring_oracle as OR
    w = W_LD
    total = np.pi * (1 - w) + w * np.pi / 2
    planet = (RP, RP)
    edges = [(k * RP, k * RP * (1.0 - RING_F)) for k in (RING_IN, RING_OUT)]
    x0, y0 = principal(time_grid(N_PREC))
    dev = []
    for x, y in zip(x0, y0):
        c = OR.ring_cols(x, y, planet, edges, [blocked_fraction()], ())
        dev.append(((1 - w) * c[0] + w * c[1]) / total)
    ref = 1.0 + np.array(dev)
    np.save(path, ref)
    return ref


def _python(code):
    """The interpreter a code runs under: its own venv where its
    dependencies conflict with the others' (adapters_ring.PYTHON)."""
    return A.PYTHON.get(code, sys.executable)


def curve(code, n):
    if code not in A.PYTHON:
        return getattr(A, f"{code}_run")(getattr(A, f"{code}_prepare")(n))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.npy")
        subprocess.run([_python(code), os.path.join(HERE, "worker_ring.py"),
                        code, "curve", str(n), path], check=True)
        return np.load(path)


def precision():
    ref = oracle_flux()
    out = {}
    for code in A.CODES:
        if code in A.OPAQUE_ONLY and SCEN == "translucent":
            continue
        f = curve(code, N_PREC)
        err = np.abs(np.asarray(f, np.float64) - ref)
        out[code] = {"max": float(err.max()), "median": float(np.median(err))}
        print(f"  precision {code:18s} max {err.max():.2e} median {np.median(err):.2e}")
    return out


def measure(code, mode, n):
    r = subprocess.run([_python(code), os.path.join(HERE, "worker_ring.py"),
                        code, mode, str(n)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def _mem(m):
    """Peak memory of one call above the process's level after importing
    the code, MB (memwatch.py: the physical footprint)."""
    return {k: m[k] for k in ("peak_mb", "mlx_peak_mb") if k in m}


def main():
    if "--precision" in sys.argv:
        # precision only, for BENCH_SCENARIO
        res = {f"precision_{SCEN}": precision()}
        path = os.path.join(HERE, "results_ring.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update(res)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
        return
    quick = "--quick" in sys.argv
    path = os.path.join(HERE, "results_ring.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    only = None
    if "--codes" in sys.argv:
        only = set(sys.argv[sys.argv.index("--codes") + 1].split(","))
    pick = (lambda codes: [c for c in codes if only is None or c in only])
    if only is None:
        res.update({"machine": platform.platform(),
                    "python": platform.python_version(),
                    "started": time.strftime("%Y-%m-%d %H:%M"),
                    "load": os.getloadavg()})
        print("precision vs the ray-integral oracle:")
        res[f"precision_{SCEN}"] = precision()
    sweep = N_SWEEP[:3] if quick else N_SWEEP
    res.setdefault("mem", {})
    for k in ("single", "batch", "grad"):
        res["mem"].setdefault(k, {})
        res.setdefault(k, {})
    for code in pick(A.CODES):
        res["single"][code] = {}
        res["mem"]["single"][code] = {}
        for n in sweep:
            m = measure(code, "single", n)
            if m is None:
                break
            res["single"][code][str(n)] = m["median_s"]
            res["mem"]["single"].setdefault(code, {})[str(n)] = _mem(m)
            print(f"  single {code:18s} N={n:>10,d}: {m['median_s'] * 1e3:10.2f} ms")
            if m["median_s"] > 8.0:          # the next decade would take minutes
                break
    for code in pick(A.BATCH_CODES):
        m = measure(code, "batch", 0)
        if m is not None:
            k = A.BATCH_SUBSET.get(code)
            res["batch"][code] = m["median_s"] * (BATCH_NPV / k if k else 1)
            if k:
                res.setdefault("batch_scaled_from", {})[code] = k
            res["mem"]["batch"][code] = _mem(m)
            print(f"  batch {code:18s} {BATCH_NPV} x {BATCH_NPT}: {m['median_s']:.3f} s")
    for code in pick(A.GRAD_CODES):
        res["grad"][code] = {}
        res["mem"]["grad"][code] = {}
        for n in GRAD_N:
            m = measure(code, "grad", n)
            if m is not None:
                res["grad"][code][str(n)] = m["median_s"]
                res["mem"]["grad"].setdefault(code, {})[str(n)] = _mem(m)
                print(f"  value+grad {code:18s} N={n:>8,d}: {m['median_s'] * 1e3:.2f} ms")
    if only is None:
        res["load_after"] = os.getloadavg()
    with open(path, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
