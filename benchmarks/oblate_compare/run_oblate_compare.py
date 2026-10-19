"""Oblate-planet cross-code benchmark: precision and speed.

    benchmarks/oblate_compare/.venv/bin/python run_oblate_compare.py [--quick]

Precision: every code on the scenario's 241-point grid against a 30-digit
mpmath polar-ray integral (tests/oblate_oracle.py, independent of every
code under test). Speed: each (code, mode, N) in its own subprocess
(worker_oblate.py). Writes results_oblate.json; make_report_oblate.py
renders RESULTS.md. Run on a quiet machine.

    --precision   precision only, into results_oblate.json
    --graze       the same for BENCH_SCENARIO=graze (precision_graze)
"""
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))   # oblate_oracle

import numpy as np  # noqa: E402

import adapters_oblate as A  # noqa: E402
from scenario_oblate import (A_AX, B_AX, BATCH_NPT, BATCH_NPV, GRAD_N,  # noqa: E402
                             N_PREC, N_SWEEP, W_LD, principal, time_grid)


def oracle_flux():
    """I(mu) = 1 - w (1 - mu^2) = (1 - w) mu^0 + w mu^2, at 30 digits."""
    import oblate_oracle as OR
    w = W_LD
    total = np.pi * (1 - w) + w * np.pi / 2
    x0, y0 = principal(time_grid(N_PREC))
    dev = []
    for x, y in zip(x0, y0):
        c = OR.columns_mp(x, y, A_AX, B_AX, (), dps=30)   # negative deviations
        dev.append(((1 - w) * c[0] + w * c[1]) / total)
    return 1.0 + np.array(dev)


def _python(code):
    """The interpreter a code runs under: its own venv where its
    dependencies conflict with the others' (adapters_oblate.PYTHON)."""
    return A.PYTHON.get(code, sys.executable)


def curve(code, n):
    if code not in A.PYTHON:
        return getattr(A, f"{code}_run")(getattr(A, f"{code}_prepare")(n))
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.npy")
        subprocess.run([_python(code), os.path.join(HERE, "worker_oblate.py"),
                        code, "curve", str(n), path], check=True)
        return np.load(path)


def precision():
    ref = oracle_flux()
    out = {}
    for code in A.CODES:
        f = curve(code, N_PREC)
        err = np.abs(np.asarray(f, np.float64) - ref)
        out[code] = {"max": float(err.max()), "median": float(np.median(err))}
        print(f"  precision {code:18s} max {err.max():.2e} median {np.median(err):.2e}")
    return out


def measure(code, mode, n):
    r = subprocess.run([_python(code), os.path.join(HERE, "worker_oblate.py"),
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
    if "--graze" in sys.argv or "--precision" in sys.argv:
        # precision only; --graze is the stress scenario (scenario_oblate.GRAZE)
        key = "precision_graze" if "--graze" in sys.argv else "precision"
        res = {key: precision()}
        path = os.path.join(HERE, "results_oblate.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update(res)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
        return
    quick = "--quick" in sys.argv
    path = os.path.join(HERE, "results_oblate.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    # a full run replaces everything but the separately run graze precision
    res = {k: v for k, v in old.items() if k == "precision_graze"}
    res.update({"machine": platform.platform(), "python": platform.python_version(),
                "started": time.strftime("%Y-%m-%d %H:%M"),
                "load": os.getloadavg()})
    print("precision vs the 30-digit oracle:")
    res["precision"] = precision()
    sweep = N_SWEEP[:3] if quick else N_SWEEP
    res["mem"] = {"single": {}, "batch": {}, "grad": {}}
    res["single"] = {}
    for code in A.CODES:
        res["single"][code] = {}
        for n in sweep:
            m = measure(code, "single", n)
            if m is None:
                break
            res["single"][code][str(n)] = m["median_s"]
            res["mem"]["single"].setdefault(code, {})[str(n)] = _mem(m)
            print(f"  single {code:18s} N={n:>10,d}: {m['median_s'] * 1e3:10.2f} ms")
            if m["median_s"] > 8.0:          # the next decade would take minutes
                break
    res["batch"] = {}
    for code in A.BATCH_CODES:
        m = measure(code, "batch", 0)
        if m is not None:
            res["batch"][code] = m["median_s"]
            res["mem"]["batch"][code] = _mem(m)
            print(f"  batch {code:18s} {BATCH_NPV} x {BATCH_NPT}: {m['median_s']:.3f} s")
    res["grad"] = {}
    for code in A.GRAD_CODES:
        res["grad"][code] = {}
        for n in GRAD_N:
            m = measure(code, "grad", n)
            if m is not None:
                res["grad"][code][str(n)] = m["median_s"]
                res["mem"]["grad"].setdefault(code, {})[str(n)] = _mem(m)
                print(f"  value+grad {code:18s} N={n:>8,d}: {m['median_s'] * 1e3:.2f} ms")
    res["load_after"] = os.getloadavg()
    with open(os.path.join(HERE, "results_oblate.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
