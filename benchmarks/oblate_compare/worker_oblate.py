"""One oblate-benchmark measurement in an isolated process.

Usage: python worker_oblate.py <code> <mode> <n>   (mode: single | batch | grad)
       python worker_oblate.py <code> curve <n> <out.npy>   (the light curve, untimed)
Emits one JSON line: {"code", "mode", "n", "median_s", "min_s", "reps"},
with the peak memory of one call (../memwatch.py) measured after the timing.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import adapters_oblate as A  # noqa: E402
import importlib  # noqa: E402

from memwatch import MemWatch  # noqa: E402


def main():
    code, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == "curve":
        f = getattr(A, f"{code}_run")(getattr(A, f"{code}_prepare")(n))
        np.save(sys.argv[4], np.asarray(f, np.float64))
        return
    for mod in A.IMPORTS[code.split("_")[0]]:
        importlib.import_module(mod)    # not counted in the memory
    watch = MemWatch()
    if mode == "single":
        state, run = getattr(A, f"{code}_prepare")(n), getattr(A, f"{code}_run")
    elif mode == "batch":
        state, run = getattr(A, f"{code}_batch_prepare")(), getattr(A, f"{code}_batch_run")
    elif mode == "grad":
        state, run = getattr(A, f"{code}_grad_prepare")(n), getattr(A, f"{code}_grad_run")
    else:
        raise SystemExit(f"unknown mode {mode}")
    ts, total = [], 0.0
    # at least 3 repetitions, stop past 2 s or 15 reps (or one slow call)
    while len(ts) < 15 and (len(ts) < 3 or total < 2.0):
        t0 = time.perf_counter()
        run(state)
        dt = time.perf_counter() - t0
        ts.append(dt)
        total += dt
        if dt > 30.0:
            break
    print(json.dumps({"code": code, "mode": mode, "n": n,
                      "median_s": float(np.median(ts)),
                      "min_s": float(np.min(ts)), "reps": len(ts),
                      **watch.measure(lambda: run(state),
                                      reps=1 if np.median(ts) > 5 else 2)}))


if __name__ == "__main__":
    main()
