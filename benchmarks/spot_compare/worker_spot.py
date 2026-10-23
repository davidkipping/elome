"""One measurement per subprocess, so no code's compilation caches or
allocator state leaks into another's timings.

    worker_spot.py CODE RES MODE CASE [OUT.npy]

MODE "curve": the relative flux on the precision grid, saved to OUT.npy.
MODE "time" / "time_grad": one light curve of N_SPEED points (value, or
value and gradient), printed as JSON: the first call (compilation
included) and the median of warm calls.
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import adapters_spot as A  # noqa: E402
from scenario_spot import N_PREC, N_SPEED, grid  # noqa: E402


def main():
    code, res, mode, case = sys.argv[1:5]
    res = None if res == "-" else int(res)
    curve, vg = A.make(code, case, res)
    if mode == "curve":
        np.save(sys.argv[5], curve(*grid(N_PREC)))
        return
    fn = curve if mode == "time" else vg
    x, y = grid(N_SPEED)
    t0 = time.perf_counter()
    fn(x, y)
    first = time.perf_counter() - t0
    warm = []
    while len(warm) < 7 and sum(warm) < 60.0:
        t0 = time.perf_counter()
        fn(x, y)
        warm.append(time.perf_counter() - t0)
    print(json.dumps({"first_s": first, "warm_s": statistics.median(warm), "reps": len(warm)}))


if __name__ == "__main__":
    main()
