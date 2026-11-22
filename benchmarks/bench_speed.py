"""Speed benchmark runner: one subprocess per (code, mode, n, threads)
so thermal state, JIT caches, and thread pools never leak between
measurements. Writes speed.json incrementally (safe to re-run; completed
measurements are skipped).

    python benchmarks/bench_speed.py [--out FILE] [--codes CODE ...] [--rounds R] [--redo]

``--out`` names the results file (default benchmarks/speed.json, this
repository's measurements): a new file starts afresh, so on another machine
pass one. ``--codes`` runs only those codes (elome_fp32, elome_fp64,
elome_gpu, batman, pytransit, roadrunner, exoplanet, jaxoplanet, pizzetti;
the hybrid2 law: elome_fp32_h2, elome_fp64_h2, elome_gpu_h2, roadrunner_h2).
``--rounds R`` times each measurement in R separate processes and keeps the
median. ``--redo`` measures the selected codes again, replacing their
results.

Axes on this machine (M2 Max):
  * CPU cores: thread counts 1/4/8/12 for codes that support threading
    (batman via OpenMP if its extension was built with it — probed;
    pytransit via numba). exoplanet-core is single-threaded by design;
    jaxoplanet's XLA pool and MLX's CPU stream are all-cores and not
    cleanly partitionable (reported as threads=0 meaning "all").
  * GPU cores: not partitionable on Apple Silicon — instead the N-sweep
    shows the single-GPU saturation curve (small N under-fills the 38
    cores; large N saturates them). Cross-chip scaling (30-core Max vs
    76-core Ultra) is the only true GPU-core axis.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "speed.json")

from scenario import N_SWEEP, THREADS  # noqa: E402

# (code, thread-counts). 0 = "all cores / not partitionable".
SINGLE = [
    ("elome_fp32", [0]),          # GPU
    ("elome_fp64", [0]),          # MLX CPU stream
    ("batman", [1]),                    # probed: this build lacks OpenMP
    ("pytransit", THREADS),             # numba
    ("roadrunner", [1, 12]),            # numba, its nthreads
    ("exoplanet", [1]),                 # single-threaded C++
    ("jaxoplanet", [0]),                # XLA CPU pool (all cores)
    ("pizzetti", [1]),                  # single-threaded numba
    # the hybrid2 law, for the codes that take it
    ("elome_fp32_h2", [0]),
    ("elome_fp64_h2", [0]),
    ("roadrunner_h2", [1, 12]),
]
BATCH = ["elome_gpu", "pytransit", "roadrunner", "jaxoplanet", "batman",
         "exoplanet", "pizzetti", "elome_gpu_h2", "roadrunner_h2"]
# eccentric sweep: same codes, each solving Kepler's equation itself, so
# the numbers include the orbit as well as the photometry.
ECC_CODES = [
    ("elome_fp32", [0]),
    ("elome_fp64", [0]),
    ("batman", [1]),
    ("pytransit", THREADS),
    ("roadrunner", [1]),
    ("exoplanet", [1]),
    ("jaxoplanet", [0]),
    ("pizzetti", [1]),
    ("elome_fp32_h2", [0]),
    ("elome_fp64_h2", [0]),
    ("roadrunner_h2", [1]),
]

# value + gradient (the differentiable codes), at these N
GRAD = [("elome_fp32", [0]), ("elome_fp64", [0]), ("jaxoplanet", [0]),
        ("elome_fp32_h2", [0]), ("elome_fp64_h2", [0])]
GRAD_N = [10_000, 100_000]

SLOW_SKIP_S = 20.0  # skip larger N once a code's median exceeds this


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return []


def save(rows):
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)


def have(rows, **kw):
    return any(all(r.get(k) == v for k, v in kw.items()) for r in rows)


def run_one(code, mode, n, threads, rounds=1):
    if rounds > 1:
        recs = [run_one(code, mode, n, threads) for _ in range(rounds)]
        good = sorted((r for r in recs if "error" not in r), key=lambda r: r["median_s"])
        if not good:
            return recs[0]
        rec = dict(good[(len(good) - 1) // 2])
        rec["rounds"] = rounds
        rec["round_medians_s"] = [r["median_s"] for r in good]
        return rec
    env = dict(os.environ)
    t = threads if threads > 0 else 12
    env.update(BENCH_NTHREADS=str(threads if threads > 0 else 1),
               NUMBA_NUM_THREADS=str(t), OMP_NUM_THREADS=str(t),
               VECLIB_MAXIMUM_THREADS=str(t))
    r = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"),
                        code, mode, str(n)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return {"code": code, "mode": mode, "n": n, "threads": threads,
                "error": (r.stderr or "")[-400:]}
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    rec["threads"] = threads
    return rec


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=OUT, help="results file (a new one starts afresh)")
    ap.add_argument("--codes", nargs="+", help="only these codes")
    ap.add_argument("--rounds", type=int, default=1,
                    help="processes per measurement, the median kept")
    ap.add_argument("--redo", action="store_true", help="re-measure the selected codes")
    a = ap.parse_args()
    if a.redo and not a.codes:
        ap.error("--redo needs --codes")
    OUT = os.path.abspath(a.out)
    keep = (lambda code: True) if not a.codes else (lambda code: code in a.codes)  # noqa: E731
    rows = load()
    if a.redo:
        rows = [r for r in rows if not keep(r["code"])]
    def measure(code, mode, n, threads):
        return run_one(code, mode, n, threads, a.rounds)
    for code, thread_list in [c for c in SINGLE if keep(c[0])]:
        for threads in thread_list:
            slow = False
            for n in N_SWEEP:
                if have(rows, code=code, mode="single", n=n,
                        threads=threads):
                    continue
                if slow:
                    break
                rec = measure(code, "single", n, threads)
                rows.append(rec)
                save(rows)
                med = rec.get("median_s")
                print(f"{code:>18s} single n={n:>9d} threads={threads}: "
                      f"{med if med is None else f'{med*1e3:10.2f} ms'} "
                      f"{rec.get('error', '')[:60]}", flush=True)
                if rec.get("error") or (med and med > SLOW_SKIP_S):
                    slow = True

    for code, thread_list in [c for c in ECC_CODES if keep(c[0])]:
        for threads in thread_list:
            slow = False
            for n in N_SWEEP:
                if have(rows, code=code, mode="ecc", n=n, threads=threads):
                    continue
                if slow:
                    break
                rec = measure(code, "ecc", n, threads)
                rows.append(rec)
                save(rows)
                med = rec.get("median_s")
                print(f"{code:>18s}    ecc n={n:>9d} threads={threads}: "
                      f"{med if med is None else f'{med*1e3:10.2f} ms'} "
                      f"{rec.get('error', '')[:60]}", flush=True)
                if rec.get("error") or (med and med > SLOW_SKIP_S):
                    slow = True

    for code in [c for c in BATCH if keep(c)]:
        if have(rows, code=code, mode="batch"):
            continue
        rec = measure(code, "batch", 0, 0)
        rows.append(rec)
        save(rows)
        med = rec.get("median_s")
        print(f"{code:>18s} batch 512x1e5: "
              f"{med if med is None else f'{med:10.3f} s'} "
              f"{rec.get('error', '')[:60]}", flush=True)

    for code, thread_list in [c for c in GRAD if keep(c[0])]:
        for threads in thread_list:
            for n in GRAD_N:
                if have(rows, code=code, mode="grad", n=n, threads=threads):
                    continue
                rec = measure(code, "grad", n, threads)
                rows.append(rec)
                save(rows)
                med = rec.get("median_s")
                print(f"{code:>18s}   grad n={n:>9d}: "
                      f"{med if med is None else f'{med*1e3:10.2f} ms'} "
                      f"{rec.get('error', '')[:60]}", flush=True)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
