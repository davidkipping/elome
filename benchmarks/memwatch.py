"""Peak memory of a benchmark call, above the process's level after the
code under test is imported.

Used by the timing workers (worker.py, oblate_compare/worker_oblate.py,
ring_compare/worker_ring.py):

    import <the code under test>        # its import cost is not counted
    watch = MemWatch()                  # baseline: here, before any data
    state = prepare(n)                  # data, JIT compilation, warm-up
    ... timing loop ...
    mem = watch.measure(lambda: run(state), reps=2)

The measure is the process's physical footprint (macOS's phys_footprint,
what Activity Monitor shows as "Memory"), polled about every millisecond
by a separate sampler PROCESS -- a thread would stall while a C extension
holds the GIL. Reported as ``peak_mb``: the peak during the measured calls
minus the footprint at the baseline, so it includes what the code keeps
between calls (compiled code, memory pools, MLX's buffer cache) and a
call's transient peak, but not the interpreter and library imports.

One number for every code. Resident set size would not do: MLX's buffers
are Metal buffers in unified memory, which RSS counts once the CPU has
touched them (an MLX float64 CPU array) and not when only the GPU has (a
float32 GPU array); the footprint counts both, once, and numpy's and
JAX's memory alike. ``mlx_peak_mb`` (MLX's allocator peak above its
baseline) is recorded alongside for reference only; it is part of
``peak_mb``. GPU memory of a non-Metal device (OpenCL) is not seen.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

_SAMPLER = r"""
import ctypes, ctypes.util, json, os, sys, time
pid, go, stop, interval = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
lib = ctypes.CDLL(ctypes.util.find_library("proc"))
class RUsage(ctypes.Structure):     # rusage_info_v4: a 16-byte uuid, then uint64s
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [("f%d" % i, ctypes.c_uint64) for i in range(40)]
def footprint():                    # ri_phys_footprint, the 8th field
    ru = RUsage()
    if lib.proc_pid_rusage(pid, 4, ctypes.byref(ru)) != 0:
        raise OSError
    return ru.f7
base = footprint()
print("ready", flush=True)
while not os.path.exists(go):
    time.sleep(0.005)
peak = footprint()
while not os.path.exists(stop):
    try:
        fp = footprint()
    except OSError:
        break
    if fp > peak:
        peak = fp
    time.sleep(interval)
print(json.dumps({"base": base, "peak": peak}), flush=True)
"""


def _sampler_python():
    """The sampler needs only the standard library."""
    return os.environ.get("BENCH_MEMWATCH_PY") or sys.executable


class MemWatch:
    """Baseline at construction; ``measure`` samples the given calls."""

    def __init__(self, interval=0.001):
        mx = sys.modules.get("mlx.core")
        self._mlx0 = None
        if mx is not None:
            mx.synchronize()
            self._mlx0 = mx.get_active_memory()
        self._tmp = tempfile.TemporaryDirectory()
        self._go = os.path.join(self._tmp.name, "go")
        self._stop = os.path.join(self._tmp.name, "stop")
        self._proc = subprocess.Popen(
            [_sampler_python(), "-c", _SAMPLER, str(os.getpid()), self._go,
             self._stop, str(interval)], stdout=subprocess.PIPE, text=True)
        assert self._proc.stdout.readline().strip() == "ready"

    def measure(self, call, reps=2):
        """Run ``call`` ``reps`` times under the sampler; returns peak_mb
        and, when MLX is loaded, mlx_peak_mb (for reference)."""
        mx = sys.modules.get("mlx.core")
        if mx is not None:
            mx.synchronize()
            mx.reset_peak_memory()
        open(self._go, "w").close()
        time.sleep(0.01)
        for _ in range(reps):
            call()
        if mx is not None:
            mx.synchronize()
        time.sleep(0.01)
        open(self._stop, "w").close()
        out = json.loads(self._proc.stdout.readline())
        self._proc.wait()
        self._tmp.cleanup()
        res = {"peak_mb": max(0, out["peak"] - out["base"]) / 2**20}
        if mx is not None:
            res["mlx_peak_mb"] = max(0, mx.get_peak_memory() - (self._mlx0 or 0)) / 2**20
        return res
