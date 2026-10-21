"""Reproduce the measured batching and long-baseline numbers quoted in the docs.

    python benchmarks/verify_doc_claims.py           # measure; writes doc_claims.json
    python benchmarks/verify_doc_claims.py --only lcs_vs_loop   # one claim
    python benchmarks/verify_doc_claims.py --check   # check the docs quote them

Every such number in the docs traces here or to vjp.json (written by
bench_vjp.py). --check measures nothing: it formats the stored numbers as
the docs print them and fails, listing each one, if a page does not
contain it. Run the measurements on an otherwise-idle GPU.

--check also reads the cross-code comparison tables (spherical, oblate and
ringed planets) back to the results files the comparison runs write
(precision*.json, speed.json, batch_scaling.json,
oblate_compare/results_oblate.json, ring_compare/results_ring.json). A
number on a page passes if the measurement rounds to it at the precision
shown ("0.44 ms" means 0.435-0.445 ms; "about 1,700" means 1,650-1,750),
and a cell with no measurement must hold no number.
"""

import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")


def _latex(x):
    """'2.7e-07' -> '2.7\\times10^{-7}', as the docs write it."""
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def check():
    claims = json.load(open(os.path.join(HERE, "doc_claims.json")))
    v = json.load(open(os.path.join(HERE, "vjp.json")))["median_ms"]
    fa, fv, fm = v["autodiff_forward"], v["analytic_forward"], v["metal_forward"]
    ga, gv, gm = v["autodiff_value_grad"], v["analytic_value_grad"], v["metal_value_grad"]
    fe, ge = v["metal_ecc_forward"], v["metal_ecc_value_grad"]
    le, lc = v["chunked_eager"], v["chunked_compiled"]
    x_ad = f"{round(ga / gm, -1):.0f}"
    b = claims["batch_vs_loop"]
    per_loop_us = b["looped_ms"] / b["npv"] * 1e3
    per_batch_ns = b["batched_ms"] / b["npv"] * 1e6
    w = claims["fp32_baseline_flux_error"]
    expect = {
        "reference/performance.md": [
            f"| {fa:.0f} ms | {ga:,.0f} ms |", f"| {fv:.0f} ms | {gv:.0f} ms |",
            f"**{fm:.1f} ms** | **{gm:.1f} ms**", f"**{fe:.1f} ms** | **{ge:.1f} ms**",
            f"{round(fv / fm)} times faster", f"about {x_ad} times faster"],
        "reference/how-it-works.md": [f"{round(ga / gv, -1):.0f}–{x_ad} times faster"],
        "spherical/fitting.md": [
            f"{fm:.1f} ms at 1024 chains", f"pays {gm:.1f} ms", f"about {x_ad} times cheaper",
            f"about {le / lc:.1f} times as slow", f"({le:.1f} against {lc:.1f} ms"],
        "getting-started/installation.md": [
            f"about {round(fv / fm)} times slower", f"about\n  {round(gv / gm)} times slower"],
        "guides/batching.md": [
            f"about {round(b['looped_ms'], -2):,.0f} ms | about {round(per_loop_us, -1):.0f} µs",
            f"about {b['batched_ms']:.1f} ms | about {round(per_batch_ns, -1):.0f} ns",
            f"about {round(b['speedup'], -2):.0f} here",
            f"in float32 this is about "
            f"{round(claims['lcs_vs_loop']['speedup'], -2):.0f} times faster",
            "several\nhundred times faster with array-valued fields"],
        "guides/gradients-and-precision.md": [
            f"${_latex(w['1'])}$ for a span of one orbit",
            f"${_latex(w['100'])}$ for 100", f"${_latex(w['1000'])}$ for 1,000"],
    }
    bad = []
    for page, needles in expect.items():
        text = open(os.path.join(DOCS, page)).read()
        bad += [f"{page}: missing {n!r}" for n in needles if n not in text]
    if not 200 <= claims["lcs_vs_loop"]["speedup_arrays"] < 1000:
        bad.append("guides/batching.md: 'several hundred times' no longer holds "
                   f"({claims['lcs_vs_loop']['speedup_arrays']})")
    n_cmp, bad_cmp, pages_cmp = check_comparisons()
    bad += bad_cmp
    pages = len(set(expect) | pages_cmp)
    print("\n".join(bad) if bad else f"docs agree with the measurements "
          f"({sum(map(len, expect.values())) + n_cmp} numbers on {pages} pages)")
    return not bad


# ---- the cross-code comparison tables -------------------------------------

_UNITS = {"ms": 1e-3, "s": 1.0, "KB": 1e-3, "MB": 1.0, "GB": 1e3}
_NUMBER = re.compile(r"\$(\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}\$"
                     r"|(\d[\d,]*(?:\.\d+)?)(?:\s*(ms|s|KB|MB|GB))?(?![\w.])")


def _shown(cell):
    """The first number in a cell, in s or MB, and half its last digit:
    '0.44 ms' -> (4.4e-4, 5e-6), 'about 1,700' -> (1700, 50). None if the
    cell holds no number."""
    m = _NUMBER.search(cell)
    if m is None:
        return None
    if m.group(1):
        digits, scale = m.group(1), 10.0 ** int(m.group(2))
    else:
        digits, scale = m.group(3).replace(",", ""), _UNITS.get(m.group(4), 1.0)
    if "." in digits:
        last = 10.0 ** -len(digits.split(".")[1])
    else:
        last = 10.0 ** (len(digits) - len(digits.rstrip("0")))
    return float(digits) * scale, 0.5 * last * scale


def _agrees(cell, value):
    shown = _shown(cell)
    if value is None or shown is None:
        return value is None and shown is None
    return abs(value - shown[0]) <= shown[1] * (1 + 1e-9)


def _table(text, after):
    """{first cell: other cells} of the first table after `after`."""
    rows, lines = {}, text[text.index(after):].splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("|"))
    for line in lines[i:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0]] = cells[1:]
    return rows


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def check_comparisons():
    """(numbers checked, failures, pages) for the comparison pages."""
    load = lambda *p: json.load(open(os.path.join(HERE, *p)))  # noqa: E731
    circ = {r["code"]: r["max_err"] for r in load("precision.json")}
    ecc = {r["code"]: r["max_err"] for r in load("precision_ecc.json")}
    speed = load("speed.json")
    scaling = [r["npv"] / r["median_s"] for r in load("batch_scaling.json")
               if r["code"] == "elome_gpu" and r["npv"] >= 256]
    ob = load("oblate_compare", "results_oblate.json")
    rg = load("ring_compare", "results_ring.json")

    thr = {"batman": 1, "pytransit": 1, "exoplanet": 1}

    def sp(code, mode, n=0, key="median_s"):
        threads = 0 if mode == "batch" else thr.get(code, 0)
        rec = next(r for r in speed if r["code"] == code and r["mode"] == mode
                   and r["n"] == n and r["threads"] == threads)
        return rec[key]

    def batch_ratio(res, gpu):
        return min(v for c, v in res.items() if not c.startswith("elome")) / res[gpu]

    sph_codes = {"Elómë, float32 (GPU)": "elome_fp32", "Elómë, float64 (CPU)": "elome_fp64",
                 "batman": "batman", "jaxoplanet": "jaxoplanet",
                 "exoplanet-core": "exoplanet", "PyTransit": "pytransit"}
    sph_batch = {"Elómë, float32 (GPU), one call": "elome_gpu",
                 "PyTransit (native arrays)": "pytransit",
                 "jaxoplanet (`jax.vmap`, CPU)": "jaxoplanet",
                 "exoplanet-core (loop)": "exoplanet", "batman (loop)": "batman"}
    sph_tables = {
        "**Accuracy**": {
            "Elómë, float64": [circ["elome fp64"], ecc["elome fp64"]],
            "exoplanet-core": [circ["exoplanet-core"], ecc["exoplanet-core"]],
            "jaxoplanet": [circ["jaxoplanet (order=50)"], ecc["jaxoplanet (order=10)"]],
            "batman": [circ["batman"], ecc["batman"]],
            "PyTransit (exact)": [circ["pytransit (exact)"], ecc["pytransit (exact)"]],
            "Elómë, float32 (GPU)": [circ["elome fp32(GPU)"], ecc["elome fp32(GPU)"]]},
        "**One light curve**": {
            label: [sp(c, "single", n) for n in (10**3, 10**5, 10**6, 10**7)]
            for label, c in sph_codes.items()},
        "**Many light curves**": {
            label: [sp(c, "batch"), 512 / sp(c, "batch"), sp(c, "batch", key="peak_mb")]
            for label, c in sph_batch.items()},
        "**Value and gradient**": {
            label: [sp(c, "grad", n) for n in (10**4, 10**5)]
            for label, c in (("Elómë, float32 (GPU)", "elome_fp32"),
                             ("jaxoplanet (CPU)", "jaxoplanet"),
                             ("Elómë, float64 (CPU)", "elome_fp64"))},
        "**Memory**": {
            label: [sp(c, "single", n, "peak_mb") for n in (10**5, 10**7)]
            for label, c in sph_codes.items()},
    }
    sph_cpu_e7 = min(sp(c, "ecc", 10**7) for c in sph_codes.values() if not c.startswith("elome"))
    sph_prose = [
        (r"take Elómë's GPU path to (\S+ ms) at \$10\^7\$ points", sp("elome_fp32", "ecc", 10**7)),
        (r"against (\S+ ms) for the fastest CPU code", sph_cpu_e7),
        (r"stays at (about \S+) light curves per second", min(scaling)),
        (r"stays at (about \S+) light curves per second", max(scaling)),
        (r"(about \S+ KB) per point", sp("elome_fp64", "single", 10**7, "peak_mb") / 1e7),
    ]

    def compare_table(r, cols, rows):
        """rows: label -> (results key path below each code, or a callable)."""
        return {label: [get(r, c) for c in cols] for label, get in rows.items()}

    def one(*path):
        return lambda r, c: _get(r, *path[:1], c, *path[1:])

    common = {
        "one light curve, $N = 10^3$": one("single", "1000"),
        "one light curve, $N = 10^5$": one("single", "100000"),
        "512 sets × $10^4$ points": one("batch"),
        "value + gradient, $N = 10^5$": one("grad", "100000"),
        "memory, $N = 10^5$": lambda r, c: _get(r, "mem", "single", c, "100000", "peak_mb"),
    }
    ob_cols = ["elome_fp32", "greenlantern", "jojo", "squishyplanet", "eclipsoid", "elome_fp64"]
    ob_rows = dict(common, **{
        "error vs 30-digit integral": one("precision", "max"),
        "one light curve, $N = 10^7$": one("single", "10000000"),
        "memory, $N = 10^7$": lambda r, c: _get(r, "mem", "single", c, "10000000", "peak_mb"),
        "memory, 512 sets × $10^4$": lambda r, c: _get(r, "mem", "batch", c, "peak_mb"),
    })
    rg_cols = ["elome_fp32", "pyppluss", "squishyplanet", "jojo", "pryngles", "elome_fp64"]
    rg_rows = dict(common, **{
        "error, opaque ring": one("precision_opaque", "max"),
        "error, translucent ($\\tau = 0.6$)": one("precision_translucent", "max"),
        "error, nearly edge-on (`ring_f = 0.95`)": one("precision_edge", "max"),
        "one light curve, $N = 10^6$": one("single", "1000000"),
        "memory, $N = 10^6$": lambda r, c: _get(r, "mem", "single", c, "1000000", "peak_mb"),
    })
    n20 = [rg["single"]["pyppluss_n20"][n] / rg["single"]["pyppluss"][n]
           for n in rg["single"]["pyppluss_n20"]]
    fp32_e7 = [sp("elome_fp32", "single", 10**7, "peak_mb"),
               ob["mem"]["single"]["elome_fp32"]["10000000"]["peak_mb"],
               rg["mem"]["single"]["elome_fp32"]["10000000"]["peak_mb"]]

    def fastest(r):
        """Elómë float32 is the fastest code at every size and in every mode."""
        for mode in ("single", "grad"):
            for n in {n for v in r[mode].values() for n in v}:
                t = {c: v[n] for c, v in r[mode].items() if n in v}
                if min(t, key=t.get) != "elome_fp32":
                    return False
        return min(r["batch"], key=r["batch"].get) == "elome_fp32"

    pages = {
        "spherical/accuracy-and-cost.md": (sph_tables, sph_prose),
        "oblate/accuracy-and-cost.md": (
            {"The benchmark is the spherical one's transit": compare_table(ob, ob_cols, ob_rows)},
            [(r"timed up to \$10\^6\$ points \((\S+ s),", ob["single"]["eclipsoid"]["1000000"]),
             (r"timed up to \$10\^6\$ points \(\S+ s, (\S+ MB)\)",
              ob["mem"]["single"]["eclipsoid"]["1000000"]["peak_mb"]),
             (r"(about \S+ KB) per point",
              ob["mem"]["single"]["elome_fp64"]["10000000"]["peak_mb"] / 1e7)]),
        "rings/accuracy-and-fitting.md": (
            {"The reference is an independent float64": compare_table(rg, rg_cols, rg_rows)},
            [(r"timed on (\S+) to \S+ sets", min(rg["batch_scaled_from"].values())),
             (r"timed on \S+ to (\S+) sets", max(rg["batch_scaled_from"].values())),
             (r"at order 20 its error is (\$[^$]+\$)",
              min(rg[k]["pyppluss_n20"]["max"] for k in rg if k.startswith("precision_"))),
             (r"at order 20 its error is \$[^$]+\$ to (\$[^$]+\$)",
              max(rg[k]["pyppluss_n20"]["max"] for k in rg if k.startswith("precision_"))),
             (r"it takes (about \S+) times as long", math.prod(n20) ** (1 / len(n20))),
             (r"which caps it near (\S+ GB)",
              max(v["peak_mb"] for v in rg["mem"]["single"]["elome_fp64"].values()))]),
        "reference/performance.md": ({}, [
            (r"from (about \S+) times \(spherical, against PyTransit\)",
             batch_ratio({c: sp(c, "batch") for c in sph_batch.values()}, "elome_gpu")),
            (r"to (about \S+) times \(ringed\)", batch_ratio(rg["batch"], "elome_fp32"))]),
        "index.md": ({}, [
            (r"integral to (\$[^$]+\$)", circ["elome fp64"]),
            (r"\$10\^7\$ points in (about \S+ ms)", sp("elome_fp32", "single", 10**7)),
            (r"(about \S+) light curves of \$10\^5\$ points", min(scaling)),
            (r"(about \S+) light curves of \$10\^5\$ points", max(scaling))]),
    }
    claims = [
        ("reference/performance.md", "fastest for oblate planets at every size",
         fastest(ob)),
        ("reference/performance.md", "fastest for ringed planets at every size",
         fastest(rg)),
        ("reference/performance.md", "float32, 10^7 points, under 300 MB for each planet type",
         max(fp32_e7) < 300),
    ]

    n, bad = 0, []
    for page, (tables, prose) in pages.items():
        text = open(os.path.join(DOCS, page)).read()
        flat = " ".join(text.split())
        for after, rows in tables.items():
            found = _table(text, after)
            for label, values in rows.items():
                cells = found.get(label)
                if cells is None or len(cells) != len(values):
                    bad.append(f"{page}: no row {label!r} with {len(values)} values")
                    continue
                for cell, v in zip(cells, values):
                    n += 1
                    if not _agrees(cell, v):
                        bad.append(f"{page}: {label!r}: {cell!r}, measured {v!r}")
        for pattern, v in prose:
            n += 1
            m = re.search(pattern, flat)
            if m is None or not _agrees(m.group(1), v):
                bad.append(f"{page}: /{pattern}/: "
                           f"{m.group(1) if m else 'not found'!r}, measured {v!r}")
    for page, what, ok in claims:
        n += 1
        if not ok:
            bad.append(f"{page}: no longer true: {what}")
    return n, bad, set(pages)


if "--check" in sys.argv:
    sys.exit(0 if check() else 1)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

# every claim by default; --only NAME [NAME ...] re-measures just those and
# keeps the others' stored numbers
only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
path = os.path.join(HERE, "doc_claims.json")
out = json.load(open(path)) if only else {}


def want(name):
    return only is None or name in only


# ---- claim 1: looped vs batched (the batching rule) ----------------------
from elome.metal import flux_dev_metal  # noqa: E402

if want("batch_vs_loop"):
    rng = np.random.default_rng(0)
    npv, npt = 10_000, 1_000
    z_row = np.abs(rng.uniform(0, 1.3, npt)).astype(np.float32)
    r_all = (0.1 * (1 + 0.1 * rng.standard_normal(npv))).astype(np.float32)
    u1 = np.full(npv, 0.4, np.float32)
    u2 = np.full(npv, 0.25, np.float32)

    z1 = mx.array(z_row[None, :])
    for _ in range(3):
        mx.eval(flux_dev_metal(z1, mx.array(r_all[:1]), mx.array(u1[:1]),
                               mx.array(u2[:1])))
    t0 = time.perf_counter()
    for i in range(200):
        mx.eval(flux_dev_metal(z1, mx.array(r_all[i:i + 1]), mx.array(u1[:1]),
                               mx.array(u2[:1])))
    t_loop = (time.perf_counter() - t0) / 200 * npv

    zb = mx.array(np.broadcast_to(z_row, (npv, npt)).copy())
    rb, u1b, u2b = mx.array(r_all), mx.array(u1), mx.array(u2)
    for _ in range(3):
        mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
        ts.append(time.perf_counter() - t0)
    t_batch = float(np.median(ts))
    out["batch_vs_loop"] = {
        "npv": npv, "npt": npt,
        "looped_ms": round(t_loop * 1e3, 1),
        "batched_ms": round(t_batch * 1e3, 2),
        "speedup": round(t_loop / t_batch),
    }
    print("batch_vs_loop:", out["batch_vs_loop"], flush=True)

# ---- claim 2: fp32 phase error vs the baseline the data span ------------
# The frontend re-centres the times on their midpoint in float64, so what
# float32 still sees is the half-span of the data. Here: transits at the
# start and at the end of a baseline of n_orbits orbits (t0 = 0), fp32
# against fp64 through the frontend, max in-transit flux error.
from elome import TransitModel, TransitParams  # noqa: E402

if want("fp32_baseline_flux_error"):

    per = 10.0
    wrap = {}
    for n_orbits in (1, 100, 1000):
        p = TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = 0.0, per, 0.1, 12.0, 88.0
        p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4, 0.25], "quadratic"
        win = np.linspace(-0.15, 0.15, 4001)
        t = np.concatenate([win, n_orbits * per + win])
        f32 = TransitModel(p, t, dtype=mx.float32).light_curve(p)
        f64 = TransitModel(p, t).light_curve(p)
        wrap[str(n_orbits)] = float(np.abs(f32 - f64).max())
    out["fp32_baseline_flux_error"] = {k: f"{v:.1e}" for k, v in wrap.items()}
    print("fp32_baseline_flux_error:", out["fp32_baseline_flux_error"], flush=True)


# ---- claim 3: light_curves vs a loop over light_curve --------------------
# 2,000 sets of a 301-point light curve, float32 on the GPU
if want("lcs_vs_loop"):
    rng = np.random.default_rng(1)
    t = np.linspace(-0.15, 0.15, 301)
    sets = []
    for j in range(2000):
        p = TransitParams()
        p.t0, p.per, p.a, p.limb_dark, p.u = 0.0, 3.45, 8.8, "quadratic", [0.4, 0.25]
        p.rp, p.inc = rng.uniform(0.05, 0.15), rng.uniform(87.0, 90.0)
        p.ecc, p.w = (0.0, 90.0) if j % 2 else (rng.uniform(0, 0.5), rng.uniform(0, 360))
        sets.append(p)
    m = TransitModel(sets[0], t, dtype=mx.float32)
    for _ in range(3):
        m.light_curves(sets)
        m.light_curve(sets[0])
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        m.light_curves(sets)
        ts.append(time.perf_counter() - t0)
    t_batch = float(np.median(ts))
    # the same sets as one TransitParams of array-valued fields
    pa = TransitParams()
    for f in ("t0", "per", "a", "limb_dark", "u"):
        setattr(pa, f, getattr(sets[0], f))
    for f in ("rp", "inc", "ecc", "w"):
        setattr(pa, f, np.array([getattr(q, f) for q in sets]))
    assert np.array_equal(m.light_curves(pa), m.light_curves(sets))
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        m.light_curves(pa)
        ts.append(time.perf_counter() - t0)
    t_arr = float(np.median(ts))
    t0 = time.perf_counter()
    for p in sets:
        m.light_curve(p)
    t_loop = time.perf_counter() - t0
    out["lcs_vs_loop"] = {"n_sets": 2000, "n_t": 301,
                          "loop_ms": round(t_loop * 1e3, 1),
                          "batched_ms": round(t_batch * 1e3, 2),
                          "speedup": round(t_loop / t_batch),
                          "batched_arrays_ms": round(t_arr * 1e3, 2),
                          "speedup_arrays": round(t_loop / t_arr)}
    print("lcs_vs_loop:", out["lcs_vs_loop"], flush=True)

with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("wrote doc_claims.json")
