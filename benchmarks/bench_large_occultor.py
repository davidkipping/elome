"""Accuracy for companions larger than their star (rp > 1), per law.

  float64   the graph, against a 30-digit direct integral of each law's
            intensity (oracle.flux_direct_law) on 41 separations a radius:
            total occultation, partial overlap and none
  float32   partial overlap only, on 2,001 float32 separations a radius,
            against the float64 graph on the same inputs: the Metal kernels
            (quadratic, hybrid2, hybrid4, hybrid5) and, for the polynomial
            law, which has no kernel, the MLX graph on the GPU

The hybrid weights carry mu^4 content (hybrid5's mu^4 coefficient is
w_1 - w_2). Nothing is timed.

    python benchmarks/bench_large_occultor.py

Writes benchmarks/large_occultor.json, which the docs check
(benchmarks/verify_doc_claims.py --check) reads.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RS = (1.5, 7.28, 20.0, 50.0)
U = (0.4, 0.25)
UP = (0.3, 0.2, -0.1, 0.05)
W = {"hybrid2": (0.3, 0.2), "hybrid4": (0.2, 0.2, 0.1, 0.1),
     "hybrid5": (0.3, 0.1, 0.1, 0.05, 0.15)}
LAWS = ("quadratic", "polynomial") + tuple(W)


def intensity(law):
    from elome.hybrid import LAWS as H, w_to_c
    if law == "quadratic":
        return lambda mu, mp: 1 - U[0] * (1 - mu) - U[1] * (1 - mu) ** 2
    if law == "polynomial":
        return lambda mu, mp: 1 - sum(u * (1 - mu) ** (n + 1)
                                      for n, u in enumerate(UP))
    c, d = w_to_c(law, W[law])
    eps = H[law].eps
    return lambda mu, mp: (c[0] + c[1] * mu ** 2 + c[2] * mu ** 4
                           + sum(dk / (mu ** 2 + e) ** 2 for dk, e in zip(d, eps)))


def graph64(z, r, law):
    import mlx.core as mx
    import elome as el
    with mx.stream(mx.cpu):
        z = mx.array(np.asarray(z, dtype=np.float64), dtype=mx.float64)
        if law == "quadratic":
            out = el.flux_dev(z, r, *U)
        elif law == "polynomial":
            out = el.flux_dev_poly(z, r, list(UP))
        else:
            out = el.flux_dev_hybrid(z, r, list(W[law]), law)
        return np.asarray(out, dtype=np.float64)


def fp32(z, r, law):
    import mlx.core as mx
    import elome as el
    from elome.metal import flux_dev_metal
    from elome.metal_hybrid import flux_dev_metal_hybrid
    z = mx.array(z)
    if law == "quadratic":
        out = flux_dev_metal(z, r, *U)
    elif law == "polynomial":
        with mx.stream(mx.gpu):
            out = el.flux_dev_poly(z, r, list(UP))
    else:
        out = flux_dev_metal_hybrid(z, r, law, list(W[law]))
    return np.asarray(out, dtype=np.float64)


def main():
    from oracle import flux_direct_law
    res = {"radii": list(RS), "weights": W, "u": U, "u_poly": UP,
           "float64": {}, "float32": {}}
    for r in RS:
        zs = np.concatenate([np.linspace(0.0, r - 1, 6)[:-1],
                             np.linspace(r - 1, r + 1, 33)[1:-1],
                             [r + 1.1, r + 2.0]])
        rf = float(np.float32(r))
        z32 = np.linspace(rf - 1, rf + 1, 2003)[1:-1].astype(np.float32)
        for law in LAWS:
            I = intensity(law)
            ref = np.array([flux_direct_law(r, z, I) - 1.0 for z in zs])
            e64 = float(np.abs(graph64(zs, r, law) - ref).max())
            ref32 = graph64(z32.astype(np.float64), rf, law)
            e32 = float(np.abs(fp32(z32, rf, law) - ref32).max())
            res["float64"].setdefault(law, {})[str(r)] = e64
            res["float32"].setdefault(law, {})[str(r)] = e32
            print(f"r = {r:5}  {law:10s}  float64 {e64:.1e}  float32 {e32:.1e}",
                  flush=True)
    path = os.path.join(HERE, "large_occultor.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
