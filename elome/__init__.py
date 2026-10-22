"""Elómë — analytic, differentiable transit light curves on Apple
Silicon (MLX).

The name is Quenya: él "star" + lómë "dusk, dimness" -- a star dimmed,
which is what a transit is. Pronounced /ɛˈloː.mɛ/, "eh-LOH-meh", stressed
on the long ó, the final ë spoken.

Quadratic limb-darkened transits per Agol, Luger & Foreman-Mackey (2020),
with a batman-style frontend for everyday use:

    import elome
    params = elome.TransitParams()
    ...
    m = elome.TransitModel(params, t)
    flux = m.light_curve(params)

and a float32-conditioned batched backend (``elome.anvil``) for
GPU sampling with the anvil MCMC engine — which is
an optional integration dependency; the core needs only mlx and numpy.
"""

from .api import TransitModel, TransitParams
from .ellip import cel, cel3
from .flux import flux_dev, light_curve
from .greens import greens_transform_np, quad_g_coeffs, quad_norm
from .kepler import (kepler, kepler_E, kepler_E_sincos,
                     separation_keplerian)
from .anchored import anchor_constants, separation_anchored
from .poly import flux_dev_poly, sn_dev_poly
from .hybrid import (HYBRID2, HYBRID2_EPS, HYBRID4, HYBRID4_BASIS,
                     HYBRID4_EPS, HYBRID5, HybridLaw, LAWS as HYBRID_LAWS,
                     hybrid4_basis,
                     flux_dev_hybrid, hybrid_norms, shape_cols)
from .oblate import flux_dev_oblate, shape_cols_oblate
from .oblate_tau import contact_offsets_oblate
from .ld import (hybrid2_from_q, hybrid2_from_q_np, hybrid2_to_q_np,
                 hybrid4_from_q, hybrid4_from_q_np, hybrid4_to_q_np,
                 q_from_simplex_np, q_to_u, q_to_u_np, simplex_from_q,
                 simplex_from_q_np, u_to_q_np)
from .metal import flux_dev_from_tau, flux_dev_metal, metal_available
from .solution import sn_dev, sn_dev_with_aux
from .vjp import flux_dev_analytic

__all__ = [
    "TransitModel",
    "TransitParams",
    "cel",
    "cel3",
    "flux_dev",
    "flux_dev_analytic",
    "flux_dev_poly",
    "sn_dev_poly",
    "light_curve",
    "flux_dev_from_tau",
    "flux_dev_metal",
    "metal_available",
    "greens_transform_np",
    "quad_g_coeffs",
    "quad_norm",
    "kepler",
    "kepler_E",
    "kepler_E_sincos",
    "separation_keplerian",
    "separation_anchored",
    "anchor_constants",
    "q_to_u",
    "q_to_u_np",
    "u_to_q_np",
    "HybridLaw",
    "HYBRID2",
    "HYBRID2_EPS",
    "HYBRID4_EPS",
    "HYBRID4_BASIS",
    "hybrid4_basis",
    "flux_dev_oblate",
    "shape_cols_oblate",
    "contact_offsets_oblate",
    "flux_dev_ring",
    "shape_cols_ring",
    "ring_kinks",
    "flux_dev_spots",
    "shape_cols_spots",
    "spot_vector",
    "HYBRID4",
    "HYBRID5",
    "HYBRID_LAWS",
    "flux_dev_hybrid",
    "hybrid_norms",
    "shape_cols",
    "hybrid2_from_q",
    "hybrid2_from_q_np",
    "hybrid2_to_q_np",
    "hybrid4_from_q",
    "hybrid4_from_q_np",
    "hybrid4_to_q_np",
    "simplex_from_q",
    "simplex_from_q_np",
    "q_from_simplex_np",
    "sn_dev",
    "sn_dev_with_aux",
]

__version__ = "0.21.0"

# Ringed planets and spotted stars, exported lazily: their modules are
# imported only when one of these is first used (or a ring or spot
# parameter is set), so a model without them never loads them.
_LAZY = {"flux_dev_ring": ("rings", "flux_dev_ring"),
         "shape_cols_ring": ("rings", "shape_cols_ring"),
         "ring_kinks": ("rings_tau", "ring_kinks"),
         "flux_dev_spots": ("spots", "flux_dev_spots"),
         "shape_cols_spots": ("spots", "shape_cols_spots"),
         "spot_vector": ("spots", "spot_vector")}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    raise AttributeError(f"module 'elome' has no attribute {name!r}")
