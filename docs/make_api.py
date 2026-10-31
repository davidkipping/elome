"""Generate the API reference (docs/reference/api.md) from the package.

    python docs/make_api.py

Run it after changing a public signature or docstring, and commit the
result. The reference is generated here rather than by autodoc at build
time, so building the documentation needs neither MLX nor a GPU.
"""
import importlib
import inspect
import pathlib
import textwrap

OUT = pathlib.Path(__file__).resolve().parent / "reference" / "api.md"

SECTIONS = [
    ("Light curves: the frontend",
     "The batman-style interface. See {doc}`../spherical/light-curves`.",
     ["elome.TransitParams", "elome.TransitModel",
      "elome.TransitModel.light_curve", "elome.TransitModel.light_curves",
      "elome.TransitModel.light_curve_mx"]),
    ("The sampler API",
     "Per-chain parameters in, per-chain light curves out, with analytic "
     "gradients. See {doc}`../guides/sampler-api`.",
     ["elome.flux_dev_from_tau", "elome.flux_dev_metal",
      "elome.contact_offsets_oblate", "elome.contact_offsets_tidal",
      "elome.tidal_q_from_hf", "elome.tidal_hf_from_q",
      "elome.oblate_tau.tidal_outline", "elome.ring_kinks",
      "elome.metal_available"]),
    ("Models and targets for anvil",
     "Ready-made models and fitting problems for the anvil engine. See "
     "{doc}`../spherical/fitting`.",
     ["elome.anvil.make_quad_transit_flux", "elome.anvil.make_ecc_transit_flux",
      "elome.anvil.make_hybrid_transit_flux", "elome.anvil.make_eclipse_flux",
      "elome.anvil.transit_param_names",
      "elome.anvil.PARAM_NAMES", "elome.anvil.PARAM_NAMES_ECC",
      "elome.anvil.make_transit_target", "elome.anvil.TransitTarget",
      "elome.anvil.make_target", "elome.anvil.QuadTransitTarget",
      "elome.anvil.make_ecc_target", "elome.anvil.EccTransitTarget",
      "elome.anvil.ecc_constraint_penalty", "elome.anvil.PenalizedLogLike",
      "elome.anvil.import_engine", "elome.anvil.DEFAULT_BOUNDS"]),
    ("Limb darkening",
     "Laws, priors and the hybrid-law basis. See "
     "{doc}`../spherical/limb-darkening` and {doc}`hybrid-laws`.",
     ["elome.q_to_u", "elome.q_to_u_np", "elome.u_to_q_np",
      "elome.hybrid2_from_q", "elome.hybrid2_from_q_np", "elome.hybrid2_to_q_np",
      "elome.hybrid4_from_q", "elome.hybrid4_from_q_np", "elome.hybrid4_to_q_np",
      "elome.simplex_from_q", "elome.simplex_from_q_np", "elome.q_from_simplex_np",
      "elome.HybridLaw", "elome.HYBRID2", "elome.HYBRID4", "elome.HYBRID5",
      "elome.HYBRID2_EPS", "elome.HYBRID4_EPS", "elome.HYBRID4_BASIS",
      "elome.HYBRID_LAWS", "elome.hybrid_norms", "elome.hybrid4_basis",
      "elome.hybrid.ladder"]),
    ("Light curves from separations (float64 reference graphs)",
     "Functions of the sky separation z (spherical), the planet's centre "
     "in its principal frame (oblate, ringed), or the planet's and the spots' "
     "positions on the sky (starspots). They follow MLX's broadcasting and "
     "type-promotion rules; see {doc}`../guides/gradients-and-precision`.",
     ["elome.flux_dev", "elome.flux_dev_analytic", "elome.flux_dev_poly",
      "elome.flux_dev_hybrid", "elome.shape_cols", "elome.flux_dev_oblate",
      "elome.shape_cols_oblate", "elome.flux_dev_ring", "elome.shape_cols_ring",
      "elome.oblate.principal_frame", "elome.flux_dev_spots", "elome.shape_cols_spots",
      "elome.spot_vector",
      "elome.light_curve", "elome.sn_dev", "elome.sn_dev_with_aux",
      "elome.sn_dev_poly"]),
    ("Orbits and time",
     "", ["elome.orbit.epoch_center_times", "elome.orbit.occultation_center_times",
          "elome.occultation_offset", "elome.eclipse.occultation_offset_dev",
          "elome.light_travel_delay", "elome.separation_keplerian",
          "elome.separation_anchored", "elome.anchor_constants",
          "elome.kepler_E", "elome.kepler.kepler", "elome.kepler.kepler_E_sincos",
          "elome.trig.sincos"]),
    ("Special functions and transforms",
     "", ["elome.cel", "elome.cel3", "elome.greens_transform_np",
          "elome.quad_g_coeffs", "elome.quad_norm"]),
]


def resolve(path):
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ImportError(path)


def _ann(a):
    return a if isinstance(a, str) else getattr(a, "__name__", repr(a))


def signature(name, obj):
    """name(arg: annotation = default, ...) -> return, without `self`."""
    target = getattr(obj, "__wrapped__", obj)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    parts, star_done = [], False
    for prm in sig.parameters.values():
        if prm.name == "self":
            continue
        if prm.kind == prm.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        if prm.kind == prm.VAR_POSITIONAL:
            star_done = True
            text = f"*{prm.name}"
        elif prm.kind == prm.VAR_KEYWORD:
            text = f"**{prm.name}"
        else:
            text = prm.name
        if prm.annotation is not prm.empty:
            text += f": {_ann(prm.annotation)}"
        if prm.default is not prm.empty:
            text += (" = " if prm.annotation is not prm.empty else "=") + repr(prm.default)
        parts.append(text)
    ret = ""
    if sig.return_annotation is not sig.empty:
        ret = f" -> {_ann(sig.return_annotation)}"
    return f"{name}({', '.join(parts)}){ret}"


def _from_source(path):
    """(signature, docstring) of a function that a decorator (such as
    mx.custom_function) has wrapped beyond inspection, read from its source."""
    import ast
    mod_path, name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    tree = ast.parse(inspect.getsource(mod))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = ast.unparse(node.args)
            ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
            return f"{name}({args}){ret}", ast.get_docstring(node)
    return None, None


def entry(path):
    obj = resolve(path)
    if type(obj).__name__ == "custom_function":
        sig, doc = _from_source(path)
        lines = [f"### `{path.removeprefix('elome.')}`", ""]
        if sig:
            lines += ["```python", sig, "```", ""]
        if doc:
            lines += ["```text", doc.strip(), "```", ""]
        return lines
    short = path.removeprefix("elome.")
    head = path.split(".")[-1]
    is_value = not callable(obj) or (not inspect.isclass(obj) and not
                                     inspect.isroutine(obj) and
                                     not hasattr(obj, "__wrapped__") and
                                     type(obj).__module__.startswith("elome"))
    lines = [f"### `{short}`", ""]
    if is_value and not inspect.isclass(obj):
        lines += ["```python", f"{head} = {obj!r}", "```", ""]
        doc = None
    else:
        sig = signature(head, obj)
        if sig:
            lines += ["```python", sig, "```", ""]
        doc = inspect.getdoc(getattr(obj, "__wrapped__", obj))
        if doc and inspect.isclass(obj) and doc == inspect.getdoc(object):
            doc = None
    if doc:
        lines += ["```text", textwrap.dedent(doc).strip(), "```", ""]
    return lines


def main():
    out = ["% GENERATED by docs/make_api.py -- edit the docstrings, then rerun it.",
           "", "# API reference", "",
           "Everything public, grouped by task. A name shown without a module "
           "(such as `flux_dev_from_tau`) is importable from `elome` directly; "
           "the others give their module (`anvil.make_transit_target` is "
           "`elome.anvil.make_transit_target`).",
           ""]
    for title, intro, names in SECTIONS:
        out += [f"## {title}", ""]
        if intro:
            out += [intro, ""]
        for n in names:
            out += entry(n)
    OUT.write_text("\n".join(out).rstrip() + "\n")
    print(f"wrote {OUT} ({sum(len(s[2]) for s in SECTIONS)} entries)")


if __name__ == "__main__":
    main()
