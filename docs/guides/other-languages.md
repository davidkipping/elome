# Calling Elómë from other languages

Elómë is a Python package, so another language reaches it through a
bridge to Python. The light curve is computed by the same code either
way: the bridge only carries arrays across. This page shows Julia; the
same pattern works from any language with a Python bridge.

Every crossing into Python costs a little, so the advice in
{doc}`batching` applies with more force: hand each call a whole light
curve, or many parameter sets at once through `light_curves`, never one
point at a time.

## Julia

[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) imports Elómë
directly. Pointed at a Python environment with Elómë installed, it uses
that environment, MLX and all:

```bash
export JULIA_CONDAPKG_BACKEND=Null              # use an existing Python
export JULIA_PYTHONCALL_EXE=/path/to/venv/bin/python
julia -e 'using Pkg; Pkg.add("PythonCall")'
```

### A light curve

```julia
using PythonCall
el = pyimport("elome")
np = pyimport("numpy")

t = collect(range(-0.1, 0.1, length=1001))
p = el.TransitParams()
p.t0 = 0.0; p.per = 3.4525; p.rp = 0.1153; p.a = 8.84; p.inc = 88.0
p.ecc = 0.0; p.w = 90.0
p.limb_dark = "quadratic"; p.u = pylist([0.4, 0.25])

m = el.TransitModel(p, np.asarray(t))
flux = pyconvert(Vector{Float64}, m.light_curve(p))
```

The result is bit for bit the light curve Python computes from the
same times.

### What to convert

- **Arrays going in**: wrap a Julia array in `np.asarray(...)` (or
  `mx.array(np.asarray(...))` for the sampler API). NumPy shares the
  Julia array's memory, so this is cheap.
- **Arrays coming out**: `pyconvert(Vector{Float64}, x)` or
  `pyconvert(Matrix{Float64}, x)` copies into a Julia array, and
  `PyArray(x)` wraps a NumPy result without copying. Convert an
  `mx.array` with `np.asarray(x)` first.
- **Limb-darkening coefficients**: pass `p.u` as `pylist([...])` or a
  NumPy array.
- **Scalars** (`p.rp = 0.1153`) need no conversion.

### Many parameter sets, and float32 on the GPU

```julia
mx = pyimport("mlx.core")
n = 64
pa = el.TransitParams()
pa.t0 = 0.0; pa.per = 3.4525; pa.a = 8.84; pa.ecc = 0.0; pa.w = 90.0
pa.limb_dark = "quadratic"; pa.u = pylist([0.4, 0.25])
pa.rp = np.asarray(collect(range(0.08, 0.14, length=n)))
pa.inc = np.asarray(fill(88.0, n))
F = pyconvert(Matrix{Float64}, m.light_curves(pa))     # 64 x 1001

m32 = el.TransitModel(p, np.asarray(t), dtype=mx.float32)   # the Metal kernels
```

The sampler API (`el.flux_dev_from_tau`) is called the same way, with
each per-chain parameter as an `mx.array`.

### Gradients

MLX differentiates a function written in Julia, as long as its
arguments and result are `mx.array`s. Float64 runs on MLX's CPU stream,
as it does in Python (`with mx.stream(mx.cpu):`):

```julia
function loss(rp, inc)
    q = el.TransitParams()
    q.t0 = 0.0; q.per = 3.4525; q.a = 8.84; q.ecc = 0.0; q.w = 90.0
    q.limb_dark = "quadratic"; q.u = pylist([0.4, 0.25])
    q.rp = rp; q.inc = inc
    return mx.sum(mx.square(m.light_curve_mx(q) - 1.0))
end

g = pywith(mx.stream(mx.cpu)) do _
    mx.grad(pyfunc(loss), argnums=pytuple((0, 1)))(
        mx.array(0.1153, dtype=mx.float64), mx.array(88.0, dtype=mx.float64))
end
d_rp = pyconvert(Float64, g[0].item())
```

## Other languages

The same pattern should work from R through
[reticulate](https://rstudio.github.io/reticulate/), from IDL through the
IDL–Python bridge (IDL 8.5 and later), and from C or C++ by embedding
the Python interpreter. These routes have not been tested. The float32
kernels need a Mac with Apple Silicon whatever the calling language; the
float64 path runs wherever MLX does.
