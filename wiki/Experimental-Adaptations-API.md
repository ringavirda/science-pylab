# API: experimental adaptations

Full signatures and usage for the structural adaptations still in trial in
`dtfit-experimental`, plus the array-backend helpers. These are **experimental
APIs and may change** until promoted. For the *ideas* behind them see
[README.md](Experimental); for where they sit in the lineage see
[../guides/lineage-and-variants.md](Guides-Lineage-and-Variants).

> #3, the overlapping-window ensemble, is retired; the robust image covers it
> (see [Methods-EAC](Methods-EAC)).

```python
from dtfit_experimental import (
    fit_lsi_basis,                 # #2 pluggable orthogonal basis
    fit_joint, JointResult,        # #4 joint shared-parameter multi-channel fit
    boosted_fit, BoostedModel,     # #5 stage-wise residual boosting
    InformationFilter,             # inverse-covariance (info-form) fusion primitive
    available_backends, resolve_backend, Backend,   # array backends
)
```

- [`fit_lsi_basis`](#fit_lsi_basis) -- #2 pluggable basis
- [`fit_joint`](#fit_joint) / [`JointResult`](#jointresult) -- #4 joint fit
- [`boosted_fit`](#boosted_fit) / [`BoostedModel`](#boostedmodel) -- #5 boosting
- [`InformationFilter`](#informationfilter) -- inverse-covariance fusion primitive
- [array backends](#backends) -- `available_backends`, `resolve_backend`, `Backend`

---

<a name="fit_lsi_basis"></a>
## `fit_lsi_basis` -- #2 pluggable orthogonal basis

**What it is.** LSI matches fingerprints on the Legendre basis, which is general
but not always *efficient*: a periodic wiggle needs many high Legendre orders to
express, whereas a **Fourier** basis captures it in a couple of harmonics, and a
pure decay is natural in a **Laguerre** basis. This keeps LSI's exact criterion
(the diagonal-weighted spectral match) but lets you pick the basis to match the
signal -- fewer coefficients, better conditioning.

```python
fit_lsi_basis(data_x, data_y, expr, var, *,
              basis="fourier", order=5, filter_data=None,
              period=None, bounds=None, p0=None) -> FittingResult
```

| arg | default | meaning |
|---|---|---|
| `data_x`, `data_y` | -- | observed samples |
| `expr`, `var` | -- | model expression and main variable |
| `basis` | `"fourier"` | `"legendre"` \| `"chebyshev"` \| `"fourier"` \| `"laguerre"` |
| `order` | `5` | spectral order (number of harmonics `K` for Fourier) |
| `filter_data` | `None` | Savitzky-Golay pre-smoothing before projection. `None` (the default) picks per basis: **off** for `"fourier"` (smoothing would erase the very cycle a Fourier basis targets) and **on** for every other basis. Pass an explicit bool to override. |
| `period` | `None` | fundamental period for Fourier (defaults to the domain length) |
| `bounds` | `None` | per-parameter `(min, max)` bounds; **supplying them switches on `solve_spectral`'s global (differential-evolution) search before the local refine** -- needed for a multimodal fit such as a free frequency |
| `p0` | `None` | initial guess |

Returns a [`FittingResult`](API-Types) (coefficients, callable model,
covariance).

```python
from dtfit_experimental import fit_lsi_basis
res = fit_lsi_basis(x, y, "A*sin(w*x + p)", "x", basis="fourier", order=4)
```

---

<a name="fit_joint"></a>
## `fit_joint` -- #4 joint shared-parameter fit

**What it is.** Several channels often share structure -- a common frequency across
x/y/z axes, a common growth rate across regions, a common time constant across a
plant's outputs. Fitting them independently wastes that coupling. `fit_joint`
stacks **all** channels' EAC area equations into one system with **shared**
parameters (estimated jointly from every channel) and **per-channel private**
parameters, solved in one pass. More equations per shared unknown means you
observe it better than any single channel could.

```python
fit_joint(channels, expr, var, shared, *,
          n_windows=6, active_ratio=1.0,
          p0_shared=None, p0_private=None,
          bounds_shared=None, bounds_private=None) -> JointResult
```

| arg | default | meaning |
|---|---|---|
| `channels` | -- | list of `(x, y)` arrays, one per channel |
| `expr`, `var` | -- | model expression and main variable |
| `shared` | -- | names of parameters tied across all channels; the rest are per-channel |
| `n_windows` | `6` | area windows per channel |
| `active_ratio` | `1.0` | leading fraction of each channel used for windows |
| `p0_shared` / `p0_private` | `None` | initial guesses (default ones) |
| `bounds_shared` / `bounds_private` | `None` | per-parameter bounds; **when given, a global search (differential evolution) precedes the local refine** -- needed for a multimodal shared parameter such as a free frequency |

<a name="jointresult"></a>
### `JointResult`

| attribute | meaning |
|---|---|
| `shared` | `{name: value}` shared-parameter estimates |
| `private` | list of `{name: value}` per-channel private estimates |
| `expr`, `var` | the model |
| `predict(channel, x)` | evaluate channel `channel`'s fitted model at `x` |

```python
from dtfit_experimental import fit_joint
ax, ay, az = (A * np.sin(1.7 * t + p) + rng.normal(0, 0.05, t.size)
              for A, p in ((1.0, 0.0), (0.7, 0.4), (0.4, 1.1)))
jr = fit_joint([(t, ax), (t, ay), (t, az)], "A*sin(w*t + p)", "t",
               shared=["w"])          # one frequency, per-axis amplitude/phase
print(jr.shared["w"], jr.private[0])
```

---

<a name="boosted_fit"></a>
## `boosted_fit` -- #5 stage-wise residual boosting

**What it is.** One parametric form may not capture both a trend and a cycle.
Boosting stages the methods: fit stage 1 to the data, subtract its prediction, fit
stage 2 to the residual, and so on. The composite is the **sum** of the stages --
e.g. an LSI trend plus an EAC-fitted oscillatory residual -- more expressive than
either alone, while each stage stays a cheap, well-posed fit. (It works because
the fingerprint transform is linear, so the fingerprint of a sum is the sum of
fingerprints.)

```python
boosted_fit(data_x, data_y, stages) -> BoostedModel
```

| arg | meaning |
|---|---|
| `data_x`, `data_y` | observed samples |
| `stages` | ordered list of stage specs, each a dict with `expr`, `var`, `method` (`"lsi"`/`"eac"`), plus any extra fitter kwargs (`p0`, `bounds`, ...) |

<a name="boostedmodel"></a>
### `BoostedModel`

| attribute | meaning |
|---|---|
| `stage_models` | the per-stage fitted callables |
| `stage_specs` | per-stage `{expr, var, method, coeffs}` (the fitted record) |
| `predict(x)` | sum of all staged contributions |

```python
from dtfit_experimental import boosted_fit
bm = boosted_fit(x, y, [
    {"expr": "a0 + a1*x", "var": "x", "method": "lsi"},      # trend
    {"expr": "A*sin(w*x + p)", "var": "x", "method": "eac"}, # cycle on the residual
])
y_hat = bm.predict(x)
```

---

<a name="informationfilter"></a>
## `InformationFilter` -- inverse-covariance fusion primitive

**What it is.** The covariance-form Kalman update the stable `EACFilter` /
`LSIFilter` run maintains `P` and inverts an `m x m` innovation covariance each
step. The **information form** maintains the inverse `Y = P^-1` (the *information
matrix*) and `yv = P^-1 p` (the *information vector*) instead, which flips two
properties that matter for sensor fusion: the measurement update is **purely
additive** (`Y += H^T R^-1 H`, `yv += H^T R^-1 z` -- no inverse to absorb a
measurement), so independent estimators **fuse by adding information** (exact,
associative, order-independent); and the readout inverts only the small `n x n`
state matrix.

This is a **standalone experimental primitive**, a recursive *linear*-Gaussian
estimator (RLS in information form, with an optional forgetting factor). It is
**not** a linearization of the nonlinear EAC/LSI filters and **shares no code**
with them -- they run the covariance form directly. It is exercised by no domain
study and has **not cleared the >=2-domain promotion gate**, so it stays in
`dtfit-experimental` until a sensor-fusion / embedded domain uses it.

```python
InformationFilter(n_params, *, prior_precision=1e-6, forgetting=1.0)
```

| arg | default | meaning |
|---|---|---|
| `n_params` | -- | state dimension `n` |
| `prior_precision` | `1e-6` | diagonal of the initial information matrix `Y0 = prior_precision * I` (a weak prior; `0` is uninformative but leaves `Y` singular until enough measurements arrive) |
| `forgetting` | `1.0` | exponential forgetting in `(0, 1]` (`1` = none); each step down-weights the accumulated information before adding the new measurement, so it tracks slowly-varying parameters |

| method / attribute | meaning |
|---|---|
| `partial_fit(h, z, r=1.0)` | absorb one measurement `z = h . theta + noise` (variance `r`); `h` is a length-`n` row for scalar `z`, or an `(m, n)` matrix for a vector `z` of length `m` (then `r` may be scalar or length-`m`). Additive, no inverse. Returns `self` |
| `fuse(other)` | add another estimator's information into this one **in place** (exact, associative, commutative); the shared prior is subtracted once so it is not double-counted. Assumes both used `forgetting == 1`. Returns `self` |
| `theta_` | current estimate, from the single small solve `Y theta = yv` (alias `p`) |
| `cov_` | parameter covariance `P = Y^-1` (alias `P`) |
| `n_updates` | measurements absorbed so far |

```python
import numpy as np
from dtfit_experimental import InformationFilter

# A line z = a0 + a1*x, streamed as rows h = [1, x], split across two "sensors"
# that are then fused into the exact state one estimator seeing all data reaches.
x = np.linspace(0, 1, 200)
z = 0.5 + 2.0 * x + np.random.default_rng(0).normal(0, 0.05, x.size)
fa, fb = InformationFilter(2), InformationFilter(2)
for i in range(x.size):
    (fa if i % 2 == 0 else fb).partial_fit([1.0, x[i]], z[i], r=0.04)
fused = fa.fuse(fb)
print(np.round(fused.theta_, 3))          # ~ [0.5, 2.0]
```

---

<a name="backends"></a>
## Array backends -- `available_backends`, `resolve_backend`, `Backend`

The channel GEMM behind `ImageStream(channels=B, backend=...)` in stable
`dtfit` is a single matrix product that runs unchanged on CPU or GPU -- only
*where the arrays live* changes. `ImageStream` selects that backend by name
(`"numpy"`, `"cupy"` or `"torch"`); `available_backends()` lists the names
usable here. `resolve_backend` builds a resolved `Backend` object instead --
what the experimental batched fitters in `dtfit_experimental.scale` accept.

```python
available_backends() -> list[str]
resolve_backend(name="auto", *, dtype="float64") -> Backend
```

- **`available_backends()`** -- the backends usable here: always `"numpy"`, plus
  `"cupy"` / `"torch"` when they import *and* a GPU is present.
- **`resolve_backend(name="auto", *, dtype="float64")`** -- build a `Backend` by
  name; `"auto"` prefers a GPU when present, else NumPy.
- **`Backend`** -- a thin object that only moves arrays on/off a device
  (`asarray` / `to_host`); the projection arithmetic stays generic (`@`, `*`,
  `.T`), which is why GPU support is a backend *choice*, not a rewrite.

```python
from dtfit_experimental import available_backends
print(available_backends())          # e.g. ['numpy'] or ['numpy', 'torch']

from dtfit import ImageStream
xs = np.linspace(0, 4, 300)
Ys = np.column_stack([xs, 2 * xs, xs ** 2])
stream = ImageStream("legendre", 6, domain=(0, 4), channels=Ys.shape[1],
                      backend="numpy")
stream.update(xs, Ys)
images = stream.images()             # one running Image per channel
```

> The channel GEMM has low arithmetic intensity (it's a thin matrix product), so a
> GPU pays off only when the data is already resident on the device -- but the code
> path is identical either way. See the big-data domain study in
> [README.md](Experimental#suite).
