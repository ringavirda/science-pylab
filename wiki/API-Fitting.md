# API: batch fitting

The core batch fitters run on [the image](Methods-Image) of the data: a
fixed-size statistic of a basis projection, additive over sample sets. All
return a
[`FittingResult`](API-Types) unless noted. Conceptual background:
[../guides/methods-explained.md](Guides-Methods-Explained); proofs:
[../methods/](Methods).

> **pandas (optional).** `data_x` / `data_y` may be a pandas `Series` or a
> single-column `DataFrame` (a multi-column `DataFrame` raises); values are
> coerced to 1-D floats. `FittingResult.predict(x)` returns a `Series` aligned to
> `x`'s index when `x` is a `Series`. pandas is an optional dependency -- dtfit
> works without it, and an ndarray/list input is unaffected.

- [`fit`](#fit) -- nonlinear least squares on an `Original` or an `Image`
- [`Original`](#original) -- the sampled signal
- [`Image`](#image) -- the basis projection of an `Original`
- [`order_for`](#order_for), [`coverage`](#coverage) -- picking and checking the order
- [`fit_lsi`](#fit_lsi) -- `fit` in the Legendre basis
- [`fit_eac`](#fit_eac) -- `fit` in the block basis
- [`ensemble_fit`](#ensemble_fit) -- overlapping-window robust ensemble (outliers)
- [`fit_dsb`](#fit_dsb) -- Differential Spectra Balance (symbolic reference)
- [`find_degree`](#find_degree) -- polynomial degree selection (DSB support)
- [`fft_frequency_seed`](#fft_frequency_seed) -- frequency seed for oscillatory fits

---

<a name="fit"></a>
## `fit`

```python
fit(model, data, var=None, *, basis="legendre", order=None, p0=None,
    bounds=None, sigma=None, absolute_sigma=False, robust=False,
    oscillatory=False, freq_param=None, param_names=None,
    solver_options=None, random_state=0) -> FittingResult
```

Fit `model` to `data` by nonlinear least squares restricted to the span of a
basis, on the data's image. `fit_lsi` and `fit_eac` are `fit` in the Legendre
and block bases respectively -- presets, not separate methods.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `model` | str \| sympy.Expr \| callable | -- | the model, in any of three equivalent forms (resolved by [`resolve_model`](#also-exported-from-dtfitmethods)): a SymPy-expression **string** `"a0 + a1*exp(a2*x)"`, a `sympy.Expr`, or a plain Python **callable** `f(x, *params)`. A symbolic model lays its parameters out **sorted by name**; a callable follows **signature order** (the parameters after the leading `x`) |
| `data` | `Original` \| `Image` | -- | an `Original` is imaged first in `basis` at `order`; an `Image` is used as given |
| `var` | str \| None | `None` | main variable name, required for a symbolic model; a label only for a callable |
| `basis` | str \| Basis | `"legendre"` | `"legendre"`, `"block"`, a `Basis` instance, or `"auto"`: the candidates are the Legendre basis with the oscillatory recipe when `oscillatory` or `freq_param` is given or the detrended spectral peak share exceeds 0.3, then the Legendre basis at its default order, then the block basis at its default order; the candidate with the lowest unweighted sample RSS wins, a later candidate only by more than 0.1 percent. Needs an `Original` -- rejected with an `Image` (`TypeError`), which does not carry the samples the routing needs |
| `order` | int \| None | `None` | basis order (polynomial degree for Legendre, window count for block). Omitted with an `Original`, it defaults to [`order_for`](#order_for) at `p0` (`osc_order` too, and the larger taken, when `oscillatory`), floored at `n_params - 1` and capped at `n_obs - 2`; for the block basis, `4 * n_params`. A `Basis` instance sets its own order; passing `order` with one that disagrees raises |
| `p0` | array \| dict \| None | `None` | initial guess (defaults to ones): positional in canonical parameter order, or a `{name: value}` dict |
| `bounds` | list[(lo, hi)] \| dict \| (lo, hi) \| None | `None` | per-parameter bounds: a pair list in canonical order, a partial `{name: (lo, hi)}` dict, or a scipy-style `(lo, hi)` 2-tuple |
| `sigma` | array \| None | `None` | per-sample standard deviations for an `Original`; builds the weights `w = 1/sigma**2`. Raises `TypeError` with an `Image` and `ValueError` if the `Original` already carries weights |
| `absolute_sigma` | bool | `False` | if `False` (default) the covariance is scaled by the residual variance `rss / (n - p)`; if `True` it is not, as in `scipy.optimize.curve_fit` with true absolute uncertainties |
| `robust` | bool | `False` | Huber-reweight the image when building it from an `Original`; raises `TypeError` with an `Image` (already built) |
| `oscillatory` | bool | `False` | declares the model has a dominant frequency: raises the default order to `osc_order` when that exceeds `order_for`; implied by `freq_param` and by `basis="auto"` routing to a cyclic signal |
| `freq_param` | str \| None | `None` | name of the frequency parameter to seed from the FFT peak of the data ([`fft_frequency_seed`](#fft_frequency_seed)) before the solve; also sets `oscillatory=True`. Overwrites `p0` for that parameter. With an `Image` the peak is read from `Image.reconstruct` on its own grid |
| `param_names` | list[str] \| None | `None` | parameter names for a callable model, in signature order; introspected from the signature when omitted, and cross-checked for a symbolic model |
| `solver_options` | dict \| None | `None` | forwarded to the least-squares stage: `xtol`, `ftol`, `gtol` and `max_nfev` go to `scipy.optimize.least_squares`; `ftol` and `gtol` also go to the polishing `scipy.optimize.minimize` call (`max_nfev` becomes its `maxfun`) when the bounded global stage runs |
| `random_state` | int \| None | `0` | seed for the bounded global stage's differential evolution; `None` makes that stage nondeterministic between calls; unused when the fit is unbounded or the local solve is not poor |

**Notes**

- `model` may be a plain Python callable `f(x, *params)`: a symbolic model
  differentiates exactly (`sympy.diff`) for its parameter sensitivities; a
  callable uses a forward-difference Jacobian instead.
- Above a coverage of 0.02 ([`coverage`](#coverage)) at the fitted order, a
  `UserWarning` says the image's truncation loses information about the
  model's parameter sensitivities -- raise `order`.
- Levenberg-Marquardt runs unbounded; trust-region with finite `bounds`; a
  differential-evolution stage only runs when every bound is finite and the
  local solve fails, has a non-finite cost, or explains less than half the
  weighted total sum of squares, and warns (`UserWarning`) that it did.
- `rss_source` on the result records where `rss` came from: `"samples"` for an
  `Original`, `"image"` for an `Image` (the identity `sumsq - S^T G^+ S + (S -
  S_f)^T G^+ (S - S_f)`, exact when the model lies in the span).
- A model whose value is not finite at a sample raises `ValueError` at `p0`
  and, once the solve is under way, is instead scored with an overflow cost so
  the optimizer can step away from it; a sensitivity that is not finite at
  isolated samples (an exponent's derivative at `x = 0`, say) is taken as
  zero there, its analytic limit.

**Returns** a [`FittingResult`](API-Types) with `coeffs`, `cov`, `converged`,
`message`, `x_range`, `n_obs`, `nfev`, `rss`, `tss`, `cost`, and `rss_source`,
`image_order`, `basis_name` recording where `rss` came from and the order and
basis of the image the fit ran on. `cov` is `None` when the degrees of freedom
are exhausted (`n_obs <= len(names)`); a parameter with a component in a null
direction of the Jacobian gets an `inf` diagonal entry and `nan` off-diagonal
entries rather than a spuriously small variance. `n_obs`, `rss` and `tss`
drive the [`rsquared`, `aic` and `bic`](API-Types#fit-quality-diagnostics-v03)
properties.

**Raises**

- `TypeError`: `robust=True`, `sigma` given, or `basis="auto"` with an
  `Image`; `data` is neither an `Original` nor an `Image`.
- `ValueError`: `freq_param` names no parameter of the model; the image has
  fewer coefficients than parameters; the model is not finite at `p0`;
  `sigma` given for an `Original` that already carries weights; a malformed
  `p0` or `bounds`.
- `RuntimeError`: the model has no free parameters; every candidate basis
  fails when `basis="auto"`.

**Example**

```python
from dtfit import Original, fit

orig = Original(x, y)
res = fit("a0 + a1*exp(a2*x)", orig, "x")
print(res.params)
```

Fitting an already-built `Image` skips the imaging step:

```python
img = orig.image("legendre", order=8)
res2 = fit("a0 + a1*exp(a2*x)", img, "x")
print(res2.params)
```

---

<a name="original"></a>
## `Original`

```python
Original(x, y, w=None, *, sigma=None, domain=None, nan_policy="raise")
```

A sampled signal `(x, y)` with per-sample weights on a domain -- the object
every fitter starts from before imaging.

Positions are sorted non-decreasing; ties are allowed, but the samples must
span an interval. `y` and the weights follow the same sort. `w` is the
inverse-variance weight of each sample, ones unless `w` or `sigma` (a
per-sample standard deviation, `w = 1/sigma**2`) is given. `domain` defaults
to `(x[0], x[-1])` and must contain every position. `nan_policy` is `"raise"`
or `"omit"`; omission drops a pair when `x`, `y` or its weight is non-finite.

**Attributes**

| attribute | meaning |
|---|---|
| `x`, `y`, `w` | float arrays of equal length `n` |
| `domain` | `(x0, x1)` |
| `grid` | the `Grid` descriptor of `x` |
| `weighted` | whether any weight differs from one |

**Methods**

- **`image(basis="legendre", order=None, *, robust=False) -> Image`** -- the
  projection of this `Original` onto a basis; see [`Image.of`](#image). The
  robust IRLS uses the Huber constant c = 1.345.
- **`fit(model, var=None, **kwargs) -> FittingResult`** -- fit a model to this
  `Original`; forwards to [`fit`](#fit).
- **`residuals(model, params, var=None) -> ndarray`** -- `y - f(x; params)`
  for a model as in [`fit`](#fit).
- **`window(i0, i1) -> Original`** -- the samples `i0:i1` as an `Original` on
  their own span; re-validates through the constructor, so `i1 - i0` must
  leave at least two samples.
- **`n`** (property) -- the sample count.

**Example**

```python
from dtfit import Original

orig = Original(x, y)
res = orig.fit("a0 + a1*exp(a2*x)", "x")
print(res.params)
print(orig.window(0, 100).n)
```

---

<a name="image"></a>
## `Image`

```python
Image(basis, domain, S, G, n, sumsq, sumy, wsum, grid, w=None, robust=False) -> None
```

The discrete image of a signal in a basis on a domain: `S = Phi^T (w y)` are
the weighted projections, `G = Phi^T diag(w) Phi` the Gram matrix of the basis
on the sample grid; with `n`, `sumsq = sum(w y^2)`, `sumy = sum(w y)` and
`wsum = sum(w)` they are the sufficient statistic of the linear model in that
basis. `w` is the per-sample weights, stored when any weight differs from one
or the image is robust; else `None`.

Images with the same basis, order and domain merge by adding their sums
whatever their sample sets ([`merge`](#imagemerge)); a Legendre image is
nested ([`truncate`](#imagetruncate)); the least-squares coefficients `beta =
G^+ S` are derived, never stored.

**Attributes**

| attribute | meaning |
|---|---|
| `basis`, `domain`, `grid` | the basis, `(x0, x1)` and sample-grid descriptor |
| `S`, `G` | the projections (length `n_coef`) and the Gram matrix (`n_coef x n_coef`) |
| `n`, `sumsq`, `sumy`, `wsum` | sample count, `sum(w y^2)`, `sum(w y)`, `sum(w)` |
| `w` | per-sample weights, or `None` |
| `order`, `n_coef` | (properties) the basis order and coefficient count |
| `weighted` | (property) `True` when the image carries a weight vector |

**Methods**

<a name="imagemerge"></a>
- **`merge(other) -> Image`** -- the image of the two signals' samples
  pooled, whatever their sample sets; requires the same basis (including
  order) and domain. The sums are additive regardless of sample order, so
  they are added; the grid is rebuilt from the sorted union of both
  position sets.
<a name="imagetruncate"></a>
- **`truncate(order) -> Image`** -- the image at a lower order; exact for
  nested bases (Legendre).
- **`transfer(domain, order=None) -> Image`** -- this image expressed in the
  same basis on a coarser `domain` that contains this one, at `order`
  (default this order). Exact for the Legendre basis when `order` is at most
  this order; for the block basis every fine window must lie inside one
  coarse window.
- **`to_dict() -> dict`** / **`from_dict(d) -> Image`** -- round-trip to
  plain, JSON-friendly data.
- **`beta`** (cached property) -- the least-squares coefficients, the
  minimum-norm solution of `G beta = S` through the Hermitian pseudo-inverse.
- **`fit(model, var=None, **kwargs) -> FittingResult`** -- fit a model to this
  `Image`; forwards to [`fit`](#fit).
- **`phi() -> ndarray`** -- the basis evaluated on the image's grid, `(n,
  n_coef)`, weights not applied.
- **`reconstruct(x) -> ndarray`** -- the least-squares reconstruction of the
  signal at `x`.
- **`simulate(sigma, rng=None) -> (x, y)`** -- the reconstruction on the
  image's grid plus Gaussian noise of std `sigma`.
- **`of(original, basis="legendre", order=None, *, robust=False) -> Image`**
  (classmethod) -- the image of an `Original`; the robust IRLS uses the Huber
  constant c = 1.345.
- **`of_model(model, params, grid, basis="legendre", order=None, *,
  var=None, domain=None, w=None) -> Image`** (classmethod) -- the image the
  model `f(x; params)` would have on `grid`.
- Two images compare equal (`==`) when their basis, domain, grid, sample
  count, robustness, sums and weights all match.

**Example**

Merging two halves of a domain and fitting the merged image:

```python
from dtfit import Original

domain = (float(x[0]), float(x[-1]))
half = x.size // 2
img1 = Original(x[:half], y[:half], domain=domain).image("legendre", order=6)
img2 = Original(x[half:], y[half:], domain=domain).image("legendre", order=6)
merged = img1.merge(img2)
res = merged.fit("a0 + a1*exp(a2*x)", "x")
print(res.params)
```

---

<a name="order_for"></a>
## `order_for`

```python
order_for(model, params, domain, *, var=None, tol=0.02, max_order=64,
          param_names=None) -> int
```

The smallest Legendre order at which every parameter sensitivity `df/dtheta`
of `model` at `params` is represented on `domain` to relative L2 error `tol`,
floored at `n_params - 1` and capped at `max_order`. Measured against the
model catalog, `tol=0.02` is at or above the order at which the projected fit
reaches NLLS efficiency for every family. Raises `RuntimeError` when the model
has no free parameters.

<a name="coverage"></a>
## `coverage`

```python
coverage(model, params, image, *, var=None, param_names=None) -> float
```

The largest relative truncation error of `model`'s parameter sensitivities at
`image`'s order, measured against a reference expansion of `max(2K, 64)`
modes. Above `0.02` the image loses information about `params`: the order is
too low to identify the model's sensitivities from this image, and [`fit`](#fit)
warns. `0.0` for a non-Legendre basis, which this measure does not apply to.

```python
from dtfit import Original, order_for
from dtfit.image import coverage

params = [0.5, 2.0, 0.5]
domain = (float(x[0]), float(x[-1]))
k = order_for("a0 + a1*exp(a2*x)", params, domain, var="x")
img = Original(x, y).image("legendre", order=k)
print(k, coverage("a0 + a1*exp(a2*x)", params, img, var="x"))
```

---

<a name="fit_lsi"></a>
## `fit_lsi`

```python
fit_lsi(data_x, data_y, expr, var=None, *, k_star=None, p0=None,
        bounds=None, sigma=None, absolute_sigma=False, oscillatory=False,
        freq_param=None, random_state=0, robust=False, solver_options=None,
        nan_policy="raise", param_names=None, **legacy) -> FittingResult
```

LSI: [`fit`](#fit) in the Legendre basis. `k_star` is the order; `None` or
`"auto"` takes the default from [`order_for`](#order_for).

Builds an `Original` from `(data_x, data_y, sigma, nan_policy)` and calls
[`fit`](#fit) on it with `basis="legendre"`; every parameter below not listed
here (`model`, canonical parameter order, the covariance and coverage rules)
behaves exactly as documented there.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples. A pandas `Series` or single-column `DataFrame` is accepted |
| `expr` | str \| sympy.Expr \| callable | -- | the model; see [`fit`](#fit) |
| `var` | str \| None | `None` | main variable name, required for a symbolic model |
| `k_star` | int \| `"auto"` \| None | `None` | the Legendre order. `None` or `"auto"` both take the order-rule default ([`order_for`](#order_for), raised for `oscillatory` per `osc_order`); an int sets it explicitly and must leave at least as many coefficients as parameters |
| `p0` | array \| dict \| None | `None` | initial guess, positional or `{name: value}` |
| `bounds` | ... | `None` | per-parameter bounds; see [`fit`](#fit) |
| `sigma` | array \| None | `None` | per-sample standard deviations, the same length as `data_y`; builds the `Original`'s weights |
| `absolute_sigma` | bool | `False` | see [`fit`](#fit) |
| `oscillatory` | bool | `False` | raises the default order to `osc_order` when that exceeds `order_for`; implied by `freq_param` |
| `freq_param` | str \| None | `None` | name of the frequency parameter to seed from the data's FFT peak ([`fft_frequency_seed`](#fft_frequency_seed)) before the solve; implies `oscillatory=True` |
| `random_state` | int \| None | `0` | seed for the bounded global stage's differential evolution; `None` is nondeterministic between calls |
| `robust` | bool | `False` | Huber-reweight the image when it is built from the samples |
| `solver_options` | dict \| None | `None` | forwarded to the least-squares stage; see [`fit`](#fit) |
| `nan_policy` | str | `"raise"` | `"raise"` rejects a non-finite sample; `"omit"` drops it before fitting |
| `param_names` | list[str] \| None | `None` | parameter names for a callable model, in signature order; cross-checked for a symbolic model |
| `**legacy` | -- | -- | retired keywords accepted for source compatibility and ignored: `filter_data`, `alpha`, `huber_c`. Any other keyword raises `TypeError` |

**Raises**

- `TypeError`: an unrecognized keyword in `**legacy`.
- `ValueError`: `freq_param` names no parameter of the model; the image has
  fewer coefficients than parameters; the model is not finite at `p0`; a
  malformed `p0`, `bounds` or `sigma`; multivariate `data_x` or `data_y`.
- `RuntimeError`: the model has no free parameters.

**Warns**

- `DeprecationWarning`: a recognized legacy keyword was passed.
- `UserWarning`: the image's coverage of the model's sensitivities at `p0` is
  poor, or the order is too low to identify the model; see [`fit`](#fit).

**Example**

```python
from dtfit import fit_lsi

res = fit_lsi(x, y, "A*sin(w*x + p)", "x", freq_param="w")   # oscillatory recipe
print({k: round(v, 3) for k, v in res.params.items()})
```

---

<a name="fit_eac"></a>
## `fit_eac`

```python
fit_eac(data_x, data_y, expr, var=None, *, n_windows=None, p0=None,
        bounds=None, sigma=None, absolute_sigma=False, robust=False,
        loss="linear", solver_options=None, nan_policy="raise",
        param_names=None, **legacy) -> FittingResult
```

EAC: [`fit`](#fit) in the block basis with `n_windows` windows (default four
per parameter). A `loss` other than `"linear"` selects the robust image.

Builds an `Original` from `(data_x, data_y, sigma, nan_policy)` and calls
[`fit`](#fit) on it with `basis="block"`; every parameter below not listed
here behaves exactly as documented there.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples. A pandas `Series` or single-column `DataFrame` is accepted |
| `expr` | str \| sympy.Expr \| callable | -- | the model; see [`fit`](#fit) |
| `var` | str \| None | `None` | main variable name, required for a symbolic model |
| `n_windows` | int \| None | `None` | number of block windows; `None` defaults to `4 * n_params`; an int sets it explicitly and must leave at least as many coefficients as parameters |
| `p0` | array \| dict \| None | `None` | initial guess, positional or `{name: value}` |
| `bounds` | ... | `None` | per-parameter bounds; see [`fit`](#fit) |
| `sigma` | array \| None | `None` | per-sample standard deviations, the same length as `data_y`; builds the `Original`'s weights |
| `absolute_sigma` | bool | `False` | see [`fit`](#fit) |
| `robust` | bool | `False` | Huber-reweight the image when it is built from the samples. Also set to `True` when `loss` is not `"linear"` |
| `loss` | str | `"linear"` | `"linear"` (default) leaves `robust` as given; any other value selects the robust image (`robust=True`) and warns (`DeprecationWarning`) |
| `solver_options` | dict \| None | `None` | forwarded to the least-squares stage; see [`fit`](#fit) |
| `nan_policy` | str | `"raise"` | `"raise"` rejects a non-finite sample; `"omit"` drops it before fitting |
| `param_names` | list[str] \| None | `None` | parameter names for a callable model, in signature order; cross-checked for a symbolic model |
| `**legacy` | -- | -- | retired keywords accepted for source compatibility and ignored: `active_ratio`, `window_mode`, `f_scale`, `huber_c`. Any other keyword raises `TypeError` |

**Raises**

- `TypeError`: an unrecognized keyword in `**legacy`.
- `ValueError`: the image has fewer coefficients than parameters; the model
  is not finite at `p0`; a malformed `p0`, `bounds` or `sigma`; multivariate
  `data_x` or `data_y`.
- `RuntimeError`: the model has no free parameters.

**Warns**

- `DeprecationWarning`: a recognized legacy keyword was passed, or `loss` is
  not `"linear"`.

**Example**

```python
from dtfit import fit_eac

res = fit_eac(x, y, "a0 + a1*exp(a2*x)", "x", n_windows=16, robust=True)
```

---

<a name="ensemble_fit"></a>
## `ensemble_fit`

```python
ensemble_fit(data_x, data_y, expr, var, *, method="eac", n_windows=8,
             overlap=0.5, aggregate="median", p0=None, **kwargs) -> EnsembleResult
```

Fit the model on many **overlapping subwindows** and aggregate the per-window
coefficients robustly -- bagging over the time axis. The **median** of the
per-window estimates rejects windows corrupted by outliers, and the
inter-window spread is a cheap empirical uncertainty band.

**Use it for outlier-contaminated data.** On clean (Gaussian-noise) data
prefer a single whole-record fit: the ensemble trades a little accuracy there
for the outlier robustness, so it is a specialised tool, not the default
path.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `data_x`, `data_y` | array | -- | observed samples |
| `expr`, `var` | str | -- | model and main variable |
| `method` | str | `"eac"` | underlying batch fitter, `"eac"` or `"lsi"` |
| `n_windows` | int | `8` | target number of overlapping subwindows |
| `overlap` | float | `0.5` | fractional overlap between consecutive windows (`0..0.9`) |
| `aggregate` | str | `"median"` | `"median"` (robust) or `"mean"` |
| `p0` | array \| dict \| None | `None` | initial guess forwarded to each window fit |
| `**kwargs` | -- | -- | extra args forwarded to the underlying fitter (e.g. `bounds`) |

**Returns** an [`EnsembleResult`](#ensembleresult) -- a [`FittingResult`](API-Types)
(so `params`, `predict`, `stderr`, `to_dict` all work) that additionally
carries the per-window `members` and their `spread`, which also fills the
covariance.

**Example**

```python
from dtfit import ensemble_fit

res = ensemble_fit(x, y, "a*exp(-b*x)", "x", method="eac", p0=[1.0, 1.0])
print(res.params, res.spread)   # robust estimate + per-parameter spread
```

<a name="ensembleresult"></a>
### `EnsembleResult`

Subclass of [`FittingResult`](API-Types) returned by `ensemble_fit`. Extra
attributes: `spread` (per-parameter inter-window standard deviation),
`members` (`(n_windows_fitted, n_params)` raw per-window coefficients),
`n_failed` (windows whose fit raised -- a `UserWarning` is emitted whenever it
is non-zero) and `last_error` (message of the last window failure, `None` if
all fit). The spread populates the covariance diagonal, so `stderr()` returns
it and `predict(return_std=True)` reports the ensemble's uncertainty.

---

<a name="fit_dsb"></a>
## `fit_dsb`

```python
fit_dsb(coeffs_poly, expr, var, *, rank=None, p0=None) -> FittingResult
```

Symbolic **reference** method: balances the model's Maclaurin spectrum against
a polynomial's, order by order, and solves symbolically. Not for noisy
production data -- use LSI/EAC. Note it takes *polynomial coefficients*, not
raw `(x, y)`: build them with [`find_degree`](#find_degree) + `np.polyfit`.

**Arguments**

| name | type | default | meaning |
|---|---|---|---|
| `coeffs_poly` | array | -- | polynomial coefficients in **ascending** order (`coeffs_poly[k]` = coefficient of `var**k` = the data's order-`k` Maclaurin coefficient). `np.polyfit` returns descending -- reverse it with `[::-1]` |
| `expr`, `var` | str | -- | model and main variable |
| `rank` | int \| None | `None` | number of balance equations (Maclaurin orders); default uses all available polynomial coefficients |
| `p0` | array \| None | `None` | initial guess for the numeric refinement/fallback |

Raises a `ValueError` if the polynomial carries fewer coefficients than the
model has parameters (the balance would be underdefined -- fit a
higher-degree polynomial). The symbolic path accepts roots with an
exactly-zero component (a model whose true parameter is 0 resolves
symbolically); only degenerate all-zero or complex roots fall through to the
numeric refinement.

**Example**

```python
from dtfit import fit_dsb, find_degree
import numpy as np

deg = find_degree(x, y)              # BIC-selected degree
pc = np.polyfit(x, y, deg)[::-1]     # ascending = the data's Maclaurin spectrum
res = fit_dsb(pc, "a*exp(b*x)", "x")
```

---

<a name="find_degree"></a>
## `find_degree`

```python
find_degree(data_x, data_y, method="bic", max_degree=12) -> int
```

Select a polynomial degree for `(data_x, data_y)` by information criterion
(the DSB pre-fit support primitive). Returns the degree in `0..max_degree`
minimizing `"bic"` (default) or `"aic"` -- a parsimony vs fit trade-off. Warns
(via the logger) if it hits `max_degree`.

---

<a name="fft_frequency_seed"></a>
## `fft_frequency_seed`

```python
fft_frequency_seed(x, y) -> float
```

Dominant **angular** frequency of `y` over the grid `x` -- the peak of the
mean-removed real FFT, with the DC bin ignored, returned as `2*pi*f`. The
samples are interpolated onto a uniform grid first (an identity when `x`
already is one). This is the seed [`fit_lsi`](#fit_lsi)'s oscillatory recipe
uses for `freq_param`; a sinusoid's frequency can't be recovered without it.

```python
from dtfit import fft_frequency_seed
w0 = fft_frequency_seed(x, y)   # ~= angular frequency of the dominant cycle
```

---

### Also exported from `dtfit.methods`

`model_params(f_sym, t)` and `taylor_coeffs(f_sym, t, order)` are the symbolic
helpers the scheme is built on (free-parameter extraction and Maclaurin
coefficients). They're available via `from dtfit.methods import model_params,
taylor_coeffs` for advanced/extension use; most users won't need them.

`resolve_model(model, var=None, *, param_names=None) -> ModelSpec` is the
public model-input resolver behind every fitter's `model`/`expr` argument. It
accepts a SymPy-expression **string**, a `sympy.Expr`, or a plain Python
**callable** `f(x, *params)`, and returns a `ModelSpec` exposing the canonical
parameter order (`.names` -- sorted-by-name for a symbolic model, **signature
order** for a callable), the numeric evaluator (`.eval`), the parameter
sensitivities (`.param_derivs`), and `.is_symbolic` / `.expr` / `.var`. It is
what makes a callable model interchangeable with an expression string across
[`fit`](#fit) / [`fit_lsi`](#fit_lsi) / [`fit_eac`](#fit_eac) /
[`auto_estimate`](API-Auto#auto_estimate) / [`NonlineRegressor`](API-Estimator)
/ [`Model`](API-Models#model) and the streaming filters. Import both via `from
dtfit.methods import resolve_model, ModelSpec`.
