# Changelog

All notable changes to `dtfit` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[SemVer](https://semver.org/) with the usual 0.x caveat - minor releases may
carry breaking changes, and each one is listed explicitly under **Changed**.

## [Unreleased]

### Added

- **`dtfit.image`**: the discrete-statistic core (`Grid`, `LegendreBasis`,
  `BlockBasis`, `Original`, `Image`, `fit`, `order_for`, `coverage`,
  `osc_order`, `fft_frequency_seed`), and the top-level names `Original`,
  `Image`, `fit`, `order_for`.
- `FittingResult` gained `rss_source`, `image_order` and `basis_name`,
  recording where `rss` was computed and the image the fit ran on.
- The Monte-Carlo gates in `tests/image/test_gates.py`, run at their full
  replicate counts with `DTFIT_NIGHTLY=1`.
- `ImageStream`, the running image over a fixed domain, in block or
  channel-batch mode; `Image.transfer`, `assemble` (merging block images
  onto a coarse domain); `DriftDetector`, the shared change-detection logic
  behind the streaming filters.
- `ImageFilter`, the streaming tracker on the window image; its `result()`
  fits the current window with the batch machinery, returning a
  `FittingResult` with a calibrated covariance; `innovation_` and `nis_`
  expose the last whitened innovation and its normalized magnitude; a
  `stream` hook (an `ImageStream`) can be fed alongside the filter;
  `noise_var` sets the measurement variance directly.
- `dtfit.image.analytics`: what an image says about itself -- `noise_sigma`
  (the noise level from the tail orders), `effective_order`, `decay`
  (geometric ratio and algebraic exponent, with the goodness of fit of
  each), `test_equal` (chi-square test that two images are of the same
  signal) and `test_structure` (chi-square test that a model explains
  everything in the basis's span), each also an `Image` method; the result
  types `Decay` and `ChiSquareTest`.
- `Original.diagnostics(model, params, var=None)`: Durbin-Watson, residual
  autocorrelation and normality for a model on the samples, without a fit
  result; `dtfit.diagnostics.residual_stats` computes them from a residual
  array.
- Gates on the noise estimate (nightly: the mean over replicates within 15
  percent of the true sigma on six smooth families) and on the equality
  test's false-alarm rate (nightly: within `[0.03, 0.08]` under Gaussian,
  Student-t and Laplace noise) in `tests/image/test_gates.py`.
- `dtfit.reference`: the exact-balance ancestor, `fit_dsb` and `find_degree`,
  documented as the method the image core is derived against rather than one
  of its routes.
- `dtfit.sklearn`: `NonlineRegressor` over `fit`, with `basis` and `order` as
  estimator parameters so a `GridSearchCV` tunes the basis and the resolution
  of the image. `import dtfit` no longer imports scikit-learn.
- `Model.fit(data)` and `suggest_models(data)` accept an `Original`, an
  `Image`, or the sample positions with values. An Image seeds from its
  reconstruction over 400 points and is fitted, and ranked, as itself: the
  criteria come off the image identity, so `suggest_models` on an image gives
  the ranking its samples would.
- `dtfit.models` exports the shared input seam: `resolve_model`, `ModelSpec`,
  `result_kwargs`, `normalize_p0`, `normalize_bounds`.

### Changed

- The stochastic tier reads every estimator, gate, forecaster and generator
  from its own additive second-order image (`SecondOrderImage`), with a
  block-stream form (`SecondOrderStream`); `garch_persistence(use=...)`,
  `decompose_trend_cycle(trend_deg=...)` and the private `_stats` / `_model`
  / `_estimators` modules are gone.
- `fit_many` now lives in `dtfit.image.parallel`.
- `fit_lsi` and `fit_eac` are now presets of `fit` on the Legendre and block
  bases. The default order comes from `order_for` (Legendre) or four windows
  per parameter (block); the covariance comes from the projected Jacobian,
  with an unidentified parameter marked `inf`/`nan` rather than given a
  spuriously small variance; robustness is the robust image (`robust=True`),
  applied when the image is built.
- The bounded solve runs a differential-evolution stage only when the local
  solve fails, reaches a non-finite cost, or explains less than half the
  variance, and warns when it does.
- `basis="auto"` fits the candidate bases and returns the one with the
  lowest sample RSS.
- The golden accuracy corpus was regenerated: values now sit at the scipy
  reference on the same draw.
- `LSIFilter` and `EACFilter` are now aliases of `ImageFilter`, fixing its
  basis to Legendre and block; the measurement is the window image
  whitened by its Gram matrix; the adaptive window is the default;
  `drift_reset` defaults to `"inflate"`; `order` is the block basis's
  window count (was `n_sub`).
- `Image.simulate` takes `(n=None, sigma=None, rng=None)`: `n` positions
  over the domain instead of the image's own grid, and a `sigma` that
  defaults to `noise_sigma()`. A non-integer `n` raises `TypeError`.
- `Image.of_model` accepts `param_names` for a callable model whose
  signature cannot be introspected.
- `Model.fit`, `suggest_models` and `NonlineRegressor` take `basis` and
  `order` in place of `method`, and all three call `fit`. On the validation
  corpus at 20 noise draws and 5 percent noise, the recovery error of the
  self-seeded path measured against `scipy.optimize.curve_fit` seeded the same
  way drops from a worst case of 1.212 to 1.054: on 99 of the 100 peaked
  family-seed draws the route lands on the Legendre basis at `order_for`
  rather than the block basis at `4 * n_params` windows, which measured 1.04
  to 1.22 times the Legendre error (lorentzian 1.21, michaelis_menten 1.08,
  gaussian 1.07, hill 1.07, double_gaussian 1.05).
- `fft_frequency_seed` removes the least-squares straight line before the FFT,
  so a cycle riding on a trend is seen: on a four-cycle trend-plus-cycle
  series it reads 1.2535 rad per unit against the true 1.257, where before it
  read 0.313, the lowest non-zero bin. `osc_order` and `fit`'s `freq_param`
  seeding read the same number.
- `fit(basis="auto")` with a `freq_param` fits two candidates instead of
  three: the flag turns the oscillatory recipe on inside every candidate, so
  the plain Legendre one repeated the oscillatory one. Measured per fit: sine
  35.4 -> 26.6 ms, damped_oscillation 43.9 -> 31.2 ms, fourier_series
  61.1 -> 40.0 ms.
- The golden accuracy corpus (`tests/accuracy/golden_baseline.json`) pins the
  median over five noise draws instead of the value at one. A single draw of a
  recovery error moves by up to 8.1x (90th percentile) between disjoint seed
  blocks, against 3.8x for the median of five, so the old snapshot pinned the
  seed rather than the method. The gate costs 35.1 s against 20.0 s.
- The top-level namespace is fifteen names and three subpackages:
  `Original`, `Image`, `ImageStream`, `ImageFilter`, `fit`, `fit_lsi`,
  `fit_eac`, `LSIFilter`, `EACFilter`, `order_for`, `fit_many`,
  `suggest_models`, `auto_forecast`, `FittingResult`, `ForecastResult`, plus
  `models`, `stochastic` and `diagnostics`. Everything else is reached through
  its own module: `dtfit.models` (`Model`, `Stochastic`, `register`,
  `unregister`), `dtfit.stochastic` (`fit_stochastic`, `StochasticModel`,
  `StochasticFilter`), `dtfit.sklearn` (`NonlineRegressor`), `dtfit.reference`
  (`fit_dsb`, `find_degree`), `dtfit.image` (`FittingProblem`,
  `fft_frequency_seed`, `coverage`) and `dtfit.log` (`enable_logging`,
  `logger`).

### Removed

- `dtfit.methods._lsi` and `dtfit.methods._eac`, replaced by `dtfit.image`.
- The keywords `filter_data`, `alpha`, `huber_c`, `active_ratio`,
  `window_mode` and `f_scale` (accepted and ignored, with a
  `DeprecationWarning`) and `loss` (maps to `robust=True`, with a warning).
- `Model.fit(method="adaptive")`.
- Curvature-placed EAC windows.
- The Savitzky-Golay pre-filter, moved to `dtfit_experimental.basis_lsi`.
- `PartitionedLSI`, `PartitionedEAC`, `PartitionedBatchLSI`,
  `fit_lsi_batched` and `project_spectra`, parked in
  `dtfit_experimental.scale` until the notebooks rerun on `ImageStream`.
- The public module `dtfit.scale` itself.
- The streaming filter keywords `r`, `adapt_r`, `adapt_noise` and `n_sub`,
  and the attributes `param_cov_` and `stderr_`, superseded by `noise_var`,
  `order` and `result()`.
- `FilterBank` and `FusedChiSquareDetector`, moved to
  `dtfit_experimental.streaming`.
- `auto_estimate`. Its routing is `fit(..., basis="auto")`, which chooses by
  outcome rather than by a shape statistic; `auto_forecast` stays, in
  `dtfit.forecast`.
- `ensemble_fit` and `EnsembleResult`. On the case their documentation gave --
  the validation corpus with 4 percent of samples replaced by 8-sigma spikes,
  five families, six noise draws -- the robust image `fit_eac(robust=True)`
  recovers to a pooled median relative parameter error of 0.008 and a pooled
  mean of 0.010, against the ensemble's 0.065 and 0.089 and the plain block
  preset's 0.114 and 0.143, and costs 0.06 s against 0.71 s for the same 30
  fits. The ensemble's own claim, roughly halving the error of a plain fit,
  held; it was never measured against the robust image that replaced the
  robust loss.
- The `dtfit.methods` and `dtfit.estimators` packages. Model and parameter
  input resolution is `dtfit.models` (`resolve_model`, `normalize_p0`,
  `normalize_bounds`), the presets are `dtfit.fit_lsi` / `dtfit.fit_eac`, DSB
  is `dtfit.reference`, and the estimator is `dtfit.sklearn`.
- `NonlineRegressor`'s `method`, `k_star`, `alpha`, `filter_data`,
  `active_ratio`, `poly_degree`, `huber_c`, `loss` and `window_mode`
  arguments, and its DSB route.

## [0.4.0] - 2026-07-09

The "adoption" release: first-class (optional) pandas support, a public model
registry, a generated docs site, and a split CI. Every addition is opt-in - with
ndarray/list inputs every path stays numerically identical to v0.3 (the golden
accuracy corpus is unchanged). pandas is an **optional** dependency; dtfit
imports and all core tests pass without it.

### Added

- **Optional pandas I/O.** `fit_lsi`, `fit_eac`, `auto_estimate`, `auto_forecast`,
  `fit_stochastic`, `Model.fit`, and `NonlineRegressor` accept a pandas `Series`
  or a single-column `DataFrame` for their data (a multi-column `DataFrame`
  raises a clear `ValueError`). The pandas handling lives in a new guarded
  `dtfit._pandas` module - pandas is never hard-imported, so it stays an
  optional install.
- **pandas out where it's natural.** `FittingResult.predict(x)` and
  `NonlineRegressor.predict(X)` return a pandas `Series` aligned to the input's
  index when the input is a `Series`/single-column `DataFrame` (an
  `(Series, Series)` pair with `return_std=True`); ndarray input still returns
  ndarray, with identical values.
- **Date-indexed forecasts.** `ForecastResult` gained `.index` (the length-horizon
  *future* index continuing the input - a `DatetimeIndex` is extended by its
  inferred frequency, an integer/`RangeIndex` by its step) and `.to_series()`
  (the pandas view). `StochasticModel.forecast` returns an index-aligned `Series`
  (and three aligned `Series` for a confidence interval) when the model was fit
  on a pandas `Series`.
- **Public model registry.** `dtfit.register(name, factory)` /
  `dtfit.unregister(name)` (also `dtfit.models.register`) add a custom model
  family to the catalog so `all_models()` and `suggest_models` see it; a name
  collision raises unless `overwrite=True`. The recommender's shortlist is
  **opened** so a custom family with an unknown shape-category is never silently
  dropped from `suggest_models`.
- A generated **docs site** (`mkdocs-material` + `mkdocstrings`, a new `docs`
  optional-dependency extra), a **"dtfit vs scipy" comparison page** with
  measured numbers, a `CITATION.cff`, and a documented **versioning/deprecation
  policy** (a `DeprecationWarning` for at least one minor release before removal).
- An explicit **1-D scope boundary**: multivariate `X` (a 2-D array with more
  than one column, or a multi-column `DataFrame`) now raises a clear, early
  error at every fitting entry point - stating that dtfit's integral criteria
  are one-dimensional and pointing to same-axis `+` composition for a
  sum-of-components signal - instead of a shape error or a silent flatten. A new
  "Multivariate data" docs page covers the composition and backfitting patterns.

### Changed

- `fit_lsi`'s `var` parameter is now optional (matching `fit_eac`): a symbolic
  model still requires it, a callable model defaults it to `"x"`.
- `fit_lsi` now threads `solver_options` through its robust-IRLS inner re-solves
  (previously only the main solve honored them), matching `fit_eac`.
- CI is split into an independent `dtfit` job (path-filtered to
  `packages/dtfit/**`) and a separate job for the research packages, so the
  library has its own fast quality signal.

### Fixed

- A sliced or derived `ForecastResult` (`fc[:3]`, a reduction, a broadcast) no
  longer carries the parent's length-horizon `.index`/`.std_band`: the
  length-dependent metadata is dropped when the array's length changes, so
  `fc[:3].std_band` is no longer a silently misaligned length-horizon band and
  `fc[:3].to_series()` gives a clear error instead of a length crash. Scalar
  provenance (`.model_name`, `.result`) still carries forward.

## [0.3.0] - 2026-07-09

The "capability" release: models can be plain Python callables, fits take
per-point measurement uncertainties, and results carry their own fit-quality
diagnostics. Every addition is opt-in - with the new arguments left at their
defaults, all fits are numerically identical to v0.2 (the golden accuracy
corpus is unchanged).

### Added

- **Callable models everywhere.** `fit_lsi`, `fit_eac`, `auto_estimate`,
  `Model`, `NonlineRegressor`, and the streaming `EACFilter`/`LSIFilter` now
  accept a plain Python function `f(x, *params)` or a `sympy.Expr`, in addition
  to the expression string. A callable is resolved via the new public
  `dtfit.methods.resolve_model`; its parameters follow the callable's
  **signature order** (symbolic models keep sorted-name order). Callable models
  fit with a forward-difference Jacobian; symbolic models keep the exact
  `sympy.diff` Jacobian. `param_names=` supplies names for a callable whose
  signature cannot be introspected (an `f(x, *params)` model or a builtin).
- **Per-point weights.** `fit_lsi` and `fit_eac` take `sigma=` (per-sample
  measurement std) and `absolute_sigma=` (scipy `curve_fit` covariance
  semantics). LSI weights the empirical Legendre spectrum by `1/sigma`; EAC
  weights each window-area residual by its propagated inverse area-std. The
  contract matches `scipy.optimize.curve_fit`: `absolute_sigma=True` scales the
  standard errors with `sigma`, `False` (default) treats it as relative.
- **`sample_weight` on `NonlineRegressor.fit`** (sklearn convention),
  translated to `sigma = 1/sqrt(weight)` and forwarded to the LSI/EAC routes.
- **Fit-quality diagnostics on `FittingResult`:** `n_obs`, `rss`, `tss`, `nfev`,
  `cost`, plus `.rsquared`, `.aic`, `.bic` properties and `.residuals(x, y)` -
  reachable from the sklearn route via `result_`, and round-tripped by
  `to_dict`/`from_dict`.
- **Solver-option passthrough.** `fit_lsi`/`fit_eac` accept
  `solver_options={"xtol": ..., "max_nfev": ...}` forwarded to the underlying
  scipy solvers; the optimizer's `nfev` is recorded on the result.
- **Structured `ForecastResult`** (exported top-level) from `auto_forecast`: an
  `np.ndarray` subclass (so every existing caller keeps working) that also
  carries `.model_name` (with fallback provenance - e.g.
  `"linear (poly diverged)"`), `.result` (the underlying `FittingResult`), and
  `.std_band` (a delta-method 1-sigma prediction band when available).
- Callable-model uncertainty: `FittingResult` gained an optional `param_model`
  so `predict(return_std=True)` produces a band for a callable-only fit (finite
  difference), and `.model` works without an expression.

### Changed

- `fit_lsi`/`fit_eac` widened their `expr` parameter to
  `str | sympy.Expr | Callable`, and `var` became optional (a label only for a
  callable; defaults to `"x"`). Existing positional calls are unaffected.
- A callable-only `FittingResult` has `expr=None`: `predict`/`.model` work, but
  `to_dict()` raises (there is no expression to serialize) - as for any
  expression-less result.
- `Model.__add__` (composition) and the seed-detrend evaluator require symbolic
  operands and raise a clear `TypeError`/error for a callable model.
- Streaming `coast()`/`coast_cov()` raise `NotImplementedError` on a
  callable-backed filter (they need symbolic time-derivatives); external
  regressors remain symbolic-only.

### Fixed

- **Sigma-length contract unified across fitters.** `fit_lsi` and `fit_eac`
  now share one `_resolve_sigma`: `sigma` is the **raw** input length and the
  same non-finite rows are dropped under `nan_policy="omit"`. Previously LSI
  validated `sigma` against the post-drop count while EAC expected full length,
  so a full-length `sigma` (as `NonlineRegressor` forwards from `sample_weight`)
  fit on EAC but raised on LSI under `nan_policy="omit"`.
- `ForecastResult`'s uncertainty band is `.std_band`, not `.std`, so it no
  longer shadows `numpy.ndarray.std` - `fc.std()` and `np.std(fc)` work.
- `Model.fit(method="auto")` (the default) now forwards a callable model's
  committed `param_names` through `auto_estimate`, so an `f(x, *params)` model
  no longer crashes on re-introspection and a renamed callable keeps its names.

## [0.2.0] - 2026-07-09

The "trust" release: fitters no longer silently modify data, drop samples,
swallow errors, or overstate convergence - and parameters can finally be
addressed by name.

### Added

- **Name-keyed `p0`/`bounds`.** `fit_lsi`, `fit_eac`, `auto_estimate`,
  `Model.fit`, and `NonlineRegressor` accept `p0={"a": 1.0, ...}` (must cover
  all parameters; a `ValueError` names anything missing or unknown) and
  `bounds={"a": (0, 10)}` (partial - unnamed parameters stay unbounded). The
  normalizers are public: `dtfit.methods.normalize_p0` / `normalize_bounds`.
- **One bounds convention.** Both fitters accept the per-parameter pair list
  (canonical), the partial dict, or the scipy-style `(lo, hi)` 2-tuple;
  `lo < hi` is validated (strictly) per parameter with the parameter named in
  the error - to pin a parameter to a constant, substitute the value into the
  model expression.
- `EnsembleResult.n_failed` and `.last_error`: per-window fit failures are
  counted and surfaced (plus a `UserWarning`) instead of silently swallowed;
  `overlap` is validated to `[0, 0.9]`.
- `NonlineRegressor.result_`: the full `FittingResult` (covariance, `stderr()`,
  `confidence_intervals()`, `converged`) is now reachable from the sklearn
  route, and the engine levers (`robust`, `huber_c`, `nan_policy`, `loss`,
  `window_mode`, plus `bounds` on the EAC route) are constructor params.
- sklearn conformance suite: `parametrize_with_checks` runs in CI (38 checks
  pass; 14 documented exclusions stem from the single-feature API).
- Fitted `NonlineRegressor` instances pickle cleanly (joblib-parallel
  cross-validation works); samples are sorted by `x` before dispatch, making
  fits sample-order invariant.
- Warnings instead of silence across the package: failed `suggest_models`
  candidates, `auto_estimate` bulk-candidate failures, `auto_forecast`
  model-fallback swaps, failed `fit_stochastic` detection stages, failed
  forecaster-backtest candidates, and the short-series (`n <= 50`) forecaster
  fallback (also visible as a `" (short-series fallback)"` suffix on
  `forecaster_name`) all emit `UserWarning`s.

### Changed

- **BREAKING - `fit_lsi(filter_data=...)` now defaults to `False`** (was
  `True`): the Savitzky-Golay pre-filter is opt-in; a fitter must not silently
  smooth your data. Recommended for very noisy telemetry.
- **BREAKING - `fit_eac(active_ratio=...)` now defaults to `1.0`** (was `0.8`):
  all samples are used; the fitter no longer silently discards the trailing
  20%. `auto_estimate`'s EAC routes explicitly pin the study-tuned
  `active_ratio=0.8` recipe, so the auto pipeline's validated behavior is
  unchanged.
- **BREAKING - 2-parameter bounds disambiguation:** a bounds 2-tuple of two
  2-sequences (e.g. `([0, 0], [10, 10])`) is now read as per-parameter
  `(lo, hi)` pairs, not the scipy `(lo_array, hi_array)` form. Pass a dict, a
  pair list, or scalars to disambiguate. All other scipy-tuple inputs keep
  working.
- `NonlineRegressor` defaults aligned with the fitters: `alpha=0.0` (was a
  divergent `0.2`), `filter_data=False`, `active_ratio=1.0` - the estimator and
  the bare fitter now give the same answer for the same data.
- Honest convergence reporting: the robust-IRLS paths of `fit_lsi`/`fit_eac`
  propagate the last inner solver's actual status (message
  `"robust IRLS (<inner status>)"`) instead of hard-coding success.
- `fit_dsb` raises `ValueError` (was `RuntimeError`) for user-input errors.
- `fit_stochastic(period=...)` is documented as being in **samples**; detected
  and supplied periods are converted to `t` units internally (see Fixed).

### Fixed

- **`fit_stochastic` time-axis bug:** seasonal periods are FFT-detected in
  sample units but were applied in `t` units, so any non-unit time axis
  (seconds, years) fit the wrong seasonal frequency. Periods are now converted
  via the median spacing of `t`; `simulate()` maps sample indices onto the
  fitted axis. Results on the default `0..n-1` axis are unchanged.
- **Streaming NaN poisoning:** `EACFilter`/`LSIFilter.partial_fit` appended the
  sample before checking finiteness, so one NaN observation silently stalled
  updates for up to a full window. Non-finite samples (in `t`, `y`, or a
  regressor) are now skipped at entry with a `RuntimeWarning` and the filter
  resumes on the next good sample.
- **Partial bounds were dropped wholesale:** if a model seeder left any
  parameter unbounded, *all* bounds were discarded (a `sigma > 0` guard
  vanished silently). Mixed bounds now reach the solver - the
  differential-evolution global stage runs only on fully-finite boxes, and the
  local trust-region solve is always constrained.
- `solve_weighted_nlls` tolerates infinite/mixed bounds (previously they were
  dropped upstream or would crash the global stage).
- `Model.fit(method="eac"/"adaptive")` no longer converts the seeded pair-list
  bounds to the ambiguous scipy tuple before calling `fit_eac`.
- `PartitionedLSI.update` crashed with `AttributeError` on plain Python
  list/tuple chunks; `PartitionedBatchLSI.update` raised `IndexError` on 0-d
  input. Array-likes now accumulate identically to ndarrays.
- `fit_dsb`'s symbolic solver discarded any root containing an exactly-zero
  component, wrongly rejecting models whose true parameter is 0; only
  degenerate all-zero, complex, or incomplete roots are dropped now.
- `fit_eac` clips the (default all-ones) initial guess into the bounds box
  before solving, matching `fit_lsi` - a named bracket excluding 1.0 no longer
  crashes with scipy's "Initial guess is outside of provided bounds".
- Degenerate `lo == hi` bounds are rejected up front with the parameter named;
  previously they succeeded or crashed with an opaque scipy error depending on
  which solver path ran.
- `PartitionedLSI`/`PartitionedBatchLSI` accept 0-d scalar chunks (treated as
  one sample); previously they crashed on a boundary carry or were silently
  dropped.
- `NonlineRegressor(method="dsb")` accepts dict `p0` like the other routes
  (it is normalized before reaching the positional-only `fit_dsb`).
- `EACFilter`/`LSIFilter` validate `drift_reset` at construction (`"full"` or
  `"inflate"`); a typo previously behaved silently as `"full"`.
- `NonlineRegressor`: standard sklearn validation errors now surface for
  sparse/NaN/empty/`y=None` inputs; plain-list 1-D inputs are promoted
  correctly; with `nan_policy="omit"` the fitter is allowed to drop non-finite
  pairs instead of being blocked by pre-validation.

## [0.1.0] - 2026-06

Initial development release: LSI/EAC/DSB batch fitters, `FittingResult`,
model catalog with self-seeding and `suggest_models`, streaming
`EACFilter`/`LSIFilter` + `FilterBank`, stochastic characterization and
forecasting pipeline, map-reduce/batched/parallel scale layer, sklearn
estimator, diagnostics, optional C kernels.
