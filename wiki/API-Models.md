# API: the model framework

Pick a *shape*, not a formula. `dtfit.models` is a catalog of named model
families that read their own starting guesses off the data, compose with `+`, and
can be ranked automatically. Background: the domain studies found that **choosing
the structurally-correct model is the whole game** -- this package makes that
choice ergonomic.

- [`Model`](#model) -- a model family (expression + self-seeder + fit)
- [`suggest_models`](#suggest_models) / [`Suggestion`](#suggestion) -- fit & rank candidates
- [the catalog](#catalog) -- the built-in families
- [`Stochastic`](#stochastic) -- the stochastic-series model in the same `.fit()` convention

```python
from dtfit import models
fit = models.logistic().fit(x, y)                    # self-seeded
fit = (models.linear() + models.sine()).fit(x, y)    # compose trend + cycle
for s in suggest_models(x, y)[:3]:
    print(s.name, s.r2, s.aic)

walk = np.cumsum(rng.normal(size=300))
m = models.Stochastic().fit(walk)                     # a random series -> StochasticModel
```

---

<a name="model"></a>
## `Model`

```python
Model(expr, var="x", *, name="", shape="bulk", category="general",
      freq_param=None, seeder=None, param_names=None)

Model.from_callable(func, names=None, *, var="x", name="", shape="bulk",
                    category="general", freq_param=None, seeder=None)
```

A model family: the model plus a routing `shape`, an optional angular-frequency
parameter, and an optional data-driven `seeder` that produces `{name: (p0, lo, hi)}`.
You normally get `Model` instances from the [catalog](#catalog) rather than
constructing them.

**`expr` may be symbolic *or* a callable.** Pass either a SymPy-expression
string (e.g. `"a*exp(b*x)"`) or a plain Python callable
`f(x, *params)` -- resolved through [`resolve_model`](#resolve_model),
so a callable's parameter order is the callable's **signature** order (not the
sorted order symbolic models use) and it fits through the same engines.
[`Model.from_callable(func, names=..., var=..., ...)`](#from_callable) is a
convenience wrapper for the callable path (`names` = the parameter names after the
leading `x`; `param_names` on the constructor does the same). Composition with `+`
and the seed-detrend evaluator require **symbolic** operands; a callable model
raises a clear error there (see [`__add__`](#__add__-composition-with-)).

**Key attributes:** `expr` (the SymPy string when symbolic, else `None`), `func`
(the Python callable when non-symbolic, else `None`), `is_symbolic` (`True` for a
string model, `False` for a callable), `var`, `name`, `shape` (`"bulk"` /
`"oscillatory"` / `"transient"` / `"peak"` / `"composite"` -- a label for the
catalog and for `suggest_models`, not a route), `category`, `freq_param`,
`params` (the parameter-name tuple, in **sorted** order for a symbolic model,
**signature** order for a callable).

### `fit(data, y=None, *, basis="auto", order=None, p0=None, bounds=None) -> FittingResult`
Fit this family to the data, **self-seeding** `p0`/`bounds` from the model's
seeder unless you override them.

`data` is an [`Original`](API-Fitting#original), an
[`Image`](API-Fitting#image), or the sample positions with `y`. An Image is
fitted as it is; its seed comes from an Original reconstructed on 400 points
over the image's domain, since the seeders read shapes off samples, and
`basis="auto"` raises on one because the routing needs the samples.

| `basis` | fits with | needs samples |
|---|---|---|
| `"auto"` (default) | [`fit`](API-Fitting#fit)'s candidates, keeping the lowest residual over the samples | yes |
| `"legendre"` | the Legendre basis at `order`, or `order_for` when `order` is `None` | no |
| `"block"` | the block basis at `order` windows, or `4 * n_params` when `order` is `None` | no |

> **Partially-bounded seeds are kept.** A seeder that bounds some parameters
> and leaves others infinite does not lose *all* its bounds: the mixed pairs
> reach the solver, constraining the local trust-region solve (the global
> differential-evolution stage runs only on fully-finite boxes). A fully
> unbounded seed maps to the unconstrained path. A failed `suggest_models`
> candidate emits a `UserWarning` naming the family instead of vanishing
> silently from the ranking.

### `seed(x, y) -> dict`
The data-driven `{name: (p0, lo, hi)}` seed map (empty if the family has no
seeder).

<a name="from_callable"></a>
### `Model.from_callable(func, names=None, *, var="x", name="", shape="bulk", category="general", freq_param=None, seeder=None) -> Model`
Build a `Model` from a Python callable `func(x, *params)` (a convenience wrapper
over the constructor's callable path). `names` are the parameter names after the
leading `x`, in call order; they are **introspected from the signature** when
omitted and are only *required* for a callable with no inspectable signature (a
builtin) or a `*params` signature. The parameter order is the callable's signature
order. A callable model fits through the same engines but **cannot compose** with
`+` (see [`__add__`](#__add__-composition-with-)).

```python
import numpy as np
from dtfit.models import Model

def decay(x, a, b):
    return a * np.exp(-b * x)

m = Model.from_callable(decay, shape="bulk")   # params ('a', 'b') in signature order
fit = m.fit(x, y)                              # same self-seeding fit path
```

### `__add__` (composition with `+`)
`model_a + model_b` builds a new additive model (e.g. `trend + seasonal`).
Colliding parameter names in the right operand are renamed, and the seeders
compose so the combined model is **still self-seeding**: the second component is
seeded on the detrended residual of the first. Both operands must share the same
`var`. **Symbolic composition only** -- both operands must be symbolic (string)
models; composing a **callable** model raises `TypeError` (a callable's expression
cannot be renamed / detrended symbolically). Fit the callable model on its own, or
compose the symbolic forms.

```python
trend_plus_cycle = models.linear() + models.sine()
fit = trend_plus_cycle.fit(x, y)
```

<a name="multivariate-data"></a>
### Multivariate data

**dtfit is one-dimensional.** LSI projects onto Legendre polynomials over a
scalar interval and EAC integrates windows on a scalar axis, so **multivariate
`X` (several predictors) is not supported** -- every fitting entry point raises a
clear error on a 2-D `X` (or a multi-column `DataFrame`) rather than silently
producing a wrong fit.

The case usually mistaken for multivariate is a **1-D signal that is a sum of
components on one axis** (trend + cycle, baseline + peak). That is still 1-D, and
the `+` composition above fits it directly: `models.linear() + models.sine()` is
`a0 + a1*t + A*sin(w*t)` **in the single variable `t`** -- not `g(t) + h(u)` over
two different inputs.

For a genuinely separable multi-predictor model `y = g1(x1) + g2(x2)` (distinct
predictors), dtfit does not fit it in one call. Either **backfit** -- alternate
1-D `fit_lsi`/`fit_eac` fits on the partial residuals (`g1` on `y - g2`, then `g2`
on `y - g1`, repeat) so each component keeps dtfit's seeding/robustness -- or use a
general nonlinear-least-squares tool (`scipy.optimize.curve_fit` / `lmfit`), which
is dimension-agnostic. dtfit is complementary to those; its edge (1-D streaming,
out-of-memory map-reduce, embedded, robustness without tuning) does not apply to a
scattered multivariate cloud.

---

<a name="suggest_models"></a>
## `suggest_models`

```python
suggest_models(data, y=None, candidates=None, *, basis="auto", top=None,
               include=None, exclude=None) -> list[Suggestion]
```

`data` is an `Original`, an `Image`, or the sample positions with `y`; `basis`
is forwarded to every candidate's [`Model.fit`](#modelfit). An Image is
shortlisted and seeded on its 400-point reconstruction but scored on the image
itself, so the ranking is the one the samples would give.

Fit candidate families to `(x, y)` and rank them **best-first by AIC**.

| arg | default | meaning |
|---|---|---|
| `candidates` | `None` | models to try; default is a **shape-based shortlist** of the catalog (oscillatory data skips peak/monotone families, etc.; ambiguous data falls back to the whole catalog so the true family is never dropped) |
| `method` | `"auto"` | fitting method passed to each model |
| `top` | `None` | if given, return only the best `top` |
| `include` | `None` | keep only candidates whose **name** (e.g. `"logistic"`) or **category** (e.g. `"decay"`, `"oscillatory"`) is in this list -- restrict the search to families you believe plausible |
| `exclude` | `None` | drop candidates whose name or category is in this list -- prune families you know don't apply (e.g. `exclude=["oscillatory"]` on a monotone series) without post-filtering. Applied **after** `include` |

Candidates whose fit fails (or yields non-finite R^2) are skipped. Each result is a
[`Suggestion`](#suggestion).

```python
for s in suggest_models(x, y, top=3):
    print(f"{s.name:20s} r2={s.r2:.4f} aic={s.aic:.1f}")

# you know the data is not oscillatory -> prune those families up front
suggest_models(x, y, exclude=["oscillatory"])
# or restrict to the exponential growth/decay families only
suggest_models(x, y, include=["growth", "decay"])
```

<a name="suggestion"></a>
### `Suggestion`

One ranked candidate. Attributes: `name`, `model` ([`Model`](#model)), `result`
([`FittingResult`](API-Types)), `report` (the full [`fit_report`](API-Diagnostics)
dict). Convenience properties: `aic`, `bic`, `r2`.

---

<a name="register"></a>
## `register` / `unregister`

```python
from dtfit.models import register, unregister

register(name, factory, *, overwrite=False)
unregister(name)
```

Add your own family to the catalog so `all_models()` **and** `suggest_models`
see it. `factory` is a zero-argument callable returning a [`Model`](#model) (like
every built-in catalog entry); it is probed once at registration for a clear
early error. A name collision raises `ValueError` unless `overwrite=True`.
A registered family with a custom `category` is **never silently dropped** from
the `suggest_models` shortlist.

```python
import numpy as np
from dtfit import suggest_models
from dtfit.models import Model, register, unregister

register("myline", lambda: Model("a0 + a1*x", name="myline"))
[s.name for s in suggest_models(x, y)]   # "myline" is a candidate now
unregister("myline")
```

---

<a name="catalog"></a>
## The catalog

`from dtfit import models` exposes one constructor per family (each returns a
self-seeding [`Model`](#model)), grouped by `category`. `models.CATALOG` is the
full registry and `models.all_models()` returns one instance of each.

| category | families (constructors) |
|---|---|
| **trend** | `linear`, `quadratic`, `cubic`, `power_law`, `logarithmic`, `sqrt_law` |
| **growth** | `exponential`, `exp_growth_offset` |
| **decay / relaxation** | `exp_decay`, `exp_decay_offset`, `first_order`, `biexponential`, `stretched_exponential` |
| **sigmoid** | `logistic`, `gompertz`, `weibull_cdf`, `tanh_step` |
| **saturating / rational** | `michaelis_menten`, `hill` |
| **peak** | `gaussian`, `lorentzian`, `double_gaussian` |
| **oscillatory** | `sine`, `damped_oscillation`, `fourier_series` |

```python
from dtfit import models

models.logistic()            # L/(1 + exp(-k*(x - x0))), shape="bulk"/sigmoid, self-seeds L,k,x0
models.gaussian()            # a peak family; auto routes it to Legendre at order_for
models.damped_oscillation()  # oscillatory, carries a freq_param

[m.name for m in models.all_models()]   # every family
```

Each constructor accepts no required arguments and reads its parameter ranges off
the data at `fit` time, so the typical use is a one-liner:
`models.<family>().fit(x, y)`.

> **Domain requirement (anchored-at-zero families).** Three trend families use an
> `x`-anchored basis and require the data to start near 0: `sqrt_law`
> (`a + b*sqrt(x)`, requires `x >= 0`), `logarithmic` (`a + b*log(x + 1)`, requires
> `x > -1`), and `power_law` (`a*(x + 1)**b`, requires `x > -1`). Their seeders
> raise `ValueError` if `x` violates the domain -- shift `x` to start at (or above)
> 0 before fitting these families.

---

<a name="stochastic"></a>
## `Stochastic`

```python
Stochastic(*, period=None, max_harmonics=4, forecaster="auto", **gates)
Stochastic.fit(x, y=None) -> StochasticModel
```

The stochastic-series model in the catalog `.fit()` convention. Unlike the families
above it does **not** fit a `y = f(x)` curve -- it characterizes a *random series*
across the stochastic routes (trend / cycle / long memory / mean reversion /
volatility) behind significance gates -- but it is driven the same way. `fit`
accepts `fit(series)` or `fit(t, series)` and returns a `StochasticModel` (also on
`.model_`) that forecasts and generates. It is not a `Model` subclass (no sympy
expression) and is not in the AIC catalog -- it just shares the convention; full
reference in [stochastic.md](API-Stochastic). Available as
`dtfit.models.Stochastic`.

```python
from dtfit.models import Stochastic
walk = np.cumsum(rng.normal(size=300))
m = Stochastic().fit(walk)          # -> a fitted StochasticModel
print(m.regime); m.forecast(12); m.simulate(200)
```

---

<a name="accuracy"></a>
## Accuracy and known limits

Every family above is exercised by an **accuracy corpus**
([`packages/dtfit/tests/accuracy/`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/tests/accuracy)): each
is fit on a known-ground-truth signal across a noise sweep, through the realistic
self-seeded `models.<family>().fit(x, y)` path, and required to stay competitive
with the `scipy.curve_fit` gold standard. A checked-in golden baseline guards
against silent accuracy drift. So the one-liner is validated for **all** families
-- but two honest limits are worth knowing.

### Parameter recovery vs. curve quality

For most families the parameters are **identifiable**: the fit recovers the true
values (relative error a few percent at 5 % noise). Three families are only weakly
identifiable -- many parameter sets reproduce the same curve, so a noisy fit lands
on *a* curve that matches, not on the true parameters:

| family | why parameters are weakly identifiable | judge by |
|---|---|---|
| `biexponential` | a sum of exponentials trades rate against amplitude (a classic ill-conditioned inverse problem) | curve quality (R^2); recovery degrades from ~10 % noise up |
| `double_gaussian` | overlapping peaks can swap labels / trade width for amplitude | curve quality (R^2) |
| `fourier_series` | individual harmonic amplitudes/phases are weakly constrained | curve quality (R^2) |

For these, trust the **fitted curve** (`R^2`, prediction) rather than the
individual coefficients, and prefer extra data / lower noise / bounds if you need
the parameters themselves.

### Cycles and the frequency seed

Recovering a frequency needs either a basis that resolves the cycle or a seed
for it, and the routing supplies both. An oscillatory family
(`models.sine()`, `models.damped_oscillation()`) names a `freq_param`, so
every candidate of [`fit(basis="auto")`](API-Fitting#fit) runs the
**oscillatory recipe**: smoothing off, the order raised to resolve the cycle,
and the frequency seeded from the peak of the *detrended* spectrum
([`fft_frequency_seed`](API-Fitting#fft_frequency_seed)). A **composite** such
as `(linear() + sine()).fit(...)` forwards its `freq_param` the same way;
removing the straight line before the FFT is what lets that seed see a cycle
riding on a trend (R^2 0.999990 and `w` 1.3003 against a true 1.3, where the
undetrended seed gave 0.794 and 0.376).

Without a `freq_param` the route still finds the cycle, by outcome rather than
by recipe: on the validation corpus's `c + A*sin(w*x + p)` scenario from bare
defaults, [`NonlineRegressor("c + A*sin(w*x + p)", "x")`](API-Estimator)
reaches R^2 1.00000 by routing to the block basis at 16 windows, where the
same fit pinned to `basis="legendre"` at its order rule (9) reaches 0.078.
Fixing the basis is what loses a cycle, not the estimator. See the
[LSI oscillatory recipe](Methods-LSI#the-oscillatory-recipe).

### `suggest_models` coverage

The default shortlist spans the whole catalog (the shape detector is permissive:
a strongly monotone series keeps the trend/growth/decay/sigmoid/saturating
families, a peak keeps the peak families, and ambiguous data falls back to
everything, so the true family is never silently dropped). `fourier_series` is the
one exception -- a parametric factory offered separately, not in the default sweep,
so a periodic signal surfaces its fundamental `sine`; add `fourier_series()`
explicitly to the `candidates` if you need the harmonic model.

---

<a name="resolve_model"></a>
## Also exported from `dtfit.models`

`resolve_model(model, var=None, *, param_names=None) -> ModelSpec` turns a
SymPy-expression string, a `sympy.Expr` or a callable `f(x, *params)` into the
one object every fitter evaluates; `ModelSpec.names` is the canonical
parameter order. `normalize_p0(p0, names)` and `normalize_bounds(bounds,
names)` put a guess and a box into that order, accepting the positional, the
`{name: value}` and the scipy forms. `result_kwargs(spec, coeffs)` bridges a
spec to a [`FittingResult`](API-Types).
