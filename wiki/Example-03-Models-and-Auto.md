# Example 03 Models and Auto

Models and automatic fitting.

Picking the structurally-correct model is most of the battle. dtfit.models is a
catalog of named, self-seeding families (they read p0/bounds off the data),
composable with "+", plus a recommender that ranks families by AIC. auto_estimate
and auto_forecast route by signal shape for callers who do not want the model
framework.

Run headless:   python examples/03_models_and_auto.py

Source: [`packages/dtfit/examples/03_models_and_auto.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/03_models_and_auto.py)

```python
from collections import defaultdict

import numpy as np

from dtfit import models, suggest_models, auto_estimate, auto_forecast
from dtfit.models import CATALOG


def catalog() -> None:
    by_cat: dict[str, list[str]] = defaultdict(list)
    for name, factory in CATALOG.items():
        by_cat[factory().category].append(name)
    print("== catalog ({} families) ==".format(len(CATALOG)))
    for cat, names in by_cat.items():
        print("  {:11s}: {}".format(cat, ", ".join(names)))


def self_seeding(rng) -> None:
    x = np.linspace(0, 10, 200)
    y = 8.0 / (1 + np.exp(-0.9 * (x - 5))) + rng.normal(0, 0.2, x.size)
    fit = models.logistic().fit(x, y)          # no p0/bounds supplied
    print("\n== models.logistic().fit (self-seeded) ==")
    print("params:", {k: round(v, 3) for k, v in fit.params.items()})


def composition(rng) -> None:
    x = np.linspace(0, 12, 240)
    y = (1.0 + 0.6 * x) + 2.5 * np.sin(1.3 * x) + rng.normal(0, 0.2, x.size)
    model = models.linear() + models.sine()    # trend + cycle
    fit = model.fit(x, y)
    print("\n== models.linear() + models.sine() ==")
    print("model:", model)
    print("rmse:", round(float(np.sqrt(np.mean((y - fit.predict(x)) ** 2))), 4))


def recommend(rng) -> None:
    x = np.linspace(0, 8, 200)
    y = 3.0 * np.exp(-((x - 4.0) ** 2) / (2 * 0.8 ** 2)) + rng.normal(0, 0.04, x.size)
    print("\n== suggest_models (ranked by AIC) ==")
    for s in suggest_models(x, y, top=5):
        print("  {:24s} r2={:.4f}  aic={:8.1f}".format(s.name, s.r2, s.aic))


def routing(rng) -> None:
    x = np.linspace(0, 10, 300)
    y = 1.5 * np.sin(2.1 * x) + rng.normal(0, 0.1, x.size)
    res = auto_estimate(x, y, "A*sin(w*x)", "x", freq_param="w")
    print("\n== auto_estimate (oscillatory route) ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})

    t = np.arange(120)
    series = 10.0 / (1 + np.exp(-0.12 * (t - 45))) + rng.normal(0, 0.015, t.size)
    cut, h = 90, 30
    fc = auto_forecast(t[:cut], series[:cut], horizon=h)     # routes to logistic
    print("\n== auto_forecast (saturating growth -> logistic) ==")
    print("forecast end:", round(float(fc[-1]), 2),
          " actual end:", round(float(series[cut + h - 1]), 2))


def main() -> None:
    rng = np.random.default_rng(0)
    catalog()
    self_seeding(rng)
    composition(rng)
    recommend(rng)
    routing(rng)


if __name__ == "__main__":
    main()
```

## Output (`python examples/03_models_and_auto.py`)

```text
== catalog (24 families) ==
  trend      : linear, quadratic, cubic, power_law, logarithmic, sqrt_law
  growth     : exponential, exp_growth_offset
  decay      : exp_decay, exp_decay_offset, first_order, biexponential, stretched_exponential
  sigmoid    : logistic, gompertz, weibull_cdf, tanh_step
  saturating : michaelis_menten, hill
  peak       : gaussian, lorentzian, double_gaussian
  oscillatory: sine, damped_oscillation

== models.logistic().fit (self-seeded) ==
params: {'L': 7.994, 'k': 0.887, 'x0': 4.99}

== models.linear() + models.sine() ==
model: Model('linear+sine', expr='(a0 + a1*x) + (A*sin(p + w*x) + c)', shape='composite')
rmse: 0.2074

== suggest_models (ranked by AIC) ==
  gaussian                 r2=0.9985  aic= -1287.2
  double_gaussian          r2=0.9985  aic= -1286.2
  lorentzian               r2=0.9536  aic=  -601.5
  quadratic                r2=0.5221  aic=  -135.0
  cubic                    r2=0.5221  aic=  -133.0

== auto_estimate (oscillatory route) ==
params: {'A': 1.505, 'w': 2.099}

== auto_forecast (saturating growth -> logistic) ==
forecast end: 10.0  actual end: 9.95
```
