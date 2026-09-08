# Models

A catalog of self-seeding model families and an AIC/BIC recommender, so you pick
*structure* (a growth curve, a saturating rise, a peak) rather than writing SymPy
strings. Catalog families live under `dtfit.models` (e.g. `models.logistic()`,
`models.exponential()`), compose with `+`, and self-seed `p0`/`bounds` from the
data on `.fit(x, y)`. Register your own family with `register`.

```python
from dtfit import models, suggest_models

fit = models.logistic().fit(x, y)                 # self-seeds from the data
fit = (models.linear() + models.sine()).fit(x, y) # compose trend + cycle

for s in suggest_models(x, y)[:3]:                # ranked best-first by AIC
    print(s.name, round(s.r2, 4), round(s.aic, 1))
```

::: dtfit.models.Model

::: dtfit.suggest_models

::: dtfit.models.register

::: dtfit.models.unregister

::: dtfit.models.resolve_model
