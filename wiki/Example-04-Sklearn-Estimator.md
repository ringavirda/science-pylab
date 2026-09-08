# Example 04 Sklearn Estimator

The scikit-learn estimator -- NonlineRegressor.

NonlineRegressor wraps dtfit.fit behind the standard estimator API (fit /
predict / score), so it composes with Pipeline, GridSearchCV and
cross_val_score. It takes a single input feature (the model's variable), and
the basis and the order of the image are estimator parameters, so a grid
search can tune them.

Run headless:   python examples/04_sklearn_estimator.py

Source: [`packages/dtfit/examples/04_sklearn_estimator.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/04_sklearn_estimator.py)

```python
import numpy as np

from dtfit.sklearn import NonlineRegressor


def fit_predict_score(rng):
    X = np.linspace(0, 3, 200).reshape(-1, 1)
    y = 1.4 * np.exp(0.8 * X.ravel()) + rng.normal(0, 0.15, X.shape[0])
    reg = NonlineRegressor("a*exp(b*x)", "x").fit(X, y)
    print("== fit / predict / score ==")
    print("coef_:", np.round(reg.coef_, 4))
    print("R2   :", round(float(reg.score(X, y)), 4))
    print("basis:", reg.result_.basis_name, "at order",
          reg.result_.image_order)
    return X, y


def grid_search(X, y) -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    pipe = Pipeline([("fit", NonlineRegressor("a0 + a1*exp(a2*x)", "x"))])
    grid = GridSearchCV(
        pipe,
        {"fit__basis": ["legendre", "block"], "fit__order": [4, 6]},
        cv=3, scoring="r2",
    )
    grid.fit(X, y)
    print("\n== GridSearchCV ==")
    print("best params:", grid.best_params_)
    print("best CV R2 :", round(float(grid.best_score_), 4))


def cross_validate(X, y) -> None:
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        NonlineRegressor("a*exp(b*x)", "x"), X, y, cv=4, scoring="r2"
    )
    print("\n== cross_val_score ==")
    print("per-fold R2:", np.round(scores, 3))
    print("mean R2    :", round(float(scores.mean()), 4))


def main() -> None:
    rng = np.random.default_rng(0)
    X, y = fit_predict_score(rng)
    grid_search(X, y)
    cross_validate(X, y)


if __name__ == "__main__":
    main()
```

## Output (`python examples/04_sklearn_estimator.py`)

```text
== fit / predict / score ==
coef_: [1.4077 0.7975]
R2   : 0.9986
basis: legendre at order 4

== GridSearchCV ==
best params: {'fit__basis': 'legendre', 'fit__order': 4}
best CV R2 : 0.9682

== cross_val_score ==
per-fold R2: [0.885 0.941 0.984 0.995]
mean R2    : 0.9512
```
