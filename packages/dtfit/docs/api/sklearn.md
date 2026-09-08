# scikit-learn estimator

`NonlineRegressor` is a scikit-learn-compatible estimator (fit / predict /
score) over `dtfit.fit`. It composes with `Pipeline`, `GridSearchCV` and the
rest of the sklearn ecosystem, so the basis and the order of the image
cross-validate like any other hyperparameter. It lives in `dtfit.sklearn` and
is the only part of dtfit that imports scikit-learn.

```python
from sklearn.model_selection import GridSearchCV
from dtfit.sklearn import NonlineRegressor

reg = NonlineRegressor(expr="a*exp(b*x)", var="x")
search = GridSearchCV(reg, {"basis": ["legendre", "block"],
                            "order": [4, 6, 12]}, cv=5)
search.fit(X, y)          # X is a 2-D column vector of the single feature
```

::: dtfit.sklearn.NonlineRegressor
