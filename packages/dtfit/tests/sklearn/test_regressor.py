"""scikit-learn compatibility of NonlineRegressor."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from dtfit.sklearn import NonlineRegressor


def _reg():
    return NonlineRegressor("a*atan(w*x)", "x", basis="block", p0=[1.0, 1.0])


def test_fit_predict_score(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.predict(x).shape == x.shape
    assert reg.n_features_in_ == 1
    assert reg.score(x, y) > 0.8


def test_clone_and_get_params():
    reg = _reg()
    assert reg.get_params()["basis"] == "block"
    cloned = clone(reg)
    assert not hasattr(cloned, "coef_")


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        _reg().predict(np.array([1.0, 2.0]))


def test_pipeline(arctan_data):
    x, y, _ = arctan_data
    pipe = Pipeline([("reg", _reg())]).fit(x.reshape(-1, 1), y)
    assert pipe.score(x.reshape(-1, 1), y) > 0.8


def test_cross_val_score(arctan_data):
    x, y, _ = arctan_data
    scores = cross_val_score(
        _reg(), x.reshape(-1, 1), y, cv=KFold(3, shuffle=True, random_state=0)
    )
    assert len(scores) == 3
    assert scores.mean() > 0.8


def test_multifeature_rejected():
    reg = NonlineRegressor("a*x", "x")
    with pytest.raises(ValueError):
        reg.fit(np.zeros((10, 2)), np.zeros(10))


def test_predict_multifeature_rejected(arctan_data):
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    with pytest.raises(ValueError):
        reg.predict(np.zeros((5, 2)))


def test_sklearn_tags():
    tags = _reg().__sklearn_tags__()
    assert tags.estimator_type == "regressor"
    assert tags.target_tags.required is True


def test_feature_names_in_(arctan_data):
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    df = pd.DataFrame({"x": x})
    reg = _reg().fit(df, y)
    assert list(reg.feature_names_in_) == ["x"]
    assert reg.n_features_in_ == 1


def test_zero_arg_constructible_and_clonable():
    """A scikit-learn estimator must construct with no arguments and give every
    ``__init__`` parameter a default. ``clone`` and meta-estimator
    introspection both rely on that."""
    import inspect

    sig = inspect.signature(NonlineRegressor.__init__)
    required = [
        name for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect._empty
    ]
    assert not required, f"__init__ params without defaults: {required}"

    reg = NonlineRegressor()
    cloned = clone(reg)
    assert isinstance(cloned, NonlineRegressor)
    assert cloned.get_params() == reg.get_params()
    assert not hasattr(cloned, "coef_")


def test_default_estimator_fits_affine():
    """The default model is an affine fit: ``NonlineRegressor()`` is usable,
    not merely constructible."""
    rng = np.random.default_rng(0)
    x = np.linspace(-2.0, 2.0, 80)
    y = 1.5 + 0.7 * x + 0.01 * rng.standard_normal(x.size)
    reg = NonlineRegressor().fit(x, y)
    assert reg.score(x, y) > 0.99


def test_result_exposes_full_fitting_result(arctan_data):
    """``fit`` stores the whole FittingResult as ``result_``; that is how
    uncertainty is reached from the sklearn route."""
    from dtfit import FittingResult

    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    assert isinstance(reg.result_, FittingResult)
    assert np.allclose(reg.result_.coeffs, reg.coef_)
    assert reg.result_.converged is not None
    assert reg.result_.cov is not None
    se = reg.result_.stderr()
    assert set(se) == {"a", "w"}
    assert all(s > 0 for s in se.values())
    ci = reg.result_.confidence_intervals()
    for name, value in reg.result_.params.items():
        lo, hi = ci[name]
        assert lo <= value <= hi


def test_dict_p0_and_bounds_pass_through(arctan_data):
    """Dict-keyed ``p0`` and ``bounds`` reach the fitter and fit the data."""
    x, y, truth = arctan_data
    reg = NonlineRegressor(
        "a*atan(w*x)",
        "x",
        basis="block",
        p0={"a": 4.0, "w": 1.0},
        bounds={"a": (0.0, 10.0)},
    ).fit(x, y)
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_nan_policy_forwarded(arctan_data):
    """``nan_policy`` reaches the fitter. The default 'raise' rejects a NaN;
    'omit' drops the bad pairs and still fits."""
    x, y, truth = arctan_data
    y_bad = y.copy()
    y_bad[3] = np.nan
    with pytest.raises(ValueError):
        _reg().fit(x, y_bad)
    reg = NonlineRegressor(
        "a*atan(w*x)", "x", basis="block", p0=[1.0, 1.0], nan_policy="omit"
    )
    assert reg.__sklearn_tags__().input_tags.allow_nan is True
    reg.fit(x, y_bad)
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_robust_levers_rescue_outliers(arctan_data):
    """The robust image rescues an outlier-contaminated fit end to end, on
    both the block and legendre bases. Per-kwarg forwarding is the spy
    test in test_basis_and_order_reach_the_fit."""
    x, y, truth = arctan_data
    y_out = y.copy()
    y_out[50] += 60.0  # gross outlier
    expected = [truth["a"], truth["w"]]
    robust = NonlineRegressor(
        "a*atan(w*x)",
        "x",
        basis="block",
        p0=[1.0, 1.0],
        robust=True,
    ).fit(x, y_out)
    assert np.allclose(robust.coef_, expected, rtol=0.15)
    plain = _reg().fit(x, y_out)
    assert np.linalg.norm(robust.coef_ - expected) <= np.linalg.norm(
        plain.coef_ - expected
    )
    robust_lsi = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0], robust=True
    ).fit(x, y_out)
    assert np.allclose(robust_lsi.coef_, expected, rtol=0.15)


def test_sample_order_invariance(arctan_data):
    """The estimator sorts by x before the integral fitters, which makes
    shuffled samples give the same fit as ordered ones."""
    x, y, _ = arctan_data
    rng = np.random.default_rng(1)
    perm = rng.permutation(x.size)
    reg_sorted = _reg().fit(x, y)
    reg_shuffled = _reg().fit(x[perm], y[perm])
    assert np.allclose(reg_sorted.coef_, reg_shuffled.coef_)


def test_pickle_round_trip(arctan_data):
    """A fitted estimator pickles, rebuilding its lambdified model on load.
    The original stays usable afterwards."""
    import pickle

    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    expected = reg.predict(x[:10])
    clone_ = pickle.loads(pickle.dumps(reg))
    assert np.allclose(clone_.predict(x[:10]), expected)
    assert np.allclose(clone_.result_.coeffs, reg.result_.coeffs)
    # __getstate__ must not strip ``model_`` off the live estimator.
    assert np.allclose(reg.predict(x[:10]), expected)


def test_plain_list_input(arctan_data):
    """A plain 1-D Python list is promoted to a single column like an array."""
    x, y, _ = arctan_data
    reg = _reg().fit(list(x), list(y))
    assert reg.n_features_in_ == 1
    assert reg.predict(list(x[:5])).shape == (5,)


def test_sparse_input_rejected(arctan_data):
    """Sparse X is refused with scikit-learn's own TypeError."""
    sparse = pytest.importorskip("scipy.sparse")
    x, y, _ = arctan_data
    with pytest.raises(TypeError, match="[Ss]parse"):
        _reg().fit(sparse.csr_matrix(x.reshape(-1, 1)), y)


def test_callable_model_fits_and_scores(arctan_data):
    """A plain callable ``f(x, a, w)`` fits on the legendre basis. Its
    parameter names come from the signature, and the result carries a
    numeric evaluator in place of an expression, which keeps ``predict``
    working."""
    x, y, truth = arctan_data

    def model(x, a, w):
        return a * np.arctan(w * x)

    reg = NonlineRegressor(
        model, "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y)
    assert reg.coef_.shape == (2,)
    assert reg.score(x, y) > 0.8
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)
    assert reg.result_.expr is None
    assert reg.result_.param_model is not None
    assert reg.result_.names == ("a", "w")
    assert reg.predict(x[:5]).shape == (5,)


def test_callable_param_names_when_signature_opaque(arctan_data):
    """A callable with an opaque ``*args`` signature is fit by passing
    ``param_names``, which the estimator forwards to the fitter."""
    x, y, truth = arctan_data

    def model(x, *p):
        return p[0] * np.arctan(p[1] * x)

    reg = NonlineRegressor(
        model, "x", basis="legendre", param_names=["a", "w"], p0=[1.0, 1.0]
    ).fit(x, y)
    assert reg.result_.names == ("a", "w")
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_callable_and_param_names_forwarded_to_fitter(
    monkeypatch, arctan_data
):
    """The callable model and ``param_names`` reach the fitter untouched. No
    behavioural outcome can prove ``param_names`` was forwarded, so the test
    spies on the call."""
    x, y, _ = arctan_data
    captured = {}

    class _Stub:
        coeffs = np.array([1.0, 1.0])
        model = staticmethod(np.asarray)

    def fake_fit(model, data, var=None, **kwargs):
        captured["model"] = model
        captured["param_names"] = kwargs.get("param_names")
        return _Stub()

    monkeypatch.setattr("dtfit.sklearn.fit", fake_fit)

    def model(x, *p):
        return p[0] * x

    NonlineRegressor(model, "x", basis="legendre",
                     param_names=["a", "b"]).fit(x, y)
    assert captured["model"] is model
    assert list(captured["param_names"]) == ["a", "b"]


def test_sample_weight_downweights_outliers(arctan_data):
    """Down-weighting a contaminated region beats an unweighted fit: the
    weighted coefficients land closer to the ground truth."""
    x, y, truth = arctan_data
    expected = np.array([truth["a"], truth["w"]])
    y_bad = y.copy()
    corrupt = (x > 4.0) & (x < 6.0)
    y_bad[corrupt] += 25.0
    weight = np.ones_like(x)
    weight[corrupt] = 1e-3  # near-zero trust in the corrupted band
    weighted = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=weight)
    plain = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y_bad)
    assert np.linalg.norm(weighted.coef_ - expected) < np.linalg.norm(
        plain.coef_ - expected
    )
    assert np.allclose(weighted.coef_, expected, rtol=0.15)


def test_sample_weight_forwarded_as_sigma(monkeypatch, arctan_data):
    """``sample_weight`` becomes the fitter's ``sigma = 1/sqrt(weight)``,
    passed as relative weights with ``absolute_sigma`` left False."""
    x, y, _ = arctan_data
    captured = {}

    class _Stub:
        coeffs = np.array([1.0, 1.0])
        model = staticmethod(np.asarray)

    def fake_fit(model, data, var=None, **kwargs):
        captured["w"] = data.w
        captured.update(kwargs)
        return _Stub()

    monkeypatch.setattr("dtfit.sklearn.fit", fake_fit)
    weight = np.linspace(0.5, 2.0, x.size)
    NonlineRegressor("a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]).fit(
        x, y, sample_weight=weight
    )
    # arctan_data x is already ascending, so the estimator's x-sort is a
    # no-op and the image weights align 1:1 with the input weights:
    # sigma = 1/sqrt(w) going in, w = 1/sigma**2 coming out.
    assert np.allclose(captured["w"], weight)
    assert captured.get("absolute_sigma", False) is False


def test_zero_sample_weight_ignores_sample(arctan_data):
    """A zero weight maps to a huge but finite sigma instead of crashing on
    1/sqrt(0). The sample drops out and the fit still lands on the truth."""
    x, y, truth = arctan_data
    weight = np.ones_like(x)
    weight[::7] = 0.0
    reg = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y, sample_weight=weight)
    assert np.all(np.isfinite(reg.coef_))
    assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_all_zero_sample_weight_raises(arctan_data):
    x, y, _ = arctan_data
    with pytest.raises(ValueError, match=r"(?i)zero"):
        NonlineRegressor(
            "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
        ).fit(x, y, sample_weight=np.zeros_like(x))


def test_negative_sample_weight_raises(arctan_data):
    """A negative weight has no meaning for a curve fit."""
    x, y, _ = arctan_data
    weight = np.ones_like(x)
    weight[0] = -1.0
    with pytest.raises(ValueError, match=r"(?i)non-negative|negative"):
        NonlineRegressor(
            "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
        ).fit(x, y, sample_weight=weight)


def test_result_has_rsquared(arctan_data):
    """The legendre fitter records the fit-quality stats and ``result_``
    exposes them."""
    x, y, _ = arctan_data
    reg = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y)
    r2 = reg.result_.rsquared
    assert r2 is not None
    assert 0.8 < r2 <= 1.0
    assert reg.result_.aic is not None
    assert reg.result_.bic is not None
    assert reg.result_.n_obs == x.size


def test_sample_weight_with_nan_omit_on_every_basis(arctan_data):
    """``sample_weight`` plus ``nan_policy='omit'`` plus a NaN row must fit in
    every basis. The hazard is sigma length: the weights and the samples have
    to agree on how many survive the dropped rows."""
    x, y, truth = arctan_data
    y = y.copy()
    y[7] = np.nan
    w = np.full(x.size, 2.0)  # full-length weights, one per raw sample
    for basis in ("auto", "legendre", "block"):
        reg = NonlineRegressor(
            "a*atan(w*x)", "x", basis=basis, p0=[1.0, 1.0],
            nan_policy="omit",
        ).fit(x, y, sample_weight=w)
        assert np.allclose(reg.coef_, [truth["a"], truth["w"]], rtol=0.2)


def test_predict_series_returns_aligned_series(arctan_data):
    """Fit on a Series x/y and predicting from a Series returns a Series that
    carries the query index, holding the plain-ndarray prediction values."""
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    idx = pd.RangeIndex(100, 100 + x.size)
    sx = pd.Series(x, index=idx)
    sy = pd.Series(y, index=idx)
    reg = _reg().fit(sx, sy)

    xq = x[:5]
    idxq = pd.Index([7, 8, 9, 10, 11])
    pred = reg.predict(pd.Series(xq, index=idxq))
    assert isinstance(pred, pd.Series)
    assert list(pred.index) == list(idxq)
    assert np.array_equal(pred.to_numpy(), reg.predict(xq))


def test_predict_ndarray_still_ndarray(arctan_data):
    """An ndarray or list input keeps returning an ndarray. The pandas branch
    must not leak into the non-pandas path."""
    x, y, _ = arctan_data
    reg = _reg().fit(x, y)
    out = reg.predict(x[:5])
    assert isinstance(out, np.ndarray)
    assert not hasattr(out, "index")
    out_list = reg.predict(list(x[:5]))
    assert isinstance(out_list, np.ndarray)


def test_predict_single_col_dataframe_returns_series(arctan_data):
    """A single-column DataFrame X predicts to a Series on the frame's row
    index."""
    pd = pytest.importorskip("pandas")
    x, y, _ = arctan_data
    # Fitting on a named single-column frame lines feature_names_in_ up with
    # the query frame, keeping sklearn's feature-name warning out of the test.
    reg = _reg().fit(pd.DataFrame({"x": x}), y)
    idxq = pd.Index([3, 4, 5, 6, 7])
    df = pd.DataFrame({"x": x[:5]}, index=idxq)
    pred = reg.predict(df)
    assert isinstance(pred, pd.Series)
    assert list(pred.index) == list(idxq)
    # identical data gives an identical fit, so the all-ndarray route is the
    # reference for the values
    expected = _reg().fit(x, y).predict(x[:5])
    assert np.array_equal(pred.to_numpy(), expected)


def test_sample_weight_as_series_works(arctan_data):
    """A Series ``sample_weight`` is coerced to an ndarray and reaches the
    fitter as sigma exactly like an ndarray weight."""
    pd = pytest.importorskip("pandas")
    x, y, truth = arctan_data
    weight = np.ones_like(x)
    corrupt = (x > 4.0) & (x < 6.0)
    y_bad = y.copy()
    y_bad[corrupt] += 25.0
    weight[corrupt] = 1e-3
    sw = pd.Series(weight, index=pd.RangeIndex(x.size))
    weighted = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=sw)
    weighted_np = NonlineRegressor(
        "a*atan(w*x)", "x", basis="legendre", p0=[1.0, 1.0]
    ).fit(x, y_bad, sample_weight=weight)
    assert np.allclose(weighted.coef_, weighted_np.coef_)
    assert np.allclose(weighted.coef_, [truth["a"], truth["w"]], rtol=0.15)


def test_basis_and_order_reach_the_fit(monkeypatch, arctan_data):
    """``basis`` and ``order`` are estimator parameters and must arrive at
    :func:`dtfit.fit` unchanged, which is what puts them in a grid search."""
    import dtfit.sklearn as sk

    x, y, _ = arctan_data
    captured = {}
    real = sk.fit

    def spy(model, data, var=None, **kwargs):
        captured.update(kwargs)
        return real(model, data, var, **kwargs)

    monkeypatch.setattr(sk, "fit", spy)
    NonlineRegressor(
        "a*atan(w*x)", "x", basis="block", order=10, p0=[1.0, 1.0],
        robust=True,
    ).fit(x, y)
    assert captured["basis"] == "block"
    assert captured["order"] == 10
    assert captured["robust"] is True


def test_grid_search_over_basis_and_order_does_not_crash(arctan_data):
    """GridSearchCV can vary ``basis`` and ``order`` without erroring, which
    is what exposing them as estimator parameters buys. The candidates here
    are not tuned for fit quality (order 6 under-covers the model), so this
    only checks the search completes and reports a score."""
    from sklearn.model_selection import GridSearchCV

    x, y, _ = arctan_data
    grid = GridSearchCV(
        NonlineRegressor("a*atan(w*x)", "x", p0=[1.0, 1.0]),
        {"basis": ["legendre", "block"], "order": [6, 12]},
        cv=3,
    )
    grid.fit(x, y)
    assert grid.best_params_["basis"] in ("legendre", "block")
    assert grid.best_params_["order"] in (6, 12)
    assert grid.best_score_ > 0.0
