"""Smoke test for the README quick-start snippets.

Each test mirrors one README block and asserts only that it runs and returns
something well-shaped; numerical accuracy is the validation suite's job. What
this catches is a rename or signature change that would leave the front page
advertising an API nobody can call.
"""

import numpy as np
import pytest


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 10, 400)
    y = 2.0 * np.arctan(1.5 * x) + rng.normal(0, 0.02, x.size)
    return x, y


def test_readme_batch_and_estimator(series):
    import dtfit as dt

    x, y = series
    result = dt.fit_eac(x, y, "a*atan(w*x)", "x")
    assert set(result.params) == {"a", "w"}

    reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x", method="lsi")
    reg.fit(x, y)
    assert reg.predict(x).shape == x.shape


def test_readme_streaming():
    import dtfit as dt

    t = np.linspace(0, 20, 600)
    flt = dt.EACFilter("A*sin(w*t)", "t", p0=[1.0, 1.0], window_size=50)
    for ti, yi in zip(t, 2.0 * np.sin(1.5 * t)):
        flt.partial_fit(ti, yi)
    assert set(flt.params_) == {"A", "w"}


def test_readme_model_framework(series):
    from dtfit import models, suggest_models

    x, y = series
    fit = models.logistic().fit(x, y)
    assert hasattr(fit, "params")
    combo = (models.linear() + models.sine()).fit(x, y)
    assert combo.predict(x).shape == x.shape
    ranked = suggest_models(x, y)[:3]
    assert ranked and all(hasattr(s, "aic") for s in ranked)


def test_readme_uncertainty_and_serialization():
    import dtfit as dt

    rng = np.random.default_rng(1)
    x = np.linspace(0.1, 3, 200)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.05, x.size)
    r = dt.fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1, 1])
    assert set(r.params) == {"a", "b"}
    r.stderr()
    r.confidence_intervals(0.95)
    y_hat, y_std = r.predict(x, return_std=True)
    assert y_hat.shape == x.shape == y_std.shape
    rt = dt.FittingResult.from_dict(r.to_dict())
    np.testing.assert_allclose(rt.coeffs, r.coeffs)


def test_readme_diagnostics():
    from dtfit import fit_lsi
    from dtfit.diagnostics import fit_report, residual_diagnostics

    rng = np.random.default_rng(2)
    x = np.linspace(0.1, 3, 200)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.05, x.size)
    r = fit_lsi(x, y, "a*exp(b*x)", "x", p0=[1, 1])
    rep = fit_report(r, x, y)
    assert {"rmse", "r2", "aic", "bic"} <= set(rep)
    residual_diagnostics(r, x, y)
