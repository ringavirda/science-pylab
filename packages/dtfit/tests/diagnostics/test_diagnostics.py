"""dtfit.diagnostics: fit-aware reports (UQ + IC + residual tests) and
displays."""

import numpy as np
import pytest

from dtfit import fit_lsi
from dtfit.diagnostics import fit_report, residual_diagnostics


@pytest.fixture
def exp_fit():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300)
    y = 2.0 * np.exp(0.8 * t) + rng.normal(0, 0.05, t.size)
    r = fit_lsi(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0])
    return r, t, y


def test_fit_report_keys_and_quality(exp_fit):
    r, t, y = exp_fit
    rep = fit_report(r, t, y)
    keys = ("n", "n_params", "rss", "rmse", "r2", "aic", "bic",
            "durbin_watson")
    for key in keys:
        assert key in rep
    assert rep["n"] == y.size and rep["n_params"] == 2
    assert rep["r2"] > 0.99
    # a covariance is available, so params and stderr are reported too
    assert "params" in rep and "stderr" in rep
    assert set(rep["params"]) == {"a", "b"}


def test_fit_report_aic_bic_prefer_correct_model(exp_fit):
    r, t, y = exp_fit
    good = fit_report(r, t, y)
    # an underspecified linear model scores worse (higher) on AIC/BIC
    bad = fit_report(fit_lsi(t, y, "a0 + a1*t", "t", p0=[1.0, 1.0]), t, y)
    assert good["aic"] < bad["aic"] and good["bic"] < bad["bic"]


def test_residual_diagnostics(exp_fit):
    r, t, y = exp_fit
    d = residual_diagnostics(r, t, y)
    assert d["residuals"].shape == y.shape
    assert 0.0 < d["durbin_watson"] < 4.0
    assert np.isfinite(d["lag1_autocorr"])


def test_fit_display_from_estimator(arctan_data):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from dtfit.diagnostics import FitDisplay
    from dtfit.sklearn import NonlineRegressor

    x, y, _ = arctan_data
    reg = NonlineRegressor("a*atan(w*x)", "x", basis="block",
                           p0=[1, 1]).fit(x, y)
    disp = FitDisplay.from_estimator(reg, x, y)
    assert disp.ax_ is not None
    assert disp.figure_ is not None
    assert disp.line_ is not None and disp.scatter_ is not None
    plt.close(disp.figure_)


def test_residuals_display_from_predictions():
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    from dtfit.diagnostics import ResidualsDisplay

    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 10, 50)
    y_pred = y_true + rng.normal(0, 0.3, 50)
    fig, ax = plt.subplots()
    disp = ResidualsDisplay.from_predictions(y_true, y_pred, ax=ax)
    assert disp.ax_ is ax
    assert disp.residuals.shape == y_true.shape
    plt.close(fig)
