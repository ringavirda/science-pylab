"""Overlapping-window ensemble fits (``ensemble_fit``)."""

import numpy as np
import pytest

import dtfit as dt
from dtfit import EnsembleResult, FittingResult


def _exp_data(seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 3, 600)
    y = 1.0 * np.exp(0.9 * t) + rng.normal(0, 0.02, t.size)
    return t, y, (1.0, 0.9)


def test_ensemble_recovers_with_spread_and_members():
    t, y, (a, b) = _exp_data()
    e = dt.ensemble_fit(t, y, "a*exp(b*t)", "t", method="eac", n_windows=6,
                        p0=[1.0, 1.0])
    assert abs(e.coeffs[1] - b) < 0.2
    assert e.spread.shape == (2,) and np.all(e.spread >= 0)
    assert e.members.ndim == 2 and e.members.shape[1] == 2


def test_ensemble_result_is_a_fitting_result():
    """It composes with the standard result API (named params, predict, cov)."""
    t, y, _ = _exp_data()
    e = dt.ensemble_fit(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0])
    assert isinstance(e, EnsembleResult) and isinstance(e, FittingResult)
    assert set(e.params) == {"a", "b"}
    assert e.cov is not None and e.cov.shape == (2, 2)
    # spread populates the covariance diagonal -> stderr equals the spread
    np.testing.assert_allclose(list(e.stderr().values()), e.spread, rtol=1e-9)
    pred = e.predict(t)
    assert pred.shape == t.shape and np.all(np.isfinite(pred))


def test_ensemble_lsi_method_and_mean_aggregate():
    t, y, (a, b) = _exp_data(1)
    e = dt.ensemble_fit(t, y, "a*exp(b*t)", "t", method="lsi",
                        aggregate="mean", p0=[1.0, 1.0])
    assert abs(e.coeffs[1] - b) < 0.25


def test_ensemble_rejects_bad_args():
    t, y, _ = _exp_data()
    with pytest.raises(ValueError, match="method"):
        dt.ensemble_fit(t, y, "a*exp(b*t)", "t", method="nope")
    with pytest.raises(ValueError, match="aggregate"):
        dt.ensemble_fit(t, y, "a*exp(b*t)", "t", aggregate="nope")


@pytest.mark.parametrize("overlap", [-0.1, 0.95, 1.0, 1.5])
def test_ensemble_rejects_bad_overlap(overlap):
    t, y, _ = _exp_data()
    with pytest.raises(ValueError, match="overlap"):
        dt.ensemble_fit(t, y, "a*exp(b*t)", "t", overlap=overlap)


def test_ensemble_counts_failed_windows_and_warns():
    """A per-window failure is counted, its error kept, and a UserWarning
    raised. None of it passes silently."""
    t, y, _ = _exp_data()
    # NaN-poison a stretch of samples. Subwindows covering it raise under the
    # default nan_policy="raise"; the later ones stay clean and fit.
    y = y.copy()
    y[100:140] = np.nan
    with pytest.warns(UserWarning, match="subwindow fits"):
        e = dt.ensemble_fit(t, y, "a*exp(b*t)", "t", n_windows=6,
                            p0=[1.0, 1.0])
    assert e.n_failed > 0
    assert e.last_error is not None and "non-finite" in e.last_error
    assert e.members.shape[0] >= 1
    assert abs(e.coeffs[1] - 0.9) < 0.2  # surviving windows still recover


def test_ensemble_clean_fit_has_no_failures():
    t, y, _ = _exp_data()
    e = dt.ensemble_fit(t, y, "a*exp(b*t)", "t", n_windows=6, p0=[1.0, 1.0])
    assert e.n_failed == 0
    assert e.last_error is None


def test_ensemble_recovers_under_outliers():
    """Median aggregation recovers the parameters under spike contamination.
    The head-to-head advantage over a plain fit is gated per family in
    tests/validation/test_outlier_robustness, not here."""
    truth = np.array([1.0, 0.9])
    err = lambda c: float(np.max(np.abs(c - truth) / truth))  # noqa: E731
    errs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 3, 400)
        clean = 1.0 * np.exp(0.9 * t)
        sig = float(clean.std())
        y = clean + rng.normal(0, 0.02 * sig, t.size)
        idx = rng.choice(t.size, 12, replace=False)
        y[idx] += rng.normal(0, 6 * sig, 12)
        errs.append(err(dt.ensemble_fit(t, y, "a*exp(b*t)", "t", p0=[1.0, 1.0]).coeffs))
    assert np.median(errs) < 0.2
