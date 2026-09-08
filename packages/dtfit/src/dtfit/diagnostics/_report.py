"""Fit-aware diagnostics for a :class:`dtfit.FittingResult`.

The generic ``(y_true, y_pred)`` scalar metrics in ``sklearn.metrics`` and
``scipy.stats`` cover plain numbers. These take the :class:`FittingResult`
itself, and can therefore report parameter uncertainty, information criteria
for comparing candidate models, and whether the residuals still carry
structure the model missed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dtfit._stats import information_criteria


def _basic_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """``r2`` and ``rmse`` of a prediction (used for plot annotations)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    resid = y_true - y_pred
    rss = float(resid @ resid)
    tss = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt(rss / y_true.size)),
        "r2": float(1.0 - rss / tss) if tss > 0 else float("nan"),
    }


def fit_report(result: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Goodness-of-fit and parsimony report for a fitted model on ``(x, y)``.

    The returned dict holds the sample and parameter counts, ``rss``,
    ``rmse`` and ``r2``, the Gaussian-likelihood ``aic`` and ``bic`` for
    comparing candidates fitted to the same data, and ``durbin_watson``
    (about 2 means no residual autocorrelation). ``converged`` appears when
    the result reports it, and ``params`` with ``stderr`` when the fit
    carries a covariance. To select a model, fit several candidates and keep
    the lowest information criterion.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(result.predict(x), dtype=float).ravel()
    resid = y - yhat
    n = y.size
    k = int(np.asarray(result.coeffs).size)
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(rss / n))
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    aic, bic = information_criteria(rss, n, k)
    dw = float(np.sum(np.diff(resid) ** 2) / rss) if rss > 0 else float("nan")
    report: dict[str, Any] = {
        "n": n, "n_params": k, "rss": rss, "rmse": rmse, "r2": r2,
        "aic": aic, "bic": bic, "durbin_watson": dw,
    }
    converged = getattr(result, "converged", None)
    if converged is not None:
        report["converged"] = bool(converged)
    if getattr(result, "cov", None) is not None:
        report["params"] = result.params
        report["stderr"] = result.stderr()
    return report


def residual_diagnostics(result: Any, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Tests for structure the model left behind in its residuals.

    A structured (DT) fit should leave white-noise residuals. Leftover
    autocorrelation means the model class is wrong, the usual case being a
    trend with no cycle fitted to a seasonal series. The returned dict holds
    the residuals with their ``mean`` and ``std``, the Durbin-Watson and
    lag-1 autocorrelation statistics, and a Shapiro-Wilk normality p-value.
    That p-value is NaN unless ``3 <= n <= 5000``, the range where the test
    applies.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    return residual_stats(
        y - np.asarray(result.predict(x), dtype=float).ravel()
    )


def residual_stats(resid: np.ndarray) -> dict[str, Any]:
    """The residual-structure statistics of :func:`residual_diagnostics`,
    from the residuals alone.

    Args:
        resid: Residuals ``y - f(x)``, a 1-D float array. If already a
            1-D float array, ``resid`` and the returned ``"residuals"``
            share memory, so mutating one mutates the other.

    Returns:
        The dict :func:`residual_diagnostics` returns: ``residuals``,
        ``durbin_watson``, ``lag1_autocorr``, ``normality_p``, ``mean``
        and ``std``. ``durbin_watson`` is NaN only for a zero residual
        (rss == 0); ``lag1_autocorr`` is NaN for a constant residual or
        fewer than three samples; ``normality_p`` is NaN unless
        ``3 <= n <= 5000``.

    Warns:
        RuntimeWarning: numpy's "Mean of empty slice" for an empty
            ``resid`` (``mean`` and ``std`` come back NaN).
        UserWarning: scipy's "Input data has range zero" for a constant
            residual in the Shapiro-Wilk range (``normality_p`` comes
            back 1.0).
    """
    resid = np.asarray(resid, dtype=float).ravel()
    n = resid.size
    rss = float(resid @ resid)
    dw = float(np.sum(np.diff(resid) ** 2) / rss) if rss > 0 else float("nan")
    lag1 = (float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
            if n > 2 and resid.std() > 0 else float("nan"))
    p_norm = float("nan")
    if 3 <= n <= 5000:
        try:
            from scipy.stats import shapiro

            p_norm = float(shapiro(resid).pvalue)
        except Exception:
            pass
    return {
        "residuals": resid,
        "durbin_watson": dw,
        "lag1_autocorr": lag1,
        "normality_p": p_norm,
        "mean": float(resid.mean()),
        "std": float(resid.std()),
    }
