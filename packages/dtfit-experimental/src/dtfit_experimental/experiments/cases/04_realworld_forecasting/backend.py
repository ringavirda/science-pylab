"""Loaders, forecasters and holdout scoring for the forecasting case study.

``04_realworld_forecasting.ipynb`` imports this module and owns the
presentation.

Four real series were picked for distinct structure: exponential growth
(COVID-19 Ukraine), exponential depreciation (USD/UAH), an eleven-year cycle
(sunspots) and trend plus seasonality (Mauna Loa CO2). Each gets a dtfit model
matched to that structure, fitted on the first 80% and extrapolated across the
last 20%, against the forecasters a practitioner would actually reach for on
the same data: ARIMA, a scikit-learn MLP, a PyTorch LSTM and the random walk.

:data:`DATASETS` binds each series to its loader, its dtfit model and its ARIMA
order. :func:`run_one` does the split, the fits and the scoring, returning
plain numbers and arrays.

dtfit forecasts by fitting a parametric form and extrapolating it. That wins
wherever a series carries clear nonlinear structure to extrapolate, and loses
to the general learners wherever the dynamics are irregular. Both outcomes
turn up in the results.
"""

from __future__ import annotations

import numpy as np

import dtfit as dt
from dtfit_experimental import fit_lsi_basis, boosted_fit

from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "load_covid", "load_uah", "load_sunspots", "load_co2",
    "dtfit_exp", "dtfit_sunspots", "dtfit_co2",
    "DATASETS", "run_one",
]


# Loaders: each returns one 1-D series.
def load_covid():
    import csv
    p = EXPERIMENTS_DIR / "data" / "covid_ukraine_confirmed.csv"
    rows = list(csv.reader(p.open()))[1:]
    cum = np.array([float(r[1]) for r in rows])
    start = next(i for i, v in enumerate(cum) if v >= 500)
    return cum[start:start + 28]  # clean exponential take-off window


def load_uah():
    import csv
    p = EXPERIMENTS_DIR / "data" / "usd_uah_2014_2015.csv"
    rows = list(csv.reader(p.open()))[1:]
    return np.array([float(r[1]) for r in rows])


def load_sunspots():
    import statsmodels.api as sm
    return sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy(float)


def load_co2():
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"]
    s = s.interpolate().bfill().ffill()
    return s.to_numpy(float)[::4]  # thin weekly->~monthly for a shorter series


# Forecasters: fit on the train split, predict over the full t.
def dtfit_exp(t_tr, y_tr, t_all):
    y0 = y_tr[0]
    r = dt.fit_lsi(t_tr, y_tr / y0, "a*exp(b*x)", "x", bounds=[(0.1, 5), (0.05, 5)])
    return np.asarray(r.model(t_all)) * y0, "LSI exp"


def dtfit_sunspots(t_tr, y_tr, t_all):
    # Adaptation #2: the Fourier basis expresses the periodic form directly.
    expr = "c + A*sin(w*x + p)"
    r = fit_lsi_basis(t_tr, y_tr, expr, "x", basis="fourier", order=8,
                      bounds=[(10, 120), (0, 200), (0.1, 1.5), (-np.pi, np.pi)])
    return np.asarray(r.model(t_all)), "Fourier-LSI (#2)"


def dtfit_co2(t_tr, y_tr, t_all):
    # Adaptation #5: stage-wise boosting fits the trend, then the seasonal
    # term on what the trend left behind.
    bm = boosted_fit(t_tr, y_tr, [
        dict(expr="a0 + a1*x + a2*x**2", var="x", method="lsi",
             p0=[y_tr[0], 1.0, 0.0]),
        dict(expr="A*sin(w*x + p)", var="x", method="lsi",
             bounds=[(0.1, 20), (0.1, 60), (-np.pi, np.pi)]),
    ])
    return bm.predict(t_all), "boosted LSI (#5)"


DATASETS = {
    "COVID-19 UA (exp growth)": (load_covid, dtfit_exp, dict(order=(2, 2, 2))),
    "USD/UAH (exp depreciation)": (load_uah, dtfit_exp, dict(order=(2, 1, 2))),
    "Sunspots (~11y cycle)": (load_sunspots, dtfit_sunspots, dict(order=(3, 0, 3))),
    "Mauna Loa CO2 (trend+season)": (load_co2, dtfit_co2, dict(order=(2, 1, 2))),
}


def run_one(name, loader, dtfit_fn, arima_kw, *, quick=True):
    """Fit every method on the 80% train split and forecast the 20% holdout.

    Returns a dict with the series ``y``, the time axis ``t``, the split index
    ``n_tr``, the per-method holdout ``preds``, the per-method ``scores``
    (R2/RMSE/MAE/MAPE) and the ``dtfit_label`` naming the parametric form used.

    ``quick`` trims the heavy learners so the whole suite runs in a couple of
    minutes: the MLP gets fewer iterations and the slow PyTorch LSTM is left
    out entirely. ``quick=False`` runs the MLP to convergence and adds the
    LSTM. A baseline whose optional dependency (statsmodels, sklearn, torch)
    is missing forecasts NaN rather than raising.
    """
    y = loader()
    n = y.size
    n_tr = int(n * 0.8)
    h = n - n_tr
    t = np.linspace(0, 1.5, n)
    t_tr, y_tr = t[:n_tr], y[:n_tr]

    preds = {}
    try:
        full, label = dtfit_fn(t_tr, y_tr, t)
        preds[label] = full[n_tr:]
    except Exception:
        preds["dtfit (failed)"] = np.full(h, np.nan)
        label = "dtfit (failed)"
    # The baselines forecast the next h points from the training array alone.
    try:
        preds["ARIMA"] = bl.arima_forecast(y_tr, h, order=arima_kw["order"])
    except Exception:
        preds["ARIMA"] = np.full(h, np.nan)
    try:
        preds["MLP"] = bl.mlp_forecast(y_tr, h, lookback=min(24, n_tr // 3),
                                       max_iter=600 if quick else 1500)
    except Exception:
        preds["MLP"] = np.full(h, np.nan)
    if not quick:
        try:
            preds["LSTM"] = bl.lstm_forecast(y_tr, h, lookback=min(24, n_tr // 3),
                                             epochs=150)
        except Exception:
            preds["LSTM"] = np.full(h, np.nan)
    preds["random walk"] = bl.random_walk_forecast(y_tr, h)

    y_te = y[n_tr:]
    scores = {m: metrics(y_te, p) for m, p in preds.items()}
    return dict(name=name, y=y, t=t, n_tr=n_tr, preds=preds, scores=scores,
                dtfit_label=label)
