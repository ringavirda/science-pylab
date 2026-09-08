"""Validate the dtfit methods against the downloaded real datasets.

Run experiments/download_data.py first, then:

    python -m dtfit_experimental.experiments.validate_methods

Four experiments:
  * COVID-19 (Ukraine), early exponential growth phase: batch fit of
    y = a*exp(b*t) with LSI and EAC next to a SciPy curve_fit baseline,
    trained on the first 80% and forecasting the last 20%;
  * the same COVID window through NonlineRegressor, the scikit-learn
    interface, scored in sample and by 4-fold cross-validation;
  * USD/UAH (2014-2015 hryvnia crisis): batch fit of the exponential
    depreciation trend with LSI and EAC;
  * the same rate series through EACFilter, streaming one-step-ahead
    forecasting against the naive "tomorrow = today" baseline.

Real downloaded data throughout, no synthetic signals.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold, cross_val_score

import dtfit as dt
from dtfit.sklearn import NonlineRegressor
from dtfit.streaming import EACFilter

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_csv(name: str) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader((DATA_DIR / name).open()))[1:]
    dates = [r[0] for r in rows]
    values = np.array([float(r[1]) for r in rows])
    return dates, values


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"R2": r2, "RMSE": rmse, "MAPE%": mape}


def fmt(m: dict[str, float]) -> str:
    return f"R2={m['R2']:+.4f}  RMSE={m['RMSE']:.4g}  MAPE={m['MAPE%']:.2f}%"


def rule(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def experiment_covid() -> None:
    rule("COVID-19 Ukraine -- exponential growth  y = a*exp(b*t)  (batch + forecast)")
    dates, cum = load_csv("covid_ukraine_confirmed.csv")

    # Take the clean early-growth window: a 4-week stretch of the take-off
    # spanning roughly 16x. A pure exponential is well-conditioned over that.
    # LSI and EAC both lose accuracy once the dynamic range grows well past it;
    # for LSI it is the empirical polynomial spectrum that goes ill-conditioned
    # across the wider range.
    length = 28
    start = next(i for i, v in enumerate(cum) if v >= 500)
    win = slice(start, start + length)
    y = cum[win]
    print(f"window: {dates[start]} .. {dates[start + length - 1]}  "
          f"({y[0]:.0f} -> {y[-1]:.0f} cases, {y[-1] / y[0]:.0f}x)")

    # Normalize the domain to [0, ~1.5] and the case counts to O(1). Both
    # rescalings are invertible, and they leave the exponential's parameters at
    # a scale the fit conditions well.
    n = y.size
    t = np.linspace(0, 1.5, n)
    y_scale = y[0]
    ys = y / y_scale

    n_train = int(n * 0.8)
    t_tr, ys_tr = t[:n_train], ys[:n_train]

    bounds = [(0.2, 5.0), (0.5, 4.0)]  # a, b  (growth)
    results = {}
    res_lsi = dt.fit_lsi(t_tr, ys_tr, "a*exp(b*x)", "x", bounds=bounds)
    res_eac = dt.fit_eac(t_tr, ys_tr, "a*exp(b*x)", "x", p0=[ys_tr[0], 1.0])
    results["LSI"] = res_lsi.coeffs
    results["EAC"] = res_eac.coeffs

    # SciPy baseline: unbounded curve_fit, i.e. Levenberg-Marquardt NLLS.
    p_sci, _ = curve_fit(
        lambda x, a, b: a * np.exp(b * x), t_tr, ys_tr, p0=[ys_tr[0], 1.0], maxfev=10000
    )
    results["SciPy curve_fit"] = p_sci

    print(f"\n{'method':18s} {'a':>10s} {'b':>10s}   "
          f"{'train (fit)':^34s} | {'forecast (last 20%)':^34s}")
    for name, (a, b) in results.items():
        pred = a * np.exp(b * t) * y_scale  # back to real case counts
        m_tr = metrics(y[:n_train], pred[:n_train])
        m_fc = metrics(y[n_train:], pred[n_train:])
        print(f"{name:18s} {a:10.4f} {b:10.4f}   {fmt(m_tr):^34s} | {fmt(m_fc):^34s}")


def experiment_covid_sklearn() -> None:
    rule("COVID-19 Ukraine -- NonlineRegressor (scikit-learn fit/score + CV)")
    _, cum = load_csv("covid_ukraine_confirmed.csv")
    length = 28
    start = next(i for i, v in enumerate(cum) if v >= 500)
    y = cum[start : start + length]
    t = np.linspace(0, 1.5, y.size)
    ys = y / y[0]

    reg = NonlineRegressor("a*exp(b*x)", "x", basis="block", p0=[1.0, 1.0])
    reg.fit(t, ys)
    print(f"  fitted coef_ (a, b): {reg.coef_}")
    print(f"  in-sample R^2      : {reg.score(t, ys):.4f}")
    cv = cross_val_score(reg, t.reshape(-1, 1), ys, cv=KFold(4, shuffle=True, random_state=0))
    print(f"  4-fold CV R^2      : {cv.round(4)}  (mean {cv.mean():.4f})")


def experiment_currency_batch() -> None:
    rule("USD/UAH 2014-2015 -- exponential depreciation trend  (batch fit)")
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    t = np.linspace(0, 1.5, n)
    r0 = rate[0]
    rs = rate / r0
    print(f"window: {dates[0]} .. {dates[-1]}  ({rate[0]:.3f} -> {rate[-1]:.3f} UAH/USD)")

    bounds = [(0.5, 3.0), (0.1, 4.0)]
    res_lsi = dt.fit_lsi(t, rs, "a*exp(b*x)", "x", bounds=bounds)
    res_eac = dt.fit_eac(t, rs, "a*exp(b*x)", "x", p0=[1.0, 1.0])

    print(f"\n{'method':18s} {'a':>10s} {'b':>10s}   {'fit over full window':^34s}")
    for name, coeffs in [("LSI", res_lsi.coeffs), ("EAC", res_eac.coeffs)]:
        a, b = coeffs
        pred = a * np.exp(b * t) * r0
        print(f"{name:18s} {a:10.4f} {b:10.4f}   {fmt(metrics(rate, pred)):^34s}")


def experiment_currency_streaming() -> None:
    rule("USD/UAH 2014-2015 -- EACFilter online tracking (bounded cost)")
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    window = 30
    # A per-sample step h keeps the local exponential model well-conditioned.
    # t stays absolute, letting the filter track parameter drift across the
    # window.
    h = 1.5 / n
    t = np.arange(n) * h
    r0 = rate[0]
    rs = rate / r0  # scale to O(1)

    flt = EACFilter(
        "a*exp(b*x)", "x", p0=[1.0, 0.0],
        window_size=window, q_diag=[5e-3, 5e-3],
    )

    track, truth = [], []
    drift_events = []  # (date, direction) for each detected regime shift
    for i in range(n):
        flt.partial_fit(t[i], rs[i])
        if flt.drift_flag_:
            drift_events.append((dates[i], "up" if flt.last_drift_direction_ > 0 else "down"))
        if len(flt._t) > 0:
            # online tracked estimate of today's level (one-sample lag)
            track.append(float(flt.predict(np.array([t[i]]))[0]) * r0)
            truth.append(rate[i])

    track = np.array(track)
    truth = np.array(truth)

    # The random walk ("tomorrow = today") is the standard and famously hard FX
    # benchmark. The filter's job is bounded-cost online tracking and drift
    # flagging, not beating the walk one step ahead; both numbers print either
    # way.
    one_step_pred = track[:-1]
    one_step_truth = truth[1:]
    naive = truth[:-1]

    print(f"online tracked points: {track.size}  (window {window})")
    print(f"  tracking fit (lag-1) : {fmt(metrics(truth, track))}")
    print(f"  1-step-ahead filter  : {fmt(metrics(one_step_truth, one_step_pred))}")
    print(f"  1-step-ahead naive RW: {fmt(metrics(one_step_truth, naive))}")
    print(f"  drift events (both directions) : {len(drift_events)}")
    for d, direction in drift_events:
        print(f"      {d}  shift {direction}")
    print(f"  final tracked params : {flt.params_}")


def main() -> None:
    if not (DATA_DIR / "covid_ukraine_confirmed.csv").exists():
        raise SystemExit("Datasets missing -- run: python -m dtfit_experimental.experiments.download_data")
    experiment_covid()
    experiment_covid_sklearn()
    experiment_currency_batch()
    experiment_currency_streaming()
    print()


if __name__ == "__main__":
    main()
