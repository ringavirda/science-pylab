"""Forecasters and benchmark protocol for the LTSF case study.

``06_benchmark_ltsf.ipynb`` imports this module and owns the presentation.

The point of this case is to put dtfit on the same long-term-forecasting
benchmark that DLinear (arXiv:2205.13504), TimesNet (arXiv:2210.02186) and
Time-LLM (arXiv:2310.01728) report on, with nothing bent in dtfit's favour: the
splits, train-fit z-score normalization, lookback-to-horizon windows and
MSE/MAE on the normalized values. Those papers' published MSE numbers are
transcribed rather than re-run, since a re-implementation of someone else's
model is a worse comparison than the number they stand behind. The data comes
through the shared :mod:`...common.datasets` loader, which reproduces the
Informer/Autoformer pipeline all three papers use.

:func:`series_extrapolate` holds three fit-then-extrapolate forecasters, each
anchored NLinear-style to the last observation: a Legendre trend alone, a
Fourier continuation alone, and a DLinear-style decomposition into trend plus a
data-driven seasonal term. :func:`evaluate` scores one of them over a
dataset's test windows; :func:`run_dataset` and :func:`seasonal_helped` batch
that up for the notebook.
"""

from __future__ import annotations

import numpy as np

from dtfit_experimental.experiments.common import datasets as ds

__all__ = [
    "PUBLISHED_MSE", "HORIZONS", "LOOKBACK", "DAMP", "SEASONAL_FRAC",
    "available", "series_extrapolate", "evaluate", "run_dataset",
    "seasonal_helped", "sample_window",
]

# Published multivariate MSE on the LTSF benchmark, transcribed from the
# papers' ar5iv editions. Horizons 96/192/336/720 across each row.
#   DLinear  : arXiv:2205.13504, table 2. Its paper reports a longer lookback
#              than the 96 the other two use, so that row is not strictly
#              like-for-like with the rest.
#   TimesNet : arXiv:2210.02186, lookback 96.
#   Time-LLM : arXiv:2310.01728, lookback 96.
PUBLISHED_MSE = {
    "ETTh1": {
        "DLinear": [0.375, 0.405, 0.439, 0.472],
        "TimesNet": [0.384, 0.436, 0.491, 0.521],
        "Time-LLM": [0.362, 0.398, 0.430, 0.442],
    },
    "ETTm1": {
        "DLinear": [0.299, 0.335, 0.369, 0.425],
        "TimesNet": [0.338, 0.374, 0.410, 0.478],
        "Time-LLM": [0.272, 0.310, 0.352, 0.383],
    },
    "weather": {
        "DLinear": [0.176, 0.220, 0.265, 0.323],
        "TimesNet": [0.172, 0.219, 0.280, 0.365],
        "Time-LLM": [0.147, 0.189, 0.262, 0.304],
    },
}
HORIZONS = [96, 192, 336, 720]
LOOKBACK = 96
DAMP = 0.95  # Damped-trend factor; saturates long-horizon trend growth.
SEASONAL_FRAC = 0.15  # Smallest energy share a harmonic may be kept on.


def available() -> list[str]:
    """LTSF dataset keys whose CSV is present locally; the loader's list."""
    return ds.available()


def series_extrapolate(look, H, order, basis="trend_seasonal", n_harm=3):
    """Forecast H steps for each channel of the ``(L, C)`` lookback.

    All three forecasters fit then extrapolate, and all three are anchored to
    the last observation NLinear-style so the forecast starts where the series
    left off.

    ``"legendre"`` restores a low-order Legendre trend, the LSI empirical
    spectrum, and nothing else; it is the baseline the other two have to beat.
    ``"fourier"`` instead continues the lookback's low-frequency harmonics
    forward (adaptation #2).
    ``"trend_seasonal"`` decomposes DLinear-style: the same Legendre trend plus
    a data-driven Fourier seasonal term fitted on the detrended residual and
    extended periodically, so the restorable structure is modelled whole
    rather than the trend alone.
    """
    L, C = look.shape
    anchor = look[-1:]            # (1, C) last observed value per channel.
    u = np.linspace(-1.0, 1.0, L)
    step = (u[-1] - u[0]) / (L - 1)
    u_fut = u[-1] + step * np.arange(1, H + 1)

    if basis == "fourier":
        d = look - anchor
        K = order
        idx = np.arange(L)
        cols = [np.ones(L)]
        for k in range(1, K + 1):
            cols += [np.cos(2 * np.pi * k * idx / L), np.sin(2 * np.pi * k * idx / L)]
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), d, rcond=None)
        fut = np.arange(L, L + H)
        colsf = [np.ones(H)]
        for k in range(1, K + 1):
            colsf += [np.cos(2 * np.pi * k * fut / L), np.sin(2 * np.pi * k * fut / L)]
        return np.column_stack(colsf) @ coef + anchor

    # Low-order Legendre trend, shared by 'legendre' and 'trend_seasonal'.
    from numpy.polynomial.legendre import legvander
    d = look - anchor
    V = legvander(u, order)                          # (L, order+1)
    tcoef, *_ = np.linalg.lstsq(V, d, rcond=None)
    trend_in = V @ tcoef + anchor                    # (L, C) in-sample trend
    # A trend fitted on a short, noisy 96-point lookback diverges if it is
    # simply continued over a 720-step horizon. The fit is kept but its growth
    # damped geometrically, holding the forecast near the last observation
    # instead of letting it run away.
    last_trend = V[-1] @ tcoef                       # (C,) trend deviation at u[-1]
    raw_dev = legvander(u_fut, order) @ tcoef - last_trend  # (H, C) undamped
    h = np.arange(1, H + 1)
    sat = (1.0 - DAMP ** h) / (1.0 - DAMP)           # saturating profile
    scale = (sat / h)[:, None]                       # (H,1) ratio, 1 at h=1
    trend_fut = anchor + raw_dev * scale             # (H, C) bounded trend
    if basis == "legendre":
        return trend_fut

    # trend_seasonal: a Fourier seasonal term on the detrended residual.
    resid = look - trend_in                          # (L, C)
    F = np.fft.rfft(resid, axis=0)                    # (L//2+1, C)
    freqs = np.fft.rfftfreq(L)                        # cycles / sample
    mag = np.abs(F); mag[0] = 0.0                     # DC lives in the trend
    fut = np.arange(L, L + H)
    seasonal_fut = np.zeros((H, C))
    seasonal_last = np.zeros(C)
    power = (mag ** 2)                                # spectral energy per bin
    total = power[1:].sum(axis=0) + 1e-12             # residual energy per channel
    for c in range(C):
        # Only a dominant spectral peak is worth continuing. A period pinned
        # from a 96-point lookback drifts out of phase once extrapolated over
        # a long horizon, and a weak peak drifts far enough to do damage, so
        # the energy-fraction gate keeps the clean strong cycles alone. An
        # aperiodic channel passes nothing and falls back to the trend.
        frac = power[1:, c] / total[c]                # energy share per non-DC bin
        cand = 1 + np.where(frac > SEASONAL_FRAC)[0]
        if cand.size == 0:
            continue
        keep = cand[np.argsort(mag[cand, c])[-n_harm:]]
        for k in keep:
            amp = 2.0 * np.abs(F[k, c]) / L
            ph = np.angle(F[k, c])
            seasonal_fut[:, c] += amp * np.cos(2 * np.pi * freqs[k] * fut + ph)
            seasonal_last[c] += amp * np.cos(2 * np.pi * freqs[k] * (L - 1) + ph)
    # Anchor the composite to the last observation. The double anchor cancels
    # algebraically: forecast = last obs + trend delta + seasonal delta.
    return trend_fut + seasonal_fut + (anchor[0] - trend_in[-1] - seasonal_last)


def evaluate(name, H, order, basis, max_windows, n_harm=3):
    """Mean (MSE, MAE) of a forecaster over the test windows of one dataset.

    Follows the LTSF protocol: train-fit z-score normalized windows, MSE and
    MAE taken on the normalized values, averaged over up to ``max_windows``
    test windows.
    """
    se = ae = cnt = 0
    for look, target in ds.test_windows(name, LOOKBACK, H, max_windows=max_windows):
        pred = series_extrapolate(look, H, order, basis, n_harm)
        se += float(np.mean((pred - target) ** 2))
        ae += float(np.mean(np.abs(pred - target)))
        cnt += 1
    return (se / cnt, ae / cnt) if cnt else (np.nan, np.nan)


def run_dataset(name, horizons, order, max_windows, n_harm=3):
    """Measure both dtfit forecasters on one dataset across ``horizons``.

    Returns ``(trend_mse, ts_mse, ts_mae)``, three ``{horizon: value}`` dicts
    holding the trend-only MSE, the trend-plus-seasonal MSE and the
    trend-plus-seasonal MAE.
    """
    trend_mse, ts_mse, ts_mae = {}, {}, {}
    for H in horizons:
        mse_tr, _ = evaluate(name, H, order, "legendre", max_windows)
        mse_ts, mae_ts = evaluate(name, H, order, "trend_seasonal", max_windows, n_harm)
        trend_mse[H] = mse_tr
        ts_mse[H] = mse_ts
        ts_mae[H] = mae_ts
    return trend_mse, ts_mse, ts_mae


def seasonal_helped(trend_mse, ts_mse, h0):
    """Split datasets by whether the seasonal term helped at horizon ``h0``.

    ``trend_mse`` and ``ts_mse`` are ``{name: {horizon: mse}}``. Returns
    ``(helped, hurt)``: the dataset names where trend-plus-seasonal beat
    trend-only at ``h0``, and the names where it did not.
    """
    helped, hurt = [], []
    for name in ts_mse:
        tr, ts = trend_mse[name].get(h0), ts_mse[name].get(h0)
        if tr and ts and np.isfinite(tr) and np.isfinite(ts) and tr > 0:
            (helped if ts < tr else hurt).append(name)
    return helped, hurt


def sample_window(name, H, order, n_harm=3):
    """One sample lookback/target window plus both forecasts, for plotting.

    Returns ``(look, target, pred_trend, pred_trend_seasonal)``, all ``(*, C)``
    arrays, for the first test window of ``name`` at horizon ``H``.
    """
    wins = list(ds.test_windows(name, LOOKBACK, H, max_windows=1))
    look, target = wins[0]
    pred_tr = series_extrapolate(look, H, order, "legendre")
    pred_ts = series_extrapolate(look, H, order, "trend_seasonal", n_harm)
    return look, target, pred_tr, pred_ts
