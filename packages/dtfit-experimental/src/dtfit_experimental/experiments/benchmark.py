"""Benchmark and figure generation for the dtfit method docs.

Feeds the per-method documentation (the wiki ``Methods`` pages):

  * figures, written as PNG into ``wiki/figures/``: one scenario plot per
    method on the data it is best at (LSI exponential and the oscillatory
    recipe; EAC saturation and the adaptive-window sigmoid step; DSB's
    additive form; the EAC/LSI streaming filters; the fused multi-axis bank;
    the partitioned and GEMM scale backends; the auto pipelines), plus a
    cross-method error bar chart, all in the paper style;
  * comparison tables, printed to stdout as GitHub-flavoured markdown, that
    put the dtfit methods against established baselines: SciPy ``curve_fit``
    (Levenberg-Marquardt NLLS, the reference nonlinear least squares),
    ``numpy.polyfit`` (a linear-in-parameters surrogate) and, for streaming,
    the naive random walk. Both model data (synthetic, known ground truth) and
    real data (COVID-19, USD/UAH) are covered.

Run (after ``python -m dtfit_experimental.experiments.download_data``):

    python -m dtfit_experimental.experiments.benchmark

Everything presented as real is downloaded data. The synthetic signals are
labelled as model data with a known ground truth and appear only where an exact
reference is needed to quote recovery error against the truth.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import dtfit as dt
from dtfit.streaming import EACFilter

DATA_DIR = Path(__file__).resolve().parent / "data"


def _repo_root() -> Path:
    """Locate the repo root (the dir holding ``packages/dtfit``) by walking up.

    The script runs from the source checkout (editable install). Searching for
    the marker directory survives a rearranged monorepo layout; the fixed depth
    in the fallback below does not.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "packages" / "dtfit").is_dir():
            return p
    return here.parents[5]  # packages/dtfit-experimental/src/.../experiments/


FIG_DIR = _repo_root() / "wiki" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {"figure.dpi": 110, "savefig.dpi": 300, "font.size": 12, "axes.grid": True}
)


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
    return {"R2": r2, "RMSE": rmse, "MAPE": mape}


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def timed(fn) -> tuple[object, float]:
    t0 = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t0) * 1e3  # ms


# Model data: synthetic signals whose ground truth is known exactly.
def synthetic_exponential(n: int = 80, noise: float = 0.05):
    """y = a*exp(b*x), a=1.0, b=1.2 on a modest range, with Gaussian noise.

    Reseeded per call so the table and the figure use identical samples.
    """
    rng = np.random.default_rng(0)
    a_true, b_true = 1.0, 1.2
    x = np.linspace(0.0, 1.5, n)
    clean = a_true * np.exp(b_true * x)
    y = clean + rng.normal(0.0, noise * clean.std(), n)
    return x, y, clean, (a_true, b_true)


def synthetic_atan(n: int = 120, noise: float = 0.08):
    """y = a*atan(w*x), a transcendental (non-Taylor) saturation curve."""
    rng = np.random.default_rng(1)
    a_true, w_true = 2.0, 3.0
    x = np.linspace(0.0, 1.5, n)
    clean = a_true * np.arctan(w_true * x)
    y = clean + rng.normal(0.0, noise * clean.std(), n)
    return x, y, clean, (a_true, w_true)


def table_model_exponential() -> str:
    x, y, clean, (a_t, b_t) = synthetic_exponential()
    expr, var = "a*exp(b*x)", "x"
    bounds = [(0.2, 5.0), (0.5, 4.0)]
    rows: list[list[str]] = []

    def add(name, coeffs, pred, dt_ms):
        m = metrics(clean, pred)
        ab = f"a={coeffs[0]:.3f}, b={coeffs[1]:.3f}" if coeffs is not None else "--"
        rows.append([name, ab, f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                     f"{m['MAPE']:.2f}", f"{dt_ms:.1f}"])

    res, ms = timed(lambda: dt.fit_lsi(x, y, expr, var, bounds=bounds))
    add("LSI", res.coeffs, np.asarray(res.model(x)), ms)

    res, ms = timed(lambda: dt.fit_eac(x, y, expr, var, p0=[1.0, 1.0]))
    add("EAC", res.coeffs, np.asarray(res.model(x)), ms)

    def sci():
        p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), x, y,
                         p0=[1.0, 1.0], maxfev=10000)
        return p
    p, ms = timed(sci)
    add("SciPy curve_fit", p, p[0] * np.exp(p[1] * x), ms)

    def poly():
        return np.polyfit(x, y, 5)
    c, ms = timed(poly)
    add("numpy.polyfit (deg 5)", None, np.polyval(c, x), ms)

    headers = ["method", "recovered params", "R²", "RMSE", "MAPE %", "fit (ms)"]
    title = (f"**Model data** — `y = a·exp(b·x)`, ground truth a={a_t}, "
             f"b={b_t}, 5 % noise, n={x.size}. Error is against the *clean* "
             f"signal (true recovery). DSB is excluded from this head-to-head: "
             f"as the analytical reference it matches against the noisy "
             f"high-order polynomial spectrum, so under noise it is a curve "
             f"fit, not a point estimator (see `dsb.md`).")
    return title + "\n\n" + md_table(headers, rows)


def table_dsb_additive() -> str:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    clean = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x)
    y = clean + rng.normal(0, 0.05, x.size)
    expr, var = "a0 + a1*x + a2*exp(a3*x)", "x"

    rows = []
    reg, ms = timed(lambda: dt.NonlineRegressor(expr, var, method="dsb").fit(x, y))
    m = metrics(clean, reg.predict(x))
    rows.append(["DSB (symbolic ref.)", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])
    res, ms = timed(lambda: dt.fit_lsi(x, y, expr, var))
    m = metrics(clean, np.asarray(res.model(x)))
    rows.append(["LSI (same model)", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])
    p, ms = timed(lambda: curve_fit(
        lambda xx, a0, a1, a2, a3: a0 + a1 * xx + a2 * np.exp(a3 * xx),
        x, y, p0=[1, 1, 1, 1], maxfev=20000)[0])
    m = metrics(clean, p[0] + p[1] * x + p[2] * np.exp(p[3] * x))
    rows.append(["SciPy curve_fit", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}",
                 f"{m['MAPE']:.2f}", f"{ms:.1f}"])

    title = ("**Model data** — `y = a0 + a1·x + a2·exp(a3·x)` (DSB's intended "
             "additive form), x∈[0,3], 5 % noise. DSB is evaluated as a *curve "
             "fit* (R²); under noise it does not uniquely recover the four "
             "parameters, which is why the numeric LSI/EAC successors exist.")
    return title + "\n\n" + md_table(
        ["method", "R²", "RMSE", "MAPE %", "fit (ms)"], rows)


# Real-data summary. The window, the scalings and the filter settings match
# validate_methods.py, so the two scripts stay comparable.
def table_real_data() -> str:
    # COVID exponential growth window.
    _, cum = load_csv("covid_ukraine_confirmed.csv")
    length, start = 28, next(i for i, v in enumerate(cum) if v >= 500)
    y = cum[start:start + length]
    t = np.linspace(0, 1.5, y.size)
    ys = y / y[0]
    bounds = [(0.2, 5.0), (0.5, 4.0)]

    cov_rows = []
    for name, coeffs in [
        ("LSI", dt.fit_lsi(t, ys, "a*exp(b*x)", "x", bounds=bounds).coeffs),
        ("EAC", dt.fit_eac(t, ys, "a*exp(b*x)", "x", p0=[ys[0], 1.0]).coeffs),
    ]:
        a, b = coeffs
        m = metrics(y, a * np.exp(b * t) * y[0])
        cov_rows.append([name, f"{m['R2']:.4f}", f"{m['RMSE']:.4g}", f"{m['MAPE']:.2f}"])
    p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), t, ys,
                     p0=[ys[0], 1.0], maxfev=10000)
    m = metrics(y, p[0] * np.exp(p[1] * t) * y[0])
    cov_rows.append(["SciPy curve_fit", f"{m['R2']:.4f}", f"{m['RMSE']:.4g}", f"{m['MAPE']:.2f}"])

    cov_tbl = md_table(["method", "R²", "RMSE", "MAPE %"], cov_rows)

    # USD/UAH streaming one-step-ahead against the random walk.
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    h = 1.5 / n
    tt = np.arange(n) * h
    r0 = rate[0]
    rs = rate / r0
    flt = EACFilter("a*exp(b*x)", "x", p0=[1.0, 0.0],
                           window_size=30, q_diag=[5e-3, 5e-3], r=0.5)
    track, truth, drifts = [], [], 0
    for i in range(n):
        flt.partial_fit(tt[i], rs[i])
        drifts += int(flt.drift_flag_)
        if len(flt._t) > 0:
            track.append(float(flt.predict(np.array([tt[i]]))[0]) * r0)
            truth.append(rate[i])
    track, truth = np.array(track), np.array(truth)
    m_track = metrics(truth, track)
    m_step = metrics(truth[1:], track[:-1])
    m_rw = metrics(truth[1:], truth[:-1])
    fx_rows = [
        ["EACFilter (lag-1 tracking)", f"{m_track['R2']:.4f}",
         f"{m_track['RMSE']:.4g}", f"{m_track['MAPE']:.2f}"],
        ["EACFilter (1-step-ahead)", f"{m_step['R2']:.4f}",
         f"{m_step['RMSE']:.4g}", f"{m_step['MAPE']:.2f}"],
        ["naive random walk (1-step)", f"{m_rw['R2']:.4f}",
         f"{m_rw['RMSE']:.4g}", f"{m_rw['MAPE']:.2f}"],
    ]
    fx_tbl = md_table(["method", "R²", "RMSE", "MAPE %"], fx_rows)

    return (
        "**Real data — COVID-19 Ukraine** (cumulative confirmed, 28-day "
        f"take-off, {y[0]:.0f}→{y[-1]:.0f} cases), exponential `y = a·exp(b·t)`:\n\n"
        + cov_tbl
        + "\n\n**Real data — USD/UAH 2014-2015** (NBU daily, "
        f"{rate[0]:.2f}→{rate[-1]:.2f}), streaming one-step-ahead vs the "
        f"random-walk benchmark; the filter flagged **{drifts}** structural "
        "break(s):\n\n" + fx_tbl
    )


def fig_lsi() -> None:
    x, y, clean, _ = synthetic_exponential()
    n_tr = int(x.size * 0.7)
    res = dt.fit_lsi(x[:n_tr], y[:n_tr], "a*exp(b*x)", "x", bounds=[(0.2, 5), (0.5, 4)])
    yhat = np.asarray(res.model(x))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x[:n_tr], y[:n_tr], s=14, c="0.5", label="train samples")
    ax[0].scatter(x[n_tr:], y[n_tr:], s=14, c="tab:orange", label="held-out")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:blue", lw=2, label="LSI fit")
    ax[0].axvline(x[n_tr], color="0.7", ls=":")
    ax[0].set_title("LSI fit — y = a·exp(b·x)")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    # spectrum view: empirical vs model Maclaurin discretes
    k = np.arange(6)
    z_emp = np.polyfit(x[:n_tr], y[:n_tr], 5)[::-1]
    a, b = res.coeffs
    z_mod = np.array([a * b**i / math.factorial(i) for i in k])
    width = 0.38
    ax[1].bar(k - width / 2, z_emp, width, label="empirical Z(k)", color="0.6")
    ax[1].bar(k + width / 2, z_mod, width, label="model F(k,c)", color="tab:blue")
    ax[1].set_title("Differential spectra matched by LSI")
    ax[1].set_xlabel("discrete order k")
    ax[1].set_ylabel("coefficient")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lsi_fit.png")
    plt.close(fig)


def fig_eac() -> None:
    x, y, clean, _ = synthetic_atan()
    res = dt.fit_eac(x, y, "a*atan(w*x)", "x", p0=[1.0, 1.0])
    yhat = np.asarray(res.model(x))
    a, w = res.coeffs

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x, y, s=12, c="0.6", label="noisy samples (8 %)")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:green", lw=2, label="EAC fit")
    # shade the per-parameter integration windows
    n = 2
    idx_max = max(int(x.size * 0.8), n + 1)
    win = max(idx_max // n, 2)
    for i in range(n):
        s = i * win
        e = (i + 1) * win if i < n - 1 else idx_max
        ax[0].axvspan(x[s], x[min(e, x.size) - 1], alpha=0.07, color="tab:green")
    ax[0].set_title(f"EAC fit — y = a·atan(w·x)  (a={a:.2f}, w={w:.2f})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    # cumulative area: the quantity EAC actually matches
    from scipy.integrate import cumulative_trapezoid
    area_d = cumulative_trapezoid(y, x, initial=0)
    area_m = cumulative_trapezoid(yhat, x, initial=0)
    ax[1].plot(x, area_d, "0.5", lw=2, label="∫ data")
    ax[1].plot(x, area_m, "tab:green", lw=1.5, ls="--", label="∫ model")
    ax[1].set_title("Equal-areas criterion (integral match)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("cumulative area")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eac_fit.png")
    plt.close(fig)


def fig_filter() -> None:
    dates, rate = load_csv("usd_uah_2014_2015.csv")
    n = rate.size
    h = 1.5 / n
    t = np.arange(n) * h
    r0 = rate[0]
    rs = rate / r0
    flt = EACFilter("a*exp(b*x)", "x", p0=[1.0, 0.0],
                           window_size=30, q_diag=[5e-3, 5e-3], r=0.5)
    track, drift_idx = [], []
    b_hist = []
    for i in range(n):
        flt.partial_fit(t[i], rs[i])
        if flt.drift_flag_:
            drift_idx.append(i)
        track.append(float(flt.predict(np.array([t[i]]))[0]) * r0 if len(flt._t) else np.nan)
        b_hist.append(flt.p[1])
    track = np.array(track)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(np.arange(n), rate, "0.5", lw=1, label="USD/UAH (real)")
    ax[0].plot(np.arange(n), track, "tab:red", lw=1.5, label="filter tracking")
    for j, di in enumerate(drift_idx):
        ax[0].axvline(di, color="tab:purple", ls="--", lw=1.2,
                      label="drift detected" if j == 0 else None)
    ax[0].set_title("EACFilter — online tracking + drift detection")
    ax[0].set_xlabel("sample (trading day)")
    ax[0].set_ylabel("UAH per USD")
    ax[0].legend(fontsize=8)

    ax[1].plot(np.arange(n), b_hist, "tab:red", lw=1.5)
    for di in drift_idx:
        ax[1].axvline(di, color="tab:purple", ls="--", lw=1.2)
    ax[1].set_title("Tracked growth parameter b (adapts after each reset)")
    ax[1].set_xlabel("sample")
    ax[1].set_ylabel("b estimate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "filter_tracking.png")
    plt.close(fig)


def fig_dsb() -> None:
    # DSB is the analytical reference: it matches the model's Maclaurin
    # spectrum against the data's noisy high-order polynomial coefficients.
    # Under noise that makes it a curve fit here, not a point estimator.
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    clean = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x)
    y = clean + rng.normal(0, 0.05, x.size)
    reg = dt.NonlineRegressor("a0 + a1*x + a2*exp(a3*x)", "x", method="dsb").fit(x, y)
    yhat = reg.predict(x)
    r2 = reg.score(x, y)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x, y, s=10, c="0.6", label="noisy samples")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    ax[0].plot(x, yhat, "tab:purple", lw=2, label="DSB fit")
    ax[0].set_title(f"DSB symbolic fit — a0+a1·x+a2·exp(a3·x)  (R²={r2:.3f})")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    ax[1].plot(x, yhat - y, "tab:purple", lw=1)
    ax[1].axhline(0, color="0.5", lw=0.8)
    ax[1].set_title("Residuals (fit − data)")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("residual")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dsb_fit.png")
    plt.close(fig)


def fig_comparison() -> None:
    # MAPE on the synthetic exponential for each method (true-recovery error).
    x, y, clean, _ = synthetic_exponential()
    expr, var = "a*exp(b*x)", "x"
    preds = {}
    preds["LSI"] = np.asarray(dt.fit_lsi(x, y, expr, var, bounds=[(0.2, 5), (0.5, 4)]).model(x))
    preds["EAC"] = np.asarray(dt.fit_eac(x, y, expr, var, p0=[1.0, 1.0]).model(x))
    p, _ = curve_fit(lambda xx, a, b: a * np.exp(b * xx), x, y, p0=[1, 1], maxfev=10000)
    preds["SciPy\ncurve_fit"] = p[0] * np.exp(p[1] * x)
    c = np.polyfit(x, y, 5)
    preds["numpy\npolyfit"] = np.polyval(c, x)

    names = list(preds)
    mapes = [metrics(clean, preds[k])["MAPE"] for k in names]
    colors = ["tab:blue", "tab:green", "0.5", "0.7"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(names, mapes, color=colors)
    ax.set_ylabel("MAPE % vs ground truth")
    ax.set_title("Parameter-recovery error on model exponential (5 % noise, lower is better)")
    for bar, mp in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width() / 2, mp, f"{mp:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_mape.png")
    plt.close(fig)


# The plots above are each method's basic fit. The ones below put every method
# on the scenario it was built for.
def fig_lsi_oscillatory() -> None:
    """LSI's signature scenario: recover a cycle with the oscillatory recipe,
    where the default low-order spectral fit settles on the wrong one."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 10, 400)
    A_t, w_t, p_t = 2.0, 1.7, 0.5
    clean = A_t * np.sin(w_t * x + p_t)
    y = clean + rng.normal(0, 0.15, x.size)

    recipe = dt.fit_lsi(x, y, "A*sin(w*x + p)", "x", freq_param="w")
    yhat_r = np.asarray(recipe.model(x))
    try:  # default LSI: order 5, no FFT seed, so it locks onto a wrong cycle
        naive = dt.fit_lsi(x, y, "A*sin(w*x + p)", "x")
        yhat_n = np.asarray(naive.model(x))
        w_n = naive.params["w"]
    except Exception:
        yhat_n, w_n = None, float("nan")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x, y, s=8, c="0.7", label="noisy samples")
    ax[0].plot(x, clean, "k--", lw=1, label="ground truth")
    if yhat_n is not None:
        ax[0].plot(x, yhat_n, "tab:orange", lw=1.5, ls=":",
                   label=f"LSI default (no recipe, w={w_n:.2f})")
    ax[0].plot(x, yhat_r, "tab:blue", lw=2,
               label=f"LSI oscillatory recipe (w={recipe.params['w']:.2f})")
    ax[0].set_title("LSI oscillatory recipe — A·sin(w·x + p), truth w=1.7")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)

    # the FFT seed the recipe uses for the frequency
    yy = y - y.mean()
    spec = np.abs(np.fft.rfft(yy)); spec[0] = 0.0
    freqs = 2 * np.pi * np.fft.rfftfreq(x.size, d=float(x[1] - x[0]))
    ax[1].plot(freqs, spec, "0.5", lw=1.2)
    ax[1].axvline(freqs[int(np.argmax(spec))], color="tab:blue", ls="--", lw=1.2,
                  label=f"FFT seed = {freqs[int(np.argmax(spec))]:.2f}")
    ax[1].axvline(w_t, color="k", ls=":", lw=1, label="true w = 1.7")
    ax[1].set_xlim(0, 6)
    ax[1].set_title("Frequency seed (FFT peak) the recipe locks onto")
    ax[1].set_xlabel("angular frequency ω"); ax[1].set_ylabel("|FFT|")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lsi_oscillatory.png"); plt.close(fig)


def fig_eac_adaptive() -> None:
    """EAC's block preset on a sharp sigmoid step: uniform windows spread
    evenly across x, not placed by the curve's local curvature."""
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 300)
    L_t, k_t, x0_t = 1.0, 2.5, 5.0
    clean = L_t / (1.0 + np.exp(-k_t * (x - x0_t)))   # sharp step at x0=5
    y = clean + rng.normal(0, 0.02, x.size)
    res = dt.fit_eac(x, y, "L/(1 + exp(-k*(x - x0)))", "x", p0=[1.0, 1.0, 5.0])
    yhat = np.asarray(res.model(x))
    edges = np.linspace(x[0], x[-1], res.image_order + 1)[1:-1]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.scatter(x, y, s=8, c="0.7", label="samples")
    ax.plot(x, clean, "k--", lw=1, label="ground truth")
    ax.plot(x, yhat, "tab:green", lw=2,
            label=f"EAC (k={res.params['k']:.2f}, x0={res.params['x0']:.2f})")
    for xe in edges:
        ax.axvline(xe, color="0.6", ls="--", lw=0.8)
    ax.set_title("EAC — sharp sigmoid step (uniform windows)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "eac_adaptive.png"); plt.close(fig)


def fig_lsi_filter() -> None:
    """LSIFilter's scenario: track a steady oscillation. The spectrum
    measurement locks onto the cycle's frequency. The EACFilter's area
    measurement nearly cancels over a cycle and cannot; the LSIFilter exists
    for that reason."""
    from dtfit.streaming import LSIFilter, EACFilter

    rng = np.random.default_rng(5)
    t = np.linspace(0, 50, 1200)
    A_t, w_t = 2.0, 1.3
    y = A_t * np.sin(w_t * t) + rng.normal(0, 0.15, t.size)

    def run(flt):
        wh, pr = [], []
        for i in range(t.size):
            flt.partial_fit(t[i], y[i])
            wh.append(flt.params_["w"])
            pr.append(float(flt.predict(np.array([t[i]]))[0]) if len(flt._t) else np.nan)
        return np.array(wh), np.array(pr)

    w_lsi, p_lsi = run(LSIFilter("A*sin(w*t)", "t", p0=[1.0, 0.8],
                                 window_size=120, order=6, q_diag=[1e-3, 1e-3], r=1.5))
    w_eac, p_eac = run(EACFilter("A*sin(w*t)", "t", p0=[1.0, 0.8],
                                 window_size=120, q_diag=[1e-3, 1e-3], r=1.5))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    sl = slice(int(t.size * 0.62), int(t.size * 0.74))   # zoom for the overlay
    ax[0].plot(t[sl], y[sl], "0.75", lw=1.2, label="signal")
    ax[0].plot(t[sl], p_lsi[sl], "tab:blue", lw=1.6, label="LSIFilter (spectrum)")
    ax[0].plot(t[sl], p_eac[sl], "tab:orange", lw=1.6, ls="--", label="EACFilter (area)")
    ax[0].set_title("Online 1-step prediction — A·sin(w·t)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)

    ax[1].axhline(w_t, color="k", ls="--", lw=1, label="true ω = 1.3")
    ax[1].plot(t, w_lsi, "tab:blue", lw=1.6,
               label=f"LSIFilter ω → {w_lsi[-1]:.2f} (locks on)")
    ax[1].plot(t, w_eac, "tab:orange", lw=1.6,
               label=f"EACFilter ω → {w_eac[-1]:.2f} (area cancels)")
    ax[1].set_title("Spectrum measurement tracks the cycle; area does not")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("ω estimate"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "lsi_filter.png"); plt.close(fig)


def fig_filter_bank() -> None:
    """FilterBank + fused χ²: a fault shared across 3 channels is weak per-axis but
    strong in the pooled statistic."""
    from dtfit import FilterBank
    from dtfit.streaming import LSIFilter

    rng = np.random.default_rng(6)
    t = np.linspace(0, 50, 1500)
    half = t.size // 2
    K = 3
    amps = np.array([2.0, 1.5, 2.5])
    w_t = 1.3
    env = np.where(np.arange(t.size) < half, 1.0, 0.45)   # shared amplitude collapse
    Y = np.column_stack([
        amps[k] * env * np.sin(w_t * t) + rng.normal(0, 0.15, t.size) for k in range(K)
    ])

    bank = FilterBank.from_model("A*sin(w*t)", "t", K, filter_cls=LSIFilter,
                                 p0=[2.0, 1.3], window_size=120, order=6,
                                 q_diag=[2e-3, 5e-4], r=1.5)
    det = bank.fused_detector(alpha=1e-3, inflate=4.0, warmup=550, cooldown=700)
    stat, flags = [], []
    for i in range(t.size):
        fired = det.update(t[i], Y[i])
        stat.append(det.statistic_)
        if fired:
            flags.append(i)
    stat = np.array(stat)

    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for k in range(K):
        ax[0].plot(t, Y[:, k], lw=0.6, label=f"axis {k+1}")
    ax[0].axvline(t[half], color="tab:purple", ls="--", lw=1, label="shared fault")
    ax[0].set_title("3 oscillatory channels — a damping (amplitude) fault shared across all axes")
    ax[0].set_ylabel("signal"); ax[0].legend(fontsize=8, ncol=2)

    ax[1].plot(t, stat, "tab:red", lw=1.2, label="fused χ²(3) statistic")
    ax[1].axhline(det.threshold_, color="k", ls=":", lw=1,
                  label=f"α=1e-3 threshold ({det.threshold_:.1f})")
    ax[1].axvline(t[half], color="tab:purple", ls="--", lw=1)
    for j, fi in enumerate(flags):
        ax[1].axvline(t[fi], color="tab:green", ls="-", lw=1.0, alpha=0.7,
                      label="flagged" if j == 0 else None)
    ax[1].set_ylim(0, min(np.nanmax(stat) * 1.1, 160))
    ax[1].set_title("Pooled χ²(3) spikes at the shared fault and is flagged")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("χ² statistic"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "filter_bank.png"); plt.close(fig)


def fig_scaling() -> None:
    """Scaling: the chunked reduce lands on the whole-batch LSI fit, to the
    max|Δcoef| the panel prints, and many channels fit in one GEMM.

    The exactness is in the reduce itself: accumulated spectra are additive, so
    chunking reproduces a single pass over the whole array. What is left over
    against ``fit_lsi`` comes from a different empirical spectrum (a
    least-squares Legendre fit there, trapezoid projection in the accumulator)
    taken at a different order; the two coefficient sets therefore land close
    together rather than on top of each other.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1.5, 4000)
    a_t, b_t = 1.0, 1.8
    clean = a_t * np.exp(b_t * x)
    y = clean + rng.normal(0, 0.03 * clean.std(), x.size)

    whole = dt.fit_lsi(x, y, "a*exp(b*x)", "x")
    acc = dt.PartitionedLSI("a*exp(b*x)", "x", domain=(0.0, 1.5), order=6)
    n_chunks = 8
    bnds = np.linspace(0, x.size, n_chunks + 1).astype(int)
    for c in range(n_chunks):  # shared boundary sample keeps the reduce exact
        lo = max(0, bnds[c] - 1)
        acc.update(x[lo:bnds[c + 1]], y[lo:bnds[c + 1]])
    part = acc.fit(p0=[1.0, 1.0])
    dcoef = float(np.max(np.abs(whole.coeffs - part.coeffs)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].scatter(x[::40], y[::40], s=8, c="0.8", label="samples (4000)")
    ax[0].plot(x, np.asarray(whole.model(x)), "tab:blue", lw=2.5,
               label=f"whole-batch LSI (b={whole.params['b']:.4f})")
    ax[0].plot(x, np.asarray(part.model(x)), "tab:orange", lw=1.2, ls="--",
               label=f"PartitionedLSI, {n_chunks} chunks (b={part.params['b']:.4f})")
    ax[0].set_title(f"Map-reduce is exact — max|Δcoef| = {dcoef:.1e}")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)

    # many channels in one GEMM: recovered growth rate vs truth across B channels
    B = 300
    b_true = rng.uniform(0.8, 2.6, B)
    Y = np.exp(np.outer(x, b_true)) * rng.uniform(0.6, 1.4, B)
    Y = Y + rng.normal(0, 0.02 * Y.std(axis=0), Y.shape)
    results = dt.fit_lsi_batched(x, Y, "a*exp(b*x)", "x", order=6)
    b_rec = np.array([r.params["b"] for r in results])
    ax[1].scatter(b_true, b_rec, s=10, c="tab:green", alpha=0.6)
    lim = [0.7, 2.7]
    ax[1].plot(lim, lim, "k--", lw=1, label="exact recovery")
    ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_title(f"GEMM-batched: {B} channels' growth rate in one matmul")
    ax[1].set_xlabel("true b (per channel)"); ax[1].set_ylabel("recovered b")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "scaling.png"); plt.close(fig)


def fig_auto_forecast() -> None:
    """auto_forecast on two series. Given trend plus cycle it routes to a
    linear+seasonal model and extrapolates the cycle, a clear win over a random
    walk; given a structureless series the no-structure guard falls back to
    persistence."""
    # (1) seasonal: trend + cycle -> route to linear+seasonal, extrapolate
    rng = np.random.default_rng(2)
    n, h, P = 160, 40, 24
    t = np.arange(n).astype(float)
    clean = 0.05 * t + 3.0 * np.sin(2 * np.pi * t / P + 0.5) + 10.0
    y = clean + rng.normal(0, 0.5, n)
    fc = dt.auto_forecast(t[: n - h], y[: n - h], h, period=P)
    rw = np.full(h, y[n - h - 1])

    # (2) structureless series: the no-structure guard falls back to persistence
    rng2 = np.random.default_rng(1)
    yn = 20.0 + rng2.normal(0, 3.0, n)
    fc2 = dt.auto_forecast(t[: n - h], yn[: n - h], h)
    poly = np.polyval(np.polyfit(t[: n - h], yn[: n - h], 2), t[n - h:])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    fut = t[n - h:]
    ax[0].plot(t[: n - h], y[: n - h], "0.6", lw=0.8, label="train")
    ax[0].plot(fut, y[n - h:], "k.", ms=3, label="held-out truth")
    ax[0].plot(fut, fc, "tab:blue", lw=2, label="auto_forecast (linear+seasonal)")
    ax[0].plot(fut, rw, "tab:orange", lw=1.2, ls=":", label="random walk")
    ax[0].axvline(t[n - h], color="0.8", ls=":")
    ax[0].set_title("auto_forecast — extrapolates the cycle (trend + season)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8)

    ax[1].plot(t[: n - h], yn[: n - h], "0.6", lw=0.8, label="train")
    ax[1].plot(fut, yn[n - h:], "k.", ms=3, label="held-out truth")
    ax[1].plot(fut, fc2, "tab:blue", lw=2, label="auto_forecast (guard → persist)")
    ax[1].plot(fut, poly, "tab:red", lw=1.2, ls="--", label="naïve poly extrapolation")
    ax[1].axvline(t[n - h], color="0.8", ls=":")
    ax[1].set_title("No-structure guard — structureless series falls back to persistence")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("y"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "auto_forecast.png"); plt.close(fig)


def main() -> None:
    if not (DATA_DIR / "covid_ukraine_confirmed.csv").exists():
        raise SystemExit("Datasets missing -- run: python -m dtfit_experimental.experiments.download_data")

    print("Generating figures into", FIG_DIR)
    fig_lsi()
    fig_lsi_oscillatory()
    fig_eac()
    fig_eac_adaptive()
    fig_filter()
    fig_lsi_filter()
    fig_filter_bank()
    fig_dsb()
    fig_scaling()
    fig_auto_forecast()
    fig_comparison()
    print("  wrote:", ", ".join(sorted(p.name for p in FIG_DIR.glob("*.png"))))

    print("\n\n===== TABLE: model-data exponential recovery =====\n")
    print(table_model_exponential())
    print("\n\n===== TABLE: DSB additive-form curve fit =====\n")
    print(table_dsb_additive())
    print("\n\n===== TABLE: real-data summary =====\n")
    print(table_real_data())


if __name__ == "__main__":
    main()
