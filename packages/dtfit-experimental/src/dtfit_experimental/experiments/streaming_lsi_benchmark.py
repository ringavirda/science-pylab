"""Streaming LSI vs streaming EAC: head-to-head filter benchmark.

Puts :class:`dtfit.streaming.LSIFilter` (online integral least squares, the
streaming counterpart of ``fit_lsi``) against
:class:`dtfit.streaming.EACFilter` (online equal areas) on two synthetic
streams, each with a known ground truth and a mid-stream parameter drift:

  * Scenario A, exponential growth ``y = a·exp(b·t)`` with a jump in the growth
    rate ``b``. This is the area filter's home turf, a monotone signal whose
    net area is highly informative, and the expectation is rough parity.
  * Scenario B, sine ``y = A·sin(w·t)`` with a jump in the frequency ``w``. A
    sine's net area over a window is ~0, leaving equal areas nearly blind to
    frequency, while the Legendre spectrum resolves the oscillation. The
    spectral filter should win on parameter tracking and on 1-step-ahead
    prediction.

Every filter/scenario pair is scored against the known truth:

  * param RMSE: relative RMS error of the tracked parameters, measured away
    from the warm-up and the post-drift transient;
  * 1-step pred RMSE: RMS one-step-ahead forecast error, quoted next to a
    random walk;
  * conv. steps: samples until the tracked params first stay within 10 % of
    truth (lower is faster observability);
  * drift lag: samples between the true drift and its detection;
  * us/step: mean per-sample wall-clock cost, the hot-path budget.

Run:

    python -m dtfit_experimental.experiments.streaming_lsi_benchmark

Synthetic data only, with a known ground truth, reseeded per scenario so the
two filters see identical samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dtfit.streaming import EACFilter, LSIFilter


@dataclass
class Scenario:
    name: str
    expr: str
    var: str
    t: np.ndarray
    y: np.ndarray
    truth: np.ndarray          # (n_samples, n_params) ground-truth params
    param_names: list[str]
    drift_idx: int
    p0: list[float]
    q_diag: list[float]


def scenario_exponential(n: int = 800) -> Scenario:
    """y = a·exp(b·t); growth rate b jumps 0.6 -> 1.1 at the midpoint."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 4.0, n)
    half = n // 2
    a_true = np.full(n, 1.0)
    b_true = np.where(np.arange(n) < half, 0.6, 1.1)
    # Build a continuous signal: integrate the piecewise-constant rate so the
    # level is continuous across the jump (a realistic regime change).
    clean = np.empty(n)
    clean[:half] = a_true[:half] * np.exp(b_true[:half] * t[:half])
    t_join = t[half - 1]
    level = a_true[half - 1] * np.exp(b_true[half - 1] * t_join)
    clean[half:] = level * np.exp(b_true[half:] * (t[half:] - t_join))
    y = clean + rng.normal(0.0, 0.03 * clean.std(), n)
    truth = np.column_stack([a_true, b_true])
    return Scenario(
        "exp growth (b: 0.6->1.1)", "a*exp(b*t)", "t", t, y, truth,
        ["a", "b"], half, p0=[1.0, 0.3], q_diag=[1e-4, 1e-3],
    )


def scenario_sine(n: int = 1200) -> Scenario:
    """y = A·sin(w·t); frequency w jumps 1.5 -> 2.5 at the midpoint."""
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 40.0, n)
    half = n // 2
    A_true = np.full(n, 3.0)
    w_true = np.where(np.arange(n) < half, 1.5, 2.5)
    # Continuous phase across the frequency change.
    phase = np.empty(n)
    dt = np.diff(t, prepend=t[0])
    phase = np.cumsum(w_true * dt)
    clean = A_true * np.sin(phase)
    y = clean + rng.normal(0.0, 0.3, n)
    truth = np.column_stack([A_true, w_true])
    return Scenario(
        "sine (w: 1.5->2.5)", "A*sin(w*t)", "t", t, y, truth,
        ["A", "w"], half, p0=[2.0, 1.5], q_diag=[1e-3, 5e-4],
    )


def run_filter(make_filter, sc: Scenario) -> dict:
    flt = make_filter(sc)
    n = sc.t.size
    p_hist = np.full((n, len(sc.param_names)), np.nan)
    pred1 = np.full(n, np.nan)   # one-step-ahead forecast of y[i]
    drift_steps: list[int] = []
    step_us: list[float] = []

    prev_ready = False
    for i in range(n):
        # one-step-ahead: predict y[i] from the estimate as it stands before
        # the sample is ingested
        if prev_ready:
            pred1[i] = float(flt.predict(np.array([sc.t[i]]))[0])
        t0 = time.perf_counter()
        flt.partial_fit(sc.t[i], sc.y[i])
        step_us.append((time.perf_counter() - t0) * 1e6)
        if getattr(flt, "drift_flag_", False):
            drift_steps.append(i)
        ready = len(flt._t) >= flt.W
        if ready:
            p_hist[i] = flt.p
        prev_ready = ready

    return {
        "p_hist": p_hist,
        "pred1": pred1,
        "drift_steps": drift_steps,
        "step_us": float(np.mean(step_us)),
        "n_drifts": getattr(flt, "n_drifts_", 0),
    }


def evaluate(res: dict, sc: Scenario) -> dict:
    p_hist, truth = res["p_hist"], sc.truth
    n = sc.t.size
    valid = ~np.isnan(p_hist[:, 0])

    # Settled regions: skip a warm-up margin after the start and another after
    # the drift, to score tracking rather than transient re-adaptation.
    margin = sc.drift_idx // 4
    settled = np.zeros(n, dtype=bool)
    settled[margin:sc.drift_idx] = True
    settled[sc.drift_idx + margin:] = True
    mask = valid & settled

    # Relative param RMSE (scale-free across the two parameters).
    rel = (p_hist - truth) / np.where(np.abs(truth) > 1e-9, np.abs(truth), 1.0)
    param_rmse = float(np.sqrt(np.nanmean(rel[mask] ** 2))) if mask.any() else np.nan

    # One-step-ahead forecast RMSE vs the naive random walk (y[i-1]).
    pred1 = res["pred1"]
    pmask = ~np.isnan(pred1)
    if pmask.any():
        pred_rmse = float(np.sqrt(np.mean((pred1[pmask] - sc.y[pmask]) ** 2)))
        rw = sc.y[:-1]
        rw_rmse = float(np.sqrt(np.mean((rw - sc.y[1:]) ** 2)))
    else:
        pred_rmse = rw_rmse = np.nan

    # Convergence: first index (pre-drift) where all params stay within 10 %.
    within = np.all(np.abs(rel) <= 0.10, axis=1) & valid
    conv = np.nan
    for i in range(n):
        if i >= sc.drift_idx:
            break
        if within[i] and np.all(within[i:sc.drift_idx]):
            conv = i
            break

    # Drift latency: first detection at/after the true drift.
    after = [d for d in res["drift_steps"] if d >= sc.drift_idx]
    drift_lag = (after[0] - sc.drift_idx) if after else np.nan

    return {
        "param_rmse": param_rmse,
        "pred_rmse": pred_rmse,
        "rw_rmse": rw_rmse,
        "conv": conv,
        "drift_lag": drift_lag,
        "n_drifts": res["n_drifts"],
        "step_us": res["step_us"],
    }


def make_eac(sc: Scenario) -> EACFilter:
    # order=2 gives the block image two windows, its fairest config.
    return EACFilter(
        sc.expr, sc.var, p0=sc.p0, window_size=50,
        q_diag=sc.q_diag, order=2,
    )


def make_lsi(sc: Scenario) -> LSIFilter:
    return LSIFilter(
        sc.expr, sc.var, p0=sc.p0, window_size=50, order=5,
        q_diag=sc.q_diag,
    )


def fmt(v, spec="{:.4g}") -> str:
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    scenarios = [scenario_exponential(), scenario_sine()]
    filters = [("EAC (block image, order=2)", make_eac),
               ("LSI (Legendre, order=5)", make_lsi)]

    headers = ["scenario", "filter", "param RMSE", "1-step RMSE",
               "(rand-walk)", "conv. steps", "drift lag", "us/step"]
    rows: list[list[str]] = []
    for sc in scenarios:
        for fname, fmake in filters:
            ev = evaluate(run_filter(fmake, sc), sc)
            rows.append([
                sc.name, fname,
                fmt(ev["param_rmse"], "{:.3f}"),
                fmt(ev["pred_rmse"], "{:.3f}"),
                fmt(ev["rw_rmse"], "{:.3f}"),
                fmt(ev["conv"], "{:.0f}"),
                fmt(ev["drift_lag"], "{:.0f}"),
                fmt(ev["step_us"], "{:.0f}"),
            ])

    print("\nStreaming LSI vs EAC -- online parameter tracking under drift")
    print("(param RMSE is relative, scale-free; lower is better everywhere "
          "except where noted)\n")
    print(md_table(headers, rows))
    print(
        "\nReading it: parity on the exponential (areas are well suited to a "
        "monotone signal); on the sine the spectral filter should track the "
        "frequency and forecast far better, since a sine's net area is ~0 and "
        "the area filter is nearly blind to w."
    )


if __name__ == "__main__":
    main()
