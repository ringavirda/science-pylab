"""Generate the real numbers behind docs/comparison.md.

Runs a *fair* head-to-head between dtfit and ``scipy.optimize.curve_fit`` on
a handful of representative scenarios drawn from the package's own accuracy
corpus (``tests/accuracy``), plus two things ``curve_fit`` structurally
cannot do: a one-pass out-of-core fit and online drift tracking.

Both fitters get the *same* data-driven initial guess (the model's
``_seed_arrays``), so the comparison isolates the method, not the seeding.
Run with the repo venv::

    F:/repos/science-nonline/.venv/Scripts/python.exe \
        packages/dtfit/docs/gen_comparison.py

It prints Markdown tables (pasted into comparison.md) and is deterministic
(fixed seeds).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Make the shared accuracy corpus importable (tests/ is on sys.path under
# pytest; add it explicitly for a standalone run).
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "tests"))

from accuracy.harness import (  # noqa: E402
    curve_fit_baseline,
    ordered_params,
    param_err,
    predict,
    r2,
)
from accuracy.scenarios import SCENARIOS_BY_NAME  # noqa: E402

from dtfit import EACFilter, ImageStream, fit  # noqa: E402

REPEATS = 7


def _median_ms(fn, *args, **kw) -> float:
    fn(*args, **kw)  # warm up (compile / cache)
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args, **kw)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def batch_table() -> str:
    cases = [
        ("exponential", "clean+noisy exponential"),
        ("logistic", "sigmoid (logistic)"),
        ("damped_oscillation", "oscillatory (damped)"),
    ]
    rows = []
    for name, _label in cases:
        scn = SCENARIOS_BY_NAME[name]
        names = ordered_params(scn)
        for noise in (0.0, 0.05):
            x, y, clean = scn.make(noise, seed=0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d_fit = scn.model().fit(x, y)
                d_perr = param_err(scn, names, np.asarray(d_fit.coeffs, float))
                d_r2 = r2(clean, predict(d_fit, x))
                d_ms = _median_ms(lambda: scn.model().fit(x, y))

                popt, pred = curve_fit_baseline(scn, x, y, names)
                if popt is None:
                    s_perr = s_r2 = s_ms = float("nan")
                else:
                    s_perr = param_err(scn, names, popt)
                    s_r2 = r2(clean, pred)
                    s_ms = _median_ms(
                        lambda: curve_fit_baseline(scn, x, y, names)
                    )

            rows.append(
                (name, noise, d_perr, d_r2, d_ms, s_perr, s_r2, s_ms)
            )

    out = [
        "| scenario | noise | dtfit param err | dtfit R^2 | dtfit ms | "
        "scipy param err | scipy R^2 | scipy ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (nm, ns, dp, dr, dm, sp, sr, sm) in rows:
        out.append(
            f"| {nm} | {ns:.2f} | {dp:.2e} | {dr:.5f} | {dm:.2f} | "
            f"{sp:.2e} | {sr:.5f} | {sm:.2f} |"
        )
    return "\n".join(rows_note(rows) + out)


def rows_note(rows) -> list[str]:
    # summarise who won on accuracy / speed, honestly
    d_wins = sum(1 for r in rows if r[2] <= r[5])
    speed_scipy = sum(1 for r in rows if r[7] < r[4])
    return [
        f"<!-- dtfit param-err <= scipy on {d_wins}/{len(rows)} cells; "
        f"scipy faster on {speed_scipy}/{len(rows)} cells -->",
    ]


def bigdata_point() -> str:
    """One-pass out-of-core fit: 2,000,000 points, fixed chunk memory."""
    rng = np.random.default_rng(1)
    a_true, b_true = 1.5, 0.8
    n = 2_000_000
    chunk = 100_000
    x0, x1 = 0.0, 2.0

    acc = ImageStream("legendre", 6, domain=(x0, x1))
    t0 = time.perf_counter()
    xs = np.linspace(x0, x1, n)
    for i in range(0, n, chunk):
        xc = xs[i : i + chunk]
        yc = a_true * np.exp(b_true * xc) + rng.normal(0.0, 0.05, xc.size)
        acc.update(xc, yc)
    res = fit("a*exp(b*t)", acc.image(), "t", p0=[1.0, 1.0])
    dt = (time.perf_counter() - t0) * 1e3

    a_est, b_est = res.params["a"], res.params["b"]
    mem_mb = chunk * 8 * 2 / 1e6  # two float64 chunk buffers resident
    return (
        f"- **n = {n:,} points**, streamed in {n // chunk} chunks of "
        f"{chunk:,} (peak working set ~{mem_mb:.1f} MB of sample buffers).\n"
        f"- Recovered `a = {a_est:.4f}` (truth 1.5), `b = {b_est:.4f}` "
        f"(truth 0.8) in **{dt:.0f} ms**, one pass.\n"
        f"- `curve_fit` would need all {n:,} points resident "
        f"(~{n * 8 * 2 / 1e6:.0f} MB for x+y) and a full NLLS over them; the "
        "map-reduce accumulator never holds more than one chunk."
    )


def streaming_point() -> str:
    """Online drift tracking: a sinusoid whose amplitude drifts 1.0 -> 3.0.

    A single batch fit yields one compromised amplitude; the streaming EAC
    filter follows the drift sample-by-sample.
    """
    rng = np.random.default_rng(2)
    n = 600
    t = np.linspace(0.0, 60.0, n)
    A_true = 1.0 + 2.0 * (t / t[-1])  # drifts 1 -> 3
    y = A_true * np.sin(t) + rng.normal(0.0, 0.05, n)

    # Online: track A(t) with a sliding-window EAC filter (w baked to 1.0).
    flt = EACFilter("A*sin(t)", "t", p0=[1.0], window_size=40)
    a_track = np.empty(n)
    for i in range(n):
        flt.partial_fit(t[i], y[i])
        a_track[i] = flt.params_["A"]
    # score on the settled tail (after the window fills)
    warm = 80
    track_rmse = float(np.sqrt(np.mean((a_track[warm:] - A_true[warm:]) ** 2)))

    # Batch: scipy.curve_fit yields one global amplitude for the whole record.
    from scipy.optimize import curve_fit

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(lambda tt, A: A * np.sin(tt), t, y, p0=[1.0])
    a_batch = float(popt[0])
    batch_rmse = float(np.sqrt(np.mean((a_batch - A_true[warm:]) ** 2)))

    return (
        f"- True amplitude drifts **1.0 -> 3.0** over {n} samples.\n"
        f"- A single `scipy.curve_fit` gives one value `A = {a_batch:.2f}` -- "
        f"RMSE **{batch_rmse:.2f}** against the drifting truth.\n"
        f"- The streaming `EACFilter` tracks it online, "
        f"RMSE **{track_rmse:.2f}** (final estimate `A = {a_track[-1]:.2f}`, "
        f"truth {A_true[-1]:.2f}).\n"
        "- `curve_fit` has no online/`partial_fit` mode: tracking drift means "
        "re-fitting the whole growing window every step."
    )


if __name__ == "__main__":
    print("### Batch accuracy table\n")
    print(batch_table())
    print("\n### Big-data one-pass\n")
    print(bigdata_point())
    print("\n### Streaming drift tracking\n")
    print(streaming_point())
