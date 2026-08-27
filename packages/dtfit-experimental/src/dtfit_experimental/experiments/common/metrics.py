"""Metrics, timing and markdown helpers shared by the experiment suite.

Argument order follows ``dtfit.metrics`` and sklearn: ``(y_true, y_pred)``.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R2 / RMSE / MAE / MAPE of a prediction against the truth."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = y_true != 0
    mape = (float(np.mean(np.abs(err[mask] / y_true[mask])) * 100)
            if mask.any() else float("nan"))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred, float) - np.asarray(y_true, float)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float))))


def timed(fn: Callable[[], object]) -> tuple[object, float]:
    """Run ``fn`` and return ``(result, elapsed_ms)``."""
    t0 = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t0) * 1e3


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v: float | None, spec: str = "{:.4g}") -> str:
    """Format a number, rendering ``None`` and NaN as ``--``."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return spec.format(v)
