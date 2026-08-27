"""LTSF benchmark loading, on the protocol the published papers report under.

DLinear (arXiv:2205.13504), TimesNet (arXiv:2210.02186) and Time-LLM
(arXiv:2310.01728) all evaluate on the long-term-forecasting benchmark through
one fixed pipeline, inherited from the Informer/Autoformer code:

* splits: ETT* use 12/4/4 months (train/val/test), the other sets 0.7/0.1/0.2
  by length;
* normalization: z-score whose statistics come from the train split alone,
  then applied to every split;
* windows: a lookback of ``L`` steps predicts the next ``H`` steps;
* metrics: MSE and MAE on the normalized values.

Following it exactly is what makes dtfit's measured numbers comparable to the
published ones. CSVs live in ``experiments/data/ltsf/``, fetched by
``download_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LTSF_DIR = Path(__file__).resolve().parent.parent / "data" / "ltsf"

FILES = {
    "ETTh1": "ETTh1.csv", "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv", "ETTm2": "ETTm2.csv",
    "weather": "weather.csv", "exchange": "exchange_rate.csv",
    "electricity": "electricity.csv", "traffic": "traffic.csv",
}


def available() -> list[str]:
    """Dataset keys whose CSV is present locally."""
    return [k for k, f in FILES.items() if (LTSF_DIR / f).exists()]


def load(name: str) -> np.ndarray:
    """Load a dataset's numeric channels as ``(T, C)`` (drops the date column)."""
    import pandas as pd

    df = pd.read_csv(LTSF_DIR / FILES[name])
    cols = [c for c in df.columns if c.lower() not in ("date", "datetime")]
    return df[cols].to_numpy(dtype=float)


def borders(name: str, n: int) -> dict[str, tuple[int, int]]:
    """Standard train/val/test index ranges for a series of length ``n``."""
    if name.startswith("ETTh"):
        b = [(0, 12 * 30 * 24), (12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24),
             (12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24)]
    elif name.startswith("ETTm"):
        b = [(0, 12 * 30 * 24 * 4),
             (12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4),
             (12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4)]
    else:
        n_tr, n_te = int(n * 0.7), int(n * 0.2)
        n_va = n - n_tr - n_te
        b = [(0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, n)]
    return {"train": b[0], "val": b[1], "test": b[2]}


def normalize(data: np.ndarray, train_range: tuple[int, int]):
    """Z-score using train-split statistics; returns ``(scaled, mean, std)``."""
    a, b = train_range
    mu = data[a:b].mean(axis=0)
    sd = data[a:b].std(axis=0) + 1e-8
    return (data - mu) / sd, mu, sd


def test_windows(name: str, lookback: int, horizon: int, *, max_windows=None,
                 stride: int = 1):
    """Yield ``(x_lookback, y_target)`` windows over the test split (normalized).

    ``x_lookback`` is ``(lookback, C)``; ``y_target`` is ``(horizon, C)``. Use
    ``max_windows`` to subsample for a quick run.
    """
    data = load(name)
    n, _ = data.shape
    bd = borders(name, n)
    scaled, _, _ = normalize(data, bd["train"])
    t0, t1 = bd["test"]
    # the standard protocol lets the lookback reach back before the test border
    start = max(t0 - lookback, 0)
    idxs = list(range(start, t1 - lookback - horizon + 1, stride))
    if max_windows is not None and len(idxs) > max_windows:
        sel = np.linspace(0, len(idxs) - 1, max_windows).astype(int)
        idxs = [idxs[i] for i in sel]
    for i in idxs:
        yield scaled[i:i + lookback], scaled[i + lookback:i + lookback + horizon]
