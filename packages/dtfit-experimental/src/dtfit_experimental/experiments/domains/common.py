"""Domain-specific compute helpers the ``backend.py`` modules import.

``dtfit_experimental.experiments.common`` owns the metric, baseline, dataset
and plotting helpers; this module adds the three utilities that belong to a
single domain: peak-memory measurement for big data, a dominant-period detector
for forecasting, and the embedded footprint formula.
"""

from __future__ import annotations

import tracemalloc
from typing import Callable

import numpy as np


# big data: peak RAM of a callable. Streaming exists to hold that peak flat
# where the whole-array path grows O(N), so measuring it is the whole claim.
def peak_memory(fn: Callable[[], object]) -> tuple[object, float]:
    """Run ``fn``, returning ``(result, peak_MiB)`` over its allocations."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak / (1024 * 1024)


# forecasting: find a dominant cycle, so the merged forecaster can decide
# whether to add a seasonal stage and at what frequency
def dominant_period(y: np.ndarray, *, min_period: int = 4) -> tuple[float, float]:
    """Return ``(period_samples, strength)`` of the strongest spectral peak of a
    linearly-detrended series. ``strength`` is the peak's share of detrended
    power in [0, 1]; a small value means no real cycle, so skip the seasonal
    stage.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    t = np.arange(n)
    trend = np.polyval(np.polyfit(t, y, 1), t)
    resid = y - trend
    spec = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0  # the DC bin: detrending should have emptied it anyway
    valid = freqs > (1.0 / max(n, 1))
    valid &= freqs <= (1.0 / min_period)
    if not valid.any() or spec[valid].sum() == 0:
        return float("nan"), 0.0
    k = np.argmax(np.where(valid, spec, 0.0))
    strength = float(spec[k] / spec[1:].sum()) if spec[1:].sum() > 0 else 0.0
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength


# embedded: the streaming filter's deployable, no-malloc state size
def embedded_footprint(n_params: int, window: int, kind: str = "eac") -> dict:
    """Words and bytes of the fixed streaming-filter C struct (Exp 9 formula).

    EAC filter state is the window buffer (t, y) plus the covariance P (n^2),
    the estimate (n), scratch (n) and about 8 words of bookkeeping. A
    Legendre-spectrum filter adds a read-only projection table, which lives in
    flash rather than SRAM.

    Returns word and byte counts for float32, the deployable size, and float64.
    """
    n = int(n_params)
    sram_words = 2 * window + n * n + 2 * n + 8
    flash_words = 0
    if kind == "legendre":
        order = max(1, n)  # rough: order ~ params; projection (order+1)*window
        flash_words = (order + 1) * window
    return {
        "sram_words": sram_words,
        "sram_bytes_f32": sram_words * 4,
        "sram_bytes_f64": sram_words * 8,
        "flash_words": flash_words,
        "flash_bytes_f32": flash_words * 4,
    }
