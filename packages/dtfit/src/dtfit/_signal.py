"""FFT-based shape detection.

Used by the shape router (:mod:`dtfit.forecast`), the model recommender
(:mod:`dtfit.models`) and the auto route's peak-share test
(:mod:`dtfit.image`); it lives here so none of the three has to import
another.
"""

from __future__ import annotations

import numpy as np


def dominant_period(
    y: np.ndarray, *, min_period: int = 4
) -> tuple[float, float]:
    """Period (in samples) and strength of the strongest spectral peak.

    The series is linearly detrended first. ``strength`` is the peak's share
    of the detrended power, in ``[0, 1]``; small means no real cycle.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 2 * min_period:
        return float("nan"), 0.0
    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    spec = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    spec[0] = 0.0
    valid = (freqs > 1.0 / max(n, 1)) & (freqs <= 1.0 / min_period)
    if not valid.any() or spec[valid].sum() == 0:
        return float("nan"), 0.0
    k = int(np.argmax(np.where(valid, spec, 0.0)))
    strength = float(spec[k] / spec[1:].sum()) if spec[1:].sum() > 0 else 0.0
    period = float(1.0 / freqs[k]) if freqs[k] > 0 else float("nan")
    return period, strength
