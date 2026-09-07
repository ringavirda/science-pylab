"""Online / real-time estimators for streaming data.

These estimators ingest one sample at a time with bounded per-update cost,
suiting control loops and Big-Data streams. Symbolic work (model and Jacobian)
happens once at construction; updates are pure NumPy/SciPy.

Each filter is the streaming twin of a batch method and carries that method's
name, keeping one method discoverable across execution modes:

- :class:`EACFilter`, twin of :func:`dtfit.fit_eac`. Its measurement is the
  block image's integrated area innovation over a sliding window.
- :class:`LSIFilter`, twin of :func:`dtfit.fit_lsi`. Its measurement is the
  block image's Legendre spectrum, a richer quantity that captures
  oscillations the area criterion partly cancels.

Both accept optional external regressors. The integral measurement can then
score a model that also depends on measured side-channels, an IMU-derived
motion basis for instance, rather than on sample position alone.

:class:`DriftDetector` is the shared change-detection logic on a whitened
innovation, usable on its own or as a building block for a filter.
"""

from ._eac import EACFilter
from ._lsi import LSIFilter
from ._bank import FilterBank, FusedChiSquareDetector
from .detect import DriftDetector

__all__ = [
    "EACFilter",
    "LSIFilter",
    "FilterBank",
    "FusedChiSquareDetector",
    "DriftDetector",
]
