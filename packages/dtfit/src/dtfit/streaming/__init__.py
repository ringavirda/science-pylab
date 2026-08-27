"""Online / real-time estimators for streaming data.

These estimators ingest one sample at a time with bounded per-update cost,
suiting control loops and Big-Data streams. Symbolic work (model and Jacobian)
happens once at construction; updates are pure NumPy/SciPy.

Each filter is the streaming twin of a batch method and carries that method's
name, keeping one method discoverable across execution modes:

- :class:`EACFilter`, twin of :func:`dtfit.fit_eac` /
  :class:`dtfit.PartitionedEAC`. Its measurement is the integrated area
  innovation over a sliding window.
- :class:`LSIFilter`, twin of :func:`dtfit.fit_lsi` /
  :class:`dtfit.PartitionedLSI`. Its measurement is the window's Legendre
  spectrum, a richer quantity that captures oscillations the area criterion
  partly cancels.

Both accept optional external regressors. The integral measurement can then
score a model that also depends on measured side-channels, an IMU-derived
motion basis for instance, rather than on sample position alone.
"""

from ._eac import EACFilter
from ._lsi import LSIFilter
from ._bank import FilterBank, FusedChiSquareDetector

__all__ = [
    "EACFilter",
    "LSIFilter",
    "FilterBank",
    "FusedChiSquareDetector",
]
