"""The overlapping-window ensemble.

``ensemble_fit`` aggregates fits over overlapping subwindows. The batch
presets ``fit_lsi`` and ``fit_eac`` live in the image core and are
re-exported here.
"""

from ._ensemble import ensemble_fit, EnsembleResult
from dtfit.image.fit import fit_lsi, fit_eac, fft_frequency_seed

__all__ = [
    "fit_lsi",
    "fit_eac",
    "ensemble_fit",
    "EnsembleResult",
    "fft_frequency_seed",
]
