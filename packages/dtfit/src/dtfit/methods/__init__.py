"""The reference method and the outlier ensemble.

``fit_dsb`` balances a model's symbolic spectrum against a polynomial's;
``find_degree`` picks that polynomial's degree; ``ensemble_fit`` aggregates
overlapping-window fits. The batch presets ``fit_lsi`` and ``fit_eac`` live
in the image core and are re-exported here.
"""

from ._common import find_degree
from ._dsb import fit_dsb
from ._ensemble import ensemble_fit, EnsembleResult
from dtfit.image.fit import fit_lsi, fit_eac, fft_frequency_seed

__all__ = [
    "fit_lsi",
    "fit_eac",
    "fit_dsb",
    "ensemble_fit",
    "EnsembleResult",
    "fft_frequency_seed",
    "find_degree",
]
