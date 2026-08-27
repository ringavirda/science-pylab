"""dtfit fitting methods.

The differential-transformation batch fitters ``fit_lsi``, ``fit_eac`` and
``fit_dsb``; the ``find_degree`` polynomial primitive that DSB builds on; the
symbolic helpers ``model_params`` and ``taylor_coeffs``; and the user-input
normalizers ``normalize_p0`` and ``normalize_bounds``.
"""

from ._common import (
    model_params,
    taylor_coeffs,
    find_degree,
    normalize_p0,
    normalize_bounds,
)
from ._modelinput import resolve_model, ModelSpec, result_kwargs
from ._dsb import fit_dsb
from ._lsi import fit_lsi, fft_frequency_seed
from ._eac import fit_eac
from ._ensemble import ensemble_fit, EnsembleResult

__all__ = [
    "fit_lsi",
    "fit_eac",
    "fit_dsb",
    "ensemble_fit",
    "EnsembleResult",
    "fft_frequency_seed",
    "find_degree",
    "model_params",
    "taylor_coeffs",
    "normalize_p0",
    "normalize_bounds",
    "resolve_model",
    "ModelSpec",
    "result_kwargs",
]
