"""Adaptation #2: pluggable orthogonal basis for LSI.

``fit_lsi`` matches spectra on the Legendre basis. A polynomial spectrum needs
many high orders before it can express one oscillation; a Fourier basis
captures the same cycle in a couple of harmonics, while a pure decay sits
naturally on a Laguerre basis. :func:`fit_lsi_basis` leaves the LSI criterion
untouched, the diagonal-weighted spectral match; only the basis it runs on
becomes the caller's choice.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from dtfit.types import FittingResult, InitialGuess
from dtfit._core._spectral import make_basis, solve_spectral


def _savgol_prefilter(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay pre-smoother for an LSI spectral projection: window
    <= 11, cubic polyorder, and a no-op below 5 samples.
    """
    y = np.asarray(y, dtype=float)
    if y.size >= 5:
        window = min(11, y.size if y.size % 2 == 1 else y.size - 1)
        if window > 3:
            return np.asarray(savgol_filter(y, window, polyorder=3), dtype=float)
    return y


def fit_lsi_basis(
    data_x: np.ndarray,
    data_y: np.ndarray,
    expr: str,
    var: str,
    *,
    basis: str = "fourier",
    order: int = 5,
    filter_data: bool | None = None,
    period: float | None = None,
    bounds: list[tuple[float, float]] | None = None,
    p0: InitialGuess = None,
) -> FittingResult:
    """LSI fit with a chosen orthogonal basis.

    Args:
        data_x, data_y: Observed samples.
        expr, var: Model expression and main variable.
        basis: ``"legendre"`` | ``"chebyshev"`` | ``"fourier"`` | ``"laguerre"``.
        order: Spectral order (number of harmonics K for ``"fourier"``).
        filter_data: Savitzky-Golay pre-smoothing before projection. The
            default ``None`` picks per basis: off for ``"fourier"``, because
            smoothing erases the very cycle a Fourier basis targets (the
            reason :func:`dtfit.fit_lsi`'s oscillatory recipe disables it too),
            on everywhere else. Pass a bool to override.
        period: Fundamental period for ``"fourier"``; defaults to the domain
            length, one cycle across the window.
        bounds: Optional per-parameter ``(min, max)``. A fully finite box puts
            a differential-evolution fallback behind ``solve_spectral``'s
            bounded local solve. A multimodal fit such as a free frequency
            needs that fallback.
        p0: Optional initial guess.

    Returns:
        FittingResult with coefficients, callable model and covariance.
    """
    x = np.asarray(data_x, dtype=float)
    y = np.asarray(data_y, dtype=float)

    if filter_data is None:
        filter_data = basis != "fourier"
    if filter_data:
        y = _savgol_prefilter(y)

    domain = (float(x[0]), float(x[-1]))
    kwargs = {"period": period} if basis == "fourier" else {}
    b = make_basis(basis, order, domain, **kwargs)
    beta_data = b.empirical(x, y)
    guess = None if p0 is None else np.asarray(p0, float)
    return solve_spectral(expr, var, b, beta_data, p0=guess, bounds=bounds)
