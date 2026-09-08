"""Polynomial-degree selection, the DSB pre-fit support."""

import numpy as np

from dtfit._stats import information_criteria
from dtfit.log import echo


def find_degree(
    data_x: np.ndarray,
    data_y: np.ndarray,
    method: str = "bic",
    max_degree: int = 12,
) -> int:
    """Select a polynomial degree for ``(data_x, data_y)`` by ``"bic"`` or
    ``"aic"``: the degree minimizing that criterion over ``0..max_degree``.
    """
    if method not in ("bic", "aic"):
        raise ValueError(
            f"Unsupported degree-selection method {method!r}; use 'bic' or 'aic'."
        )
    degree = _find_degree_direct(data_x, data_y, max_degree, method)
    if degree == max_degree:
        echo(f"Warning: maximum degree {max_degree} reached.")
    echo(f"Best polynomial degree selected by {method}: {degree}")
    return degree


def _find_degree_direct(
    data_x: np.ndarray,
    data_y: np.ndarray,
    max_degree: int,
    criterion: str,
) -> int:
    """Degree minimizing AIC/BIC computed from the residual sum of squares."""
    if data_y is None or data_y.size == 0:
        raise ValueError("data must be a non-empty 1-D array")

    n = data_y.size
    max_degree = int(min(max_degree, max(0, n - 1)))
    best_degree, best_score = 0, np.inf

    for deg in range(0, max_degree + 1):
        try:
            coeffs = np.polyfit(data_x, data_y, deg)
        except Exception:
            continue
        model = np.poly1d(coeffs)(data_x)
        rss = float(np.sum((data_y - model) ** 2))
        if rss <= 0:
            return deg
        aic, bic = information_criteria(rss, n, deg + 1)
        score = aic if criterion == "aic" else bic
        if score < best_score:
            best_degree, best_score = deg, score

    return int(best_degree)
