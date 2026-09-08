"""Statistics of a least-squares fit: the Gauss-Newton parameter covariance
and the Gaussian-likelihood information criteria.

The covariance here takes the residual and the parameter count; the image
core's ``fit`` has its own, which takes a noise variance already computed
from the image identity.
"""

import numpy as np


def nlls_covariance(
    jac: np.ndarray, res: np.ndarray, n_params: int, *,
    absolute_sigma: bool = False,
) -> np.ndarray | None:
    """Gauss-Newton covariance ``sigma^2 (J^T J)^-1`` from the residual
    Jacobian.

    Taken from the SVD of ``J`` rather than by inverting ``J^T J``. Forming
    ``J^T J`` squares the condition number, which makes ``inv(J^T J)`` both
    slower and far less accurate exactly when the parameters are
    near-degenerate, the case a covariance is most needed. With
    ``J = U S V^T``, ``(J^T J)^-1 = V diag(1/s^2) V^T``; singular values
    below a relative tolerance count as null directions (Moore-Penrose), as
    in ``scipy.optimize.curve_fit``'s SVD-based covariance.

    Args:
        jac: ``(m, n_params)`` Jacobian of the residual at the solution.
        res: Length-``m`` residual at the solution.
        n_params: Number of free parameters.
        absolute_sigma: If ``False`` (default) the raw ``(J^T J)^-1`` is
            scaled by the reduced chi-square ``res @ res / (m - n_params)``:
            the residual carries the unknown noise scale. With ``True``
            there is no such rescaling (``sigma^2 = 1``), the residual being
            assumed already scaled by ``1/sigma``, so the covariance
            reflects those absolute measurement errors. Same meaning as
            ``scipy.optimize.curve_fit``'s ``absolute_sigma`` flag.

    Returns:
        The ``(n_params, n_params)`` covariance, or ``None`` for an exactly-
        or under-determined system (``m <= n_params``) or when ``J`` is
        entirely singular.
    """
    m = res.size
    if m <= n_params:
        return None
    try:
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s[0] == 0.0:
        return None
    tol = s[0] * max(jac.shape) * float(np.finfo(float).eps)
    inv_s2 = np.where(s > tol, 1.0 / (s * s), 0.0)
    jtj_inv = (vt.T * inv_s2) @ vt
    sigma2 = 1.0 if absolute_sigma else float(res @ res) / (m - n_params)
    return sigma2 * jtj_inv


def information_criteria(rss: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian-likelihood ``(AIC, BIC)`` from a residual sum of squares.

    ``AIC = n*ln(rss/n) + 2k`` and ``BIC = n*ln(rss/n) + k*ln(n)`` for ``n``
    samples and ``k`` parameters. A perfect fit (``rss <= 0``) returns
    ``-inf``.
    """
    if rss <= 0:
        return float("-inf"), float("-inf")
    base = n * np.log(rss / n)
    return float(base + 2 * k), float(base + k * np.log(n))
