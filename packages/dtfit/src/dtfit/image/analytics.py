"""What an image says about itself: noise level, resolved order and the
decay of its coefficients.

Every function here reads an :class:`~dtfit.image.Image` only: its
coefficients ``beta = G^+ S`` and their covariance per unit noise variance
``V = G^+``. The :class:`~dtfit.image.Image` methods of the same names
forward to these.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .image import Image

# Median of the chi-square distribution with one degree of freedom; turns a
# median of squared standardized coefficients into a variance estimate.
_CHI2_1_MEDIAN = 0.4549364231195728


@dataclass(frozen=True)
class Decay:
    """How an image's coefficients fall off with the order.

    Both laws are fitted to ``log |beta_j|`` over orders 2 to the effective
    order, so both are always reported and ``kind`` names the better fit.

    Attributes:
        ratio: The geometric ratio ``r`` of ``|beta_j| ~ a r^j``, in
            ``(0, inf)``; below 1 for a decaying spectrum.
        exponent: The algebraic exponent ``p`` of ``|beta_j| ~ c j^-p``,
            positive for a decaying spectrum; negative when the
            coefficients grow.
        geometric_r2: Coefficient of determination of the geometric fit on
            the log coefficients, in ``[0, 1]``.
        algebraic_r2: The same for the algebraic fit.
        kind: ``"geometric"`` or ``"algebraic"``, whichever has the larger
            ``r2``; ties go to ``"geometric"``.
    """

    ratio: float
    exponent: float
    geometric_r2: float
    algebraic_r2: float
    kind: str


def _require_legendre(image: "Image", what: str) -> None:
    if image.basis.name != "legendre":
        raise ValueError(
            f"{what} reads the coefficient index as a resolution scale, "
            f"which holds for the legendre basis; this image is in the "
            f"{image.basis.name} basis"
        )


def _scaled_squares(image: "Image") -> np.ndarray:
    """``beta_j^2 / V_jj``: each coefficient in units of its own variance
    per unit noise variance, so every entry estimates the noise variance
    where the signal has died out."""
    v = np.diag(np.linalg.pinv(image.G, hermitian=True))
    return np.asarray(image.beta ** 2 / v, dtype=float)


def _last_above(r: np.ndarray, threshold: float) -> int:
    idx = np.flatnonzero(r > threshold)
    return int(idx[-1]) if idx.size else 0


def _effective_order(r: np.ndarray) -> int:
    """The last index of ``r`` standing three sigma above the noise scale.

    The scale starts as the median of the upper half of ``r`` rescaled by
    the chi-square median, which no single large coefficient can drag, and
    is then refined up to eight times from the mean of the tail the current
    answer leaves.
    """
    k = r.size
    q = max(2, k // 2)
    s2 = float(np.median(r[k - q:])) / _CHI2_1_MEDIAN
    e = _last_above(r, 9.0 * s2)
    for _ in range(8):
        tail = r[e + 1:]
        if tail.size < 4:
            break
        s2 = float(np.mean(tail))
        nxt = _last_above(r, 9.0 * s2)
        if nxt == e:
            break
        e = nxt
    return e


def _tail_variance(image: "Image") -> float | None:
    """The noise variance from the orders above the effective order, or
    ``None`` when the basis is not Legendre or fewer than eight orders are
    left. Silent: the warning belongs to :func:`noise_sigma`."""
    if image.basis.name != "legendre":
        return None
    r = _scaled_squares(image)
    e = _effective_order(r)
    if image.order - e < 8:
        return None
    return float(np.mean(r[e + 1:]))


def effective_order(image: "Image") -> int:
    """The highest order whose coefficient stands above the noise.

    The largest ``j`` with ``|beta_j| > 3 sqrt(s2 V_jj)``, where ``s2`` is
    the noise variance estimated from the orders above the answer itself
    (seeded from the median over the upper half of the orders and refined
    up to eight times). ``0`` means no coefficient beyond the constant
    stands out, which is also what a constant signal gives.

    Args:
        image: A Legendre image.

    Returns:
        An order in ``[0, image.order]``.

    Raises:
        ValueError: the image is not in the Legendre basis, where the
            coefficient index is not a resolution scale.
    """
    _require_legendre(image, "effective_order")
    return _effective_order(_scaled_squares(image))


def noise_sigma(image: "Image") -> float | None:
    """The noise standard deviation read off the image's tail orders.

    ``sqrt(mean_j(beta_j^2 / V_jj))`` over the orders above
    :func:`effective_order`, where the signal has died out and each
    coefficient is noise alone. With unit weights this is the standard
    deviation of the samples' noise in the units of ``y``; with weights it
    is the scale factor on the weights, 1 when they are the true inverse
    variances.

    The estimate is an upper bound when the image order is close to the
    signal's own effective order, since the first tail coefficients still
    hold signal: measured on the ``damped_oscillation`` scenario, 1.76
    times the true sigma at order 24, 1.10 at order 32 and 1.04 at
    order 40.

    Args:
        image: A Legendre image.

    Returns:
        The standard deviation, or ``None`` when fewer than eight orders
        stand above the effective order.

    Raises:
        ValueError: the image is not in the Legendre basis.

    Warns:
        RuntimeWarning: the return is ``None`` for lack of tail orders;
            rebuild the image at a higher order to get a number.
    """
    _require_legendre(image, "noise_sigma")
    v = _tail_variance(image)
    if v is None:
        warnings.warn(
            "noise_sigma: fewer than 8 orders stand above the effective "
            f"order of this image (order {image.order}); no tail is left "
            "to read the noise from",
            RuntimeWarning, stacklevel=2,
        )
        return None
    return float(np.sqrt(v))


def _log_slope(u: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Slope and ``r2`` of the least-squares line ``z ~ a + b u``."""
    A = np.column_stack([np.ones(u.size), u])
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    res = z - A @ c
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - float(res @ res) / ss_tot if ss_tot > 0.0 else 0.0
    return float(c[1]), r2


def decay(image: "Image") -> Decay:
    """Geometric and algebraic decay rates of the image's coefficients.

    Both laws are fitted by least squares to ``log |beta_j|`` over orders 2
    to :func:`effective_order`: ``log|beta_j| = log a + j log r`` for the
    geometric ratio and ``log|beta_j| = log c - p log j`` for the algebraic
    exponent. Orders 0 and 1 are excluded because the mean and the trend of
    a signal carry no information about how fast the rest falls off.

    Args:
        image: A Legendre image.

    Returns:
        A :class:`Decay` with both rates, both goodness-of-fit values and
        the name of the better law.

    Raises:
        ValueError: the image is not in the Legendre basis; or fewer than
            three non-zero coefficients lie between order 2 and the
            effective order, where both laws fit exactly and ``kind``
            would carry no information.
    """
    _require_legendre(image, "decay")
    e = effective_order(image)
    j = np.arange(2, e + 1, dtype=float)
    a = np.abs(np.asarray(image.beta[2:e + 1], dtype=float))
    keep = a > 0.0
    j, a = j[keep], a[keep]
    if j.size < 3:
        raise ValueError(
            "decay needs at least three non-zero coefficients between "
            f"order 2 and the effective order; the effective order is {e}"
        )
    z = np.log(a)
    g_slope, g_r2 = _log_slope(j, z)
    a_slope, a_r2 = _log_slope(np.log(j), z)
    return Decay(
        float(np.exp(g_slope)), float(-a_slope), g_r2, a_r2,
        "geometric" if g_r2 >= a_r2 else "algebraic",
    )
