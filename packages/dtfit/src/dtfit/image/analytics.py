"""What an image says about itself: noise level, resolved order, coefficient
decay, and the chi-square tests of equality and of structure.

Every function here reads an :class:`~dtfit.image.Image` only: its
coefficients ``beta = G^+ S`` and their covariance per unit noise variance
``V = G^+``. The :class:`~dtfit.image.Image` methods of the same names
forward to these.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import chi2

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


@dataclass(frozen=True)
class ChiSquareTest:
    """The outcome of one chi-square test on an image.

    Attributes:
        statistic: The quadratic form, non-negative.
        dof: Degrees of freedom, at least 1.
        pvalue: Upper-tail probability of ``statistic`` under the null, in
            ``[0, 1]``.
        alpha: The significance level the verdict was taken at, in
            ``(0, 1)``.
    """

    statistic: float
    dof: int
    pvalue: float
    alpha: float

    @property
    def reject(self) -> bool:
        """``True`` when ``pvalue < alpha``: the null hypothesis is rejected.
        The null is equality for :func:`test_equal` and "the model explains
        everything in the span" for :func:`test_structure`."""
        return bool(self.pvalue < self.alpha)


def _pinv_rank(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Hermitian pseudo-inverse of ``M`` and its numerical rank, singular
    values below ``1e-15`` of the largest dropped (the rule
    :attr:`~dtfit.image.Image.beta` solves by)."""
    _, s, vt = np.linalg.svd(M, hermitian=True)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        return np.zeros_like(M), 0
    keep = s > 1e-15 * s[0]
    inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    return (vt.T * inv) @ vt, int(np.count_nonzero(keep))


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
    answer leaves. The threshold is floored at a tiny fraction of the
    largest entry, since with no noise at all the tail is floating-point
    roundoff and both the seed and the refined scale would otherwise latch
    onto whichever roundoff coefficient happens to be largest.
    """
    k = r.size
    q = max(2, k // 2)
    floor = 1e-24 * float(r.max()) if r.size else 0.0
    s2 = float(np.median(r[k - q:])) / _CHI2_1_MEDIAN
    e = _last_above(r, max(9.0 * s2, floor))
    for _ in range(8):
        tail = r[e + 1:]
        if tail.size < 4:
            break
        s2 = float(np.mean(tail))
        nxt = _last_above(r, max(9.0 * s2, floor))
        if nxt == e:
            break
        e = nxt
    return e


def _tail_variance(image: "Image") -> float | None:
    """The noise variance from the orders above the effective order, or
    ``None`` when the basis is not Legendre, the Gram is rank-deficient
    (an unidentified coefficient's minimum-norm value over a near-zero
    ``V_jj`` would corrupt every entry of the tail), or fewer than eight
    orders are left. Silent: the warning belongs to :func:`noise_sigma`."""
    if image.basis.name != "legendre":
        return None
    if _pinv_rank(image.G)[1] < image.n_coef:
        return None
    r = _scaled_squares(image)
    e = _effective_order(r)
    if image.order - e < 8:
        return None
    return float(np.mean(r[e + 1:]))


def _residual_variance(image: "Image") -> float:
    """``(sumsq - S^T beta) / (n - rank(G))``: the weighted residual
    variance of the basis regression, the noise scale the chi-square tests
    use. The divisor is the rank of ``G``, the same rank the tests' degrees
    of freedom use, not ``n_coef``."""
    dof = image.n - _pinv_rank(image.G)[1]
    if dof <= 0:
        return float("nan")
    rss = max(float(image.sumsq - image.S @ image.beta), 0.0)
    return rss / dof


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


def _warn_no_tail(image: "Image", stacklevel: int) -> None:
    """The ``noise_sigma`` warning, with ``stacklevel`` set by the caller
    so it points at the user's call site whether that is this module's
    :func:`noise_sigma` or :meth:`~dtfit.image.Image.noise_sigma`."""
    warnings.warn(
        "noise_sigma: fewer than 8 orders stand above the effective "
        f"order of this image (order {image.order}); no tail is left "
        "to read the noise from",
        RuntimeWarning, stacklevel=stacklevel,
    )


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
        _warn_no_tail(image, stacklevel=3)
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


def test_equal(
    image: "Image", other: "Image", alpha: float = 0.05, *,
    sigma: float | None = None,
) -> ChiSquareTest:
    """Chi-square test that two images are of the same signal.

    With ``d = beta_a - beta_b``, the statistic is
    ``d^T (s2_a V_a + s2_b V_b)^+ d``, which is chi-square distributed with
    as many degrees of freedom as the covariance has rank when both images
    observe the same signal under noise. ``s2_a`` and ``s2_b`` are each
    image's own noise variance, not a single pooled scale, since an image
    built with ``sigma=`` and an unweighted image carry their residuals in
    different units; using one image's scale for the other's covariance
    term is what lets a weighted and an unweighted image be compared
    without the test going silently blind. The two images must cover the
    same domain: on a block basis, an image whose windows are empty over
    part of the domain has zero coefficients there, which the test then
    reads as a difference from an image that has them; the Legendre basis
    has no such windows. Beyond that, the two images may have any sample
    sets, weights and sample counts; only the basis, its order and the
    domain must agree, since the coefficients are compared entry by entry.

    Measured false-alarm rate at ``alpha=0.05``, 2000 replicates of a
    logistic signal at order 12: 0.046 under Gaussian noise, 0.052 under
    Student-t noise with three degrees of freedom, 0.048 under Laplace
    noise; the heavy tails do not break the test. Measured over 400
    replicates of the same signal, one image weighted with ``sigma=`` and
    the other unweighted: 0.050, the same rate as two unweighted images.

    Args:
        image: The first image.
        other: The second image; same basis, order and domain.
        alpha: Significance level of the verdict, in ``(0, 1)``.
        sigma: The noise standard deviation, if known, shared by both
            images. ``None`` takes each image's own noise variance from
            its basis-regression residual,
            ``(sumsq - S^T beta) / (n - rank(G))``, falling back to the
            other image's estimate when one image has no residual degree
            of freedom.

    Returns:
        A :class:`ChiSquareTest`; ``reject`` is ``True`` when the images
        differ by more than noise.

    Raises:
        ValueError: the two images differ in basis, order or domain;
            ``alpha`` outside ``(0, 1)``; ``sigma`` not finite and
            positive; or, with ``sigma=None``, neither image has a residual
            degree of freedom to estimate a noise scale from.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    if image.basis != other.basis or image.domain != other.domain:
        raise ValueError(
            "images to compare must share basis, order and domain; got "
            f"{image.basis.to_dict()} on {image.domain} and "
            f"{other.basis.to_dict()} on {other.domain}"
        )
    if sigma is not None:
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be finite and positive; got {sigma}")
        s2_a = s2_b = float(sigma) ** 2
    else:
        s2_a, s2_b = _residual_variance(image), _residual_variance(other)
        if not np.isfinite(s2_a) and not np.isfinite(s2_b):
            raise ValueError(
                "neither image has a residual degree of freedom to estimate "
                "the noise from; pass sigma"
            )
        s2_a = s2_a if np.isfinite(s2_a) else s2_b
        s2_b = s2_b if np.isfinite(s2_b) else s2_a
    d = image.beta - other.beta
    cov = (s2_a * np.linalg.pinv(image.G, hermitian=True)
           + s2_b * np.linalg.pinv(other.G, hermitian=True))
    inv, rank = _pinv_rank(cov)
    if rank < 1:
        raise ValueError(
            "the pooled covariance of the two images has rank 0; the noise "
            "scale is zero or the Gram matrices are degenerate"
        )
    stat = float(d @ (inv @ d))
    return ChiSquareTest(stat, rank, float(chi2.sf(stat, rank)), alpha)


def test_structure(
    image: "Image", model: Any, params: Any, var: str | None = None, *,
    alpha: float = 0.05, param_names: Any = None,
    sigma: float | None = None, fitted: bool = True,
) -> ChiSquareTest:
    """Chi-square test that a model explains everything in the image's span.

    The model is projected on the image's own grid with the image's own
    weights (:meth:`~dtfit.image.Image.of_model`), and the leftover
    projections ``d = S - S_f`` are tested against their covariance
    ``s2 G``: the statistic is ``d^T (s2 G)^+ d``, the drop in residual sum
    of squares between the model and the best fit in the span, in units of
    the noise variance. A large value means the basis still resolves
    structure the model does not. The chi-square distribution holds when
    ``s2``'s own residual degrees of freedom, ``n - rank(G)``, dominate
    the rank being tested; the statistic is really ``rank * F(rank, n -
    rank(G))``, and the false-alarm rate at ``alpha=0.05`` drifts above
    nominal as the residual degrees of freedom shrink relative to the
    rank (measured on the true model at order 10: 0.050 at 489 residual
    degrees of freedom, 0.101 at 29, 0.181 at 9).

    Args:
        image: The image to test against.
        model: A SymPy expression string, a ``sympy.Expr``, or a callable
            ``f(x, *params)``, as :func:`~dtfit.image.fit` takes.
        params: Parameter values in canonical order (sorted names for a
            symbolic model, signature order for a callable).
        var: The main variable name; required for a symbolic model.
        alpha: Significance level of the verdict, in ``(0, 1)``.
        param_names: Parameter names for a callable model; see
            :func:`~dtfit.models.resolve_model`.
        sigma: The noise standard deviation, if known. ``None`` takes it
            from the image's own basis-regression residual,
            ``sqrt((sumsq - S^T beta) / (n - rank(G)))``.
        fitted: ``True`` (default) when ``params`` were estimated from this
            image, which costs one degree of freedom per parameter;
            ``False`` for parameters fixed beforehand.

    Returns:
        A :class:`ChiSquareTest`; ``reject`` is ``True`` when the model
        leaves structure in the span unexplained.

    Raises:
        ValueError: ``alpha`` outside ``(0, 1)``; ``sigma`` not finite and
            positive; the model is not finite on the image's grid; the
            image has no degrees of freedom left after the model's
            parameters; the noise scale cannot be estimated.
        RuntimeError: the model has no free parameters.
    """
    from dtfit._input import resolve_model
    from .image import Image

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    spec = resolve_model(model, var, param_names=param_names)
    if not spec.names:
        raise RuntimeError("Model has no free parameters.")
    if sigma is not None:
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be finite and positive; got {sigma}")
        s2 = float(sigma) ** 2
    else:
        s2 = _residual_variance(image)
        if not np.isfinite(s2) or s2 <= 0.0:
            raise ValueError(
                "the image has no residual variance to test against "
                f"({image.n} samples, {image.n_coef} coefficients); pass "
                "sigma"
            )
    S_f = Image.of_model(
        model, params, image.grid, image.basis, var=var,
        domain=image.domain, w=image.w, param_names=param_names,
    ).S
    inv, rank = _pinv_rank(image.G)
    dof = rank - (len(spec.names) if fitted else 0)
    if dof < 1:
        raise ValueError(
            f"test_structure is left with {dof} degrees of freedom: the "
            f"image's {rank} identifiable coefficients do not exceed the "
            f"model's {len(spec.names)} parameters"
        )
    d = image.S - S_f
    stat = float(d @ (inv @ d)) / s2
    return ChiSquareTest(stat, dof, float(chi2.sf(stat, dof)), alpha)
