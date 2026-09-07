"""The exactness comparison both datasets use: the raw least-squares
reference, the score against it, the Legendre order the image needs, and
the initial guess taken from the image alone."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from dtfit.image import Image, fit, u_of
from dtfit.types import FittingResult

EXACTNESS_TOL = 1e-8
# dtfit.image.coverage above this says the image's order cannot represent
# the model's sensitivities; the row is reported UNDERSAMPLED, not failed.
COVERAGE_TOL = 0.02
# Denominator floor for a reference value that is zero to rounding,
# relative to the largest reference magnitude.
_SCORE_FLOOR = 1e-12
# Samples per coefficient below which a fixed order is not a coverage
# failure of the sensitivities but a plain shortage of rows; shared by
# legendre_order's density cap and the NOAA gate's density floor.
DENSITY_PER_COEF = 4


def _scores(
    fitted: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    """Per-parameter relative differences; the shared half of
    :func:`param_score` and :func:`worst_param`."""
    ref = {k: float(v) for k, v in reference.items()}
    m = max((abs(v) for v in ref.values()), default=0.0)
    if m == 0.0:
        return {
            k: 0.0 if float(fitted[k]) == 0.0 else float("inf") for k in ref
        }
    return {
        name: abs(float(fitted[name]) - r) / max(abs(r), _SCORE_FLOOR * m)
        for name, r in ref.items()
    }


def param_score(
    fitted: Mapping[str, float], reference: Mapping[str, float]
) -> float:
    """The worst relative parameter difference between a fit and its
    reference.

    Every parameter is scored ``|f - ref| / max(|ref|, 1e-12 * M)`` with
    ``M`` the largest reference magnitude; the station's score is the
    maximum. The floor only stops a division by a reference value that is
    zero to rounding -- it is twelve orders below ``M``, so the score is a
    relative difference for every parameter these models carry. When every
    reference value is zero the score is 0.0 if every fitted value is zero
    too and ``inf`` otherwise.

    Raises:
        KeyError: ``fitted`` is missing a name ``reference`` carries.
    """
    return max(_scores(fitted, reference).values(), default=0.0)


def worst_param(
    fitted: Mapping[str, float], reference: Mapping[str, float]
) -> str:
    """The parameter name :func:`param_score` scored highest; the first
    name in ``reference``'s order when every score ties."""
    scores = _scores(fitted, reference)
    return max(scores, key=lambda k: scores[k]) if scores else ""


def raw_lstsq(
    design: np.ndarray, y: np.ndarray, names: Sequence[str]
) -> dict[str, float]:
    """Unweighted least squares of ``y`` on ``design``: the reference an
    image fit is compared against. ``names`` label the columns in the
    model's canonical (sorted) parameter order."""
    beta = np.linalg.lstsq(
        np.asarray(design, dtype=float), np.asarray(y, dtype=float),
        rcond=None,
    )[0]
    return {n: float(b) for n, b in zip(names, beta)}


def raw_bic(design: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(rss, bic)`` of the unweighted least-squares fit of ``y`` on
    ``design``.

    ``bic = n*log(rss/n) + k*log(n)`` with ``n = y.size`` and ``k`` the
    column count: the formula :attr:`dtfit.types.FittingResult.bic` uses,
    so an image BIC and this one are directly comparable. A perfect fit
    (``rss <= 0``) gives ``-inf``, as the library's does.
    """
    X = np.asarray(design, dtype=float)
    yy = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    rss = float(np.sum((yy - X @ beta) ** 2))
    n, k = int(yy.size), int(X.shape[1])
    if rss <= 0.0:
        return rss, float("-inf")
    return rss, float(n * math.log(rss / n) + k * math.log(n))


def legendre_order(
    span: float,
    n_samples: int,
    *,
    per_unit: float = 8.0,
    margin: int = 16,
    floor: int = 16,
    per_coef: int = DENSITY_PER_COEF,
) -> int:
    """The Legendre order an image needs to reproduce the raw fit.

    ``ceil(per_unit * span) + margin``, floored at ``floor`` and capped
    twice: at ``n_samples - 2`` (an image of order ``k`` needs ``k + 2``
    samples) and at ``n_samples // per_coef``, the density floor that
    keeps at least ``per_coef`` samples per coefficient. The margin carries
    short spans through the 1e-8 exactness gate; the density cap keeps a
    sparse station representable, at the price of an order too low for
    the model, which the station's ``coverage`` column then reports. The
    result is never below 1.
    """
    order = max(floor, math.ceil(per_unit * float(span)) + margin)
    n = int(n_samples)
    return int(max(1, min(order, n - 2, n // max(1, int(per_coef)))))


def gram_rebuild_error(image: Image) -> float:
    """How far ``G`` rebuilt from the grid is from the accumulated ``G``.

    ``G = Phi^T diag(w) Phi`` is a deterministic function of the grid, the
    basis and the order, all of which travel with the image, so a receiver
    could rebuild it instead of receiving it. Returns the maximum relative
    difference between the rebuild and the stored ``G``, which is a
    rounding-level number for a chunk-accumulated image and is what the
    report quotes when it says what shipping ``S`` and the grid alone
    would cost. Returns ``inf`` when the stored ``G`` is all zeros.
    """
    x = image.grid.positions()
    Phi = image.basis.evaluate(u_of(x, *image.domain))
    w = np.ones(x.size) if image.w is None else np.asarray(
        image.w, dtype=float
    )
    rebuilt = Phi.T @ (w[:, None] * Phi)
    scale = float(np.max(np.abs(image.G)))
    if not scale > 0.0:
        return float("inf")
    return float(np.max(np.abs(rebuilt - image.G)) / scale)


def p0_from_image(
    image: Image,
    names: Sequence[str],
    *,
    level: str = "c",
    slope: str | None = "v",
) -> dict[str, float]:
    """The initial guess for a fit, taken from the image alone.

    Every parameter starts at zero except ``level``, which starts at the
    image's reconstruction at the domain's left edge, and ``slope``, which
    starts at the reconstruction's mean rate across the domain. No sample
    is read, so a machine holding only the image starts the solver at the
    same point as the machine that reduced it.
    """
    t0, t1 = float(image.domain[0]), float(image.domain[1])
    r0, r1 = image.reconstruct(np.array([t0, t1], dtype=float))
    p0 = {str(n): 0.0 for n in names}
    if level in p0:
        p0[level] = float(r0)
    if slope is not None and slope in p0:
        p0[slope] = float((r1 - r0) / (t1 - t0)) if t1 > t0 else 0.0
    return p0


def fit_from_image(
    expr: str,
    image: Image,
    names: Sequence[str],
    *,
    var: str = "t",
    level: str = "c",
    slope: str | None = "v",
    **kwargs: Any,
) -> FittingResult:
    """Fit ``expr`` to ``image`` from :func:`p0_from_image`; ``kwargs`` go
    to :func:`dtfit.image.fit`."""
    p0 = p0_from_image(image, names, level=level, slope=slope)
    return fit(expr, image, var, p0=p0, **kwargs)
