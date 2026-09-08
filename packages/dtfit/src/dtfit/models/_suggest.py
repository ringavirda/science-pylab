"""Model inference: fit a set of candidate families and rank them.

``suggest_models`` fits every candidate family self-seeded and ranks the fits
by AIC, which penalises parameter count. The family then comes off a scored
shortlist rather than out of a guessed string.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from dtfit.types import FittingResult
from dtfit.diagnostics import fit_report
from dtfit._signal import dominant_period
from dtfit.image import Image
from ._model import Model, as_fit_data
from ._catalog import CATALOG


# Which family categories are plausible for each detected coarse shape.
_SHAPE_CATEGORIES = {
    "oscillatory": {"oscillatory"},
    "peak": {"peak"},
    "monotone": {"trend", "growth", "decay", "sigmoid", "saturating"},
}

# Every category the shape detector can reason about. A family outside this
# set, such as a registered one left at the default "general", carries no
# shape signal to prune on; ``_shortlist`` keeps it unconditionally.
_KNOWN_CATEGORIES = frozenset().union(*_SHAPE_CATEGORIES.values())


def _detect_categories(x: np.ndarray, y: np.ndarray) -> set[str]:
    """Coarse shape -> plausible family categories, for shortlisting.

    An ambiguous shape returns every category.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    cats: set[str] = set()
    _, strength = dominant_period(y)
    # A real cycle needs both a strong spectral peak and several zero
    # crossings of the detrended signal, i.e. two full periods or more. The
    # crossing count is what tells an oscillation from a single broad hump;
    # the spectral period alone cannot.
    t = np.arange(n)
    resid = y - np.polyval(np.polyfit(t, y, 1), t)
    crossings = int(np.count_nonzero(np.diff(np.sign(resid)) != 0))
    # Monotonicity via Spearman rank correlation. A per-sample diff-sign test
    # is noise-dominated where the true slope is small: a noisy sigmoid's flat
    # tails fail it, dropping the sigmoid and saturating families for exactly
    # the S-curves they describe. Spearman tolerates it, scoring ~0.98 on a
    # logistic and ~0 on a sine. scipy's typed stub omits ``statistic`` from
    # its private result class, though the attribute exists at runtime.
    rho = (spearmanr(x, y).statistic  # pyright: ignore[reportAttributeAccessIssue]
           if n > 2 and float(np.std(y)) > 0 else 0.0)
    monotone = abs(float(np.nan_to_num(rho))) > 0.85
    # Additive, not exclusive: a cycle on a trend gets the monotone tag too.
    oscillatory = strength > 0.12 and crossings >= 4
    if oscillatory:
        cats |= _SHAPE_CATEGORIES["oscillatory"]
    # An interior extremum the series rises to and falls from. A handful of
    # separated peaks (a double Gaussian) read as a low-frequency cycle to the
    # crossing test; both families stay shortlisted and the AIC ranking picks.
    i = int(np.argmax(np.abs(y - np.median(y))))
    if not monotone and n > 10 and 0.1 * n < i < 0.9 * n:
        cats |= _SHAPE_CATEGORIES["peak"]
    if monotone:
        cats |= _SHAPE_CATEGORIES["monotone"]
    if not cats:  # ambiguous -> try everything
        cats = set().union(*_SHAPE_CATEGORIES.values())
    cats.add("trend")  # always keep a polynomial baseline in the running
    return cats


def _shortlist(x: np.ndarray, y: np.ndarray) -> list[Model]:
    cats = _detect_categories(x, y)
    out: list[Model] = []
    for factory in CATALOG.values():
        m = factory()
        # A known category is pruned to the detected shape. An unknown one
        # (a registered family) carries no shape signal and is always kept.
        if m.category in cats or m.category not in _KNOWN_CATEGORIES:
            out.append(m)
    return out


def _image_report(res: FittingResult) -> dict[str, Any]:
    """Goodness-of-fit report for a candidate fitted to an Image.

    The image is the data here: :func:`dtfit.fit` records ``rss`` and
    ``n_obs`` against the image identity, so the criteria are those of the
    samples the image was built from rather than of any reconstruction. An
    image carries no ordered sample residual, so the report has no
    ``durbin_watson`` key.
    """
    n = int(res.n_obs) if res.n_obs else 0
    rss = float(res.rss) if res.rss is not None else float("nan")
    r2, aic, bic = res.rsquared, res.aic, res.bic
    rep: dict[str, Any] = {
        "n": n,
        "n_params": int(np.asarray(res.coeffs).size),
        "rss": rss,
        "rmse": float(np.sqrt(rss / n)) if n else float("nan"),
        "r2": float(r2) if r2 is not None else float("nan"),
        "aic": float(aic) if aic is not None else float("inf"),
        "bic": float(bic) if bic is not None else float("inf"),
        "converged": bool(res.converged),
    }
    if res.cov is not None:
        rep["params"] = res.params
        rep["stderr"] = res.stderr()
    return rep


@dataclass
class Suggestion:
    """One ranked candidate: family, fit, and goodness-of-fit report."""

    name: str
    model: Model
    result: FittingResult
    report: dict

    @property
    def aic(self) -> float:
        return float(self.report["aic"])

    @property
    def bic(self) -> float:
        return float(self.report["bic"])

    @property
    def r2(self) -> float:
        return float(self.report["r2"])

    def __repr__(self) -> str:
        return (f"Suggestion({self.name!r}, r2={self.r2:.4f}, "
                f"aic={self.aic:.1f}, params={self.result.params})")


def suggest_models(
    data: Any,
    y: Any = None,
    candidates: list[Model] | None = None,
    *,
    basis: Any = "auto",
    top: int | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Suggestion]:
    """Fit candidate model families to ``data`` and rank them by AIC.

    Args:
        data: An :class:`~dtfit.Original`, an :class:`~dtfit.Image`, or the
            sample positions with ``y``. An Image is shortlisted and seeded
            on its reconstruction over 400 evenly spaced positions, and
            fitted and scored on the image itself.
        y: The sample values, when ``data`` is a bare positions array.
        candidates: Models to try. Defaults to a shape-based shortlist of the
            catalog: oscillatory data skips the peak and monotone families,
            ambiguous data falls back to the whole catalog. Each candidate is
            fit self-seeded via :meth:`Model.fit`.
        basis: Forwarded to :meth:`Model.fit` for every candidate;
            ``"auto"`` (default) routes each fit by outcome, except on an
            Image, where the routing has no samples to measure against and
            the image's own basis is used instead.
        top: If given, return only the best ``top`` suggestions.
        include: Keep only candidates whose name (``"logistic"``) or category
            (``"decay"``, ``"oscillatory"``) is in this list, restricting the
            search to the families you believe plausible.
        exclude: Drop candidates whose name or category is in this list, to
            prune families you know don't apply (``exclude=["oscillatory"]``
            on a monotone series) without post-filtering the result. Applied
            after ``include``.

    Returns:
        :class:`Suggestion` list sorted best-first (lowest AIC). A candidate
        whose fit fails is skipped with a :class:`UserWarning` naming it.
        ``[s.name for s in suggest_models(x, y)][:3]`` gives a quick
        shortlist; ``.report`` holds the full diagnostics. For an Image the
        report comes from the image identity, so ``rss``, ``n``, ``r2``,
        ``aic`` and ``bic`` are those of the samples the image was built
        from and not of the reconstruction the shortlist was read off, and
        it carries no ``durbin_watson``, an image having no ordered sample
        residual.

    Raises:
        TypeError: ``y`` given with an Original or an Image, or ``data`` is
            a bare array and ``y`` is missing.
    """
    target, samples = as_fit_data(data, y)
    x, y = samples.x, samples.y
    image = target if isinstance(target, Image) else None
    if image is not None and basis == "auto":
        # The route measures its candidate bases by their residual over the
        # samples, which an Image does not carry. It is already in one
        # basis, so that is the basis every candidate is fitted in.
        basis = image.basis.name
    models = candidates if candidates is not None else _shortlist(x, y)
    if include is not None:
        inc = set(include)
        models = [m for m in models if m.name in inc or m.category in inc]
    if exclude is not None:
        exc = set(exclude)
        models = [m for m in models if m.name not in exc and m.category not in exc]
    out: list[Suggestion] = []
    for m in models:
        try:
            res = m.fit(target, basis=basis)
            rep = (fit_report(res, x, y) if image is None
                   else _image_report(res))
        except Exception as exc:
            # Warn rather than skip quietly: a family missing from the
            # ranking because it errored should say as much.
            warnings.warn(
                f"candidate model '{m.name}' failed: {exc}",
                UserWarning, stacklevel=2,
            )
            continue
        if not np.isfinite(rep["r2"]):
            continue
        out.append(Suggestion(m.name, m, res, rep))
    out.sort(key=lambda s: s.aic if np.isfinite(s.aic) else np.inf)
    return out[:top] if top else out
