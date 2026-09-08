"""Adaptation #5: stage-wise residual boosting (LSI then EAC).

One parametric form rarely captures both a trend and a cycle. Boosting stages
the methods instead: fit stage 1 to the data, subtract its prediction, fit
stage 2 to what is left, and so on down the list. The composite model is the
sum of the stages, say an LSI exponential or polynomial trend plus an
EAC-fitted oscillatory residual. Every stage stays a cheap, well-posed fit
while the composite reaches past what either method expresses alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from dtfit import fit_lsi, fit_eac
from dtfit.types import FittingResult

_FITTERS: dict[str, Callable[..., FittingResult]] = {"lsi": fit_lsi, "eac": fit_eac}


@dataclass
class BoostedModel:
    """Additive composite of staged fits."""

    stage_models: list[Callable[..., Any]]
    stage_specs: list[dict]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        total = np.zeros_like(x)
        for model in self.stage_models:
            v = model(x)
            total = total + (np.full_like(x, float(v)) if np.ndim(v) == 0 else np.asarray(v, float))
        return total


def boosted_fit(
    data_x: np.ndarray,
    data_y: np.ndarray,
    stages: Sequence[dict],
) -> BoostedModel:
    """Fit additive stages, each to the running residual.

    Args:
        data_x, data_y: Observed samples.
        stages: Ordered stage specs. Each is a dict of ``expr``, ``var``,
            ``method`` (``"lsi"`` or ``"eac"``, default ``"lsi"``) and any
            extra fitter kwargs such as ``p0`` or ``bounds``.

    Returns:
        BoostedModel whose ``predict`` sums the staged contributions.
    """
    x = np.asarray(data_x, dtype=float)
    residual = np.asarray(data_y, dtype=float).copy()
    models: list[Callable[..., Any]] = []
    specs: list[dict] = []

    for stage in stages:
        spec = dict(stage)
        method = spec.pop("method", "lsi")
        expr = spec.pop("expr")
        var = spec.pop("var")
        fitter = _FITTERS.get(method)
        if fitter is None:
            raise ValueError(f"stage method must be 'lsi'/'eac', got {method!r}")
        res = fitter(x, residual, expr, var, **spec)
        models.append(res.model)
        specs.append({"expr": expr, "var": var, "method": method, "coeffs": res.coeffs})
        pred = res.model(x)
        pred = np.full_like(x, float(pred)) if np.ndim(pred) == 0 else np.asarray(pred, float)
        residual = residual - pred

    return BoostedModel(stage_models=models, stage_specs=specs)
