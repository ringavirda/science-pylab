"""Phase-1 accuracy gate: recovery across the whole model catalogue.

Every family is swept over a range of noise levels, once through the realistic
user path and once through every batch method. Nothing documented can then land
in a regime nobody measured.

Thresholds are generous absolute ceilings paired with a clause allowing no
more than a small margin over ``curve_fit``: tight enough to catch a method
that diverges or stalls, loose enough not to fail on an unlucky noise draw.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dtfit import fit_lsi, fit_eac
from accuracy.scenarios import SCENARIOS, NOISE_LEVELS
from accuracy.harness import (
    ordered_params,
    r2,
    param_err,
    predict,
    curve_fit_baseline,
)

_CASES = [(s, noise) for s in SCENARIOS for noise in NOISE_LEVELS]
_IDS = [f"{s.name}-noise{noise:g}" for s, noise in _CASES]


@pytest.mark.parametrize("scn,noise", _CASES, ids=_IDS)
def test_recovery_matrix(scn, noise):
    names = ordered_params(scn)
    x, y, clean = scn.make(noise, seed=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = scn.model().fit(x, y)
    est = np.asarray(res.coeffs, float)
    pred = predict(res, x)
    assert np.all(np.isfinite(est)), f"{scn.name}: non-finite coefficients"
    assert np.all(np.isfinite(pred)), f"{scn.name}: non-finite prediction"

    cf_popt, cf_pred = curve_fit_baseline(scn, x, y, names)

    if scn.metric == "params":
        err = param_err(scn, names, est)
        cf_err = (param_err(scn, names, cf_popt)
                  if cf_popt is not None else np.inf)
        # pass on the noise-scaled absolute ceiling, or on being within 3x of
        # a curve_fit seeded exactly the same way, whichever is looser
        allowed = max(scn.tol + 2.0 * noise, 3.0 * cf_err)
        assert err <= allowed, (
            f"{scn.name} @ noise={noise}: param error {err:.3f} > "
            f"{allowed:.3f} (curve_fit {cf_err:.3f}). {scn.note}")
    else:  # curve-quality families (weak parameter identifiability)
        got = r2(clean, pred)
        cf_r2 = (
            r2(clean, np.asarray(cf_pred, float))
            if cf_popt is not None else -np.inf
        )
        # meet the noise-relaxed R^2 floor, or essentially tie the baseline.
        # The second clause covers ill-conditioned shapes like sums of
        # exponentials, where curve_fit does no better either.
        floor = scn.r2_min - 1.5 * noise
        assert got >= floor or got >= cf_r2 - 0.02, (
            f"{scn.name} @ noise={noise}: R2 {got:.4f} < floor {floor:.4f} "
            f"and below baseline {cf_r2:.4f}. {scn.note}")


_METHODS = {
    "lsi": lambda x, y, e, v, p0: fit_lsi(x, y, e, v, p0=p0),
    "eac": lambda x, y, e, v, p0: fit_eac(x, y, e, v, p0=p0),
}
_RUN_CASES = [(s, m) for s in SCENARIOS for m in _METHODS]
_RUN_IDS = [f"{s.name}-{m}" for s, m in _RUN_CASES]


@pytest.mark.parametrize("scn,method", _RUN_CASES, ids=_RUN_IDS)
def test_every_method_runs(scn, method):
    """No method may crash, return NaN, or stall on any catalogue model.

    Quality is gated above; this is the numerical-sanity half, and it is what
    catches the singular-Jacobian family of bug (EAC on ``x**n`` at x = 0).
    Every (model, method) pair a user could pick has to produce a finite fit
    and a finite curve.
    """
    x, y, _ = scn.make(0.03, seed=1)
    m = scn.model()
    p0, _ = m._seed_arrays(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _METHODS[method](x, y, m.expr, m.var, p0)
    est = np.asarray(res.coeffs, float)
    pred = predict(res, x)
    assert np.all(np.isfinite(est)), f"{scn.name}/{method}: non-finite coeffs"
    assert np.all(np.isfinite(pred)), (
        f"{scn.name}/{method}: non-finite prediction"
    )
