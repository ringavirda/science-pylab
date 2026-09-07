"""Outlier-robustness gate for ``ensemble_fit``.

The Gaussian-noise corpus never exercises outliers. This is where the case for
``ensemble_fit`` is actually measured: on spike-contaminated data its
overlapping-window median must beat the plain whole-record fit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import dtfit as dt
from accuracy.scenarios import SCENARIOS
from accuracy.harness import ordered_params, param_err

# Families the whole-record fit already nails on clean data. Anything that
# degrades it here is the contamination, not the family.
_OUTLIER_FAMILIES = [
    "exponential", "exp_decay", "power_law", "michaelis_menten", "first_order",
]
_SEEDS = range(6)


def _errs(name):
    scn = next(s for s in SCENARIOS if s.name == name)
    names = ordered_params(scn)
    plain, ens = [], []
    for seed in _SEEDS:
        x, y, _ = scn.make(0.03, seed=seed, outlier_frac=0.04, outlier_scale=8.0)
        m = scn.model()
        p0, _ = m._seed_arrays(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pe = param_err(scn, names,
                           dt.fit_eac(x, y, m.expr, m.var, p0=p0).coeffs)
            ee = param_err(scn, names,
                           dt.fit_eac(x, y, m.expr, m.var, p0=p0,
                                      robust=True).coeffs)
        plain.append(pe)
        ens.append(ee)
    return np.array(plain), np.array(ens)


@pytest.mark.parametrize("name", _OUTLIER_FAMILIES)
def test_robust_image_beats_plain_under_outliers(name):
    """Per family: under 4% spike contamination the robust image's median
    recovery error beats the plain fit and stays usable in absolute terms."""
    plain, rob = _errs(name)
    assert np.median(rob) < np.median(plain), (
        f"{name}: robust median {np.median(rob):.3f} "
        f"not better than plain {np.median(plain):.3f}")
    assert np.median(rob) <= 0.15, f"{name}: robust median {np.median(rob):.3f}"


def test_ensemble_pooled_robustness():
    """Pooled across families and seeds, both the median and the mean error
    drop. The mean is the interesting one: a single fit can blow up on a bad
    window and the ensemble cannot."""
    P, E = [], []
    for name in _OUTLIER_FAMILIES:
        p, e = _errs(name)
        P.append(p)
        E.append(e)
    P, E = np.concatenate(P), np.concatenate(E)
    assert np.median(E) < np.median(P)
    assert np.mean(E) < np.mean(P)
