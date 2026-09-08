"""Phase-3 seed and regime robustness.

A method validated on one favourable RNG draw with a hand-fed p0 can look good
purely by luck of the seed. These tests require recovery to hold across many
noise realizations and from a perturbed initial guess, so that a passing fit
means a basin and not a knife-edge.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dtfit import fit_lsi
from accuracy.scenarios import SCENARIOS
from accuracy.harness import ordered_params, param_err, r2, predict

_SEEDS = range(6)


@pytest.mark.parametrize("scn", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_recovery_stable_across_noise_realizations(scn):
    """Worst case over 6 independent noise draws must stay in tolerance, making
    the result a property of the method rather than of one lucky seed."""
    names = ordered_params(scn)
    worst_perr, worst_r2 = 0.0, 1.0
    for seed in _SEEDS:
        x, y, clean = scn.make(0.05, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scn.model().fit(x, y)
        assert np.all(np.isfinite(res.coeffs))
        worst_perr = max(worst_perr, param_err(scn, names, res.coeffs))
        worst_r2 = min(worst_r2, r2(clean, predict(res, x)))
    if scn.metric == "params":
        assert worst_perr <= 0.15, (
            f"{scn.name}: seed-dependent param error up to "
            f"{worst_perr:.3f}. {scn.note}")
    else:
        assert worst_r2 >= 0.99, (
            f"{scn.name}: seed-dependent R2 down to {worst_r2:.4f}. "
            f"{scn.note}")


# Families where a local LSI solve should converge from a wrong-but-bracketed
# seed. Oscillatory and weak-id families need the recipe/bounds path and are
# covered by the noise-realization sweep above.
_BASIN_MODELS = ["exponential", "exp_decay", "logistic", "michaelis_menten",
                 "gaussian", "first_order", "gompertz"]


@pytest.mark.parametrize("name", _BASIN_MODELS)
@pytest.mark.parametrize("factor", [0.6, 1.4])
def test_basin_stability_to_seed_perturbation(name, factor):
    """A wrong initial guess, scaled +-40% off the data-driven seed, must still
    converge. Recovery cannot hinge on a perfectly-placed p0."""
    scn = next(s for s in SCENARIOS if s.name == name)
    x, y, _ = scn.make(0.02, seed=0)
    m = scn.model()
    p0, _ = m._seed_arrays(x, y)
    assert p0 is not None  # the catalogue scenarios are all self-seeding
    perturbed = [v * factor for v in p0]
    names = ordered_params(scn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_lsi(x, y, m.expr, m.var, p0=perturbed)
    assert np.all(np.isfinite(res.coeffs))
    assert param_err(scn, names, res.coeffs) <= 0.12, (
        f"{name} from {factor}x seed: "
        f"err {param_err(scn, names, res.coeffs):.3f}")
