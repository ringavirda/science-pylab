"""Phase-4 golden-regression guard.

Pins the recovery accuracy of every scenario to a checked-in snapshot,
``accuracy/golden_baseline.json``. A change that quietly degrades any catalogue
family fails here. An intended improvement is adopted by regenerating the
snapshot with ``python -m accuracy.make_golden`` and reviewing the diff.

Every entry is the median over the five noise draws of
``accuracy.harness.SEEDS``, so a routing change moves the number in
proportion to what it did rather than flipping on one lucky draw.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from accuracy.scenarios import SCENARIOS, NOISE_LEVELS
from accuracy.harness import metrics_for

_GOLDEN = json.loads(
    (Path(__file__).parents[1] / "accuracy" / "golden_baseline.json")
    .read_text()
)

# Allowed drift before a change counts as a regression. A purely absolute
# slack is toothless here: most baseline param-errors sit far below 0.03 (many
# ~1e-3, and ~1e-15 at noise=0), leaving room to regress tenfold and still
# pass. The guard is relative, with a small absolute floor so a near-zero
# baseline is not held to machine precision.
_PERR_REL = 1.5      # +50% relative param-error worsening, or
_PERR_ABS = 0.01     # +0.01 absolute, whichever is larger
_R2_REL = 1.5        # the residual (1 - R^2) may grow 50%, plus
_R2_ABS = 1e-4       # a small absolute floor

_CASES = [(s, noise) for s in SCENARIOS for noise in NOISE_LEVELS]
_IDS = [f"{s.name}@{noise:g}" for s, noise in _CASES]


def test_golden_covers_every_case():
    expected = {
        f"{s.name}@{noise:g}" for s in SCENARIOS for noise in NOISE_LEVELS
    }
    assert set(_GOLDEN) == expected, (
        "golden_baseline.json is stale; "
        "regenerate with `python -m accuracy.make_golden`")


@pytest.mark.parametrize("scn,noise", _CASES, ids=_IDS)
def test_no_accuracy_regression(scn, noise):
    key = f"{scn.name}@{noise:g}"
    base = _GOLDEN[key]
    now = metrics_for(scn, noise)
    if base["metric"] == "params":
        limit = max(base["perr"] * _PERR_REL, base["perr"] + _PERR_ABS)
        assert now["perr"] <= limit, (
            f"{key}: param error regressed {base['perr']:.4g} -> "
            f"{now['perr']:.4g} (limit {limit:.4g})")
    else:
        # guard the residual (1 - R^2) relatively: a 0.9999 -> 0.999 drop
        # trips
        limit = _R2_REL * (1.0 - base["r2"]) + _R2_ABS
        assert (1.0 - now["r2"]) <= limit, (
            f"{key}: R2 regressed {base['r2']:.5f} -> {now['r2']:.5f} "
            f"(residual limit {limit:.4g})")
