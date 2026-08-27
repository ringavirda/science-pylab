"""Shared fixtures: synthetic datasets with known ground-truth parameters."""

import numpy as np
import pytest


@pytest.fixture
def exp_data():
    """Exponential decay y = 2.5 * exp(-1.2 x) on a short LSI-friendly span."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 2, 80)
    y = 2.5 * np.exp(-1.2 * x) + rng.normal(0, 0.05, x.size)
    return x, y, {"a": 2.5, "b": -1.2}


@pytest.fixture
def arctan_data():
    """Saturating y = 5 * arctan(1.5 x)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 400)
    y = 5.0 * np.arctan(1.5 * x) + rng.normal(0, 0.3, x.size)
    return x, y, {"a": 5.0, "w": 1.5}


@pytest.fixture
def lint_exp_data():
    """Linear + exponential y = 0.5 + 0.2 x + 0.3 exp(0.4 x) (DSB-friendly)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 150)
    y = 0.5 + 0.2 * x + 0.3 * np.exp(0.4 * x) + rng.normal(0, 0.05, x.size)
    return x, y
