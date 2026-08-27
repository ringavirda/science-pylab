"""The Western parameter-estimation baselines: Prony, Matrix Pencil / ESPRIT,
variable projection, method of moments.

These are the signal-processing and system-ID foils dtfit is scored against.
Each has to recover the nonlinear parameters dtfit targets, the rates,
frequencies and amplitudes, on synthetic data where the truth is known.
"""

import numpy as np
import pytest
from sklearn.metrics import r2_score

from dtfit_experimental.experiments.common.baselines import (
    prony_fit,
    matrix_pencil_fit,
    varpro_fit,
    moment_match_fit,
)


# Prony, the algebraic original
def test_prony_recovers_single_exponential():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4, 400)
    y = 1.5 * np.exp(0.6 * t) + rng.normal(0, 1e-3, t.size)
    m = prony_fit(t, y, n_modes=1)
    # a pure exponential is one real mode, no conjugate pair
    assert m.rate.size == 1
    assert m.rate[0].real == pytest.approx(0.6, abs=0.05)
    assert float(np.real(m.amp[0])) == pytest.approx(1.5, abs=0.1)
    assert r2_score(y, m.predict(t)) > 0.99


def test_prony_needs_enough_samples():
    with pytest.raises(ValueError):
        prony_fit(np.linspace(0, 1, 4), np.ones(4), n_modes=3)


# Matrix Pencil / ESPRIT, the SVD-robust successor
def test_matrix_pencil_recovers_biexponential_rates():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 6, 500)
    y = 2.0 * np.exp(-2.0 * t) + 1.0 * np.exp(-0.3 * t) + rng.normal(0, 2e-3, t.size)
    m = matrix_pencil_fit(t, y, n_modes=2)
    damping = np.sort(m.damping)  # -Re(rate); the truth is 0.3 and 2.0
    assert damping[0] == pytest.approx(0.3, abs=0.05)
    assert damping[1] == pytest.approx(2.0, abs=0.2)
    assert r2_score(y, m.predict(t)) > 0.99


def test_matrix_pencil_recovers_sinusoid_frequency():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 4 * np.pi, 400)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 5e-3, t.size)
    m = matrix_pencil_fit(t, y, n_modes=2)  # real sinusoid: one conjugate pair
    freq = m.frequency[m.frequency > 0]
    assert float(np.max(freq)) == pytest.approx(1.5, abs=0.05)
    assert r2_score(y, m.predict(t)) > 0.99


# variable projection, Golub-Pereyra separable NLLS
def test_varpro_recovers_biexponential():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 6, 400)
    y = 2.0 * np.exp(-2.0 * t) + 1.0 * np.exp(-0.3 * t) + rng.normal(0, 5e-3, t.size)

    def design(alpha, tt):  # linear amplitudes a, c; nonlinear rates b, d
        b, d = alpha
        return np.column_stack([np.exp(-b * tt), np.exp(-d * tt)])

    m = varpro_fit(t, y, design, alpha0=[1.5, 0.5],
                   bounds=([0.05, 0.05], [5.0, 5.0]))
    rates = np.sort(m.alpha)
    assert rates[0] == pytest.approx(0.3, abs=0.05)
    assert rates[1] == pytest.approx(2.0, abs=0.15)
    assert r2_score(y, m.predict(t)) > 0.99


# method of moments / GMM
def test_moment_match_recovers_exponential_growth():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 4, 300)
    y = 1.5 * np.exp(0.6 * t) + rng.normal(0, 5e-3, t.size)

    def f(tt, a, b):
        return a * np.exp(b * tt)

    a, b = moment_match_fit(t, y, f, p0=[1.0, 1.0],
                            bounds=([0.2, 0.05], [5.0, 2.0]))
    assert a == pytest.approx(1.5, abs=0.15)
    assert b == pytest.approx(0.6, abs=0.05)


# the domain head-to-head wiring
def test_subspace_rate_recovery_head_to_head():
    """Every method the domain helper reports stays inside its error budget: 2
    percent for dtfit LSI, SciPy NLLS and Matrix Pencil / ESPRIT on both tasks,
    10 percent for classical Prony and only on the clean exponential. On the
    noisy sinusoid Prony is asked for nothing but a finite number, which is the
    textbook reason the subspace methods replaced it."""
    from dtfit_experimental.experiments.domains.parameter_estimation.backend import (
        subspace_rate_recovery,
    )
    rows = subspace_rate_recovery(np.random.default_rng(0), noise=0.03)
    assert len(rows) == 2
    for row in rows:
        for method in ("dtfit LSI", "SciPy NLLS", "Matrix Pencil/ESPRIT"):
            assert row[method] < 2.0, (row["task"], method, row[method])
    # Prony is held to a number only on rows[0], the clean exponential
    assert rows[0]["Prony"] < 10.0
    # nothing crashes: every method returns a finite error on both tasks
    assert all(np.isfinite(row[m]) for row in rows for m in
               ("dtfit LSI", "SciPy NLLS", "Prony", "Matrix Pencil/ESPRIT"))
