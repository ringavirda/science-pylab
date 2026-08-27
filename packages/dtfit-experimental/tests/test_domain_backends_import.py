"""Import smoke test for the per-domain experiment backends.

Each domain's ``backend.py`` is the single source of truth its notebook and its
paper scripts both import. A rename or a moved symbol breaks that link quietly:
nothing fails until someone runs a heavy notebook or re-renders a figure.
Importing every backend and checking the names its tooling calls turns that
into a CI failure instead.
"""

from __future__ import annotations

import importlib

import pytest

_BASE = "dtfit_experimental.experiments.domains"

# backend module -> the public names its notebook and paper scripts call.
# An empty tuple leaves the import itself as the whole check.
_BACKENDS = {
    "forecasting": ("evaluate_series", "merged_forecaster", "win_summary",
                    "best_oracle", "collapse_oracle_scores", "exp_model_mismatch"),
    "parameter_estimation": ("MODELS", "gen", "est_eac", "est_lsi", "est_nlls",
                             "param_err", "FAMILY_REASON", "load_puromycin",
                             "real_puromycin", "exp_model_mismatch"),
    "big_data": (),
    "embedded_control": ("clean_accuracy", "sweep_perr_all", "exp_model_mismatch"),
    "stochastic_series": ("exp_ar_discrimination", "exp_ar_order_recovery",
                          "exp_fracdiff_whitening", "exp_student_t"),
    "realtime_gps": (),
}


@pytest.mark.parametrize("domain,symbols", list(_BACKENDS.items()),
                         ids=list(_BACKENDS))
def test_backend_imports_and_exposes_symbols(domain, symbols):
    mod = importlib.import_module(f"{_BASE}.{domain}.backend")
    missing = [s for s in symbols if not hasattr(mod, s)]
    assert not missing, f"{domain}.backend missing public symbols: {missing}"
