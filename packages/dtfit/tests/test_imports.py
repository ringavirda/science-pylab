"""Package surface: the public names, submodule importability, what left."""

import importlib
import pkgutil
import sys

import pytest

import dtfit

# The whole top-level surface. Fifteen entry points and the three
# subpackages that carry the rest; everything else is reached through its
# own module.
PUBLIC = {
    "Original",
    "Image",
    "ImageStream",
    "ImageFilter",
    "fit",
    "fit_lsi",
    "fit_eac",
    "LSIFilter",
    "EACFilter",
    "order_for",
    "fit_many",
    "models",
    "suggest_models",
    "auto_forecast",
    "FittingResult",
    "ForecastResult",
    "stochastic",
    "diagnostics",
    "__version__",
}


def test_top_level_api_is_exactly_the_public_surface():
    assert set(dtfit.__all__) == PUBLIC
    for name in PUBLIC:
        assert hasattr(dtfit, name), f"missing public name: {name}"


def test_all_submodules_import():
    for mod in pkgutil.walk_packages(dtfit.__path__, "dtfit."):
        importlib.import_module(mod.name)


def test_importing_dtfit_does_not_import_scikit_learn():
    """The estimator is the only part that needs it, and it is opt-in."""
    out = __import__("subprocess").run(
        [sys.executable, "-c",
         "import dtfit, sys; print('sklearn' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize(
    "mod",
    [
        "dtfit.methods",
        "dtfit.estimators",
        "dtfit.auto",
        "dtfit.scale",
        "dtfit.stochastic._estimators",
        "dtfit.stochastic._model",
        "dtfit.stochastic._stats",
        "dtfit.stochastic._forecast",
        "dtfit.stochastic._simulate",
        "dtfit.stochastic._filter",
    ],
)
def test_removed_modules_are_gone(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


@pytest.mark.parametrize(
    "name,module",
    [
        ("NonlineRegressor", "dtfit.sklearn"),
        ("fit_dsb", "dtfit.reference"),
        ("find_degree", "dtfit.reference"),
        ("Model", "dtfit.models"),
        ("Stochastic", "dtfit.models"),
        ("register", "dtfit.models"),
        ("unregister", "dtfit.models"),
        ("resolve_model", "dtfit.models"),
        ("fit_stochastic", "dtfit.stochastic"),
        ("StochasticModel", "dtfit.stochastic"),
        ("StochasticFilter", "dtfit.stochastic"),
        ("FittingProblem", "dtfit.image"),
        ("fft_frequency_seed", "dtfit.image"),
        ("coverage", "dtfit.image"),
        ("enable_logging", "dtfit.log"),
        ("logger", "dtfit.log"),
    ],
)
def test_the_rest_is_reached_through_its_module(name, module):
    assert not hasattr(dtfit, name), f"{name} is not a top-level name"
    assert hasattr(importlib.import_module(module), name)


def test_stochastic_surface():
    from dtfit import stochastic
    for name in ("SecondOrderImage", "SecondOrderStream", "fit_stochastic",
                 "StochasticModel", "StochasticFilter", "FORECASTERS",
                 "is_nonstationary", "dickey_fuller", "sample_acf",
                 "hurst_aggvar", "hurst_spectral", "ar1_reversion",
                 "ar_order", "fit_ar", "fractional_difference",
                 "garch_persistence", "cycle_period",
                 "decompose_trend_cycle"):
        assert hasattr(stochastic, name), f"missing public name: {name}"


def test_removed_names_left_the_library():
    for name in ("PartitionedLSI", "PartitionedEAC", "PartitionedBatchLSI",
                 "fit_lsi_batched", "project_spectra", "auto_estimate",
                 "ensemble_fit", "EnsembleResult", "FilterBank",
                 "FusedChiSquareDetector"):
        assert not hasattr(dtfit, name)
    assert callable(dtfit.fit_many) and dtfit.ImageStream is not None
    assert "dtfit.scale" not in sys.modules
