"""Package surface: public names, submodule importability, removed modules."""

import importlib
import pkgutil
import sys

import pytest

import dtfit


def test_top_level_api():
    for name in [
        "NonlineRegressor",
        "ImageFilter",
        "EACFilter",
        "fit_lsi",
        "fit_eac",
        "fit_dsb",
        "models",
        "suggest_models",
        "auto_estimate",
        "enable_logging",
    ]:
        assert hasattr(dtfit, name), f"missing public name: {name}"


def test_all_submodules_import():
    for mod in pkgutil.walk_packages(dtfit.__path__, "dtfit."):
        importlib.import_module(mod.name)


@pytest.mark.parametrize(
    "mod",
    [
        "dtfit.methods.dsbi",
        "dtfit.methods.dsbe",
        # Hand-written differential-transform tables replaced by generic Taylor.
        "dtfit.methods.discretes",
        "dtfit.methods.spectrum",
        "dtfit.scale",
        "dtfit.stochastic._estimators",
        "dtfit.stochastic._model",
        "dtfit.stochastic._stats",
        "dtfit.stochastic._forecast",
        "dtfit.stochastic._simulate",
        "dtfit.stochastic._filter",
    ],
)
def test_removed_methods_are_gone(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)


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


def test_scale_names_left_the_library():
    for name in ("PartitionedLSI", "PartitionedEAC", "PartitionedBatchLSI",
                 "fit_lsi_batched"):
        assert not hasattr(dtfit, name)
    assert callable(dtfit.fit_many) and dtfit.ImageStream is not None
    assert "dtfit.scale" not in sys.modules


def test_filter_bank_names_left_the_library():
    for name in ("FilterBank", "FusedChiSquareDetector"):
        assert not hasattr(dtfit, name)
