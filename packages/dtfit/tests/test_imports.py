"""Package surface: public names, submodule importability, removed modules."""

import importlib
import pkgutil

import pytest

import dtfit


def test_top_level_api():
    for name in [
        "NonlineRegressor",
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
    ],
)
def test_removed_methods_are_gone(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(mod)
