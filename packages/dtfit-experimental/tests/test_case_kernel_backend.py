"""The parallel-scaling case backend runs on the numpy kernel.

Case dirs start with a digit, so the module is reached through importlib.
After the host compiled kernel is removed the backend must import with no
reference to ``dtfit._core._native`` and expose no ``HAVE_NATIVE`` flag,
the same import-is-the-check contract as test_domain_backends_import.
"""

from __future__ import annotations

import importlib


def test_parallel_scaling_backend_has_no_native_handle() -> None:
    mod = importlib.import_module(
        "dtfit_experimental.experiments.cases."
        "07_parallel_scaling.backend"
    )
    assert hasattr(mod, "kernel_scaling")
    assert not hasattr(mod, "HAVE_NATIVE")
