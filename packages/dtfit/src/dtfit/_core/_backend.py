"""Pluggable array backend for the GEMM-batched projection.

The batched LSI/EAC projection is a single matrix product ``Dᵀ·(w⊙Y)`` (see
:mod:`dtfit_experimental.scale._batched`). :class:`ImageStream` uses the
same backend for its channel batch's ``S`` projection, ``Phi^T w Y`` in
``image/stream.py``. Both are written with plain ``@``, ``*`` and ``.T``.
A :class:`Backend` only has to move arrays to and from a device:

* ``numpy``: always available, multithreaded BLAS GEMM on the CPU.
* ``cupy``: NumPy-API GPU arrays on cuBLAS, given an install and a GPU.
* ``torch``: CUDA tensors, given an install and ``torch.cuda.is_available()``.

The projection has very low arithmetic intensity; a GPU only pays off once the
data is already resident on it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


class Backend:
    """Moves arrays on and off a compute device. Arithmetic stays generic."""

    def __init__(
        self,
        name: str,
        asarray: Callable[[Any], Any],
        to_host: Callable[[Any], np.ndarray],
    ) -> None:
        self.name = name
        self._asarray = asarray
        self._to_host = to_host

    def asarray(self, a: Any) -> Any:
        """Put ``a`` on this backend's device with the backend dtype."""
        return self._asarray(a)

    def to_host(self, a: Any) -> np.ndarray:
        """Bring a device array back to a host NumPy array."""
        return self._to_host(a)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Backend({self.name!r})"


def _numpy_backend(dtype: Any) -> Backend:
    npd = np.dtype(dtype)
    return Backend(
        "numpy", lambda a: np.asarray(a, dtype=npd), lambda a: np.asarray(a)
    )


def _cupy_backend(dtype: Any) -> Backend:  # pragma: no cover - requires a GPU
    import cupy as cp

    cpd = cp.dtype(dtype)
    return Backend(
        "cupy", lambda a: cp.asarray(a, dtype=cpd), lambda a: cp.asnumpy(a)
    )


def _torch_backend(dtype: Any) -> Backend:  # pragma: no cover - requires a GPU
    import torch

    td = torch.float32 if np.dtype(dtype) == np.float32 else torch.float64
    return Backend(
        "torch",
        lambda a: torch.as_tensor(np.asarray(a), dtype=td, device="cuda"),
        lambda a: a.detach().cpu().numpy(),
    )


def available_backends() -> list[str]:
    """Backends usable here: ``numpy``, plus any GPU one that has a device."""
    out = ["numpy"]
    try:  # pragma: no cover - depends on environment
        import cupy  # noqa: F401

        out.append("cupy")
    except Exception:
        pass
    try:  # pragma: no cover - depends on environment
        import torch

        if torch.cuda.is_available():
            out.append("torch")
    except Exception:
        pass
    return out


def resolve_backend(name: str = "auto", *, dtype: Any = "float64") -> Backend:
    """Build a :class:`Backend` by name. ``"auto"`` prefers a GPU."""
    avail = available_backends()
    if name == "auto":
        if "cupy" in avail:
            name = "cupy"
        elif "torch" in avail:
            name = "torch"
        else:
            name = "numpy"
    if name == "numpy":
        return _numpy_backend(dtype)
    if name in ("cupy", "torch") and name not in avail:
        # A known name with no device behind it: report it like an unknown
        # name instead of an ImportError from inside the factory.
        raise ValueError(
            f"backend {name!r} is not available (install it / a working GPU); "
            f"available: {avail}"
        )
    if name == "cupy":
        return _cupy_backend(dtype)  # pragma: no cover - requires a GPU
    if name == "torch":
        return _torch_backend(dtype)  # pragma: no cover - requires a GPU
    raise ValueError(f"unknown backend {name!r}; available: {avail}")
