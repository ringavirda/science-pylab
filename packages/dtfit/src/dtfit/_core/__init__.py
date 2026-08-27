"""Private numeric core. Not part of the public API.

- ``_backend``: the array-backend registry (NumPy, optional CuPy/torch).
- ``_kernels``: wrappers over the optional compiled C kernels
  (``dtfit._core._native``, built by ``build_native.py``), with NumPy/SciPy
  fallbacks.
- ``_spectral``: orthogonal-basis construction and spectral NLLS solving.

Import paths under ``dtfit._core`` are internal and may change.
"""
