"""Private numeric core. Not part of the public API.

- ``_backend``: the array-backend registry (NumPy, optional CuPy/torch).
- ``_kernels``: numpy/scipy kernels for the window-projection inner loops.
- ``_spectral``: orthogonal-basis construction and spectral NLLS solving.

Import paths under ``dtfit._core`` are internal and may change.
"""
