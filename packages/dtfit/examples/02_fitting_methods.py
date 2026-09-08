"""The three batch fitters -- LSI, EAC, DSB.

Each differential-transformation fitter uses a different *measurement* of "fit":

- LSI (fit_lsi)  -- integral least-squares in a reconditioned Legendre spectrum;
                    the general default, with an oscillatory recipe for cycles.
- EAC (fit_eac)  -- equal-areas integral matching over windows; robust to sparse
                    outliers (robust=True) and good on transients.
- DSB (fit_dsb)  -- symbolic differential-spectra balance against a polynomial
                    pre-fit; an analytical reference method.

Run headless:   python examples/02_fitting_methods.py
"""

import numpy as np

from dtfit import fit_lsi, fit_eac
from dtfit.image import fft_frequency_seed
from dtfit.reference import find_degree, fit_dsb


def lsi_basic(rng) -> None:
    x = np.linspace(0, 4, 300)
    y = 0.5 + 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, x.size)
    # k_star sets the Legendre spectral order matched against the data.
    res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x", k_star=6)
    print("== LSI: offset + exponential ==")
    print("params:", {k: round(v, 4) for k, v in res.params.items()})


def lsi_oscillatory(rng) -> None:
    # A smoothed low-order spectral fit erases cycles. Naming the angular-frequency
    # parameter (freq_param) seeds it from the data's FFT peak and turns on the
    # oscillatory recipe (no smoothing, raised order).
    x = np.linspace(0, 10, 400)
    y = 2.0 * np.sin(1.7 * x + 0.5) + rng.normal(0, 0.10, x.size)
    print("\n== LSI oscillatory recipe: A*sin(w*x + p) ==")
    print("FFT frequency seed:", round(fft_frequency_seed(x, y), 4))
    res = fit_lsi(x, y, "A*sin(w*x + p)", "x", freq_param="w")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def eac_robust(rng) -> None:
    # robust=True builds the image with per-sample Huber weights from an
    # IRLS regression on the basis, so outliers are down-weighted before
    # any model is involved.
    x = np.linspace(0, 5, 250)
    y = 3.0 * np.arctan(1.5 * x) + rng.normal(0, 0.1, x.size)  # truth a=3, w=1.5
    idx = rng.choice(x.size, 12, replace=False)
    y[idx] += rng.normal(0, 3.0, 12)                          # scattered outliers
    res = fit_eac(x, y, "a*atan(w*x)", "x", robust=True)
    print("\n== EAC robust (robust=True) ==")
    print("truth a=3.0 w=1.5 ->", {k: round(v, 3) for k, v in res.params.items()})


def eac_transient(rng) -> None:
    # The block preset's uniform windows condition a localized transient/peak
    # well without any window-placement tuning.
    x = np.linspace(0, 6, 300)
    y = 5.0 * x * np.exp(-1.2 * x) + rng.normal(0, 0.03, x.size)
    res = fit_eac(x, y, "a*x*exp(-b*x)", "x")
    print("\n== EAC block preset on a transient peak ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def dsb(rng) -> None:
    # DSB equates the model's Maclaurin spectrum to a polynomial pre-fit's, order
    # by order. Build ascending polynomial coeffs (the data Taylor spectrum) with
    # find_degree + np.polyfit, then balance.
    x = np.linspace(0, 1.5, 200)
    y = 1.5 * np.exp(1.1 * x) + rng.normal(0, 0.02, x.size)   # truth a=1.5, b=1.1
    deg = find_degree(x, y)
    pc = np.polyfit(x, y, deg)[::-1]
    res = fit_dsb(pc, "a*exp(b*x)", "x")
    print("\n== DSB (polynomial degree {}) ==".format(deg))
    print("params:", {k: round(v, 3) for k, v in res.params.items()})


def main() -> None:
    rng = np.random.default_rng(0)
    lsi_basic(rng)
    lsi_oscillatory(rng)
    eac_robust(rng)
    eac_transient(rng)
    dsb(rng)


if __name__ == "__main__":
    main()
