"""dtfit quickstart -- your first fit in a minute.

dtfit fits models that are *nonlinear in their parameters* (exponential,
transcendental, oscillatory, mixed). You bring a model as a small sympy
expression string (e.g. "a*exp(b*t)") and your data; dtfit recovers the
parameters and returns a self-describing FittingResult.

Run headless:   python examples/01_quickstart.py
"""

import numpy as np

from dtfit import Original, fit, fit_lsi


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 200)
    y = 1.4 * np.exp(0.8 * x) + rng.normal(0, 0.15, x.size)

    # 1. A first fit. Everything in the expression except the variable "t" is a
    #    free parameter -- here a and b.
    res = fit_lsi(x, y, "a*exp(b*t)", "t")
    print("== fit_lsi: a*exp(b*t) ==")
    print(res.summary())
    print("params:", {k: round(v, 4) for k, v in res.params.items()})
    # The optimizer's verdict travels with the result: check it before
    # trusting a fit (converged is False on a misspecified model or a bad
    # seed).
    print("converged:", res.converged)

    # 2. Uncertainty: an overdetermined fit carries a parameter covariance,
    #    so it reports standard errors, confidence intervals and a
    #    prediction band.
    print("\n== uncertainty ==")
    print("stderr:", {k: round(v, 4) for k, v in res.stderr().items()})
    print("95% CI:", {k: tuple(round(b, 3) for b in ci)
                      for k, ci in res.confidence_intervals().items()})
    xs = np.linspace(x.min(), x.max(), 5)
    y_hat, y_sd = res.predict(xs, return_std=True)
    print("predict(return_std) sd:", np.round(y_sd, 4))

    # opt-in extrapolation guard: predicting past the fitted range is the
    # classic curve-fitting footgun, so warn_extrapolation flags it instead
    # of silently returning a wild value.
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res.predict(np.array([x.max() + 5.0]), warn_extrapolation=True)
    print("extrapolation warned:", bool(caught))

    # 3. Don't want to choose the basis? basis="auto" fits the candidates and
    # keeps whichever leaves the smallest residual over the samples.
    res2 = fit("a*exp(b*t)", Original(x, y), "t", basis="auto")
    print("\n== fit(basis=\"auto\") ==")
    print("params:", {k: round(v, 4) for k, v in res2.params.items()})


if __name__ == "__main__":
    main()
