# Example 07 Diagnostics

Diagnostics, serialization and logging.

Evaluate a fitted dtfit model: information criteria, residual-structure tests,
ready-made plots, opt-in logging, and round-trip serialization. (For plain scalar
metrics on arrays, use sklearn.metrics / scipy.stats directly.)

Run headless:        python examples/07_diagnostics.py
Show the plots too:  python examples/07_diagnostics.py --plot   (needs the viz extra)

Source: [`packages/dtfit/examples/07_diagnostics.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/07_diagnostics.py)

```python
import sys

import numpy as np

from dtfit import fit_lsi, FittingResult


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 250)
    y = 0.5 + 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, x.size)
    res = fit_lsi(x, y, "a0 + a1*exp(a2*x)", "x")

    # fit_report -- sample/param counts, RSS, RMSE, r2, AIC/BIC, Durbin-Watson.
    from dtfit.diagnostics import fit_report, residual_diagnostics

    rep = fit_report(res, x, y)
    print("== fit_report ==")
    for k in ("n", "rmse", "r2", "aic", "bic", "durbin_watson", "converged"):
        if k in rep:
            print("  {:14s}: {}".format(k, round(rep[k], 4)
                                        if isinstance(rep[k], float) else rep[k]))

    # residual_diagnostics -- autocorrelation / normality of the residuals.
    rd = residual_diagnostics(res, x, y)
    print("\n== residual_diagnostics ==")
    print("  durbin_watson :", round(rd["durbin_watson"], 3))
    print("  lag1_autocorr :", round(rd["lag1_autocorr"], 3))
    print("  normality_p   :", round(rd["normality_p"], 3))

    # The image has its own diagnostics, read from the projections alone: the
    # noise level, how many orders carry signal, and a chi-square test of
    # whether the model left structure the basis still resolves. Original
    # .diagnostics runs the residual tests above for a model and parameters,
    # without a FittingResult.
    from dtfit import Original

    orig = Original(x, y)
    img = orig.image("legendre", 40)
    print("\n== the image's own diagnostics ==")
    print("  noise_sigma    :", round(img.noise_sigma(), 4))
    print("  effective_order:", img.effective_order())
    left = img.test_structure("a0 + a1*exp(a2*x)", res.coeffs, "x")
    print("  structure left :", left.reject, "p =", round(left.pvalue, 3))
    print("  wrong model    :",
          img.test_structure("a0 + a1*x", [0.5, 5.0], "x").reject)
    print("  durbin_watson  :",
          round(orig.diagnostics("a0 + a1*exp(a2*x)", res.coeffs, "x")
                ["durbin_watson"], 3))

    # Serialize -- everything needed to rebuild the model round-trips through a
    # JSON-friendly dict.
    blob = res.to_dict()
    restored = FittingResult.from_dict(blob)
    print("\n== to_dict / from_dict round-trip ==")
    print("  params:", {k: round(v, 3) for k, v in restored.params.items()})

    # Opt-in logging -- dtfit logs under the "dtfit" logger with a NullHandler by
    # default; enable_logging(DEBUG) surfaces the fitting internals.
    import logging
    from dtfit import enable_logging

    enable_logging(logging.WARNING)   # quiet here; use DEBUG to see fit detail
    logging.getLogger("dtfit").handlers = [logging.NullHandler()]

    # Optional plots (scikit-learn-style Display objects; need matplotlib).
    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt
        from dtfit.diagnostics import FitDisplay, ResidualsDisplay

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        FitDisplay.from_predictions(x, y, res.predict(x), ax=axes[0],
                                    estimator_name="LSI")
        ResidualsDisplay.from_predictions(y, res.predict(x), ax=axes[1],
                                          estimator_name="LSI")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
```

## Output (`python examples/07_diagnostics.py`)

```text
== fit_report ==
  n             : 250
  rmse          : 0.203
  r2            : 0.9968
  aic           : -791.3853
  bic           : -780.8209
  durbin_watson : 1.9154
  converged     : True

== residual_diagnostics ==
  durbin_watson : 1.915
  lag1_autocorr : 0.04
  normality_p   : 0.892

== the image's own diagnostics ==
  noise_sigma    : 0.233
  effective_order: 3
  structure left : False p = 0.088
  wrong model    : True
  durbin_watson  : 1.915

== to_dict / from_dict round-trip ==
  params: {'a0': 0.534, 'a1': 1.984, 'a2': 0.501}
```
