# Example 05 Streaming

Streaming / online trackers.

These estimators ingest one sample at a time with bounded per-update cost
(partial_fit(t, y)), for control loops and big-data streams. Each filter is
the streaming twin of a batch method and carries built-in drift detection.
Start from the .tracking() / .robust() presets instead of the ~20 raw knobs.

- EACFilter    -- streaming equal-areas (twin of fit_eac).
- LSIFilter    -- streaming Legendre spectrum (twin of fit_lsi).
- result()     -- the window as a batch fit, with a calibrated covariance.
- fused nis_   -- pooling several filters' innovations into one fault test.

Run headless:   python examples/05_streaming.py

Source: [`packages/dtfit/examples/05_streaming.py`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/examples/05_streaming.py)

```python
import numpy as np

from dtfit import EACFilter, LSIFilter


def track_drifting_parameter(rng) -> None:
    # The exponential rate b jumps mid-stream; the filter re-adapts (its drift
    # test detects the change and re-arms the covariance).
    T = 500
    t = np.linspace(0, 8, T)
    b_true = np.where(t < 4, 0.30, 0.55)
    y = np.exp(b_true * t) + rng.normal(0, 0.05, T)

    flt = EACFilter("exp(b*t)", "t", p0=[0.2], window_size=40, q_diag=[1e-4])
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    print("== EACFilter: track a mid-stream step ==")
    print("final b estimate:", round(flt.params_["b"], 3), " (true 0.55)")
    print("drifts detected :", flt.n_drifts_)


def preset(rng) -> None:
    # The .tracking() preset turns on auto window sizing; .robust() turns on
    # the outlier-resilient gains. Both keep the full kwargs for overrides.
    t = np.linspace(0, 6, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    flt = EACFilter.tracking("A*sin(w*x)", "x")
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    print("\n== EACFilter.tracking() preset ==")
    print("params:", {k: round(v, 3) for k, v in flt.params_.items()})


def lsi_filter(rng) -> None:
    # LSIFilter is the streaming twin of fit_lsi: its measurement is the
    # window's Legendre spectrum (order+1 independent equations per step),
    # which identifies an oscillation's amplitude AND frequency -- shape the
    # single area measurement partly cancels. Here it recovers both online
    # from a noisy sinusoid.
    t = np.linspace(0, 20, 500)
    y = 2.0 * np.sin(1.3 * t) + rng.normal(0, 0.05, t.size)
    flt = LSIFilter.tracking("A*sin(w*x)", "x", p0=[1.0, 1.0])
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    print("\n== LSIFilter.tracking(): online amplitude + frequency ==")
    print("params:", {k: round(v, 3) for k, v in flt.params_.items()},
          " (true A=2.0, w=1.3)")


def window_result(rng) -> None:
    # result() fits the model on the current window with the batch machinery,
    # so the streamed estimate comes with a calibrated covariance.
    t = np.linspace(0, 12, 400)
    y = 1.5 * np.exp(0.25 * t) + rng.normal(0, 0.05, t.size)
    flt = LSIFilter("a*exp(b*t)", "t", p0=[1.0, 0.1], window_size=40)
    for ti, yi in zip(t, y):
        flt.partial_fit(ti, yi)
    res = flt.result()
    print("\n== result(): the window as a batch fit ==")
    print("params:", {k: round(v, 3) for k, v in res.params.items()},
          " stderr:", {k: round(v, 4) for k, v in res.stderr().items()})


def fused_detection(rng) -> None:
    # A change that hits every stream is weak in any one innovation and
    # strong in the pooled statistic: the sum of the filters' nis_ is
    # chi-square with the summed degrees of freedom under the model.
    from scipy.stats import chi2
    K = 3
    t = np.linspace(0, 40, 600)
    amp = np.where(t < 20, 1.0, 0.5)
    phases = (0.0, 0.7, 1.4)
    Y = np.column_stack(
        [amp * np.sin(1.2 * t + p) + rng.normal(0, 0.05, t.size)
         for p in phases])
    flts = [LSIFilter("A*sin(1.2*t + p)", "t", p0=[1.0, 0.0], window_size=40,
                      order=4, adaptive_window=False, alpha=1e-15,
                      cusum_k=float("inf")) for _ in range(K)]
    dof = sum(f.basis.n_coef for f in flts)
    threshold = chi2.ppf(1 - 1e-4, dof)
    first = None
    for i in range(t.size):
        for k, f in enumerate(flts):
            f.partial_fit(t[i], Y[i, k])
        pooled = sum(f.nis_ for f in flts)
        if (i > 120 and np.isfinite(pooled) and pooled > threshold
                and first is None):
            first = t[i]
    print("\n== fused detection: pooled nis_ over three streams ==")
    print("first flag at t =",
          None if first is None else round(first, 1), "(fault at 20)")


def main() -> None:
    rng = np.random.default_rng(0)
    track_drifting_parameter(rng)
    preset(rng)
    lsi_filter(rng)
    window_result(rng)
    fused_detection(rng)


if __name__ == "__main__":
    main()
```

## Output (`python examples/05_streaming.py`)

```text
== EACFilter: track a mid-stream step ==
final b estimate: 0.55  (true 0.55)
drifts detected : 1

== EACFilter.tracking() preset ==
params: {'A': 2.02, 'w': 1.503}

== LSIFilter.tracking(): online amplitude + frequency ==
params: {'A': 2.02, 'w': 1.3}  (true A=2.0, w=1.3)

== result(): the window as a batch fit ==
params: {'a': 1.509, 'b': 0.249}  stderr: {'a': 0.0162, 'b': 0.0009}

== fused detection: pooled nis_ over three streams ==
first flag at t = 20.1 (fault at 20)
```
