# Streaming

`ImageFilter` tracks parameters online on the sliding window's image, one
sample at a time (`partial_fit`), at bounded per-update cost. `LSIFilter`
and `EACFilter` fix its basis to Legendre and block. Start from the
`.tracking()` / `.robust()` presets. `result()` reads the current window
off as a calibrated batch fit; `filter.coast(...)` dead-reckons through
measurement dropouts.

```python
from dtfit import LSIFilter

flt = LSIFilter.tracking("a*exp(b*x)", "x")
for xi, yi in zip(x, y):
    flt.partial_fit(xi, yi)
print(flt.params_)      # latest estimate
```

::: dtfit.ImageFilter

::: dtfit.LSIFilter

::: dtfit.EACFilter

::: dtfit.streaming.DriftDetector
