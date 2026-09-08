# The reference method

DSB (differential spectra balance) recovers a model's parameters by equating
its Maclaurin spectrum to a polynomial pre-fit's, order by order, and solving
the balance symbolically. It is the ancestor the image methods are derived
against, not part of `fit`.

```python
import numpy as np
from dtfit.reference import find_degree, fit_dsb

deg = max(find_degree(x, y, method="bic"), n_params - 1, 1)
res = fit_dsb(np.polyfit(x, y, deg)[::-1], "a0 + a1*exp(a2*x)", "x")
```

::: dtfit.reference.fit_dsb

::: dtfit.reference.find_degree
