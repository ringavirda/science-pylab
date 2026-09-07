# Versioning & deprecation policy

dtfit follows [Semantic Versioning](https://semver.org/) (`MAJOR.MINOR.PATCH`),
with the standard **0.x caveat**.

## While on 0.x (now)

The project is pre-1.0 (`Development Status :: 3 - Alpha`). Under SemVer, the
public API may still change between **minor** releases:

- **`0.MINOR` (e.g. 0.3 -> 0.4)** may add features and may make backward-
  incompatible changes to the public API. Breaking changes are called out in the
  [CHANGELOG](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/CHANGELOG.md)
  and, wherever feasible, staged through the deprecation process below.
- **`0.MINOR.PATCH` (e.g. 0.4.0 -> 0.4.1)** is bug-fix only: no intended API
  breakage.

We still try hard not to break you: additive changes are strongly preferred, and
a removal is deprecated first whenever a one-release warning window is possible.

## After 1.0

Once the API is declared stable at `1.0.0`, standard SemVer applies:

- **MAJOR** -- backward-incompatible public API changes.
- **MINOR** -- backward-compatible additions.
- **PATCH** -- backward-compatible bug fixes.

## The public API

The public API is what you can reach without a leading underscore:

- names exported from `dtfit` (see `dtfit.__all__`) and its public submodules
  (`dtfit.methods`, `dtfit.models`, `dtfit.streaming`, `dtfit.image`,
  `dtfit.stochastic`, `dtfit.diagnostics`);
- their documented parameters, attributes and return types.

Anything under a `_private` module or prefixed with `_` (e.g. `dtfit._core`,
`Model._seed_arrays`) is internal and may change at any time without notice.

## Deprecation convention

When a public name or behavior is scheduled for removal:

1. It keeps working and emits a `DeprecationWarning` naming the replacement, for
   **at least one full minor release** before removal (e.g. deprecated in 0.5,
   earliest removal in 0.6).
2. The deprecation and its replacement are recorded in the CHANGELOG under the
   release that introduced the warning.
3. Removal happens in a later release, noted in the CHANGELOG.

```python
import warnings

def old_name(*args, **kwargs):
    warnings.warn(
        "old_name() is deprecated and will be removed in a future release; "
        "use new_name() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return new_name(*args, **kwargs)
```

`DeprecationWarning` is silent by default in application code, so it will not spam
end users, but you can surface them while testing:

```bash
python -W error::DeprecationWarning -m pytest
```

## Supported Python & dependencies

dtfit supports the Python versions listed in its
[`pyproject.toml`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit/pyproject.toml)
classifiers (currently 3.10-3.13) and the minimum NumPy / SciPy / SymPy /
scikit-learn floors declared there. Dropping support for an end-of-life Python or
raising a dependency floor is treated as a minor-version change and noted in the
CHANGELOG.
