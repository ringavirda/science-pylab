# science-nonline — dtfit monorepo

Differential-transformation fitting: nonlinear smoothing and forecasting via
differential / non-Taylor transformations. Developed as part of a PhD
dissertation on mathematical models of nonlinear smoothing and prediction.

This repository is a **monorepo of three distributions**:

| package | path | what it is |
|---|---|---|
| **`dtfit`** | [`packages/dtfit`](packages/dtfit) | the stable, published library -- the public API (`NonlineRegressor`, `ImageFilter` (with `LSIFilter`/`EACFilter` as basis aliases), `fit_lsi`/`fit_eac`/`fit_dsb`, `ImageStream`, the `models` framework + `suggest_models`, `auto_estimate`/`auto_forecast`, diagnostics). |
| **`dtfit-experimental`** | [`packages/dtfit-experimental`](packages/dtfit-experimental) | experimental EAC/LSI adaptations + the full experiment / validation suite. Depends on `dtfit`; **never ships inside the `dtfit` wheel**. |
| **`dtfit-hardware`** | [`packages/dtfit-hardware`](packages/dtfit-hardware) | the real-silicon rig: Arduino firmware, the USB/BLE host telemetry link, the real-log comparison harness, and a React Native phone app. Depends on `dtfit-experimental`. |

The dependency is one-directional: `dtfit-hardware` -> `dtfit-experimental` -> `dtfit`.
When an experimental adaptation proves itself across the experiment suite it is
**promoted** -- physically moved into `dtfit` (as `ensemble_fit` was) and
re-exported from there, never imported back out of the experimental package.

## Install (editable, from the repo root)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate   |   Linux/mac: source .venv/bin/activate

pip install -e packages/dtfit                          # stable library
pip install -e 'packages/dtfit[dev]'                   # + test/lint tooling

# optional — the experimental adaptations + experiment suite:
pip install -e packages/dtfit-experimental
pip install -e 'packages/dtfit-experimental[bench]'    # + suite baselines/plotting

# optional — the hardware rig (firmware, USB/BLE host link, phone app):
pip install -e 'packages/dtfit-hardware[bench]'        # pulls dtfit-experimental
```

## Layout

```
packages/
  dtfit/                     # stable library (pyproject, src/dtfit, tests, examples, build_native.py)
  dtfit/examples/            # runnable, headless example scripts (the guides)
  dtfit-experimental/        # experimental package (pyproject, src/dtfit_experimental, tests)
  dtfit-hardware/            # real-silicon rig (src/dtfit_hardware: firmware + host link; mobile/ phone app)
```

See each package's README for details:
[`packages/dtfit/README.md`](packages/dtfit/README.md) ·
[`packages/dtfit-experimental/README.md`](packages/dtfit-experimental/README.md) ·
[`packages/dtfit-hardware/README.md`](packages/dtfit-hardware/README.md).
The full documentation -- guides, API reference, method math, and validation --
lives in the [project wiki](https://github.com/ringavirda/science-nonline/wiki).
