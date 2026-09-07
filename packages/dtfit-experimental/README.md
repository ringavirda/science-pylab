# dtfit-experimental

Experimental structural adaptations of the `dtfit` EAC / LSI methods, plus the
full experiment / validation suite. This is a **separate distribution** that
depends on the stable [`dtfit`](../dtfit) package: it never ships inside the published
`dtfit` wheel, so the public API stays lean while new method modifications are
prototyped and evaluated here.

## What lives here

- **`dtfit_experimental`** -- the experimental adaptations that remain in trial
  (`fit_lsi_basis`, `fit_joint`, `boosted_fit`). These build directly on `dtfit`'s
  internals (`dtfit.methods`, `dtfit._spectral`, `dtfit._backend`). (The
  overlapping-window ensemble `ensemble_fit` has been **promoted** into stable
  `dtfit`.)
- **`dtfit_experimental.experiments`** -- the experiment suite: `cases/` (each
  adaptation in isolation), `domains/` (per-application-domain validation against
  the established baselines), shared `common/` framework, and `data/`.

When an adaptation proves effective across enough domains it is **promoted into
stable `dtfit`** and physically moved there; it is then imported from `dtfit`,
not from here. Already promoted: the LSI **oscillatory recipe**
(`dtfit.fit_lsi(oscillatory=..., freq_param=...)` +
`dtfit.fft_frequency_seed`), and the multi-axis `FusedChiSquareDetector`.
Adaptive-window EAC (#6, curvature-placed windows) is retired; `fit_eac`
places equal windows. The map-reduce estimators (`PartitionedLSI` /
`PartitionedEAC`, #1) and the GEMM-batched `fit_lsi_batched` /
`project_spectra` / `PartitionedBatchLSI` live here in
`dtfit_experimental.scale` until the notebooks rerun on `ImageStream`.

## Install

From the repo root (both packages are editable; this one pulls in `dtfit`):

```bash
pip install -e packages/dtfit                    # stable dtfit
pip install -e packages/dtfit-experimental       # this package
pip install -e "packages/dtfit-experimental[bench]"  # + matplotlib/torch/statsmodels/pandas
```

## Run the suites

The experiments are self-contained **Jupyter notebooks** -- `cases/` (per-adaptation
studies) and `domains/` (per-application-domain studies). Each experiment folder
holds a `backend.py` (the compute) and the notebook (the report: tables, figures,
narrative). Fetch the datasets once, then open or execute any notebook:

```bash
python -m dtfit_experimental.experiments.download_data          # fetch datasets

# open and re-run interactively
jupyter lab src/dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb

# or execute headless (writes outputs + figures in place)
jupyter nbconvert --to notebook --execute --inplace \
    src/dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb
```

Each notebook has a config block of knobs near the top (sized for a few-minute run
by default; comments show how to scale up). The indexes
[`cases/REPORTS.md`](src/dtfit_experimental/experiments/cases/REPORTS.md) and
[`domains/DOMAINS.md`](src/dtfit_experimental/experiments/domains/DOMAINS.md) link
every notebook. See
[`src/dtfit_experimental/experiments/README.md`](src/dtfit_experimental/experiments/README.md)
for the real-data validation details.
