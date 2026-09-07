# dtfit experiment suite -- reports index

Each experiment is a self-contained folder with a `backend.py` (the compute) and a Jupyter notebook (the report) plus its `figures/`.


## Experiments

| # | experiment | report | status | runtime (s) |
|---|---|---|---|---|
| 1 | Control-systems system identification | [01_control_systems.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb) | ok | 2 |
| 2 | Big data / streaming (scaling law) | [02_big_data_streaming.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/02_big_data_streaming/02_big_data_streaming.ipynb) | ok | 5 |
| 3 | Noise & robustness sweep | [03_noise_robustness.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/03_noise_robustness/03_noise_robustness.ipynb) | ok | 4 |
| 4 | Real-world forecasting (train/holdout) | [04_realworld_forecasting.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/04_realworld_forecasting/04_realworld_forecasting.ipynb) | ok | 1 |
| 5 | GPS positioning & trajectory forecast | [05_gps_trajectory.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/05_gps_trajectory/05_gps_trajectory.ipynb) | ok | 1 |
| 6 | LTSF benchmark vs published R&D results | [06_benchmark_ltsf.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/06_benchmark_ltsf/06_benchmark_ltsf.ipynb) | ok | 5 |
| 7 | Parallel scaling & architecture adaptability | [07_parallel_scaling.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/07_parallel_scaling/07_parallel_scaling.ipynb) | ok | 17 |
| 8 | GEMM-batched projection throughput (CPU/GPU) | [08_gpu_batched_projection.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/08_gpu_batched_projection/08_gpu_batched_projection.ipynb) | ok | 1 |
| 9 | Embedded footprint (latency, memory, MCU fit) | [09_embedded_footprint.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/09_embedded_footprint/09_embedded_footprint.ipynb) | ok | 0 |
| 10 | Fused map-reduce + GEMM-batched LSI (multi-channel big data) | [10_fused_partitioned_batched.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/10_fused_partitioned_batched/10_fused_partitioned_batched.ipynb) | ok | 3 |

## Architecture-adaptation effectiveness matrix

Each novel EAC/LSI adaptation, scored across the experiments that exercised it (win / partial / loss / n/a). **Promotion gate**: a clear win on >= 2 distinct application domains.

| adaptation | per-domain result | decision |
|---|---|---|
| #1 map-reduce LSI/EAC (PartitionedLSI/EAC) | big data: win, parallel: win, forecasting: n/a | In `dtfit_experimental.scale`, covered by `ImageStream` -- exact one-pass distributed estimator behind the big-data scaling law and the parallel map-reduce. |
| #2 pluggable orthogonal basis (fit_lsi_basis) | control: partial, forecasting: partial, ltsf: loss | Keep experimental -- Fourier basis expresses periodic models cleanly, but a window-local seasonal term only helped the cleanest periodic LTSF series (electricity) and hurt elsewhere (a 96-point period estimate drifts out of phase over long horizons); it did not beat the tuned forecasting baselines. |
| #3 overlapping-window ensemble (ensemble_fit) | noise/outliers: partial, gps: n/a | PROMOTED (specialized tool) -- on densely-contaminated data it is the whole-window-rejection route that stays stable where `fit_eac`'s robust loss diverges, so it ships in stable `dtfit` as `ensemble_fit` -- the complement to the robust loss, not a general-purpose default. |
| #4 joint multi-channel fit (fit_joint) | control: loss, gps: n/a | Keep experimental -- on cleanly-identifiable channels the dedicated solver already wins; value is parameter parsimony / consistency, not accuracy. |
| #5 stage-wise boosting (boosted_fit) | forecasting: win, ltsf: n/a | Keep experimental -- a clear win on CO2 (trend+season) but only one domain demonstrated; promote if a second domain confirms. |
| #6 adaptive-window EAC | transient fit: win | Retired -- `fit_eac` places equal windows. |

## Promotion outcome

- **Promoted to the stable API:** the overlapping-window ensemble (`ensemble_fit`, #3 -- the complementary whole-window-rejection robustness tool), the fused `FusedChiSquareDetector`, the LSI oscillatory recipe, and the stochastic-series solution, all re-exported from `dtfit` and documented. The map-reduce estimators (`PartitionedLSI` / `PartitionedEAC`, #1) and the GEMM-batched projection (`fit_lsi_batched`) live in `dtfit_experimental.scale`, covered by `ImageStream`; adaptive-window EAC (#6) is retired, `fit_eac` places equal windows.

- **Kept experimental in `dtfit_experimental`:** #2 (`fit_lsi_basis`), #4 (`fit_joint`) and #5 (`boosted_fit`), with the honest per-experiment findings above -- they have not (yet) cleared the >=2-domain promotion gate.


