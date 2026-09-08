# dtfit domain validation suite — index

Where the `cases` suite evaluates each EAC/LSI adaptation in isolation, this suite is a **comprehensive per-application-domain study**: each domain tests every applicable dtfit method against the **established domain-standard methods**, on synthetic *and real* data, checking validity, applicability and usefulness. Each domain is a self-contained folder with a `backend.py` (the compute) and a Jupyter notebook (the report: *Methods under test*, *Baseline methods*, tables, figures, narrative) plus its `figures/`. Open a notebook in Jupyter and re-run it, or execute it headless with `jupyter nbconvert --to notebook --execute --inplace <notebook>`.


## Domains

| domain | notebook |
|---|---|
| Forecasting — comprehensive cross-method study | [forecasting.ipynb](forecasting/forecasting.ipynb) |
| Model parameter estimation — comprehensive | [parameter_estimation.ipynb](parameter_estimation/parameter_estimation.ipynb) |
| Big-data processing — batch, streaming & distributed | [big_data.ipynb](big_data/big_data.ipynb) |
| Embedded real-time control — comprehensive online ID | [embedded_control.ipynb](embedded_control/embedded_control.ipynb) |
| Real-time GPS/inertial trajectory — control-systems notebook | [realtime_gps.ipynb](realtime_gps/realtime_gps.ipynb) |
| Real-time GPS/inertial trajectory — **well-known-trajectory benchmarks** (exact truth, IMM/CT-EKF baselines) | [benchmark_trajectories.ipynb](realtime_gps/benchmark_trajectories.ipynb) |
| Real-time GPS/inertial trajectory — **hardware rig** (real silicon; now the [`dtfit-hardware`](../../../../../dtfit-hardware/README.md) package) | [realtime_gps_hw.ipynb](../../../../../dtfit-hardware/src/dtfit_hardware/realtime_gps_hw.ipynb) |
| The image showcase -- two public datasets larger than RAM | [image_showcase.ipynb](image_showcase/image_showcase.ipynb) |

## dtfit methods tested vs baselines, per domain

| domain | dtfit methods + established baselines |
|---|---|
| Forecasting | dtfit LSI, EAC, #2 Fourier-LSI, #5 boosting, auto-merged pipeline — on 12 series × 2 horizons (8 measured datasets + 4 physics/signal waveforms: RLC transient, AC+harmonics, AM, chirp), with the structurally-correct model per series, vs random walk, seasonal-naïve, drift, poly-extrap, Holt-Winters ETS, Theta, (S)ARIMA, MLP, LSTM |
| Model parameter estimation | dtfit LSI, EAC, #6 adaptive-EAC, #3 ensemble, #4 joint, merged selector — on 15 nonlinear model families (oscillatory/exp/multi-exp/peak/sigmoid/rational-saturating/power) × an applicability map, noise & outlier sweeps, sparse/transient/short-record/multi-channel regimes, real COVID/FX recovery, vs NLLS, robust NLLS, MLP, Gaussian process |
| Big-data processing | dtfit GEMM batch (fit_lsi_batched), fused streaming PartitionedBatchLSI, distributed merge (#1), streaming EACFilter — 4 multi-channel panels + real 321-channel electricity; exactness, GB-scale memory wall, numerical stability (float32/float64/Kahan), mergeability, O(1)/sample online cost; vs per-channel NLLS, vectorised polynomial lstsq, sklearn SGD partial_fit, RLS |
| Embedded real-time control | dtfit EACFilter, LSIFilter, FilterBank + fused χ² detector + inflate — 4 plant shapes + applicability map, robustness profile (Gaussian noise / outliers / dropout), multi-axis fault detection, deployable footprint, real FX streaming; vs EKF, RLS, constant-accel Kalman, sliding-window refit |
| Real-time GPS/inertial trajectory | dtfit **integral** estimators — streaming LSI/EAC (now with **external regressors**) and a full-IMU strapdown fused *inside the LSI filter* + fused NIS/CUSUM maneuver detector — on a simulated 9-DOF rig (3-D maneuvering target, GPS fixes, 3-axis gyro + accelerometer), with dropouts and multipath glitches; smoothing/forecast, dropout coasting, glitch robustness, maneuver detection, on-MCU budget, generalization batch; **vs the established baselines: constant-acceleration Kalman and the gyro-aided coordinated-turn EKF**. De-risks the planned embedded paper |
| Real-time GPS/inertial trajectory (hardware) — **now the `dtfit-hardware` package** | The **on-silicon** reproduction of the simulated rig: the streaming integral trackers running on an Arduino Nano 33 BLE Sense M4F over a live NEO-M8N fix stream + onboard IMU, with BLE logging — validated against held-out forecasts, surveyed paths and **public RTK/INS datasets** (not a live-cm claim). Host control + telemetry in `dtfit-hardware/src/dtfit_hardware/backend.py`, firmware in `.../firmware/`, plus a phone-side BLE monitor app; build in `papers/embedded_hardware_bom.md`. Turns the de-risked simulation into the embedded paper |
| The image showcase | dtfit `ImageStream` (accumulator, yearly and daily blocks, channel-form GEMM), `dtfit.image.fit` and `assemble` from stored images, `LSIFilter` with `DriftDetector`, the image protocol over TCP -- on 17 GB of NGL GPS series and 51.6 GB of NOAA Global Hourly 2024: exactness against `numpy.linalg.lstsq` on the same rows (gate 1e-8), flat reducer memory, velocities against published MIDAS, step recall against the NGL step database, annual and diurnal amplitudes against NOAA's hourly normals, the whole analysis re-run on a Raspberry Pi 5 from the images alone, and block images streamed between the two machines |

## Relationship to the `cases` suite

The cases isolate one optimization/adaptation per folder; these domains exercise them **together and against the methods a practitioner in that domain actually uses**. Every notebook keeps the cases' honest-negative tone: dtfit trails the classical toolkit on near-RW / irregular series, the integral methods miss weakly-identifiable parameters (Michaelis–Menten), the area filter is the wrong measurement for oscillations (use the spectrum filter), robustness wants a dedicated robust loss over ensembling, streaming trades throughput for bounded memory, and online detection is SNR-limited.