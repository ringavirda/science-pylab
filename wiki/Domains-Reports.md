# dtfit domain validation suite -- index

Where the `cases` suite evaluates each EAC/LSI adaptation in isolation, this suite is a **comprehensive per-application-domain study**: each domain tests every applicable dtfit method against the **established domain-standard methods**, on synthetic *and real* data, checking validity, applicability and usefulness. Each domain is a `backend.py` (the compute) + a Jupyter notebook (the report) that opens with a detailed *Methods under test* and *Baseline methods* section.


## Domains

| domain | notebook | status | runtime (s) |
|---|---|---|---|
| Forecasting -- comprehensive cross-method study | [forecasting.ipynb](Domain-Forecasting) | ok | 33 |
| Model parameter estimation -- comprehensive | [parameter_estimation.ipynb](Domain-Parameter-Estimation) | ok | 15 |
| Big-data processing -- batch, streaming & distributed | [big_data.ipynb](Domain-Big-Data) | ok | 53 |
| Embedded real-time control -- comprehensive online ID | [embedded_control.ipynb](Domain-Embedded-Control) | ok | 6 |
| Real-time GPS/inertial trajectory -- control-systems sensor fusion | [realtime_gps.ipynb](Domain-Realtime-GPS) | ok | -- |
| Real-time GPS/inertial -- **on-silicon hardware rig** | [Domain-Realtime-GPS-Hardware](Domain-Realtime-GPS-Hardware) ([realtime_gps_hw.ipynb](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-hardware/src/dtfit_hardware/realtime_gps_hw.ipynb)) | ok | 50 |
| Stochastic series -- dtfit on random data | [stochastic_series.ipynb](Domain-Stochastic-Series) | ok | -- |
| The image showcase -- two public datasets larger than RAM | [image_showcase.ipynb](Domain-Image-Showcase) | ok | -- |

## dtfit methods tested vs baselines, per domain

| domain | dtfit methods + established baselines |
|---|---|
| Forecasting | dtfit LSI, EAC, #2 Fourier-LSI, #5 boosting, auto-merged pipeline -- on 12 series x 2 horizons (8 measured datasets + 4 physics/signal waveforms: RLC transient, AC+harmonics, AM, chirp), with the structurally-correct model per series, vs random walk, seasonal-naive, drift, poly-extrap, Holt-Winters ETS, Theta, (S)ARIMA, MLP, LSTM |
| Model parameter estimation | dtfit LSI, EAC, #6 adaptive-EAC, #3 ensemble, #4 joint, merged selector -- on 16 nonlinear model families (oscillatory/exp/multi-exp/peak/sigmoid/rational-saturating/power) x an applicability map, noise & outlier sweeps, sparse/transient/short-record/multi-channel regimes, real COVID/FX recovery, vs NLLS, robust NLLS, MLP, Gaussian process |
| Big-data processing | dtfit GEMM batch (fit_lsi_batched), fused streaming PartitionedBatchLSI, distributed merge (#1) -- in `dtfit_experimental.scale`, covered by `ImageStream` -- and streaming EACFilter -- 4 multi-channel panels + real 321-channel electricity; exactness, GB-scale memory wall, numerical stability (float32/float64/Kahan), mergeability, O(1)/sample online cost; vs per-channel NLLS, vectorised polynomial lstsq, sklearn SGD partial_fit, RLS |
| Embedded real-time control | dtfit EACFilter, LSIFilter, FilterBank + fused chi^2 detector + inflate -- 4 plant shapes + applicability map, robustness profile (Gaussian noise / outliers / dropout), multi-axis fault detection, deployable footprint, real FX streaming; vs EKF, RLS, constant-accel Kalman, sliding-window refit |
| Real-time GPS/inertial trajectory | dtfit streaming LSIFilter/EACFilter (now with external regressors) + full-IMU strapdown fused inside the LSI filter + fused NIS/CUSUM maneuver detector -- on a simulated 9-DOF rig (3-D maneuvering target, GPS fixes, 3-axis gyro + accelerometer + magnetometer) with dropouts & multipath glitches: smoothing/forecast, dropout coasting, glitch robustness, maneuver detection, on-MCU budget, generalization batch; well-known-trajectory benchmarks (exact truth, CT-EKF/IMM) + public RTK/INS datasets; vs constant-accel Kalman, gyro-aided coordinated-turn EKF. On-silicon twin in `dtfit-hardware` |
| Real-time GPS/inertial -- **hardware rig** | the on-silicon reproduction ([Domain-Realtime-GPS-Hardware](Domain-Realtime-GPS-Hardware)): on-MCU float32 streaming LSI on an Arduino Nano 33 BLE Sense + NEO-M8N, real logged car drives (1 Hz & 5 Hz), on-MCU cost (~267 us/update, sub-kB, golden 4.4e-16), complementary-filter gyro fusion, + public **comma2k19** highway and **UrbanNav** deep-urban benchmarks with absolute truth; vs Kalman-CA, CT-EKF |
| Stochastic series | dtfit fit_stochastic, StochasticModel.simulate, StochasticFilter, the functional estimators (Hurst / AR(1) / GARCH / cycle), vendored ADF (promoted to `dtfit.stochastic`) -- 6 process families (ARFIMA, AR(1)/OU, GARCH, AR(2) cycle, trend+cycle) recovered vs known truth + regime router, forecast skill, 7-series real gallery, generative round-trip, streaming tracking + break detection; vs OLS/GPH, R/S, DFA, lag-1 ACF, FFT-peak, random walk, drift, AR(1), ARIMA, ETS, Theta, seasonal-naive, LSIFilter (filter cost) |
| The image showcase | dtfit `ImageStream` (accumulator, yearly and daily blocks, channel-form GEMM), `dtfit.image.fit` and `assemble` from stored images, `LSIFilter` with `DriftDetector`, the image protocol over TCP -- on 17 GB of NGL GPS series and 49 GB of NOAA Global Hourly 2024: exactness against `numpy.linalg.lstsq` on the same rows (gate 1e-8), flat reducer memory, velocities against published MIDAS, step recall against the NGL step database, annual and diurnal amplitudes against NOAA's hourly normals, the whole analysis re-run on a Raspberry Pi 5 from the images alone, and block images streamed between the two machines |

## Relationship to the `cases` suite

The cases isolate one optimization/adaptation per folder; these domains exercise them **together and against the methods a practitioner in that domain actually uses**. Every report keeps the cases' honest-negative tone: dtfit trails the classical toolkit on near-RW / irregular series, the integral methods miss weakly-identifiable parameters (Michaelis-Menten), the area filter is the wrong measurement for oscillations (use the spectrum filter), robustness wants a dedicated robust loss over ensembling, streaming trades throughput for bounded memory, and online detection is SNR-limited.


