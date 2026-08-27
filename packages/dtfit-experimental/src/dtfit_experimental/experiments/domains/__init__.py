"""Domain validation suite for the dtfit methods.

Where ``experiments/cases/`` evaluates each EAC/LSI adaptation in isolation,
one optimization or structural idea per folder scored on the promotion matrix,
this suite asks a different question: for each real application domain, which
combination of the methods is best, and does the merged pipeline hold up in a
realistic setting?

Each domain is its own folder holding a ``backend.py`` (the single source of
truth for its simulation, estimation and data infrastructure) and a notebook
that imports it and writes the report, its tables and figures, into
``figures/``:

* ``forecasting``: structured fit-then-extrapolate, merging the trend (LSI) and
  seasonal (Fourier-basis, boosting) levers into one auto-composed forecaster.
* ``parameter_estimation``: physical-parameter recovery, merging batch LSI/EAC,
  adaptive-window EAC for transients and joint multi-channel fitting behind a
  scenario-aware selector, against the NLLS gold standard.
* ``big_data``: batch and streaming both, merging the map-reduce estimators
  (PartitionedLSI/EAC), the fused multi-channel ``PartitionedBatchLSI`` and the
  whole-array GEMM projection behind a regime dispatcher.
* ``embedded_control``: real-time online identification, merging the streaming
  filters, the multi-stream ``FilterBank`` and the fused drift detector with
  the deployable embedded-footprint accounting, against a Kalman baseline.
* ``realtime_gps``: streaming LSI/EAC with external regressors, and a full-IMU
  strapdown fused inside the LSI filter, against a constant-accel Kalman and a
  gyro-aided coordinated-turn EKF, on a simulated 9-DOF rig.
* ``stochastic_series``: can the deterministic fitters touch a random series,
  economic or financial data? It fits dtfit to a stochastic process's
  deterministic functionals (autocovariance, spectrum, aggregated variance,
  trend/cycle) and scores which parameter-recovery routes are viable (Hurst and
  long memory, mean reversion, volatility persistence, stochastic cycle)
  against known-truth simulators.

The hardware twin of ``realtime_gps``, which drives the real Arduino Nano 33
BLE and NEO-M8N rig on silicon, lives in ``packages/dtfit-hardware`` (firmware,
USB/BLE host link, phone app) and depends on this package's ``realtime_gps``.

Open a notebook directly (``jupyter lab forecasting/forecasting.ipynb``) or run
it headless through ``jupyter nbconvert --execute``; ``DOMAINS.md`` indexes
them. The backends share ``experiments/common`` (metrics, baselines, datasets,
plotting) with the case experiments, and that shared floor keeps the two
consistent.
"""
