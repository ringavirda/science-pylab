# The experimental package -- adaptations and validation

`dtfit-experimental` is a **separate package** that sits on top of stable
`dtfit`. It is where new ways of *composing* the core methods are prototyped,
evaluated across a large experiment suite, and -- if they prove themselves --
**promoted** into the stable `dtfit` API (where they then physically live).
Nothing here ships inside the published `dtfit` wheel, so the public API stays
lean.

This page explains, accessibly:

- [the promotion model](#promotion) -- how an idea graduates from experimental to stable;
- [the adaptations](#adaptations) -- the structural extensions still in trial,
  each with intuition + the math it rests on;
- [the experiment suite](#suite) -- the *cases* and *domains* studies that decide
  promotion;
- and, in a companion file, [every baseline](Experimental-Baselines) the methods are
  compared against and **why each was selected**.

Source: [`packages/dtfit-experimental/`](https://github.com/ringavirda/science-nonline/blob/main/packages/dtfit-experimental).

---

<a name="promotion"></a>
## 1. The promotion model -- why two packages

The project separates **what is proven** from **what is being tried**:

- `dtfit` (stable) -- the methods and adaptations that have been validated across
  multiple application domains. This is the lean, public, supported API.
- `dtfit-experimental` -- new adaptations, the full benchmark/validation suite,
  and the datasets. It *depends on* `dtfit`; it is never depended on *by* it.

An adaptation graduates only after the experiment suite shows it helps across a
*range* of applications, not just one cherry-picked case. On promotion it is
**physically moved** into `dtfit` and imported from there -- there is no
re-export shim, so the dependency only ever points one way.

**Promoted** (in stable `dtfit`):

| adaptation | in `dtfit` as |
|---|---|
| the LSI oscillatory recipe | `fit_lsi(oscillatory=..., freq_param=...)`, `fft_frequency_seed` |
| fused multi-axis fault detection | `FusedChiSquareDetector` |
| #3 overlapping-window ensemble | retired -- the robust image (`fit_eac(robust=True)`) |

The map-reduce estimators (`PartitionedLSI`, `PartitionedEAC`) and the
GEMM-batched projection (`fit_lsi_batched`, `project_spectra`,
`PartitionedBatchLSI`) live in `dtfit_experimental.scale`, covered by
`ImageStream` and its channel axis; the curvature-adaptive windows are
retired, `fit_eac` places equal windows.

**Still experimental** (the three adaptations below): `fit_lsi_basis`,
`fit_joint`, `boosted_fit`. The inverse-covariance **`InformationFilter`** (an
information-form fusion primitive) also lives in this tier -- it was **moved out of
stable `dtfit`** because it is exercised by no domain study and shares no code with
the covariance-form `EACFilter` / `LSIFilter`, so it has not cleared the
>=2-domain promotion gate (`from dtfit_experimental import InformationFilter`).

---

<a name="adaptations"></a>
## 2. The adaptations still in trial

Every adaptation is grounded in the *same* math the core methods use -- the
**linearity of integration**, **orthogonal-basis projection**, and the
**additivity of areas**. None of them is an ad-hoc trick; each is a structural
recombination of the existing fingerprint machinery.

### #2 -- Pluggable basis LSI (`fit_lsi_basis`)

**Intuition.** LSI matches fingerprints on the Legendre (polynomial) basis. But
the "best measuring sticks" depend on the signal: a **periodic** signal needs
*many* polynomial orders to express a wiggle, whereas a **Fourier** basis
(sines/cosines) captures it in two or three harmonics; a pure **decay** is
natural in a **Laguerre** basis. This adaptation keeps LSI's exact criterion but
lets you choose the basis to match the signal -- fewer coefficients, better
conditioning.

**The math it rests on.** The LSI derivation (see
[../methods/lsi.md](Methods-LSI)) only needs the basis to be *orthogonal*
on the interval; nothing about it is specific to Legendre. Swap in any orthogonal
family and the same diagonal least-squares match holds.

### #3 -- Overlapping-window ensemble -- **retired**

Retired: the robust image (`fit_eac(robust=True)`) covers the same
contamination at a twelfth of the cost and a pooled median recovery error of
0.008 against the ensemble's 0.065 (five families, 4 percent of samples
replaced by 8-sigma spikes, six draws).

### #4 -- Joint shared-parameter fit (`fit_joint`)

**Intuition.** Often several channels share structure -- the x/y/z axes of a
trajectory share a common frequency; several regions share a growth rate; a
multi-output plant shares a time constant. Fitting each channel alone throws that
coupling away. `fit_joint` stacks **all** channels' area equations into one big
system, with the **shared** parameters estimated jointly from every channel and
the **per-channel private** parameters estimated locally, solved in a single pass.
More equations per shared unknown means you observe it better than any channel
could alone.

**The math it rests on.** EAC's area equations are just rows of a least-squares
system; rows from different channels referring to the same shared parameter
simply stack. The stacked system is still linear in the residual/Jacobian
structure EAC already builds.

### #5 -- Stage-wise residual boosting (`boosted_fit`)

**Intuition.** One parametric form may not capture *both* a trend and a cycle.
Boosting stages the methods: fit stage 1 (say an LSI exponential/polynomial
trend), subtract its prediction, fit stage 2 (say an EAC-fitted oscillatory
residual) to what's left, and sum the stages. Each stage stays a cheap,
well-posed fit, but the composite is more expressive than either method alone.

**The math it rests on.** Because the fingerprint transform is **linear**, the
fingerprint of a sum of components is the sum of their fingerprints -- so fitting
components one at a time and adding them is consistent with matching the whole
signal's fingerprint. (This is the additive, gradient-boosting idea applied to
parametric curve components.)

**Full signatures, arguments and return types** for the four experimental
adaptations (plus the array-backend helpers) are in
[adaptations-api.md](Experimental-Adaptations-API); the promoted ones are in
[../api/](API).

---

<a name="suite"></a>
## 3. The experiment suite -- how adaptations are judged

There are **two** complementary suites, both driven by one shared runner:

### `cases/` -- each lever in isolation

Ten focused experiments, each isolating *one* optimization or adaptation
(control systems, big-data streaming, noise robustness, real-world forecasting,
GPS trajectory, an LTSF deep-learning benchmark, parallel scaling, GPU-batched
projection, embedded footprint, fused partitioned-batched). The point is to
measure each lever cleanly, on its own.

Each case is a self-contained folder with a `backend.py` (the compute) and a
Jupyter notebook (the report); open/run the notebook directly (index:
`cases/REPORTS.md`).

### `domains/` -- the levers together, against the real toolkit

Six **application-domain** studies, each testing *every applicable dtfit method*
against the **established methods a practitioner in that domain actually uses**,
on synthetic *and real* data. This is the suite that decides whether an
adaptation earns promotion. The six domains and their honest headline results:

| domain | what it tests | headline result |
|---|---|---|
| **Forecasting** | LSI, EAC, Fourier-LSI, boosting, the auto-merged pipeline -- on 12 series x 2 horizons -- vs random walk, seasonal-naive, drift, poly-extrap, Holt-Winters, Theta, (S)ARIMA, MLP, LSTM | dtfit wins where the series has real *extrapolable nonlinear structure*; trails the general learners on near-random-walk / irregular series (and says so) |
| **Parameter estimation** | LSI, EAC, adaptive-EAC, ensemble, joint, the merged selector -- across 15+ nonlinear model families, noise/outlier/sparse/short/multi-channel regimes, real recovery -- vs NLLS, robust NLLS, MLP, Gaussian process | with the **shape-matched variant**, dtfit's integral estimators **tie the NLLS gold standard** across the families; pointwise NLLS keeps a slight edge only on the heavy-tailed Lorentzian |
| **Big-data processing** | GEMM batch, fused streaming, distributed merge, streaming filter -- multi-channel panels + a real 321-channel set -- exactness, memory/throughput scaling, numerical stability, mergeability, online cost -- vs per-channel NLLS, vectorized poly lstsq, SGD `partial_fit`, RLS | the additive projection is exact across batch/streaming/distributed routes and scales with bounded memory; trades peak throughput for that bounded memory |
| **Embedded control** | EACFilter, LSIFilter, FilterBank + fused chi^2 detector -- 4 plant shapes, robustness profile, multi-axis fault detection, sub-KiB footprint, real streaming -- vs EKF, RLS, constant-accel Kalman, sliding-window refit | the *integral* measurement wins under outliers/dropouts at fixed O(1)/sample cost; online fault detection is SNR-limited (and reported as such) |
| **Real-time GPS/inertial** | streaming LSI/EAC with external regressors + a full-IMU strapdown fused *inside* the LSI filter + fused NIS/CUSUM maneuver detector -- 9-DOF maneuvering-target rig, dropouts/multipath, plus well-known-trajectory benchmarks and a hardware rig (`dtfit-hardware`) -- vs constant-accel Kalman and the gyro-aided coordinated-turn EKF | the integral trackers match/beat the constant-accel Kalman and hold up in benign (static/pedestrian) regimes, but structurally **trail the CT-EKF on aggressive maneuvers**; the coast/glitch/on-MCU results de-risk the embedded paper |
| **Stochastic series** | the stochastic tier (`fit_stochastic` / `StochasticModel` / `StochasticFilter`) recovering long-memory Hurst, AR(1) reversion, GARCH persistence, stochastic-cycle period and trend+cycle from a process's *functionals* -- vs the standard estimator for each (aggregated-variance/GPH, ACF-exp, GARCH-QMLE, ...) | dtfit recovers the **regime and its parameters** with a single coherent estimator->forecast->generator API; it cannot out-forecast a martingale and a dedicated MLE (GARCH-QMLE) stays a touch sharper on the raw parameter (reported per possibility as VIABLE / MARGINAL / NOT VIABLE) |

Each domain is a self-contained folder with a `backend.py` (the compute) and a
Jupyter notebook (the report); open/run the notebook directly (index:
`domains/DOMAINS.md`).

Every report keeps an **honest-negative tone**: where dtfit trails the classical
toolkit, the report says so and explains why (near-random-walk series, weakly
identifiable parameters, the area filter being the wrong measurement for
oscillations, etc.). That honesty is the point of the suite -- promotion requires a
*broad* win, not a cherry-picked one.

---

## 4. Install & run

```bash
pip install -e packages/dtfit                       # stable dtfit
pip install -e packages/dtfit-experimental          # this package
pip install -e "packages/dtfit-experimental[bench]" # + matplotlib/torch/statsmodels/pandas

python -m dtfit_experimental.experiments.download_data       # fetch real datasets

# the experiments are Jupyter notebooks -- open one and re-run it, or headless:
jupyter nbconvert --to notebook --execute --inplace \
    packages/dtfit-experimental/src/dtfit_experimental/experiments/cases/01_control_systems/01_control_systems.ipynb
```

Each notebook has a config block of knobs near the top (sized for a few-minute run
by default; comments show how to scale up). Next:
**[baselines.md](Experimental-Baselines)** -- what every comparison method is and why it was
chosen.
