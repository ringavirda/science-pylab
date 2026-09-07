"""Latency, state-size and MCU-fit measurements for the embedded case study.

``09_embedded_footprint.ipynb`` imports this module and owns the presentation.

One question drives it: can the streaming filters run on the sort of
microcontroller you would bolt to a GPS module, an Arduino, STM32 or ESP32?
Three things decide that. Each is measured separately: the state-size
functions count the words a minimal, no-malloc C port must keep resident, which
is the number that has to fit in SRAM. The latency functions time one update of
each estimator. The :data:`MCUS` datasheet table and the :data:`CONFIGS` filter
setups then pair the two against real parts.

The two halves are not equally direct evidence, and it matters which is which.
NumPy does not run on an AVR, so the latency figures are a desktop reference
for the shape of the algorithm rather than a prediction of MCU timing; the
memory figures are the C-struct size that genuinely deploys.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
from scipy.optimize import curve_fit


from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "CONFIGS", "MCUS",
    "state_doubles_eac", "state_bytes", "state_doubles_lsi_ram",
    "const_doubles_lsi", "state_doubles_kalman", "approx_flops_per_update",
    "measure_latency", "measure_kalman_latency", "measure_curvefit_latency",
    "measure_resident_python",
]


# The filter setups that get measured and sized, one per realistic embedded
# job: (label, model expr, var, p0, filter kwargs, n params, window W).
CONFIGS = [
    ("CA quadratic (GPS axis)", "c0 + c1*t + c2*t**2", "t", [0.0, 0.0, 0.0],
     dict(window_size=15, q_diag=[1e-2] * 3, order=3), 3, 15),
    ("damped sine (control ID)", "A*exp(-d*t)*sin(w*t)", "t", [1.0, 0.1, 1.0],
     dict(window_size=50, q_diag=[1e-2] * 3, order=3), 3, 50),
    ("linear (range smoother)", "a + b*t", "t", [0.0, 0.0],
     dict(window_size=20, q_diag=[1e-1, 1e-2], order=2), 2, 20),
]


# MCU classes: name, SRAM bytes, clock MHz, has FPU, effective MFLOP/s. SRAM
# and clock are datasheet values; the MFLOP/s figure is a deliberately rough
# order-of-magnitude estimate, penalised where soft-float is the only option.
MCUS = [
    ("AVR ATmega328 (Uno/Nano)", 2 * 1024, 16, False, 0.05),
    ("ARM Cortex-M0+ (SAMD21/Zero)", 32 * 1024, 48, False, 0.3),
    ("ARM Cortex-M4F (STM32F4/Teensy)", 192 * 1024, 168, True, 30.0),
    ("ESP32 (Xtensa LX6 FPU)", 520 * 1024, 240, True, 40.0),
]


# Deployable state size: the C struct, the number that has to fit.
def state_doubles_eac(n: int, w: int, n_sub: int = 2) -> int:
    """Floating-point words a minimal C port of EACFilter keeps alive between
    samples, as one fixed-size, no-malloc struct:

      * ring buffer  t[W], y[W]                        -> 2W
      * parameter estimate p[n]                        -> n
      * covariance P[n*n]                              -> n*n
      * process-noise diagonal Q[n]                    -> n
      * measurement and detector scalars (R, EWMA
        scales, CUSUM arms, counters)                  -> 8

    Scratch for the per-step solve (h_mat, S, gain) is transient and lives on
    the stack, so it is not counted as resident state.
    """
    return 2 * w + n * n + 2 * n + 8


def state_bytes(n: int, w: int, dtype_bytes: int) -> int:
    return state_doubles_eac(n, w) * dtype_bytes


def state_doubles_lsi_ram(n: int, w: int) -> int:
    """Mutable per-sample RAM words for LSIFilter: the ring buffer (2W), the
    covariance P (n*n), one n-word parameter vector and about 9 scalars.

    The filter also precomputes projection and quadrature tables. Those are
    read-only constants and belong in flash or PROGMEM rather than SRAM, so
    they are counted separately by :func:`const_doubles_lsi`.
    """
    return 2 * w + n * n + n + 9


def const_doubles_lsi(n: int, w: int, order: int) -> int:
    """Read-only tables the Legendre filter precomputes; they belong in flash,
    not SRAM: the projection pseudo-inverse (order+1)*W, the Gauss-Legendre
    nodes and weights (2*n_quad), the quadrature Vandermonde n_quad*(order+1),
    and the per-order weights 2*(order+1)."""
    n_quad = max(2 * (order + 1), 16)
    return (order + 1) * w + 2 * n_quad + n_quad * (order + 1) + 2 * (order + 1)


def state_doubles_kalman(dim: int = 3) -> int:
    """Mutable RAM words for the CA Kalman: state x[3] plus covariance P[3x3]
    comes to 12 per axis. The transition, noise and measurement matrices
    F, Q, H, R are constants and can sit in flash. The Kalman keeps no history
    window at all, making it leaner than the integral filters. That is a real
    architectural difference and is reported as one.
    """
    return 12 * dim


def approx_flops_per_update(n: int, w: int, n_sub: int = 2) -> int:
    """Rough FLOP count for one EACFilter update with a degree-(n-1)
    polynomial model.

    Evaluating the model and its n derivatives over the W-point window
    dominates, at roughly 2*W*n each, with the small (n_sub x n) Kalman algebra
    on top. Order-of-magnitude only; it exists for the MCU compute sanity
    check, not as a cycle count.
    """
    model_eval = 2 * w * n          # Horner over the window
    jac_eval = 2 * w * n * n        # n derivative rows over the window
    integrate = 2 * w * (n + 1)     # Simpson sums for e_vec and the h_mat rows
    kalman = n * n * n_sub + n_sub ** 3 + n * n_sub ** 2 + n * n
    return model_eval + jac_eval + integrate + kalman


# Measured per-sample latency.
def measure_latency(make_filter, n_warm: int, n_timed: int = 4000) -> float:
    """Mean wall-clock us per `partial_fit` on a warmed-up filter."""
    flt = make_filter()
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, n_warm + n_timed)
    y = np.sin(3 * t) + rng.normal(0, 0.1, t.size)
    for i in range(n_warm):  # Fill the window; only steady state is timed.
        flt.partial_fit(float(t[i]), float(y[i]))
    t0 = time.perf_counter()
    for i in range(n_warm, n_warm + n_timed):
        flt.partial_fit(float(t[i]), float(y[i]))
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_kalman_latency(n_timed: int = 4000) -> float:
    """Mean us per 3-axis CA Kalman update (recursive, no window)."""
    kf = bl.KalmanCA(dim=3, dt=0.03, q=5e-2, r=0.5)
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (n_timed + 10, 3))
    for i in range(10):
        kf.update(z[i])
    t0 = time.perf_counter()
    for i in range(10, 10 + n_timed):
        kf.update(z[i])
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_curvefit_latency(w: int, n_timed: int = 400) -> float:
    """Mean us per step for the batch alternative to a streaming filter.

    Refits a CA quadratic on the trailing W-sample window with unbounded
    `scipy.optimize.curve_fit`, which takes the Levenberg-Marquardt path, at
    every new sample. One axis only.
    """
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, w + n_timed)
    y = 1.0 + 2.0 * t + 0.5 * t**2 + rng.normal(0, 0.1, t.size)

    def f(tt, c0, c1, c2):
        return c0 + c1 * tt + c2 * tt**2

    p0 = np.zeros(3)
    t0 = time.perf_counter()
    for i in range(w, w + n_timed):
        tw, yw = t[i - w:i], y[i - w:i]
        try:
            p0, _ = curve_fit(f, tw, yw, p0=p0, maxfev=2000)
        except Exception:
            pass
    return (time.perf_counter() - t0) / n_timed * 1e6


def measure_resident_python(make_filter, n_warm: int) -> int:
    """Python-object resident bytes of one warmed filter, via tracemalloc.

    This is the interpreter footprint, not the embeddable struct. It is
    reported only to show how far the Python objects inflate the algorithmic
    state that a C port would actually carry.
    """
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    flt = make_filter()
    t = np.linspace(0, 1, n_warm)
    for ti in t:
        flt.partial_fit(float(ti), float(np.sin(3 * ti)))
    snap = tracemalloc.take_snapshot()
    stats = snap.compare_to(base, "filename")
    total = sum(s.size_diff for s in stats)
    tracemalloc.stop()
    return max(total, 0)
