"""Trajectory, front end, filters and baselines for the GPS case study.

``05_gps_trajectory.ipynb`` imports this module and owns the presentation.

The scenario is deliberately unkind to a curve fitter. A GNSS receiver rides a
genuinely maneuvering object, flying a sequence of coordinated turns with
changing heading, speed and climb rate that are integrated into position. No
closed-form per-axis function describes that trajectory, so no estimator here
can be handed one; each tracks a generic constant-acceleration model instead.

dtfit is not a multilateration solver. The front end is a standard
``scipy.optimize.least_squares`` pseudorange solve, and dtfit's job is the
temporal smoothing and short-horizon forecasting of the fix stream that comes
out of it. Its competitor is the constant-acceleration Kalman filter, given the
same information class, the same measurements and the same adaptive re-arming,
so neither side starts with a structural advantage.

:func:`per_satellite_smoothing` is the suite's documented negative result and
is kept for that reason; see its own docstring.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from dtfit.streaming import EACFilter, LSIFilter

from dtfit_experimental.streaming import FilterBank

from dtfit_experimental.experiments.common import baselines as bl

__all__ = [
    "SATS", "MANEUVERS", "MANEUVER_ONSETS", "WARMUP", "INFLATE",
    "trajectory", "pseudoranges", "multilaterate", "rmse3", "build_scenario",
    "per_satellite_smoothing", "FusedManeuverDetector",
    "dtfit_eval", "kalman_eval", "rw_pred", "roll_rmse", "match_onsets",
]

# Satellites high above in km, spread out for a realistic solve geometry.
SATS = np.array([
    [0.0, 0.0, 200.0], [150.0, 20.0, 210.0], [-120.0, 90.0, 190.0],
    [60.0, -140.0, 205.0], [-80.0, -60.0, 220.0], [130.0, 130.0, 198.0],
])


# Flight plan: piecewise-constant controls (t_onset, turn rate omega, speed v,
# climb rate zdot). Heading psi integrates omega, (xdot, ydot) is
# v*(cos psi, sin psi), and position is the integral of that. Heading, speed
# and climb all change at the onsets below. That is what leaves the target
# without a fixed per-axis formula.
MANEUVERS = [
    (0.0,  0.00,  8.0,  1.5),   # climb-out, straight
    (3.0,  0.55,  8.0,  1.5),   # bank into a left turn
    (6.0,  0.00, 13.0,  0.0),   # roll out, accelerate, level off
    (9.0, -0.70, 13.0, -1.0),   # hard right turn while descending
]
MANEUVER_ONSETS = [m[0] for m in MANEUVERS[1:]]  # the true regime-change times

WARMUP = 30  # Skip the filters' fill-window transient when scoring.
# A gentle covariance nudge on a detected maneuver. Aggressive re-arming, say
# x100, backfires: at this measurement SNR the added false-alarm variance costs
# more than the subtle maneuvers it catches.
INFLATE = 3.0


def _cumtrapz(f, t):
    """Cumulative trapezoid integral of f over t, with a leading 0."""
    out = np.zeros_like(f, dtype=float)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))
    return out


def _controls(t):
    """Active (turn-rate, speed, climb-rate) at each time from the flight plan."""
    t = np.asarray(t, float)
    omega = np.zeros_like(t)
    v = np.zeros_like(t)
    zdot = np.zeros_like(t)
    for ts, om, sp, zd in MANEUVERS:
        m = t >= ts
        omega[m], v[m], zdot[m] = om, sp, zd
    return omega, v, zdot


def trajectory(t):
    """Maneuvering object: coordinated turns with changing heading/speed/climb.

    Integrated from the piecewise-constant ``MANEUVERS`` controls. Direction,
    speed and climb angle all change over the flight, leaving no closed-form
    per-axis function for an estimator to be handed.
    """
    t = np.asarray(t, float)
    omega, v, zdot = _controls(t)
    psi = _cumtrapz(omega, t)                      # heading angle
    x = 2.0 + _cumtrapz(v * np.cos(psi), t)
    y = 0.0 + _cumtrapz(v * np.sin(psi), t)
    z = 5.0 + _cumtrapz(zdot, t)
    return np.stack([x, y, z], axis=-1)


def pseudoranges(p, rng, sigma=0.6, clock_bias=1.5):
    """Noisy pseudoranges from a position p (n,3) to each satellite."""
    d = np.linalg.norm(p[:, None, :] - SATS[None, :, :], axis=2)  # (n, nsat)
    return d + clock_bias + rng.normal(0, sigma, d.shape)


def multilaterate(rho_row, p0):
    """Solve one epoch's (x,y,z,bias) from satellite pseudoranges (NLLS)."""
    def resid(s):
        p, b = s[:3], s[3]
        return np.linalg.norm(p[None, :] - SATS, axis=1) + b - rho_row
    sol = least_squares(resid, np.r_[p0, 0.0], method="lm")
    return sol.x[:3]


def rmse3(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def build_scenario(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, n)
    truth = trajectory(t)
    rho = pseudoranges(truth, rng)
    # Instantaneous multilateration fixes, the standard GNSS solve. Each epoch
    # starts from the previous fix, as a receiver would.
    fixes = np.zeros_like(truth)
    p0 = truth[0]
    for i in range(n):
        fixes[i] = multilaterate(rho[i], p0)
        p0 = fixes[i]
    return t, truth, rho, fixes


def per_satellite_smoothing(t, rho, seed=0):
    """Smooth each pseudorange stream with its own EAC filter, then solve.

    This is the suite's documented negative result, kept because the lesson is
    worth more than the number. Multilateration needs the ranges at one instant
    to be mutually consistent; smoothing each stream independently gives every
    satellite its own lag. That destroys the synchrony and sends the geometry
    solve far off. Smooth at the trajectory level, not the raw-range level.
    """
    nsat = rho.shape[1]
    bank = FilterBank.from_model(
        "a + b*t", "t", nsat, p0=[rho[0].mean(), 0.0],
        window_size=20, q_diag=[1e-1, 1e-2], order=2)
    out = bank.run(t, rho, n_jobs=1, track=True)
    smoothed = out["track"]
    # Solve from the smoothed ranges, falling back to the raw row wherever the
    # bank has no finite track yet.
    n = t.size
    fixes = np.zeros((n, 3))
    p0 = trajectory(t[:1])[0]
    for i in range(n):
        row = smoothed[i]
        if np.any(np.isnan(row)):
            row = rho[i]
        fixes[i] = multilaterate(row, p0)
        p0 = fixes[i]
    return fixes


class FusedManeuverDetector:
    """Fuse several streams' one-step innovations into one chi-square(K) NIS
    statistic and run a CUSUM on it.

    A maneuver moves several axes at once, so the fused statistic carries a far
    better signal-to-noise ratio than any single axis. That is why a per-axis
    detector on noisy position fixes misses subtle coordinated turns: each axis
    alone reaches only ~1.4-3.2x baseline, while fused they reach ~4x. The
    CUSUM then accumulates sustained surprise, so a brief onset transient is
    not averaged away the way the full-window area statistic averages it.

    Two entry points keep the comparison fair. ``update_residuals``
    standardizes raw residuals by a running EWMA scale, for dtfit, whose
    innovation covariance is not readily available; ``update_nis`` consumes an
    already-normalized NIS, for the Kalman, which provides one. Both feed the
    identical CUSUM.
    """

    def __init__(self, k: int, *, ewma: float = 0.05, slack: float = 1.5,
                 h: float = 10.0, warmup: int = 60) -> None:
        self.k = k
        self.ewma = ewma
        self.slack = slack
        self.h = h
        self.warmup = warmup
        self._sc2 = np.zeros(k)
        self._g = 0.0
        self._n = 0

    def _cusum(self, nis: float) -> bool:
        self._n += 1
        if self._n <= self.warmup:
            return False
        # Accumulate surprise above the expected dof plus slack; fire on
        # crossing h, then re-arm from zero.
        self._g = max(0.0, self._g + nis - self.k - self.slack)
        if self._g > self.h:
            self._g = 0.0
            return True
        return False

    def update_residuals(self, residuals) -> bool:
        r = np.asarray(residuals, dtype=float)
        if not np.all(np.isfinite(r)):
            return False
        z2 = np.where(self._sc2 > 0.0, r * r / np.where(self._sc2 > 0.0, self._sc2, 1.0), 0.0)
        self._sc2 = (1.0 - self.ewma) * self._sc2 + self.ewma * r * r
        return self._cusum(float(z2.sum()))

    def update_nis(self, nis: float) -> bool:
        if not np.isfinite(nis):
            return False
        return self._cusum(float(nis))


def _make_axis_filters(kind, fixes, off):
    """Build the 3 per-axis dtfit filters for the generic CA quadratic model.

    ``kind="eac"`` gives EACFilter and its window-area measurement;
    ``kind="lsi"`` gives the streaming sibling LSIFilter, projecting each
    window onto the first ``order+1`` Legendre moments instead. Both share the
    same model, window and p0, so a difference between them is a difference in
    the measurement and not in the tuning.
    """
    model = "c0 + c1*t + c2*t**2"  # Local CA model; params sort c0, c1, c2.
    if kind == "lsi":
        return [LSIFilter(
            model, "t", p0=[float(fixes[0, ax]), 0.0, 0.0], window_size=15,
            order=3, q_diag=[1e-2, 1e-2, 1e-2],
            drift_reset="inflate", **off) for ax in range(3)]
    return [EACFilter(
        model, "t", p0=[float(fixes[0, ax]), 0.0, 0.0], window_size=15,
        q_diag=[1e-2, 1e-2, 1e-2], order=3,
        drift_reset="inflate", **off) for ax in range(3)]


def dtfit_eval(t, fixes, horizons, *, kind="eac", fused=False, inflate=INFLATE,
               det_h=10.0):
    """Online per-axis tracking with a generic model, plus rolling forecasts.

    Nothing about the maneuver is given away. Each axis is tracked by a local
    constant-acceleration quadratic ``c0 + c1*t + c2*t**2`` over a short
    sliding window, the same information class as the Kalman baseline.

    ``kind`` selects the measurement, ``"eac"`` for area or ``"lsi"`` for the
    Legendre spectrum. With ``fused=False`` each filter uses its own per-axis
    detector. With ``fused=True`` that is switched off and re-adaptation is
    driven by a :class:`FusedManeuverDetector` over the three axes' forecast
    residuals, so the maneuver is detected from the joint innovation and all
    three axes are re-armed together by covariance inflation.

    Returns ``(smoothed, pred, drift_times)``.
    """
    n = t.size
    smoothed = np.zeros((n, 3))
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    drift_times: set[float] = set()
    # Fused mode mutes the internal per-axis detector with a huge NIS
    # threshold and no CUSUM, leaving the fused detector as the only driver of
    # re-adaptation.
    off = dict(alpha=1e-12, cusum_h=float("inf")) if fused else {}
    flts = _make_axis_filters(kind, fixes, off)
    det = FusedManeuverDetector(3, h=det_h) if fused else None
    for i in range(n):
        for ax in range(3):
            flts[ax].partial_fit(t[i], fixes[i, ax])
            if not fused and flts[ax].drift_flag_:
                drift_times.add(round(float(t[i]), 2))
            smoothed[i, ax] = float(flts[ax].predict(np.array([t[i]]))[0])
        if det is not None and det.update_residuals(
                [flts[ax].last_residual_ for ax in range(3)]):
            drift_times.add(round(float(t[i]), 2))
            for ax in range(3):
                flts[ax].inflate(inflate)  # Re-arm every axis, not just one.
        for h in horizons:
            if i + h < n:
                for ax in range(3):
                    pred[h][i + h, ax] = float(flts[ax].predict(np.array([t[i + h]]))[0])
    return smoothed, pred, sorted(drift_times)


def kalman_eval(t, fixes, horizons, q=5e-2, *, adaptive=False, inflate=INFLATE,
                det_h=10.0):
    """Constant-acceleration Kalman tracking with rolling h-step forecasts.

    ``adaptive=True`` applies the same fused-NIS CUSUM re-arming the adaptive
    dtfit filter gets, inflating the covariance on a detected maneuver. The
    gold-standard baseline is extended the identical courtesy.
    """
    kf = bl.KalmanCA(dim=3, dt=float(t[1] - t[0]), q=q, r=0.5)
    det = FusedManeuverDetector(3, h=det_h) if adaptive else None
    n = t.size
    smoothed = np.zeros((n, 3))
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    drift_times: set[float] = set()
    hmax = max(horizons)
    for i in range(n):
        smoothed[i] = kf.update(fixes[i])
        if det is not None and det.update_residuals(kf.last_residuals_):
            drift_times.add(round(float(t[i]), 2))
            kf.inflate(inflate)
        fc = kf.forecast(hmax)  # (hmax, 3); forecasting must not mutate kf.
        for h in horizons:
            if i + h < n:
                pred[h][i + h] = fc[h - 1]
    return smoothed, pred, sorted(drift_times)


def rw_pred(fixes, horizons):
    """Random-walk forecast: position persists, making the h-ahead estimate
    the last fix seen."""
    n = fixes.shape[0]
    pred = {h: np.full((n, 3), np.nan) for h in horizons}
    for h in horizons:
        pred[h][h:] = fixes[: n - h]
    return pred


def roll_rmse(pred_h, truth):
    """Position RMSE of a rolling forecast, minus NaN and warm-up rows."""
    mask = ~np.isnan(pred_h[:, 0])
    mask[:WARMUP] = False
    return rmse3(pred_h[mask], truth[mask])


def match_onsets(flags, onsets=None):
    """(onsets caught, false alarms) for a list of flagged drift times.

    A flag counts as a catch from 0.3 before an onset to 1.5 after it on the
    trajectory's time axis; the asymmetry allows for the lag a windowed filter
    needs before the break reaches its window.
    """
    onsets = MANEUVER_ONSETS if onsets is None else onsets
    caught = sum(any(o - 0.3 <= f <= o + 1.5 for f in flags) for o in onsets)
    fa = sum(not any(o - 0.3 <= f <= o + 1.5 for o in onsets) for f in flags)
    return caught, fa
