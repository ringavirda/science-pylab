"""Simulation and estimation infrastructure for the embedded real-time control
experiment.

``embedded_control.ipynb`` imports this module and does the presentation. Pure
compute here, with no ``matplotlib``: every function returns numbers, arrays or
dicts the notebook renders.

The study tests both dtfit streaming filters and the fused multi-axis
``FilterBank`` across the concerns that actually decide an embedded estimator,
against the online toolkit an engineer would otherwise reach for: an EKF, RLS,
a constant-acceleration Kalman and a sliding-window ``curve_fit``. What it
holds:

* the plant model families, :data:`PLANTS` and :func:`gen_plant`: the
  oscillatory, sustained-cycle, monotone and polynomial signal classes, with
  noise, outliers and dropout;
* the uniform estimator adapters :class:`EAAd`, :class:`LegAd`, :class:`EKFAd`,
  :class:`RLSAd` and :class:`RefitAd`, driven by :func:`drive`;
* the identification-accuracy sweep across plant shapes, and the data-driven
  applicability map :data:`FILTER_REASON`;
* the robustness profile, :func:`sweep_perr` and :func:`dropout_perr` over
  Gaussian noise, outliers and dropout;
* the multi-axis fault detection, :func:`make_multi`, :class:`MergedTracker`,
  :func:`run_tracker` and :func:`kalman_multi`, on a fused chi-square detector
  and ``inflate``;
* the embedded-footprint accounting, :func:`footprint_rows`, :data:`MCUS` and a
  re-exported :func:`embedded_footprint`;
* the real-data online tracking, :func:`load_fx` and :func:`fx_track`.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from dtfit.streaming import EACFilter, LSIFilter

from dtfit_experimental.streaming import FilterBank

from dtfit_experimental.experiments.common import EXPERIMENTS_DIR, metrics
from dtfit_experimental.experiments.common import baselines as bl
from dtfit_experimental.experiments.common.baselines import (
    KalmanCA, EKFParam, RLSPredictor,
)
from dtfit_experimental.experiments.domains.common import embedded_footprint

__all__ = [
    "OSC", "PLANTS", "MCUS", "FILTER_REASON",
    "gen_plant",
    "EAAd", "LegAd", "EKFAd", "RLSAd", "RefitAd", "perr", "drive", "adapters",
    "clean_accuracy",
    "sweep_perr", "dropout_perr", "sweep_perr_all", "exp_model_mismatch",
    "make_multi", "MergedTracker", "run_tracker", "kalman_multi",
    "footprint_rows", "embedded_footprint",
    "load_fx", "fx_track",
    "EACFilter", "LSIFilter", "FilterBank", "KalmanCA", "EKFParam", "RLSPredictor",
    "metrics",
]

OSC = "A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)"


# the plant model families, each a real embedded signal class
def _f_damped(t, A, w, z):
    return A * np.exp(-z * w * t) * np.sin(w * np.sqrt(1 - z ** 2) * t)


def _f_acsine(t, A, c, w):
    return c + A * np.sin(w * t)


def _f_firstorder(t, K, tau):
    return K * (1 - np.exp(-t / tau))


def _f_catraj(t, c0, c1, c2):
    return c0 + c1 * t + c2 * t ** 2


PLANTS = [
    dict(key="damped_osc", app="control / vibration ID", shape="oscillatory",
         expr=OSC, func=_f_damped, true={"A": 2.0, "w": 2.5, "z": 0.12},
         p0=[2.0, 2.0, 0.1], bounds=([0.1, 1, 0.01], [5, 6, 0.9]),
         T=12, n=700, window=60, q=[1e-3, 1e-3, 1e-3]),
    dict(key="ac_sine", app="AC / power monitoring", shape="sustained cycle",
         expr="c + A*sin(w*t)", func=_f_acsine, true={"A": 2.0, "c": 0.5, "w": 2.5},
         p0=[1.5, 0.5, 2.0], bounds=([0.1, -2, 1], [5, 3, 6]),
         T=12, n=700, window=60, q=[1e-3, 1e-3, 1e-3]),
    dict(key="first_order", app="RC / thermal / DC-motor", shape="monotone",
         expr="K*(1-exp(-t/tau))", func=_f_firstorder, true={"K": 3.0, "tau": 1.2},
         p0=[1.0, 1.0], bounds=([0.1, 0.05], [10, 5]),
         T=6, n=600, window=50, q=[1e-2, 1e-2]),
    dict(key="ca_traj", app="GPS / inertial trajectory", shape="polynomial",
         expr="c0 + c1*t + c2*t**2", func=_f_catraj,
         true={"c0": 1.0, "c1": 2.0, "c2": 0.5}, p0=[0.0, 0.0, 0.0],
         bounds=([-10, -10, -10], [10, 10, 10]),
         T=6, n=600, window=40, q=[1e-2, 1e-2, 1e-2]),
]


def gen_plant(plant, rng, *, noise=0.05, outliers=0.0, drop=0.0):
    """One noisy real-time stream from a plant: the clean signal plus Gaussian
    noise, optionally with gross outliers (sensor spikes, multipath) and
    optionally with dropout, a fraction of samples removed to leave an
    irregular stream. Returns ``(t, y, clean)``."""
    t = np.linspace(0, plant["T"], plant["n"])
    clean = plant["func"](t, *[plant["true"][k] for k in sorted(plant["true"])])
    scale = clean.std() + 1e-9
    y = clean + rng.normal(0, noise * scale, t.size)
    if outliers > 0:
        m = rng.random(t.size) < outliers
        y[m] += rng.normal(0, 8 * scale, int(m.sum()))
    if drop > 0:
        keep = np.sort(rng.choice(t.size, int(t.size * (1 - drop)), replace=False))
        t, y, clean = t[keep], y[keep], clean[keep]
    return t, y, clean


# the uniform estimator adapters
class _Ad:
    gives_params = True

    def params(self):
        return None


class EAAd(_Ad):
    name = "dtfit EACFilter"

    def __init__(self, plant, window=None):
        self.f = EACFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=len(plant["p0"]), q_diag=list(plant["q"]))

    def step(self, t, y):
        self.f.partial_fit(t, y)

    def predict(self, t):
        return float(self.f.predict(np.array([t]))[0]) if len(self.f._t) else np.nan

    def params(self):
        return dict(self.f.params_)


class LegAd(_Ad):
    name = "dtfit LSIFilter"

    def __init__(self, plant, window=None):
        self.f = LSIFilter(plant["expr"], "t", p0=list(plant["p0"]),
                           window_size=window or plant["window"],
                           order=5, q_diag=list(plant["q"]))

    def step(self, t, y):
        self.f.partial_fit(t, y)

    def predict(self, t):
        return float(self.f.predict(np.array([t]))[0]) if len(self.f._t) else np.nan

    def params(self):
        return dict(self.f.params_)


class EKFAd(_Ad):
    name = "EKF (params-as-state)"

    def __init__(self, plant):
        self.f = EKFParam(plant["expr"], "t", list(plant["p0"]), q=1e-4, r=0.5,
                          p_init=5.0)

    def step(self, t, y):
        self.f.update(t, y)

    def predict(self, t):
        return float(self.f.predict(t))

    def params(self):
        return dict(self.f.params_)


class RLSAd(_Ad):
    name = "RLS (AR predictor)"
    gives_params = False

    def __init__(self, plant, order=4):
        self.f = RLSPredictor(order=order, lam=1.0, delta=1e3)
        self._last = float("nan")

    def step(self, t, y):
        self.f.update(y)
        self._last = self.f.last_pred_

    def predict(self, t):
        return self._last

    def params(self):
        return None


class RefitAd(_Ad):
    name = "sliding-window curve_fit"

    def __init__(self, plant, refit_every=20):
        self.plant = plant
        self.W = plant["window"]
        self.every = refit_every
        self.t, self.y = [], []
        self.p = np.array(plant["p0"], float)
        self._k = 0

    def step(self, t, y):
        self.t.append(t); self.y.append(y)
        if len(self.t) > self.W:
            self.t.pop(0); self.y.pop(0)
        self._k += 1
        if len(self.t) >= self.W and self._k % self.every == 0:
            try:
                self.p = bl.scipy_curve_fit(np.array(self.t), np.array(self.y),
                                            self.plant["func"], self.p,
                                            bounds=self.plant["bounds"])
            except Exception:
                pass

    def predict(self, t):
        return float(self.plant["func"](t, *self.p))

    def params(self):
        return dict(zip(sorted(self.plant["true"]), self.p))


def perr(params, true):
    """Mean relative parameter error (%), or ``None`` for a params-free method."""
    if params is None:
        return None
    return float(np.mean([abs(params[k] - true[k]) / abs(true[k]) for k in true]) * 100)


def drive(adapter, t, y, clean, warm):
    """Run one adapter sample by sample over a stream. Returns
    ``(rmse_vs_clean, median_latency_us, track)``, the RMSE scored past
    warm-up."""
    track = np.full(t.size, np.nan)
    lat = []
    for i in range(t.size):
        t0 = time.perf_counter()
        adapter.step(float(t[i]), float(y[i]))
        lat.append((time.perf_counter() - t0) * 1e6)
        track[i] = adapter.predict(float(t[i]))
    valid = np.isfinite(track)
    valid[:warm] = False
    rmse = (float(np.sqrt(np.mean((track[valid] - clean[valid]) ** 2)))
            if valid.any() else np.nan)
    return rmse, float(np.median(lat)), track


def adapters(plant):
    return [EAAd(plant), LegAd(plant), EKFAd(plant), RLSAd(plant), RefitAd(plant)]


def clean_accuracy(plant, seeds, *, noise=0.05):
    """Multi-seed clean accuracy (E1) for one plant: run every adapter over
    ``seeds`` independent noisy realisations and report the distribution of the
    recovered-parameter error as mean and standard deviation, so the headline
    number cannot be single-seed luck. RMSE and latency are averaged too, both
    being stable across seeds. From seed 0 it also returns the noisy stream and
    the best-dtfit and EKF tracks the overlay figure draws, as
    ``{"rows": {name: {...}}, "overlay": {...}}``.

    ``rows[name]`` carries ``perr_mean`` and ``perr_std`` (None for a
    params-free method), ``rmse``, ``lat``, and ``gp`` for whether it gives
    physical parameters."""
    warm = plant["window"] + 15
    names = [ad.name for ad in adapters(plant)]
    acc = {nm: dict(perr=[], rmse=[], lat=[], gp=True) for nm in names}
    overlay = None
    for s in range(seeds):
        rng = np.random.default_rng(s)
        t, y, clean = gen_plant(plant, rng, noise=noise)
        tracks = {}
        for ad in adapters(plant):
            rmse, lat, track = drive(ad, t, y, clean, warm)
            acc[ad.name]["rmse"].append(rmse)
            acc[ad.name]["lat"].append(lat)
            acc[ad.name]["gp"] = ad.gives_params
            acc[ad.name]["perr"].append(perr(ad.params(), plant["true"]))
            tracks[ad.name] = track
        if s == 0:
            dt_names = ["dtfit EACFilter", "dtfit LSIFilter"]
            bestf = min(dt_names,
                        key=lambda nm: (np.inf if acc[nm]["perr"][0] is None
                                        else acc[nm]["perr"][0]))
            overlay = dict(t=t, y=y, clean=clean, bestf=bestf,
                           dttrack=tracks[bestf],
                           ekftrack=tracks["EKF (params-as-state)"])
    rows = {}
    for nm in names:
        pe = [v for v in acc[nm]["perr"] if v is not None]
        rows[nm] = dict(
            perr_mean=(float(np.mean(pe)) if pe else None),
            perr_std=(float(np.std(pe)) if pe else None),
            rmse=float(np.nanmean(acc[nm]["rmse"])),
            lat=float(np.nanmean(acc[nm]["lat"])),
            gp=acc[nm]["gp"])
    return dict(rows=rows, overlay=overlay)


# the robustness sweeps: Gaussian noise, outliers, dropout
def sweep_perr(plant, kind, levels, seeds):
    """Sweep ``kind``, "noise" or "outliers", over ``levels``, averaging the
    mean parameter error over ``seeds`` for LSI, EAC and the EKF. Returns
    ``{method: [errs]}``."""
    methods = ["dtfit LSIFilter", "dtfit EACFilter", "EKF (params-as-state)"]
    out = {m: [] for m in methods}
    for lv in levels:
        acc = {m: [] for m in methods}
        for s in range(seeds):
            rng = np.random.default_rng(s)
            if kind == "noise":
                t, y, _ = gen_plant(plant, rng, noise=lv)
            else:
                t, y, _ = gen_plant(plant, rng, noise=0.05, **{kind: lv})
            for ad in (LegAd(plant), EAAd(plant), EKFAd(plant)):
                for i in range(t.size):
                    ad.step(float(t[i]), float(y[i]))
                acc[ad.name].append(perr(ad.params(), plant["true"]))
        for m in methods:
            out[m].append(float(np.nanmean(acc[m])))
    return out


def sweep_perr_all(kind, levels, seeds, *, plants=None):
    """Run :func:`sweep_perr` for every plant, by default all of
    :data:`PLANTS`, so the outlier and noise robustness claim is validated
    across shapes rather than on the damped oscillator alone. Returns
    ``{plant_key: {method: [errs per level]}}``."""
    plants = plants if plants is not None else PLANTS
    return {p["key"]: sweep_perr(p, kind, levels, seeds) for p in plants}


def dropout_perr(plant, drops, seeds):
    """Mean parameter error against dropout fraction for LSI, EAC and the EKF.
    Returns a list of ``(method_label, [errs per drop])`` rows."""
    rows = []
    for m_cls, mname in [(LegAd, "dtfit LSIFilter"), (EAAd, "dtfit EACFilter"),
                         (EKFAd, "EKF")]:
        cells = []
        for d in drops:
            errs = []
            for s in range(seeds):
                rng = np.random.default_rng(10 + s)
                t, y, _ = gen_plant(plant, rng, noise=0.05, drop=d)
                ad = m_cls(plant)
                for i in range(t.size):
                    ad.step(float(t[i]), float(y[i]))
                errs.append(perr(ad.params(), plant["true"]))
            cells.append(float(np.nanmean(errs)))
        rows.append((mname, cells))
    return rows


# Model mismatch, the negative control: the wrong physical model on-device.
# Each pair is the true plant that generates the stream and the wrong plant
# whose model is fitted to it.
_MISMATCH_PAIRS = [
    ("damped_osc", "first_order"),  # a decaying sinusoid fitted as a saturating rise
    ("first_order", "ac_sine"),     # a monotone RC rise fitted as a pure sinusoid
    ("ca_traj", "first_order"),     # an accelerating trajectory fitted as a plateau
]


def _plant_by_key(key):
    return next(p for p in PLANTS if p["key"] == key)


# RMSE ceiling. A wrong-model EKF can diverge to numerical overflow through
# covariance windup or an exponential blow-up, so clip to a finite ceiling and
# flag it rather than let a ~1e88 value swamp the table: "diverged" is the
# honest read.
_MM_CEIL = 1e4


def _mismatch_scores(adapter, t, y, clean, warm):
    """Drive one adapter over a stream and return
    ``(rmse_vs_clean, insample_residual_rmse, diverged)``. The residual, track
    against the noisy observations, is the on-device self-diagnosis signal: a
    wrong model cannot fit even the data it sees, so its residual stays
    structured and large. ``diverged`` is set when the estimate blew up, going
    non-finite or past :data:`_MM_CEIL`."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _, _, track = drive(adapter, t, y, clean, warm)
    valid = np.isfinite(track)
    valid[:warm] = False
    if not valid.any():
        return float("nan"), float("nan"), True
    rmse_clean = float(np.sqrt(np.mean((track[valid] - clean[valid]) ** 2)))
    resid = float(np.sqrt(np.mean((track[valid] - y[valid]) ** 2)))
    diverged = not np.isfinite(rmse_clean) or rmse_clean > _MM_CEIL
    return min(rmse_clean, _MM_CEIL), min(resid, _MM_CEIL), diverged


def exp_model_mismatch(seeds=5):
    """Negative control: drive an online estimator configured for the wrong
    plant model over another plant's stream. Two things come out of it. No
    online estimator rescues a mis-specified physical model, the error being a
    property of the model rather than the filter, though the integral
    ``LSIFilter`` degrades gracefully and stays bounded where the pointwise EKF
    can diverge outright. And the wrong model leaves a large in-sample
    residual, the streaming analogue of an in-sample R^2 collapse and the very
    innovation the fused chi-square detector already watches, so a mismatch is
    self-diagnosing on-device with no ground truth. Returns one row per (true
    plant, wrong model, estimator)."""
    rows = []
    for true_key, wrong_key in _MISMATCH_PAIRS:
        tp, wp = _plant_by_key(true_key), _plant_by_key(wrong_key)
        warm = max(tp["window"], wp["window"]) + 15
        for cls, est in [(LegAd, "dtfit LSIFilter"), (EKFAd, "EKF")]:
            cc, cw, rc, rw, dv = [], [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                t, y, clean = gen_plant(tp, rng, noise=0.05)
                c_clean, c_resid, _ = _mismatch_scores(cls(tp), t, y, clean, warm)
                w_clean, w_resid, w_div = _mismatch_scores(cls(wp), t, y, clean,
                                                           warm)
                cc.append(c_clean); cw.append(w_clean)
                rc.append(c_resid); rw.append(w_resid); dv.append(w_div)
            correct = float(np.nanmean(cc))
            wrong = float(np.nanmean(cw))
            rows.append(dict(
                true=true_key, wrong_model=wrong_key, estimator=est,
                rmse_correct=correct, rmse_wrong=wrong,
                blowup=(wrong / correct if correct and correct > 0
                        else float("nan")),
                resid_correct=float(np.nanmean(rc)),
                resid_wrong=float(np.nanmean(rw)),
                diverged=bool(np.any(dv)))
            )
    return rows


# fault detection and on-device re-adaptation, by the multi-axis fused detector
def make_multi(rng, n=900, noise=0.05):
    """A 3-axis damped oscillator carrying a damping fault: zeta jumps on every
    axis at the midpoint. Returns ``(t, noisy, clean, half)``."""
    t = np.linspace(0, 18, n)
    half = n // 2
    A = np.array([2.0, 1.5, 2.5]); w = np.array([2.5, 2.0, 3.0])
    z1 = np.array([0.08, 0.10, 0.06]); z2 = np.array([0.30, 0.28, 0.25])
    clean = np.zeros((n, 3))
    dtt = np.diff(t, prepend=t[0])
    for d in range(3):
        z_arr = np.where(np.arange(n) < half, z1[d], z2[d])
        wd = w[d] * np.sqrt(1 - z_arr ** 2)
        clean[:, d] = A[d] * np.exp(-z_arr * w[d] * t) * np.sin(np.cumsum(wd * dtt))
    return t, clean + rng.normal(0, noise, clean.shape), clean, half


class MergedTracker:
    """A multi-axis oscillator tracker on dtfit's streaming API: a
    :class:`~dtfit_experimental.streaming.FilterBank` of per-axis
    Legendre-spectrum filters driven by
    :class:`~dtfit_experimental.streaming.FusedChiSquareDetector`, which
    pools the per-axis one-step innovations into a fused chi^2(n_axes)
    fault statistic and re-arms the bank through ``inflate`` on a
    detection."""

    def __init__(self, n_axes, p0, *, window=60, fuse_alpha=1e-4, inflate=4.0):
        self.bank = FilterBank.from_model(
            OSC, "t", n_axes, filter_cls=LSIFilter, p0=list(p0),
            window_size=window, order=5, q_diag=[1e-3] * len(p0),
            cusum_h=np.inf)
        self.n_axes = n_axes
        self.detector = self.bank.fused_detector(alpha=fuse_alpha, inflate=inflate)

    @property
    def flags_(self):
        return self.detector.flags_

    def step(self, i, t, y):
        self.detector.update(t, y)

    def predict(self, t):
        return self.bank.predict(np.array([t]))


def run_tracker(t, Y, clean, half, inflate):
    """Run the fused FilterBank over a multi-axis stream. Returns
    ``(tracker, pred, rmse, false_alarms, detect_latency_steps, warmup)``."""
    n = Y.shape[0]
    tr = MergedTracker(3, [2.0, 2.5, 0.1], inflate=inflate)
    pred = np.full((n, 3), np.nan)
    for i in range(n):
        tr.step(i, float(t[i]), Y[i])
        pred[i] = tr.predict(float(t[i]))
    warm = tr.bank.filters[0].W
    valid = np.all(np.isfinite(pred), axis=1)
    valid[:warm] = False
    rmse = (float(np.sqrt(np.mean((pred[valid] - clean[valid]) ** 2)))
            if valid.any() else np.nan)
    flags_post = [i for i in tr.flags_ if i >= half]
    fp = len([i for i in tr.flags_ if i < half])
    lat = (flags_post[0] - half) if flags_post else None
    return tr, pred, rmse, fp, lat, warm


def kalman_multi(t, Y, clean, warm):
    """Constant-acceleration Kalman over the multi-axis stream, identifying no
    plant. Returns ``(pred, rmse)``."""
    kf = KalmanCA(dim=3, dt=float(np.mean(np.diff(t))), q=1e-2, r=0.5)
    pred = np.full((Y.shape[0], 3), np.nan)
    for i in range(Y.shape[0]):
        kf.update(Y[i]); pred[i] = kf.position()
    valid = np.all(np.isfinite(pred), axis=1)
    valid[:warm] = False
    rmse = float(np.sqrt(np.mean((pred[valid] - clean[valid]) ** 2)))
    return pred, rmse


# deployable footprint and latency
MCUS = [
    ("AVR ATmega328 (Uno)", 2 * 1024, "no (soft)"),
    ("ARM Cortex-M0+ (SAMD21)", 32 * 1024, "no (soft)"),
    ("ARM Cortex-M4F (STM32F4)", 192 * 1024, "yes"),
    ("ESP32 (LX6 FPU)", 520 * 1024, "yes"),
]


def footprint_rows(lat, *, n=3, W=60):
    """Live no-malloc state per estimator, which does not grow with the stream:
    the deployable word and byte counts of a hand-coded C struct. ``lat`` is
    the ``{estimator_name: latency_us}`` map from the accuracy sweep. Returns a
    dict holding the per-estimator state table, the MCU-fit table, and the
    resident-state sweep the figure draws."""
    ea = embedded_footprint(n, W, kind="eac")
    leg = embedded_footprint(n, W, kind="legendre")
    ekf_words = n * n + 2 * n + 8
    rls_words = 4 * 4 + 4 + 4
    kf_words = 3 * (3 * 3 + 3) + 8
    state = [
        dict(estimator="dtfit EACFilter", state_words=ea["sram_words"],
             float32_B=str(ea["sram_bytes_f32"]), window_buffer=f"yes (W={W})",
             params="yes", latency_us=lat["dtfit EACFilter"]),
        dict(estimator="dtfit LSIFilter", state_words=leg["sram_words"],
             float32_B=f"{leg['sram_bytes_f32']} +{leg['flash_bytes_f32']}B flash",
             window_buffer=f"yes (W={W})", params="yes",
             latency_us=lat["dtfit LSIFilter"]),
        dict(estimator="EKF (params-as-state)", state_words=ekf_words,
             float32_B=str(ekf_words * 4), window_buffer="no", params="yes",
             latency_us=lat["EKF (params-as-state)"]),
        dict(estimator="RLS (AR predictor)", state_words=rls_words,
             float32_B=str(rls_words * 4), window_buffer="no", params="no",
             latency_us=lat["RLS (AR predictor)"]),
        dict(estimator="Kalman-CA (3-axis)", state_words=kf_words,
             float32_B=str(kf_words * 4), window_buffer="no", params="no",
             latency_us=None),
    ]
    track32 = ea["sram_bytes_f32"] * 3        # a 3-axis tracker
    mcu = [dict(MCU=name, SRAM_KB=sram // 1024, FPU=fpu,
                fits=("yes" if track32 < sram * 0.5
                      else ("tight" if track32 < sram else "no")))
           for name, sram, fpu in MCUS]
    Ws = np.arange(10, 110, 5)
    sweep = {nn: [embedded_footprint(nn, int(w))["sram_bytes_f32"] for w in Ws]
             for nn in (2, 3, 5)}
    return dict(state=state, track32=track32, mcu=mcu,
                sweep_W=Ws, sweep=sweep, lat=lat)


# real-data online tracking
def load_fx(limit=220):
    """Load the daily USD/UAH 2014-15 crisis rate, normalised as
    ``rate/rate[0]``. Returns ``(t, y)``."""
    path = EXPERIMENTS_DIR / "data" / "usd_uah_2014_2015.csv"
    rows = list(csv.reader(path.open()))[1:]
    rate = np.array([float(r[1]) for r in rows])[:limit]
    y = rate / rate[0]
    t = np.linspace(0, 1.5, y.size)
    return t, y


def fx_track(t, y):
    """Stream the FX series and track a local exponential ``a*exp(b*t)`` online,
    one step ahead, against the EKF, RLS and a random walk. Returns
    ``(actual, {method: predictions})``."""
    ea = EACFilter("a*exp(b*t)", "t", p0=[1.0, 0.5], window_size=40, order=2,
                   q_diag=[1e-4, 1e-4])
    ekf = EKFParam("a*exp(b*t)", "t", [1.0, 0.5], q=1e-5, r=0.1)
    rls = RLSPredictor(order=2, lam=1.0, delta=1e3)
    preds = {"dtfit EACFilter": [], "EKF": [], "RLS": [], "random walk": []}
    actual = []
    for i in range(1, y.size):
        preds["dtfit EACFilter"].append(
            float(ea.predict(np.array([t[i]]))[0]) if len(ea._t) >= ea.W else y[i - 1])
        preds["EKF"].append(float(ekf.predict(t[i])))
        preds["RLS"].append(rls.predict_next())
        preds["random walk"].append(y[i - 1])
        actual.append(y[i])
        ea.partial_fit(t[i], y[i]); ekf.update(t[i], y[i]); rls.update(y[i])
    return np.array(actual), {k: np.array(v) for k, v in preds.items()}


# the data-driven applicability map, which carries the headline reasoning
FILTER_REASON = {
    "damped_osc": ("EACFilter ~= Legendre",
                   "A clean damped oscillation is easy for both -- param error <1% "
                   "each (EAF marginally better on params, Legendre better on "
                   "tracking RMSE). Use the lean EAF unless you need the robustness."),
    "ac_sine": ("EACFilter ~= Legendre",
                "A sustained sinusoid -- both recover it within ~1%; the EAF is "
                "leaner and slightly more accurate on the parameters here. The "
                "filters separate under stress, not on this clean cycle."),
    "first_order": ("LSIFilter",
                    "A saturating exponential is where the spectrum clearly helps: "
                    "its several orthogonal coefficients pin (K, tau) far better than "
                    "a single area, which leaves tau weakly constrained (3.8% vs ~19% "
                    "param error)."),
    "ca_traj": ("LSIFilter",
                "A polynomial trajectory -- the multi-coefficient measurement edges "
                "the single area (16% vs 19%); both find the trajectory parameters "
                "harder than the EKF, though they track the path itself well."),
}
