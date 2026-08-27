"""Validation on well-known trajectories: the literature-standard companion to
the synthetic-random ``realtime_gps`` study and the real-rig
``realtime_gps_hw`` study.

Two kinds of well-known trajectory:

1. Standard maneuvering-target benchmarks with exact analytic ground truth: a
   fixed coordinated-turn maneuver (the canonical CV-CT-CV-CT-CV scenario) and
   a figure-8 lemniscate. Since the truth is known at every step, the score is
   the true position RMSE rather than the held-out-fix proxy the no-RTK rig is
   limited to, Monte-Carlo averaged. Both are generated at the sim's native
   scale (10 Hz, GPS sigma about 1.5 m, about 12 m/s), so every tracker runs in
   the regime it is tuned for and the comparison is fair by construction.

2. Public real datasets with dm-level RTK/INS truth, wired through
   :func:`load_external` so a real log drops straight in; the EXTERNAL DATASETS
   comment further down names them.

Methods under test, the same family as the sim: dtfit LSI-cubic, dtfit LSI
coordinated-turn (``c0+c1*t+c2*sin(c3*t+c4)``, the nonlinear model a single
Kalman cannot represent), and dtfit EAC, the honest negative, area being the
wrong measurement for oscillatory motion.

The baselines a maneuvering-target practitioner actually deploys, all
position-only and so on the fair information set: Kalman-CA on a single
constant-acceleration model, CT-EKF (pos-only), a coordinated-turn EKF that
estimates the turn-rate from position, and IMM (CV+CT), the gold-standard
interacting-multiple-model tracker. The last two are absent from
``common/baselines.py``, which carries only the gyro-aided ``CTEKFGyro``, so
they live here.

Fairness rules, the lesson from the rig harness: dtfit's turn model is not
handed the generating turn-rate (it estimates ``c3`` online), every tracker
sees only the noisy position, RMSE is against the true trajectory, and numbers
are averaged over Monte-Carlo seeds.

The finding, adversarially audited for truth leaks and for handicapping dtfit:
on these idealized-Gaussian-noise benchmarks the model-matched recursive
trackers win. IMM, CT-EKF and Kalman-CA beat dtfit on clean smoothing (on the
CT, IMM 1.02 against dtfit 1.33) and especially on dropout coasting (IMM 4.3
against dtfit 11.4). That dropout gap is structural: a windowed integral fit
extrapolates a local polynomial across the blank and diverges, where the
recursive filters dead-reckon an explicit velocity and turn-rate state. dtfit
runs its own near-optimal config here, a window/order/q sweep confirming
window-15 poly-cubic is dtfit's optimum, so the loss is real rather than a
handicap. dtfit's robust winsorized LSI is a genuine, cheap lever, recovering
most of the glitch error at roughly zero clean cost, and it is competitive with
a symmetrically Huber-hardened Kalman without beating one: a tie on the
coordinated turn (2.01 against 2.02) and a slight loss on the smooth figure-8
(1.86 against 1.58). So this is the recursive filters' home turf, where they
are optimal under Gaussian noise with known dynamics. dtfit's measured edge
lives on real non-Gaussian, drifting GPS: ``realtime_gps_hw`` records dtfit LSI
beating the Kalman by about 2x on the actual rig.
"""
from __future__ import annotations

import numpy as np

from dtfit_experimental.experiments.domains.realtime_gps import backend as G

SIGMA = G.GPS_SIGMA      # per-axis noise (m): the sim's NEO-M8N-grade 1.5 m
WARM = G.WARMUP          # warm-up samples excluded from scoring
N = 600                  # 10 Hz over the 60 s scenario (the sim's native rate)

# A fixed, named coordinated-turn maneuver: straight, left turn, straight, hard
# right, straight, at a constant 12 m/s and planar (climb 0). The
# piecewise-constant turn-rate plan is in the sim's
# ``(onset_s, turn_rate_rad_s, speed_m_s, climb_m_s)`` format, so it runs
# through the identical, already-tuned ``build_rig`` / ``dtfit_track``
# machinery at the native scale.
CT_PLAN = [
    (0.0,  0.00, 12.0, 0.0),
    (12.0, 0.40, 12.0, 0.0),
    (24.0, 0.00, 12.0, 0.0),
    (36.0, -0.70, 12.0, 0.0),
    (48.0, 0.00, 12.0, 0.0),
]


# (1) standard trajectories with exact ground truth
def ct_benchmark(seed=0, *, sigma=SIGMA, plan=None, n=N):
    """The canonical 2-D coordinated-turn maneuver through the sim generator:
    exact truth plus GPS-noise fixes at the native scale. Returns
    ``(t, truth(n,3), meas(n,3), labels(n,))``, the labels marking each sample
    ``"cv"`` on a straight leg or ``"turn"`` inside a turn."""
    plan = plan or CT_PLAN
    t, truth, fixes, _, _ = G.build_rig(n, seed=seed, plan=plan, gps_sigma=sigma)
    om, _, _ = G._controls(t, plan)
    return t, truth, fixes, np.where(np.abs(om) > 1e-9, "turn", "cv")


def figure8_benchmark(seed=0, *, sigma=SIGMA, n=N, scale=150.0):
    """The figure-8 (Gerono lemniscate) at the native scale:
    ``x = scale*cos(phi)``, ``y = scale*sin(2*phi)/2`` over one period, a
    smooth and continuously curving path peaking near 16 m/s. Exact ground
    truth."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, G.DURATION, n)
    ph = 2.0 * np.pi * t / G.DURATION
    truth = np.stack([2.0 + scale * np.cos(ph), scale * np.sin(2.0 * ph) / 2.0,
                      np.full(n, 5.0)], axis=1)
    return t, truth, truth + rng.normal(0.0, sigma, truth.shape), np.array(["curve"] * n)


# (2) the fair maneuver baselines the repo lacks: CT-EKF and IMM
class _CT5:
    """One coordinated-turn EKF over the state ``[x, vx, y, vy, w]`` on a
    position-only measurement, so the turn-rate ``w`` is estimated rather than
    measured. Reused both as a stand-alone tracker and as an IMM mode. Call
    ``predict`` then ``update(z)`` with ``z = [x, y]``; ``update`` returns the
    Gaussian measurement likelihood the IMM weights its modes by. The defaults
    suit the native 10 Hz and GPS-sigma scale: ``r`` is the measurement
    variance, ``q_acc`` covers the ~8 m/s^2 centripetal load, and ``q_w`` lets
    the turn-rate slew across the onsets."""

    def __init__(self, dt, r, q_acc=8.0, q_w=0.30):
        self.dt = float(dt); self.r = float(r)
        qb = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * float(q_acc)
        Q = np.zeros((5, 5)); Q[0:2, 0:2] = qb; Q[2:4, 2:4] = qb; Q[4, 4] = float(q_w) * dt
        self.Q = Q
        self.x = np.zeros(5); self.P = np.eye(5)

    def _prop(self, s):
        x, vx, y, vy, w = s; dt = self.dt; th = w * dt
        if abs(w) < 1e-6:                                  # constant-velocity limit
            return np.array([x + vx * dt, vx, y + vy * dt, vy, w])
        sn, cs = np.sin(th), np.cos(th); a, b = sn / w, (1.0 - cs) / w
        return np.array([x + a * vx - b * vy, cs * vx - sn * vy,
                         y + b * vx + a * vy, sn * vx + cs * vy, w])

    def _F(self, s):
        F = np.zeros((5, 5)); f0 = self._prop(s)
        for j in range(5):
            sp = s.copy(); h = 1e-6 * max(1.0, abs(s[j])); sp[j] += h
            F[:, j] = (self._prop(sp) - f0) / h
        return F

    def predict(self):
        F = self._F(self.x)
        self.x = self._prop(self.x)
        self.P = F @ self.P @ F.T + self.Q

    _H = np.array([[1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0]])

    def update(self, z):
        H = self._H; R = np.eye(2) * self.r
        yk = np.asarray(z, float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ yk
        self.P = (np.eye(5) - K @ H) @ self.P
        det = float(np.linalg.det(2.0 * np.pi * S))
        return max(float(np.exp(-0.5 * yk @ np.linalg.solve(S, yk)) / np.sqrt(max(det, 1e-12))),
                   1e-300)

    def init_state(self, z):
        self.x = np.array([z[0], 0.0, z[1], 0.0, 0.0])
        self.P = np.diag([self.r, 100.0, self.r, 100.0, 0.5])


def ctekf_pos_track(t, meas, *, sigma=SIGMA, q_acc=8.0, q_w=0.30):
    """A single coordinated-turn EKF with the turn-rate estimated from position
    and no gyro. Missing fixes (NaN rows) coast predict-only on the CT
    dynamics, which is the fair dropout dead-reckon."""
    n = len(t); f = _CT5(t[1] - t[0], sigma ** 2, q_acc, q_w)
    out = np.zeros((n, 3))
    for i in range(n):
        z = meas[i, :2]
        if i == 0:
            f.init_state(z if np.isfinite(z).all() else np.zeros(2))
        elif not np.isfinite(z).all():
            f.predict()                        # coast through the gap
        else:
            f.predict(); f.update(z)
        out[i, :2] = [f.x[0], f.x[2]]
    return out


class IMM2:
    """Interacting Multiple Model over two coordinated-turn EKFs sharing the
    state ``[x, vx, y, vy, w]``: a near-CV mode whose tiny ``q_w`` holds w near
    zero, and a maneuver mode whose larger ``q_w`` lets w adapt. The standard
    IMM cycle applies: mixing, model-matched predict and update,
    mode-probability update, combination. ``Pij`` is the mode-transition
    matrix."""

    def __init__(self, dt, sigma=SIGMA, q_acc=8.0, q_w_cv=1e-3, q_w_ct=0.5,
                 Pij=((0.95, 0.05), (0.10, 0.90))):
        r = sigma ** 2
        self.models = [_CT5(dt, r, q_acc, q_w_cv), _CT5(dt, r, q_acc, q_w_ct)]
        self.mu = np.array([0.5, 0.5])
        self.Pij = np.asarray(Pij, float)

    def init_state(self, z):
        for m in self.models:
            m.init_state(z)

    def step(self, z):
        cbar = self.Pij.T @ self.mu                          # predicted mode probs (j)
        Wij = (self.Pij * self.mu[:, None]) / cbar[None, :]   # mixing weights W[i,j]
        mixed = []
        for j in range(2):                                    # mixed initial conditions
            xj = sum(Wij[i, j] * self.models[i].x for i in range(2))
            Pj = np.zeros((5, 5))
            for i in range(2):
                dx = self.models[i].x - xj
                Pj = Pj + Wij[i, j] * (self.models[i].P + np.outer(dx, dx))
            mixed.append((xj, Pj))
        L = np.zeros(2)                                       # model-matched filtering
        for j in range(2):
            self.models[j].x, self.models[j].P = mixed[j]
            self.models[j].predict()
            L[j] = self.models[j].update(z)
        self.mu = cbar * L                                    # mode-probability update
        self.mu = self.mu / (self.mu.sum() + 1e-300)
        xc = sum(self.mu[j] * self.models[j].x for j in range(2))  # combination
        return np.array([xc[0], xc[2]])

    def coast(self):
        """A measurement-free step through a GPS gap: mix, predict each model,
        combine, with the mode probabilities taken from the Markov prediction
        since there is no likelihood to update them with."""
        cbar = self.Pij.T @ self.mu
        Wij = (self.Pij * self.mu[:, None]) / cbar[None, :]
        mixed = []
        for j in range(2):
            xj = sum(Wij[i, j] * self.models[i].x for i in range(2))
            Pj = np.zeros((5, 5))
            for i in range(2):
                dx = self.models[i].x - xj
                Pj = Pj + Wij[i, j] * (self.models[i].P + np.outer(dx, dx))
            mixed.append((xj, Pj))
        for j in range(2):
            self.models[j].x, self.models[j].P = mixed[j]
            self.models[j].predict()
        self.mu = cbar
        xc = sum(self.mu[j] * self.models[j].x for j in range(2))
        return np.array([xc[0], xc[2]])


def imm_track(t, meas, *, sigma=SIGMA, q_acc=8.0):
    n = len(t); imm = IMM2(t[1] - t[0], sigma=sigma, q_acc=q_acc)
    out = np.zeros((n, 3))
    for i in range(n):
        z = meas[i, :2]
        if i == 0:
            imm.init_state(z if np.isfinite(z).all() else np.zeros(2)); out[i, :2] = z
        elif not np.isfinite(z).all():
            out[i, :2] = imm.coast()           # dead-reckon through the gap
        else:
            out[i, :2] = imm.step(z)
    return out


class _CA1:
    """1-D constant-acceleration Kalman over (pos, vel, acc), taking the
    measurement variance ``r`` directly and offering optional Huber outlier
    down-weighting: when the standardized innovation ``|y|/sqrt(S)`` exceeds
    the threshold, the update's measurement variance is scaled up in
    proportion, softly discounting the spike. That is the fair pointwise
    analogue of dtfit's winsorized integral, which lets the glitch column
    compare hardened against hardened. A hard reject is avoided deliberately:
    on a CA model it lets the acceleration state run away during a real
    maneuver, a 275 m blow-up when measured, so the soft rescale is the
    competent choice."""

    def __init__(self, dt, r, q):
        self.r = float(r)
        self.F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1.0]])
        g = np.array([dt * dt / 2, dt, 1.0])
        self.Q = np.outer(g, g) * float(q)
        self.x = np.zeros(3); self.P = np.eye(3)

    def init(self, z):
        self.x = np.array([z, 0.0, 0.0]); self.P = np.diag([self.r, 100.0, 100.0])

    def predict(self):
        self.x = self.F @ self.x; self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, huber=None):
        S = self.P[0, 0] + self.r; y = z - self.x[0]
        if huber is not None:
            nu = abs(y) / np.sqrt(S)
            if nu > huber:
                S = self.P[0, 0] + self.r * (nu / huber)      # soft down-weight
        K = self.P[:, 0] / S
        self.x = self.x + K * y
        self.P = self.P - np.outer(K, self.P[0, :])


def kalman_ca_track(t, meas, *, sigma=SIGMA, q=5e-2, r=0.5, huber=None):
    """Per-axis constant-acceleration Kalman, optionally Huber-hardened,
    coasting predict-only through NaN gaps. ``r`` defaults to 0.5, the value
    the sim's tuned ``KalmanCA`` uses: a deliberately tight R that tracks the
    maneuver with little lag, where the principled sigma^2 trades about 0.5 m
    of clean accuracy for lag. The plain run therefore matches the deployed
    ``Kalman-CA`` baseline, and the Huber run is a fair, symmetric hardening of
    that same filter."""
    n = len(t); dt = t[1] - t[0]
    fl = [_CA1(dt, r, q) for _ in range(2)]
    out = np.zeros((n, 3))
    for i in range(n):
        z = meas[i, :2]
        for ax in range(2):
            if i == 0:
                fl[ax].init(z[ax] if np.isfinite(z[ax]) else 0.0)
            elif not np.isfinite(z[ax]):
                fl[ax].predict()
            else:
                fl[ax].predict(); fl[ax].update(z[ax], huber=huber)
            out[i, ax] = fl[ax].x[0]
    return out


# methods and scoring
def run_methods(t, meas, *, sigma=SIGMA):
    """Run every dtfit tracker and baseline on the noisy ``meas``, returning
    name -> smoothed. The Kalman appears twice, plain and Huber-hardened, so
    the glitch column can compare a hardened dtfit (``robust=True``) against a
    symmetrically hardened pointwise filter instead of a soft target."""
    return {
        "dtfit LSI-cubic": G.dtfit_track(t, meas, (1,), kind="lsi", model="poly")[0],
        "dtfit LSI-turn": G.dtfit_track(t, meas, (1,), kind="lsi", model="turn")[0],
        "dtfit EAC (area)": G.dtfit_track(t, meas, (1,), kind="eac", model="poly")[0],
        "Kalman-CA": G.kalman_track(t, meas, (1,))[0],
        "Kalman-CA (Huber)": kalman_ca_track(t, meas, sigma=sigma, huber=3.0),
        "CT-EKF (pos-only)": ctekf_pos_track(t, meas, sigma=sigma),
        "IMM (CV+CT)": imm_track(t, meas, sigma=sigma),
    }


def run_methods_realdata(t, meas, *, window=5, q_acc=20.0, kalman_q=5e-2, huber=3.0):
    """``run_methods`` configured for a real low-rate road trip. A 15-sample
    sim window is about 15 s there, hundreds of metres at road speed, so dtfit
    takes a short ``window`` (scale it to the trip's speed and rate) and the
    CA/CT baselines take a regime-appropriate process noise. Both dtfit and the
    Kalman carry a robust variant, keeping the multipath column hardened
    against hardened. NaN rows coast predict-only."""
    md = G.MODELS["poly"]; n = len(t)

    def dtfit(robust):
        fl = [G.LSIFilter(md["expr"], "t", p0=[float(meas[0, ax])] + list(md["rest"]),
                          window_size=window, order=md["order"], q_diag=[1e-2] * 4,
                          adapt_noise=True, robust=robust, drift_reset="inflate")
              for ax in range(3)]
        sm = np.zeros((n, 3))
        for i in range(n):
            miss = np.any(np.isnan(meas[i]))
            for ax in range(3):
                if not miss:
                    fl[ax].partial_fit(t[i], meas[i, ax])
                sm[i, ax] = float(fl[ax].predict(np.array([t[i]]))[0])
        return sm

    return {
        "dtfit LSI-cubic": dtfit(False),
        "dtfit LSI-cubic (robust)": dtfit(True),
        "Kalman-CA": G.kalman_track(t, meas, (1,), q=kalman_q)[0],
        "Kalman-CA (Huber)": kalman_ca_track(t, meas, q=kalman_q, huber=huber),
        "CT-EKF (pos-only)": ctekf_pos_track(t, meas, q_acc=q_acc),
        "IMM (CV+CT)": imm_track(t, meas, q_acc=q_acc),
    }


def pos_stats(est, truth, *, warm=WARM):
    """(RMSE, median, p95) of the 2-D position error past warm-up. The median
    and p95 expose spike rejection that a mean RMSE, dominated by a few huge
    multipath jumps, hides."""
    base = np.ones(len(truth), bool); base[:warm] = False
    e = np.linalg.norm(est[base, :2] - truth[base, :2], axis=1)
    return float(np.sqrt(np.mean(e ** 2))), float(np.median(e)), float(np.percentile(e, 95))


def _pos_rmse(est, truth, mask):
    d = est[mask, :2] - truth[mask, :2]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1)))) if mask.any() else float("nan")


def score(res, truth, labels, *, warm=WARM):
    """Per-method overall + per-segment-type position RMSE (warm-up excluded)."""
    n = len(truth); base = np.ones(n, bool); base[:warm] = False
    seg_types = list(dict.fromkeys(labels.tolist()))
    out = {}
    for name, est in res.items():
        row = {"overall": _pos_rmse(est, truth, base)}
        for s in seg_types:
            row[s] = _pos_rmse(est, truth, base & (labels == s))
        out[name] = row
    return out


def monte_carlo(bench_fn, seeds=range(50), *, sigma=SIGMA, **bench_kw):
    """Average per-method/per-segment RMSE over ``seeds`` independent noise realizations."""
    acc = None
    for sd in seeds:
        t, truth, meas, labels = bench_fn(seed=sd, sigma=sigma, **bench_kw)
        r = score(run_methods(t, meas, sigma=sigma), truth, labels)
        if acc is None:
            acc = {nm: {k: [] for k in row} for nm, row in r.items()}
        for nm, row in r.items():
            for k, v in row.items():
                acc[nm][k].append(v)
    return {nm: {k: float(np.mean(v)) for k, v in row.items()} for nm, row in acc.items()}


# Stress scenarios. Clean smoothing is not where the integral methods earn
# their keep; per the sim's E2 and E3 their edge is coasting through GPS
# dropouts and rejecting multipath glitches. A benchmark that skips both tells
# half the story.
def _gap_mask(n, gap=20, period=120):
    m = np.zeros(n, bool)
    for st in range(WARM + 20, n - gap, max(period, gap * 3)):
        m[st:st + gap] = True
    return m


def dropout_score(t, truth, meas, *, gap=20, sigma=SIGMA):
    """Blank out ``gap``-sample GPS dropouts, let each tracker coast, and score
    the coast against true position at the blanked samples. That is the real
    dead-reckoning error, not a held-out-fix proxy for it."""
    n = len(t); gm = _gap_mask(n, gap); mg = meas.copy(); mg[gm] = np.nan
    res = run_methods(t, mg, sigma=sigma)
    return {nm: _pos_rmse(est, truth, gm) for nm, est in res.items()}


def glitch_score(t, truth, meas, *, frac=0.06, mag=12.0, seed=0, sigma=SIGMA):
    """Inject multipath spikes, N(0, ``mag``) on a ``frac`` of the fixes, and
    score each tracker against true position at the spiked samples. Adds
    dtfit's winsorized ``robust=True`` LSI, the integral method's robustness
    lever that the pointwise filters lack.

    Glitch placement uses a decorrelated child of ``seed``
    (``SeedSequence.spawn``) rather than a correlated integer offset, so the
    glitch stream is statistically independent of the noise realization
    ``bench_fn`` drew from the same ``seed``; an offset could alias the two."""
    n = len(t)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    gl = np.zeros(n, bool); idx = np.arange(WARM + 10, n)
    gl[rng.choice(idx, size=int(frac * idx.size), replace=False)] = True
    mgl = meas.copy(); mgl[gl, :2] += rng.normal(0.0, mag, (int(gl.sum()), 2))
    res = run_methods(t, mgl, sigma=sigma)
    res["dtfit LSI robust"] = G.dtfit_track(t, mgl, (1,), kind="lsi", model="poly",
                                            robust=True)[0]
    return {nm: _pos_rmse(est, truth, gl) for nm, est in res.items()}


def scenarios_mc(bench_fn, seeds=range(30), *, sigma=SIGMA, **bench_kw):
    """Monte-Carlo the three scenarios (clean smoothing / dropout coasting / glitch
    robustness) on ``bench_fn``; return ``{scenario: {method: mean RMSE}}``."""
    accs = {"clean": None, "dropout": None, "glitch": None}
    for sd in seeds:
        t, truth, meas, labels = bench_fn(seed=sd, sigma=sigma, **bench_kw)
        tabs = {
            "clean": {nm: row["overall"] for nm, row in
                      score(run_methods(t, meas, sigma=sigma), truth, labels).items()},
            "dropout": dropout_score(t, truth, meas, sigma=sigma),
            "glitch": glitch_score(t, truth, meas, seed=sd, sigma=sigma),
        }
        for k, tab in tabs.items():
            if accs[k] is None:
                accs[k] = {nm: [] for nm in tab}
            for nm, v in tab.items():
                accs[k][nm].append(v)
    return {k: {nm: float(np.mean(v)) for nm, v in acc.items()} for k, acc in accs.items()}


# (2) public real datasets: the drop-in loader.
#
# EXTERNAL DATASETS (download separately; see the notebook's dataset section):
#   * Google Smartphone Decimeter Challenge (GSDC 2022), Kaggle, dm-level RTK
#     truth.
#   * UrbanNav (HK / Tokyo urban canyon), GitHub, SPAN-CPT RTK/INS ground
#     truth.
#   * comma2k19, highway, global-pose reference from a fused INS+RTK.
# Provide a CSV carrying a measured fix and a truth fix per row; this projects
# both to local ENU about the first truth point and runs the identical
# run_methods / score pipeline.
def load_external(path, *, lat="lat", lon="lon", truth_lat="truth_lat",
                  truth_lon="truth_lon", t_col="t", alt=None, truth_alt=None):
    """Load a public-dataset CSV into ``(t, truth(n,3), meas(n,3), labels)`` in
    local ENU metres, scored against the dataset's own RTK/INS truth: real
    dm-level ground truth, where the rig has only a held-out-fix proxy. Column
    names are configurable per dataset."""
    import csv
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"no rows in {path}")
    def g(r, k):
        return float(r[k])
    lat0, lon0 = g(rows[0], truth_lat), g(rows[0], truth_lon)
    cl = np.cos(np.radians(lat0)); md = 111320.0
    n = len(rows)
    t = (np.array([g(r, t_col) for r in rows]) if t_col in rows[0] else np.arange(n, dtype=float))
    t = t - t[0]

    def enu(la, lo, al):
        return np.stack([(np.array(lo) - lon0) * cl * md, (np.array(la) - lat0) * md,
                         np.array(al) if al is not None else np.zeros(n)], axis=1)

    meas = enu([g(r, lat) for r in rows], [g(r, lon) for r in rows],
               [g(r, alt) for r in rows] if alt else None)
    truth = enu([g(r, truth_lat) for r in rows], [g(r, truth_lon) for r in rows],
                [g(r, truth_alt) for r in rows] if truth_alt else None)
    return t, truth, meas, np.array(["real"] * n)


def _ecef_to_geodetic(x, y, z):
    """WGS84 ECEF (m) -> (lat deg, lon deg, alt m), vectorized, by Bowring's
    closed form."""
    a = 6378137.0; e2 = 6.69437999014e-3
    b = a * np.sqrt(1.0 - e2); ep2 = (a * a - b * b) / (b * b)
    p = np.sqrt(x * x + y * y); th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), alt


def load_gsdc(gnss_path, gt_path, *, tol_ms=500):
    """Load a GSDC-2022 trip: the phone's WLS baseline (``device_gnss``, ECEF,
    one row per satellite) as the noisy measurement and the RTK
    ``ground_truth`` (lat/lon per epoch) as truth, aligned by epoch timestamp
    to the nearest within ``tol_ms`` and projected to local ENU. Returns
    ``(t, truth, meas, labels)``, exactly the shape ``run_methods`` and
    ``score`` consume, so a real dm-truthed drive flows straight through the
    benchmark. ``device_gnss`` repeats the WLS position across each epoch's
    satellite rows, hence the dedupe by ``utcTimeMillis`` first."""
    import csv
    wls = {}
    for r in csv.DictReader(open(gnss_path)):
        tm = int(float(r["utcTimeMillis"]))
        if tm not in wls:
            wls[tm] = (float(r["WlsPositionXEcefMeters"]), float(r["WlsPositionYEcefMeters"]),
                       float(r["WlsPositionZEcefMeters"]))
    wt = np.array(sorted(wls))
    wx = np.array([wls[k] for k in wt])
    wla, wlo, wal = _ecef_to_geodetic(wx[:, 0], wx[:, 1], wx[:, 2])

    gt = list(csv.DictReader(open(gt_path)))
    gtt = np.array([int(float(r["UnixTimeMillis"])) for r in gt])
    gla = np.array([float(r["LatitudeDegrees"]) for r in gt])
    glo = np.array([float(r["LongitudeDegrees"]) for r in gt])
    gal = np.array([float(r.get("AltitudeMeters") or 0.0) for r in gt])

    idx = np.clip(np.searchsorted(wt, gtt), 1, len(wt) - 1)
    pick = np.where(np.abs(gtt - wt[idx - 1]) <= np.abs(gtt - wt[idx]), idx - 1, idx)
    ok = np.abs(gtt - wt[pick]) <= tol_ms
    gla, glo, gal, gtt, pick = gla[ok], glo[ok], gal[ok], gtt[ok], pick[ok]
    lat0, lon0, cl, mdg = gla[0], glo[0], np.cos(np.radians(gla[0])), 111320.0

    def enu(la, lo, al, al0):
        return np.stack([(lo - lon0) * cl * mdg, (la - lat0) * mdg, al - al0], axis=1)

    truth = enu(gla, glo, gal, gal[0])
    meas = enu(wla[pick], wlo[pick], wal[pick], wal[pick][0])
    return (gtt - gtt[0]) / 1000.0, truth, meas, np.array(["real"] * len(gtt))


if __name__ == "__main__":
    import sys
    bench = figure8_benchmark if (len(sys.argv) > 1 and sys.argv[1] == "fig8") else ct_benchmark
    sc = scenarios_mc(bench, seeds=range(30))
    names = list(sc["glitch"])                      # glitch has the extra robust row
    print(f"{'method':<22}{'clean':>9}{'dropout':>9}{'glitch':>9}")
    for nm in names:
        row = "".join(f"{sc[k].get(nm, float('nan')):9.2f}" for k in ("clean", "dropout", "glitch"))
        print(f"{nm:<22}{row}")
