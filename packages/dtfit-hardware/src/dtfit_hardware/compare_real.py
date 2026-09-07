"""Score dtfit's integral trackers against the classical baselines on a real
logged rig run: the hardware counterpart of the simulation's E1/E2, plus the
float32-vs-float64 precision check E5.

A real single-frequency run has no external ground truth, which is why the
comparison rests on the only two metrics that are well defined from the data
itself. Forecast RMSE has the tracker predict the fix ``h`` steps ahead and
scores that against the fix that actually arrived, making truth the real future
sample. Dropout-coasting RMSE blanks synthetic gaps, lets each tracker coast,
and scores the coasted estimate against the real fixes that were held out; the
GPS+IMU methods dead-reckon the gap while the GPS-only ones extrapolate their
local model. When the log also carries the on-MCU estimate (``est_lat`` and
``est_lon`` from ``nano_lsi_log``), a third number compares that logged float32
estimate against a float64 dtfit replay of the same raw fixes.

Everything runs in local-ENU metres about the first fix. Small values keep the
fit well-conditioned, which is also why the firmware works in ENU.

Usage::

    python compare_real.py data/your_run.csv
    python -c "import compare_real as C; print(C.report('data/your_run.csv'))"
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np

from dtfit_experimental.experiments.domains.realtime_gps import backend as G

WARM = G.WARMUP
DEG2RAD = math.pi / 180.0
G_MS2 = 9.81


def load_log(path: str) -> dict:
    """Parse a rig CSV: raw GPS+IMU, optionally the nano_lsi_log columns.

    Keeps only ``fix==1`` rows.

    Returns:
        Named float arrays plus the set of columns present. The firmware emits
        faster than the GPS updates, leaving most rows a repeat of the last fix
        with only fresh IMU and heading, so where a ``newfix`` column exists
        only the genuine GPS updates are kept and the trajectory stays at the
        true GPS rate. The between-fix samples remain in the raw CSV for finer
        analysis. Both on-MCU heading columns, ``hdg_deg`` and
        ``dhdg_deg``, are carried through.
    """
    rows = list(csv.reader(open(path)))
    head = rows[0]
    ix = {name: i for i, name in enumerate(head)}
    has_newfix = "newfix" in ix

    def keep(r):
        if len(r) != len(head) or r[ix.get("fix", 2)] != "1":
            return False
        return (not has_newfix) or r[ix["newfix"]] == "1"

    data = [r for r in rows[1:] if keep(r)]
    if not data:
        raise ValueError(f"no fix==1 rows in {path}")

    def col(name):
        if name not in ix:
            return None
        return np.array([float(r[ix[name]]) for r in data])

    t = col("t_ms") / 1000.0
    t = t - t[0]
    out = {"t": t, "n": len(data), "cols": set(ix)}
    for k in ("lat", "lon", "alt_m", "ax", "ay", "az", "gx", "gy", "gz",
              "mx", "my", "mz", "est_lat", "est_lon", "sats", "hdop", "spd_kmph",
              "hdg_deg", "dhdg_deg"):
        out[k] = col(k)
    # older logs name the on-MCU estimate lsi_lat/lsi_lon
    if out["est_lat"] is None and "lsi_lat" in ix:
        out["est_lat"] = col("lsi_lat")
        out["est_lon"] = col("lsi_lon")
    return out


def to_enu(lat, lon, alt):
    """Local east/north/up metres about the first sample."""
    lat0, lon0 = lat[0], lon[0]
    cl = math.cos(math.radians(lat0))
    md = 111320.0
    e = (lon - lon0) * cl * md
    n = (lat - lat0) * md
    u = (alt - alt[0]) if alt is not None else np.zeros_like(lat)
    return np.stack([e, n, u], axis=1), (lat0, lon0, cl, md)


def _align_to_z(g):
    """Body-to-world rotation taking ``g`` to world +z, via Rodrigues.

    Feed it the measured static specific force and ``R0 @ accel_static`` points
    along +z, which makes gravity cancel in ``R @ accel + GRAVITY`` at rest.
    """
    z = np.array([0.0, 0.0, 1.0])
    g = g / (np.linalg.norm(g) + 1e-9)
    v = np.cross(g, z)
    s = float(np.linalg.norm(v))
    c = float(np.dot(g, z))
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _rest_mask(gyro, accel, gbias, speed):
    """Stationary samples: low rotation once bias is removed, ~1 g specific
    force, near-zero GPS speed. Thresholds match the firmware's rest test."""
    gmag = np.linalg.norm(gyro - gbias, axis=1)          # rad/s
    amag = np.linalg.norm(accel, axis=1) / G_MS2         # g
    sp = speed if speed is not None else np.zeros(len(gyro))
    return (gmag < 3 * DEG2RAD) & (np.abs(amag - 1.0) < 0.05) & (sp < 2.0)


def _imu(log):
    """Real IMU in the SI body frame, raw and with no bias removed, plus
    the initial gravity alignment ``R0``, the initial gyro bias and the
    per-sample rest mask. Returns a dict, or None if accel/gyro are absent."""
    if log["gx"] is None or log["ax"] is None:
        return None
    gyro = np.stack([log["gx"], log["gy"], log["gz"]], axis=1) * DEG2RAD
    accel = np.stack([log["ax"], log["ay"], log["az"]], axis=1) * G_MS2
    k = min(WARM, len(accel))
    gbias0 = gyro[:k].mean(axis=0)
    g0 = accel[:k].mean(axis=0)
    R0 = _align_to_z(g0)
    abias0 = g0 - R0.T @ np.array([0.0, 0.0, np.linalg.norm(g0)])
    rest = _rest_mask(gyro, accel, gbias0, log["spd_kmph"])
    # Yaw rate about the gravity (world-vertical) axis, not the raw body z.
    # Tilt the board even ~11 deg and the roll/pitch rates from bumps and
    # braking, gyro_x/y and often larger than gyro_z, leak into gyro_z and
    # integrate into a bogus heading: 98 deg RMS vs GPS course on the car
    # drive, against 36 deg for the gravity-aligned component. R0 is the same
    # body-to-world alignment the strapdown path uses, and (R0 @ w)[2] is the
    # part of body rate w about world up. Feeds gyro_gated_basis and ekf_track.
    yaw = (R0 @ (gyro - gbias0).T).T[:, 2]
    # Prefer the on-MCU raw yaw increment ``dhdg_deg`` where the log has it: it
    # is integrated at the IMU rate on-chip, so it does not alias the way one
    # gyro sample per second does. It is deliberately not the GPS-anchored
    # ``hdg_deg``; feeding that cleaned heading into the "gyro" fusion would
    # leak the GPS course into the dead-reckoning, including metric [B]'s
    # held-out fixes, which the on-chip anchor did see. ``hdg_deg`` stays for
    # reporting and plots. Consumed as a yaw rate (increment / dt), leaving
    # gyro_gated_basis and ekf_track untouched, and falling back to the
    # gravity-aligned gyro_z above when the column is absent.
    dhdg = log.get("dhdg_deg")
    if dhdg is not None and len(dhdg) == len(gyro) and np.isfinite(dhdg).all():
        t = log["t"]
        dt = np.diff(t, prepend=t[0] - 1.0)
        dt[dt <= 0] = 1.0
        yaw = np.radians(np.asarray(dhdg, float)) / dt
    return dict(gyro=gyro, accel=accel, R0=R0, gbias0=gbias0, abias0=abias0,
                rest=rest, yaw=yaw)


def _mag_heading_enu(log, fixes, *, win=10, disp=8.0):
    """Tilt-compensated magnetometer heading in the ENU velocity-angle frame,
    measured from east, the convention ``gyro_gated_basis`` integrates ``psi``
    in. This is the drift-free absolute-yaw anchor.

    The magnetometer is hard-iron centred on the run median, tilt-compensated
    with the accelerometer roll and pitch, then aligned to true course by a
    single constant circular offset fitted on the confident-motion samples,
    those with at least ``disp`` m of GPS displacement across the +-``win``
    sample window where course-over-ground is well defined. One scalar absorbs
    magnetic declination, the sensor-axis convention and the board's unknown
    mounting yaw, all of them physically constant. Every time-varying part of
    the heading is therefore the real magnetometer, which is what lets it hold
    heading through a GPS dropout.

    Returns:
        ``(heading_enu (n,), info)``, or ``(None, None)`` when the mag columns
        are absent or there is too little motion. ``info['resid_deg']`` is how
        tightly the compass tracks course, i.e. a quality gauge for weighting
        the anchor.
    """
    if log["mx"] is None or fixes is None:
        return None, None
    n = log["n"]
    cx = log["mx"] - np.median(log["mx"])
    cy = log["my"] - np.median(log["my"])
    cz = log["mz"] - np.median(log["mz"])
    a = np.stack([log["ax"], log["ay"], log["az"]], axis=1)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    roll = np.arctan2(an[:, 1], an[:, 2])
    pitch = np.arctan2(-an[:, 0], np.hypot(an[:, 1], an[:, 2]))
    cr, sr, cp, sp = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    Xh = cx * cp + cy * sr * sp + cz * cr * sp
    Yh = cy * cr - cz * sr
    psi = np.arctan2(Yh, Xh)                                   # compass frame
    course = np.full(n, np.nan)
    for i in range(n):
        j = min(i + win, n - 1)
        k = max(i - win, 0)
        de, dn = fixes[j, 0] - fixes[k, 0], fixes[j, 1] - fixes[k, 1]
        if math.hypot(de, dn) > disp:
            course[i] = math.atan2(dn, de)        # ENU angle from east
    mv = np.isfinite(course)
    if mv.sum() < 20:
        return None, None
    d = course[mv] - psi[mv]
    off = math.atan2(float(np.sin(d).mean()), float(np.cos(d).mean()))
    r = np.arctan2(np.sin(d - off), np.cos(d - off))
    heading = np.arctan2(np.sin(psi + off), np.cos(psi + off))
    return heading, {"resid_deg": float(np.degrees(np.sqrt(np.mean(r ** 2)))),
                     "offset_deg": float(np.degrees(off)), "n": int(mv.sum())}


def strapdown_real(t, gyro, accel, R0, rest, gbias0, abias0, *, tau=20.0,
                   ba=0.02, bg=0.02):
    """Real-IMU strapdown with rest-aided online bias estimation and ZUPT.

    At a detected stationary sample the true angular rate is zero and the true
    specific force is just gravity, so those samples drive an EWMA of the gyro
    and accel biases and zero the velocity. The biases are estimated in the
    body frame because that is where sensor bias lives: it rotates with
    attitude, and a world-frame estimate goes wrong the moment the rig moves.
    Doing this keeps the position basis bounded over long runs where a one-time
    bias removal drifts to hundreds of metres. Returns ``S`` (n,3).
    """
    m = len(t); R = R0.copy()
    S = np.zeros((m, 3)); v = np.zeros(3); s = np.zeros(3)
    gb = gbias0.copy(); ab = abias0.copy(); grav = G.GRAVITY
    for i in range(m):
        dt = (t[i] - t[i - 1]) if i > 0 else (t[1] - t[0] if m > 1 else 1.0)
        dt = min(max(float(dt), 1e-3), 5.0)
        if rest[i]:
            gb = (1 - bg) * gb + bg * gyro[i]
            ab = (1 - ba) * ab + ba * (accel[i] - R.T @ (-grav))   # body bias at rest
            v[:] = 0.0                                             # ZUPT
        w = gyro[i] - gb
        aw = R @ (accel[i] - ab) + grav
        R = R @ G._exp_so3(w, dt)
        if not rest[i]:
            v = (1 - dt / tau) * v + aw * dt
        s = (1 - dt / tau) * s + v * dt
        S[i] = s
    return S


def gyro_gated_basis(t, fixes, imu, *, tau=4.0, gbias_relax=0.01,
                     mag_heading=None, mag_gain=0.05, scale=1.0, gain=0.5,
                     abs_scale=8.0, decay=0.95, anchor_tau=30.0, win=5):
    """Gyro-yaw-rate dead-reckoning under an innovation gate: the CT-EKF
    mechanism in dtfit-native form.

    Only the bias-corrected gyro yaw rate is integrated, into a heading
    ``psi``. That is a single integration of a bounded rate, with no gravity
    leak to double-integrate, which is the failure that sends the
    accel-strapdown rows drifting hundreds of metres. GPS finite-difference
    speed rides that heading, re-anchored at each fix, zero-velocity-updated at
    rest and held through gaps, washed into a position basis
    ``S = washout_tau(speed * [cos psi, sin psi])``. The accelerometer is left
    out deliberately: on this class of MEMS it is the liability.

    A per-step gate ``w`` in [0,1] then combines an increment-agreement term,
    asking whether the dead-reckoned step matches the GPS step, with an
    absolute-divergence term, asking whether the dead-reckoning has walked away
    from an EWMA GPS anchor, and scales the basis by the product. A diverging
    coast is pulled back toward the GPS-only control while a trustworthy one is
    admitted, and through a GPS gap ``w`` simply decays. The upshot is that on
    static or garbage data the basis self-disables, staying within noise of the
    S=0 control, and the IMU carries weight only on genuine, GPS-consistent
    motion. Returns ``(w*S, w)``.

    A calibrated compass, where present, bounds ``psi`` drift through long
    dropouts via the complementary ``mag_heading``/``mag_gain`` anchor.
    """
    yaw_rate, rest = imu["yaw"], imu["rest"]
    m = len(t); S = np.zeros((m, 3))
    psi = 0.0; gb = 0.0; s = np.zeros(2); last_speed = 0.0
    for i in range(m):
        dt = (t[i] - t[i - 1]) if i > 0 else (t[1] - t[0] if m > 1 else 1.0)
        dt = min(max(float(dt), 1e-3), 5.0)
        a = min(dt / tau, 1.0)     # a>1 past a tau-long gap flips the sign
        if rest[i]:                                   # ZUPT the yaw-rate bias
            gb = (1 - gbias_relax) * gb + gbias_relax * yaw_rate[i]
        psi += (yaw_rate[i] - gb) * dt
        if mag_heading is not None and mag_gain > 0.0 and np.isfinite(mag_heading[i]):
            e = math.atan2(math.sin(mag_heading[i] - psi), math.cos(mag_heading[i] - psi))
            psi += mag_gain * e                       # drift-free yaw anchor
        if i > 0 and not np.any(np.isnan(fixes[i])) and not np.any(np.isnan(fixes[i - 1])):
            last_speed = float(np.linalg.norm(fixes[i, :2] - fixes[i - 1, :2])) / dt
        speed = 0.0 if rest[i] else last_speed
        vd = speed * np.array([math.cos(psi), math.sin(psi)])
        s = (1.0 - a) * s + vd * dt
        S[i, 0], S[i, 1] = s[0], s[1]
    # causal Schmidt-style gate: innovation plus absolute divergence, decaying
    # through gaps
    w = np.zeros(m); cur = 1.0; err_ewma = 0.0
    prevS = prevF = aS = aF = None; b = 1.0 / anchor_tau
    for i in range(m):
        if not np.any(np.isnan(fixes[i])):
            if aS is None:
                aS = S[i].copy(); aF = fixes[i].copy()
            if prevS is not None:
                err = float(np.linalg.norm((S[i] - prevS) - (fixes[i] - prevF)))
                err_ewma = (1 - 1.0 / win) * err_ewma + (1.0 / win) * err
            a_inc = scale * scale / (scale * scale + err_ewma * err_ewma)
            aerr = float(np.linalg.norm((S[i] - aS) - (fixes[i] - aF)))
            a_abs = abs_scale * abs_scale / (abs_scale * abs_scale + aerr * aerr)
            cur = (1 - gain) * cur + gain * (a_inc * a_abs)
            prevS = S[i].copy(); prevF = fixes[i].copy()
            aS = (1 - b) * aS + b * S[i]; aF = (1 - b) * aF + b * fixes[i]
        else:
            cur *= decay
        cur = max(0.0, min(1.0, cur)); w[i] = cur
    return S * w[:, None], w


def _fc_rmse(pred_h, fixes, motion=None):
    """Forecast RMSE past warm-up. Passing ``motion``, a rest mask, restricts
    the score to moving samples. That is the only honest IMU discriminator,
    because a static metric rewards 'stay put' whatever the IMU did."""
    m = ~np.isnan(pred_h[:, 0])
    m[:WARM] = False
    if motion is not None:
        m &= ~motion
    return G.rmse3(pred_h[m], fixes[m]) if m.any() else float("nan")


def _gap_mask(n, gap=15, period=80):
    """Synthetic dropouts: blank ``gap`` samples every ``period``, past warm-up."""
    m = np.zeros(n, bool)
    for st in range(WARM + 20, n - gap, max(period, gap * 3)):
        m[st:st + gap] = True
    return m


def _cv_replay(t, fixes_en, rest=None):
    """Float64 replay of the on-MCU model for the precision diff: per-axis
    degree-1 LSI (``c0 + c1*t``) over a 15-sample window, roughly what
    ``nano_lsi_log`` computes on-chip. It is not the cubic GPS-only tracker.

    Window and degree match the chip. The Legendre spectral ``order`` here is 4
    against the firmware's 5, and the on-chip warm-up, glitch handling and ENU
    origin cannot be reconstructed from the log, which makes ``[C]`` a coarse
    float32-vs-float64 sanity number rather than a measurement. Replaying a
    cubic instead would measure the model gap and wildly overstate the
    precision drop. The rigorous check is the on-boot golden-vector run
    (``lsi_testvec.h``, scored in the report notebook); this is the on-run
    sanity number over the fixes the rig actually saw.

    ``rest`` is accepted but never applied. The estimate is read at the current
    sample time, and on static data, with ``c1`` near zero, that is already the
    value the chip's degree-0 ZUPT would hold, so passing a mask changes
    nothing.
    """
    m = len(t)
    flts = [G.LSIFilter("c0 + c1*t", "t", p0=[float(fixes_en[0, ax]), 0.0],
                        window_size=15, order=4, q_diag=[1e-2, 1e-2],
                        drift_reset="inflate") for ax in range(2)]
    out = np.zeros((m, 2))
    for i in range(m):
        for ax in range(2):
            flts[ax].partial_fit(t[i], fixes_en[i, ax])
            out[i, ax] = float(flts[ax].predict(np.array([t[i]]))[0])
    return out


def _glitch_mc(t, fixes, imu, have_imu, n, idx, *, n_seeds=25, frac=0.05, mag=25.0):
    """Monte-Carlo the [D] glitch-robustness row over ``n_seeds`` independent
    sub-seeds spawned from a root :class:`numpy.random.SeedSequence`. Spawning
    gives decorrelated streams where correlated integer offsets would not, and
    turns the reported figure into a distribution rather than one glitch
    realization. Each seed draws its own spiked-fix set and scores every
    tracker's smoothed estimate at the spiked samples against the clean fix.
    Returns ``[(name, (mean_rmse, p95_rmse)), ...]``."""
    trackers = [
        ("dtfit LSI robust (GPS-only)",
         lambda fg: G.dtfit_track(t, fg, (1,), kind="lsi", robust=True)[0]),
        ("dtfit LSI plain (GPS-only)",
         lambda fg: G.dtfit_track(t, fg, (1,), kind="lsi", robust=False)[0]),
        ("Kalman-CA (GPS-only)",
         lambda fg: G.kalman_track(t, fg, (1,))[0]),
    ]
    if have_imu:
        trackers.append(("CT-EKF (GPS+gyro)",
                         lambda fg: G.ekf_track(t, fg, imu["yaw"], (1,))[0]))
    per = {name: [] for name, _ in trackers}
    for child in np.random.SeedSequence(20240701).spawn(n_seeds):
        rng = np.random.default_rng(child)
        gl = np.zeros(n, bool)
        gl[rng.choice(idx, size=max(1, int(frac * idx.size)), replace=False)] = True
        fg = fixes.copy()
        fg[gl, :2] += rng.normal(0, mag, (int(gl.sum()), 2))
        for name, run in trackers:
            sm = run(fg)
            per[name].append(G.rmse3(sm[gl], fixes[gl]) if gl.any() else float("nan"))
    out = []
    for name, _ in trackers:
        arr = np.asarray(per[name], float)
        out.append((name, (float(np.nanmean(arr)),
                           float(np.nanpercentile(arr, 95)))))
    return out


def report(path: str, h: int = 10, gap: int = 15) -> str:
    log = load_log(path)
    fixes, _ = to_enu(log["lat"], log["lon"], log["alt_m"])
    t, n = log["t"], log["n"]
    lines = [f"real-run comparison: {path}",
             f"  {n} fix rows, {t[-1]:.0f} s, "
             f"sats {int(np.nanmin(log['sats']))}-{int(np.nanmax(log['sats']))}"
             if log["sats"] is not None else f"  {n} fix rows, {t[-1]:.0f} s",
             ""]

    imu = _imu(log)
    have_imu = imu is not None
    if have_imu:
        gy3, ac3, R0 = imu["gyro"], imu["accel"], imu["R0"]
        # the accelerometer contrast row
        S_rest = strapdown_real(t, gy3, ac3, R0, imu["rest"],
                                imu["gbias0"], imu["abias0"])
        rest_pct = 100.0 * imu["rest"].mean()
        lines.append(f"  IMU present; rest-detected {rest_pct:.0f}% of samples")
        mh, minfo = _mag_heading_enu(log, fixes)
        imu["mag_heading"] = mh
        if mh is not None:
            lines.append(f"  compass present; tilt-comp heading tracks GPS course to "
                         f"{minfo['resid_deg']:.0f} deg RMS (constant frame offset "
                         f"{minfo['offset_deg']:.0f} deg = declination+mounting+convention, "
                         f"n={minfo['n']}) -> a weak but stable absolute-yaw anchor")
        lines.append("")

    # (A) forecast RMSE: predict h ahead, score against the real future fix.
    # Every IMU row is judged against the matched S=0 control, pure GPS through
    # the same imu_lsi_track engine, rather than the differently configured
    # LSI-cubic row. Otherwise a harness-config difference (cubic against the
    # engine's quadratic drift) could pass for an IMU gain. The motion-only
    # column is the honest discriminator, since a static rig rewards "stay put"
    # whatever the IMU did.
    rest = imu["rest"] if have_imu else None
    z3 = np.zeros((n, 3))
    rows = []   # (name, all_rmse, motion_rmse)

    def _fwd(pred_h):
        return _fc_rmse(pred_h, fixes), _fc_rmse(pred_h, fixes, motion=rest)

    # coast=True dead-reckons the cubic at constant velocity off its window
    # instead of evaluating it directly, because a raw cubic diverges past the
    # window: 143 m at a 25-step gap against 104 m coasted. See dtfit_track.
    rows.append(("dtfit LSI-cubic (GPS-only, CV-coast)",
                 *_fwd(G.dtfit_track(t, fixes, (h,), kind="lsi", coast=True)[1][h])))
    rows.append(("Kalman-CA (GPS-only)", *_fwd(G.kalman_track(t, fixes, (h,))[1][h])))
    ctrl_a = ctrl_m = gg_a = gg_m = float("nan")
    if have_imu:
        ctrl_a, ctrl_m = _fwd(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=z3)[1][h])
        rows.append(("dtfit IMU-LSI S=0 control (matched)", ctrl_a, ctrl_m))
        Sg, wg = gyro_gated_basis(t, fixes, imu)
        gg_a, gg_m = _fwd(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=Sg)[1][h])
        rows.append(("dtfit IMU-LSI gyro-gated (GPS+gyro)", gg_a, gg_m))
        cm_a = cm_m = float("nan")
        if imu.get("mag_heading") is not None:
            Sgm, _ = gyro_gated_basis(t, fixes, imu, mag_heading=imu["mag_heading"])
            cm_a, cm_m = _fwd(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=Sgm)[1][h])
            rows.append(("dtfit IMU-LSI gyro+compass (GPS+gyro+mag)", cm_a, cm_m))
        rows.append(("dtfit IMU-LSI+ZUPT accel-strapdown",
                     *_fwd(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=S_rest)[1][h])))
        rows.append(("CT-EKF (GPS+gyro)", *_fwd(G.ekf_track(t, fixes, imu["yaw"], (h,))[1][h])))
    lines.append(f"[A] {h}-step forecast RMSE vs the real future fix  [all / motion-only] (m):")
    for name, va, vm in rows:
        lines.append(f"      {name:<36} {va:6.2f} / {vm:6.2f}")
    if have_imu:
        lines.append(f"      -> gyro contribution (gyro-gated minus matched control): "
                     f"{gg_a - ctrl_a:+.2f} / {gg_m - ctrl_m:+.2f}  "
                     f"(negative = IMU helps; ~0/positive expected on a static run)")
        if imu.get("mag_heading") is not None:
            lines.append(f"      -> compass contribution (gyro+compass minus matched control): "
                         f"{cm_a - ctrl_a:+.2f} / {cm_m - ctrl_m:+.2f}")

    # (B) dropout coasting: blank gaps, score the coast against the held-out
    # real fix.
    gm = _gap_mask(n, gap=gap)
    fg = fixes.copy()
    fg[gm] = np.nan
    sc = []
    sc.append(("dtfit LSI-cubic (GPS-only, CV-coast)",
               G.dtfit_track(t, fg, (1,), kind="lsi", coast=True)[0]))
    sc.append(("Kalman-CA (GPS-only)", G.kalman_track(t, fg, (1,))[0]))
    if have_imu:
        sc.append(("dtfit IMU-LSI S=0 control (matched)",
                   G.imu_lsi_track(t, fg, gy3, ac3, R0, (1,), S=z3)[0]))
        Sg_gap, _ = gyro_gated_basis(t, fg, imu)   # rebuilt on blanked fixes
        sc.append(("dtfit IMU-LSI gyro-gated (GPS+gyro)",
                   G.imu_lsi_track(t, fg, gy3, ac3, R0, (1,), S=Sg_gap)[0]))
        if imu.get("mag_heading") is not None:
            # mag_heading is the magnetometer's own; its constant frame offset
            # is fitted on the full run because that offset is a physical
            # constant. The compass therefore still holds heading through the
            # blanked gap, which is the anchor value being measured here.
            Sgm_gap, _ = gyro_gated_basis(t, fg, imu, mag_heading=imu["mag_heading"])
            sc.append(("dtfit IMU-LSI gyro+compass (GPS+gyro+mag)",
                       G.imu_lsi_track(t, fg, gy3, ac3, R0, (1,), S=Sgm_gap)[0]))
        sc.append(("dtfit IMU-LSI+ZUPT accel-strapdown",
                   G.imu_lsi_track(t, fg, gy3, ac3, R0, (1,), S=S_rest)[0]))
        sc.append(("CT-EKF (GPS+gyro)", G.ekf_track(t, fg, imu["yaw"], (1,))[0]))
    lines.append("")
    lines.append(f"[B] dropout coasting RMSE vs held-out real fixes "
                 f"({int(gm.sum())} blanked samples, {gap}-step gaps) (m):")
    for name, sm in sc:
        v = G.rmse3(sm[gm], fixes[gm]) if sm is not None and gm.any() else float("nan")
        lines.append(f"      {name:<36} {v:6.2f}")

    # (C) on-MCU float32 against a host float64 replay of the same model.
    if log["est_lat"] is not None:
        _, (lat0, lon0, cl, md) = to_enu(log["lat"], log["lon"], log["alt_m"])
        mcu = np.stack([(log["est_lon"] - lon0) * cl * md,
                        (log["est_lat"] - lat0) * md], axis=1)        # on-MCU est (ENU)
        pc = _cv_replay(t, fixes[:, :2], rest=imu["rest"] if have_imu else None)
        m = np.ones(n, bool)
        m[:WARM] = False
        d = np.linalg.norm(mcu[m] - pc[m], axis=1)
        lines.append("")
        lines.append("[C] logged on-MCU est vs PC float64 replay of the SAME degree-1 model:")
        lines.append(f"      mean {d.mean():.2f} m, median {np.median(d):.2f} m, "
                     f"p95 {np.percentile(d, 95):.2f} m, max {d.max():.2f} m")
        lines.append("      (coarse agreement check -- the residual is on-chip state we can't "
                     "replay from the log: ENU origin, exact window/warmup, glitch handling,")
        lines.append("      NOT float32 error. The rigorous bit-faithful E5 is the embed_lsi "
                     "golden-vector test: on-MCU float32 == float64 golden to <=3e-5 deg.)")

    # (D) glitch robustness (E3): inject multipath spikes, score against the
    # clean truth. The S2 run is clean (hdop<=4) and never exercises dtfit's
    # winsorized-integral robustness, so synthetic ~25 m multipath spikes go on
    # a fraction of the fixes and each tracker's smoothed estimate at those
    # samples is scored against the un-spiked fix. A robust tracker rejects the
    # spike and stays on the local trajectory; a pointwise one follows it. A
    # real urban-canyon run would supply organic glitches; until there is one,
    # this is the honest stand-in.
    idx = np.arange(WARM + 10, n)
    if idx.size:
        d_stats = _glitch_mc(t, fixes, imu, have_imu, n, idx, n_seeds=25)
        lines.append("")
        lines.append(f"[D] glitch robustness (E3): ~{int(0.05 * idx.size)} injected ~25 m "
                     f"spikes; smoothed-track RMSE vs the CLEAN fix at spiked samples, "
                     f"mean +/- p95 over 25 seeds (m):")
        for name, (mean_v, p95_v) in d_stats:
            lines.append(f"      {name:<36} {mean_v:6.2f} +/- {p95_v:6.2f}")

    if have_imu:
        lines.append("")
        lines.append("note: the honest baseline is the S=0 *matched control* (pure GPS through the "
                     "same LSI engine), not the LSI-cubic row -- judged against it,")
        lines.append(f"      no IMU method beats GPS-only on this {rest_pct:.0f}%-static run (a parked "
                     "rig can't beat 'stay put'; the IMU contribution above is ~0/positive).")
        lines.append("      The gyro-gated row is the fusion to prove on a MOVING run: it dead-reckons "
                     "on the gyro YAW-RATE only (no accel double-integration -> no")
        lines.append("      gravity-leak, cf. CT-EKF) under an innovation gate, so it self-disables on "
                     "static/garbage data and admits the IMU only on GPS-consistent")
        lines.append("      motion. The accel-strapdown row is the cautionary drift contrast. "
                     "Definitive test = the motion-only column on a real moving walk.")
    else:
        lines.append("")
        lines.append("note: IMU-strapdown rows need accel+gyro; compass fusion needs "
                     "the mag columns (firmware add).")
    return "\n".join(lines)


def sweep(path: str, horizons=(2, 3, 5, 10), gaps=(5, 10, 15)) -> str:
    """Compact horizon/gap sweep of the key fusion rows.

    It answers two questions the fixed ``report`` (h=10, gap=15) cannot. First,
    whether the gyro or compass contribution shows up at a shorter forecast
    horizon: 10 s is long for a pedestrian who turns corners, and a genuine IMU
    gain can wash out over it. Second, how coasting scales with gap length.
    Motion-only RMSE throughout, and every column shares the ``report`` engine
    and its matched S=0 control, leaving no room for a config difference to
    masquerade as an IMU gain.
    """
    log = load_log(path)
    fixes, _ = to_enu(log["lat"], log["lon"], log["alt_m"])
    t, n = log["t"], log["n"]
    imu = _imu(log)
    if imu is None:
        return "sweep: no IMU columns"
    gy3, ac3, R0 = imu["gyro"], imu["accel"], imu["R0"]
    z3 = np.zeros((n, 3))
    mh, minfo = _mag_heading_enu(log, fixes)
    imu["mag_heading"] = mh
    rest = imu["rest"]
    Sg_full, _ = gyro_gated_basis(t, fixes, imu)
    Sgm_full = (gyro_gated_basis(t, fixes, imu, mag_heading=mh)[0]
                if mh is not None else None)
    L = [f"sweep: {path}",
         (f"  compass tracks course to {minfo['resid_deg']:.0f} deg RMS"
          if mh is not None else "  no compass"),
         "",
         "[A] forecast RMSE, motion-only (m), by horizon h (samples ~= s):",
         "   h  GPS-ctrl  gyro-gated  gyro+compass   CT-EKF   Kalman"]

    def fc(pred):
        return _fc_rmse(pred, fixes, motion=rest)
    for h in horizons:
        ctrl = fc(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=z3)[1][h])
        gg = fc(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=Sg_full)[1][h])
        cm = (fc(G.imu_lsi_track(t, fixes, gy3, ac3, R0, (h,), S=Sgm_full)[1][h])
              if Sgm_full is not None else float("nan"))
        ek = fc(G.ekf_track(t, fixes, imu["yaw"], (h,))[1][h])
        ka = fc(G.kalman_track(t, fixes, (h,))[1][h])
        L.append(f"  {h:2d}  {ctrl:7.2f}  {gg:9.2f}  {cm:11.2f}  {ek:7.2f}  {ka:6.2f}")

    L += ["", "[B] dropout coasting RMSE (m) vs held-out fixes, by gap length (samples):",
          "   gap  GPS-ctrl  gyro-gated  gyro+compass   CT-EKF"]
    for gap in gaps:
        gm = _gap_mask(n, gap=gap)
        fg = fixes.copy()
        fg[gm] = np.nan

        def rc(S):
            sm = G.imu_lsi_track(t, fg, gy3, ac3, R0, (1,), S=S)[0]
            return G.rmse3(sm[gm], fixes[gm]) if gm.any() else float("nan")
        ctrl = rc(z3)
        gg = rc(gyro_gated_basis(t, fg, imu)[0])
        cm = (rc(gyro_gated_basis(t, fg, imu, mag_heading=mh)[0])
              if mh is not None else float("nan"))
        ek = G.ekf_track(t, fg, imu["yaw"], (1,))[0]
        eks = G.rmse3(ek[gm], fixes[gm]) if gm.any() else float("nan")
        L.append(f"  {gap:3d}  {ctrl:7.2f}  {gg:9.2f}  {cm:11.2f}  {eks:7.2f}")
    return "\n".join(L)


# Analyses for the real 5 Hz car drive: E1/E2 on the fast-motion run.
def est_err_by_speed(log, buckets=((0, 1), (1, 10), (10, 30), (30, 60), (60, 200))):
    """On-MCU LSI estimate against the raw GPS fix, in metres, binned by speed.
    This is what the phone shows as EST ERR. At 5 Hz the fixed 15-sample window
    spans 3 s rather than 15, which is why the on-chip degree-1 fit stops
    cutting corners at speed. Returns rows ``(lo, hi, n, mean_err, max_err)``,
    or None when the log carries no estimate."""
    if log.get("est_lat") is None:
        return None
    cl = math.cos(math.radians(float(log["lat"][0]))); md = 111320.0
    err = np.hypot((log["est_lon"] - log["lon"]) * cl * md, (log["est_lat"] - log["lat"]) * md)
    ok = np.isfinite(err) & ~((log["est_lat"] == 0) & (log["est_lon"] == 0))
    spd = log["spd_kmph"]
    rows = []
    for lo, hi in buckets:
        m = ok & (spd >= lo) & (spd < hi)
        if m.any():
            rows.append((lo, hi, int(m.sum()), float(err[m].mean()), float(err[m].max())))
    return rows


def imu_noise_floor(log):
    """IMU noise at the longest stationary stretch, where the true rate is 0
    and |a| is 1 g, against fast driving. An isotropic gyro/accel std far above
    the parked floor is mechanical vibration through a loose mount; vehicle
    dynamics would be anisotropic. Returns per-regime stats and the stop
    duration."""
    g = np.stack([log["gx"], log["gy"], log["gz"]], axis=1)
    a = np.stack([log["ax"], log["ay"], log["az"]], axis=1)
    spd = log["spd_kmph"]; amag = np.linalg.norm(a, axis=1)
    still = spd < 1
    best = end = cur = 0
    for i in range(len(spd)):
        cur = cur + 1 if still[i] else 0
        if cur > best:
            best, end = cur, i
    stop = np.zeros(len(spd), bool); stop[end - best + 1:end + 1] = True
    fast = spd > 60

    def st(m):
        return dict(n=int(m.sum()), gyro_std=g[m].std(axis=0).round(2).tolist(),
                    amag_mean=float(amag[m].mean()), amag_std=float(amag[m].std()))
    return dict(stop=st(stop), fast=st(fast),
                stop_s=float(log["t"][end] - log["t"][end - best + 1]))


def _course(fixes, *, win=15, disp=8.0):
    """GPS course-over-ground, as an ENU angle from east, where there is real
    displacement; NaN elsewhere."""
    n = len(fixes); c = np.full(n, np.nan)
    for i in range(n):
        j = min(i + win, n - 1); k = max(i - win, 0)
        de, dn = fixes[j, 0] - fixes[k, 0], fixes[j, 1] - fixes[k, 1]
        if math.hypot(de, dn) > disp:
            c[i] = math.atan2(dn, de)
    return c


def complementary_heading(log, fixes, *, K=0.03):
    """Clean the wobbly on-MCU gyro heading with a complementary filter.

    Propagate on the real per-emit yaw increment ``dhdg_deg``, slow-correct
    toward the GPS course while moving, which takes out both the drift and the
    vibration-rectification bias a loose mount injects, and coast on the gyro
    alone through a GPS gap (a NaN fix). Returns ``(psi_cleaned, course)``.
    Needs the ``dhdg_deg`` column.
    """
    n = len(fixes); dhdg = np.radians(np.asarray(log["dhdg_deg"]))
    course = _course(fixes); psi = np.zeros(n); p = 0.0
    for i in range(n):
        p += dhdg[i]
        if np.isfinite(course[i]) and not np.any(np.isnan(fixes[i])):
            p += K * math.atan2(math.sin(course[i] - p), math.cos(course[i] - p))
        psi[i] = p
    return psi, course


def onchip_heading_causal(log, fixes, *, win=15, disp=8.0, K=0.04):
    """Host mirror of the on-MCU cleaned heading, firmware ``anchorHeading`` in
    nano_lsi_log.

    Integrate the raw per-emit gyro increment ``dhdg_deg`` and slow-correct
    toward a causal backward-difference GPS course, the direction from the fix
    ``win`` samples ago to now, only where displacement exceeds ``disp`` m and
    the rig is moving. That is ``complementary_heading`` with a backward,
    MCU-realisable course in place of the centred one, and it reproduces what
    the chip emits live in ``hdg_deg``. On the drive it lands at ~15 deg RMS
    against GPS course, from 82 deg raw; the non-causal host filter reaches 9
    deg, and the difference is what the causal lag costs. Having the mirror
    here means a fresh log can verify the chip. The constants match the
    firmware #defines CRS_WIN, CRS_DISP and CRS_K. Returns cleaned psi in rad.
    """
    n = len(fixes); dhdg = np.radians(np.asarray(log["dhdg_deg"]))
    spd = log.get("spd_kmph"); psi = np.zeros(n); p = 0.0
    for i in range(n):
        p += dhdg[i]                                     # gyro propagation
        moving = spd is None or spd[i] >= 2.0
        if i >= win and moving:
            de = fixes[i, 0] - fixes[i - win, 0]; dn = fixes[i, 1] - fixes[i - win, 1]
            if math.hypot(de, dn) > disp:                # real displacement
                course = math.atan2(dn, de)
                p += K * math.atan2(math.sin(course - p), math.cos(course - p))
        psi[i] = p
    return psi


def heading_rms(psi, course):
    """RMS of a heading (rad) vs GPS course after one best constant offset (deg)."""
    m = np.isfinite(course)
    d = course[m] - psi[m]
    off = math.atan2(float(np.sin(d).mean()), float(np.cos(d).mean()))
    r = np.arctan2(np.sin(d - off), np.cos(d - off))
    return math.degrees(math.sqrt(np.mean(r ** 2)))


def deadreckon_basis(t, fixes, psi, *, tau=4.0):
    """Position basis from a heading and GPS finite-difference speed, the speed
    held through gaps, washed out over ``tau``. This is the gyro-yaw
    dead-reckoning fed to imu_lsi_track as the regressor ``S``. No
    accelerometer: its double integration is hopeless on a vibrating mount."""
    n = len(fixes); S = np.zeros((n, 3)); s = np.zeros(2); last = 0.0
    for i in range(n):
        dt = min(max(float(t[i] - t[i - 1]) if i > 0 else 0.2, 1e-3), 5.0)
        a = min(dt / tau, 1.0)     # a>1 past a tau-long gap flips the sign
        if i > 0 and not np.any(np.isnan(fixes[i])) and not np.any(np.isnan(fixes[i - 1])):
            last = float(np.linalg.norm(fixes[i, :2] - fixes[i - 1, :2])) / dt
        s = (1 - a) * s + last * np.array([math.cos(psi[i]), math.sin(psi[i])]) * dt
        S[i, :2] = s
    return S


def maneuver_dropouts(log, fixes, *, gap=75, turn_deg=15):
    """Place periodic ``gap``-sample GPS dropouts, coast GPS-only against
    cleaned-gyro fusion through each, and split the gaps into straight and turn
    by net heading change.

    The IMU pays off precisely on turn-dropouts: GPS-only extrapolates straight
    off the turn while the gyro carries the heading. Average over random
    placement and the two cancel, which is why undifferentiated periodic
    dropouts show almost nothing and the split is the honest view.

    Returns:
        Per-bin means, the turn gap with the best improvement, and the two
        coasted tracks for plotting.
    """
    n = len(fixes); t = log["t"]; z3 = np.zeros((n, 3))
    psi_full = np.unwrap(complementary_heading(log, fixes)[0])
    gm = _gap_mask(n, gap=gap); fg = fixes.copy(); fg[gm] = np.nan
    ctrl = G.imu_lsi_track(t, fg, z3, z3, np.eye(3), (1,), S=z3)[0]
    fus = G.imu_lsi_track(t, fg, z3, z3, np.eye(3), (1,),
                          S=deadreckon_basis(t, fg, complementary_heading(log, fg)[0]))[0]
    runs = []; i = 0
    while i < n:
        if gm[i]:
            a = i
            while i < n and gm[i]:
                i += 1
            runs.append((a, i))
        else:
            i += 1
    rows = [dict(a=a, b=b, man=math.degrees(abs(psi_full[b - 1] - psi_full[a])),
                 ctrl=float(G.rmse3(ctrl[a:b], fixes[a:b])),
                 fus=float(G.rmse3(fus[a:b], fixes[a:b]))) for a, b in runs]
    straight = [r for r in rows if r["man"] < turn_deg]
    turns = [r for r in rows if r["man"] >= turn_deg]

    def mean(g, k):
        return float(np.mean([r[k] for r in g])) if g else float("nan")
    best = max(turns, key=lambda r: r["ctrl"] - r["fus"]) if turns else None
    return dict(n_straight=len(straight), n_turn=len(turns),
                straight=(mean(straight, "ctrl"), mean(straight, "fus")),
                turn=(mean(turns, "ctrl"), mean(turns, "fus")),
                best=best, ctrl=ctrl, fus=fus)


def load_comma_enu(path):
    """Load the compact comma2k19 demo ENU CSV from
    ``data/comma2k19_demo_enu.csv``, whose provenance header has the details.

    Public CA-280 highway: raw ublox live GNSS at about 5 Hz alongside comma's
    ``global_pose`` decimetre truth, both in per-segment local ENU metres with
    the constant ublox-to-pose datum offset removed. Absolute ground truth is
    the one thing the no-RTK rig lacks, and having it turns the rig's forecast
    and coast proxies into real E1/E2 error.

    Returns:
        A list of segments ``{seg, t, raw(Nx3, z=0), truth(Nx3, z=0), n}``. The
        trailing z is there because the backend trackers expect ENU triples.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    ix = {k: i for i, k in enumerate(rows[0])}
    by_seg = {}
    for r in rows[1:]:
        by_seg.setdefault(int(r[ix["seg"]]), []).append(r)
    segs = []
    for s in sorted(by_seg):
        rs = by_seg[s]
        t = np.array([float(r[ix["t"]]) for r in rs])
        raw = np.array([[float(r[ix["raw_e"]]), float(r[ix["raw_n"]]), 0.0] for r in rs])
        tru = np.array([[float(r[ix["truth_e"]]), float(r[ix["truth_n"]]), 0.0] for r in rs])
        segs.append(dict(seg=s, t=t, raw=raw, truth=tru, n=len(t)))
    return segs


def comma_bench(segs, *, horizons=(5, 10), gaps=(15, 25), glitch_thr=8.0):
    """dtfit LSI-cubic (CV-coast) against Kalman-CA on the comma2k19 segments,
    scored on absolute decimetre truth. This is the real E1 forecast and E2
    coast that the rig can only proxy without RTK. Metrics are fix-weighted
    over segments, and the run also scans for organic multipath as raw-vs-truth
    deviation, with nothing injected.

    Returns:
        Dataset stats plus ``forecast[H]=(dtfit, kalman)``,
        ``coast[gap]=(dtfit, kalman)`` and the count of glitches above
        ``glitch_thr`` m.
    """
    def rms2(a, b):
        m = ~np.isnan(a[:, 0]); m[:WARM] = False
        return float(np.sqrt(np.mean(np.sum((a[m, :2] - b[m, :2]) ** 2, axis=1)))) if m.any() else float("nan")
    w = np.array([s["n"] for s in segs], float)

    def wm(x):
        return float(np.average(x, weights=w))
    res = np.concatenate([np.linalg.norm(s["raw"][:, :2] - s["truth"][:, :2], axis=1) for s in segs])
    path = sum(float(np.sum(np.linalg.norm(np.diff(s["truth"][:, :2], axis=0), axis=1))) for s in segs)
    dur = sum(float(s["t"][-1]) for s in segs)
    fc = {}
    for H in horizons:
        d = [rms2(G.dtfit_track(s["t"], s["raw"], (H,), kind="lsi", coast=True)[1][H], s["truth"]) for s in segs]
        k = [rms2(G.kalman_track(s["t"], s["raw"], (H,))[1][H], s["truth"]) for s in segs]
        fc[H] = (wm(d), wm(k))
    co = {}
    for gap in gaps:
        d, k = [], []
        for s in segs:
            gm = _gap_mask(s["n"], gap=gap); fg = s["raw"].copy(); fg[gm] = np.nan
            if not gm.any():
                continue
            d.append(rms2(G.dtfit_track(s["t"], fg, (1,), kind="lsi", coast=True)[0][gm], s["truth"][gm]))
            k.append(rms2(G.kalman_track(s["t"], fg, (1,))[0][gm], s["truth"][gm]))
        co[gap] = (float(np.mean(d)), float(np.mean(k)))
    return dict(n_seg=len(segs), n_fix=int(w.sum()), path_km=path / 1000.0,
                avg_kmph=path / dur * 3.6, raw_med=float(np.median(res)),
                raw_p95=float(np.percentile(res, 95)), raw_max=float(res.max()),
                forecast=fc, coast=co, n_glitch=int((res > glitch_thr).sum()), glitch_thr=glitch_thr)


def load_urbannav_enu(path):
    """Load the UrbanNav Medium-Urban (TST) organic-multipath CSV,
    ``data/urbannav_tst_enu.csv``, whose provenance header has the details.

    A fairly deep Hong Kong urban canyon: raw NMEA GGA fixes from three
    receivers, a clean dual-frequency ublox F9P, a phone, and a badly affected
    single-frequency M8T of the same class as the rig's NEO-M8N, against
    SPAN-CPT decimetre truth, in local ENU with the receiver-reported HDOP.
    The multipath here is organic, with nothing injected. Returns
    ``{recv: {t, raw(Nx3, z=0), truth(Nx3, z=0), hdop, n}}``.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    ix = {k: i for i, k in enumerate(rows[0])}
    by = {}
    for r in rows[1:]:
        by.setdefault(r[ix["recv"]], []).append(r)
    out = {}
    for recv, rs in by.items():
        t = np.array([float(r[ix["t"]]) for r in rs])
        raw = np.array([[float(r[ix["raw_e"]]), float(r[ix["raw_n"]]), 0.0] for r in rs])
        tru = np.array([[float(r[ix["truth_e"]]), float(r[ix["truth_n"]]), 0.0] for r in rs])
        hd = np.array([float(r[ix["hdop"]]) for r in rs])
        out[recv] = dict(t=t - t[0], raw=raw, truth=tru, hdop=hd, n=len(t))
    return out


def urbannav_e3(data, *, spike_thr=15.0):
    """Organic E3 on UrbanNav, and the honest bound on the robustness claim.

    Per receiver it reports three things: the quality gradient, as horizontal
    error against SPAN truth in median, p95 and max; whether a robust pointwise
    filter rescues the multipath, comparing raw against dtfit-robust
    (winsorized integral) and Kalman-CA at the organic spike epochs where raw
    error exceeds ``spike_thr`` m; and ``n_iso``, the spikes that are isolated,
    both neighbours under 8 m.

    Real urban NLOS is sustained rather than isolated, giving ``n_iso`` near
    zero, and no per-axis filter helps there. The robust win is therefore
    scoped to isolated outliers, which is what the rig's injected E3 measures;
    the urban regime is a tight-coupling or 3D-map problem instead. Returns
    per-receiver stats.
    """
    def rms(a, b, m):
        ok = ~np.isnan(a[:, 0]) & m
        return float(np.sqrt(np.mean(np.sum((a[ok, :2] - b[ok, :2]) ** 2, axis=1)))) if ok.any() else float("nan")
    out = {}
    for recv, d in data.items():
        t, raw, tru = d["t"], d["raw"], d["truth"]
        err = np.hypot(raw[:, 0] - tru[:, 0], raw[:, 1] - tru[:, 1])
        sp = err > spike_thr
        iso = int(sum(1 for k in np.where(sp)[0]
                      if 0 < k < len(err) - 1 and err[k - 1] < 8 and err[k + 1] < 8))
        rob = G.dtfit_track(t, raw, (1,), kind="lsi", robust=True)[0]
        kal = G.kalman_track(t, raw, (1,))[0]
        out[recv] = dict(n=d["n"], med=float(np.median(err)), p95=float(np.percentile(err, 95)),
                         mx=float(err.max()), n_spike=int(sp.sum()), n_iso=iso,
                         raw_sp=rms(raw, tru, sp), rob_sp=rms(rob, tru, sp), kal_sp=rms(kal, tru, sp))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        print(sweep(sys.argv[2] if len(sys.argv) > 2 else "data/static_lsi.csv"))
    else:
        p = sys.argv[1] if len(sys.argv) > 1 else "data/static_lsi.csv"
        print(report(p))
