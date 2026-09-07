"""Leg 3: the streaming filter on both datasets.

The drift detector tests once per effective window, so a filter cannot
report a change sooner than its window: a 1000-sample window can track a
velocity but cannot meet a 60-day detection rule. Each dataset therefore
gets the configurations it can actually serve -- a short window for
detection, a long one for tracking -- and every row says which it came
from.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dtfit.streaming import LSIFilter

from . import isd, ngl

DAYS_PER_YEAR = 365.25
# ImageFilter builds its DriftDetector with warmup=3 and tests it once
# per window_size full-window samples, so after a detection no test can
# fire for another warmup * window samples. The reachability rule below
# mirrors that; changing one without the other makes the recall
# denominator wrong.
WARMUP = 3


@dataclass(frozen=True)
class FilterConfig:
    """One streaming-filter configuration.

    ``window`` is the sample cap and therefore the detector's stride;
    ``horizon_days`` is how long after an event a flag still counts as
    explaining it. ``q_diag`` is the per-parameter process noise, or None
    for the filter's default of 0.01. ``adaptive`` is False in both
    configurations: an adaptive window makes the stride follow the data
    and measurably loses steps (ruling 15).
    """

    key: str
    expr: str
    names: tuple[str, ...]
    order: int
    window: int
    q_diag: tuple[float, ...] | None = None
    horizon_days: float = 60.0
    time_unit_days: float = 1.0
    adaptive: bool = False


NGL_CONFIGS = {
    "detection": FilterConfig(
        key="detection", expr="c + v*t", names=("c", "v"), order=5,
        window=40, horizon_days=60.0, time_unit_days=DAYS_PER_YEAR,
    ),
    "tracking": FilterConfig(
        key="tracking",
        expr=(
            "c + v*t + a1*cos(2*pi*t) + b1*sin(2*pi*t)"
            " + a2*cos(4*pi*t) + b2*sin(4*pi*t)"
        ),
        names=("a1", "a2", "b1", "b2", "c", "v"), order=24, window=1000,
        horizon_days=60.0, time_unit_days=DAYS_PER_YEAR,
    ),
}

ISD_CONFIG = FilterConfig(
    key="diurnal",
    expr="c + v*t + a*cos(2*pi*t) + b*sin(2*pi*t)",
    names=("a", "b", "c", "v"), order=16, window=48,
    horizon_days=2.0, time_unit_days=1.0,
)

FILTER_COLUMNS = [
    "station", "component", "config", "n_updates", "station_years",
    "seconds", "us_per_update", "n_flags", "n_steps", "n_detected",
    "n_reachable", "n_detected_reachable", "recall_reachable",
    "median_delay_days", "n_false_alarms", "false_alarms_per_year",
    "event_window_share", "chance_false_alarms_per_year", "v", "c",
]
ISD_FILTER_COLUMNS = [
    "station", "component", "config", "n_updates", "station_years",
    "seconds", "us_per_update", "n_flags", "n_events",
    "n_flags_explained", "n_flags_unexplained", "explained_share",
    "event_window_share", "v", "c", "n_moves", "n_quality_failures",
]
STEP_COLUMNS = [
    "station", "component", "config", "step_year", "step_index", "code",
    "magnitude", "distance_km", "threshold_km", "dist_ratio", "mag_bin",
    "dist_bin", "ratio_bin", "reachable", "detected", "delay_days",
]
ISD_FLAG_COLUMNS = [
    "station", "field", "config", "flag_day", "explained_by",
    "event_day", "lag_days",
]


def magnitude_bin(magnitude: float | None) -> str:
    """The reporting bin of an earthquake magnitude; ``"none"`` for a step
    with no magnitude (an equipment change)."""
    if magnitude is None:
        return "none"
    m = float(magnitude)
    if m < 5.0:
        return "<5"
    if m < 6.0:
        return "5-6"
    if m < 7.0:
        return "6-7"
    return ">=7"


def distance_bin(distance_km: float | None) -> str:
    """The reporting bin of an epicentral distance in km; ``"none"``
    without one."""
    if distance_km is None:
        return "none"
    d = float(distance_km)
    if d < 20.0:
        return "<20"
    if d < 100.0:
        return "20-100"
    return ">=100"


def ratio_bin(
    distance_km: float | None, threshold_km: float | None
) -> str:
    """The reporting bin of ``distance / threshold``, the step database's
    own proxy for whether an offset is expected at all.

    The database lists a candidate whenever the event is inside its
    magnitude-dependent threshold radius, so a ratio near 1 is a distant
    event that most likely moved nothing; a ratio well under 1 is the
    near-field case an offset is expected in. ``"none"`` when either
    number is missing or the threshold is not positive; magnitude alone
    is not enough, because 34.6 percent of the database's earthquake
    entries are magnitude 7 or above and most of those are far away.
    """
    if distance_km is None or threshold_km is None:
        return "none"
    thr = float(threshold_km)
    if not thr > 0.0:
        return "none"
    r = float(distance_km) / thr
    if r < 0.25:
        return "<0.25"
    if r < 0.5:
        return "0.25-0.5"
    if r < 1.0:
        return "0.5-1"
    return ">=1"


def event_window_share(
    event_times: Sequence[float],
    *,
    t0: float,
    t1: float,
    before: float = 0.0,
    after: float = 0.0,
) -> float:
    """The fraction of ``[t0, t1]`` covered by the union of the windows
    ``[e - before, e + after]`` around ``event_times``.

    This is the null the observed false-alarm rate is read against: a
    flag dropped uniformly at random inside the span is "explained" with
    this probability. The NGL rule explains a flag with a step on either
    side, so it passes ``before = after = horizon``; the NOAA rule only
    looks backwards, so it passes ``before = 0``. On a typical NGL
    station with a median 22 database steps over 12 years the share is
    around 0.6, so a false-alarm rate quoted without it says little about
    the filter. Returns 0.0 for an empty span and is capped at 1.0.
    """
    span = float(t1) - float(t0)
    if not span > 0.0:
        return 0.0
    windows = sorted(
        (max(float(t0), float(e) - float(before)),
         min(float(t1), float(e) + float(after)))
        for e in event_times
    )
    total = 0.0
    edge: float | None = None
    for lo, hi in windows:
        if hi <= lo:
            continue
        if edge is None or lo > edge:
            total += hi - lo
            edge = hi
        elif hi > edge:
            total += hi - edge
            edge = hi
    return float(min(1.0, total / span))


def reachable_events(
    event_indices: Sequence[int],
    flag_indices: Sequence[int],
    *,
    min_window: int,
    window: int,
    warmup: int = WARMUP,
) -> list[bool]:
    """Which events the detector could have reported at all.

    An event is unreachable when it falls inside the detector's blind
    start (``min_window + warmup * window`` samples from the series
    start) or within ``warmup * window`` samples after the last flag
    raised before it: no test can fire there whatever the data does.
    Both lists are sample indices into the same series, sorted
    increasing. Returns one boolean per event, in order. Measured over
    60 real stations, 63.8 percent of consecutive step pairs sit closer
    than one blind period, so the raw step count is not a recall
    denominator.
    """
    blind_start = int(min_window) + int(warmup) * int(window)
    stride = int(warmup) * int(window)
    flags = sorted(int(f) for f in flag_indices)
    out: list[bool] = []
    for raw in event_indices:
        idx = int(raw)
        ok = idx >= blind_start
        if ok:
            before = [f for f in flags if f < idx]
            if before and idx - before[-1] < stride:
                ok = False
        out.append(bool(ok))
    return out


def run_filter(
    t: np.ndarray,
    y: np.ndarray,
    config: FilterConfig,
    *,
    p0: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run one filter over a series, recording its drift flags.

    ``p0`` defaults to zeros with the offset ``c`` at ``y[0]``:
    :class:`~dtfit.streaming.LSIFilter` takes a positional sequence in
    canonical parameter order, not a mapping.

    Returns:
        ``{"flags", "flag_indices", "n_updates", "n_drifts", "seconds",
        "us_per_update", "min_window", "params"}``; ``flags`` is an array
        of the sample times at which a flag was raised, in the series' own
        units, and ``flag_indices`` the sample indices of the same flags.
        ``min_window`` is the filter's own value, which the reachability
        rule needs.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if p0 is None:
        p0 = [float(y[0]) if n == "c" else 0.0 for n in config.names]
    kwargs: dict[str, Any] = {
        "order": config.order, "window_size": config.window,
        "adaptive_window": config.adaptive, "p0": list(p0),
    }
    if config.q_diag is not None:
        kwargs["q_diag"] = list(config.q_diag)
    filt = LSIFilter(config.expr, "t", **kwargs)
    flags: list[float] = []
    flag_indices: list[int] = []
    started = time.perf_counter()
    for k in range(t.size):
        filt.partial_fit(t[k], y[k])
        if filt.drift_flag_:
            flags.append(float(t[k]))
            flag_indices.append(int(k))
    seconds = time.perf_counter() - started
    return {
        "flags": np.array(flags, dtype=float),
        "flag_indices": flag_indices,
        "n_updates": int(t.size),
        "n_drifts": int(filt.n_drifts_),
        "seconds": seconds,
        "us_per_update": (
            seconds / t.size * 1e6 if t.size else 0.0
        ),
        "min_window": int(filt.min_window),
        "params": dict(filt.params_),
    }


def match_flags(
    flag_times: np.ndarray,
    event_times: Sequence[float],
    *,
    horizon: float,
) -> dict[str, Any]:
    """Match drift flags to known events.

    An event is detected by the earliest unused flag in ``(event, event +
    horizon]``; a flag that explains no event and has none within
    ``horizon`` either side is a false alarm. Times and ``horizon`` share
    whatever unit the series uses.

    Returns:
        ``{"n_events", "n_detected", "delays", "median_delay",
        "n_false_alarms", "false_alarm_times"}``; ``delays`` has one entry
        per event, ``None`` where it was not detected.
    """
    flags = np.asarray(flag_times, dtype=float)
    events = [float(e) for e in event_times]
    used: set[int] = set()
    delays: list[float | None] = []
    for event in events:
        best: tuple[float, int] | None = None
        for i, flag in enumerate(flags):
            if i in used:
                continue
            lag = float(flag) - event
            if 0.0 <= lag <= horizon and (best is None or lag < best[0]):
                best = (lag, i)
        if best is None:
            delays.append(None)
        else:
            used.add(best[1])
            delays.append(best[0])
    false_times = [
        float(f) for i, f in enumerate(flags)
        if i not in used
        and not any(abs(float(f) - e) <= horizon for e in events)
    ]
    hit = [d for d in delays if d is not None]
    return {
        "n_events": len(events), "n_detected": len(hit), "delays": delays,
        "median_delay": float(np.median(hit)) if hit else None,
        "n_false_alarms": len(false_times),
        "false_alarm_times": false_times,
    }


def ngl_filter_station(
    path: Any,
    steps: Sequence[Any],
    *,
    configs: dict[str, FilterConfig] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter one station's three components under every configuration.

    ``steps`` are :class:`ngl.Step` entries for this station; their
    ``year`` is the decimal year, matched against the filter's own time
    axis (years from the station's first epoch). Returns
    ``(summary_rows, step_rows)``: one summary per component and
    configuration, and one row per step, component and configuration.

    Every step row carries ``reachable``, and the summary carries the
    recall over the reachable subset beside the recall over every step,
    because the detector is structurally blind for
    ``WARMUP * window`` samples after each flag. The summary also carries
    ``event_window_share`` and the false-alarm rate a randomly placed
    flag would produce at that share, so the observed rate is read
    against its null.
    """
    configs = NGL_CONFIGS if configs is None else configs
    path = Path(path)
    times: list[np.ndarray] = []
    cols: dict[str, list[np.ndarray]] = {c: [] for c in ngl.COMPONENTS}
    for chunk in ngl.read_tenv3(path):
        times.append(chunk.t)
        for comp in ngl.COMPONENTS:
            cols[comp].append(getattr(chunk, comp))
    t = np.concatenate(times)
    tc = t - t[0]
    span_years = float(tc[-1]) if tc.size else 0.0
    kept = [s for s in steps if t[0] <= float(s.year) <= t[-1]]
    events = [float(s.year) - float(t[0]) for s in kept]
    event_indices = [
        int(np.searchsorted(tc, e, side="left")) for e in events
    ]
    summary: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for comp in ngl.COMPONENTS:
        y = np.concatenate(cols[comp])
        for key, config in configs.items():
            horizon = config.horizon_days / config.time_unit_days
            out = run_filter(tc, y, config)
            matched = match_flags(out["flags"], events, horizon=horizon)
            reach = reachable_events(
                event_indices, out["flag_indices"],
                min_window=out["min_window"], window=config.window,
            )
            n_reachable = sum(1 for r in reach if r)
            hit_reachable = sum(
                1 for r, d in zip(reach, matched["delays"])
                if r and d is not None
            )
            share = event_window_share(
                events, t0=0.0, t1=span_years,
                before=horizon, after=horizon,
            )
            n_flags = int(out["flags"].size)
            summary.append({
                "station": path.stem, "component": comp, "config": key,
                "n_updates": out["n_updates"],
                "station_years": round(span_years, 4),
                "seconds": round(out["seconds"], 4),
                "us_per_update": round(out["us_per_update"], 2),
                "n_flags": n_flags,
                "n_steps": matched["n_events"],
                "n_detected": matched["n_detected"],
                "n_reachable": n_reachable,
                "n_detected_reachable": hit_reachable,
                "recall_reachable": (
                    round(hit_reachable / n_reachable, 4)
                    if n_reachable else None
                ),
                "median_delay_days": (
                    None if matched["median_delay"] is None
                    else round(matched["median_delay"]
                               * config.time_unit_days, 2)
                ),
                "n_false_alarms": matched["n_false_alarms"],
                "false_alarms_per_year": (
                    round(matched["n_false_alarms"] / span_years, 4)
                    if span_years > 0 else None
                ),
                "event_window_share": round(share, 4),
                "chance_false_alarms_per_year": (
                    round(n_flags * (1.0 - share) / span_years, 4)
                    if span_years > 0 else None
                ),
                "v": out["params"].get("v"),
                "c": out["params"].get("c"),
            })
            for j, (step, delay) in enumerate(
                zip(kept, matched["delays"])
            ):
                ratio = None
                if (step.distance_km is not None
                        and step.threshold_km
                        and float(step.threshold_km) > 0.0):
                    ratio = round(
                        float(step.distance_km)
                        / float(step.threshold_km), 4
                    )
                step_rows.append({
                    "station": path.stem, "component": comp,
                    "config": key, "step_year": float(step.year),
                    "step_index": event_indices[j],
                    "code": int(step.code), "magnitude": step.magnitude,
                    "distance_km": step.distance_km,
                    "threshold_km": step.threshold_km,
                    "dist_ratio": ratio,
                    "mag_bin": magnitude_bin(step.magnitude),
                    "dist_bin": distance_bin(step.distance_km),
                    "ratio_bin": ratio_bin(step.distance_km,
                                           step.threshold_km),
                    "reachable": reach[j],
                    "detected": delay is not None,
                    "delay_days": (
                        None if delay is None
                        else round(delay * config.time_unit_days, 2)
                    ),
                })
    return summary, step_rows


def isd_filter_station(
    csv_path: Any, *, field: str = "TMP"
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter one station-year's hourly series and explain its flags.

    A flag is explained by the latest quality failure or coordinate change
    within :attr:`ISD_CONFIG.horizon_days` before it (one detector
    stride); a coordinate change wins over a quality failure at the same
    lag. Returns ``(summary_rows, flag_rows)``.

    The summary counts events, explained flags and unexplained flags
    under those names: a quality-failed row is not a step and an
    explained flag is not a detection, so reusing the NGL names would
    invite a reader to compute a recall that means nothing. A real
    station-year has hundreds of quality failures.
    ``event_window_share`` gives the share of the year covered by the
    explanation windows, which is what an unexplained-flag count has to
    be read against.
    """
    t, y, info = isd.station_year(csv_path, field)
    out = run_filter(t, y, ISD_CONFIG)
    dropped = isd.dropped_times(csv_path, field)
    moves = isd.coordinate_changes(csv_path, field)
    events = (
        [("move", float(m["t"])) for m in moves]
        + [("quality", float(v)) for v in dropped["quality"]]
    )
    flag_rows: list[dict[str, Any]] = []
    explained = 0
    for flag in out["flags"]:
        best: tuple[str, float, float] | None = None
        for kind, when in events:
            lag = float(flag) - when
            if 0.0 <= lag <= ISD_CONFIG.horizon_days:
                if best is None or lag < best[2] or (
                    lag == best[2] and kind == "move"
                ):
                    best = (kind, when, lag)
        if best is not None:
            explained += 1
        flag_rows.append({
            "station": info["station"], "field": field,
            "config": ISD_CONFIG.key, "flag_day": float(flag),
            "explained_by": "none" if best is None else best[0],
            "event_day": None if best is None else best[1],
            "lag_days": None if best is None else round(best[2], 4),
        })
    n_flags = int(out["flags"].size)
    share = event_window_share(
        [when for _kind, when in events],
        t0=float(t[0]) if t.size else 0.0,
        t1=float(t[-1]) if t.size else 0.0,
        before=0.0, after=ISD_CONFIG.horizon_days,
    )
    summary = [{
        "station": info["station"], "component": field,
        "config": ISD_CONFIG.key, "n_updates": out["n_updates"],
        "station_years": round(info["n"] / (24.0 * 365.25), 4),
        "seconds": round(out["seconds"], 4),
        "us_per_update": round(out["us_per_update"], 2),
        "n_flags": n_flags,
        "n_events": len(events),
        "n_flags_explained": explained,
        "n_flags_unexplained": n_flags - explained,
        "explained_share": (
            round(explained / n_flags, 4) if n_flags else None
        ),
        "event_window_share": round(share, 4),
        "v": out["params"].get("v"), "c": out["params"].get("c"),
        "n_moves": len(moves),
        "n_quality_failures": int(dropped["quality"].size),
    }]
    return summary, flag_rows


def _ngl_one(
    args: tuple[str, list[Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path, steps = args
    try:
        return ngl_filter_station(path, steps)
    except Exception as exc:                     # keep the batch alive
        return ([{"station": Path(path).stem, "component": "",
                  "config": f"ERROR: {type(exc).__name__}: {exc}"}], [])


def _isd_one(
    path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        return isd_filter_station(path)
    except Exception as exc:                     # keep the batch alive
        return ([{"station": Path(path).stem, "component": "",
                  "config": f"ERROR: {type(exc).__name__}: {exc}"}], [])


def _fan_out_pairs(
    func: Any, jobs: Sequence[Any], workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if workers <= 1:
        results = [func(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(func, jobs, chunksize=4))
    first = [row for group, _ in results for row in group]
    second = [row for _, group in results for row in group]
    return first, second


def run_ngl_filters(
    jobs: Sequence[tuple[Any, Sequence[Any]]], *, workers: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """:func:`ngl_filter_station` over many ``(path, steps)`` jobs."""
    return _fan_out_pairs(
        _ngl_one, [(str(p), list(s)) for p, s in jobs], workers
    )


def run_isd_filters(
    paths: Sequence[Any], *, workers: int = 1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """:func:`isd_filter_station` over many station-year files."""
    return _fan_out_pairs(_isd_one, [str(p) for p in paths], workers)
