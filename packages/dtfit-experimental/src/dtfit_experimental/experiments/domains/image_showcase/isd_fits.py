"""Fits from the stored NOAA images: the annual cycle from the
station-year image, the diurnal cycle from the one-day images, the
exactness against the raw least squares, and the check against NOAA's
published hourly normals.

Only :func:`exactness_year` reads a station CSV; everything else works
from the ``.npz`` images, so the Pi runs it on what it received.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dtfit.image import coverage

from . import isd
from .compare import (
    COVERAGE_TOL, EXACTNESS_TOL, fit_from_image, param_score, raw_lstsq,
    worst_param,
)
from .isd_reduce import (
    ANNUAL_EXPR, ANNUAL_NAMES, ANNUAL_OMEGA, DAY_POSITIONS, DIURNAL_EXPR,
    DIURNAL_NAMES, annual_design, diurnal_design, project_day_grids,
)
from .store import load_images

# The model's annual period in days; the phases are reported against it,
# not against the calendar year's 365 or 366 days.
PERIOD = 2.0 * np.pi / ANNUAL_OMEGA
# How many stored day images exactness_year gates against their raw rows.
DAY_GATE_SAMPLE = 8

_LENGTHS = {
    False: (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
    True: (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),
}

YEAR_COLUMNS = [
    "station", "year", "field", "n", "order", "days",
    *ANNUAL_NAMES, "annual_amp", "annual_phase_day", "semi_amp",
    "semi_phase_day", "rss", "bic",
]
DAY_COLUMNS = [
    "station", "year", "field", "day", "month", "n", "diurnal_amp",
    "diurnal_range", "diurnal_peak_hour", "mean", "rss",
]
NORMALS_ROW_COLUMNS = [
    "station", "normals_id", "kind", "month", "amp_image", "amp_normals",
    "phase_image_day", "phase_normals_day", "phase_diff_days",
    "range_image", "range_normals", "n_days",
]
EXACT_ISD_COLUMNS = [
    "station", "year", "field", "n", "order", "score", "coverage",
    "day_score", "n_days_checked", "gate", "worst_param", "amp_image",
    "amp_raw",
]


def amp_phase(
    cos_coef: float, sin_coef: float, period: float
) -> tuple[float, float]:
    """Amplitude and peak position of ``a cos(w t) + b sin(w t)``.

    Returns ``(hypot(a, b), phase)`` with ``phase`` in the same units as
    ``period``, in ``[0, period)``: the position of the first maximum. A
    zero pair gives ``(0.0, 0.0)``.
    """
    a, b = float(cos_coef), float(sin_coef)
    amp = float(np.hypot(a, b))
    if amp == 0.0:
        return 0.0, 0.0
    phase = float(np.arctan2(b, a)) / (2.0 * np.pi) * float(period)
    return amp, phase % float(period)


def month_of_day(day: int, year: int) -> tuple[int, int]:
    """``(month, day_of_month)`` of a day index counted from 0 on
    1 January. A day index past the year's end is clamped to 31
    December."""
    lengths = _LENGTHS[isd.days_in_year(year) == 366]
    remaining = int(day)
    for month, length in enumerate(lengths, start=1):
        if remaining < length:
            return month, remaining + 1
        remaining -= length
    return 12, 31


def year_rows(
    npz_path: Any, fields: Sequence[str] = ("TMP",)
) -> list[dict[str, Any]]:
    """Fit the annual model to each stored ``year_<FIELD>`` image.

    One row per field, with the parameters, the annual and semiannual
    amplitude and peak day, and the fit's ``rss`` and ``bic``. A field
    with no stored image is skipped.
    """
    images, info = load_images(npz_path)
    station = str(info.get("station", Path(npz_path).stem))
    year = int(info.get("year", 0))
    days = float(info.get("days", isd.days_in_year(year) if year else 365))
    out: list[dict[str, Any]] = []
    for name in fields:
        key = f"year_{name}"
        if key not in images:
            continue
        image = images[key]
        res = fit_from_image(ANNUAL_EXPR, image, ANNUAL_NAMES)
        p = res.params
        amp1, phase1 = amp_phase(p["a1"], p["b1"], PERIOD)
        amp2, phase2 = amp_phase(p["a2"], p["b2"], PERIOD / 2.0)
        row = {
            "station": station, "year": year, "field": name,
            "n": int(image.n), "order": int(image.order),
            "days": int(days),
            "annual_amp": amp1, "annual_phase_day": phase1,
            "semi_amp": amp2, "semi_phase_day": phase2,
            "rss": None if res.rss is None else float(res.rss),
            "bic": None if res.bic is None else float(res.bic),
        }
        for k in ANNUAL_NAMES:
            row[k] = float(p[k])
        out.append(row)
    return out


def day_rows(npz_path: Any, field: str = "TMP") -> list[dict[str, Any]]:
    """Fit the diurnal model to every stored ``day<NNN>_<FIELD>`` image.

    One row per day with the first-harmonic amplitude (degrees Celsius),
    the range of the day image's reconstruction at its own sample
    positions, the peak hour and the day's mean. The range is the
    statistic the published normals' own diurnal range is compared
    against: a real day is not a pure sinusoid and its range exceeds
    twice the first harmonic by 10 to 25 percent, so comparing
    ``2 * amplitude`` against a max-minus-min would be biased low by
    construction. Returns an empty list for a station reduced without
    ``with_days``.
    """
    images, info = load_images(npz_path)
    station = str(info.get("station", Path(npz_path).stem))
    year = int(info.get("year", 0))
    out: list[dict[str, Any]] = []
    for key, image in images.items():
        if not key.startswith("day") or not key.endswith(f"_{field}"):
            continue
        day = int(key[3:key.index("_")])
        res = fit_from_image(
            DIURNAL_EXPR, image, DIURNAL_NAMES, slope=None
        )
        p = res.params
        amp, peak = amp_phase(p["a"], p["b"], 24.0)
        # The reconstruction is taken at the image's own sample
        # positions, never on a nominal hourly lattice: a station
        # reporting at :53 has no sample at the domain's left edge, and a
        # Legendre reconstruction extrapolated there diverges.
        shape = image.reconstruct(image.grid.positions())
        month, _ = month_of_day(day, year) if year else (0, 0)
        out.append({
            "station": station, "year": year, "field": field, "day": day,
            "month": month, "n": int(image.n), "diurnal_amp": amp,
            "diurnal_range": float(np.max(shape) - np.min(shape)),
            "diurnal_peak_hour": peak, "mean": float(p["c"]),
            "rss": None if res.rss is None else float(res.rss),
        })
    out.sort(key=lambda r: r["day"])
    return out


def normals_annual(
    normals: dict[str, np.ndarray], year: int
) -> dict[str, float]:
    """The annual model fitted to the normals' daily means.

    The 24 hourly normals of each day are averaged, the resulting daily
    series is fitted by least squares with :func:`annual_design` at
    ``t = day + 0.5``, and the amplitude and peak day of the annual and
    semiannual terms are returned along with the mean. Days with no finite
    normal are dropped.

    The published normals are a 365-day climatology with no 29 February,
    so each ``(month, day)`` is mapped through ``year``'s own calendar:
    against a leap year that shifts every date after February by one day,
    which is exactly the offset the phase comparison would otherwise
    carry.
    """
    day_index: dict[tuple[int, int], list[float]] = {}
    for m, d, temp in zip(normals["month"], normals["day"],
                          normals["temp_c"]):
        if np.isfinite(temp):
            day_index.setdefault((int(m), int(d)), []).append(float(temp))
    keys = sorted(day_index)
    t = np.array(
        [_day_number(m, d, year) + 0.5 for m, d in keys], dtype=float
    )
    y = np.array([float(np.mean(day_index[k])) for k in keys])
    ref = raw_lstsq(annual_design(t), y, ANNUAL_NAMES)
    amp1, phase1 = amp_phase(ref["a1"], ref["b1"], PERIOD)
    amp2, phase2 = amp_phase(ref["a2"], ref["b2"], PERIOD / 2.0)
    return {
        "amp": amp1, "phase_day": phase1, "semi_amp": amp2,
        "semi_phase_day": phase2, "mean": ref["c"], "n_days": len(keys),
    }


def _day_number(month: int, day: int, year: int) -> int:
    """The 0-based day index of a (month, day) in ``year``'s calendar."""
    lengths = _LENGTHS[isd.days_in_year(year) == 366]
    return sum(lengths[: month - 1]) + day - 1


def normals_monthly_range(
    normals: dict[str, np.ndarray]
) -> dict[int, float]:
    """The mean daily temperature range (max minus min over the 24 hourly
    normals) of each month, in degrees Celsius. A month with no finite day
    is absent."""
    per_day: dict[tuple[int, int], list[float]] = {}
    for m, d, temp in zip(normals["month"], normals["day"],
                          normals["temp_c"]):
        if np.isfinite(temp):
            per_day.setdefault((int(m), int(d)), []).append(float(temp))
    by_month: dict[int, list[float]] = {}
    for (m, _d), values in per_day.items():
        if len(values) >= 2:
            by_month.setdefault(m, []).append(max(values) - min(values))
    return {m: float(np.mean(v)) for m, v in sorted(by_month.items())}


def normals_rows(
    npz_path: Any, normals_path: Any, *, field: str = "TMP"
) -> list[dict[str, Any]]:
    """Compare one station's image fits against its published normals.

    Returns one ``kind="annual"`` row (amplitude and peak day, image
    against normals) and one ``kind="diurnal"`` row per month present in
    both: the month's median reconstructed daily range from the day
    images against the normals' mean daily range, the same statistic on
    both sides, with the first-harmonic amplitude kept beside it in
    ``amp_image``. A station with no day images contributes the annual
    row alone.
    """
    normals = isd.read_normals(normals_path)
    ranges = normals_monthly_range(normals)
    years = year_rows(npz_path, (field,))
    if not years:
        return []
    y = years[0]
    ann = normals_annual(normals, int(y["year"]))
    diff = y["annual_phase_day"] - ann["phase_day"]
    diff = (diff + PERIOD / 2.0) % PERIOD - PERIOD / 2.0
    out = [{
        "station": y["station"], "normals_id": Path(normals_path).stem,
        "kind": "annual", "month": 0,
        "amp_image": y["annual_amp"], "amp_normals": ann["amp"],
        "phase_image_day": y["annual_phase_day"],
        "phase_normals_day": ann["phase_day"],
        "phase_diff_days": float(diff),
        "range_image": None, "range_normals": None,
        "n_days": ann["n_days"],
    }]
    per_day = day_rows(npz_path, field)
    by_month: dict[int, list[tuple[float, float]]] = {}
    for row in per_day:
        by_month.setdefault(int(row["month"]), []).append(
            (float(row["diurnal_amp"]), float(row["diurnal_range"]))
        )
    for month in sorted(set(by_month) & set(ranges)):
        amps = np.array([a for a, _r in by_month[month]])
        spans = np.array([r for _a, r in by_month[month]])
        out.append({
            "station": y["station"],
            "normals_id": Path(normals_path).stem,
            "kind": "diurnal", "month": month,
            "amp_image": float(np.median(amps)), "amp_normals": None,
            "phase_image_day": None, "phase_normals_day": None,
            "phase_diff_days": None,
            "range_image": float(np.median(spans)),
            "range_normals": ranges[month], "n_days": int(amps.size),
        })
    return out


def _day_gate(
    csv_path: Any, field: str, day_sample: int, n_days: int
) -> tuple[float | None, int]:
    """The worst exactness score of the channel form's day images against
    ``lstsq`` on the same 24 hourly rows, and how many days were checked.

    The days are ``day_sample`` evenly spaced across the year. Each one
    is binned to the nominal positions :data:`~.isd_reduce.DAY_POSITIONS`
    and projected through :func:`~.isd_reduce.project_day_grids`, which
    is the batch the GPU row measures, and compared against the least
    squares of the same model on the same 24 values at the same
    positions. The reducer's own ``day<NNN>`` block images are a
    different statistic -- they sit on the rows' true minute offsets, so
    a station reporting at :53 has no sample at a nominal hour -- and are
    not what this gates.

    Costs one extra streaming pass over the station-year CSV, which is
    why the gate samples days rather than taking all of them.

    Returns ``(None, 0)`` when no sampled day filled its 24 bins.
    """
    if day_sample <= 0 or n_days <= 0:
        return None, 0
    step = max(1, int(n_days) // int(day_sample))
    days = list(range(0, int(n_days), step))[:int(day_sample)]
    grids = isd.day_grids(csv_path, days, field)
    usable = [(d, grids[d]) for d in days if grids.get(d) is not None]
    if not usable:
        return None, 0
    images, _seconds = project_day_grids([g for _d, g in usable])
    design = diurnal_design(DAY_POSITIONS)
    worst = 0.0
    for (_day, grid), image in zip(usable, images):
        ref = raw_lstsq(design, grid, DIURNAL_NAMES)
        res = fit_from_image(
            DIURNAL_EXPR, image, DIURNAL_NAMES, slope=None
        )
        worst = max(worst, param_score(res.params, ref))
    return worst, len(usable)


def exactness_year(
    csv_path: Any,
    npz_path: Any,
    field: str = "TMP",
    *,
    day_sample: int = DAY_GATE_SAMPLE,
) -> dict[str, Any]:
    """The gate for NOAA: the station-year image fit against the raw least
    squares on the same rows, plus a sample of the day images against
    theirs. Reads the CSV again, so it runs where the samples are.

    ``gate`` is ``"UNDERSAMPLED"`` when :func:`dtfit.image.coverage` on
    the station-year image exceeds :data:`compare.COVERAGE_TOL` (a
    station-year that kept too few rows for order 24: reported and
    counted, never a failure of the run), ``"FAIL"`` when either the
    annual score or the worst day score misses
    :data:`compare.EXACTNESS_TOL`, and ``"ok"`` otherwise.
    ``day_score`` is empty and ``n_days_checked`` zero when no sampled
    day filled its 24 hour bins.
    """
    t, y, info = isd.station_year(csv_path, field)
    images, meta = load_images(npz_path)
    image = images[f"year_{field}"]
    ref = raw_lstsq(annual_design(t), y, ANNUAL_NAMES)
    res = fit_from_image(ANNUAL_EXPR, image, ANNUAL_NAMES)
    got = res.params
    score = param_score(got, ref)
    cover = float(coverage(
        ANNUAL_EXPR, [float(got[k]) for k in ANNUAL_NAMES], image,
        var="t",
    ))
    day_score, checked = _day_gate(
        csv_path, field, day_sample, int(info["days"])
    )
    missed = score > EXACTNESS_TOL or (
        day_score is not None and day_score > EXACTNESS_TOL
    )
    if cover > COVERAGE_TOL:
        gate = "UNDERSAMPLED"
    elif missed:
        gate = "FAIL"
    else:
        gate = "ok"
    return {
        "station": info["station"], "year": info["year"], "field": field,
        "n": int(image.n), "order": int(image.order),
        "score": float(score), "coverage": cover,
        "day_score": day_score, "n_days_checked": checked,
        "gate": gate, "worst_param": worst_param(got, ref),
        "amp_image": amp_phase(got["a1"], got["b1"], PERIOD)[0],
        "amp_raw": amp_phase(ref["a1"], ref["b1"], PERIOD)[0],
    }


def _year_one(args: tuple[str, list[str]]) -> list[dict[str, Any]]:
    npz_path, fields = args
    return year_rows(npz_path, fields)


def _normals_one(args: tuple[str, str]) -> list[dict[str, Any]]:
    npz_path, normals_path = args
    return normals_rows(npz_path, normals_path)


def _exact_isd_one(args: tuple[str, str]) -> list[dict[str, Any]]:
    csv_path, npz_path = args
    try:
        return [exactness_year(csv_path, npz_path)]
    except Exception as exc:                     # keep the batch alive
        return [{"station": Path(npz_path).stem, "year": 0, "field": "",
                 "n": 0, "order": 0, "score": float("inf"),
                 "coverage": None, "day_score": None,
                 "n_days_checked": 0,
                 "gate": f"ERROR: {type(exc).__name__}: {exc}",
                 "worst_param": "", "amp_image": None, "amp_raw": None}]


def _fan_out(func: Any, jobs: Sequence[Any], workers: int) -> list[Any]:
    if workers <= 1:
        results = [func(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(func, jobs, chunksize=8))
    return [row for group in results for row in group]


def run_year_fits(
    npz_paths: Sequence[Any], *, fields: Sequence[str] = ("TMP",),
    workers: int = 1,
) -> list[dict[str, Any]]:
    """:func:`year_rows` over many stations, in input order."""
    return _fan_out(
        _year_one, [(str(p), list(fields)) for p in npz_paths], workers
    )


def run_normals(
    pairs: Sequence[tuple[Any, Any]], *, workers: int = 1
) -> list[dict[str, Any]]:
    """:func:`normals_rows` over many ``(npz, normals csv)`` pairs."""
    return _fan_out(
        _normals_one, [(str(a), str(b)) for a, b in pairs], workers
    )


def run_exactness_isd(
    pairs: Sequence[tuple[Any, Any]], *, workers: int = 1
) -> list[dict[str, Any]]:
    """:func:`exactness_year` over many ``(csv, npz)`` pairs; a station
    that raises contributes a row whose ``gate`` starts with
    ``ERROR``."""
    return _fan_out(
        _exact_isd_one, [(str(a), str(b)) for a, b in pairs], workers
    )
