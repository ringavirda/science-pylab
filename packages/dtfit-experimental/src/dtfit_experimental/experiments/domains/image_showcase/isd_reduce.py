"""The NOAA reductions: one image per station-year and field (the annual
model's carrier), the optional per-day block images, and the channel
batch that projects many stations' shared day grid through one GEMM.

The station-year image carries the annual and semiannual cycle only: a
Legendre image resolves a cycle at about eight coefficients per period, so
365 diurnal periods would need thousands. The diurnal cycle lives in the
one-day images instead, where one period needs order 16.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dtfit.image import Image, ImageStream

from . import isd
from .store import image_nbytes, save_images

ANNUAL_OMEGA = 2.0 * np.pi / 365.25
ANNUAL_EXPR = (
    "c + v*t"
    f" + a1*cos({ANNUAL_OMEGA!r}*t) + b1*sin({ANNUAL_OMEGA!r}*t)"
    f" + a2*cos({2.0 * ANNUAL_OMEGA!r}*t)"
    f" + b2*sin({2.0 * ANNUAL_OMEGA!r}*t)"
)
ANNUAL_NAMES = ["a1", "a2", "b1", "b2", "c", "v"]
ANNUAL_ORDER = 24

DIURNAL_EXPR = "c + a*cos(2*pi*t) + b*sin(2*pi*t)"
DIURNAL_NAMES = ["a", "b", "c"]
DIURNAL_ORDER = 16
DAY_POSITIONS = np.arange(24, dtype=float) / 24.0

ISD_REDUCE_COLUMNS = [
    "station", "year", "field", "days", "n", "dropped_quality",
    "dropped_missing", "dropped_repeat", "n_days", "dropped_days",
    "coef_bytes", "gram_bytes", "grid_bytes", "raw_bytes", "npz_bytes",
    "seconds", "error",
]


def annual_design(t: np.ndarray) -> np.ndarray:
    """The annual model's columns at ``t`` (days from the year's start),
    ordered like :data:`ANNUAL_NAMES`."""
    t = np.asarray(t, dtype=float)
    w = ANNUAL_OMEGA
    return np.column_stack([
        np.cos(w * t), np.cos(2.0 * w * t),
        np.sin(w * t), np.sin(2.0 * w * t),
        np.ones_like(t), t,
    ])


def diurnal_design(t: np.ndarray) -> np.ndarray:
    """The diurnal model's columns at ``t`` (days), ordered like
    :data:`DIURNAL_NAMES`."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t), np.ones_like(t),
    ])


@dataclass
class YearReduction:
    """One station-year's images and facts.

    ``images`` keys are ``year_<FIELD>`` and, with ``with_days``,
    ``day<NNN>_<FIELD>`` for the days that filled at least
    ``DIURNAL_ORDER + 2`` samples.
    """

    station: str
    year: int
    images: dict[str, Image] = dc_field(default_factory=dict)
    info: dict[str, Any] = dc_field(default_factory=dict)


def reduce_station_year(
    path: Any,
    fields: Sequence[str] = ("TMP",),
    *,
    chunk: int = 100_000,
    with_days: bool = False,
) -> YearReduction:
    """Reduce one station-year CSV to its images.

    Args:
        path: The station-year file.
        fields: ``"TMP"``, ``"SLP"`` or both. Each field is a separate
            pass over the file: the two have different quality and
            missing rows, so they cannot share a sample set.
        chunk: Rows per read block, at most 100,000.
        with_days: Also build the one-day block images of the first
            field. They cost about 2.6 kB a day, so the full run enables
            them only for the stations that have published normals.

    Returns:
        A :class:`YearReduction`.

    Raises:
        ValueError: the file has no data row, or a field keeps fewer than
            ``ANNUAL_ORDER + 2`` samples.
    """
    path = Path(path)
    station, year = isd.station_header(path)
    if not year:
        raise ValueError(f"{path.name}: no data row")
    days = float(isd.days_in_year(year))
    images: dict[str, Image] = {}
    info: dict[str, Any] = {
        "station": station, "year": year, "days": int(days),
        "fields": list(fields), "n": {}, "dropped_quality": {},
        "dropped_missing": {}, "dropped_repeat": {},
        "n_days": 0, "dropped_days": 0,
    }
    for k, name in enumerate(fields):
        stream = ImageStream(
            "legendre", ANNUAL_ORDER, domain=(0.0, days), grid="explicit"
        )
        blocks = None
        if with_days and k == 0:
            blocks = ImageStream(
                "legendre", DIURNAL_ORDER, domain=(0.0, days), block=1.0,
                detect="previous", grid="explicit",
                keep_fine=int(days) + 1, fold=int(days) + 1,
            )
        dq = dm = dr = 0
        n = 0
        for c in isd.read_isd(path, name, chunk=chunk):
            stream.update(c.t, c.y)
            if blocks is not None:
                blocks.update(c.t, c.y)
            dq += c.dropped_quality
            dm += c.dropped_missing
            dr += c.dropped_repeat
            n += int(c.t.size)
        if n < ANNUAL_ORDER + 2:
            raise ValueError(
                f"{path.name}: {name} kept {n} rows, need "
                f"{ANNUAL_ORDER + 2}"
            )
        images[f"year_{name}"] = stream.image(0)
        info["n"][name] = n
        info["dropped_quality"][name] = dq
        info["dropped_missing"][name] = dm
        info["dropped_repeat"][name] = dr
        if blocks is not None:
            blocks.close()
            found = blocks.blocks(0.0, days)
            for img in found:
                day = int(round(img.domain[0]))
                images[f"day{day:03d}_{name}"] = img
            info["n_days"] = len(found)
            info["dropped_days"] = int(blocks.dropped_)
            info["day_flags"] = [
                [float(i), float(d0), float(d1)]
                for i, (d0, d1) in blocks.flags_
            ]
    return YearReduction(
        station=station, year=year, images=images, info=info
    )


def reduce_year_to_file(
    path: Any, out_dir: Any, fields: Sequence[str] = ("TMP",),
    **kwargs: Any,
) -> tuple[Path, YearReduction]:
    """Reduce one station-year and write
    ``<out_dir>/<STATION>.npz``."""
    red = reduce_station_year(path, fields, **kwargs)
    out = save_images(
        Path(out_dir) / f"{red.station}.npz", red.images, red.info
    )
    return out, red


def _reduce_year_one(
    args: tuple[str, str, list[str], dict[str, Any]]
) -> dict[str, Any]:
    """Worker: reduce one station-year to a row; a failure is reported in
    ``error`` rather than raised."""
    path_s, out_s, fields, kwargs = args
    path = Path(path_s)
    row: dict[str, Any] = {c: "" for c in ISD_REDUCE_COLUMNS}
    row["station"] = path.stem
    row["field"] = ",".join(fields)
    row["raw_bytes"] = path.stat().st_size
    started = time.perf_counter()
    try:
        npz, red = reduce_year_to_file(path, out_s, fields, **kwargs)
    except Exception as exc:                     # keep the batch alive
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    coef = gram = grid = 0
    for img in red.images.values():
        c, g, x = image_nbytes(img)
        coef += c
        gram += g
        grid += x
    first = red.info["fields"][0]
    row.update({
        "station": red.station, "year": red.year,
        "days": red.info["days"], "n": red.info["n"][first],
        "dropped_quality": red.info["dropped_quality"][first],
        "dropped_missing": red.info["dropped_missing"][first],
        "dropped_repeat": red.info["dropped_repeat"][first],
        "n_days": red.info["n_days"],
        "dropped_days": red.info["dropped_days"],
        "coef_bytes": coef, "gram_bytes": gram, "grid_bytes": grid,
        "npz_bytes": npz.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3), "error": "",
    })
    return row


def reduce_many_years(
    paths: Sequence[Any],
    out_dir: Any,
    *,
    fields: Sequence[str] = ("TMP",),
    workers: int = 1,
    with_days: bool = False,
) -> list[dict[str, Any]]:
    """Reduce many station-years, optionally across processes; one row per
    station in input order, with the columns of
    :data:`ISD_REDUCE_COLUMNS` and a message in ``error`` for a station
    that failed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(p), str(out_dir), list(fields), {"with_days": with_days})
        for p in paths
    ]
    if workers <= 1:
        return [_reduce_year_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_reduce_year_one, jobs, chunksize=8))


def project_day_grids(
    columns: Sequence[np.ndarray],
    *,
    order: int = DIURNAL_ORDER,
    backend: str = "numpy",
) -> tuple[list[Image], float]:
    """Project 24-value day grids through one channel-form GEMM.

    ``columns`` are complete day grids on the nominal positions
    :data:`DAY_POSITIONS`, so the projection is the single
    ``S = Phi^T Y`` with ``Y`` of shape ``(24, len(columns))``: the form
    the GPU row measures and the exactness gate checks. ``backend`` is
    ``"numpy"``, ``"cupy"`` or ``"torch"`` and goes straight to
    :class:`~dtfit.image.ImageStream`.

    Returns:
        ``(images, seconds)`` -- one image per column, in order, and the
        projection's wall time. Empty ``columns`` gives ``([], 0.0)``.
    """
    if not columns:
        return [], 0.0
    Y = np.column_stack(list(columns))
    stream = ImageStream(
        "legendre", int(order), domain=(0.0, 1.0), grid="explicit",
        channels=Y.shape[1], backend=backend,
    )
    started = time.perf_counter()
    stream.update(DAY_POSITIONS, Y)
    images = stream.images()
    return images, time.perf_counter() - started


def day_batch(
    paths: Sequence[Any],
    day: int,
    field: str = "TMP",
    *,
    order: int = DIURNAL_ORDER,
    backend: str = "numpy",
) -> tuple[list[str], list[Image], dict[str, Any]]:
    """Image one day of every station that filled all 24 hour bins, in one
    channel batch.

    The qualifying stations share the nominal positions
    :data:`DAY_POSITIONS`, so the projection is the single GEMM
    ``S = Phi^T Y`` with ``Y`` of shape ``(24, B)``: the form the GPU row
    measures. ``backend`` is ``"numpy"``, ``"cupy"`` or ``"torch"`` and
    goes straight to :class:`~dtfit.image.ImageStream`.

    Returns:
        ``(station_ids, images, info)``; ``info`` carries ``day``,
        ``n_read`` (files opened), ``n_stations`` (files that qualified),
        ``backend``, ``seconds_read`` and ``seconds_project``. With no
        qualifying station the lists are empty and ``images`` is ``[]``.
    """
    files = [Path(p) for p in paths]
    ids: list[str] = []
    columns: list[np.ndarray] = []
    started = time.perf_counter()
    for p in files:
        grid = isd.day_grid(p, day, field)
        if grid is not None:
            ids.append(p.stem)
            columns.append(grid)
    read_seconds = time.perf_counter() - started
    info = {
        "day": int(day), "field": field, "n_read": len(files),
        "n_stations": len(ids), "backend": backend,
        "seconds_read": round(read_seconds, 4), "seconds_project": 0.0,
        "samples": 24 * len(ids),
    }
    if not ids:
        return [], [], info
    images, seconds = project_day_grids(
        columns, order=order, backend=backend
    )
    info["seconds_project"] = round(seconds, 6)
    return ids, images, info
