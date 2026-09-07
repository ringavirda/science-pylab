"""Reader for the NOAA NCEI Integrated Surface Database's Global Hourly
CSV (one file per station-year), and the hourly normals the annual and
diurnal amplitudes are checked against.

The reader streams with the ``csv`` module: a year directory is 49 GB and
a station-year up to a few tens of thousands of rows, so nothing here ever
holds more than one chunk. Values are tenths with a quality code; codes
0, 1, 4, 5 and 9 are kept and 2, 3, 6 and 7 dropped, each drop counted.
"""

from __future__ import annotations

import csv
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

# field name -> (the missing sentinel as the file writes it, the unit
# the reader returns after dividing the integer by ten)
ISD_FIELDS: dict[str, tuple[int, str]] = {
    "TMP": (9999, "degC"),
    "SLP": (99999, "hPa"),
}
GOOD_QUALITY = frozenset("01459")
NORMALS_BASE = (
    "https://www.ncei.noaa.gov/data/normals-hourly/1991-2020/access/"
)
NORMALS_COLUMNS = (
    "STATION", "month", "day", "hour", "HLY-TEMP-NORMAL",
    "HLY-PRES-NORMAL",
)

_CUM = {
    False: (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334),
    True: (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335),
}
_HREF = re.compile(r'href="([A-Z]{2}W[0-9]{8})\.csv"')


def days_in_year(year: int) -> int:
    """366 in a leap year, else 365."""
    y = int(year)
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    return 366 if leap else 365


def iso_days(stamp: str, leap: bool) -> float:
    """Days from the start of the stamp's year, from an ISD ``DATE``
    such as ``2024-01-02T06:00:00`` (UTC). ``leap`` says whether that year
    has 366 days.

    Raises:
        ValueError: the stamp is shorter than ``YYYY-MM-DDTHH:MM:SS``.
    """
    if len(stamp) < 19:
        raise ValueError(f"not an ISD timestamp: {stamp!r}")
    month = int(stamp[5:7])
    day = int(stamp[8:10])
    secs = (int(stamp[11:13]) * 3600 + int(stamp[14:16]) * 60
            + int(stamp[17:19]))
    return (_CUM[leap][month - 1] + day - 1) + secs / 86400.0


def parse_field(raw: str, missing: int) -> tuple[float | None, str]:
    """One ``"value,quality"`` field as ``(value, quality)``.

    The value is the integer part divided by ten (tenths of a degree
    Celsius, tenths of a hectopascal); ``None`` when the field is empty,
    unparsable, or equal to ``missing``. The quality code is returned even
    for a missing value, and is ``""`` when the field carries none.
    """
    parts = str(raw).split(",")
    if len(parts) < 2 or not parts[0]:
        return None, ""
    try:
        value = int(parts[0])
    except ValueError:
        return None, ""
    if abs(value) == abs(int(missing)):
        return None, parts[1]
    return value / 10.0, parts[1]


@dataclass
class IsdChunk:
    """One block of a station-year's kept rows.

    ``t`` is days from the year's start (UTC), ``y`` the field in degrees
    Celsius (TMP) or hectopascals (SLP), and ``lat``, ``lon``, ``elev``
    the coordinates the row carried. The three ``dropped_*`` counts cover
    this block only.
    """

    t: np.ndarray
    y: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    elev: np.ndarray
    dropped_quality: int = 0
    dropped_missing: int = 0
    dropped_repeat: int = 0


def _float(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def read_isd(
    path: Any, field: str = "TMP", *, chunk: int = 100_000
) -> Iterator[IsdChunk]:
    """Stream one station-year CSV as :class:`IsdChunk` blocks.

    Rows are kept in file order and the first usable row of a timestamp
    wins: a timestamp at or before the last kept one is dropped
    (``dropped_repeat``), so a duplicate whose earlier copies all failed
    the filters can still contribute. A row failing the quality filter
    (``dropped_quality``) or carrying the missing sentinel
    (``dropped_missing``) is dropped without advancing the timestamp. A
    block whose rows are all dropped is skipped rather than yielded
    empty, and its counts carry into the next block that keeps a row; if
    the file ends on such a run with no later block to carry them, a
    final zero-length chunk yields the residual counts instead, so a
    station-year's drop counts are never lost.

    Raises:
        ValueError: ``field`` is not in :data:`ISD_FIELDS`, or the file's
            header carries neither ``DATE`` nor the field.
    """
    if field not in ISD_FIELDS:
        raise ValueError(
            f"field must be one of {sorted(ISD_FIELDS)}, got {field!r}"
        )
    missing = ISD_FIELDS[field][0]
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return
        try:
            i_date = header.index("DATE")
            i_val = header.index(field)
            i_lat = header.index("LATITUDE")
            i_lon = header.index("LONGITUDE")
            i_elev = header.index("ELEVATION")
        except ValueError as exc:
            raise ValueError(f"{path}: unexpected header") from exc
        leap = None
        last_t = -1.0
        t: list[float] = []
        y: list[float] = []
        la: list[float] = []
        lo: list[float] = []
        el: list[float] = []
        counts = [0, 0, 0]

        def flush() -> IsdChunk | None:
            """The buffered rows as a chunk, or None when this block kept
            nothing; a None result leaves ``counts`` untouched so a later
            call's chunk carries them, and the caller covers the case
            where no later call ever yields."""
            if not t:
                return None
            out = IsdChunk(
                np.array(t), np.array(y), np.array(la), np.array(lo),
                np.array(el), *counts,
            )
            del t[:]
            del y[:]
            del la[:]
            del lo[:]
            del el[:]
            counts[:] = [0, 0, 0]
            return out

        for row in reader:
            if len(row) <= i_val:
                continue
            stamp = row[i_date]
            if leap is None:
                leap = days_in_year(int(stamp[:4])) == 366
            when = iso_days(stamp, leap)
            if when <= last_t:
                counts[2] += 1
                continue
            value, quality = parse_field(row[i_val], missing)
            if quality and quality not in GOOD_QUALITY:
                counts[0] += 1
                continue
            if value is None:
                counts[1] += 1
                continue
            last_t = when
            t.append(when)
            y.append(value)
            la.append(_float(row[i_lat]))
            lo.append(_float(row[i_lon]))
            el.append(_float(row[i_elev]))
            if len(t) >= chunk:
                out = flush()
                if out is not None:
                    yield out
        out = flush()
        if out is not None:
            yield out
        elif any(counts):
            yield IsdChunk(
                np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros(0), *counts,
            )


def station_header(path: Any) -> tuple[str, int]:
    """The station id and the year of a station-year file, read from its
    first data row; ``("", 0)`` for a file with no data row."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        first = next(reader, None)
        if header is None or first is None:
            return "", 0
        return (first[header.index("STATION")],
                int(first[header.index("DATE")][:4]))


def station_year(
    path: Any, field: str = "TMP", *, chunk: int = 100_000
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One station-year as ``(t, y, info)``.

    The arrays are the whole kept series (at most a few tens of thousands
    of rows for one station-year, the bound this function relies on);
    ``info`` carries ``station``, ``year``, ``days`` (365 or 366), ``n``
    and the three drop counts.
    """
    path = Path(path)
    ts: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dq = dm = dr = 0
    for c in read_isd(path, field, chunk=chunk):
        ts.append(c.t)
        ys.append(c.y)
        dq += c.dropped_quality
        dm += c.dropped_missing
        dr += c.dropped_repeat
    t = np.concatenate(ts) if ts else np.zeros(0)
    y = np.concatenate(ys) if ys else np.zeros(0)
    station, year = station_header(path)
    info = {
        "station": station, "year": year,
        "days": days_in_year(year) if year else 0,
        "n": int(t.size), "dropped_quality": dq,
        "dropped_missing": dm, "dropped_repeat": dr,
    }
    return t, y, info


def coordinate_changes(
    path: Any, field: str = "TMP"
) -> list[dict[str, Any]]:
    """Every change of ``LATITUDE``, ``LONGITUDE`` or ``ELEVATION`` in one
    station-year, as ``{"t", "lat", "lon", "elev"}`` at the first row that
    carries the new position. The first row is the reference, not a
    change; an empty file gives an empty list."""
    out: list[dict[str, Any]] = []
    prev: tuple[float, float, float] | None = None
    for c in read_isd(path, field):
        for k in range(c.t.size):
            here = (float(c.lat[k]), float(c.lon[k]), float(c.elev[k]))
            if prev is not None and here != prev:
                out.append({"t": float(c.t[k]), "lat": here[0],
                            "lon": here[1], "elev": here[2]})
            prev = here
    return out


def day_grids(
    path: Any, days: Sequence[int], field: str = "TMP"
) -> dict[int, np.ndarray | None]:
    """The 24 hourly values of each requested day, in one pass.

    ``days`` are day indices from the year's start (0 for 1 January);
    duplicates collapse and the result has one entry per distinct day. A
    kept row falls in the hour bin it lies in (hour 23 holds 23:00 to
    23:59, so a station reporting at :53 fills every bin) and the first
    row in a bin wins; a day is ``None`` unless all 24 bins fill, which
    is what lets many stations share one nominal grid and one projection
    matrix. The values sit at the nominal positions ``k / 24``, and the
    raw reference the channel form is compared against uses those same
    positions, so the comparison measures the image machinery and not the
    binning. One pass matters: the caller that gates several days of a
    station would otherwise reread a multi-megabyte CSV per day.
    """
    wanted = {int(d) for d in days}
    if not wanted:
        return {}
    bins: dict[int, dict[int, float]] = {d: {} for d in wanted}
    last = max(wanted)
    for c in read_isd(path, field):
        for k in range(c.t.size):
            day = int(c.t[k])
            if day > last:
                return {d: _grid_or_none(bins[d]) for d in wanted}
            slot = bins.get(day)
            if slot is None:
                continue
            hour = int((float(c.t[k]) - day) * 24.0)
            if hour not in slot:
                slot[hour] = float(c.y[k])
    return {d: _grid_or_none(bins[d]) for d in wanted}


def day_grid(
    path: Any, day: int, field: str = "TMP"
) -> np.ndarray | None:
    """The 24 hourly values of one day, or ``None`` when an hour is
    missing; :func:`day_grids` for one day."""
    return day_grids(path, [int(day)], field)[int(day)]


def _grid_or_none(bins: dict[int, float]) -> np.ndarray | None:
    if len(bins) != 24:
        return None
    return np.array([bins[h] for h in range(24)], dtype=float)


def station_files(
    year_dir: Any,
    stations: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """The station CSVs of one year directory, sorted by station id;
    ``stations`` keeps only those ids, ``limit`` truncates."""
    root = Path(year_dir)
    if stations is not None:
        found = [root / f"{s}.csv" for s in sorted(set(stations))]
        out = [p for p in found if p.exists()]
    else:
        out = sorted(root.glob("*.csv"))
    return out[:limit] if limit is not None else out


def wban(station_id: str) -> str | None:
    """The WBAN part of an 11-character ISD id, or ``None`` when the id is
    the wrong length or its WBAN is the ``99999`` placeholder."""
    s = str(station_id)
    if len(s) != 11:
        return None
    part = s[6:11]
    return None if part == "99999" else part


def _open(opener: Callable[..., Any] | None) -> Callable[..., Any]:
    return urllib.request.urlopen if opener is None else opener


def normals_index(
    *,
    base_url: str = NORMALS_BASE,
    opener: Callable[..., Any] | None = None,
    timeout: int = 60,
) -> list[str]:
    """The WBAN-pattern station ids the hourly-normals directory offers,
    sorted: only ids matching ``[A-Z]{2}W[0-9]{8}`` are read from the
    listing, so a non-WBAN normals station present there is silently
    absent from the result. Harmless for :func:`match_normals`, which is
    WBAN-keyed on the ISD side too.

    ``opener(url, timeout=...)`` returns a binary file object; the default
    is :func:`urllib.request.urlopen`, so a test injects its own and makes
    no network call.
    """
    with _open(opener)(base_url, timeout=timeout) as fh:
        html = fh.read().decode("utf-8", "replace")
    return sorted(set(_HREF.findall(html)))


def match_normals(
    isd_ids: Sequence[str], normals_ids: Sequence[str]
) -> dict[str, str]:
    """``{isd id: normals id}`` for the ISD stations whose WBAN matches a
    normals station's last five characters. Ids with a ``99999`` WBAN and
    unmatched stations are absent."""
    by_wban = {
        Path(str(n)).stem[-5:]: Path(str(n)).stem for n in normals_ids
    }
    out: dict[str, str] = {}
    for sid in isd_ids:
        w = wban(sid)
        if w is not None and w in by_wban:
            out[str(sid)] = by_wban[w]
    return out


def download_normals(
    dest: Any,
    normals_ids: Sequence[str],
    *,
    base_url: str = NORMALS_BASE,
    opener: Callable[..., Any] | None = None,
    timeout: int = 60,
    overwrite: bool = False,
) -> list[Path]:
    """Fetch the hourly normals of ``normals_ids`` into ``dest``, keeping
    only :data:`NORMALS_COLUMNS`.

    The published file is 6.5 MB of 100-odd columns; the reduced copy is
    about 300 kB. An existing file is left alone unless ``overwrite``. A
    station the server does not serve is skipped. Returns the paths
    written or already present, in input order.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for sid in normals_ids:
        target = dest / f"{sid}.csv"
        if target.exists() and not overwrite:
            out.append(target)
            continue
        try:
            with _open(opener)(
                f"{base_url}{sid}.csv", timeout=timeout
            ) as fh:
                text = fh.read().decode("utf-8", "replace")
        except Exception:                        # not served: skip it
            continue
        reader = csv.reader(text.splitlines())
        header = next(reader, None)
        if header is None:
            continue
        idx = [header.index(c) for c in NORMALS_COLUMNS
               if c in header]
        if len(idx) != len(NORMALS_COLUMNS):
            continue
        with open(target, "w", newline="") as out_fh:
            writer = csv.writer(out_fh)
            writer.writerow(NORMALS_COLUMNS)
            for row in reader:
                if len(row) > max(idx):
                    writer.writerow([row[i].strip() for i in idx])
        out.append(target)
    return out


def read_normals(path: Any) -> dict[str, np.ndarray]:
    """One reduced normals file as arrays.

    Returns ``{"month", "day", "hour"}`` as integer arrays and
    ``{"temp_c", "pres_hpa"}`` as float arrays; the published temperature
    is degrees Fahrenheit to one decimal and is converted, the pressure is
    hectopascals and is not. A missing value (the ``-9999`` sentinel) is
    NaN.
    """
    month: list[int] = []
    day: list[int] = []
    hour: list[int] = []
    temp: list[float] = []
    pres: list[float] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            month.append(int(row["month"]))
            day.append(int(row["day"]))
            hour.append(int(row["hour"]))
            f = _float(row["HLY-TEMP-NORMAL"])
            p = _float(row["HLY-PRES-NORMAL"])
            temp.append((f - 32.0) * 5.0 / 9.0 if f > -9000 else np.nan)
            pres.append(p if p > -9000 else np.nan)
    return {
        "month": np.array(month, dtype=int),
        "day": np.array(day, dtype=int),
        "hour": np.array(hour, dtype=int),
        "temp_c": np.array(temp, dtype=float),
        "pres_hpa": np.array(pres, dtype=float),
    }
