"""Readers for the Nevada Geodetic Laboratory's daily position series and
for the two published baselines the showcase compares against: the MIDAS
velocities and the step database.

Every reader streams: a station file is handed to the caller in chunks of
at most ``chunk`` rows, so no reduction ever holds a dataset. The tables
(steps, MIDAS, holdings) are small enough to load whole; the largest,
``steps.txt``, is 142,474 lines.

Positions are returned relative to the first row's integer metre column
(``offset_e``, ``offset_n``, ``offset_u`` on every chunk), which is what
makes a relative exactness gate meaningful: the absolute coordinate is up
to 5.4 million metres while the annual amplitudes are millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

TENV3_COLUMNS = 23
# Rows per read block. A 100,000-row block leaves a 2.3 million element
# <U15 token array alive, 252 MiB traced and 537 MiB resident; 10,000
# keeps that under 30 MiB, and no NGL station exceeds 11,850 rows anyway.
TENV3_CHUNK = 10_000
MJD_ORDINAL = 678576                      # date(1858, 11, 17).toordinal()
COMPONENTS = ("east", "north", "up")
# Columns the reducer needs, in the order _rows returns them: decimal
# year, the three integer/fraction position pairs, the three sigmas.
_NEEDED = (2, 7, 8, 9, 10, 11, 12, 14, 15, 16)

_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def decimal_year(ymd: str) -> float:
    """The NGL decimal year of a ``YYMMMDD`` date such as ``08MAR27``.

    NGL's own column is ``2000 + (MJD - 51544) / 365.25`` and this
    reproduces it exactly, so a step date and an epoch are directly
    comparable. Two-digit years of 80 and above are 19xx, the rest 20xx.

    Raises:
        ValueError: the string is not ``YYMMMDD`` with a known month.
    """
    s = str(ymd).strip().upper()
    if len(s) != 7 or s[2:5] not in _MONTHS:
        raise ValueError(f"not a YYMMMDD date: {ymd!r}")
    try:
        yy, dd = int(s[:2]), int(s[5:])
    except ValueError as exc:
        raise ValueError(f"not a YYMMMDD date: {ymd!r}") from exc
    year = 1900 + yy if yy >= 80 else 2000 + yy
    mjd = date(year, _MONTHS[s[2:5]], dd).toordinal() - MJD_ORDINAL
    return 2000.0 + (mjd - 51544) / 365.25


@dataclass
class TenvChunk:
    """One block of a station's rows, already deduplicated by epoch.

    ``t`` is the decimal year; ``east``, ``north`` and ``up`` are metres
    measured from the first row's integer metre column, which travels
    unchanged on every chunk of the file as ``offset_e``, ``offset_n``,
    ``offset_u`` (add one back to recover the absolute coordinate).
    ``sig_*`` are the per-component standard deviations in metres, read
    and carried but used by no fit in this domain.
    """

    t: np.ndarray
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    sig_e: np.ndarray
    sig_n: np.ndarray
    sig_u: np.ndarray
    offset_e: float = 0.0
    offset_n: float = 0.0
    offset_u: float = 0.0


def _rows(fh: Any, chunk: int) -> Iterator[np.ndarray]:
    """Yield ``(m, 10)`` float arrays of the :data:`_NEEDED` columns,
    ``m <= chunk``.

    The whitespace tokens are parsed and released inside this loop rather
    than handed on, so the caller never holds a string array.

    Raises:
        ValueError: a block whose token count is not a multiple of 23.
    """
    while True:
        lines = list(islice(fh, chunk))
        if not lines:
            return
        tokens = np.array("".join(lines).split())
        if tokens.size % TENV3_COLUMNS:
            raise ValueError(
                f"a tenv3 row must have {TENV3_COLUMNS} columns; got "
                f"{tokens.size} tokens in a block of {len(lines)} lines"
            )
        cols = tokens.reshape(-1, TENV3_COLUMNS)[:, _NEEDED].astype(float)
        del tokens
        yield cols


def read_tenv3(path: Any, chunk: int = TENV3_CHUNK) -> Iterator[TenvChunk]:
    """Stream a ``.tenv3`` station file as :class:`TenvChunk` blocks.

    The header line is skipped when present. A row whose epoch is not
    strictly greater than every epoch before it is dropped -- repeats keep
    their first copy and a row that goes backwards is discarded rather
    than left in a non-monotonic series -- and the running maximum carries
    across chunk boundaries. A chunk whose rows are all dropped is skipped,
    not yielded empty, so every yielded chunk holds at least one sample.

    Raises:
        ValueError: a block whose token count is not a multiple of 23.
    """
    with open(path) as fh:
        first = fh.readline()
        if first and not first.lstrip().lower().startswith("site"):
            fh.seek(0)
        last_t = -np.inf
        offsets: tuple[float, float, float] | None = None
        for cols in _rows(fh, chunk):
            if offsets is None:
                offsets = (float(cols[0, 1]), float(cols[0, 3]),
                           float(cols[0, 5]))
            t = cols[:, 0]
            running = np.maximum.accumulate(
                np.concatenate(([last_t], t))
            )
            keep = t > running[:-1]
            last_t = float(running[-1])
            if not keep.any():
                continue
            def col(i: int, j: int, off: float) -> np.ndarray:
                return (cols[keep, i] - off) + cols[keep, j]
            yield TenvChunk(
                t=t[keep],
                east=col(1, 2, offsets[0]),
                north=col(3, 4, offsets[1]),
                up=col(5, 6, offsets[2]),
                sig_e=cols[keep, 7],
                sig_n=cols[keep, 8],
                sig_u=cols[keep, 9],
                offset_e=offsets[0], offset_n=offsets[1],
                offset_u=offsets[2],
            )


def read_epochs(path: Any, chunk: int = TENV3_CHUNK) -> np.ndarray:
    """The station's deduplicated epochs (decimal year) and nothing else.

    The first of the reducer's two passes: at 11,850 epochs for the
    longest station this is 95 kB, which fixes the domains, the segment
    boundaries and every order before a projection runs.
    """
    parts = [c.t for c in read_tenv3(path, chunk)]
    return np.concatenate(parts) if parts else np.zeros(0)


@dataclass
class Step:
    """One entry of the NGL step database.

    ``code`` 1 is an equipment change (``label`` names it) and 2 a
    possible earthquake step, for which ``threshold_km``, ``distance_km``,
    ``magnitude`` and ``event`` (the USGS id) are set and ``label`` is
    None. ``year`` is the date on the epochs' decimal-year scale.
    """

    sta: str
    ymd: str
    year: float
    code: int
    label: str | None = None
    threshold_km: float | None = None
    distance_km: float | None = None
    magnitude: float | None = None
    event: str | None = None


def read_steps(path: Any) -> dict[str, list[Step]]:
    """The whole step database keyed by station, each list in file order.

    A line that is blank, or whose date or code cannot be parsed, is
    skipped silently: the file is machine-generated daily and a station
    with no parsable step simply has no entry.
    """
    out: dict[str, list[Step]] = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            sta, ymd, code_s = parts[0], parts[1], parts[2]
            try:
                year, code = decimal_year(ymd), int(code_s)
            except ValueError:
                continue
            step = Step(sta=sta, ymd=ymd, year=year, code=code)
            if code == 2 and len(parts) >= 7:
                step.threshold_km = float(parts[3])
                step.distance_km = float(parts[4])
                step.magnitude = float(parts[5])
                step.event = parts[6]
            elif len(parts) >= 4:
                step.label = parts[3]
            out.setdefault(sta, []).append(step)
    return out


@dataclass
class Midas:
    """One station's published MIDAS velocity (metres per year) and its
    uncertainty, with the span the estimate covers."""

    sta: str
    first: float
    last: float
    span: float
    n_epochs: int
    n_good: int
    ve: float
    vn: float
    vu: float
    se: float
    sn: float
    su: float
    n_steps: int


def read_midas(path: Any) -> dict[str, Midas]:
    """The MIDAS velocity table keyed by station. Columns follow
    ``midas.readme.txt``: 3 and 4 the first and last epoch, 5 the span, 6
    and 7 the epoch counts, 9 to 11 the east/north/up velocities, 12 to 14
    their uncertainties, 24 the number of steps assumed. Short lines are
    skipped."""
    out: dict[str, Midas] = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 24:
                continue
            out[p[0]] = Midas(
                sta=p[0], first=float(p[2]), last=float(p[3]),
                span=float(p[4]), n_epochs=int(p[5]), n_good=int(p[6]),
                ve=float(p[8]), vn=float(p[9]), vu=float(p[10]),
                se=float(p[11]), sn=float(p[12]), su=float(p[13]),
                n_steps=int(p[23]),
            )
    return out


@dataclass
class Holding:
    """One row of ``DataHoldings.txt``: where a station is and how much
    data it has. ``first`` and ``last`` are ``YYYY-MM-DD`` strings."""

    sta: str
    lat: float
    lon: float
    height: float
    first: str
    last: str
    n_sol: int


def read_holdings(path: Any) -> dict[str, Holding]:
    """The station index keyed by station; the header line and any short
    line are skipped."""
    out: dict[str, Holding] = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 11 or p[0] == "Sta":
                continue
            try:
                out[p[0]] = Holding(
                    sta=p[0], lat=float(p[1]), lon=float(p[2]),
                    height=float(p[3]), first=p[7], last=p[8],
                    n_sol=int(p[10]),
                )
            except ValueError:
                continue
    return out


def station_files(
    tenv3_dir: Any,
    stations: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """The ``.tenv3`` files under ``tenv3_dir``, sorted by station name.

    ``stations`` keeps only those names (a name with no file is skipped);
    ``limit`` truncates the list. Returns an empty list for a directory
    with no matching file.
    """
    root = Path(tenv3_dir)
    if stations is not None:
        found = [root / f"{s}.tenv3" for s in sorted(set(stations))]
        out = [p for p in found if p.exists()]
    else:
        out = sorted(root.glob("*.tenv3"))
    return out[:limit] if limit is not None else out
