"""One command for every real-data run of the showcase.

    python -m dtfit_experimental.experiments.domains.image_showcase.cli \\
        ngl-reduce --workers 16 --images ~/data/showcase/images/ngl

Each subcommand writes one or two CSV files into the domain's tracked
``results/`` directory and prints what it wrote. ``--suffix`` renames the
output (the Pi's runs use ``--suffix _pi``), so the two machines' numbers
sit side by side. The exactness subcommands exit 1 when any station misses
the gate, so a run that breaks the claim fails visibly.

Every process runs one BLAS thread: the parent through
:func:`threadpoolctl.threadpool_limits`, a pool's workers through
the ``*_NUM_THREADS`` variables they inherit. A ``cpu-N`` row then
counts N cores, and N workers never spawn N times the core count
in threads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from dtfit.image import ImageStream, assemble
from dtfit.streaming import LSIFilter

from . import (
    compare, filters, isd, isd_fits, isd_reduce, ngl, ngl_fits,
    ngl_reduce, paths, stream, throughput,
)
from .store import load_images, write_table

RESULTS = {
    "ngl-reduce": "ngl_reduce",
    "ngl-fits": "ngl_fits",
    "ngl-exact": "ngl_exact",
    "ngl-rank": "ngl_rank",
    "isd-reduce": "isd_reduce",
    "isd-fits": "isd_year_fits",
    "isd-exact": "isd_exact",
    "normals-fetch": "normals_fetch",
    "isd-normals": "isd_normals",
    "filters-ngl": "filters_ngl",
    "filters-isd": "filters_isd",
    "throughput": "throughput",
    "gemm": "gemm",
    "stream-serve": "leg5_serve",
    "stream-replay": "leg5_replay",
    "stream-track": "leg5_track",
    "leg5-tables": "leg5_tables",
}

# One entry per leg-5 run this domain has produced: the JSON summary's
# stem under results/leg5/, its optional producer-side sibling, the
# direction and block length the table reports it under, and the image
# subtree its groups reassemble from (see task 18's brief). A source
# whose JSON is absent is skipped, so a partial leg 5 (one direction
# unmeasured) still tables what exists.
LEG5_SOURCES: tuple[tuple[str, str | None, str, float, str], ...] = (
    ("pi_to_pc", "pi_to_pc_producer", "pi->pc", 1.0, "ngl"),
    ("pc_to_pi", "pc_to_pi_producer", "pc->pi", 1.0, "ngl"),
    ("block_0.25", None, "local", 0.25, "ngl-b0.25"),
    ("block_4.0", None, "local", 4.0, "ngl-b4.0"),
)
LEG5_RATE_TAGS = ("1000", "10000", "100000", "max")

LEG5_STREAM_COLUMNS = [
    "direction", "block_years", "n_images", "bytes_header",
    "bytes_payload", "seconds", "images_per_second",
    "latency_ms_median", "latency_ms_p90", "clock_delta_s", "groups",
    "n_flags", "mismatched",
]
LEG5_GROUP_COLUMNS = [
    "direction", "block_years", "station", "field", "n_blocks",
    "samples", "bytes_header", "bytes_payload", "n_flags", "digest",
    "mismatched",
]
LEG5_REPLAY_COLUMNS = [
    "direction", "rate_requested", "rate_achieved", "n_samples",
    "n_frames", "dropped_sender", "dropped_receiver", "us_per_update",
    "n_flags", "n_blocks_back", "bytes_forward", "bytes_back",
    "seconds", "flags_match",
]

TIMING_COLUMNS = ["host", "command", "rows", "seconds"]

# The models the stream roles use, resolved here so that stream.py needs
# no import from the dataset modules or from filters.py. The model has to
# match the images the producer sends: `assemble` cannot raise the order
# above the block order, so an assembly of order-12 yearly NGL blocks
# carries a trend and nothing finer -- fitting the full model on it warns
# about coverage and returns a number worth nothing. The whole-span and
# station-year images do carry their full models.
STREAM_MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ngl-trend": ("c + v*t", ("c", "v")),
    "ngl-whole": (ngl_reduce.NGL_EXPR, tuple(ngl_reduce.NGL_NAMES)),
    "isd-diurnal": (
        isd_reduce.DIURNAL_EXPR, tuple(isd_reduce.DIURNAL_NAMES)
    ),
    "isd-year": (
        isd_reduce.ANNUAL_EXPR, tuple(isd_reduce.ANNUAL_NAMES)
    ),
}

# How many rows the running subcommand wrote, for the timing row.
_WRITTEN = [0]


def _out(name: str, suffix: str) -> Path:
    return paths.results_dir() / f"{name}{suffix}.csv"


def _write(name: str, suffix: str, rows: Sequence[dict[str, Any]],
           columns: Sequence[str]) -> Path:
    path = write_table(_out(name, suffix), list(rows), columns)
    _WRITTEN[0] += len(rows)
    print(f"{len(rows)} rows -> {path}")
    return path


def _steps_by_station(root: Path) -> dict[str, list[float]]:
    """Step decimal years per station, or an empty mapping when the
    database is not next to the data."""
    steps_file = paths.ngl_dir(root) / "steps.txt"
    if not steps_file.exists():
        return {}
    return {
        sta: [s.year for s in group]
        for sta, group in ngl.read_steps(steps_file).items()
    }


def _ngl_inputs(args: argparse.Namespace) -> list[Path]:
    stations = args.stations.split(",") if args.stations else None
    return ngl.station_files(
        paths.ngl_dir() / "tenv3", stations=stations, limit=args.limit
    )


def _npz_files(
    images: Any, limit: int | None = None, stations: str | None = None
) -> list[Path]:
    """The ``.npz`` files under ``images``, sorted by name.

    ``stations`` is the comma-separated list ``--stations`` carries and
    keeps only those stems (a named station with no file is skipped);
    ``limit`` truncates. Returns an empty list for a directory with no
    match, which is how a command run before its reducer reports
    nothing rather than raising.
    """
    root = Path(images)
    if stations:
        named = sorted(set(stations.split(",")))
        found = [root / f"{s}.npz" for s in named]
        out = [p for p in found if p.exists()]
    else:
        out = sorted(root.glob("*.npz"))
    return out[:limit] if limit else out


def cmd_ngl_reduce(args: argparse.Namespace) -> int:
    rows = ngl_reduce.reduce_many(
        _ngl_inputs(args), args.images,
        _steps_by_station(paths.data_root()),
        workers=args.workers, block_len=args.block,
    )
    _write(RESULTS["ngl-reduce"], args.suffix, rows,
           ngl_reduce.REDUCE_COLUMNS)
    failed = [r for r in rows if r["error"]]
    if failed:
        print(f"{len(failed)} stations failed; see the error column")
    good = [r for r in rows if not r["error"]]
    if good:
        raw = sum(int(r["raw_bytes"]) for r in good)
        img = sum(int(r["coef_bytes"]) + int(r["gram_bytes"])
                  + int(r["grid_bytes"]) for r in good)
        print(f"images {img / 1e9:.3f} GB against {raw / 1e9:.3f} GB raw "
              f"({img / raw:.3f}x)")
    return 0


def cmd_ngl_fits(args: argparse.Namespace) -> int:
    midas_file = paths.ngl_dir() / "midas.IGS20.txt"
    midas = ngl.read_midas(midas_file) if midas_file.exists() else {}
    rows = ngl_fits.run_fits(
        _npz_files(args.images, args.limit, args.stations), midas,
        workers=args.workers,
    )
    _write(RESULTS["ngl-fits"], args.suffix, rows, ngl_fits.FIT_COLUMNS)
    return 0


def _regate(rows: Sequence[dict[str, Any]], tol: float) -> None:
    """Recompute ``gate`` from ``tol`` in both directions, in place.

    ``exactness_rows`` set it against the module's own tolerance, so a
    looser ``--tol`` has to be able to clear a FAIL as well as a tighter
    one has to be able to raise one. ``UNDERSAMPLED`` and ``ERROR``
    verdicts are left alone: neither is about the tolerance.
    """
    for row in rows:
        if str(row["gate"]).startswith(("UNDERSAMPLED", "ERROR")):
            continue
        day = row.get("day_score")
        bound = float(row["design_cond"]) ** 2 * (
            float(row["coverage"])
            + compare.EPS * float(row["gram_cond"])
        )
        row["gate"] = compare.verdict(
            float(row["score"]), bound, tol,
            also_missed=day is not None and float(day) > tol,
        )


def _gate_exit(rows: Sequence[dict[str, Any]], tol: float) -> int:
    """Print the summary line and return the process exit code: 1 only
    when a row actually failed the tolerance. The worst score is over the
    rows the tolerance judged; the ill-conditioned rows' own worst is
    printed beside it."""
    def worst(gates: tuple[str, ...]) -> float:
        scores = [
            float(r["score"]) for r in rows
            if r["gate"] in gates and np.isfinite(float(r["score"]))
        ]
        return max(scores) if scores else float("nan")

    under = sum(1 for r in rows if r["gate"] == "UNDERSAMPLED")
    ill = sum(1 for r in rows if r["gate"] == "ILL-CONDITIONED")
    failed = sum(1 for r in rows if str(r["gate"]).startswith(
        ("FAIL", "ERROR")
    ))
    print(f"worst score {worst(('ok', 'FAIL')):.3e} against {tol:.1e}; "
          f"{under} undersampled, {ill} ill-conditioned (worst "
          f"{worst(('ILL-CONDITIONED',)):.3e}), {failed} failed")
    return 1 if failed else 0


def cmd_ngl_exact(args: argparse.Namespace) -> int:
    pairs = [
        (paths.ngl_dir() / "tenv3" / f"{p.stem}.tenv3", p)
        for p in _npz_files(args.images, args.limit, args.stations)
    ]
    rows = ngl_fits.run_exactness(pairs, workers=args.workers)
    _regate(rows, args.tol)
    _write(RESULTS["ngl-exact"], args.suffix, rows,
           ngl_fits.EXACT_COLUMNS)
    errs = [
        float(r["gram_rebuild_err"]) for r in rows
        if r["gram_rebuild_err"] is not None
    ]
    if errs:
        print(f"worst G rebuild error {max(errs):.3e} over {len(errs)} "
              f"images (what shipping S and the grid alone would cost)")
    return _gate_exit(rows, args.tol)


def cmd_ngl_rank(args: argparse.Namespace) -> int:
    pairs = [
        (p, None if args.no_raw
         else paths.ngl_dir() / "tenv3" / f"{p.stem}.tenv3")
        for p in _npz_files(args.images, args.limit, args.stations)
    ]
    rows = ngl_fits.run_rankings(pairs, workers=args.workers)
    _write(RESULTS["ngl-rank"], args.suffix, rows, ngl_fits.RANK_COLUMNS)
    return 0


def _isd_images(args: argparse.Namespace) -> Path:
    """``<images>/<year>`` where ``--images`` already ends in ``isd``:
    the layout ruling 11 documents, ``<root>/images/isd/<year>/``."""
    return Path(args.images) / str(args.year)


def cmd_isd_reduce(args: argparse.Namespace) -> int:
    stations = args.stations.split(",") if args.stations else None
    files = isd.station_files(
        paths.isd_dir(args.year), stations=stations, limit=args.limit
    )
    rows = isd_reduce.reduce_many_years(
        files, _isd_images(args),
        fields=tuple(args.fields.split(",")), workers=args.workers,
        with_days=args.with_days,
    )
    _write(RESULTS["isd-reduce"], args.suffix, rows,
           isd_reduce.ISD_REDUCE_COLUMNS)
    good = [r for r in rows if not r["error"]]
    if good:
        raw = sum(int(r["raw_bytes"]) for r in good)
        img = sum(int(r["coef_bytes"]) + int(r["gram_bytes"])
                  + int(r["grid_bytes"]) for r in good)
        print(f"images {img / 1e6:.1f} MB against {raw / 1e9:.3f} GB raw "
              f"({raw / max(img, 1):.0f}x)")
    return 0


def cmd_isd_fits(args: argparse.Namespace) -> int:
    rows = isd_fits.run_year_fits(
        _npz_files(_isd_images(args), args.limit, args.stations),
        fields=tuple(args.fields.split(",")), workers=args.workers,
    )
    _write(RESULTS["isd-fits"], args.suffix, rows, isd_fits.YEAR_COLUMNS)
    return 0


def cmd_isd_exact(args: argparse.Namespace) -> int:
    year_dir = paths.isd_dir(args.year)
    pairs = [
        (year_dir / f"{p.stem}.csv", p)
        for p in _npz_files(_isd_images(args), args.limit, args.stations)
    ]
    rows = isd_fits.run_exactness_isd(pairs, workers=args.workers)
    _regate(rows, args.tol)
    _write(RESULTS["isd-exact"], args.suffix, rows,
           isd_fits.EXACT_ISD_COLUMNS)
    checked = sum(int(r["n_days_checked"]) for r in rows)
    print(f"{checked} day images gated against their raw hourly rows")
    return _gate_exit(rows, args.tol)


def cmd_normals_fetch(args: argparse.Namespace) -> int:
    ids = isd.normals_index()
    stations = [p.stem for p in
                isd.station_files(paths.isd_dir(args.year))]
    matched = isd.match_normals(stations, ids)
    written = isd.download_normals(
        paths.normals_dir(), sorted(set(matched.values()))
    )
    rows = [
        {"isd_station": sid, "normals_id": nid,
         "path": str(paths.normals_dir() / f"{nid}.csv")}
        for sid, nid in sorted(matched.items())
        if (paths.normals_dir() / f"{nid}.csv").exists()
    ]
    _write(RESULTS["normals-fetch"], args.suffix, rows,
           ["isd_station", "normals_id", "path"])
    print(f"{len(written)} normals files in {paths.normals_dir()}")
    return 0


def cmd_isd_normals(args: argparse.Namespace) -> int:
    matched = {
        r["isd_station"]: r["normals_id"]
        for r in _read_rows(_out(RESULTS["normals-fetch"], args.suffix))
    }
    year_images = _isd_images(args)
    pairs = [
        (year_images / f"{sid}.npz", paths.normals_dir() / f"{nid}.csv")
        for sid, nid in sorted(matched.items())
        if (year_images / f"{sid}.npz").exists()
    ]
    rows = isd_fits.run_normals(pairs, workers=args.workers)
    _write(RESULTS["isd-normals"], args.suffix, rows,
           isd_fits.NORMALS_ROW_COLUMNS)
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV this tool wrote earlier; empty when it is missing."""
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def cmd_filters_ngl(args: argparse.Namespace) -> int:
    steps = ngl.read_steps(paths.ngl_dir() / "steps.txt")
    holdings = ngl.read_holdings(paths.ngl_dir() / "DataHoldings.txt")
    candidates = [
        sta for sta, group in sorted(steps.items())
        if any(s.code == 2 for s in group)
        and holdings.get(sta) is not None
        and holdings[sta].n_sol >= 1000
    ][: args.limit or 200]
    jobs = [
        (paths.ngl_dir() / "tenv3" / f"{sta}.tenv3", steps[sta])
        for sta in candidates
        if (paths.ngl_dir() / "tenv3" / f"{sta}.tenv3").exists()
    ]
    summary, step_rows = filters.run_ngl_filters(
        jobs, workers=args.workers
    )
    _write(RESULTS["filters-ngl"], args.suffix, summary,
           filters.FILTER_COLUMNS)
    _write(RESULTS["filters-ngl"] + "_steps", args.suffix, step_rows,
           filters.STEP_COLUMNS)
    return 0


def cmd_filters_isd(args: argparse.Namespace) -> int:
    matched = [
        r["isd_station"]
        for r in _read_rows(_out(RESULTS["normals-fetch"], args.suffix))
    ]
    year_dir = paths.isd_dir(args.year)
    files = [year_dir / f"{sid}.csv" for sid in matched
             if (year_dir / f"{sid}.csv").exists()][: args.limit or 200]
    summary, flag_rows = filters.run_isd_filters(
        files, workers=args.workers
    )
    _write(RESULTS["filters-isd"], args.suffix, summary,
           filters.ISD_FILTER_COLUMNS)
    _write(RESULTS["filters-isd"] + "_flags", args.suffix, flag_rows,
           filters.ISD_FLAG_COLUMNS)
    return 0


def cmd_throughput(args: argparse.Namespace) -> int:
    files = _ngl_inputs(args)
    half = max(1, len(files) // 2)
    disk = throughput.disk_read_rate(files[:half])
    machine = throughput.machine_row()
    rows: list[dict[str, Any]] = [{
        "dataset": "ngl", "route": "disk", "workers": 0,
        "n_files": disk["n_files"], "samples": None,
        "seconds": disk["seconds"],
        "samples_per_second": None, "peak_mib": None,
        "peak_rss_mib": None,
        "raw_bytes": disk["bytes"], "coef_bytes": None,
        "gram_bytes": None, "grid_bytes": None, "image_bytes": None,
        "reduction_ratio": None, "host": machine["host"],
        "backend": "none",
        "note": f"{disk['mb_per_second']} MB/s raw read",
    }]
    for workers in sorted({1, args.workers}):
        rows.append(throughput.reduce_rate(
            files[half:], Path(args.images) / f"cpu{workers}",
            dataset="ngl", workers=workers,
            steps_by_station=_steps_by_station(paths.data_root()),
        ))
    if files:
        # The largest file by bytes, not the last of the (limited,
        # name-sorted) list: the spec's leg 1 wants this measured on the
        # longest series, and file size tracks epoch count directly.
        longest = max(files, key=lambda p: p.stat().st_size)
        t_list, y_list = [], []
        for chunk in ngl.read_tenv3(longest):
            t_list.append(chunk.t)
            y_list.append(chunk.east)
        t = np.concatenate(t_list)
        y = np.concatenate(y_list)
        err = throughput.float32_error(
            t - t[0], y, compare.legendre_order(float(t[-1] - t[0]), t.size)
        )
        rows.append({
            "dataset": "ngl", "route": "float32", "workers": 0,
            "n_files": 1, "samples": err["n"], "seconds": None,
            "samples_per_second": None, "peak_mib": None,
            "peak_rss_mib": None,
            "raw_bytes": None, "coef_bytes": None, "gram_bytes": None,
            "grid_bytes": None, "image_bytes": None,
            "reduction_ratio": None, "host": machine["host"],
            "backend": "numpy",
            "note": (f"rel S float32 {err['rel_S_float32']:.2e}, "
                     f"float64 {err['rel_S_float64']:.2e}"),
        })
    _write(RESULTS["throughput"], args.suffix, rows,
           throughput.THROUGHPUT_COLUMNS)
    return 0


def _gemm_batch(
    files: Sequence[Path], days: Sequence[int]
) -> tuple[np.ndarray, int, float]:
    """``(Y, n_columns, gather_seconds)`` for a batch.

    One column per (station, day) that filled all 24 hour bins, all on
    the same nominal grid, so the projection is a single
    ``Phi^T Y`` with ``Phi`` 24 by ``order + 1``. Concatenating several
    days is what lets the batch grow past the station count. Returns an
    empty array when nothing qualified.
    """
    columns: list[np.ndarray] = []
    started = time.perf_counter()
    for path in files:
        grids = isd.day_grids(path, days, "TMP")
        for day in days:
            grid = grids.get(day)
            if grid is not None:
                columns.append(grid)
    gather = time.perf_counter() - started
    if not columns:
        return np.zeros((24, 0)), 0, gather
    Y = np.column_stack(columns)
    return Y, Y.shape[1], gather


def cmd_gemm(args: argparse.Namespace) -> int:
    """Sweep the channel-form batch over station counts and day counts.

    The projection is 24 by 17 by B, an arithmetic intensity so low that
    the interesting number is how the rate moves with B, not any single
    point. The Gram update stays on the host whatever the backend
    (``_Sums.add`` computes ``Phi.T @ (w * Phi)`` in numpy), so only
    ``S`` is accelerated; the note on every row says so.
    """
    year_dir = paths.isd_dir(args.year)
    every = isd.station_files(year_dir)
    limits = [n for n in (500, 5000, len(every)) if n <= len(every)]
    limits = sorted(set(limits))
    day_counts = [int(d) for d in args.days.split(",")]
    rows: list[dict[str, Any]] = []
    for n_stations in limits:
        files = every[:n_stations]
        for n_days in day_counts:
            days = list(range(args.day, args.day + n_days))
            Y, channels, gather = _gemm_batch(files, days)
            if not channels:
                print(f"{n_stations} stations, {n_days} days: none "
                      f"filled the 24 bins")
                continue
            for backend in ("numpy", "cupy"):
                note = (
                    f"{n_stations} files, {n_days} days, {channels} "
                    f"qualified; the Gram stays on the host"
                )
                if backend != "numpy":
                    ok, message = throughput.gpu_probe(backend)
                    if not ok:
                        # A skipped GPU row is reported, never dropped.
                        rows.append({
                            "dataset": "isd", "backend": backend,
                            "order": isd_reduce.DIURNAL_ORDER,
                            "channels": channels, "days": n_days,
                            "samples": int(Y.size), "repeats": 0,
                            "seconds": None,
                            "elements_per_second": None,
                            "gather_seconds": round(gather, 3),
                            "note": f"skipped: {message}",
                        })
                        continue
                got = throughput.channel_gemm_rate(
                    Y, order=isd_reduce.DIURNAL_ORDER, backend=backend,
                    repeats=args.repeats,
                )
                got.pop("images", None)
                got["dataset"] = "isd"
                got["days"] = n_days
                got["gather_seconds"] = round(gather, 3)
                got["note"] = note
                rows.append(got)
    if not rows:
        print("no station filled a day's 24 bins; nothing to measure")
        return 0
    _write(RESULTS["gemm"], args.suffix, rows, throughput.GEMM_COLUMNS)
    return 0


def cmd_stream_serve(args: argparse.Namespace) -> int:
    """The leg-5 consumer with the dataset's model filled in: it
    assembles each station's blocks, fits that model on the assembly and
    tallies the drift flags the producer's stream raised."""
    expr, names = STREAM_MODELS[args.model]
    summary = stream.serve(
        args.host, args.port, expect=args.expect,
        port_file=args.port_file, timeout=args.timeout,
        expr=expr, names=list(names),
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    rows = [
        dict({"group": key}, **{
            "n_blocks": g["n_blocks"], "n": g["n"],
            "n_flags": g["n_flags"], "digest": g["digest"],
            "bytes_header": g["bytes_header"],
            "bytes_payload": g["bytes_payload"],
            "samples": g["samples"],
            **{f"p_{k}": v for k, v in (g["params"] or {}).items()},
        })
        for key, g in sorted(summary["groups"].items())
    ]
    columns = ["group", "n_blocks", "n", "n_flags", "digest",
               "bytes_header", "bytes_payload", "samples"]
    columns += [f"p_{n}" for n in names]
    _write(RESULTS["stream-serve"], args.suffix, rows, columns)
    return 0


def cmd_stream_replay(args: argparse.Namespace) -> int:
    """The leg-5 replayer: stream one station's raw samples at a
    requested rate. The tenv3 reading lives here, not in ``stream.py``,
    for the same reason the tracker's filter does."""
    stations = args.stations.split(",") if args.stations else None
    files = ngl.station_files(
        paths.ngl_dir() / "tenv3", stations=stations, limit=args.limit
    )

    def chunks():
        for path in files:
            first_t = None
            for c in ngl.read_tenv3(path, args.chunk):
                if first_t is None:
                    first_t = c.t[0]
                yield path.stem, args.field, c.t - first_t, getattr(
                    c, args.field
                )

    out = stream.replay(
        args.host, args.port, chunks(), rate=args.rate,
        timeout=args.timeout,
    )
    row = dict(out)
    row["host"] = throughput.machine_row()["host"]
    row["n_stations"] = len(files)
    columns = ["host", "n_stations", "rate_requested",
               "samples_per_second", "n_frames", "n_samples",
               "n_dropped", "bytes_sent", "n_frames_back", "seconds"]
    _write(RESULTS["stream-replay"], args.suffix, [row], columns)
    return 0


def cmd_stream_track(args: argparse.Namespace) -> int:
    """The leg-5 tracker: run an ``LSIFilter`` over the replayed samples
    and send block images back. The filter is built here, so
    ``stream.py`` needs no import from ``filters.py``.

    One connection carries every replayed station in turn; a fresh
    filter and block stream are built for each one (matching
    ``filters.run_filter``'s ``p0`` and ``q_diag``) so that a station's
    state never leaks into the next.
    """
    config = filters.NGL_CONFIGS.get(args.config) or (
        filters.ISD_CONFIG if args.config == "isd-diurnal" else None
    )
    if config is None:
        raise SystemExit(f"unknown filter config {args.config!r}")

    def reset(_station: str, _t: np.ndarray, y: np.ndarray) -> Any:
        p0 = [float(y[0]) if n == "c" else 0.0 for n in config.names]
        kwargs: dict[str, Any] = {
            "order": config.order, "window_size": config.window,
            "adaptive_window": config.adaptive, "p0": p0,
        }
        if config.q_diag is not None:
            kwargs["q_diag"] = list(config.q_diag)
        filt = LSIFilter(config.expr, "t", **kwargs)
        blocks = ImageStream(
            "legendre", args.order, domain=(0.0, args.span),
            block=args.block, detect="previous", grid="explicit",
            keep_fine=int(args.span / args.block) + 2,
            fold=int(args.span / args.block) + 2,
        )
        return filt, blocks

    out = stream.track(
        args.host, args.port, None, block_stream=None,
        station=args.station, field=args.field,
        port_file=args.port_file, timeout=args.timeout,
        reset=reset,
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n"
        )
    row = {k: v for k, v in out.items() if k not in ("flags", "params")}
    row["host"] = throughput.machine_row()["host"]
    row["config"] = args.config
    row["flag_times"] = " ".join(f"{f:.6f}" for f in out["flags"])
    columns = ["host", "config", "n_frames", "n_samples", "n_dropped",
               "seconds", "samples_per_second", "us_per_update",
               "n_flags", "n_blocks", "bytes_back", "flag_times"]
    _write(RESULTS["stream-track"], args.suffix, [row], columns)
    return 0


def _leg5_group_mismatch(
    images_root: Path, subdir: str, key: str, group: dict[str, Any]
) -> int | None:
    """0 or 1: a local reassembly of ``key``'s blocks digests differently
    from ``group``'s remote assembly. ``None`` when the station's images
    are not on this machine (or carry no matching block), so the column
    stays blank rather than claiming a match nothing checked."""
    station, field = key.split("/")
    npz = images_root / subdir / f"{station}.npz"
    if not npz.exists():
        return None
    images, _info = load_images(npz)
    blocks = [
        img for name, img in images.items()
        if name.startswith("blk") and name.endswith(f"_{field}")
    ]
    if not blocks:
        return None
    local = assemble(blocks[: int(group["n_blocks"])],
                     order=min(b.order for b in blocks))
    return int(stream.image_digest(local) != group["digest"])


def _leg5_local_flags(
    stations: Sequence[str], tenv3_dir: Path, field: str = "east"
) -> int:
    """Total drift flags the NGL detection filter raises on ``field`` for
    ``stations``, run fresh off their ``.tenv3`` files: the number the
    wire replay's flag count is checked against, not the 200-station
    ``filters_ngl.csv`` run (a different station set)."""
    config = {"detection": filters.NGL_CONFIGS["detection"]}
    total = 0
    for sta in stations:
        path = tenv3_dir / f"{sta}.tenv3"
        if not path.exists():
            continue
        summary, _steps = filters.ngl_filter_station(path, [],
                                                       configs=config)
        total += sum(
            r["n_flags"] for r in summary if r["component"] == field
        )
    return total


def leg5_tables(
    leg5_dir: Path,
    images_root: Path,
    results_dir: Path,
    tenv3_dir: Path,
    *,
    sources: Sequence[tuple[str, str | None, str, float, str]] =
    LEG5_SOURCES,
    rate_tags: Sequence[str] = LEG5_RATE_TAGS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build and write the leg-5 tables from the run's own JSON summaries.

    ``leg5_dir`` holds one ``<name>.json`` per entry of ``sources`` that
    was actually run (and its producer sibling, when named); ``mismatched``
    is recomputed here rather than trusted from the run, by reassembling
    each group locally from ``images_root`` and comparing digests
    (:func:`_leg5_group_mismatch`). ``rate_tags`` names the sample-rate
    sweep whose ``leg5_replay_rate_<tag>.csv``/``leg5_track_rate_<tag>.csv``
    pairs live in ``results_dir``; each sweep row's ``flags_match`` is
    checked against a fresh local run of the detection filter on
    ``tenv3_dir`` over the stations the localhost block-length sources
    used -- the same twenty stations the sample-rate sweep replayed.

    Writes ``leg5_stream.csv``, ``leg5_groups.csv`` and ``leg5_replay.csv``
    into ``results_dir`` and returns the three row lists written.
    """
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for name, producer, direction, block, subdir in sources:
        path = leg5_dir / f"{name}.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        p: dict[str, Any] = {}
        if producer and (leg5_dir / f"{producer}.json").exists():
            p = json.loads((leg5_dir / f"{producer}.json").read_text())
        mismatched = 0
        for key, g in sorted(s["groups"].items()):
            bad = _leg5_group_mismatch(images_root, subdir, key, g)
            if bad is not None:
                mismatched += bad
            station, field = key.split("/")
            group_rows.append({
                "direction": direction, "block_years": block,
                "station": station, "field": field,
                "n_blocks": g["n_blocks"], "samples": g["samples"],
                "bytes_header": g["bytes_header"],
                "bytes_payload": g["bytes_payload"],
                "n_flags": g["n_flags"], "digest": g["digest"],
                "mismatched": bad,
            })
        rows.append({
            "direction": direction, "block_years": block,
            "n_images": s["n_images"], "bytes_header": s["bytes_header"],
            "bytes_payload": s["bytes_payload"], "seconds": s["seconds"],
            "images_per_second": s["images_per_second"],
            "latency_ms_median": p.get("latency_ms_median"),
            "latency_ms_p90": p.get("latency_ms_p90"),
            "clock_delta_s": s["clock_delta_s"],
            "groups": len(s["groups"]),
            "n_flags": sum(g["n_flags"] for g in s["groups"].values()),
            "mismatched": mismatched,
        })

    stations = sorted({
        r["station"] for r in group_rows if r["direction"] == "local"
    })
    local_flags = _leg5_local_flags(stations, tenv3_dir) if stations else None

    sweep: list[dict[str, Any]] = []
    for tag in rate_tags:
        send = results_dir / f"leg5_replay_rate_{tag}.csv"
        recv = results_dir / f"leg5_track_rate_{tag}.csv"
        if not (send.exists() and recv.exists()):
            continue
        s_row = _read_rows(send)[0]
        r_row = _read_rows(recv)[0]
        matches = (
            local_flags is not None
            and int(s_row["n_dropped"]) == 0
            and int(r_row["n_dropped"]) == 0
            and int(r_row["n_flags"]) == local_flags
        )
        sweep.append({
            "direction": "pc->pi",
            "rate_requested": s_row["rate_requested"],
            "rate_achieved": s_row["samples_per_second"],
            "n_samples": s_row["n_samples"],
            "n_frames": s_row["n_frames"],
            "dropped_sender": s_row["n_dropped"],
            "dropped_receiver": r_row["n_dropped"],
            "us_per_update": r_row["us_per_update"],
            "n_flags": r_row["n_flags"],
            "n_blocks_back": r_row["n_blocks"],
            "bytes_forward": s_row["bytes_sent"],
            "bytes_back": r_row["bytes_back"],
            "seconds": r_row["seconds"], "flags_match": matches,
        })

    write_table(results_dir / "leg5_stream.csv", rows, LEG5_STREAM_COLUMNS)
    write_table(results_dir / "leg5_groups.csv", group_rows,
                LEG5_GROUP_COLUMNS)
    write_table(results_dir / "leg5_replay.csv", sweep, LEG5_REPLAY_COLUMNS)
    return rows, group_rows, sweep


def cmd_leg5_tables(args: argparse.Namespace) -> int:
    """Fold the leg-5 JSON summaries (``results/leg5/``) into the three
    published tables; see :func:`leg5_tables`. ``--images`` is the image
    tree root, defaulting to ``<data>/images``."""
    rows, group_rows, sweep = leg5_tables(
        paths.results_dir() / "leg5", Path(args.images),
        paths.results_dir(), paths.ngl_dir() / "tenv3",
    )
    _WRITTEN[0] += len(rows) + len(group_rows) + len(sweep)
    print(f"{len(rows)} stream rows, {len(group_rows)} group rows, "
          f"{len(sweep)} sweep rows")
    return 0


BLAS_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
)

_COMMANDS = {
    "ngl-reduce": cmd_ngl_reduce,
    "ngl-fits": cmd_ngl_fits,
    "ngl-exact": cmd_ngl_exact,
    "ngl-rank": cmd_ngl_rank,
    "isd-reduce": cmd_isd_reduce,
    "isd-fits": cmd_isd_fits,
    "isd-exact": cmd_isd_exact,
    "normals-fetch": cmd_normals_fetch,
    "isd-normals": cmd_isd_normals,
    "filters-ngl": cmd_filters_ngl,
    "filters-isd": cmd_filters_isd,
    "throughput": cmd_throughput,
    "gemm": cmd_gemm,
    "stream-serve": cmd_stream_serve,
    "stream-replay": cmd_stream_replay,
    "stream-track": cmd_stream_track,
    "leg5-tables": cmd_leg5_tables,
}


def main(argv: list[str] | None = None) -> int:
    """Run one subcommand; returns its exit code (1 when an exactness
    gate fails on the tolerance, 0 when it only reports undersampled
    stations)."""
    parser = argparse.ArgumentParser(
        prog="image_showcase", description=__doc__,
    )
    parser.add_argument("command", choices=sorted(_COMMANDS))
    parser.add_argument("--images", default=None,
                        help="the image tree; default <data>/images/ngl "
                             "for the ngl commands and "
                             "<data>/images/isd for the isd ones")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--day", type=int, default=1,
                        help="first day index from 1 January, for gemm")
    parser.add_argument("--days", default="1,7,30",
                        help="gemm: day counts to concatenate into one "
                             "batch, comma separated")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stations", default=None,
                        help="comma-separated station ids")
    parser.add_argument("--fields", default="TMP")
    parser.add_argument("--field", default="east",
                        help="stream: the single component to replay "
                             "or track")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=ngl.TENV3_CHUNK)
    parser.add_argument("--block", type=float, default=1.0,
                        help="block length in the data's own time unit")
    parser.add_argument("--order", type=int,
                        default=ngl_reduce.BLOCK_ORDER,
                        help="stream-track: the block image order")
    parser.add_argument("--span", type=float, default=40.0,
                        help="stream-track: the block stream's domain "
                             "length, in the data's own time unit")
    parser.add_argument("--with-days", action="store_true")
    parser.add_argument("--no-raw", action="store_true",
                        help="ngl-rank: rank from the images alone, "
                             "touching no station file")
    parser.add_argument("--tol", type=float,
                        default=compare.EXACTNESS_TOL)
    parser.add_argument("--model", default="ngl-trend",
                        choices=sorted(STREAM_MODELS),
                        help="stream-serve: the model to fit on each "
                             "assembled image; it must be one the "
                             "streamed images' order can carry")
    parser.add_argument("--config", default="detection",
                        help="stream-track: detection, tracking or "
                             "isd-diurnal")
    parser.add_argument("--station", default="",
                        help="stream-track: the label on the images it "
                             "sends back")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--port-file", default=None)
    parser.add_argument("--expect", type=int, default=None)
    parser.add_argument("--rate", type=float, default=None,
                        help="stream-replay: samples per second; "
                             "unbounded when absent")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", default=None,
                        help="stream: also write the JSON summary here")
    # An underscore, not a dash: argparse would read "-pi" as an
    # option, not as this option's value.
    parser.add_argument("--suffix", default="",
                        help="appended to the output name, e.g. _pi")
    args = parser.parse_args(argv)
    if args.images is None:
        base = paths.images_dir()
        args.images = str(
            base / "ngl" if args.command.startswith("ngl")
            else base / "isd" if args.command.startswith("isd")
            else base
        )
    for name in BLAS_THREAD_VARS:
        os.environ.setdefault(name, "1")
    _WRITTEN[0] = 0
    started = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        code = _COMMANDS[args.command](args)
    seconds = round(time.perf_counter() - started, 3)
    write_table(
        _out(RESULTS[args.command] + "_timing", args.suffix),
        [{"host": throughput.machine_row()["host"],
          "command": args.command, "rows": _WRITTEN[0],
          "seconds": seconds}],
        TIMING_COLUMNS,
    )
    print(f"{args.command} took {seconds}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
