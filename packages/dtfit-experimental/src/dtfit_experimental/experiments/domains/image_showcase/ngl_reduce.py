"""The NGL reduction: one pass over a station file produces its whole-span
image per component, one image per segment between database steps, and a
stream of yearly block images.

The file is read twice. The first read keeps only the deduplicated epochs
(95 kB at the longest station), which fixes the domains, the segment
boundaries and every Legendre order; the second read projects. Both reads
apply the same deduplication, so the row indices of the second read match
the epoch vector of the first.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dtfit.image import Image, ImageStream

from . import ngl
from .compare import legendre_order
from .store import image_nbytes, save_images

NGL_VAR = "t"
NGL_EXPR = (
    "c + v*t + a1*cos(2*pi*t) + b1*sin(2*pi*t)"
    " + a2*cos(4*pi*t) + b2*sin(4*pi*t)"
)
# The canonical (sorted) parameter order dtfit uses for a symbolic model.
NGL_NAMES = ["a1", "a2", "b1", "b2", "c", "v"]
BLOCK_ORDER = 12
MIN_EPOCHS = 24

REDUCE_COLUMNS = [
    "sta", "n", "span", "order", "n_segments", "skipped_segments",
    "n_blocks", "dropped_blocks", "n_flags", "coef_bytes", "gram_bytes",
    "grid_bytes", "raw_bytes", "npz_bytes", "seconds", "error",
]


def ngl_design(t: np.ndarray) -> np.ndarray:
    """The model's columns at ``t`` (years from the span's start), ordered
    like :data:`NGL_NAMES`: the reference design for the raw least
    squares."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.cos(2 * np.pi * t), np.cos(4 * np.pi * t),
        np.sin(2 * np.pi * t), np.sin(4 * np.pi * t),
        np.ones_like(t), t,
    ])


def segment_bounds(
    tc: np.ndarray,
    step_offsets: Sequence[float],
    *,
    min_span: float = 1.0,
    min_epochs: int = 200,
) -> tuple[list[tuple[int, int]], int]:
    """Half-open index ranges of the segments between steps.

    Args:
        tc: Increasing epochs measured from the span's start (years).
        step_offsets: Step times on the same scale; a step splits at the
            first epoch on or after it, and a step outside ``tc``'s range
            splits nothing.
        min_span: Segments shorter than this (years) are dropped.
        min_epochs: Segments with fewer epochs are dropped.

    Returns:
        ``(segments, skipped)``: the kept ranges in time order and the
        number of candidate segments dropped by either floor. A station
        with no step yields one segment covering everything.
    """
    tc = np.asarray(tc, dtype=float)
    cuts = sorted({
        int(np.searchsorted(tc, float(s), side="left"))
        for s in step_offsets
        if tc.size and tc[0] <= float(s) <= tc[-1]
    })
    edges = sorted({0, *cuts, int(tc.size)})
    kept: list[tuple[int, int]] = []
    skipped = 0
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 - i0 >= min_epochs and tc[i1 - 1] - tc[i0] >= min_span:
            kept.append((i0, i1))
        else:
            skipped += 1
    return kept, skipped


@dataclass
class StationReduction:
    """One station's images and the facts a table row needs.

    ``images`` keys are ``whole_<component>``, ``seg<kk>_<component>`` and
    ``blk<kk>_<component>``, the block index being the block's own domain
    divided by the block length; ``info`` carries the station name, the
    epoch count, the span, the whole-span order, the block length, the
    segment descriptions, the per-component drift flags (each
    ``[block index, t0, t1]``), the dropped-block counts and ``offsets``,
    the three integer metre columns :mod:`ngl` removed.
    """

    sta: str
    images: dict[str, Image] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def reduce_station(
    path: Any,
    step_years: Sequence[float] = (),
    *,
    chunk: int = ngl.TENV3_CHUNK,
    block_order: int = BLOCK_ORDER,
    block_len: float = 1.0,
    with_segments: bool = True,
) -> StationReduction:
    """Reduce one ``.tenv3`` file to its images.

    Args:
        path: The station file.
        step_years: Step dates in decimal years (absolute, as
            :func:`ngl.decimal_year` returns them); converted to the
            station's own origin here.
        chunk: Rows per read block.
        block_order: Legendre order of the yearly block images.
        block_len: Block length in years.
        with_segments: Build the between-steps segment images as well.
            A station whose steps leave a single segment gets none: that
            segment is the whole span, already imaged.

    Returns:
        A :class:`StationReduction`.

    Raises:
        ValueError: fewer than ``MIN_EPOCHS`` epochs after deduplication,
            or a span of zero.
    """
    path = Path(path)
    t_all = ngl.read_epochs(path, chunk)
    if t_all.size < MIN_EPOCHS:
        raise ValueError(
            f"{path.name}: {t_all.size} epochs, need {MIN_EPOCHS}"
        )
    t0 = float(t_all[0])
    tc = t_all - t0
    span = float(tc[-1])
    if not span > 0.0:
        raise ValueError(f"{path.name}: the epochs span no interval")
    order = legendre_order(span, tc.size)
    domain = (0.0, span)

    whole = ImageStream(
        "legendre", order, domain=domain, grid="explicit", channels=3
    )
    segs: list[tuple[int, int]] = []
    skipped = 0
    seg_streams: list[ImageStream] = []
    if with_segments:
        segs, skipped = segment_bounds(
            tc, [float(s) - t0 for s in step_years]
        )
        if len(segs) <= 1:
            # One segment is the whole span again; storing it twice buys
            # nothing, so the station keeps only its whole-span images.
            segs = []
        for i0, i1 in segs:
            seg_streams.append(ImageStream(
                "legendre",
                legendre_order(float(tc[i1 - 1] - tc[i0]), i1 - i0),
                domain=(float(tc[i0]), float(tc[i1 - 1])),
                grid="explicit", channels=3,
            ))
    # keep_fine above the block count: the default folds the oldest
    # blocks into coarse ones, which would destroy the per-year images.
    keep = int(span / max(block_len, 1e-9)) + 2
    blocks = {
        comp: ImageStream(
            "legendre", block_order, domain=domain, block=float(block_len),
            detect="previous", grid="explicit", keep_fine=keep, fold=keep,
        )
        for comp in ngl.COMPONENTS
    }

    pos = 0
    offsets = {comp: 0.0 for comp in ngl.COMPONENTS}
    for c in ngl.read_tenv3(path, chunk):
        m = int(c.t.size)
        offsets = {"east": c.offset_e, "north": c.offset_n,
                   "up": c.offset_u}
        x = c.t - t0
        Y = np.column_stack([c.east, c.north, c.up])
        whole.update(x, Y)
        for (i0, i1), stream in zip(segs, seg_streams):
            a, b = max(i0, pos), min(i1, pos + m)
            if b > a:
                stream.update(x[a - pos:b - pos], Y[a - pos:b - pos])
        for k, comp in enumerate(ngl.COMPONENTS):
            blocks[comp].update(x, Y[:, k])
        pos += m

    images: dict[str, Image] = {}
    for k, comp in enumerate(ngl.COMPONENTS):
        images[f"whole_{comp}"] = whole.image(k)
    seg_info: list[dict[str, Any]] = []
    for j, ((i0, i1), stream) in enumerate(zip(segs, seg_streams)):
        for k, comp in enumerate(ngl.COMPONENTS):
            images[f"seg{j:02d}_{comp}"] = stream.image(k)
        seg_info.append({
            "index": j, "i0": i0, "i1": i1,
            "t0": float(tc[i0]), "t1": float(tc[i1 - 1]),
            "span": float(tc[i1 - 1] - tc[i0]), "n": int(i1 - i0),
            "order": stream.order,
        })
    flags: dict[str, list[list[float]]] = {}
    dropped: dict[str, int] = {}
    n_blocks = 0
    for comp in ngl.COMPONENTS:
        stream = blocks[comp]
        stream.close()
        found = stream.blocks(domain[0], domain[1])
        for img in found:
            # The key is the block's own domain, not its position in the
            # list: ImageStream also advances its internal index for the
            # blocks it drops, so a station with a gap year would otherwise
            # store an index that no longer matches the drift flags or the
            # wire header.
            j = int(round(float(img.domain[0]) / block_len))
            images[f"blk{j:02d}_{comp}"] = img
        n_blocks = max(n_blocks, len(found))
        flags[comp] = [
            [float(round(float(d0) / block_len)), float(d0), float(d1)]
            for _i, (d0, d1) in stream.flags_
        ]
        dropped[comp] = int(stream.dropped_)

    info = {
        "sta": path.stem, "n": int(tc.size), "span": span, "t0": t0,
        "order": order, "block_order": block_order,
        "block_len": float(block_len),
        "n_segments": len(segs), "skipped_segments": skipped,
        "segments": seg_info, "n_blocks": n_blocks, "flags": flags,
        "dropped": dropped, "offsets": offsets,
    }
    return StationReduction(sta=path.stem, images=images, info=info)


def reduce_to_file(
    path: Any, out_dir: Any, step_years: Sequence[float] = (), **kwargs: Any
) -> tuple[Path, StationReduction]:
    """Reduce one station and write ``<out_dir>/<STA>.npz``; returns the
    path written and the reduction."""
    red = reduce_station(path, step_years, **kwargs)
    out = save_images(
        Path(out_dir) / f"{red.sta}.npz", red.images, red.info
    )
    return out, red


def _reduce_one(
    args: tuple[str, str, list[float], dict[str, Any]]
) -> dict[str, Any]:
    """Worker: reduce one station to a row. Never raises; a failure is
    reported in the row's ``error`` field so a pool run survives it."""
    path_s, out_s, steps, kwargs = args
    path = Path(path_s)
    row: dict[str, Any] = {c: "" for c in REDUCE_COLUMNS}
    row["sta"] = path.stem
    row["raw_bytes"] = path.stat().st_size
    started = time.perf_counter()
    try:
        npz, red = reduce_to_file(path, out_s, steps, **kwargs)
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
    info = red.info
    row.update({
        "n": info["n"], "span": round(info["span"], 4),
        "order": info["order"], "n_segments": info["n_segments"],
        "skipped_segments": info["skipped_segments"],
        "n_blocks": info["n_blocks"],
        "dropped_blocks": sum(info["dropped"].values()),
        "n_flags": sum(len(v) for v in info["flags"].values()),
        "coef_bytes": coef, "gram_bytes": gram, "grid_bytes": grid,
        "npz_bytes": npz.stat().st_size,
        "seconds": round(time.perf_counter() - started, 3), "error": "",
    })
    return row


def reduce_many(
    paths: Sequence[Any],
    out_dir: Any,
    steps_by_station: Mapping[str, Sequence[float]],
    *,
    workers: int = 1,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Reduce many stations, optionally across processes.

    Args:
        paths: Station files.
        out_dir: Where the per-station ``.npz`` files go; created.
        steps_by_station: Step decimal years per station name; a station
            with no entry is reduced without segments beyond the whole
            span.
        workers: 1 runs in this process; more uses a
            :class:`ProcessPoolExecutor` of that size.
        kwargs: Forwarded to :func:`reduce_station`.

    Returns:
        One row per station, in input order, with the columns of
        :data:`REDUCE_COLUMNS`; a station that failed carries the message
        in ``error`` and empty measurements.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(p), str(out_dir),
         [float(s) for s in steps_by_station.get(Path(p).stem, ())], kwargs)
        for p in paths
    ]
    if workers <= 1:
        return [_reduce_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_reduce_one, jobs, chunksize=8))
