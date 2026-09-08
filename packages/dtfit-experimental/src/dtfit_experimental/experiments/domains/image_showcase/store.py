"""The image container on disk and the CSV writer every run uses.

One ``.npz`` holds every image of one station (or station-year) under a
name: ``S``, ``G``, the explicit grid positions and any weights as arrays,
every scalar as one JSON string. The images a run ships to the Pi are
exactly these files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dtfit.image import Grid, Image, make_basis

_META = "__meta__"


def save_images(
    path: Any,
    images: Mapping[str, Image],
    info: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``images`` to ``path`` as one compressed ``.npz``.

    Args:
        path: Destination file; its parent directory is created.
        images: ``{name: Image}``; the names are the keys
            :func:`load_images` returns, in this order.
        info: Station-level facts (counts, spans, flags) stored alongside;
            any JSON-serializable mapping. ``None`` stores ``{}``.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"images": {}, "info": dict(info or {})}
    for name, img in images.items():
        meta["images"][name] = {
            "basis": img.basis.to_dict(),
            "domain": [float(img.domain[0]), float(img.domain[1])],
            "n": int(img.n), "sumsq": float(img.sumsq),
            "sumy": float(img.sumy), "wsum": float(img.wsum),
            "robust": bool(img.robust),
            "grid": {"kind": img.grid.kind, "n": int(img.grid.n),
                     "x0": float(img.grid.x0), "x1": float(img.grid.x1)},
            "has_w": img.w is not None,
        }
        arrays[f"{name}__S"] = np.asarray(img.S, dtype=float)
        arrays[f"{name}__G"] = np.asarray(img.G, dtype=float)
        if img.grid.kind == "explicit":
            arrays[f"{name}__x"] = img.grid.positions()
        if img.w is not None:
            arrays[f"{name}__w"] = np.asarray(img.w, dtype=float)
    arrays[_META] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    return path


def load_images(path: Any) -> tuple[dict[str, Image], dict[str, Any]]:
    """Read back what :func:`save_images` wrote.

    Returns:
        ``(images, info)``: the images keyed by their stored names in
        write order, and the ``info`` mapping (``{}`` when none was
        stored).
    """
    out: dict[str, Image] = {}
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z[_META]))
        for name, m in meta["images"].items():
            g = m["grid"]
            if g["kind"] == "explicit":
                grid = Grid("explicit", int(g["n"]), float(g["x0"]),
                            float(g["x1"]),
                            np.asarray(z[f"{name}__x"], dtype=float))
            else:
                grid = Grid("uniform", int(g["n"]), float(g["x0"]),
                            float(g["x1"]))
            w = (np.asarray(z[f"{name}__w"], dtype=float)
                 if m["has_w"] else None)
            out[name] = Image(
                make_basis(m["basis"]["name"], m["basis"]["order"]),
                (float(m["domain"][0]), float(m["domain"][1])),
                np.asarray(z[f"{name}__S"], dtype=float),
                np.asarray(z[f"{name}__G"], dtype=float),
                int(m["n"]), float(m["sumsq"]), float(m["sumy"]),
                float(m["wsum"]), grid, w, bool(m["robust"]),
            )
    return out, meta["info"]


def image_nbytes(image: Image) -> tuple[int, int, int]:
    """``(coef_bytes, gram_bytes, grid_bytes)`` of one image as float64.

    ``S``, ``G``, and the explicit grid positions plus any weights (0 for
    a uniform grid without weights). The three are reported separately
    because they scale differently: ``S`` as the order, ``G`` as the
    order squared, the grid as the sample count. For NGL, where the order
    follows the span, ``G`` is what makes the image tree the same order of
    size as the raw files.
    """
    coef = int(image.S.size) * 8
    gram = int(image.G.size) * 8
    grid = image.grid.n * 8 if image.grid.kind == "explicit" else 0
    if image.w is not None:
        grid += int(image.w.size) * 8
    return int(coef), int(gram), int(grid)


def write_table(
    path: Any,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> Path:
    """Write ``rows`` as a CSV with a header row.

    Args:
        path: Destination file; its parent directory is created.
        rows: Mappings, one per line; a key missing from a row writes an
            empty field.
        columns: Column order; defaults to the first row's keys. Required
            to write a header for empty ``rows``, which otherwise writes
            an empty file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else (
        list(rows[0]) if rows else []
    )
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        if cols:
            writer.writerow(cols)
            for row in rows:
                writer.writerow(
                    ["" if row.get(c) is None else row[c] for c in cols]
                )
    return path
