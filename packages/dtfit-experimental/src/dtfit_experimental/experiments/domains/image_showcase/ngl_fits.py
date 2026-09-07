"""Fits from the stored NGL images: the velocity against MIDAS, the
exactness against the raw least squares, and the model ranking by the
image's BIC.

Everything except :func:`exactness_rows` and the raw half of
:func:`ranking_rows` reads images alone, so the Pi runs the same code on
the shipped ``.npz`` files without the station files.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dtfit.image import Image, coverage

from . import ngl
from .compare import (
    COVERAGE_TOL, EXACTNESS_TOL, fit_from_image, gram_rebuild_error,
    param_scores, raw_bic, raw_lstsq,
)
from .ngl_reduce import NGL_EXPR, NGL_NAMES, ngl_design
from .store import load_images


def _design_trend(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.column_stack([np.ones_like(t), t])


def _design_trend_annual(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
        np.ones_like(t), t,
    ])


def _design_trend_quad(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.cos(2 * np.pi * t), np.cos(4 * np.pi * t),
        np.sin(2 * np.pi * t), np.sin(4 * np.pi * t),
        np.ones_like(t), t * t, t,
    ])


@dataclass(frozen=True)
class CatalogModel:
    """One alternative in the ranking.

    ``key`` names it, ``expr`` is the symbolic form the image fit uses,
    ``names`` are its parameters in canonical (sorted) order, and
    ``design`` builds its columns in that same order for the raw
    least-squares arm. The two arms are independent by construction: the
    image side solves from ``S`` and ``G``, the raw side from the samples.
    """

    key: str
    expr: str
    names: tuple[str, ...]
    design: Callable[[np.ndarray], np.ndarray]


MODELS = [
    CatalogModel("trend", "c + v*t", ("c", "v"), _design_trend),
    CatalogModel(
        "trend_annual",
        "c + v*t + a1*cos(2*pi*t) + b1*sin(2*pi*t)",
        ("a1", "b1", "c", "v"),
        _design_trend_annual,
    ),
    CatalogModel(
        "trend_annual_semi", NGL_EXPR, tuple(NGL_NAMES), ngl_design
    ),
    CatalogModel(
        "trend_quad_annual_semi",
        "c + v*t + q*t**2 + a1*cos(2*pi*t) + b1*sin(2*pi*t)"
        " + a2*cos(4*pi*t) + b2*sin(4*pi*t)",
        ("a1", "a2", "b1", "b2", "c", "q", "v"),
        _design_trend_quad,
    ),
]

FIT_COLUMNS = [
    "sta", "kind", "index", "component", "n", "span", "order",
    *NGL_NAMES, *[f"stderr_{n}" for n in NGL_NAMES],
    "rss", "bic", "converged",
    "midas_v", "midas_sigma", "midas_span", "midas_steps",
    "diff", "diff_over_sigma", "within_sigma",
]
EXACT_COLUMNS = [
    "sta", "component", "n", "span", "order", "score", "coverage",
    "gram_rebuild_err", "gate", "worst_param", "v_image", "v_raw",
]
RANK_COLUMNS = [
    "sta", "component", "model", "n_params", "bic_image", "bic_raw",
    "rss_image", "rss_raw", "rank_image", "rank_raw",
]

_MIDAS_FIELDS = {
    "east": ("ve", "se"), "north": ("vn", "sn"), "up": ("vu", "su"),
}


def _row_from_result(res: Any, image: Image) -> dict[str, Any]:
    """The parameter half of a fit row: values, standard errors, fit
    quality. Missing standard errors (no covariance) become empty."""
    params = res.params
    try:
        errs = res.stderr()
    except Exception:                            # no covariance
        errs = {}
    row: dict[str, Any] = {
        "n": int(image.n),
        "span": round(float(image.domain[1] - image.domain[0]), 6),
        "order": int(image.order),
        "rss": None if res.rss is None else float(res.rss),
        "bic": None if res.bic is None else float(res.bic),
        "converged": bool(res.converged),
    }
    for name in NGL_NAMES:
        row[name] = float(params[name]) if name in params else None
        row[f"stderr_{name}"] = (
            float(errs[name]) if name in errs else None
        )
    return row


def _midas_columns(
    component: str, velocity: float, entry: Any
) -> dict[str, Any]:
    """The MIDAS comparison columns for one component, or empty ones when
    the station has no MIDAS entry."""
    if entry is None:
        return {k: None for k in (
            "midas_v", "midas_sigma", "midas_span", "midas_steps", "diff",
            "diff_over_sigma", "within_sigma",
        )}
    vfield, sfield = _MIDAS_FIELDS[component]
    mv = float(getattr(entry, vfield))
    ms = float(getattr(entry, sfield))
    diff = float(velocity) - mv
    return {
        "midas_v": mv, "midas_sigma": ms,
        "midas_span": float(getattr(entry, "span", float("nan"))),
        "midas_steps": int(getattr(entry, "n_steps", 0)),
        "diff": diff,
        "diff_over_sigma": diff / ms if ms > 0 else None,
        "within_sigma": bool(ms > 0 and abs(diff) <= ms),
    }


def station_rows(
    npz_path: Any, midas: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Fit every stored whole-span and segment image of one station.

    Returns one row per image (``kind`` ``"whole"`` or ``"seg"``) plus, for
    each component with segments, an epoch-count-weighted mean row
    (``kind`` ``"segmean"``, the steps-aware velocity MIDAS is comparable
    to). The MIDAS columns are filled on the ``"whole"`` and ``"segmean"``
    rows when the station has an entry; block images are not fitted.
    """
    images, info = load_images(npz_path)
    sta = str(info.get("sta", Path(npz_path).stem))
    entry = (midas or {}).get(sta)
    rows: list[dict[str, Any]] = []
    seg_rows: dict[str, list[dict[str, Any]]] = {}
    for name, image in images.items():
        if name.startswith("blk"):
            continue
        head, comp = name.split("_", 1)
        kind = "whole" if head == "whole" else "seg"
        index = 0 if kind == "whole" else int(head[3:])
        res = fit_from_image(NGL_EXPR, image, NGL_NAMES)
        row = {"sta": sta, "kind": kind, "index": index,
               "component": comp}
        row.update(_row_from_result(res, image))
        row.update(_midas_columns(
            comp, row["v"], entry if kind == "whole" else None
        ))
        rows.append(row)
        if kind == "seg":
            seg_rows.setdefault(comp, []).append(row)
    for comp, group in seg_rows.items():
        w = np.array([r["n"] for r in group], dtype=float)
        mean = {"sta": sta, "kind": "segmean", "index": -1,
                "component": comp, "n": int(w.sum()),
                "span": float(sum(r["span"] for r in group)),
                "order": max(int(r["order"]) for r in group),
                "rss": None, "bic": None, "converged": True}
        for name in NGL_NAMES:
            vals = np.array([r[name] for r in group], dtype=float)
            mean[name] = float(w @ vals / w.sum())
            mean[f"stderr_{name}"] = None
        mean.update(_midas_columns(comp, mean["v"], entry))
        rows.append(mean)
    return rows


def exactness_rows(
    tenv3_path: Any, npz_path: Any, chunk: int = ngl.TENV3_CHUNK
) -> list[dict[str, Any]]:
    """The gate: the whole-span image fit against the raw least squares on
    the same rows, one row per component.

    Reads the station file again (this runs on the machine that holds the
    samples, never on the Pi). Each row carries three verdicts' worth of
    evidence:

    - ``score``, the worst relative parameter difference
      (:func:`compare.param_scores`), against :data:`compare.EXACTNESS_TOL`;
    - ``coverage``, :func:`dtfit.image.coverage` of the fitted model on
      this image, against :data:`compare.COVERAGE_TOL`;
    - ``gram_rebuild_err``, how far ``G`` rebuilt from the grid is from
      the accumulated ``G``, which is what shipping ``S`` and the grid
      alone would cost.

    ``gate`` is ``"UNDERSAMPLED"`` when the coverage is above tolerance
    (the image's order cannot represent the model on this station's
    sampling: reported and counted, never a failure of the run),
    ``"FAIL"`` when the score misses the gate, and ``"ok"`` otherwise.
    """
    images, info = load_images(npz_path)
    parts: dict[str, list[np.ndarray]] = {c: [] for c in ngl.COMPONENTS}
    times: list[np.ndarray] = []
    for c in ngl.read_tenv3(tenv3_path, chunk):
        times.append(c.t)
        parts["east"].append(c.east)
        parts["north"].append(c.north)
        parts["up"].append(c.up)
    t = np.concatenate(times)
    tc = t - t[0]
    design = ngl_design(tc)
    out: list[dict[str, Any]] = []
    for comp in ngl.COMPONENTS:
        y = np.concatenate(parts[comp])
        image = images[f"whole_{comp}"]
        ref = raw_lstsq(design, y, NGL_NAMES)
        res = fit_from_image(NGL_EXPR, image, NGL_NAMES)
        got = res.params
        scores = param_scores(got, ref)
        score = max(scores.values(), default=0.0)
        cover = float(coverage(
            NGL_EXPR, [float(got[k]) for k in NGL_NAMES], image, var="t"
        ))
        if cover > COVERAGE_TOL:
            gate = "UNDERSAMPLED"
        elif score <= EXACTNESS_TOL:
            gate = "ok"
        else:
            gate = "FAIL"
        out.append({
            "sta": str(info.get("sta", Path(npz_path).stem)),
            "component": comp, "n": int(image.n),
            "span": round(float(image.domain[1]), 6),
            "order": int(image.order), "score": float(score),
            "coverage": cover,
            "gram_rebuild_err": gram_rebuild_error(image),
            "gate": gate,
            "worst_param": (
                max(scores, key=lambda k: scores[k]) if scores else ""
            ),
            "v_image": float(got["v"]), "v_raw": float(ref["v"]),
        })
    return out


def ranking_rows(
    npz_path: Any, tenv3_path: Any = None, chunk: int = ngl.TENV3_CHUNK
) -> list[dict[str, Any]]:
    """Rank :data:`MODELS` on the whole-span image of each component by the
    image's BIC, and, when ``tenv3_path`` is given, by the BIC of an
    ordinary least-squares fit of each model's own design matrix to the
    raw samples.

    The two arms share no code: the image arm solves from ``S`` and ``G``
    through :func:`dtfit.image.fit`, the raw arm through
    :func:`numpy.linalg.lstsq` on the samples with
    ``n_obs = t.size``. That is what makes "the ranking from the image
    agrees with the ranking from the data" a measurement rather than an
    identity. ``bic_raw``, ``rss_raw`` and ``rank_raw`` are empty without
    the samples, which is how the Pi runs it.
    """
    images, info = load_images(npz_path)
    sta = str(info.get("sta", Path(npz_path).stem))
    samples: dict[str, np.ndarray] = {}
    tc = np.zeros(0)
    if tenv3_path is not None:
        parts: dict[str, list[np.ndarray]] = {
            c: [] for c in ngl.COMPONENTS
        }
        times: list[np.ndarray] = []
        for c in ngl.read_tenv3(tenv3_path, chunk):
            times.append(c.t)
            for comp in ngl.COMPONENTS:
                parts[comp].append(getattr(c, comp))
        t = np.concatenate(times)
        tc = t - t[0]
        for comp in ngl.COMPONENTS:
            samples[comp] = np.concatenate(parts[comp])
    rows: list[dict[str, Any]] = []
    for comp in ngl.COMPONENTS:
        image = images[f"whole_{comp}"]
        group: list[dict[str, Any]] = []
        for model in MODELS:
            res_i = fit_from_image(model.expr, image, model.names)
            rss_raw: float | None = None
            bic_raw: float | None = None
            if comp in samples:
                rss_raw, bic_raw = raw_bic(
                    model.design(tc), samples[comp]
                )
            group.append({
                "sta": sta, "component": comp, "model": model.key,
                "n_params": len(model.names),
                "bic_image": (
                    None if res_i.bic is None else float(res_i.bic)
                ),
                "bic_raw": bic_raw,
                "rss_image": (
                    None if res_i.rss is None else float(res_i.rss)
                ),
                "rss_raw": rss_raw,
                "rank_image": None, "rank_raw": None,
            })
        for key in ("image", "raw"):
            vals = [r[f"bic_{key}"] for r in group]
            if any(v is None for v in vals):
                continue
            order = np.argsort(np.array(vals, dtype=float))
            for rank, idx in enumerate(order):
                group[int(idx)][f"rank_{key}"] = rank
        rows.extend(group)
    return rows


def _fits_one(args: tuple[str, Any]) -> list[dict[str, Any]]:
    npz_path, entry = args
    midas = {} if entry is None else {Path(npz_path).stem: entry}
    return station_rows(npz_path, midas)


def _exact_one(args: tuple[str, str]) -> list[dict[str, Any]]:
    src, npz_path = args
    try:
        return exactness_rows(src, npz_path)
    except Exception as exc:                     # keep the batch alive
        return [{"sta": Path(npz_path).stem, "component": "", "n": 0,
                 "span": 0.0, "order": 0, "score": float("inf"),
                 "coverage": None, "gram_rebuild_err": None,
                 "gate": f"ERROR: {type(exc).__name__}: {exc}",
                 "worst_param": "", "v_image": None, "v_raw": None}]


def _rank_one(args: tuple[str, str | None]) -> list[dict[str, Any]]:
    npz_path, src = args
    return ranking_rows(npz_path, src)


def _fan_out(func: Any, jobs: Sequence[Any], workers: int) -> list[Any]:
    """Map ``func`` over ``jobs``, flattening the per-job lists and
    keeping the input order."""
    if workers <= 1:
        results = [func(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(func, jobs, chunksize=8))
    return [row for group in results for row in group]


def run_fits(
    npz_paths: Sequence[Any], midas: Mapping[str, Any], *, workers: int = 1
) -> list[dict[str, Any]]:
    """:func:`station_rows` over many stations, in input order.

    Each job carries only its own station's MIDAS entry (keyed by the
    ``.npz`` stem, which is always the station code), not a copy of the
    whole table: with 21798 MIDAS entries, embedding the full mapping in
    every job would pickle gigabytes across a multi-worker pool.
    """
    jobs = [(str(p), midas.get(Path(p).stem)) for p in npz_paths]
    return _fan_out(_fits_one, jobs, workers)


def run_exactness(
    pairs: Sequence[tuple[Any, Any]], *, workers: int = 1
) -> list[dict[str, Any]]:
    """:func:`exactness_rows` over many ``(tenv3, npz)`` pairs; a station
    that raises contributes one row whose ``gate`` starts with
    ``ERROR``."""
    jobs = [(str(a), str(b)) for a, b in pairs]
    return _fan_out(_exact_one, jobs, workers)


def run_rankings(
    pairs: Sequence[tuple[Any, Any]], *, workers: int = 1
) -> list[dict[str, Any]]:
    """:func:`ranking_rows` over many ``(npz, tenv3 or None)`` pairs."""
    jobs = [(str(a), None if b is None else str(b)) for a, b in pairs]
    return _fan_out(_rank_one, jobs, workers)
