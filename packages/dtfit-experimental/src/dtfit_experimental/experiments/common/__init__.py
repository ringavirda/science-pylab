"""Shared helpers for the dtfit experiment suite.

Every experiment is a notebook over a sibling ``backend.py``. This package
holds what those backends have in common: metrics, baselines, dataset loaders,
the few plotting helpers the notebooks import directly (``fit_overlay`` and
friends), and the ``EXPERIMENTS_DIR`` anchor that locates the bundled data.
Pure compute only; the notebooks own the presentation.
"""

from pathlib import Path

from .metrics import metrics, mse, mae, timed, md_table, fmt
from . import plotting
from . import baselines
from . import datasets

# ``.../dtfit_experimental/experiments``; its ``data/`` holds the CSVs.
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent

__all__ = [
    "metrics", "mse", "mae", "timed", "md_table", "fmt",
    "EXPERIMENTS_DIR",
    "plotting", "baselines", "datasets",
]
