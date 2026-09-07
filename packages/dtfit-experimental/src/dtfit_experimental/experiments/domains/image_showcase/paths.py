"""Default locations of the showcase's inputs and outputs.

Every function in the domain takes its paths as arguments; these are the
defaults only, read from the environment so that no machine's directory
layout is written into the package. ``SHOWCASE_DATA`` (default
``~/data/showcase``) holds the NGL copy, the reduced images and the
normals; ``SHOWCASE_ISD`` (default ``<SHOWCASE_DATA>/isd``) holds the NOAA
year directories, which on the PC live on a second NVMe.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA = "~/data/showcase"


def data_root() -> Path:
    """The data root from ``SHOWCASE_DATA``; not created here."""
    return Path(os.environ.get("SHOWCASE_DATA", DEFAULT_DATA)).expanduser()


def ngl_dir(root: Path | None = None) -> Path:
    """The NGL copy ``<root>/ngl``: ``tenv3/``, ``steps.txt``,
    ``midas.IGS20.txt``, ``DataHoldings.txt``."""
    return (data_root() if root is None else Path(root)) / "ngl"


def isd_dir(year: int | None = None) -> Path:
    """The NOAA copy from ``SHOWCASE_ISD``, or ``<data_root()>/isd``; with
    ``year``, that year's directory of one CSV per station."""
    env = os.environ.get("SHOWCASE_ISD")
    base = Path(env).expanduser() if env else data_root() / "isd"
    return base if year is None else base / str(year)


def normals_dir(root: Path | None = None) -> Path:
    """Where the reduced NOAA hourly normals are stored."""
    return (data_root() if root is None else Path(root)) / "normals"


def images_dir(root: Path | None = None) -> Path:
    """Where the reduced images are written: ``<root>/images``, with
    ``ngl/`` and ``isd/<year>/`` beneath it."""
    return (data_root() if root is None else Path(root)) / "images"


def domain_dir() -> Path:
    """This package's directory inside the repository."""
    return Path(__file__).resolve().parent


def results_dir() -> Path:
    """The tracked CSV output directory, created if missing."""
    p = domain_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir() -> Path:
    """The tracked figure output directory, created if missing."""
    p = domain_dir() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p
