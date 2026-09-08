"""The image showcase's compute surface: the names the notebook and the
paper scripts import.

``image_showcase.ipynb`` does ``from ... import backend as B`` and reads
the CSV files in ``results/``; the modules behind this one hold the
implementation (``ngl``, ``ngl_reduce``, ``ngl_fits``, ``isd``,
``isd_reduce``, ``isd_fits``, ``filters``, ``throughput``, ``stream``,
``store``, ``compare``, ``paths``).
"""

from __future__ import annotations

from . import (  # noqa: F401  (re-exported for the notebook)
    compare, filters, isd, isd_fits, isd_reduce, ngl, ngl_fits,
    ngl_reduce, paths, store, stream, throughput,
)
from .compare import (
    COVERAGE_TOL, EXACTNESS_TOL, fit_from_image, gram_rebuild_error,
    legendre_order, p0_from_image, param_score, raw_bic, raw_lstsq,
    worst_param,
)
from .filters import (
    ISD_CONFIG, NGL_CONFIGS, event_window_share, isd_filter_station,
    match_flags, ngl_filter_station, ratio_bin, reachable_events,
    run_filter, run_isd_filters, run_ngl_filters,
)
from .isd_fits import (
    amp_phase, day_rows, exactness_year, normals_rows, run_normals,
    run_year_fits, year_rows,
)
from .isd_reduce import (
    ANNUAL_EXPR, ANNUAL_NAMES, ANNUAL_ORDER, DIURNAL_EXPR, DIURNAL_ORDER,
    day_batch, reduce_many_years, reduce_station_year, reduce_year_to_file,
)
from .ngl_fits import (
    MODELS, exactness_rows, ranking_rows, run_exactness, run_fits,
    run_rankings, station_rows,
)
from .ngl_reduce import (
    NGL_EXPR, NGL_NAMES, ngl_design, reduce_many, reduce_station,
    reduce_to_file, segment_bounds,
)
from .paths import (
    data_root, figures_dir, images_dir, isd_dir, ngl_dir, normals_dir,
    results_dir,
)
from .store import image_nbytes, load_images, save_images, write_table
from .stream import (
    image_digest, image_from_frame, image_header, produce, recv_frame,
    replay, send_image, send_samples, serve, track,
)
from .throughput import (
    channel_gemm_rate, disk_read_rate, float32_error, gpu_probe,
    machine_row, peak_rss_mib, reduce_rate,
)

__all__ = [
    # the modules
    "compare", "filters", "isd", "isd_fits", "isd_reduce", "ngl",
    "ngl_fits", "ngl_reduce", "paths", "store", "stream", "throughput",
    # comparison and storage
    "EXACTNESS_TOL", "COVERAGE_TOL", "fit_from_image", "legendre_order",
    "p0_from_image", "param_score", "worst_param", "raw_lstsq",
    "raw_bic", "gram_rebuild_error", "image_nbytes", "load_images",
    "save_images", "write_table",
    # NGL
    "NGL_EXPR", "NGL_NAMES", "ngl_design", "reduce_station",
    "reduce_to_file", "reduce_many", "segment_bounds", "station_rows",
    "exactness_rows", "ranking_rows", "run_fits", "run_exactness",
    "run_rankings", "MODELS",
    # NOAA
    "ANNUAL_EXPR", "ANNUAL_NAMES", "ANNUAL_ORDER", "DIURNAL_EXPR",
    "DIURNAL_ORDER", "reduce_station_year", "reduce_year_to_file",
    "reduce_many_years", "day_batch", "year_rows", "day_rows",
    "normals_rows", "exactness_year", "amp_phase", "run_year_fits",
    "run_normals",
    # filters
    "NGL_CONFIGS", "ISD_CONFIG", "run_filter", "match_flags",
    "reachable_events", "event_window_share", "ratio_bin",
    "ngl_filter_station", "isd_filter_station", "run_ngl_filters",
    "run_isd_filters",
    # measurements and the wire
    "machine_row", "gpu_probe", "peak_rss_mib", "disk_read_rate",
    "float32_error", "channel_gemm_rate", "reduce_rate", "image_header",
    "image_from_frame", "image_digest", "send_image", "send_samples",
    "recv_frame", "produce", "serve", "replay", "track",
    # paths
    "data_root", "ngl_dir", "isd_dir", "normals_dir", "images_dir",
    "results_dir", "figures_dir",
]
