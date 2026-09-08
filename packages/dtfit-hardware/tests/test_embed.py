"""Guards for the embedded-LSI codegen: the facts the on-silicon numbers
rest on, neither needing a board.

* The float64 golden reproduces the real ``dtfit.streaming.LSIFilter`` on a
  uniform, drift-free window, making the firmware the dtfit method and not
  a lookalike there; the level-shift and jitter cases below bound how far
  the two part ways once drift fires or the grid stops being uniform.
* The checked-in flash tables match the generator, and every sketch dir carries
  an identical copy, since Arduino needs sketch-local headers. A config change
  therefore cannot ship one sketch stale.
* The compiled C hot path, run over the real BLE sample vector, tracks the
  float64 golden to float32 rounding.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dtfit_hardware.tools import embed_lsi


def test_golden_matches_real_lsi_filter() -> None:
    # The embedded float64 golden tracks the configured LSIFilter to 1e-6 in
    # every parameter on a uniform, drift-free window -- the regime the port
    # targets. See test_golden_diverges_under_level_shift and
    # test_golden_diverges_on_jittered_grid for what happens outside it.
    assert embed_lsi.cross_check() < 1e-6


def test_golden_diverges_under_level_shift() -> None:
    # Outside cross_check's no-drift regime: LSIFilter's jump test fires
    # (cusum_k=inf only disables the two CUSUM arms), the golden has no
    # model of it, and the two do not re-converge. Regression guard on the
    # gap's size, not a bound to shrink here.
    assert embed_lsi.cross_check_level_shift() > 50.0


def test_golden_diverges_on_jittered_grid() -> None:
    # tables() freezes B on a uniform grid; LSIFilter rebuilds it from the
    # window's actual sample times. cross_check's fixed 0.1 s step cannot
    # see this. A 1% jitter, close to the sketches' millis()-gated 1 Hz
    # loop, already clears the 1e-6 bound by more than an order.
    assert embed_lsi.cross_check_jitter(0.01) > 1e-5


def test_c_hot_path_matches_golden() -> None:
    # Compiles and runs the real C header (not a reimplementation) over the
    # checked-in test vector, and checks its float32 output against the
    # float64 golden on the same (t, y). The vector is read back from
    # lsi_testvec.h, the file the C actually compiles, not load_sample(),
    # which falls back to a synthetic ramp on a checkout with no recorded
    # BLE CSV.
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    t, y = embed_lsi.load_testvec()
    p0 = np.array([y[0]] + [0.0] * (embed_lsi.N - 1))
    golden = embed_lsi.golden_run(t, y, p0)

    fw_dir = embed_lsi.FIRMWARE / "nano_lsi_onboard"
    src = embed_lsi.HERE / "test_lsi.cpp"
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "test_lsi"
        subprocess.run(
            ["g++", "-O2", "-ffp-contract=off", "-I", str(fw_dir),
             str(src), "-o", str(exe)],
            check=True,
        )
        out = subprocess.run(
            [str(exe)], capture_output=True, text=True, check=True,
        ).stdout

    rows = [ln.split() for ln in out.splitlines() if ln.startswith("VAL")]
    idx = [int(r[1]) for r in rows]
    c_p = np.array([[float(r[2]), float(r[3])] for r in rows])
    assert np.max(np.abs(golden[idx] - c_p)) < 1e-4


def test_checked_in_tables_match_generator() -> None:
    generated = embed_lsi.render_header()
    for target in embed_lsi.FIRMWARE_TARGETS:
        header = embed_lsi.FIRMWARE / target / "lsi_tables.h"
        assert header.is_file(), f"missing firmware/{target}/lsi_tables.h"
        assert header.read_text(encoding="utf-8") == generated, (
            f"firmware/{target}/lsi_tables.h is stale -- run "
            "`python -m dtfit_hardware.tools.embed_lsi` to regenerate"
        )


def test_shared_headers_are_in_sync_across_sketch_dirs() -> None:
    # dtfit_lsi.h is hand-written C copied into each sketch dir. Let the copies
    # drift and one sketch ships a different filter.
    dirs = [embed_lsi.FIRMWARE / t for t in embed_lsi.FIRMWARE_TARGETS]
    for fname in ("dtfit_lsi.h", "lsi_tables.h"):
        texts = {d.name: (d / fname).read_text(encoding="utf-8")
                 for d in dirs if (d / fname).is_file()}
        assert len(set(texts.values())) == 1, (
            f"{fname} differs across sketch dirs: {sorted(texts)}"
        )


def test_no_block_image_firmware_is_shipped() -> None:
    # Ruling: the embedded tier ships the window (LSI) image only; the block
    # (EAC) image stays host-side (EACFilter / ImageStream over logged
    # fixes). This guards the scope from silently half-drifting into a
    # block firmware path with no board to validate it.
    fw = embed_lsi.FIRMWARE
    for pat in ("*block*", "*eac*", "block_tables.h", "dtfit_block.h"):
        assert not list(fw.rglob(pat)), f"unexpected block firmware: {pat}"


def test_embed_names_no_retired_symbol() -> None:
    # The hardware tier rides the current image core; none of the
    # removed-surface names may reappear in the host glue.
    import dtfit_hardware.compare_real as CR
    import dtfit_hardware.backend as BK
    from pathlib import Path

    retired = (
        "PartitionedLSI", "PartitionedEAC", "project_spectra",
        "fit_lsi_batched", "ensemble_fit", "auto_estimate", "FilterBank",
        "FusedChiSquareDetector", "fit_eac_adaptive", "window_mode",
        "active_ratio", "f_scale", "adapt_r", "adapt_noise",
        "param_cov_", "stderr_",
    )
    for mod in (CR, BK, embed_lsi):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        hits = [s for s in retired if s in text]
        assert not hits, f"{mod.__name__} names retired symbols: {hits}"
