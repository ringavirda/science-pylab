"""Guards for the embedded-LSI codegen: the two facts the on-silicon numbers
rest on, neither needing a board.

* The float64 golden reproduces the real ``dtfit.streaming.LSIFilter``, making
  the firmware the dtfit method rather than a lookalike.
* The checked-in flash tables match the generator, and every sketch dir carries
  an identical copy, since Arduino needs sketch-local headers. A config change
  therefore cannot ship one sketch stale.
"""
from __future__ import annotations

from dtfit_hardware.tools import embed_lsi


def test_golden_matches_real_lsi_filter() -> None:
    # The embedded float64 golden tracks the configured LSIFilter to 1e-6 in
    # every parameter, so the two are the same filter and not merely similar.
    assert embed_lsi.cross_check() < 1e-6


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
