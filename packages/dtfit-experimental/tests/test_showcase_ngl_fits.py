"""NGL fits from stored images: the MIDAS comparison, the exactness rows
and the BIC ranking, all on a synthetic station."""

from __future__ import annotations

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, ngl_fits, ngl_reduce,
)

HEADER = (
    "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0 _e _n0 _n _u0 _u"
    " _ant sig_e sig_n sig_u c_en c_eu c_nu lat lon hgt\n"
)
TRUTH = {
    "east": [0.004, 0.001, -0.002, 0.0005, 0.83, 0.036],
    "north": [-0.003, 0.0008, 0.001, -0.0004, -0.28, 0.058],
    "up": [0.006, 0.002, 0.003, 0.001, 0.84, -0.0007],
}


def write_station(path, t, cols):
    with open(path, "w") as fh:
        fh.write(HEADER)
        for k in range(t.size):
            fh.write(
                f"AAAA 08MAR27 {t[k]:.4f} 54552 1472 4 130.8"
                f" 4781 {cols['east'][k]:.6f} -1378706"
                f" {cols['north'][k]:.6f} 104 {cols['up'][k]:.6f} 0.0000"
                f" 0.001193 0.001135 0.005615 0.047261 0.011975 -0.050796"
                f" -12.4 -229.1 104.8\n"
            )
    return path


def station(tmp_path, name="AAAA", span=6.0, seed=4):
    rng = np.random.default_rng(seed)
    t = 2008.0 + np.arange(0.0, span, 1.0 / 365.25)
    t = np.round(t[rng.random(t.size) > 0.10], 4)
    t = t[np.diff(t, prepend=t[0] - 1.0) > 0]
    tc = t - t[0]
    cols = {
        k: (ngl_reduce.ngl_design(tc) @ np.array(v)
            + 0.002 * rng.standard_normal(tc.size))
        for k, v in TRUTH.items()
    }
    src = write_station(tmp_path / f"{name}.tenv3", t, cols)
    npz, _ = ngl_reduce.reduce_to_file(src, tmp_path / "images")
    return src, npz, t, cols


def test_station_rows_carry_the_parameters_and_the_midas_comparison(tmp_path):
    src, npz, t, cols = station(tmp_path)
    midas = {"AAAA": type("M", (), {
        "ve": 0.0361, "vn": 0.0578, "vu": -0.0005,
        "se": 0.0002, "sn": 0.0003, "su": 0.0009,
        "span": 6.0, "n_steps": 0,
    })()}
    rows = ngl_fits.station_rows(npz, midas)
    whole = [r for r in rows if r["kind"] == "whole"]
    assert {r["component"] for r in whole} == {"east", "north", "up"}
    east = [r for r in whole if r["component"] == "east"][0]
    assert abs(east["v"] - 0.036) < 1e-3
    assert abs(east["a1"] - 0.004) < 1e-3
    assert east["stderr_v"] > 0.0
    assert abs(east["midas_v"] - 0.0361) < 1e-12
    assert abs(east["diff"] - (east["v"] - 0.0361)) < 1e-12
    assert east["diff_over_sigma"] == east["diff"] / 0.0002
    assert np.isfinite(east["bic"]) and east["rss"] > 0.0
    assert set(ngl_fits.FIT_COLUMNS) >= set(east)


def test_station_rows_average_the_segments_by_epoch_count(tmp_path):
    src, npz, t, cols = station(tmp_path, name="BBBB", span=6.0, seed=8)
    npz2, red = ngl_reduce.reduce_to_file(
        src, tmp_path / "images2", step_years=[float(t[0]) + 3.0]
    )
    rows = ngl_fits.station_rows(npz2)
    kinds = {r["kind"] for r in rows}
    assert {"whole", "seg", "segmean"} <= kinds
    seg_east = [r for r in rows
                if r["kind"] == "seg" and r["component"] == "east"]
    assert len(seg_east) == 2
    mean = [r for r in rows
            if r["kind"] == "segmean" and r["component"] == "east"][0]
    w = np.array([r["n"] for r in seg_east], dtype=float)
    v = np.array([r["v"] for r in seg_east], dtype=float)
    assert abs(mean["v"] - float(w @ v / w.sum())) < 1e-12
    assert mean["n"] == int(w.sum())


def test_exactness_rows_meet_the_gate_on_every_component(tmp_path):
    src, npz, t, cols = station(tmp_path, name="CCCC", span=8.0, seed=11)
    rows = ngl_fits.exactness_rows(src, npz)
    assert len(rows) == 3
    for r in rows:
        assert r["score"] <= compare.EXACTNESS_TOL
        assert r["coverage"] <= compare.COVERAGE_TOL
        assert r["gram_rebuild_err"] < 1e-10
        assert r["gate"] == "ok"
        assert r["n"] == len(t)
        assert abs(r["v_image"] - r["v_raw"]) < 1e-6
        assert set(ngl_fits.EXACT_COLUMNS) >= set(r)


def test_exactness_rows_report_an_undersampled_station(tmp_path):
    # 40 epochs over 6 years: the density floor holds the order to 10, so
    # the model's sensitivities are not represented and the row is
    # reported UNDERSAMPLED rather than failed.
    rng = np.random.default_rng(21)
    t = np.round(np.sort(2008.0 + rng.uniform(0.0, 6.0, 40)), 4)
    t = t[np.diff(t, prepend=t[0] - 1.0) > 0]
    tc = t - t[0]
    cols = {
        k: (ngl_reduce.ngl_design(tc) @ np.array(v)
            + 0.002 * rng.standard_normal(tc.size))
        for k, v in TRUTH.items()
    }
    src = write_station(tmp_path / "EEEE.tenv3", t, cols)
    npz, _ = ngl_reduce.reduce_to_file(src, tmp_path / "sparse")
    # dtfit.image.fit warns about the same thing the coverage column
    # records, so the warning is part of what this test asserts
    with pytest.warns(UserWarning, match="image coverage"):
        rows = ngl_fits.exactness_rows(src, npz)
    assert all(r["order"] == compare.legendre_order(float(tc[-1]), t.size)
               for r in rows)
    assert any(r["gate"] == "UNDERSAMPLED" for r in rows)
    assert all(r["gate"] != "FAIL" for r in rows)


def test_ranking_agrees_with_an_independent_raw_least_squares(tmp_path):
    src, npz, t, cols = station(tmp_path, name="DDDD", span=6.0, seed=13)
    rows = ngl_fits.ranking_rows(npz, src)
    east = [r for r in rows if r["component"] == "east"]
    by_image = sorted(east, key=lambda r: r["bic_image"])
    by_raw = sorted(east, key=lambda r: r["bic_raw"])
    assert by_image[0]["model"] in (
        "trend_annual_semi", "trend_quad_annual_semi"
    )
    # The raw arm is numpy.linalg.lstsq on each model's own design matrix,
    # so the agreement below is measured, not structural: the image
    # identity has to reproduce the residual sum of squares first.
    for r in east:
        assert abs(r["rss_image"] - r["rss_raw"]) <= 1e-8 * r["rss_raw"]
        assert abs(r["bic_image"] - r["bic_raw"]) <= 1e-8 * abs(
            r["bic_raw"]
        )
    assert [r["model"] for r in by_image] == [r["model"] for r in by_raw]
    assert set(ngl_fits.RANK_COLUMNS) >= set(east[0])


def test_ranking_without_samples_leaves_the_raw_arm_empty(tmp_path):
    src, npz, t, cols = station(tmp_path, name="FFFF", span=5.0, seed=15)
    rows = ngl_fits.ranking_rows(npz, None)
    assert rows and all(r["bic_raw"] is None for r in rows)
    assert all(r["rank_raw"] is None for r in rows)
    assert all(r["rank_image"] is not None for r in rows)


def test_run_helpers_fan_out_and_keep_the_input_order(tmp_path):
    a = station(tmp_path, name="AAAA", span=5.0, seed=1)
    b = station(tmp_path, name="BBBB", span=5.0, seed=2)
    rows = ngl_fits.run_fits([a[1], b[1]], {}, workers=1)
    assert [r["sta"] for r in rows][:3] == ["AAAA"] * 3
    exact = ngl_fits.run_exactness([(a[0], a[1]), (b[0], b[1])], workers=1)
    assert {r["sta"] for r in exact} == {"AAAA", "BBBB"}
    assert all(r["gate"] == "ok" for r in exact)
