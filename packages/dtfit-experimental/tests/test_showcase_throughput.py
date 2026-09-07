"""Leg 1's measurement helpers. No GPU, no real dataset."""

from __future__ import annotations

import numpy as np

from dtfit.image import Image, Original

from dtfit_experimental.experiments.domains.image_showcase import (
    throughput,
)


def test_machine_row_names_the_host_and_the_toolchain():
    row = throughput.machine_row()
    assert set(row) >= {"host", "machine", "cpu_count", "python",
                        "numpy"}
    assert row["cpu_count"] >= 1
    assert row["python"].count(".") >= 1


def test_gpu_probe_reports_the_failure_instead_of_raising():
    ok, message = throughput.gpu_probe("not-a-backend")
    assert ok is False and "not-a-backend" in message


def test_disk_read_rate_measures_bytes_and_seconds(tmp_path):
    blob = b"x" * (256 * 1024)
    paths = []
    for k in range(3):
        p = tmp_path / f"f{k}.bin"
        p.write_bytes(blob)
        paths.append(p)
    got = throughput.disk_read_rate(paths)
    assert got["bytes"] == 3 * len(blob)
    assert got["seconds"] > 0.0 and got["mb_per_second"] > 0.0
    assert got["n_files"] == 3


def test_float32_error_is_larger_than_float64_and_both_are_small():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 10.0, 4000))
    t = t - t[0]
    y = 4781.0 + 0.03 * t + 0.002 * rng.standard_normal(t.size)
    got = throughput.float32_error(t, y, 40, chunk=500)
    assert got["rel_S_float64"] <= 1e-12
    assert got["rel_S_float32"] > got["rel_S_float64"]
    assert got["rel_S_float32"] < 1e-2
    assert got["n"] == t.size and got["order"] == 40


def test_channel_gemm_rate_projects_every_channel(tmp_path):
    rng = np.random.default_rng(1)
    Y = rng.standard_normal((24, 12))
    got = throughput.channel_gemm_rate(Y, order=16, repeats=2)
    assert got["channels"] == 12 and got["samples"] == 24 * 12
    assert got["backend"] == "numpy"
    assert got["seconds"] > 0.0 and got["elements_per_second"] > 0.0
    # the batch equals a per-channel image, channel by channel
    x = np.arange(24) / 24.0
    one = Image.of(Original(x, Y[:, 3], domain=(0.0, 1.0)), "legendre", 16)
    np.testing.assert_allclose(got["images"][3].S, one.S, rtol=0, atol=1e-12)


def test_reduce_rate_reports_samples_per_second(tmp_path):
    header = (
        "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0 _e _n0 _n"
        " _u0 _u _ant sig_e sig_n sig_u c_en c_eu c_nu lat lon hgt\n"
    )
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(2)
    for name in ("AAAA", "BBBB"):
        t = 2008.0 + np.arange(800) / 365.25
        y = 0.03 * (t - t[0]) + 0.002 * rng.standard_normal(t.size)
        with open(src / f"{name}.tenv3", "w") as fh:
            fh.write(header)
            for i in range(t.size):
                fh.write(
                    f"{name} 08MAR27 {t[i]:.4f} 54552 1472 4 130.8 4781"
                    f" {y[i]:.6f} -1378706 {y[i]:.6f} 104 {y[i]:.6f} 0.0"
                    f" 0.001 0.001 0.005 0.0 0.0 0.0 -12.4 -229.1 104.8\n"
                )
    got = throughput.reduce_rate(
        sorted(src.glob("*.tenv3")), tmp_path / "images", dataset="ngl"
    )
    assert got["n_files"] == 2 and got["samples"] == 1600
    assert got["samples_per_second"] > 0.0
    assert got["peak_mib"] > 0.0
    # tracemalloc and the resident set are different numbers; the memory
    # claim is stated against the second
    assert got["peak_rss_mib"] is None or got["peak_rss_mib"] > 0.0
    assert (got["coef_bytes"] + got["gram_bytes"] + got["grid_bytes"]
            == got["image_bytes"] > 0)
    assert got["gram_bytes"] > got["coef_bytes"]
    assert got["raw_bytes"] > 0
    assert set(throughput.THROUGHPUT_COLUMNS) >= set(got)


def test_peak_rss_is_a_resident_set_or_absent():
    value = throughput.peak_rss_mib()
    assert np.isnan(value) or value > 0.0
