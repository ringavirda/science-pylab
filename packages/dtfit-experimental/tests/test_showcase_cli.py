"""The showcase command line, driven over a tiny synthetic dataset the
test lays out exactly as the real one is laid out."""

from __future__ import annotations

import csv
import os
import threading
import time

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import (
    cli, filters, ngl, ngl_reduce, paths, stream,
)

HEADER = (
    "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0 _e _n0 _n _u0 _u"
    " _ant sig_e sig_n sig_u c_en c_eu c_nu lat lon hgt\n"
)


def make_dataset(root, n_stations=2, n=900, seed=5):
    """``<root>/ngl/tenv3/*.tenv3`` plus a steps file and a MIDAS table,
    the layout the real copy has."""
    rng = np.random.default_rng(seed)
    tenv3 = root / "ngl" / "tenv3"
    tenv3.mkdir(parents=True)
    names = [f"ST{k:02d}" for k in range(n_stations)]
    for name in names:
        t = 2008.0 + np.arange(n) / 365.25
        tc = t - t[0]
        y = (ngl_reduce.ngl_design(tc)
             @ np.array([0.004, 0.001, -0.002, 0.0005, 0.83, 0.036])
             + 0.002 * rng.standard_normal(n))
        with open(tenv3 / f"{name}.tenv3", "w") as fh:
            fh.write(HEADER)
            for i in range(n):
                fh.write(
                    f"{name} 08MAR27 {t[i]:.4f} 54552 1472 4 130.8 4781"
                    f" {y[i]:.6f} -1378706 {y[i]:.6f} 104 {y[i]:.6f} 0.0"
                    f" 0.001 0.001 0.005 0.0 0.0 0.0 -12.4 -229.1 104.8\n"
                )
    (root / "ngl" / "steps.txt").write_text(
        f"{names[0]}  09JUL01  1  Antenna_Type_Changed\n"
    )
    (root / "ngl" / "midas.IGS20.txt").write_text("".join(
        f"{name} MIDAS5 2008.2355 2010.7324 2.4969 900 900 800"
        "   0.036100   0.058500  -0.000700  0.000200 0.000300 0.000900"
        "   4781.8 -1378706.3 104.9 0.07 0.11 0.08 0.002 0.002 0.008"
        "   1 -12.4 -229.2 104.8\n"
        for name in names
    ))
    return root


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_cli_reduces_fits_and_gates_a_small_dataset(tmp_path,
                                                    monkeypatch):
    root = make_dataset(tmp_path / "data")
    images = tmp_path / "images"
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")

    assert cli.main([
        "ngl-reduce", "--images", str(images), "--suffix", "_test",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "ngl_reduce_test.csv")
    assert len(rows) == 2 and all(r["error"] == "" for r in rows)
    assert (images / "ST00.npz").exists()

    assert cli.main([
        "ngl-fits", "--images", str(images), "--suffix", "_test",
    ]) == 0
    fits = read_csv(tmp_path / "results" / "ngl_fits_test.csv")
    assert {r["component"] for r in fits} == {"east", "north", "up"}
    east = [r for r in fits
            if r["kind"] == "whole" and r["component"] == "east"]
    assert len(east) == 2
    assert all(abs(float(r["diff_over_sigma"])) < 10.0 for r in east)

    assert cli.main([
        "ngl-exact", "--images", str(images), "--suffix", "_test",
    ]) == 0
    exact = read_csv(tmp_path / "results" / "ngl_exact_test.csv")
    assert len(exact) == 6 and all(r["gate"] == "ok" for r in exact)
    assert all(float(r["gram_rebuild_err"]) < 1e-10 for r in exact)

    # every subcommand leaves a timing row beside its table
    timing = read_csv(tmp_path / "results" / "ngl_exact_timing_test.csv")
    assert len(timing) == 1
    assert timing[0]["command"] == "ngl-exact"
    assert int(timing[0]["rows"]) == 6
    assert float(timing[0]["seconds"]) > 0.0
    assert timing[0]["host"]


def test_cli_rank_can_leave_the_station_files_untouched(tmp_path,
                                                        monkeypatch):
    root = make_dataset(tmp_path / "data")
    images = tmp_path / "images"
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    assert cli.main(["ngl-reduce", "--images", str(images)]) == 0

    def boom(*_args, **_kwargs):
        raise AssertionError("--no-raw must not open a station file")

    with monkeypatch.context() as m:
        m.setattr(ngl, "read_tenv3", boom)
        assert cli.main([
            "ngl-rank", "--images", str(images), "--no-raw",
        ]) == 0
    rows = read_csv(tmp_path / "results" / "ngl_rank.csv")
    assert rows and all(r["bic_raw"] == "" for r in rows)
    assert all(r["rank_image"] != "" for r in rows)
    assert cli.main(["ngl-rank", "--images", str(images),
                     "--suffix", "_raw"]) == 0
    with_raw = read_csv(tmp_path / "results" / "ngl_rank_raw.csv")
    assert all(r["bic_raw"] != "" for r in with_raw)


def test_cli_exact_fails_visibly_when_a_station_misses_the_gate(
    tmp_path, monkeypatch
):
    root = make_dataset(tmp_path / "data", n_stations=1)
    images = tmp_path / "images"
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    assert cli.main(["ngl-reduce", "--images", str(images)]) == 0
    # An impossible tolerance turns every station into a failure, which
    # is how the run reports a real one.
    code = cli.main([
        "ngl-exact", "--images", str(images), "--tol", "1e-30",
    ])
    assert code == 1
    rows = read_csv(tmp_path / "results" / "ngl_exact.csv")
    assert any(r["gate"] == "FAIL" for r in rows)
    # and a looser tolerance clears it again: --tol moves the verdict in
    # both directions, it does not only tighten
    assert cli.main([
        "ngl-exact", "--images", str(images), "--tol", "1e-4",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "ngl_exact.csv")
    assert all(r["gate"] == "ok" for r in rows)


def test_cli_reports_an_unknown_subcommand(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["not-a-command"])


def test_cli_runs_every_process_on_one_blas_thread(tmp_path, monkeypatch):
    from threadpoolctl import threadpool_info

    seen: dict[str, object] = {}

    def probe(args):
        seen["threads"] = {
            (p["internal_api"], p["num_threads"]) for p in threadpool_info()
            if p["user_api"] == "blas"
        }
        seen["env"] = {n: os.environ.get(n) for n in cli.BLAS_THREAD_VARS}
        return 0

    for name in cli.BLAS_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(cli._COMMANDS, "probe", probe)
    monkeypatch.setitem(cli.RESULTS, "probe", "probe")
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    assert cli.main(["probe"]) == 0
    assert seen["env"] == {n: "1" for n in cli.BLAS_THREAD_VARS}
    assert all(n == 1 for _, n in seen["threads"])


def test_cli_throughput_writes_a_row(tmp_path, monkeypatch):
    root = make_dataset(tmp_path / "data")
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    assert cli.main([
        "throughput", "--images", str(tmp_path / "images"),
        "--limit", "2", "--suffix", "_test",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "throughput_test.csv")
    kinds = {r["route"] for r in rows}
    assert "disk" in kinds and any(k.startswith("cpu-") for k in kinds)
    cpu = [r for r in rows if r["route"] == "cpu-1"][0]
    assert cpu["peak_mib"] and cpu["gram_bytes"]


def test_cli_stream_serve_moves_a_station_over_localhost(
    tmp_path, monkeypatch
):
    root = make_dataset(tmp_path / "data", n_stations=1)
    images = tmp_path / "images"
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    assert cli.main(["ngl-reduce", "--images", str(images)]) == 0
    port_file = tmp_path / "port"
    result = {}

    def consume():
        result["code"] = cli.main([
            "stream-serve", "--host", "127.0.0.1", "--port", "0",
            "--port-file", str(port_file), "--model", "ngl-trend",
            "--timeout", "60",
        ])

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        deadline = time.time() + 20.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert port_file.exists()
        port = int(port_file.read_text().strip())
        summary = stream.produce(
            "127.0.0.1", port,
            stream.iter_images(images / "ngl" if (images / "ngl").is_dir()
                               else images, prefix="blk"),
        )
    finally:
        thread.join(timeout=30.0)
    assert result["code"] == 0
    assert summary["n_acks"] == summary["n_images"] > 0
    rows = read_csv(tmp_path / "results" / "leg5_serve.csv")
    assert rows and all(r["p_v"] for r in rows)
    assert all(r["digest"] for r in rows)


def test_cli_stream_track_resets_per_station_to_match_the_local_filter(
    tmp_path, monkeypatch
):
    """Two stations replayed over one connection must raise the same
    flags a fresh local filter per station does, which only holds if
    the tracker rebases each station's own clock and starts a fresh
    filter and block stream for it (rulings behind the leg-5 gate)."""
    root = make_dataset(tmp_path / "data", n_stations=2)
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")

    config = {"detection": filters.NGL_CONFIGS["detection"]}
    local_flags = 0
    for name in ("ST00", "ST01"):
        summary, _steps = filters.ngl_filter_station(
            root / "ngl" / "tenv3" / f"{name}.tenv3", [], configs=config,
        )
        local_flags += sum(
            r["n_flags"] for r in summary if r["component"] == "east"
        )

    port_file = tmp_path / "track_port"
    result = {}

    def track():
        result["code"] = cli.main([
            "stream-track", "--host", "127.0.0.1", "--port", "0",
            "--port-file", str(port_file), "--config", "detection",
            "--field", "east", "--order", "12", "--block", "1.0",
            "--span", "40.0", "--station", "replay", "--timeout", "20",
            "--suffix", "_test",
        ])

    thread = threading.Thread(target=track)
    thread.start()
    try:
        deadline = time.time() + 20.0
        while not port_file.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert port_file.exists()
        port = int(port_file.read_text().strip())
        assert cli.main([
            "stream-replay", "--host", "127.0.0.1", "--port", str(port),
            "--field", "east", "--stations", "ST00,ST01",
            "--chunk", "100", "--timeout", "20", "--suffix", "_test",
        ]) == 0
    finally:
        thread.join(timeout=30.0)
    assert result["code"] == 0
    rows = read_csv(tmp_path / "results" / "leg5_track_test.csv")
    row = rows[0]
    over_wire = [x for x in row["flag_times"].split() if x]
    assert len(over_wire) == local_flags
    # one time-reset-per-chunk bug pins every sample in block 0; two
    # full stations of yearly data must clear several blocks each.
    assert int(row["n_blocks"]) > 2
