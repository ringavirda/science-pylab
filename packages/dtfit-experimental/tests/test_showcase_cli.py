"""The showcase command line, driven over a tiny synthetic dataset the
test lays out exactly as the real one is laid out."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import threading
import time

import numpy as np
import pytest

from dtfit.image import assemble

from dtfit_experimental.experiments.domains.image_showcase import (
    cli, filters, ngl, ngl_reduce, paths, store, stream,
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


ISD_HEAD = ('"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION",'
            '"NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND",'
            '"CIG","VIS","TMP","DEW","SLP"\n')
ISD_ROW = ('"{sta}","{date}","4","70.9","-8.6","9.0","A STATION","FM-12",'
           '"99999","V020","318,1,N,0061,1","99999,9,9,9",'
           '"999999,9,9,9","{tmp},1","-0130,1","10130,1"\n')
ISD_TRUTH = [12.0, 2.0, -5.0, 1.0, 18.0, 0.001]      # a1 a2 b1 b2 c v


def make_isd_year(path, sta, year=2024, seed=3):
    """A full hourly station-year (the module's own annual design plus a
    diurnal cycle and noise), the same shape
    ``tests/test_showcase_isd_reduce.py`` fits against."""
    from dtfit_experimental.experiments.domains.image_showcase import (
        isd, isd_reduce,
    )

    rng = np.random.default_rng(seed)
    days = isd.days_in_year(year)
    t = np.arange(0.0, days, 1.0 / 24.0)
    y = (isd_reduce.annual_design(t) @ np.array(ISD_TRUTH)
         + 8.0 * np.cos(2.0 * np.pi * t - 1.0)
         + 0.2 * rng.standard_normal(t.size))
    with open(path, "w") as fh:
        fh.write(ISD_HEAD)
        month, dom = 1, 1
        lengths = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        for k in range(t.size):
            hour = k % 24
            if hour == 0 and k > 0:
                dom += 1
                if dom > lengths[month - 1]:
                    dom = 1
                    month += 1
            date = f"{year}-{month:02d}-{dom:02d}T{hour:02d}:00:00"
            fh.write(ISD_ROW.format(
                sta=sta, date=date, tmp=f"{int(round(y[k] * 10)):+05d}",
            ))


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
    # An image one percent off its station file is what a real failure
    # looks like: a miss far beyond what the conditioning explains.
    npz = next(images.glob("*.npz"))
    imgs, info = store.load_images(npz)
    east = imgs["whole_east"]
    imgs["whole_east"] = dataclasses.replace(east, S=east.S * 1.01)
    store.save_images(npz, imgs, info)
    code = cli.main(["ngl-exact", "--images", str(images)])
    assert code == 1
    rows = read_csv(tmp_path / "results" / "ngl_exact.csv")
    assert [r["gate"] for r in rows if r["component"] == "east"] == ["FAIL"]
    assert all(r["gate"] == "ok" for r in rows if r["component"] != "east")
    # and a looser tolerance clears it again: --tol moves the verdict in
    # both directions, it does not only tighten
    assert cli.main([
        "ngl-exact", "--images", str(images), "--tol", "0.1",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "ngl_exact.csv")
    assert all(r["gate"] == "ok" for r in rows)


def test_cli_reports_an_unknown_subcommand(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["not-a-command"])


def _pool_worker_blas_threads():
    """Run in a pool worker: its own BLAS thread counts, the half of the
    ruling that matters (the workers, not just the parent, inherit the
    cap). Module level so a spawned or forkserver worker can pickle it."""
    from threadpoolctl import threadpool_info

    return {
        (p["internal_api"], p["num_threads"]) for p in threadpool_info()
        if p["user_api"] == "blas"
    }


def test_cli_runs_every_process_on_one_blas_thread(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor

    from threadpoolctl import threadpool_info

    seen: dict[str, object] = {}

    def probe(args):
        seen["threads"] = {
            (p["internal_api"], p["num_threads"]) for p in threadpool_info()
            if p["user_api"] == "blas"
        }
        with ProcessPoolExecutor(max_workers=1) as pool:
            seen["worker_threads"] = pool.submit(
                _pool_worker_blas_threads
            ).result()
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
    assert all(n == 1 for _, n in seen["worker_threads"])


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


def test_leg5_tables_recomputes_mismatched_and_flags_match(tmp_path):
    """The three published tables come from :func:`cli.leg5_tables`, not
    from typing: a corrupted digest must show up as ``mismatched`` and a
    wrong wire flag count must fail ``flags_match``, both recomputed
    locally rather than trusted from the run's own JSON summary."""
    root = tmp_path / "data"
    make_dataset(root, n_stations=2)
    images_root = tmp_path / "images"
    files = ngl.station_files(root / "ngl" / "tenv3")
    rows = ngl_reduce.reduce_many(
        files, images_root / "ngl", {}, workers=1, block_len=1.0,
    )
    assert all(not r["error"] for r in rows)

    groups = {}
    for name in ("ST00", "ST01"):
        imgs, _info = store.load_images(images_root / "ngl" / f"{name}.npz")
        blocks = [img for n, img in imgs.items()
                  if n.startswith("blk") and n.endswith("_east")]
        whole = assemble(blocks, order=min(b.order for b in blocks))
        digest = stream.image_digest(whole)
        groups[f"{name}/east"] = {
            "n_blocks": len(blocks), "order": int(whole.order),
            "domain": [float(whole.domain[0]), float(whole.domain[1])],
            "n": int(whole.n),
            # ST01's digest is deliberately wrong: the writer must catch
            # this by reassembling locally, not by trusting the summary.
            "digest": digest if name == "ST00" else "0" * 64,
            "bytes_header": 100, "bytes_payload": 200,
            "samples": int(whole.n), "n_flags": 0,
            "params": None, "rss": None, "converged": None,
        }
    summary = {
        "n_images": sum(g["n_blocks"] for g in groups.values()),
        "n_other": 0, "bytes_header": 300, "bytes_payload": 400,
        "seconds": 1.0, "images_per_second": 1.0, "clock_delta_s": 0.0,
        "groups": groups,
    }
    leg5_dir = tmp_path / "results" / "leg5"
    leg5_dir.mkdir(parents=True)
    (leg5_dir / "run_a.json").write_text(json.dumps(summary))

    tenv3_dir = root / "ngl" / "tenv3"
    local_flags = cli._leg5_local_flags(["ST00", "ST01"], tenv3_dir)
    results_dir = tmp_path / "results"
    replay_row = {
        "host": "h", "n_stations": 2, "rate_requested": 1000.0,
        "samples_per_second": 999.0, "n_frames": 10, "n_samples": 100,
        "n_dropped": 0, "bytes_sent": 1000, "n_frames_back": 5,
        "seconds": 1.0,
    }
    store.write_table(
        results_dir / "leg5_replay_rate_1000.csv", [replay_row],
        list(replay_row),
    )

    def write_track(n_flags):
        row = {
            "host": "h", "config": "detection", "n_frames": 10,
            "n_samples": 100, "n_dropped": 0, "seconds": 1.0,
            "samples_per_second": 100.0, "us_per_update": 1.0,
            "n_flags": n_flags, "n_blocks": 4, "bytes_back": 500,
            "flag_times": "",
        }
        store.write_table(
            results_dir / "leg5_track_rate_1000.csv", [row], list(row),
        )

    sources = (("run_a", None, "local", 1.0, "ngl"),)

    write_track(local_flags)
    rows, group_rows, sweep = cli.leg5_tables(
        leg5_dir, images_root, results_dir, tenv3_dir, sources=sources,
    )
    by_station = {r["station"]: r for r in group_rows}
    assert by_station["ST00"]["mismatched"] == 0
    assert by_station["ST01"]["mismatched"] == 1
    assert rows[0]["mismatched"] == 1
    assert sweep[0]["flags_match"] is True

    write_track(local_flags + 1)
    _, _, sweep = cli.leg5_tables(
        leg5_dir, images_root, results_dir, tenv3_dir, sources=sources,
    )
    assert sweep[0]["flags_match"] is False

    stream_rows = read_csv(results_dir / "leg5_stream.csv")
    group_csv_rows = read_csv(results_dir / "leg5_groups.csv")
    replay_rows = read_csv(results_dir / "leg5_replay.csv")
    assert len(stream_rows) == 1 and len(group_csv_rows) == 2
    assert len(replay_rows) == 1


def test_cli_isd_reduce_then_exact_gates_a_small_dataset(tmp_path,
                                                          monkeypatch):
    root = tmp_path / "data"
    year_dir = root / "isd" / "2024"
    year_dir.mkdir(parents=True)
    make_isd_year(year_dir / "72278023183.csv", "72278023183")
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    images = tmp_path / "images"

    assert cli.main([
        "isd-reduce", "--images", str(images), "--year", "2024",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "isd_reduce.csv")
    assert len(rows) == 1 and rows[0]["error"] == ""

    assert cli.main([
        "isd-exact", "--images", str(images), "--year", "2024",
    ]) == 0
    exact = read_csv(tmp_path / "results" / "isd_exact.csv")
    assert exact and all(r["gate"] == "ok" for r in exact)

    # An image well off its raw fit must fail the gate visibly, the
    # second gate this domain relies on (spec's `isd-exact`, `_regate`'s
    # day_score arm).
    npz = images / "2024" / "72278023183.npz"
    imgs, info = store.load_images(npz)
    year_img = imgs["year_TMP"]
    imgs["year_TMP"] = dataclasses.replace(year_img, S=year_img.S * 1.01)
    store.save_images(npz, imgs, info)
    code = cli.main([
        "isd-exact", "--images", str(images), "--year", "2024",
    ])
    assert code == 1
    exact = read_csv(tmp_path / "results" / "isd_exact.csv")
    assert exact[0]["gate"] == "FAIL"


def test_cli_normals_fetch_then_isd_normals_uses_the_injected_opener(
    tmp_path, monkeypatch
):
    from dtfit_experimental.experiments.domains.image_showcase import isd

    root = tmp_path / "data"
    year_dir = root / "isd" / "2024"
    year_dir.mkdir(parents=True)
    sta = "72278023183"
    make_isd_year(year_dir / f"{sta}.csv", sta)
    monkeypatch.setenv("SHOWCASE_DATA", str(root))
    monkeypatch.setattr(paths, "results_dir", lambda: tmp_path / "results")
    images = tmp_path / "images"
    assert cli.main([
        "isd-reduce", "--images", str(images), "--year", "2024",
    ]) == 0

    index_html = b'<html><a href="USW00023183.csv">a</a></html>'
    normals_body = (
        b"STATION,NAME,LATITUDE,LONGITUDE,ELEVATION,DATE,month,day,"
        b"hour,HLY-TEMP-NORMAL,HLY-PRES-NORMAL\n"
        + b"\n".join(
            b'"USW00023183","X","1","2","3","%02d-%02dT%02d:00:00",'
            b'"%02d","%02d","%02d","    50.9","  1017.8"'
            % (m, d, h, m, d, h)
            for m in (1,) for d in range(1, 29) for h in (0, 12)
        )
        + b"\n"
    )

    def opener(url, timeout=None):
        return io.BytesIO(index_html if url.endswith("/") else normals_body)

    monkeypatch.setattr(isd.urllib.request, "urlopen", opener)
    assert cli.main(["normals-fetch", "--year", "2024"]) == 0
    fetched = read_csv(tmp_path / "results" / "normals_fetch.csv")
    assert fetched == [{"isd_station": sta, "normals_id": "USW00023183",
                        "path": str(paths.normals_dir() / "USW00023183.csv")}]

    assert cli.main([
        "isd-normals", "--images", str(images), "--year", "2024",
    ]) == 0
    rows = read_csv(tmp_path / "results" / "isd_normals.csv")
    assert rows and all(r["station"] == sta for r in rows)
