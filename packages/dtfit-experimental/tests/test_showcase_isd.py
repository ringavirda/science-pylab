"""The NOAA Global Hourly reader and the normals, on fixtures the test
writes. No network call and no real file."""

from __future__ import annotations

import io

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import isd

HEAD = ('"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION",'
        '"NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND","CIG",'
        '"VIS","TMP","DEW","SLP"\n')
ROW = ('"{sta}","{date}","4","{lat}","{lon}","{elev}","A STATION",'
       '"FM-12","99999","V020","318,1,N,0061,1","99999,9,9,9",'
       '"999999,9,9,9","{tmp}","-0130,1","{slp}"\n')


def write_isd(path, rows, sta="72278023183"):
    """rows: (date, tmp field, slp field, lat, lon, elev)."""
    with open(path, "w") as fh:
        fh.write(HEAD)
        for date, tmp, slp, lat, lon, elev in rows:
            fh.write(ROW.format(sta=sta, date=date, tmp=tmp, slp=slp,
                                lat=lat, lon=lon, elev=elev))
    return path


def test_parse_field_scales_tenths_and_reads_the_quality_code():
    assert isd.parse_field("-0070,1", 9999) == (-7.0, "1")
    assert isd.parse_field("10208,1", 99999) == (1020.8, "1")
    assert isd.parse_field("+9999,9", 9999) == (None, "9")
    assert isd.parse_field("99999,9", 99999) == (None, "9")
    assert isd.parse_field("", 9999) == (None, "")
    assert isd.parse_field("garbage", 9999) == (None, "")


def test_iso_days_counts_from_the_years_start():
    assert isd.iso_days("2024-01-01T00:00:00", True) == 0.0
    assert isd.iso_days("2024-01-02T06:00:00", True) == pytest.approx(1.25)
    assert isd.iso_days("2024-03-01T00:00:00", True) == 60.0
    assert isd.iso_days("2023-03-01T00:00:00", False) == 59.0
    assert isd.days_in_year(2024) == 366 and isd.days_in_year(2023) == 365


def test_read_isd_filters_quality_missing_and_repeats(tmp_path):
    p = write_isd(tmp_path / "s.csv", [
        ("2024-01-01T00:00:00", "-0070,1", "10208,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T00:53:00", "-0065,5", "10204,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T00:53:00", "-0060,1", "10200,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T02:00:00", "-0055,3", "10198,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T03:00:00", "+9999,9", "10196,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T04:00:00", "-0050,4", "10194,1", "71.0", "-8.7", "9.0"),
    ])
    chunks = list(isd.read_isd(p, "TMP"))
    assert len(chunks) == 1
    c = chunks[0]
    np.testing.assert_allclose(c.t, [0.0, 0.0 + 53.0 / 1440.0,
                                     4.0 / 24.0])
    np.testing.assert_allclose(c.y, [-7.0, -6.5, -5.0])
    assert c.dropped_quality == 1 and c.dropped_missing == 1
    assert c.dropped_repeat == 1
    np.testing.assert_allclose(c.lat, [70.9, 70.9, 71.0])


def test_read_isd_chunks_and_carries_the_dedup_across_blocks(tmp_path):
    rows = [
        (f"2024-01-01T{h:02d}:00:00", f"-00{40 + h:02d},1", "10208,1",
         "70.9", "-8.6", "9.0")
        for h in range(6)
    ]
    rows.insert(3, rows[2])
    p = write_isd(tmp_path / "c.csv", rows)
    chunks = list(isd.read_isd(p, "TMP", chunk=2))
    t = np.concatenate([c.t for c in chunks])
    assert t.size == 6 and np.all(np.diff(t) > 0)
    assert sum(c.dropped_repeat for c in chunks) == 1


def test_read_isd_carries_drop_counts_past_the_last_chunk(tmp_path):
    good = [
        (f"2024-01-01T{h:02d}:00:00", f"-00{40 + h:02d},1", "10208,1",
         "70.9", "-8.6", "9.0")
        for h in range(4)
    ]
    bad = [
        (f"2024-01-01T{h:02d}:00:00", "+9999,9", "10208,1",
         "70.9", "-8.6", "9.0")
        for h in range(4, 9)
    ]
    p = write_isd(tmp_path / "e.csv", good + bad)
    chunks = list(isd.read_isd(p, "TMP", chunk=4))
    assert sum(c.t.size for c in chunks) == 4
    assert sum(c.dropped_missing for c in chunks) == 5
    assert chunks[-1].t.size == 0                # the residual-count chunk


def test_read_isd_all_rows_dropped_reports_no_chunk_and_no_counts_lost(
    tmp_path,
):
    p = write_isd(tmp_path / "f.csv", [
        ("2024-01-01T00:00:00", "+9999,9", "10208,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T01:00:00", "-0070,3", "10208,1", "70.9", "-8.6", "9.0"),
    ])
    chunks = list(isd.read_isd(p, "TMP"))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.t.size == 0
    assert c.dropped_missing == 1 and c.dropped_quality == 1


def test_station_files_filters_and_truncates(tmp_path):
    for sta in ("72278023183", "72278023184", "72278023185"):
        (tmp_path / f"{sta}.csv").write_text(HEAD)
    got = isd.station_files(tmp_path)
    assert [p.name for p in got] == [
        "72278023183.csv", "72278023184.csv", "72278023185.csv",
    ]
    filtered = isd.station_files(
        tmp_path, stations=["72278023185", "72278023183", "missing"]
    )
    assert [p.name for p in filtered] == [
        "72278023183.csv", "72278023185.csv",
    ]
    assert len(isd.station_files(tmp_path, limit=2)) == 2


def test_station_year_returns_the_series_and_its_counts(tmp_path):
    rows = [
        (f"2024-01-{1 + d:02d}T12:00:00", f"-00{50 + d:02d},1", "10208,1",
         "70.9", "-8.6", "9.0")
        for d in range(5)
    ]
    p = write_isd(tmp_path / "y.csv", rows)
    t, y, info = isd.station_year(p, "TMP")
    assert t.size == 5 and y.size == 5
    assert info["station"] == "72278023183" and info["year"] == 2024
    assert info["days"] == 366 and info["n"] == 5
    assert info["dropped_quality"] == 0
    assert t[0] == pytest.approx(0.5) and t[-1] == pytest.approx(4.5)
    assert isd.station_header(p) == ("72278023183", 2024)


def test_coordinate_changes_reports_a_station_move(tmp_path):
    p = write_isd(tmp_path / "m.csv", [
        ("2024-01-01T00:00:00", "-0070,1", "10208,1", "70.9", "-8.6", "9.0"),
        ("2024-01-01T01:00:00", "-0070,1", "10208,1", "70.9", "-8.6", "9.0"),
        ("2024-06-01T00:00:00", "-0070,1", "10208,1", "71.5", "-8.6", "9.0"),
    ])
    moves = isd.coordinate_changes(p)
    assert len(moves) == 1
    assert moves[0]["lat"] == pytest.approx(71.5)
    assert moves[0]["t"] == pytest.approx(152.0)


def test_day_grid_needs_all_twenty_four_bins(tmp_path):
    full = [
        (f"2024-02-02T{h:02d}:53:00", f"-00{10 + h:02d},1", "10208,1",
         "70.9", "-8.6", "9.0")
        for h in range(24)
    ]
    p = write_isd(tmp_path / "d.csv", full)
    grid = isd.day_grid(p, 32, "TMP")
    assert grid is not None and grid.shape == (24,)
    np.testing.assert_allclose(grid[0], -1.0)
    np.testing.assert_allclose(grid[23], -3.3)
    p2 = write_isd(tmp_path / "d2.csv", full[:23])
    assert isd.day_grid(p2, 32, "TMP") is None
    # several days in one pass, including one the file does not reach
    many = isd.day_grids(p, [32, 33], "TMP")
    assert sorted(many) == [32, 33]
    np.testing.assert_allclose(many[32], grid)
    assert many[33] is None
    assert isd.day_grids(p, [], "TMP") == {}


def test_wban_and_normals_matching():
    assert isd.wban("72278023183") == "23183"
    assert isd.wban("01001099999") is None
    assert isd.wban("short") is None
    got = isd.match_normals(
        ["72278023183", "01001099999", "70026027502"],
        ["USW00023183.csv", "AQW00061705.csv", "USW00027502.csv"],
    )
    assert got == {"72278023183": "USW00023183",
                   "70026027502": "USW00027502"}


def test_normals_index_and_download_use_the_injected_opener(tmp_path):
    index = (b'<html><a href="USW00023183.csv">a</a>'
             b'<a href="AQW00061705.csv">b</a></html>')
    body = (b"STATION,NAME,LATITUDE,LONGITUDE,ELEVATION,DATE,month,day,"
            b"hour,HLY-TEMP-NORMAL,HLY-PRES-NORMAL\n"
            b'"USW00023183","X","1","2","3","01-01T00:00:00","01","01",'
            b'"00","    50.9","  1017.8"\n'
            b'"USW00023183","X","1","2","3","01-01T01:00:00","01","01",'
            b'"01","    49.9","-9999"\n')

    def opener(url, timeout=None):
        return io.BytesIO(index if url.endswith("/") else body)

    ids = isd.normals_index(opener=opener)
    assert ids == ["AQW00061705", "USW00023183"]
    written = isd.download_normals(tmp_path, ["USW00023183"], opener=opener)
    assert [p.name for p in written] == ["USW00023183.csv"]
    n = isd.read_normals(written[0])
    assert n["hour"].tolist() == [0, 1]
    # 50.9 F is 10.5 C; the pressure of the second row is missing.
    np.testing.assert_allclose(n["temp_c"], [10.5, 9.94444444], atol=1e-6)
    assert np.isnan(n["pres_hpa"][1])
    assert n["pres_hpa"][0] == pytest.approx(1017.8)
