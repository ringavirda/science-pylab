"""Leg 3: the streaming filters, on synthetic series the test builds."""

from __future__ import annotations

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import (
    filters, isd, ngl,
)

HEADER = (
    "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0 _e _n0 _n _u0 _u"
    " _ant sig_e sig_n sig_u c_en c_eu c_nu lat lon hgt\n"
)
HEAD_CSV = ('"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION",'
            '"NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND",'
            '"CIG","VIS","TMP","DEW","SLP"\n')
ROW_CSV = ('"72278023183","{date}","4","{lat}","-8.6","9.0","A","FM-12",'
           '"99999","V020","318,1,N,0061,1","99999,9,9,9","999999,9,9,9",'
           '"{tmp},{q}","-0130,1","10208,1"\n')


def ngl_station(path, n=1600, step_index=800, step=0.02, seed=0):
    """A daily three-component station with a common step."""
    rng = np.random.default_rng(seed)
    t = 2008.0 + np.arange(n) / 365.25
    tc = t - t[0]
    base = 0.036 * tc + 0.004 * np.cos(2 * np.pi * tc)
    cols = {}
    for k, comp in enumerate(ngl.COMPONENTS):
        y = base + 0.002 * rng.standard_normal(n)
        y[step_index:] += step
        cols[comp] = y
    with open(path, "w") as fh:
        fh.write(HEADER)
        for i in range(n):
            fh.write(
                f"AAAA 08MAR27 {t[i]:.4f} 54552 1472 4 130.8"
                f" 4781 {cols['east'][i]:.6f} -1378706"
                f" {cols['north'][i]:.6f} 104 {cols['up'][i]:.6f} 0.0"
                f" 0.001 0.001 0.005 0.0 0.0 0.0 -12.4 -229.1 104.8\n"
            )
    return path, float(t[step_index])


def isd_fragment(path, days=20, shift_day=12, shift=4.0, seed=1):
    """Hourly temperature over ``days`` days with a level shift, a moved
    station after the shift and one quality-failed row."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        for h in range(24):
            t = d + h / 24.0
            c = (18.0 + 6.0 * np.cos(2 * np.pi * t - 1.0)
                 + 0.5 * rng.standard_normal())
            if t >= shift_day:
                c += shift
            lat = "70.9" if t < shift_day else "71.4"
            q = "3" if (d == 2 and h == 5) else "1"
            rows.append((f"2024-01-{d + 1:02d}T{h:02d}:00:00", c, lat, q))
    # 20 days of hourly rows put about six detector tests before the
    # shift; the detector arms after three.
    with open(path, "w") as fh:
        fh.write(HEAD_CSV)
        for date, c, lat, q in rows:
            fh.write(ROW_CSV.format(
                date=date, lat=lat, q=q,
                tmp=f"{int(round(c * 10)):+05d}",
            ))
    return path


def test_match_flags_counts_detections_delays_and_false_alarms():
    flags = np.array([1.0, 5.0, 50.0])
    events = np.array([0.5, 40.0])
    got = filters.match_flags(flags, events, horizon=2.0)
    assert got["n_events"] == 2 and got["n_detected"] == 1
    assert got["delays"] == [0.5, None]
    assert got["n_false_alarms"] == 2        # 5.0 and 50.0 explain nothing
    assert got["median_delay"] == 0.5
    empty = filters.match_flags(np.zeros(0), events, horizon=2.0)
    assert empty["n_detected"] == 0 and empty["median_delay"] is None


def test_run_filter_flags_a_step_and_leaves_a_clean_series_alone():
    rng = np.random.default_rng(3)
    n = 1200
    t = np.arange(n) / 365.25
    clean = 4781.83 + 0.036 * t + 0.002 * rng.standard_normal(n)
    stepped = clean.copy()
    stepped[600:] += 0.02
    cfg = filters.NGL_CONFIGS["detection"]
    out = filters.run_filter(t, stepped, cfg)
    assert out["n_updates"] == n and out["seconds"] > 0.0
    hit = [f for f in out["flags"]
           if t[600] < f <= t[600] + 60.0 / 365.25]
    assert hit, "the 2 cm step was not flagged within 60 days"
    assert filters.run_filter(t, clean, cfg)["flags"].size == 0
    assert out["us_per_update"] > 0.0


def test_run_filter_tracks_the_velocity_with_the_tracking_config():
    rng = np.random.default_rng(4)
    n = 2000
    t = np.arange(n) / 365.25
    y = (4781.83 + 0.036 * t + 0.004 * np.cos(2 * np.pi * t)
         + 0.002 * rng.standard_normal(n))
    out = filters.run_filter(t, y, filters.NGL_CONFIGS["tracking"])
    assert abs(out["params"]["v"] - 0.036) < 5e-3
    assert abs(out["params"]["a1"] - 0.004) < 2e-3


def test_event_window_share_is_the_union_over_the_span():
    # two events 1.0 apart with a 0.25 horizon either side: two disjoint
    # windows of 0.5 in a span of 10
    assert filters.event_window_share(
        [2.0, 3.0], t0=0.0, t1=10.0, before=0.25, after=0.25
    ) == pytest.approx(0.1)
    # overlapping windows are counted once: [1.75, 2.25] and [1.95, 2.45]
    # union to 0.7, not to 1.0
    assert filters.event_window_share(
        [2.0, 2.2], t0=0.0, t1=10.0, before=0.25, after=0.25
    ) == pytest.approx(0.07)
    # a backward-looking rule covers half as much
    assert filters.event_window_share(
        [2.0, 3.0], t0=0.0, t1=10.0, before=0.0, after=0.25
    ) == pytest.approx(0.05)
    assert filters.event_window_share([], t0=0.0, t1=0.0) == 0.0


def test_reachable_events_excludes_the_blind_start_and_blind_period():
    reach = filters.reachable_events(
        [50, 200, 260, 400], [210], min_window=6, window=40,
    )
    # blind start is 6 + 3 * 40 = 126, and 260 is inside 120 samples of
    # the flag at 210
    assert reach == [False, True, False, True]
    assert filters.reachable_events([300], [], min_window=6,
                                    window=40) == [True]


def test_ratio_bin_uses_the_databases_own_threshold():
    assert filters.ratio_bin(10.0, 100.0) == "<0.25"
    assert filters.ratio_bin(30.0, 100.0) == "0.25-0.5"
    assert filters.ratio_bin(90.0, 100.0) == "0.5-1"
    assert filters.ratio_bin(120.0, 100.0) == ">=1"
    assert filters.ratio_bin(None, 100.0) == "none"
    assert filters.ratio_bin(10.0, 0.0) == "none"


def test_ngl_filter_station_reports_summaries_and_step_rows(tmp_path):
    path, step_year = ngl_station(tmp_path / "AAAA.tenv3")
    steps = [ngl.Step(sta="AAAA", ymd="10JAN01", year=step_year, code=2,
                      threshold_km=100.0, distance_km=30.0,
                      magnitude=6.2, event="usx")]
    summary, step_rows = filters.ngl_filter_station(path, steps)
    keys = {(r["component"], r["config"]) for r in summary}
    assert keys == {(c, k) for c in ngl.COMPONENTS
                    for k in ("detection", "tracking")}
    row = [r for r in summary
           if r["config"] == "detection" and r["component"] == "east"][0]
    assert row["n_updates"] == 1600 and row["n_steps"] == 1
    assert row["station_years"] > 4.0
    assert row["n_reachable"] == 1        # the step is at sample 800
    assert row["recall_reachable"] in (0.0, 1.0)
    assert 0.0 <= row["event_window_share"] <= 1.0
    assert row["chance_false_alarms_per_year"] is not None
    detect_rows = [r for r in step_rows if r["config"] == "detection"]
    assert len(detect_rows) == 3
    assert any(r["detected"] for r in detect_rows)
    hit = [r for r in detect_rows if r["detected"]][0]
    assert hit["mag_bin"] == "6-7" and hit["dist_bin"] == "20-100"
    assert hit["ratio_bin"] == "0.25-0.5"
    assert hit["dist_ratio"] == pytest.approx(0.3)
    assert hit["reachable"] is True and hit["step_index"] == 800
    assert hit["delay_days"] is not None and hit["delay_days"] <= 60.0
    assert set(filters.FILTER_COLUMNS) >= set(row)
    assert set(filters.STEP_COLUMNS) >= set(hit)


def test_dropped_times_reports_the_quality_failures(tmp_path):
    p = isd_fragment(tmp_path / "72278023183.csv")
    times = isd.dropped_times(p, "TMP")
    assert times["quality"].size == 1
    assert abs(float(times["quality"][0]) - (2 + 5 / 24.0)) < 1e-9
    assert times["missing"].size == 0


def test_isd_filter_station_flags_the_shift_and_explains_it(tmp_path):
    p = isd_fragment(tmp_path / "72278023183.csv")
    summary, flag_rows = filters.isd_filter_station(p)
    assert len(summary) == 1
    row = summary[0]
    assert row["station"] == "72278023183"
    assert row["n_updates"] > 400 and row["n_flags"] >= 1
    assert row["n_moves"] == 1 and row["n_quality_failures"] == 1
    # events, explained flags and unexplained flags: none of the three is
    # a step count or a detection count
    assert row["n_events"] == 2
    assert (row["n_flags_explained"] + row["n_flags_unexplained"]
            == row["n_flags"])
    assert 0.0 <= row["event_window_share"] <= 1.0
    assert "n_steps" not in row and "n_detected" not in row
    assert set(filters.ISD_FLAG_COLUMNS) >= set(flag_rows[0])
    assert set(filters.ISD_FILTER_COLUMNS) >= set(row)
    assert any(r["explained_by"] == "move" for r in flag_rows)
