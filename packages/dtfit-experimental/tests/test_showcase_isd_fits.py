"""NOAA fits from stored images and the comparison against the published
normals, on synthetic data the test writes."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, isd, isd_fits, isd_reduce,
)

HEAD = ('"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION",'
        '"NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND","CIG",'
        '"VIS","TMP","DEW","SLP"\n')
ROW = ('"{sta}","{date}","4","70.9","-8.6","9.0","A STATION","FM-12",'
       '"99999","V020","318,1,N,0061,1","99999,9,9,9","999999,9,9,9",'
       '"{tmp},1","-0130,1","10208,1"\n')
# a1 a2 b1 b2 c v: an annual amplitude of 13 C peaking near day 200
TRUTH = [12.0, 2.0, -5.0, 1.0, 18.0, 0.0]
DIURNAL = 6.0
MONTHS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def date_of(year, day, hour, minute=53):
    dom, month = day + 1, 1
    lengths = MONTHS if isd.days_in_year(year) == 366 else (
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    )
    for m, length in enumerate(lengths, start=1):
        if dom <= length:
            month = m
            break
        dom -= length
    return f"{year}-{month:02d}-{dom:02d}T{hour:02d}:{minute:02d}:00"


def write_year(path, sta="72278023183", year=2024, seed=3, skew=0.0):
    """An hourly station-year at :53. ``skew`` adds a second diurnal
    harmonic, which is what makes a real day's range exceed twice its
    first-harmonic amplitude."""
    rng = np.random.default_rng(seed)
    days = isd.days_in_year(year)
    t = np.arange(0.0, days, 1.0 / 24.0) + 53.0 / 1440.0
    y = (isd_reduce.annual_design(t) @ np.array(TRUTH)
         + DIURNAL * np.cos(2.0 * np.pi * t)
         + skew * np.cos(4.0 * np.pi * t)
         + 0.5 * rng.standard_normal(t.size))
    with open(path, "w") as fh:
        fh.write(HEAD)
        for k in range(t.size):
            day = int(t[k])
            hour = int(round((t[k] - day) * 24.0 - 53.0 / 60.0))
            fh.write(ROW.format(
                sta=sta, date=date_of(year, day, min(max(hour, 0), 23)),
                tmp=f"{int(round(y[k] * 10)):+05d}",
            ))
    return path


def write_normals(path, sta="USW00023183", seed=4):
    """A reduced normals file for a 365-day climatology with the same
    annual and diurnal shape, in degrees Fahrenheit."""
    rng = np.random.default_rng(seed)
    months, doms, hours, when = [], [], [], []
    day = 0
    for month, length in enumerate(
        (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), start=1
    ):
        for dom in range(1, length + 1):
            for hour in range(24):
                months.append(month)
                doms.append(dom)
                hours.append(hour)
                when.append(day + hour / 24.0)
            day += 1
    t = np.array(when)
    c = (isd_reduce.annual_design(t) @ np.array(TRUTH)
         + DIURNAL * np.cos(2.0 * np.pi * t)
         + 0.05 * rng.standard_normal(t.size))
    degf = c * 9.0 / 5.0 + 32.0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(isd.NORMALS_COLUMNS)
        for k in range(t.size):
            w.writerow([sta, f"{months[k]:02d}", f"{doms[k]:02d}",
                        f"{hours[k]:02d}", f"{degf[k]:.1f}", "1013.0"])
    return path


def test_amp_phase_reads_a_cosine_pair():
    amp, phase = isd_fits.amp_phase(3.0, 4.0, 10.0)
    assert amp == pytest.approx(5.0)
    assert 0.0 <= phase < 10.0
    # the peak of 3 cos(wt) + 4 sin(wt) is where wt = atan2(4, 3)
    assert phase == pytest.approx(np.arctan2(4.0, 3.0) / (2 * np.pi) * 10.0)
    amp0, phase0 = isd_fits.amp_phase(0.0, 0.0, 10.0)
    assert amp0 == 0.0 and phase0 == 0.0


def test_month_of_day_handles_both_year_lengths():
    assert isd_fits.month_of_day(0, 2024) == (1, 1)
    assert isd_fits.month_of_day(59, 2024) == (2, 29)
    assert isd_fits.month_of_day(59, 2023) == (3, 1)
    assert isd_fits.month_of_day(365, 2024) == (12, 31)


def test_year_rows_recover_the_annual_amplitude_and_phase(tmp_path):
    src = write_year(tmp_path / "72278023183.csv")
    npz, _ = isd_reduce.reduce_year_to_file(src, tmp_path / "images")
    rows = isd_fits.year_rows(npz)
    assert len(rows) == 1
    row = rows[0]
    assert row["station"] == "72278023183" and row["field"] == "TMP"
    assert row["order"] == isd_reduce.ANNUAL_ORDER
    assert row["annual_amp"] == pytest.approx(13.0, abs=0.2)
    assert row["semi_amp"] == pytest.approx(np.hypot(2.0, 1.0), abs=0.2)
    assert 0.0 <= row["annual_phase_day"] < 366.0
    assert np.isfinite(row["bic"]) and row["rss"] > 0.0
    assert set(isd_fits.YEAR_COLUMNS) >= set(row)


def test_day_rows_recover_the_diurnal_amplitude(tmp_path):
    src = write_year(tmp_path / "72278023183.csv", seed=6)
    npz, _ = isd_reduce.reduce_year_to_file(
        src, tmp_path / "images", with_days=True
    )
    rows = isd_fits.day_rows(npz)
    assert len(rows) >= 350
    amps = np.array([r["diurnal_amp"] for r in rows])
    assert np.median(amps) == pytest.approx(DIURNAL, abs=0.3)
    assert set(isd_fits.DAY_COLUMNS) >= set(rows[0])
    assert all(0 <= r["month"] <= 12 for r in rows)


def test_day_rows_range_is_the_reconstruction_not_twice_the_amplitude(
    tmp_path,
):
    # A second diurnal harmonic makes the day's range exceed 2 * a1; the
    # normals comparison must see the reconstructed range, not the
    # doubled first harmonic.
    src = write_year(tmp_path / "72278023183.csv", seed=7, skew=3.0)
    npz, _ = isd_reduce.reduce_year_to_file(
        src, tmp_path / "images", with_days=True
    )
    rows = isd_fits.day_rows(npz)
    amps = np.median([r["diurnal_amp"] for r in rows])
    spans = np.median([r["diurnal_range"] for r in rows])
    assert spans > 2.0 * amps * 1.05


def test_exactness_year_meets_the_gate_on_the_year_and_the_days(tmp_path):
    src = write_year(tmp_path / "72278023183.csv", seed=8)
    npz, _ = isd_reduce.reduce_year_to_file(src, tmp_path / "images")
    row = isd_fits.exactness_year(src, npz)
    assert row["score"] <= compare.EXACTNESS_TOL
    assert row["coverage"] <= compare.COVERAGE_TOL
    # the day arm gates the channel form, so it needs no day images in
    # the .npz: it bins and projects the raw rows itself
    assert row["n_days_checked"] == isd_fits.DAY_GATE_SAMPLE
    assert row["day_score"] <= compare.EXACTNESS_TOL
    assert row["gate"] == "ok"
    assert set(isd_fits.EXACT_ISD_COLUMNS) >= set(row)


def test_exactness_year_leaves_the_day_arm_empty_without_full_days(
    tmp_path,
):
    src = write_year(tmp_path / "72278023183.csv", seed=12)
    npz, _ = isd_reduce.reduce_year_to_file(src, tmp_path / "images")
    row = isd_fits.exactness_year(src, npz, day_sample=0)
    assert row["day_score"] is None and row["n_days_checked"] == 0
    assert row["gate"] == "ok"


def test_normals_rows_compare_amplitude_phase_and_diurnal_range(tmp_path):
    src = write_year(tmp_path / "72278023183.csv", seed=9)
    npz, _ = isd_reduce.reduce_year_to_file(
        src, tmp_path / "images", with_days=True
    )
    nrm = write_normals(tmp_path / "USW00023183.csv")
    rows = isd_fits.normals_rows(npz, nrm)
    annual = [r for r in rows if r["kind"] == "annual"][0]
    assert annual["amp_image"] == pytest.approx(annual["amp_normals"],
                                                abs=0.5)
    # the normals are a 365-day climatology and the station-year is 2024;
    # mapping them through 2024's calendar removes the one-day offset
    assert abs(annual["phase_diff_days"]) < 2.0
    monthly = [r for r in rows if r["kind"] == "diurnal"]
    assert len(monthly) == 12
    for r in monthly:
        assert r["range_image"] == pytest.approx(r["range_normals"],
                                                 abs=1.0)
        assert r["amp_image"] > 0.0
    assert set(isd_fits.NORMALS_ROW_COLUMNS) >= set(annual)


def test_run_helpers_keep_the_input_order(tmp_path):
    paths = []
    for sta in ("72278023183", "72278023184"):
        src = write_year(tmp_path / f"{sta}.csv", sta=sta, seed=11)
        npz, _ = isd_reduce.reduce_year_to_file(src, tmp_path / "images")
        paths.append((src, npz))
    rows = isd_fits.run_year_fits([b for _, b in paths], workers=1)
    assert [r["station"] for r in rows] == ["72278023183", "72278023184"]
    exact = isd_fits.run_exactness_isd(paths, workers=1)
    assert all(r["gate"] == "ok" for r in exact)


def test_run_normals_fans_out_over_pairs(tmp_path):
    nrm = write_normals(tmp_path / "USW00023183.csv")
    pairs = []
    for sta in ("72278023183", "72278023184"):
        src = write_year(tmp_path / f"{sta}.csv", sta=sta, seed=13)
        npz, _ = isd_reduce.reduce_year_to_file(
            src, tmp_path / "images_n", with_days=True
        )
        pairs.append((npz, nrm))
    rows = isd_fits.run_normals(pairs, workers=1)
    annual = [r for r in rows if r["kind"] == "annual"]
    assert [r["station"] for r in annual] == [
        "72278023183", "72278023184"
    ]
    assert all(r["normals_id"] == "USW00023183" for r in rows)
    assert len([r for r in rows if r["kind"] == "diurnal"]) == 24
