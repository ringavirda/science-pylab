"""The NOAA reducers on a synthetic station-year the test writes."""

from __future__ import annotations

import numpy as np

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, isd, isd_reduce, store,
)

HEAD = ('"STATION","DATE","SOURCE","LATITUDE","LONGITUDE","ELEVATION",'
        '"NAME","REPORT_TYPE","CALL_SIGN","QUALITY_CONTROL","WND","CIG",'
        '"VIS","TMP","DEW","SLP"\n')
ROW = ('"{sta}","{date}","4","70.9","-8.6","9.0","A STATION","FM-12",'
       '"99999","V020","318,1,N,0061,1","99999,9,9,9","999999,9,9,9",'
       '"{tmp},1","-0130,1","{slp},1"\n')
TRUTH = [12.0, 2.0, -5.0, 1.0, 18.0, 0.001]      # a1 a2 b1 b2 c v


def synthetic_year(path, sta="72278023183", year=2024, seed=2,
                   diurnal=8.0, keep=0.95):
    """An hourly station-year at :53 with ``1 - keep`` of the rows absent,
    an annual and semiannual cycle, a diurnal cycle and noise. Returns the
    kept times and temperatures."""
    rng = np.random.default_rng(seed)
    days = isd.days_in_year(year)
    t = np.arange(0.0, days, 1.0 / 24.0) + 53.0 / 1440.0
    t = t[rng.random(t.size) > (1.0 - keep)]
    y = (isd_reduce.annual_design(t) @ np.array(TRUTH)
         + diurnal * np.cos(2.0 * np.pi * t - 1.0)
         + 2.0 * rng.standard_normal(t.size))
    with open(path, "w") as fh:
        fh.write(HEAD)
        for k in range(t.size):
            day = int(t[k])
            hour = int(round((t[k] - day) * 24.0 - 53.0 / 60.0))
            hour = min(max(hour, 0), 23)
            month, dom = 1, day + 1
            for m, length in enumerate(
                (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), start=1
            ):
                if dom <= length:
                    month = m
                    break
                dom -= length
            date = f"{year}-{month:02d}-{dom:02d}T{hour:02d}:53:00"
            fh.write(ROW.format(
                sta=sta, date=date, tmp=f"{int(round(y[k] * 10)):+05d}",
                slp=f"{int(round((1013.0 + 0.1 * y[k]) * 10)):05d}",
            ))
    return t, y


def synthetic_year_with_thin_day(path, sta="72278023183", year=2024,
                                 seed=17, thin_day=100, thin_keep=5):
    """A full hourly station-year except ``thin_day``, which keeps only
    ``thin_keep`` rows -- fewer than ``DIURNAL_ORDER + 2`` -- so it is the
    one day :func:`~isd_reduce.reduce_station_year` should drop."""
    rng = np.random.default_rng(seed)
    days = isd.days_in_year(year)
    t = np.arange(0.0, days, 1.0 / 24.0) + 53.0 / 1440.0
    day_of = t.astype(int)
    thin_idx = np.flatnonzero(day_of == thin_day)
    drop = rng.choice(thin_idx, size=thin_idx.size - thin_keep,
                      replace=False)
    keep = np.ones(t.size, dtype=bool)
    keep[drop] = False
    t = t[keep]
    y = (isd_reduce.annual_design(t) @ np.array(TRUTH)
         + 8.0 * np.cos(2.0 * np.pi * t - 1.0)
         + 2.0 * rng.standard_normal(t.size))
    with open(path, "w") as fh:
        fh.write(HEAD)
        for k in range(t.size):
            day = int(t[k])
            hour = int(round((t[k] - day) * 24.0 - 53.0 / 60.0))
            hour = min(max(hour, 0), 23)
            month, dom = 1, day + 1
            for m, length in enumerate(
                (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), start=1
            ):
                if dom <= length:
                    month = m
                    break
                dom -= length
            date = f"{year}-{month:02d}-{dom:02d}T{hour:02d}:53:00"
            fh.write(ROW.format(
                sta=sta, date=date, tmp=f"{int(round(y[k] * 10)):+05d}",
                slp=f"{int(round((1013.0 + 0.1 * y[k]) * 10)):05d}",
            ))
    return t, y


def test_reduce_station_year_meets_the_gate_on_the_annual_model(tmp_path):
    p = tmp_path / "72278023183.csv"
    synthetic_year(p)
    red = isd_reduce.reduce_station_year(p, ("TMP",))
    assert red.station == "72278023183" and red.year == 2024
    assert red.info["days"] == 366
    image = red.images["year_TMP"]
    assert image.order == isd_reduce.ANNUAL_ORDER
    assert image.domain == (0.0, 366.0)
    t, y, _ = isd.station_year(p, "TMP")
    ref = compare.raw_lstsq(isd_reduce.annual_design(t), y,
                            isd_reduce.ANNUAL_NAMES)
    res = compare.fit_from_image(isd_reduce.ANNUAL_EXPR, image,
                                 isd_reduce.ANNUAL_NAMES)
    assert compare.param_score(res.params, ref) <= compare.EXACTNESS_TOL
    assert abs(res.params["a1"] - 12.0) < 0.5


def test_reduce_station_year_can_add_the_daily_blocks(tmp_path):
    p = tmp_path / "72278023183.csv"
    synthetic_year(p)
    red = isd_reduce.reduce_station_year(p, ("TMP",), with_days=True)
    days = [k for k in red.images if k.startswith("day")]
    assert len(days) >= 350                       # a few days lose rows
    first = red.images[days[0]]
    assert first.order == isd_reduce.DIURNAL_ORDER
    assert red.info["dropped_days"] >= 0
    assert red.info["n_days"] == len(days)


def test_reduce_station_year_drops_a_day_with_too_few_samples(tmp_path):
    p = tmp_path / "72278023183.csv"
    synthetic_year_with_thin_day(p, thin_day=100, thin_keep=5)
    red = isd_reduce.reduce_station_year(p, ("TMP",), with_days=True)
    assert red.info["dropped_days"] == 1
    assert "day100_TMP" not in red.images


def test_reduce_year_to_file_round_trips(tmp_path):
    p = tmp_path / "72278023183.csv"
    synthetic_year(p, keep=0.6)
    out, red = isd_reduce.reduce_year_to_file(p, tmp_path / "images")
    images, info = store.load_images(out)
    assert images["year_TMP"] == red.images["year_TMP"]
    assert info["station"] == "72278023183"
    rows = isd_reduce.reduce_many_years([p], tmp_path / "images2",
                                        workers=1)
    assert rows[0]["station"] == "72278023183" and rows[0]["error"] == ""
    assert rows[0]["coef_bytes"] == 25 * 8
    assert rows[0]["gram_bytes"] == 25 * 25 * 8
    assert rows[0]["grid_bytes"] > 0
    assert set(isd_reduce.ISD_REDUCE_COLUMNS) >= set(rows[0])


def test_day_batch_images_every_station_on_one_grid(tmp_path):
    paths = []
    for k, sta in enumerate(("72278023183", "72278023184")):
        p = tmp_path / f"{sta}.csv"
        synthetic_year(p, sta=sta, seed=10 + k, keep=1.0)
        paths.append(p)
    ids, images, info = isd_reduce.day_batch(paths, 40, "TMP")
    assert ids == ["72278023183", "72278023184"]
    assert len(images) == 2
    assert info["n_stations"] == 2 and info["n_read"] == 2
    assert info["backend"] == "numpy"
    assert info["seconds_project"] >= 0.0
    for image, path in zip(images, paths):
        assert image.order == isd_reduce.DIURNAL_ORDER
        assert image.domain == (0.0, 1.0)
        grid = isd.day_grid(path, 40, "TMP")
        ref = compare.raw_lstsq(
            isd_reduce.diurnal_design(isd_reduce.DAY_POSITIONS), grid,
            isd_reduce.DIURNAL_NAMES,
        )
        res = compare.fit_from_image(
            isd_reduce.DIURNAL_EXPR, image, isd_reduce.DIURNAL_NAMES,
            slope=None,
        )
        assert compare.param_score(res.params, ref) <= compare.EXACTNESS_TOL


def test_day_batch_skips_a_station_with_an_incomplete_day(tmp_path):
    good = tmp_path / "72278023183.csv"
    synthetic_year(good, keep=1.0)
    thin = tmp_path / "72278023184.csv"
    synthetic_year(thin, sta="72278023184", keep=0.3, seed=21)
    ids, images, info = isd_reduce.day_batch([good, thin], 40, "TMP")
    assert ids == ["72278023183"] and info["n_read"] == 2
    assert info["n_stations"] == 1
