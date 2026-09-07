"""The NGL readers, on fixtures the test writes. No real file is read."""

from __future__ import annotations

import numpy as np
import pytest

from dtfit_experimental.experiments.domains.image_showcase import ngl

HEADER = (
    "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m)"
    " _north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m)"
    " __corr_en __corr_eu __corr_nu _latitude(deg) _longitude(deg)"
    " __height(m)\n"
)
ROW = (
    "00NA {ymd} {dec} {mjd} 1472 4  130.8   4781  {east}  -1378706"
    " {north}   104  {up}  0.0000 0.001193 0.001135 0.005615  0.047261"
    "  0.011975 -0.050796 -12.4666414929 -229.1560136341   104.83888\n"
)


def write_tenv3(path, rows):
    """rows: (ymd, decimal year, mjd, east frac, north frac, up frac)."""
    with open(path, "w") as fh:
        fh.write(HEADER)
        for ymd, dec, mjd, e, n, u in rows:
            fh.write(ROW.format(ymd=ymd, dec=dec, mjd=mjd, east=e,
                                north=n, up=u))
    return path


def test_decimal_year_matches_the_files_own_column():
    # The tenv3 decimal year is 2000 + (MJD - 51544) / 365.25; both pairs
    # are copied from real header examples in README_tenv3.txt.
    assert ngl.decimal_year("08MAR27") == pytest.approx(2008.2355, abs=5e-5)
    assert ngl.decimal_year("10JUL28") == pytest.approx(2010.5708, abs=5e-5)
    assert ngl.decimal_year("94JAN17") == pytest.approx(1994.0452, abs=5e-5)
    with pytest.raises(ValueError):
        ngl.decimal_year("08XXX27")


def test_read_tenv3_parses_positions_sigmas_and_drops_repeats(tmp_path):
    p = write_tenv3(tmp_path / "00NA.tenv3", [
        ("08MAR27", "2008.2355", "54552", " 0.834463", "-0.289444",
         " 0.838880"),
        ("08MAR28", "2008.2382", "54553", " 0.836223", "-0.284531",
         " 0.851196"),
        ("08MAR28", "2008.2382", "54553", " 0.900000", "-0.200000",
         " 0.900000"),
        ("08MAR29", "2008.2410", "54554", " 0.840000", "-0.280000",
         " 0.860000"),
    ])
    chunks = list(ngl.read_tenv3(p))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.t.size == 3
    np.testing.assert_allclose(c.t, [2008.2355, 2008.2382, 2008.2410])
    # the first row's integer metres are removed and travel as offsets
    np.testing.assert_allclose(c.east, [0.834463, 0.836223, 0.840000])
    np.testing.assert_allclose(c.north, [-0.289444, -0.284531, -0.280000])
    np.testing.assert_allclose(c.up, [0.838880, 0.851196, 0.860000])
    assert c.offset_e == 4781.0
    assert c.offset_n == -1378706.0
    assert c.offset_u == 104.0
    np.testing.assert_allclose(c.sig_e, [0.001193] * 3)
    np.testing.assert_allclose(c.sig_u, [0.005615] * 3)


def test_read_tenv3_keeps_the_series_continuous_across_an_integer_jump(
    tmp_path,
):
    # The integer column re-initialises after a jump of more than 10 m;
    # subtracting the first row's integer, not each row's, keeps the
    # series continuous.
    p = tmp_path / "JUMP.tenv3"
    with open(p, "w") as fh:
        fh.write(HEADER)
        for dec, mjd, e0, frac in (
            ("2008.2355", "54552", "4781", " 0.900000"),
            ("2008.2382", "54553", "4791", " 0.100000"),
            ("2008.2410", "54554", "4791", " 0.300000"),
        ):
            fh.write(
                f"00NA 08MAR27 {dec} {mjd} 1472 4  130.8   {e0} {frac}"
                " -1378706 -0.289444   104  0.838880  0.0000"
                " 0.001193 0.001135 0.005615  0.047261  0.011975"
                " -0.050796 -12.4666414929 -229.1560136341   104.83888\n"
            )
    c = next(iter(ngl.read_tenv3(p)))
    np.testing.assert_allclose(c.east, [0.9, 10.1, 10.3])
    assert c.offset_e == 4781.0


def test_read_tenv3_chunks_and_dedupes_across_the_boundary(tmp_path):
    rows = []
    for k in range(7):
        rows.append((f"08APR{k + 1:02d}", f"{2008.3 + k * 0.0027:.4f}",
                     str(54560 + k), f" {0.8 + k * 0.001:.6f}",
                     "-0.289444", " 0.838880"))
    rows.insert(3, rows[2])          # a repeated epoch at a chunk edge
    rows.insert(5, rows[1])          # and one that goes backwards
    p = write_tenv3(tmp_path / "AAAA.tenv3", rows)
    chunks = list(ngl.read_tenv3(p, chunk=3))
    assert len(chunks) == 3
    t = np.concatenate([c.t for c in chunks])
    assert t.size == 7 and np.all(np.diff(t) > 0)
    np.testing.assert_allclose(t, ngl.read_epochs(p, chunk=3))


def test_read_tenv3_rejects_a_row_with_the_wrong_column_count(tmp_path):
    p = tmp_path / "BAD.tenv3"
    p.write_text(HEADER + "00NA 08MAR27 2008.2355 54552\n")
    with pytest.raises(ValueError, match="23 columns"):
        list(ngl.read_tenv3(p))


def test_read_steps_splits_equipment_from_earthquakes(tmp_path):
    p = tmp_path / "steps.txt"
    p.write_text(
        "00NA  17MAY04  1  Antenna_Type_Changed\n"
        "00NA  17MAY04  1  Receiver_Make_and_Model_Changed\n"
        "GOL2  94JAN17  2   363.078   201.965  6.7 ci3144585\n"
    )
    steps = ngl.read_steps(p)
    assert sorted(steps) == ["00NA", "GOL2"]
    assert len(steps["00NA"]) == 2
    eq = steps["GOL2"][0]
    assert eq.code == 2 and eq.magnitude == pytest.approx(6.7)
    assert eq.distance_km == pytest.approx(201.965)
    assert eq.threshold_km == pytest.approx(363.078)
    assert eq.event == "ci3144585"
    assert eq.year == pytest.approx(1994.0452, abs=5e-5)
    eqp = steps["00NA"][0]
    assert eqp.code == 1 and eqp.label == "Antenna_Type_Changed"
    assert eqp.magnitude is None and eqp.distance_km is None


def test_read_midas_takes_the_velocities_and_their_uncertainties(tmp_path):
    p = tmp_path / "midas.IGS20.txt"
    p.write_text(
        "00NA MIDAS5 2008.2355 2018.7324 10.4969  3185  2964   5167"
        "   0.036184   0.058504  -0.000718  0.000234 0.000267 0.000892"
        "   4781.8336  -1378706.2819  104.8530 0.072 0.114 0.080"
        " 0.002151 0.002403 0.008181   1 -12.4666386578 -229.1560119207"
        "   104.84908\n"
    )
    m = ngl.read_midas(p)["00NA"]
    assert m.ve == pytest.approx(0.036184) and m.se == pytest.approx(0.000234)
    assert m.vn == pytest.approx(0.058504) and m.vu == pytest.approx(-0.000718)
    assert m.span == pytest.approx(10.4969) and m.n_steps == 1
    assert m.n_epochs == 3185 and m.n_good == 2964


def test_read_holdings_gives_the_station_index(tmp_path):
    p = tmp_path / "DataHoldings.txt"
    p.write_text(
        "Sta  Lat(deg)   Long(deg) Hgt(m)  X(m)           Y(m)"
        "         Z(m)          Dtbeg      Dtend      Dtmod      NumSol"
        " StaOrigName\n"
        "00NA -12.4666   130.8440  104.851 -4073662.2759  4712064.7454"
        " -1367874.5096 2008-03-27 2018-09-25 2025-09-25   3185\n"
    )
    h = ngl.read_holdings(p)["00NA"]
    assert h.lat == pytest.approx(-12.4666) and h.n_sol == 3185
    assert h.first == "2008-03-27" and h.last == "2018-09-25"


def test_station_files_selects_and_limits(tmp_path):
    for name in ("AAAA", "BBBB", "CCCC"):
        (tmp_path / f"{name}.tenv3").write_text(HEADER)
    assert [p.stem for p in ngl.station_files(tmp_path)] == [
        "AAAA", "BBBB", "CCCC"
    ]
    assert [p.stem for p in ngl.station_files(tmp_path, limit=2)] == [
        "AAAA", "BBBB"
    ]
    got = ngl.station_files(tmp_path, stations=["CCCC", "ZZZZ"])
    assert [p.stem for p in got] == ["CCCC"]
