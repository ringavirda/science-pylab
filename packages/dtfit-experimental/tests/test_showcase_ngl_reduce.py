"""The NGL reducer on a synthetic station file the test writes."""

from __future__ import annotations

import numpy as np
import pytest

from dtfit.image import Image, Original

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, ngl_reduce, store,
)

HEADER = (
    "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0 _e _n0 _n _u0 _u"
    " _ant sig_e sig_n sig_u c_en c_eu c_nu lat lon hgt\n"
)
# The reader subtracts the first row's integer metre column, and these
# fixtures keep that column constant, so the values it returns are exactly
# the fractional columns written below.
INTEGER = {"east": 4781.0, "north": -1378706.0, "up": 104.0}


def write_station(path, t, east, north, up):
    """A tenv3 file whose decimal-year column is ``t`` and whose position
    columns are ``4781 + east``, ``-1378706 + north``, ``104 + up``."""
    with open(path, "w") as fh:
        fh.write(HEADER)
        for k in range(t.size):
            fh.write(
                f"AAAA 08MAR27 {t[k]:.4f} 54552 1472 4 130.8"
                f" 4781 {east[k]:.6f} -1378706 {north[k]:.6f}"
                f" 104 {up[k]:.6f} 0.0000 0.001193 0.001135 0.005615"
                f" 0.047261 0.011975 -0.050796 -12.4 -229.1 104.8\n"
            )
    return path


def synthetic(span=4.0, seed=3, start=2008.0):
    """Daily epochs over ``span`` years with 10 percent missing days, and
    three components with a trend and annual and semiannual cycles."""
    rng = np.random.default_rng(seed)
    t = start + np.arange(0.0, span, 1.0 / 365.25)
    t = np.round(t[rng.random(t.size) > 0.10], 4)
    t = t[np.diff(t, prepend=t[0] - 1.0) > 0]
    tc = t - t[0]
    truth = {
        "east": [0.004, 0.001, -0.002, 0.0005, 0.83, 0.036],
        "north": [-0.003, 0.0008, 0.001, -0.0004, -0.28, 0.058],
        "up": [0.006, 0.002, 0.003, 0.001, 0.84, -0.0007],
    }
    cols = {}
    for name, p in truth.items():
        # Rounded to the six decimals the file format writes, so a
        # comparison against these columns sees exactly the samples the
        # reader will return.
        cols[name] = np.round(
            ngl_reduce.ngl_design(tc) @ np.array(p)
            + 0.002 * rng.standard_normal(tc.size), 6
        )
    return t, cols


def test_segment_bounds_splits_at_the_first_epoch_on_or_after_a_step():
    tc = np.arange(0.0, 10.0, 1.0 / 365.25)
    segs, skipped = ngl_reduce.segment_bounds(tc, [3.0, 3.5, 7.0])
    assert skipped == 1                      # the 0.5-year middle segment
    assert len(segs) == 3
    assert segs[0][0] == 0
    assert tc[segs[0][1] - 1] < 3.0 <= tc[segs[0][1]]
    assert tc[segs[1][0]] >= 3.5 and tc[segs[1][1] - 1] < 7.0
    assert segs[-1][1] == tc.size


def test_segment_bounds_without_steps_is_one_segment():
    tc = np.arange(0.0, 5.0, 1.0 / 365.25)
    segs, skipped = ngl_reduce.segment_bounds(tc, [])
    assert segs == [(0, tc.size)] and skipped == 0


def test_segment_bounds_ignores_steps_outside_the_span():
    tc = np.arange(0.0, 5.0, 1.0 / 365.25)
    segs, skipped = ngl_reduce.segment_bounds(tc, [-2.0, 9.0])
    assert segs == [(0, tc.size)] and skipped == 0


def test_segment_bounds_skips_a_segment_with_too_few_epochs():
    tc = np.concatenate([np.arange(0.0, 4.0, 1.0 / 365.25),
                         np.arange(6.0, 8.0, 1.0 / 12.0)])
    segs, skipped = ngl_reduce.segment_bounds(tc, [6.0])
    # the second piece spans two years but holds 24 epochs
    assert len(segs) == 1 and skipped == 1


def test_reduce_station_matches_the_direct_image_and_the_raw_fit(tmp_path):
    t, cols = synthetic()
    p = write_station(tmp_path / "AAAA.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    red = ngl_reduce.reduce_station(p)
    tc = t - t[0]
    order = compare.legendre_order(float(tc[-1]), tc.size)
    assert red.info["order"] == order
    assert red.info["n"] == tc.size
    assert red.info["offsets"] == INTEGER
    for comp in ("east", "north", "up"):
        img = red.images[f"whole_{comp}"]
        y = cols[comp]
        direct = Image.of(Original(tc, y), "legendre", order)
        rel = np.max(np.abs(img.S - direct.S)) / np.max(np.abs(direct.S))
        assert rel < 1e-10
        ref = compare.raw_lstsq(ngl_reduce.ngl_design(tc), y,
                                ngl_reduce.NGL_NAMES)
        res = compare.fit_from_image(ngl_reduce.NGL_EXPR, img,
                                     ngl_reduce.NGL_NAMES)
        assert compare.param_score(res.params, ref) <= compare.EXACTNESS_TOL


def test_reduce_station_emits_yearly_blocks_and_segment_images(tmp_path):
    t, cols = synthetic(span=4.0, seed=5)
    p = write_station(tmp_path / "BBBB.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    step = float(t[0]) + 2.0
    red = ngl_reduce.reduce_station(p, step_years=[step])
    blocks = [k for k in red.images if k.startswith("blk") and
              k.endswith("_east")]
    # the index in the key is the block's own domain, not its position in
    # the list, so a dropped year leaves a gap rather than a shift
    assert blocks == [f"blk{j:02d}_east" for j in range(4)]
    assert red.images[blocks[0]].order == ngl_reduce.BLOCK_ORDER
    assert red.images[blocks[0]].domain == (0.0, 1.0)
    segs = [k for k in red.images if k.startswith("seg") and
            k.endswith("_east")]
    assert len(segs) == 2 and red.info["n_segments"] == 2
    s0 = red.info["segments"][0]
    assert s0["order"] == compare.legendre_order(s0["span"], s0["n"])
    assert set(red.info["flags"]) == {"east", "north", "up"}


def test_reduce_station_drops_a_block_with_too_few_epochs(tmp_path):
    # A middle year thinned below BLOCK_ORDER + 2 = 14 samples closes
    # through the mid-stream skip path (the trailing block, if it were
    # thin, would just be left unimaged by close() and not counted): the
    # block is discarded and its year leaves a gap in the block keys.
    t, cols = synthetic(span=5.0, seed=6)
    tc = t - t[0]
    sparse = (tc >= 2.0) & (tc < 3.0)
    keep = ~sparse
    keep[np.where(sparse)[0][:5]] = True
    t = t[keep]
    cols = {k: v[keep] for k, v in cols.items()}
    p = write_station(tmp_path / "KKKK.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    red = ngl_reduce.reduce_station(p)
    assert red.info["dropped"] == {"east": 1, "north": 1, "up": 1}
    blocks = sorted(k for k in red.images
                    if k.startswith("blk") and k.endswith("_east"))
    assert "blk02_east" not in blocks


def test_reduce_station_flags_a_drift_past_the_detector_warmup(tmp_path):
    # DriftDetector warms up over 20 blocks; at block_len=1.0 that is 20
    # years, so only a station past roughly that span can ever raise a
    # flag. A 5 m step (2500x the 0.002 noise amplitude) at year 25 of a
    # 30-year station is past warmup and must flag; unflagged components
    # carry no step and must not.
    t, cols = synthetic(span=30.0, seed=13)
    tc = t - t[0]
    jump = tc >= 25.0
    cols = dict(cols)
    cols["up"] = cols["up"].copy()
    cols["up"][jump] += 5.0
    p = write_station(tmp_path / "HHHH.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    red = ngl_reduce.reduce_station(p)
    assert red.info["flags"]["up"] == [[25.0, 25.0, 26.0]]
    assert red.info["flags"]["east"] == []
    assert red.info["flags"]["north"] == []


def test_reduce_station_matches_across_a_chunk_boundary_inside_a_segment(
    tmp_path,
):
    # 1057 of 21798 real MIDAS stations exceed TENV3_CHUNK; a chunk this
    # small crosses both segment boundaries, exercising the cross-chunk
    # slice ``a, b = max(i0, pos), min(i1, pos + m)`` that a single-chunk
    # run never reaches.
    t, cols = synthetic(span=6.0, seed=8)
    p = write_station(tmp_path / "JJJJ.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    step = float(t[0]) + 3.0
    one = ngl_reduce.reduce_station(p, step_years=[step], chunk=100_000)
    many = ngl_reduce.reduce_station(p, step_years=[step], chunk=250)
    assert set(one.images) == set(many.images)
    for key, img in one.images.items():
        other = many.images[key]
        assert np.allclose(img.S, other.S, atol=1e-9, rtol=1e-9)
        assert np.allclose(img.G, other.G, atol=1e-9, rtol=1e-9)


def test_block_keys_follow_the_domain_across_a_missing_year(tmp_path):
    t, cols = synthetic(span=5.0, seed=17)
    tc = t - t[0]
    gap = (tc < 2.0) | (tc > 3.0)         # the third year is missing
    t = t[gap]
    cols = {k: v[gap] for k, v in cols.items()}
    p = write_station(tmp_path / "GGGG.tenv3", t, cols["east"],
                      cols["north"], cols["up"])
    red = ngl_reduce.reduce_station(p)
    blocks = sorted(k for k in red.images
                    if k.startswith("blk") and k.endswith("_east"))
    assert "blk02_east" not in blocks
    for key in blocks:
        img = red.images[key]
        assert int(key[3:5]) == int(round(img.domain[0]))


def test_reduce_station_rejects_a_station_that_is_too_short(tmp_path):
    t = 2008.0 + np.arange(10) / 365.25
    z = np.zeros(10)
    p = write_station(tmp_path / "CCCC.tenv3", t, z, z, z)
    with pytest.raises(ValueError, match="epochs"):
        ngl_reduce.reduce_station(p)


def test_reduce_to_file_and_reduce_many_write_readable_images(tmp_path):
    t, cols = synthetic(span=3.0, seed=9)
    src = tmp_path / "src"
    src.mkdir()
    p = write_station(src / "DDDD.tenv3", t, cols["east"], cols["north"],
                      cols["up"])
    out = tmp_path / "images"
    path, red = ngl_reduce.reduce_to_file(p, out)
    images, info = store.load_images(path)
    assert images["whole_east"] == red.images["whole_east"]
    assert info["sta"] == "DDDD"
    rows = ngl_reduce.reduce_many([p], out, {"DDDD": []}, workers=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["sta"] == "DDDD" and row["error"] == ""
    assert row["n"] == red.info["n"]
    assert row["coef_bytes"] > 0 and row["gram_bytes"] > 0
    assert row["grid_bytes"] > 0
    # G is the order squared and dominates S for any real station
    assert row["gram_bytes"] > row["coef_bytes"]
    assert row["raw_bytes"] == p.stat().st_size
