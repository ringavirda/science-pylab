"""The showcase's foundation: paths, the image container, the exactness
comparison. No test here reads a real dataset."""

from __future__ import annotations

import csv
import dataclasses

import numpy as np
import pytest

from dtfit.image import Image, Original

from dtfit_experimental.experiments.domains.image_showcase import (
    compare, paths, store,
)

EXPR = ("c + v*t + a1*cos(2*pi*t) + b1*sin(2*pi*t)"
        " + a2*cos(4*pi*t) + b2*sin(4*pi*t)")
NAMES = ["a1", "a2", "b1", "b2", "c", "v"]


def design(t):
    """The NGL model's columns in canonical (sorted) parameter order."""
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        np.cos(2 * np.pi * t), np.cos(4 * np.pi * t),
        np.sin(2 * np.pi * t), np.sin(4 * np.pi * t),
        np.ones_like(t), t,
    ])


def irregular_series(span, seed, offset=0.83):
    """Daily samples over ``span`` years with 12 percent missing days and a
    four-month outage: the NGL sampling shape, without NGL data.

    ``offset`` is the intercept at the span's start. The default is metre
    scale because ruling 2 has the reader subtract the first row's integer
    metre column, so that is what the reducer actually sees."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, span, 1.0 / 365.25)
    t = t[rng.random(t.size) > 0.12]
    mid = span / 2.0
    t = t[(t < mid) | (t > mid + 0.33)]
    t = t - t[0]
    truth = np.array([0.004, 0.001, -0.002, 0.0005, offset, 0.036])
    y = design(t) @ truth + 0.002 * rng.standard_normal(t.size)
    return t, y


def test_paths_follow_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SHOWCASE_DATA", str(tmp_path / "d"))
    monkeypatch.delenv("SHOWCASE_ISD", raising=False)
    assert paths.data_root() == tmp_path / "d"
    assert paths.ngl_dir() == tmp_path / "d" / "ngl"
    assert paths.isd_dir() == tmp_path / "d" / "isd"
    assert paths.isd_dir(2024) == tmp_path / "d" / "isd" / "2024"
    assert paths.images_dir() == tmp_path / "d" / "images"
    assert paths.normals_dir() == tmp_path / "d" / "normals"
    monkeypatch.setenv("SHOWCASE_ISD", str(tmp_path / "elsewhere"))
    assert paths.isd_dir(2024) == tmp_path / "elsewhere" / "2024"
    assert paths.results_dir().is_dir()
    assert paths.figures_dir().is_dir()
    assert paths.domain_dir().name == "image_showcase"


def test_images_round_trip_through_the_npz_container(tmp_path):
    t, y = irregular_series(4.0, 1)
    explicit = Image.of(Original(t, y), "legendre", 48)
    xu = np.linspace(0.0, 1.0, 200)
    uniform = Image.of(Original(xu, np.sin(xu)), "legendre", 12)
    out = store.save_images(
        tmp_path / "sub" / "STA.npz",
        {"whole_east": explicit, "blk00_east": uniform},
        info={"sta": "STA", "span": 4.0},
    )
    assert out.exists()
    images, info = store.load_images(out)
    assert list(images) == ["whole_east", "blk00_east"]
    assert images["whole_east"] == explicit
    assert images["blk00_east"] == uniform
    assert images["blk00_east"].grid.kind == "uniform"
    assert info == {"sta": "STA", "span": 4.0}


def test_weighted_image_round_trips_with_its_weights(tmp_path):
    x = np.linspace(0.0, 1.0, 60)
    w = np.linspace(1.0, 2.0, 60)
    img = Image.of(Original(x, np.cos(x), w), "legendre", 8)
    images, info = store.load_images(
        store.save_images(tmp_path / "w.npz", {"a": img})
    )
    assert images["a"] == img
    assert images["a"].w is not None
    assert info == {}


def test_image_nbytes_splits_S_from_G_from_the_grid():
    t, y = irregular_series(3.0, 2)
    img = Image.of(Original(t, y), "legendre", 40)
    coef, gram, grid = store.image_nbytes(img)
    assert coef == 41 * 8
    assert gram == 41 * 41 * 8
    assert grid == img.n * 8
    xu = np.linspace(0.0, 1.0, 50)
    coef_u, gram_u, grid_u = store.image_nbytes(
        Image.of(Original(xu, xu ** 2), "legendre", 6)
    )
    assert coef_u == 7 * 8 and gram_u == 49 * 8 and grid_u == 0


def test_write_table_writes_a_header_and_the_rows(tmp_path):
    p = store.write_table(
        tmp_path / "t.csv",
        [{"sta": "AAAA", "score": 1e-12}, {"sta": "BBBB"}],
        columns=["sta", "score"],
    )
    with open(p, newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["sta", "score"]
    assert rows[1] == ["AAAA", "1e-12"]
    assert rows[2] == ["BBBB", ""]
    empty = store.write_table(tmp_path / "e.csv", [], columns=["a", "b"])
    with open(empty, newline="") as fh:
        assert list(csv.reader(fh)) == [["a", "b"]]


def test_param_score_is_relative_above_the_floor():
    ref = {"c": 100.0, "v": 5.0}
    assert compare.param_score(dict(ref), ref) == 0.0
    # v doubles, so the worst relative miss is 1.0, not c's 1e-2: a
    # parameter above the floor is scored against itself
    got = compare.param_score({"c": 101.0, "v": 10.0}, ref)
    assert got == pytest.approx(1.0, rel=1e-12)
    assert compare.worst_param({"c": 101.0, "v": 10.0}, ref) == "v"
    # a parameter below one percent of the largest is scored against
    # that one-percent level, a reference of exactly zero included
    small = {"c": 100.0, "v": 1e-4}
    assert compare.param_score({"c": 100.0, "v": 1e-4 + 1e-9}, small) == (
        pytest.approx(1e-9 / (compare.SCORE_FLOOR * 100.0))
    )
    zero = {"c": 100.0, "v": 0.0}
    assert compare.param_score({"c": 100.0, "v": 1e-10}, zero) == (
        pytest.approx(1e-10 / (compare.SCORE_FLOOR * 100.0))
    )
    assert compare.param_score({"a": 0.0}, {"a": 0.0}) == 0.0
    assert compare.param_score({"a": 5.0}, {"a": 0.0}) == float("inf")
    with pytest.raises(KeyError):
        compare.param_score({"b": 0.0}, {"a": 0.0})


def test_legendre_order_follows_the_measured_rule():
    assert compare.legendre_order(2.0, 500) == 32
    assert compare.legendre_order(11.0, 3400) == 104
    assert compare.legendre_order(32.0, 10200) == 272
    assert compare.legendre_order(0.5, 100) == 20
    assert compare.legendre_order(0.1, 100) == 17
    # the density floor bites before the n - 2 cap
    assert compare.legendre_order(32.0, 60) == 15
    assert compare.legendre_order(11.0, 200) == 50
    # the density cap on a series too short for any model
    assert compare.legendre_order(32.0, 8) == 2


def test_gram_rebuild_error_is_at_rounding_for_a_direct_image():
    t, y = irregular_series(3.0, 12)
    img = Image.of(Original(t, y), "legendre", 40)
    assert compare.gram_rebuild_error(img) < 1e-12
    # a perturbed G is reported at the size of the perturbation, and an
    # all-zero G as infinite
    bent = dataclasses.replace(img, G=img.G * (1.0 + 1e-6))
    assert compare.gram_rebuild_error(bent) == pytest.approx(1e-6, rel=1e-3)
    empty = dataclasses.replace(img, G=np.zeros_like(img.G))
    assert compare.gram_rebuild_error(empty) == float("inf")


def test_raw_bic_matches_the_library_formula():
    t, y = irregular_series(3.0, 14)
    X = design(t)
    rss, bic = compare.raw_bic(X, y)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    expected = float(np.sum((y - X @ beta) ** 2))
    assert rss == pytest.approx(expected, rel=1e-12)
    n, k = t.size, X.shape[1]
    assert bic == pytest.approx(
        n * np.log(rss / n) + k * np.log(n), rel=1e-12
    )


def test_image_fit_reproduces_the_raw_least_squares_at_the_gate():
    for span, seed, offset in ((3.0, 5, 0.83), (11.0, 6, -2.47)):
        t, y = irregular_series(span, seed, offset)
        order = compare.legendre_order(span, t.size)
        img = Image.of(Original(t, y), "legendre", order)
        ref = compare.raw_lstsq(design(t), y, NAMES)
        res = compare.fit_from_image(EXPR, img, NAMES)
        assert compare.param_score(res.params, ref) <= compare.EXACTNESS_TOL


def test_an_absolute_coordinate_would_defeat_the_gate():
    # The reader subtracts the first row's integer metre column. Kept, an
    # ALBH-scale north coordinate leaves the small parameters far above
    # 1e-8 of themselves while the score, floored at one percent of that
    # coordinate, no longer sees it: the gate would pass for the wrong
    # reason, which is why the subtraction exists.
    t, y = irregular_series(11.0, 6, 5361769.0)
    img = Image.of(Original(t, y), "legendre",
                   compare.legendre_order(11.0, t.size))
    ref = compare.raw_lstsq(design(t), y, NAMES)
    res = compare.fit_from_image(EXPR, img, NAMES)
    own = max(abs(res.params[k] - ref[k]) / abs(ref[k])
              for k in ("v", "a1", "b1"))
    assert own > compare.EXACTNESS_TOL
    assert compare.param_score(res.params, ref) <= compare.EXACTNESS_TOL


def test_p0_comes_from_the_image_alone():
    t, y = irregular_series(5.0, 7)
    img = Image.of(Original(t, y), "legendre", 56)
    p0 = compare.p0_from_image(img, NAMES)
    assert set(p0) == set(NAMES)
    assert p0["a1"] == 0.0 and p0["b2"] == 0.0
    assert p0["c"] == pytest.approx(0.83, abs=0.05)
    assert p0["v"] == pytest.approx(0.036, abs=0.01)
    no_slope = compare.p0_from_image(img, NAMES, slope=None)
    assert no_slope["v"] == 0.0
