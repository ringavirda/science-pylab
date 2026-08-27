"""FilterBank: a bank of independent streaming filters runs in parallel."""

from typing import Any

import numpy as np
import pytest

from dtfit.streaming import EACFilter, FilterBank


def _streams(K=5, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4, n)
    bs = np.linspace(0.5, 1.0, K)
    Y = np.column_stack(
        [1.0 * np.exp(b * t) + rng.normal(0, 0.02, n) for b in bs]
    )
    return t, Y, bs


def test_filter_bank_recovers_per_stream_params():
    t, Y, bs = _streams()
    bank = FilterBank.from_model(
        "a*exp(b*t)", "t", len(bs), p0=[1.0, 0.3], window_size=40,
        q_diag=[1e-4, 1e-3], r=0.3,
    )
    out = bank.run(t, Y, n_jobs=1)
    est_b = out["params"][:, 1]
    np.testing.assert_allclose(est_b, bs, atol=0.1)


def test_filter_bank_threaded_matches_serial():
    t, Y, bs = _streams()
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    a = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(t, Y, n_jobs=1)
    b = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(t, Y, n_jobs=4)
    np.testing.assert_allclose(a["params"], b["params"], rtol=1e-9, atol=1e-9)


def test_filter_bank_process_backend_matches_serial():
    """The process backend runs the filters in separate interpreters, free of
    the GIL. It must still reproduce the serial driver to 1e-12, with the drift
    counts exact and the tracking aligned."""
    t, Y, bs = _streams(K=4, n=300)
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    a = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(
        t, Y, n_jobs=1, track=True)
    b = FilterBank.from_model("a*exp(b*t)", "t", len(bs), **kw).run(
        t, Y, n_jobs=2, backend="process", track=True)
    np.testing.assert_allclose(a["params"], b["params"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a["n_drifts"], b["n_drifts"])
    np.testing.assert_allclose(
        np.nan_to_num(a["track"]), np.nan_to_num(b["track"]), rtol=1e-12, atol=1e-12)


def test_filter_bank_matches_standalone_filters():
    t, Y, bs = _streams(K=3)
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    bank = FilterBank.from_model("a*exp(b*t)", "t", 3, **kw)
    bank.run(t, Y, n_jobs=1)
    for k in range(3):
        flt = EACFilter("a*exp(b*t)", "t", **kw)
        for s in range(t.size):
            flt.partial_fit(t[s], Y[s, k])
        np.testing.assert_allclose(bank[k].p, flt.p, rtol=1e-9, atol=1e-9)


def test_filter_bank_skips_nan_sample_without_shape_corruption():
    """A NaN observation in one stream is skipped at ingestion, with a warning.
    The readout shapes stay intact and the poisoned stream still recovers its
    parameter, because the skipped step leaves its filter untouched."""
    t, Y, bs = _streams(K=3, n=300)
    Y[150, 1] = np.nan
    kw: dict[str, Any] = dict(
        p0=[1.0, 0.3], window_size=40, q_diag=[1e-4, 1e-3], r=0.3
    )
    bank = FilterBank.from_model("a*exp(b*t)", "t", 3, **kw)
    with pytest.warns(RuntimeWarning, match="non-finite sample skipped"):
        out = bank.run(t, Y, n_jobs=1, track=True)
    assert out["params"].shape == (3, 2)
    assert out["n_drifts"].shape == (3,)
    assert out["track"].shape == (300, 3)
    assert np.all(np.isfinite(out["params"]))
    np.testing.assert_allclose(out["params"][:, 1], bs, atol=0.1)


def test_filter_bank_predict_and_readout_shapes():
    t, Y, bs = _streams(K=4)
    bank = FilterBank.from_model("a*exp(b*t)", "t", 4, p0=[1.0, 0.3],
                                 window_size=40)
    bank.run(t, Y, n_jobs=1)
    assert bank.params_array().shape == (4, 2)
    assert bank.predict(t[:6]).shape == (4, 6)
    assert bank.predict(t[:1]).shape == (4,)
    assert len(bank.params_) == 4
