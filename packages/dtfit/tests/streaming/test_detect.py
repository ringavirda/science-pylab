import numpy as np
import pytest

from dtfit.streaming import DriftDetector


def _run(det, innovations):
    return [det.update(e) for e in innovations]


def test_white_innovations_rarely_alarm():
    rng = np.random.default_rng(0)
    det = DriftDetector(5)
    flags = _run(det, rng.standard_normal((300, 5)))
    assert sum(flags) <= 1
    assert det.n_drifts_ == sum(flags)


def test_energy_jump_is_flagged_after_warmup():
    rng = np.random.default_rng(1)
    det = DriftDetector(5, alpha=0.001)
    _run(det, rng.standard_normal((30, 5)))
    assert det.n_drifts_ == 0
    assert det.update(np.full(5, 4.0)) is True
    assert det.flag_ and det.n_drifts_ == 1


def test_sustained_shift_in_first_component_is_flagged_by_cusum():
    rng = np.random.default_rng(2)
    det = DriftDetector(5, alpha=1e-9, cusum_k=0.5, cusum_h=5.0)
    _run(det, rng.standard_normal((40, 5)))
    shifted = rng.standard_normal((40, 5))
    shifted[:, 0] += 1.5
    flags = _run(det, shifted)
    assert any(flags)
    first = flags.index(True)
    assert first < 25
    assert det.last_direction_ == 1


def test_reset_and_warmup_after_a_detection():
    rng = np.random.default_rng(3)
    det = DriftDetector(3, warmup=10)
    _run(det, rng.standard_normal((30, 3)))
    det.update(np.full(3, 6.0))
    assert det.n_drifts_ == 1 and det.n_tests_ == 0
    flags = _run(det, rng.standard_normal((10, 3)))
    assert not any(flags)


def test_state_round_trip_reproduces_flags():
    rng = np.random.default_rng(4)
    inno = rng.standard_normal((60, 4))
    inno[40:, 0] += 2.0
    a = DriftDetector(4)
    _run(a, inno[:30])
    b = DriftDetector(4)
    b.restore(a.state())
    fa = _run(a, inno[30:])
    fb = _run(b, inno[30:])
    assert fa == fb and a.state() == b.state()


def test_bad_input_raises():
    det = DriftDetector(3)
    with pytest.raises(ValueError, match="3"):
        det.update(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        det.update(np.array([0.0, np.nan, 0.0]))
    with pytest.raises(ValueError):
        DriftDetector(0)
