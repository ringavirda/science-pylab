"""The EAC block preset, the LSI oscillatory recipe and the fused detector.

* ``fit_eac``, the uniform-window block preset, aimed at concentrated
  transients;
* the ``fit_lsi`` oscillatory recipe (``oscillatory=`` / ``freq_param=`` with
  ``fft_frequency_seed``), which recovers a sinusoid the default order at
  p0 does not resolve;
* ``FusedChiSquareDetector``, the multi-axis fault detector on a
  ``FilterBank``.
"""

import numpy as np
import pytest

from dtfit import (
    fit_lsi,
    fit_eac,
    fft_frequency_seed,
    FilterBank,
    FusedChiSquareDetector,
    LSIFilter,
)


def test_eac_recovers_transient():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 3, 400)
    y = 2.0 * (1 - np.exp(-3.0 * t)) + rng.normal(0, 0.02, t.size)
    r = fit_eac(t, y, "K*(1-exp(-a*x))", "x", p0=[1.0, 1.0])
    assert abs(r.coeffs[0] - 2.0) < 0.2 and abs(r.coeffs[1] - 3.0) < 0.5


def test_fft_frequency_seed_finds_dominant_cycle():
    t = np.linspace(0, 4 * np.pi, 400)
    y = 2.0 * np.sin(1.5 * t)
    assert fft_frequency_seed(t, y) == pytest.approx(1.5, rel=0.05)


def test_oscillatory_recipe_recovers_sine_where_default_fails():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    w_true = 1.5
    y = 2.0 * np.sin(w_true * t) + rng.normal(0, 0.05, t.size)

    osc = fit_lsi(t, y, "A*sin(w*x)", "x", freq_param="w", p0=[1.0, 1.0])
    names = ["A", "w"]  # sympy sorts the parameters, so A comes before w
    w_osc = osc.coeffs[names.index("w")]
    assert abs(w_osc - w_true) < 0.1

    # without the recipe the default order at p0 need not resolve the cycle
    plain = fit_lsi(t, y, "A*sin(w*x)", "x", p0=[1.0, 1.0])
    w_plain = plain.coeffs[names.index("w")]
    assert abs(w_osc - w_true) <= abs(w_plain - w_true)


def test_oscillatory_flag_raises_order_under_bounds():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6 * np.pi, 400)
    y = np.sin(2.0 * t) + rng.normal(0, 0.05, t.size)
    # the bounds path: a global search brackets the frequency; the recipe
    # still helps
    r = fit_lsi(t, y, "A*sin(w*x)", "x", oscillatory=True,
                bounds=[(0.1, 5.0), (0.5, 4.0)])
    assert abs(r.coeffs[1] - 2.0) < 0.2


def test_freq_param_unknown_raises():
    t = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="freq_param"):
        fit_lsi(t, np.sin(t), "A*sin(w*x)", "x", freq_param="omega")


OSC = "A*exp(-z*w*t)*sin(w*sqrt(1-z**2)*t)"


def _multiaxis(rng, n=600, fault_at=None, noise=0.03):
    fault_at = n // 2 if fault_at is None else fault_at
    t = np.linspace(0, 18, n)
    A = np.array([2.0, 1.5, 2.5])
    w = np.array([2.5, 2.0, 3.0])
    z1 = np.array([0.08, 0.10, 0.06])
    z2 = np.array([0.30, 0.28, 0.25])
    dtt = np.diff(t, prepend=t[0])
    Y = np.zeros((n, 3))
    for d in range(3):
        z = np.where(np.arange(n) < fault_at, z1[d], z2[d])
        wd = w[d] * np.sqrt(1 - z ** 2)
        Y[:, d] = A[d] * np.exp(-z * w[d] * t) * np.sin(np.cumsum(wd * dtt))
    return t, Y + rng.normal(0, noise, Y.shape), fault_at


def test_fused_detector_flags_multiaxis_fault():
    rng = np.random.default_rng(0)
    t, Y, fault_at = _multiaxis(rng, n=600)
    bank = FilterBank.from_model(
        OSC, "t", 3, filter_cls=LSIFilter, p0=[2.0, 2.5, 0.1],
        window_size=60, order=5, q_diag=[1e-3] * 3, r=0.5, adapt_r=True,
        cusum_h=np.inf)
    det = bank.fused_detector(alpha=1e-4, inflate=4.0)
    for i in range(Y.shape[0]):
        det.update(float(t[i]), Y[i])
    post = [i for i in det.flags_ if i >= fault_at]
    pre = [i for i in det.flags_ if i < fault_at]
    assert post, "fault after the regime change was not flagged"
    assert len(pre) == 0  # no false alarm before the fault
    assert det.threshold_ > 0 and det.n_flags_ == len(det.flags_)


def test_fused_detector_factory_matches_class():
    bank = FilterBank.from_model(
        OSC, "t", 2, filter_cls=LSIFilter, p0=[2.0, 2.5, 0.1],
        window_size=40, order=4)
    det = bank.fused_detector(alpha=1e-3)
    assert isinstance(det, FusedChiSquareDetector)
    assert det.k == 2
