"""The image's own statistics: noise level, resolved order and decay."""

import numpy as np
import pytest

from dtfit.image import Original
from dtfit.image.analytics import Decay, decay, effective_order, noise_sigma

X = np.linspace(-1.0, 1.0, 400)


def _legendre_signal(degree, seed, sigma=0.05, order=24):
    """An image of a Legendre polynomial of known degree plus white noise."""
    yc = np.polynomial.legendre.legval(X, np.ones(degree + 1))
    y = yc + sigma * np.random.default_rng(seed).standard_normal(X.size)
    return Original(X, y).image("legendre", order)


def test_effective_order_finds_the_degree_of_a_polynomial():
    for degree in (0, 2, 4, 6, 9):
        img = _legendre_signal(degree, 2000)
        assert effective_order(img) == degree
        assert img.effective_order() == degree


def test_effective_order_is_the_degree_for_the_median_of_many_draws():
    """A noise coefficient crosses three sigma in about one draw in seven at
    order 24, so the estimate is judged over draws, not on one."""
    for degree in (0, 4, 9):
        got = [effective_order(_legendre_signal(degree, 2000 + s))
               for s in range(15)]
        assert int(np.median(got)) == degree
        assert min(got) == degree           # never below the true degree


def test_effective_order_of_a_noise_free_polynomial_is_exact():
    yc = np.polynomial.legendre.legval(X, np.ones(6))
    assert Original(X, yc).image("legendre", 20).effective_order() == 5


def test_effective_order_rejects_a_non_legendre_basis():
    img = Original(X, np.exp(X)).image("block", 8)
    with pytest.raises(ValueError, match="legendre"):
        effective_order(img)


def test_noise_sigma_recovers_the_noise_of_a_white_record():
    ratios = []
    for s in range(8):
        rng = np.random.default_rng(11 + s)
        img = Original(
            np.linspace(0.0, 1.0, 600), 0.2 * rng.standard_normal(600)
        ).image("legendre", 40)
        v = img.noise_sigma()
        assert v is not None
        ratios.append(v / 0.2)
    assert 0.70 < min(ratios) and max(ratios) < 1.35
    assert abs(float(np.mean(ratios)) - 1.0) < 0.10


def test_noise_sigma_scales_with_the_noise():
    def sigma_of(scale):
        rng = np.random.default_rng(5)
        img = Original(
            np.linspace(0.0, 1.0, 600), scale * rng.standard_normal(600)
        ).image("legendre", 40)
        return img.noise_sigma()

    small, large = sigma_of(0.02), sigma_of(0.5)
    assert small is not None and large is not None
    assert large / small == pytest.approx(25.0, rel=1e-9)


def test_noise_sigma_is_none_without_a_tail():
    x = np.linspace(0.0, 4.0, 200)
    img = Original(x, x ** 2).image("legendre", 6)
    with pytest.warns(RuntimeWarning, match="no tail"):
        assert noise_sigma(img) is None


def test_noise_sigma_rejects_a_non_legendre_basis():
    img = Original(X, np.exp(X)).image("block", 8)
    with pytest.raises(ValueError, match="legendre"):
        img.noise_sigma()


def _from_coeffs(c):
    x = np.linspace(-1.0, 1.0, 800)
    return Original(x, np.polynomial.legendre.legval(x, c)).image(
        "legendre", c.size - 1)


def test_decay_reads_a_geometric_ratio():
    for r in (0.5, 0.75):
        d = _from_coeffs(r ** np.arange(21)).decay()
        assert isinstance(d, Decay)
        assert d.ratio == pytest.approx(r, rel=1e-3)
        assert d.kind == "geometric"
        assert d.geometric_r2 > 0.999 and d.geometric_r2 > d.algebraic_r2


def test_decay_reads_an_algebraic_exponent():
    for p in (2.0, 3.0):
        j = np.arange(21)
        c = np.where(j == 0, 1.0, 1.0 / np.maximum(j, 1) ** p)
        d = decay(_from_coeffs(c))
        assert d.exponent == pytest.approx(p, rel=1e-3)
        assert d.kind == "algebraic"
        assert d.algebraic_r2 > 0.999 and d.algebraic_r2 > d.geometric_r2


def test_decay_needs_three_orders_above_the_first():
    """Two points fit both laws exactly, so the report would be empty."""
    x = np.linspace(-1.0, 1.0, 800)
    for c in ([1.0, 2.0], [1.0, 2.0, 0.5, 0.25]):
        img = Original(
            x, np.polynomial.legendre.legval(x, np.array(c))
        ).image("legendre", 12)
        with pytest.raises(ValueError, match="at least three"):
            img.decay()


def test_decay_rejects_a_non_legendre_basis():
    img = Original(X, np.exp(X)).image("block", 8)
    with pytest.raises(ValueError, match="legendre"):
        decay(img)
