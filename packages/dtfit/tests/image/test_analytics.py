"""The image's own statistics: noise level, resolved order, decay, and the
two chi-square tests."""

import numpy as np
import pytest

from dtfit.image import Original, analytics
from dtfit.image.analytics import (
    ChiSquareTest, Decay, decay, effective_order, noise_sigma,
)

# analytics.test_equal and analytics.test_structure are reached through the
# module: imported by name, pytest would collect them as test functions.

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


def _exp_image(seed, slope=0.0, sigma=0.05, order=10, basis="legendre"):
    """An image of a decaying exponential plus noise; ``slope`` tilts it."""
    x = np.linspace(0.0, 4.0, 500)
    y = (2.0 * np.exp(-0.7 * x) + 0.3 + slope * x
         + sigma * np.random.default_rng(seed).standard_normal(x.size))
    return Original(x, y).image(basis, order)


def test_test_equal_passes_two_images_of_one_signal():
    t = analytics.test_equal(_exp_image(1), _exp_image(2))
    assert isinstance(t, ChiSquareTest)
    assert t.dof == 11 and not t.reject and t.pvalue > 0.05
    assert 0.0 <= t.statistic


def test_test_equal_fails_two_different_signals():
    t = _exp_image(1).test_equal(_exp_image(3, slope=0.04))
    assert t.reject and t.pvalue < 1e-6


def test_test_equal_works_on_the_block_basis():
    a = _exp_image(1, basis="block", order=8)
    b = _exp_image(2, basis="block", order=8)
    c = _exp_image(3, slope=0.04, basis="block", order=8)
    assert a.test_equal(b).dof == 8 and not a.test_equal(b).reject
    assert a.test_equal(c).reject


def test_test_equal_accepts_different_sample_sets():
    """Only basis, order and domain must agree; the grids need not."""
    x = np.linspace(0.0, 4.0, 500)
    yc = 2.0 * np.exp(-0.7 * x) + 0.3
    rng = np.random.default_rng(4)
    a = Original(x, yc + 0.05 * rng.standard_normal(x.size)).image(
        "legendre", 10)
    xb = np.sort(rng.uniform(0.0, 4.0, 300))
    yb = 2.0 * np.exp(-0.7 * xb) + 0.3 + 0.05 * rng.standard_normal(xb.size)
    b = Original(xb, yb, domain=(0.0, 4.0)).image("legendre", 10)
    assert not a.test_equal(b).reject


def test_test_equal_rejects_mismatched_images():
    a = _exp_image(1)
    for other in (
        _exp_image(2, order=8),
        _exp_image(2, basis="block", order=10),
    ):
        with pytest.raises(ValueError, match="basis, order and domain"):
            a.test_equal(other)
    x = np.linspace(0.0, 4.0, 500)
    wide = Original(x, np.exp(-x), domain=(0.0, 5.0)).image("legendre", 10)
    with pytest.raises(ValueError, match="basis, order and domain"):
        a.test_equal(wide)


def test_test_equal_validates_alpha_and_sigma():
    a, b = _exp_image(1), _exp_image(2)
    with pytest.raises(ValueError, match="alpha"):
        analytics.test_equal(a, b, alpha=0.0)
    with pytest.raises(ValueError, match="sigma"):
        analytics.test_equal(a, b, sigma=0.0)


def test_test_equal_takes_a_known_sigma():
    a, b = _exp_image(1), _exp_image(2)
    tight = analytics.test_equal(a, b, sigma=0.005)
    loose = analytics.test_equal(a, b, sigma=0.5)
    assert tight.statistic > loose.statistic
    assert tight.reject and not loose.reject


def test_test_structure_passes_the_generating_model():
    """Hands the true generating parameters, not a fit: fitted=False takes
    the full coefficient count, since no degree of freedom was spent."""
    img = _exp_image(1)
    t = analytics.test_structure(
        img, "a*exp(-b*t) + c", [2.0, 0.7, 0.3], "t", fitted=False,
    )
    assert t.dof == 11 and not t.reject
    assert t.pvalue > 0.05


def test_test_structure_fails_a_wrong_model():
    img = _exp_image(1)
    t = img.test_structure("a + b*t", [2.0, -0.4], "t")
    assert t.dof == 11 - 2 and t.reject and t.pvalue < 1e-6


def test_test_structure_at_the_fitted_parameters():
    from dtfit.image import fit

    img = _exp_image(1)
    res = fit("a*exp(-b*t) + c", img, "t", p0=[2.0, 0.7, 0.3])
    t = img.test_structure("a*exp(-b*t) + c", res.coeffs, "t")
    assert not t.reject and t.statistic < 11.0


def test_test_structure_counts_free_parameters_only_when_fitted():
    img = _exp_image(1)
    expr, p = "a*exp(-b*t) + c", [2.0, 0.7, 0.3]
    fitted = analytics.test_structure(img, expr, p, "t")
    fixed = analytics.test_structure(img, expr, p, "t", fitted=False)
    assert fitted.dof == 8 and fixed.dof == 11
    assert fitted.statistic == pytest.approx(fixed.statistic)


def test_test_structure_on_a_weighted_and_a_robust_image():
    x = np.linspace(0.0, 4.0, 500)
    y = (2.0 * np.exp(-0.7 * x) + 0.3
         + 0.05 * np.random.default_rng(1).standard_normal(x.size))
    weighted = Original(x, y, sigma=0.5 + 0.1 * x).image("legendre", 10)
    robust = Original(x, y, sigma=0.5 + 0.1 * x).image(
        "legendre", 10, robust=True)
    for img in (weighted, robust):
        t = img.test_structure("a*exp(-b*t) + c", [2.0, 0.7, 0.3], "t")
        assert t.pvalue > 0.01


def test_test_structure_takes_a_callable_model():
    def model(x, a, b, c):
        return a * np.exp(-b * x) + c

    img = _exp_image(1)
    assert not img.test_structure(model, [2.0, 0.7, 0.3]).reject


def test_test_structure_validates_its_arguments():
    img = _exp_image(1)
    with pytest.raises(ValueError, match="alpha"):
        img.test_structure("a + b*t", [1.0, 1.0], "t", alpha=1.0)
    with pytest.raises(ValueError, match="sigma"):
        img.test_structure("a + b*t", [1.0, 1.0], "t", sigma=-1.0)
    small = _exp_image(1, order=3)
    with pytest.raises(ValueError, match="degrees of freedom"):
        small.test_structure(
            "a + b*t + c*t**2 + d*t**3 + e*t**4",
            [1.0, 1.0, 1.0, 1.0, 1.0], "t",
        )


def test_chi_square_test_reject_follows_alpha():
    t = ChiSquareTest(statistic=3.0, dof=1, pvalue=0.083, alpha=0.05)
    assert not t.reject
    assert ChiSquareTest(3.0, 1, 0.083, 0.10).reject
