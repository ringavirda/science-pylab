"""resolve_model / ModelSpec: unify string, sympy.Expr and callable models."""

import numpy as np
import pytest
import sympy as sp

from dtfit.models import resolve_model, ModelSpec, result_kwargs


# string / sympy.Expr / callable resolution
def test_resolve_string_expr():
    spec = resolve_model("a*exp(b*t)", "t")
    assert isinstance(spec, ModelSpec)
    assert spec.is_symbolic
    assert spec.names == ("a", "b")  # sorted by name
    assert spec.var == "t"
    assert spec.expr == "a*exp(b*t)"


def test_resolve_sympy_expr():
    expr = sp.sympify("a*exp(b*t)")
    spec = resolve_model(expr, "t")
    assert spec.is_symbolic
    assert spec.names == ("a", "b")
    assert spec.expr is not None  # stringified expression stored


def test_resolve_callable_signature_order():
    def f(x, a, b):
        return a * np.exp(b * x)

    spec = resolve_model(f)
    assert not spec.is_symbolic
    assert spec.names == ("a", "b")  # signature order (drop the leading x)
    assert spec.var == "x"  # callable var defaults to a label
    assert spec.expr is None


def test_resolve_callable_param_names_override():
    def f(x, a, b):
        return a * np.exp(b * x)

    spec = resolve_model(f, "time", param_names=["p", "q"])
    assert spec.names == ("p", "q")
    assert spec.var == "time"


def test_callable_names_are_signature_order_not_sorted():
    def g(x, b, a):
        return b * x + a

    spec = resolve_model(g)
    assert spec.names == ("b", "a")
    # the symbolic spelling of the same model sorts its names
    sym = resolve_model("b*x + a", "x")
    assert sym.names == ("a", "b")


# edge cases
def test_symbolic_requires_var():
    with pytest.raises(ValueError, match="var is required"):
        resolve_model("a*t", None)


def test_symbolic_param_names_validated():
    with pytest.raises(ValueError, match="do not match"):
        resolve_model("a*exp(b*t)", "t", param_names=["a", "c"])
    # a matching set in any order is accepted; the names stay sorted
    spec = resolve_model("a*exp(b*t)", "t", param_names=["b", "a"])
    assert spec.names == ("a", "b")


def test_callable_uninspectable_requires_param_names():
    def f(x, *params):
        return sum(params) * x

    with pytest.raises(ValueError, match="param_names"):
        resolve_model(f)
    spec = resolve_model(f, param_names=["a", "b", "c"])
    assert spec.names == ("a", "b", "c")


def test_callable_param_names_length_mismatch():
    def f(x, a, b):
        return a * x + b

    with pytest.raises(ValueError, match="param_names"):
        resolve_model(f, param_names=["a", "b", "c"])


def test_bad_model_type_raises():
    with pytest.raises(TypeError):
        resolve_model(123)  # type: ignore[arg-type]


# eval / param_derivs numerics
def test_eval_matches_lambdify_to_1e_12():
    expr = "a*exp(b*t) + c"
    spec = resolve_model(expr, "t")
    t = sp.Symbol("t")
    f = sp.lambdify((t, sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c")),
                    sp.sympify(expr), "numpy")
    x = np.linspace(0.0, 2.0, 50)
    coeffs = np.array([1.3, 0.7, -0.4])  # (a, b, c) in sorted name order
    got = spec.eval(x, coeffs)
    want = f(x, *coeffs)
    assert got.shape == x.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_eval_matches_callable_and_broadcasts_constant():
    def f(x, a, b):
        return a * np.sin(b * x)

    spec = resolve_model(f)
    x = np.linspace(0.0, 3.0, 40)
    c = np.array([2.0, 1.5])
    np.testing.assert_allclose(spec.eval(x, c), f(x, *c), atol=1e-12)

    # a callable returning a python float (a constant model) fills to x's shape
    def const(x, a):
        return float(a)

    cspec = resolve_model(const)
    out = cspec.eval(x, [3.5])
    assert out.shape == x.shape
    np.testing.assert_allclose(out, 3.5)

    # a 0-d array return is filled the same way
    def const0(x, a):
        return np.float64(a) * np.ones(())

    out0 = resolve_model(const0).eval(x, [1.25])
    assert out0.shape == x.shape
    np.testing.assert_allclose(out0, 1.25)


def test_param_derivs_symbolic_matches_finite_difference():
    expr = "a*exp(b*t) + c*t"
    x = np.linspace(0.1, 2.0, 30)
    coeffs = np.array([1.1, 0.6, -0.3])  # (a, b, c)

    sym = resolve_model(expr, "t")
    d_sym = sym.param_derivs(x, coeffs)

    def f(t, a, b, c):
        return a * np.exp(b * t) + c * t

    cal = resolve_model(f)
    d_fd = cal.param_derivs(x, coeffs)

    assert len(d_sym) == len(d_fd) == 3
    for ds, df in zip(d_sym, d_fd):
        assert ds.shape == x.shape and df.shape == x.shape
        np.testing.assert_allclose(ds, df, rtol=1e-5, atol=1e-6)


def test_param_derivs_symbolic_are_exact():
    # d/da (a*t^2) = t^2 ; d/db (b) constant -> broadcast to full length
    spec = resolve_model("a*t**2 + b", "t")
    x = np.linspace(0.0, 2.0, 25)
    da, db = spec.param_derivs(x, np.array([1.0, 1.0]))
    np.testing.assert_allclose(da, x**2, atol=1e-12)
    assert db.shape == x.shape
    np.testing.assert_allclose(db, np.ones_like(x), atol=1e-12)


def test_bound_model_is_fixed_coeff_closure():
    def f(x, a, b):
        return a * x + b

    spec = resolve_model(f)
    m = spec.bound_model([2.0, 1.0])
    x = np.linspace(-1.0, 1.0, 11)
    np.testing.assert_allclose(m(x), 2.0 * x + 1.0, atol=1e-12)


# result_kwargs bridge
def test_result_kwargs_symbolic_vs_callable():
    sym = resolve_model("a*exp(b*t)", "t")
    kw = result_kwargs(sym, np.array([2.0, 0.5]))
    assert kw["expr"] == "a*exp(b*t)" and kw["var"] == "t"
    assert kw["names"] == ("a", "b")
    assert "param_model" not in kw  # symbolic keeps the lambdify path

    def f(x, a, b):
        return a * np.exp(b * x)

    cal = resolve_model(f)
    kwc = result_kwargs(cal, np.array([2.0, 0.5]))
    assert kwc["expr"] is None
    assert kwc["names"] == ("a", "b")
    assert callable(kwc["model"]) and callable(kwc["param_model"])
    x = np.linspace(0.0, 1.0, 8)
    np.testing.assert_allclose(kwc["model"](x), f(x, 2.0, 0.5), atol=1e-12)
    np.testing.assert_allclose(
        kwc["param_model"](x, np.array([2.0, 0.5])), f(x, 2.0, 0.5), atol=1e-12
    )
