"""The symbolic (Maclaurin) spectrum of a model expression.

The differential transform of ``f`` about ``t0 = 0`` with sampling interval
``H`` is ``F(k) = (H**k / k!) * f^(k)(0)``. In a spectra balance every
equation sets a model discrete equal to the data discrete at the same order
``k``, so the common ``H**k`` factor cancels and the balance reduces to
matching plain Maclaurin coefficients ``f^(k)(0) / k!``, which SymPy produces
for any differentiable expression.
"""

from typing import cast

import sympy as sp


def model_params(f_sym: sp.Expr, t: sp.Symbol) -> list[sp.Symbol]:
    """Return the free parameters of ``f_sym`` (all symbols except ``t``),
    ordered by name for a stable coefficient layout."""
    params = sorted((s for s in f_sym.free_symbols if s != t), key=str)
    return cast("list[sp.Symbol]", params)


def taylor_coeffs(f_sym: sp.Expr, t: sp.Symbol, order: int) -> list[sp.Expr]:
    """Symbolic Maclaurin coefficients ``a_k = f^(k)(0) / k!`` for
    ``k = 0 .. order`` inclusive: the ``H``-free differential spectrum."""
    coeffs: list[sp.Expr] = []
    deriv = f_sym
    for k in range(order + 1):
        coeffs.append(sp.simplify(deriv.subs(t, 0) / sp.factorial(k)))
        if k < order:
            deriv = sp.diff(deriv, t)
    return coeffs
