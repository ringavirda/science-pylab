"""Unified model-input resolution for the fitting methods.

:func:`resolve_model` accepts a model in three equivalent forms, a SymPy
expression string, a :class:`sympy.Expr`, or a plain Python callable
``f(x, *params)``, behind one :class:`ModelSpec` interface. A fitter can then
evaluate the model, its parameter sensitivities and a bound ``f(x)`` closure
without caring which form the caller supplied.

The canonical parameter order (:attr:`ModelSpec.names`) is the order used for
coefficients, ``p0``, bounds, covariance and
:class:`~dtfit.types.FittingResult` everywhere downstream:

* symbolic models sort their parameters by name
  (:func:`dtfit.methods.model_params`);
* callables use signature order, the parameters after the leading ``x``,
  because a callable is invoked positionally and the coefficients have to line
  up with it.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import sympy as sp

from ._common import model_params


def _fill(v: Any, x: np.ndarray) -> np.ndarray:
    """Coerce a model / derivative return to a float array shaped like ``x``.

    A constant (a python float, a 0-d array, or any scalar the evaluator emits
    for a constant sub-expression) is broadcast to fill ``x``; an array is
    returned as float. Mirrors the ``constant -> full_like`` broadcasting the
    LSI / EAC residual evaluators already use.
    """
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0:
        return np.full(x.shape, float(arr))
    return arr


def _introspect_names(func: Callable[..., Any]) -> tuple[str, ...] | None:
    """Parameter names of a callable model, in signature order (skipping ``x``).

    Returns the names of the positional parameters after the first one, the
    ``x`` variable. ``None`` means the names cannot be determined, from a
    builtin with no signature or a ``*args`` model with no fixed trailing
    parameters, and the caller must supply ``param_names`` explicitly.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    positional: list[str] = []
    has_var_positional = False
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional.append(p.name)
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
    if not positional:
        return None
    rest = tuple(positional[1:])  # drop the leading x variable
    if has_var_positional and not rest:
        # f(x, *params): the parameter count is open, so names are unknowable.
        return None
    return rest


class ModelSpec:
    """A resolved model in one of three forms, behind a uniform interface.

    Produced by :func:`resolve_model`. Exposes the canonical parameter order
    (:attr:`names`), numeric evaluation (:meth:`eval`), parameter sensitivities
    (:meth:`param_derivs`) and a fixed-coefficient closure (:meth:`bound_model`)
    regardless of whether the underlying model is symbolic (a SymPy expression)
    or a plain Python callable.

    Attributes:
        names: Canonical parameter order. Sorted by name for a symbolic model
            (matching :func:`dtfit.methods.model_params`); signature order for a
            callable. This is the layout of ``coeffs`` / ``p0`` / ``bounds`` /
            the covariance and of :attr:`dtfit.types.FittingResult.names`.
        var: The main variable name (meaningful for a symbolic model; a label
            only for a callable, defaulting to ``"x"``).
        expr: The SymPy expression string when symbolic, else ``None``.
        is_symbolic: ``True`` for a string / :class:`sympy.Expr` model.
    """

    def __init__(
        self,
        names: Sequence[str],
        var: str,
        expr: str | None,
        is_symbolic: bool,
        *,
        sym: tuple[Any, Any, Any, tuple[Any, ...]] | None = None,
        callable_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._names: tuple[str, ...] = tuple(str(n) for n in names)
        self._var = str(var)
        self._expr = expr
        self._is_symbolic = bool(is_symbolic)
        # Symbolic backing ``(sp module, t symbol, f_sym, params tuple)`` or the
        # user callable; exactly one is set. Lambdified evaluators are built and
        # cached lazily so constructing a spec to read only ``.names`` is cheap.
        self._sym = sym
        self._callable = callable_fn
        self._eval_func: Callable[..., Any] | None = None
        self._deriv_funcs: list[Callable[..., Any]] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def var(self) -> str:
        return self._var

    @property
    def expr(self) -> str | None:
        return self._expr

    @property
    def is_symbolic(self) -> bool:
        return self._is_symbolic

    def __repr__(self) -> str:
        what = repr(self._expr) if self._is_symbolic else "<callable>"
        return f"ModelSpec({what}, var={self._var!r}, names={self._names!r})"

    # lazily-built symbolic evaluators
    def _eval_lambda(self) -> Callable[..., Any]:
        if self._eval_func is None:
            assert self._sym is not None
            spm, t, f_sym, params = self._sym
            self._eval_func = spm.lambdify((t, *params), f_sym, "numpy")
        return self._eval_func

    def _deriv_lambdas(self) -> list[Callable[..., Any]]:
        if self._deriv_funcs is None:
            assert self._sym is not None
            spm, t, f_sym, params = self._sym
            self._deriv_funcs = [
                spm.lambdify((t, *params), spm.diff(f_sym, p), "numpy")
                for p in params
            ]
        return self._deriv_funcs

    def eval(self, x: np.ndarray, coeffs: Sequence[float] | np.ndarray) -> np.ndarray:
        """Model values at ``x`` for ``coeffs`` given in :attr:`names` order.

        Always returns a 1-D float array broadcast to ``x``'s shape. A
        constant model, or a callable returning a python float or 0-d array,
        is filled to the full length.
        """
        x = np.asarray(x, dtype=float)
        c = np.asarray(coeffs, dtype=float)
        if self._is_symbolic:
            v = self._eval_lambda()(x, *c)
        else:
            fn = self._callable
            assert fn is not None
            v = fn(x, *c)
        return _fill(v, x)

    def param_derivs(
        self, x: np.ndarray, coeffs: Sequence[float] | np.ndarray
    ) -> list[np.ndarray]:
        """``d f / d p_k`` at ``x``, one array per parameter in :attr:`names` order.

        Symbolic models differentiate exactly (:func:`sympy.diff`); callables use
        a forward difference with step ``1e-6 * max(1, |c_k|)``. Each entry is
        broadcast to ``x``'s shape (a constant sensitivity is filled).
        """
        x = np.asarray(x, dtype=float)
        c = np.asarray(coeffs, dtype=float)
        if self._is_symbolic:
            return [_fill(func(x, *c), x) for func in self._deriv_lambdas()]
        y0 = self.eval(x, c)
        out: list[np.ndarray] = []
        for k in range(c.size):
            step = 1e-6 * max(1.0, abs(float(c[k])))
            cp = c.copy()
            cp[k] += step
            out.append((self.eval(x, cp) - y0) / step)
        return out

    def bound_model(
        self, coeffs: Sequence[float] | np.ndarray
    ) -> Callable[[np.ndarray], np.ndarray]:
        """A plain ``f(x)`` closure with the coefficients frozen at ``coeffs``.

        Used as :attr:`dtfit.types.FittingResult.model` when no expression is
        available (the callable path). Returns the same broadcast float array as
        :meth:`eval`.
        """
        c = np.asarray(coeffs, dtype=float)

        def _model(x: np.ndarray) -> np.ndarray:
            return self.eval(x, c)

        return _model


def _resolve_symbolic(
    expr_str: str, var: str | None, param_names: Sequence[str] | None
) -> ModelSpec:
    if var is None:
        raise ValueError(
            "var is required for a symbolic (string / sympy.Expr) model "
            "(e.g. 't' or 'x')."
        )
    t = sp.Symbol(str(var))
    f_sym = cast(sp.Expr, sp.sympify(expr_str))
    params = model_params(f_sym, t)  # sorted by name: the canonical order
    names = tuple(str(p) for p in params)
    if param_names is not None:
        given = tuple(str(n) for n in param_names)
        if sorted(given) != sorted(names):
            raise ValueError(
                f"param_names {list(given)} do not match the model's parameters "
                f"{list(names)} (parsed from the expression)."
            )
    return ModelSpec(
        names, str(var), expr_str, True, sym=(sp, t, f_sym, tuple(params))
    )


def _resolve_callable(
    func: Callable[..., Any], var: str | None, param_names: Sequence[str] | None
) -> ModelSpec:
    v = "x" if var is None else str(var)
    if param_names is not None:
        names = tuple(str(n) for n in param_names)
        introspected = _introspect_names(func)
        if introspected is not None and len(introspected) != len(names):
            raise ValueError(
                f"param_names has {len(names)} name(s) {list(names)} but the "
                f"callable takes {len(introspected)} parameter(s) after {v!r}."
            )
    else:
        introspected = _introspect_names(func)
        if introspected is None:
            raise ValueError(
                "cannot introspect the parameter names of the callable model "
                "(it has no inspectable signature or uses *args); pass "
                "param_names explicitly."
            )
        names = introspected
    return ModelSpec(names, v, None, False, callable_fn=func)


def resolve_model(
    model: str | sp.Expr | Callable[..., Any],
    var: str | None = None,
    *,
    param_names: Sequence[str] | None = None,
) -> ModelSpec:
    """Resolve a model given as a string, a :class:`sympy.Expr`, or a callable.

    Args:
        model: The model. A SymPy expression string (e.g. ``"a*exp(b*t)"``), a
            :class:`sympy.Expr`, or a Python callable ``f(x, *params)``.
        var: The main variable name. Required for a symbolic model, where it
            names the free variable in the expression; for a callable it is a
            label only and defaults to ``"x"``.
        param_names: Parameter names. For a callable, the names of the
            parameters after the leading ``x`` (in signature order); introspected
            from the signature when omitted. For a symbolic model it is optional
            and, if given, is validated against the names parsed from the
            expression (which stay the canonical sorted order).

    Returns:
        A :class:`ModelSpec` exposing the canonical parameter order and the
        numeric evaluation / sensitivity / closure helpers.

    Raises:
        ValueError: A symbolic model without ``var``; a ``param_names`` that does
            not match the parsed / introspected parameters; or a callable whose
            names cannot be introspected and were not supplied.
        TypeError: ``model`` is not a string, :class:`sympy.Expr`, or callable.
    """
    if isinstance(model, str):
        return _resolve_symbolic(model, var, param_names)
    if callable(model):
        return _resolve_callable(model, var, param_names)
    if isinstance(model, sp.Expr):
        return _resolve_symbolic(str(model), var, param_names)
    raise TypeError(
        "model must be a sympy-expression string, a sympy.Expr, or a callable "
        f"f(x, *params); got {type(model).__name__}."
    )


def result_kwargs(
    spec: ModelSpec, coeffs: Sequence[float] | np.ndarray
) -> dict[str, Any]:
    """Keyword arguments for building a :class:`~dtfit.types.FittingResult`.

    Bridges :func:`resolve_model` to the result type:

    * symbolic model: pass ``expr`` / ``var`` / ``names``, which is what the
      lambdify path behind prediction std bands and ``to_dict`` runs on;
    * callable model: there is no expression, so pass a bound ``f(x)``
      closure as ``model`` and the numeric params-explicit evaluator as
      ``param_model``, letting the result still finite-difference a
      prediction std band.
    """
    if spec.is_symbolic:
        return {"expr": spec.expr, "var": spec.var, "names": spec.names}
    return {
        "expr": None,
        "var": spec.var,
        "names": spec.names,
        "model": spec.bound_model(coeffs),
        "param_model": spec.eval,
    }
