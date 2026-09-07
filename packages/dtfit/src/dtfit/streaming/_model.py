"""A model compiled once for the streaming hot path: evaluator, parameter
Jacobian and time derivatives, with optional external regressors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np


class CompiledModel:
    """The model of a streaming filter, compiled at construction.

    Args:
        model: A SymPy-expression string, a ``sympy.Expr`` or a callable
            ``f(t, *params)``.
        var: The main variable name (a label only for a callable).
        regressors: Name(s) of external-regressor symbols in a symbolic
            model; every other free symbol is a parameter. Not accepted
            with a callable.
        param_names: A callable's parameter names in signature order,
            required when its signature cannot be introspected.

    Attributes:
        names: Parameter names in canonical order (sorted for a symbolic
            model, signature order for a callable).
        symbolic: True for a string or ``sympy.Expr`` model.
        regressors: Regressor names, ``[]`` without any.
        f, jac: The evaluator ``f(t[, *regressors], *params)`` and the
            per-parameter sensitivities, same signature.
        dfdt, d2fdt2, dfdt_jac, d2fdt2_jac: Time derivatives and their
            parameter sensitivities for coasting; ``None`` and ``[]`` for a
            callable model.
        f_reg, f_drift, f_drift_dt, f_drift_d2t: The regressor-dependent
            and time-only parts of a regressor model, ``None`` otherwise.

    Raises:
        ValueError: regressors given with a callable model.
        RuntimeError: the model has no free parameters.
    """

    def __init__(
        self,
        model: Any,
        var: str,
        regressors: str | Sequence[str] | None = None,
        param_names: Sequence[str] | None = None,
    ) -> None:
        import sympy as sp

        from dtfit.methods._modelinput import resolve_model

        self.source = model
        self.var = str(var)
        if regressors is None:
            self.regressors: list[str] = []
        elif isinstance(regressors, str):
            self.regressors = [regressors]
        else:
            self.regressors = list(regressors)
        self.has_regressors = bool(self.regressors)
        self.dfdt: Callable[..., Any] | None = None
        self.d2fdt2: Callable[..., Any] | None = None
        self.dfdt_jac: list[Callable[..., Any]] = []
        self.d2fdt2_jac: list[Callable[..., Any]] = []
        self.f_reg: Callable[..., Any] | None = None
        self.f_drift: Callable[..., Any] | None = None
        self.f_drift_dt: Callable[..., Any] | None = None
        self.f_drift_d2t: Callable[..., Any] | None = None
        self.f: Callable[..., Any]
        self.jac: list[Callable[..., Any]]

        if callable(model) and not isinstance(model, (str, sp.Expr)):
            if self.regressors:
                raise ValueError(
                    "external regressors are only supported for symbolic "
                    "(string / sympy.Expr) models, not for a callable "
                    "model; embed the side-channel in the callable or "
                    "pass an expression string."
                )
            spec = resolve_model(model, var, param_names=param_names)
            self.names = list(spec.names)
            if not self.names:
                raise RuntimeError(
                    "Model callable has no free parameters to fit."
                )
            self.symbolic = False
            self._spec = spec

            def _f(t: Any, *p: float) -> np.ndarray:
                return spec.eval(np.asarray(t, dtype=float), p)

            self.f = _f
            self.jac = [
                self._callable_deriv(spec, k)
                for k in range(len(self.names))
            ]
            return

        t_sym = sp.Symbol(var)
        reg_syms = [sp.Symbol(r_) for r_ in self.regressors]
        _locals = {var: t_sym, **dict(zip(self.regressors, reg_syms))}
        expr = sp.sympify(
            model, locals=_locals
        )  # pyright: ignore[reportCallIssue]
        _exclude = {t_sym, *reg_syms}
        syms = sorted(
            (s for s in expr.free_symbols if s not in _exclude), key=str
        )
        if not syms:
            raise RuntimeError(
                "Model expression has no free parameters to fit."
            )
        self.names = [str(s) for s in syms]
        self.symbolic = True
        args = [t_sym, *reg_syms, *syms]
        self.f = sp.lambdify(args, expr, "numpy")
        self.jac = [
            sp.lambdify(args, sp.diff(expr, p), "numpy") for p in syms
        ]
        self.dfdt = sp.lambdify(args, sp.diff(expr, t_sym), "numpy")
        self.d2fdt2 = sp.lambdify(args, sp.diff(expr, t_sym, 2), "numpy")
        self.dfdt_jac = [
            sp.lambdify(args, sp.diff(sp.diff(expr, t_sym), p), "numpy")
            for p in syms
        ]
        self.d2fdt2_jac = [
            sp.lambdify(args, sp.diff(sp.diff(expr, t_sym, 2), p), "numpy")
            for p in syms
        ]
        if not reg_syms:
            return
        reg_set = set(reg_syms)
        terms = sp.Add.make_args(sp.expand(expr))
        f_reg = sp.Add(*[tm for tm in terms if tm.free_symbols & reg_set])
        f_drift = sp.Add(
            *[tm for tm in terms if not (tm.free_symbols & reg_set)]
        )
        self.f_reg = sp.lambdify(args, f_reg, "numpy")
        self.f_drift = sp.lambdify([t_sym, *syms], f_drift, "numpy")
        self.f_drift_dt = sp.lambdify(
            [t_sym, *syms], sp.diff(f_drift, t_sym), "numpy"
        )
        self.f_drift_d2t = sp.lambdify(
            [t_sym, *syms], sp.diff(f_drift, t_sym, 2), "numpy"
        )

    @property
    def n(self) -> int:
        return len(self.names)

    @staticmethod
    def _callable_deriv(spec: Any, k: int) -> Callable[..., np.ndarray]:
        def _deriv(t: Any, *p: float) -> np.ndarray:
            return spec.param_derivs(np.asarray(t, dtype=float), p)[k]

        return _deriv

    def eval(
        self,
        t: np.ndarray,
        reg_cols: list[np.ndarray] | None,
        p: np.ndarray,
    ) -> np.ndarray:
        """``f`` on ``t`` at ``p``; a scalar result is broadcast to ``t``'s
        shape. ``reg_cols`` are the regressor columns aligned with ``t``."""
        v = self.f(t, *p) if reg_cols is None else self.f(t, *reg_cols, *p)
        if np.ndim(v) == 0:
            return np.full(np.shape(t), float(v), dtype=float)
        return np.asarray(v, dtype=float)

    def jacobian(
        self,
        t: np.ndarray,
        reg_cols: list[np.ndarray] | None,
        p: np.ndarray,
    ) -> np.ndarray:
        """``d f / d p`` on ``t`` at ``p``, shape ``(len(t), n)``."""
        cols = []
        for jk in self.jac:
            d = jk(t, *p) if reg_cols is None else jk(t, *reg_cols, *p)
            if np.ndim(d) == 0:
                d = np.full(np.shape(t), float(d), dtype=float)
            cols.append(np.asarray(d, dtype=float))
        return np.column_stack(cols)

    def reg_tuple(self, regressors: Any) -> tuple:
        """One regressor sample as a tuple ordered like ``regressors``.

        Raises:
            ValueError: ``None`` for a regressor model, or a wrong count.
        """
        if regressors is None:
            raise ValueError(
                "this model declares external regressors; pass them to "
                "partial_fit"
            )
        if isinstance(regressors, Mapping):
            return tuple(float(regressors[r_]) for r_ in self.regressors)
        vals = np.atleast_1d(np.asarray(regressors, dtype=float))
        if vals.size != len(self.regressors):
            raise ValueError(
                f"expected {len(self.regressors)} regressors, "
                f"got {vals.size}"
            )
        return tuple(float(v) for v in vals)

    def predict_cols(
        self, xa: np.ndarray, regressors: Any
    ) -> list[np.ndarray]:
        """Regressor columns broadcast to ``xa`` for prediction.

        Raises:
            ValueError: ``regressors`` is ``None``.
        """
        if regressors is None:
            raise ValueError(
                "predict() needs regressor values for this model"
            )
        if isinstance(regressors, Mapping):
            return [
                np.broadcast_to(
                    np.asarray(regressors[r_], float), xa.shape
                )
                for r_ in self.regressors
            ]
        arr = np.asarray(regressors, float)
        if arr.ndim == 2 and arr.shape[1] == len(self.regressors):
            return [arr[:, c] for c in range(arr.shape[1])]
        if (len(self.regressors) == 1 and arr.ndim == 1
                and arr.size == xa.size and xa.size > 1):
            return [arr]
        return [
            np.broadcast_to(arr.reshape(-1)[c], xa.shape)
            for c in range(len(self.regressors))
        ]
