"""The :class:`Model`: a named, self-seeding, composable model family.

A :class:`Model` bundles the expression, its parameters, a shape tag that
decides the estimator variant, and a seeder that reads initial values and
bounds off the data. Fitting routes through :func:`dtfit.auto_estimate`, and
models compose with ``+`` (trend plus seasonal, a sum of peaks).

The model itself is either a SymPy expression string or a plain Python
callable ``f(x, *params)``. A callable is resolved through
:func:`dtfit.models.resolve_model` and keeps its signature parameter order,
where a symbolic model sorts its names; both fit through the same engines.
Composition with ``+`` and the seed-detrend evaluator need symbolic operands,
and a callable raises a clear error there (see :meth:`__add__`).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import sympy as sp

from dtfit.types import FittingResult
from dtfit._input import resolve_model
from dtfit.image.fit import fit_lsi, fit_eac
from dtfit.auto import auto_estimate

# A seeder reads (x, y) and returns ``{param_name: (p0, lo, hi)}``.
Seeder = Callable[[np.ndarray, np.ndarray], dict[str, tuple[float, float, float]]]

# A model is a SymPy-expression string (symbolic) or a callable ``f(x, *params)``.
ModelExpr = str | Callable[..., Any]


def _params_of(
    expr: ModelExpr, var: str, param_names: tuple[str, ...] | None = None
) -> tuple[str, ...]:
    """Canonical parameter order of ``expr``.

    A symbolic expression parses its free symbols and sorts them by name. A
    callable is delegated to :func:`dtfit.models.resolve_model`, whose
    :attr:`~dtfit.models.ModelSpec.names` follow the signature order, either
    introspected or taken from ``param_names``.
    """
    if callable(expr):
        return resolve_model(expr, var, param_names=param_names).names
    t = sp.Symbol(var)
    f = sp.sympify(expr)
    names = tuple(sorted((str(s) for s in f.free_symbols if s != t)))
    if param_names is not None:
        given = tuple(str(n) for n in param_names)
        if sorted(given) != sorted(names):
            raise ValueError(
                f"param_names {list(given)} do not match the expression's "
                f"parameters {list(names)}."
            )
    return names


class Model:
    """A model family: expression + parameters + shape + data-driven seeder.

    Args:
        expr: The model. Either a SymPy expression string such as
            ``"a*exp(b*x)"``, or a plain Python callable ``f(x, *params)``
            (see :meth:`from_callable`). A callable is resolved via
            :func:`dtfit.models.resolve_model` and its parameters keep their
            signature order.
        var: The main variable name. For a callable it is a label only,
            defaulting to ``"x"``.
        name: A short human label.
        shape: Routing tag, one of ``"bulk"``, ``"oscillatory"``,
            ``"transient"``, ``"peak"``, ``"composite"``. It picks the
            estimator variant in :meth:`fit` under ``method="auto"``.
        freq_param: Name of the angular-frequency parameter, if oscillatory.
            Forwarded to the LSI oscillatory recipe.
        seeder: ``(x, y) -> {name: (p0, lo, hi)}`` producing data-driven
            initial values and bounds. ``None`` falls back to ones and no
            bounds.
        param_names: For a callable ``expr`` whose parameter names cannot be
            introspected (a builtin, or an ``f(x, *params)`` signature), the
            names of the parameters after the leading ``x``, in call order.
            Ignored for a symbolic ``expr``, beyond being validated against
            the parsed names when given.

    Attributes:
        is_symbolic: ``True`` for a symbolic (string) model, ``False`` for a
            callable one.
        expr: The SymPy expression string when symbolic, else ``None``.
        func: The Python callable when non-symbolic, else ``None``.
        params: The parameter names in canonical order, sorted for a symbolic
            model and in signature order for a callable. This is the layout of
            ``p0``, of ``bounds``, and of
            :attr:`dtfit.types.FittingResult.names`.
    """

    def __init__(
        self,
        expr: ModelExpr,
        var: str = "x",
        *,
        name: str = "",
        shape: str = "bulk",
        category: str = "general",
        freq_param: str | None = None,
        seeder: Seeder | None = None,
        param_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        pnames = tuple(param_names) if param_names is not None else None
        if callable(expr):
            # Resolve once to validate the callable and pin its signature-order
            # names; the callable itself is kept for the fitters.
            spec = resolve_model(expr, var, param_names=pnames)
            self.is_symbolic = False
            self.expr: str | None = None
            self.func: Callable[..., Any] | None = expr
            self.var = spec.var
            self.params: tuple[str, ...] = spec.names
            self.name = name or getattr(expr, "__name__", "") or "callable"
        else:
            expr_str = expr if isinstance(expr, str) else str(expr)
            self.is_symbolic = True
            self.expr = expr_str
            self.func = None
            self.var = var
            self.params = _params_of(expr_str, var, pnames)
            self.name = name or expr_str
        self.shape = shape
        self.category = category
        self.freq_param = freq_param
        self.seeder = seeder

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        names: tuple[str, ...] | list[str] | None = None,
        *,
        var: str = "x",
        name: str = "",
        shape: str = "bulk",
        category: str = "general",
        freq_param: str | None = None,
        seeder: Seeder | None = None,
    ) -> "Model":
        """Build a :class:`Model` from a Python callable ``func(x, *params)``.

        A convenience wrapper over the constructor's callable path. ``names``
        are the parameter names after the leading ``x``, in call order,
        introspected from the signature when omitted. They are required only
        for a callable with no inspectable signature (a builtin) or an
        ``*params`` one. Parameter order is the signature order.
        """
        return cls(
            func,
            var,
            name=name,
            shape=shape,
            category=category,
            freq_param=freq_param,
            seeder=seeder,
            param_names=names,
        )

    def __repr__(self) -> str:
        what = repr(self.expr) if self.is_symbolic else "<callable>"
        return f"Model({self.name!r}, expr={what}, shape={self.shape!r})"

    def seed(self, x: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float, float]]:
        """The data-driven ``{name: (p0, lo, hi)}`` seed map (empty if none)."""
        if self.seeder is None:
            return {}
        return self.seeder(np.asarray(x, float), np.asarray(y, float))

    def _eval_seed(self, x: np.ndarray, seed: dict) -> np.ndarray | None:
        """Evaluate the model at its seed parameter values.

        A cheap approximate curve, not a fit; it detrends the data before the
        next component of a composition is seeded. Symbolic only, since a
        callable model cannot compose (see :meth:`__add__`), and one returns
        ``None`` here.
        """
        if not seed or not self.is_symbolic:
            return None
        assert self.expr is not None
        t = sp.Symbol(self.var)
        f = sp.sympify(self.expr)
        params = sorted((s for s in f.free_symbols if s != t), key=str)
        vals = [seed.get(str(p), (1.0, 0.0, 0.0))[0] for p in params]
        try:
            fn = sp.lambdify(t, f.subs(dict(zip(params, vals))), "numpy")
            v = np.asarray(fn(np.asarray(x, float)), dtype=float)
        except Exception:
            return None
        x = np.asarray(x, float)
        if v.ndim == 0:
            v = np.full_like(x, float(v))
        return v if np.all(np.isfinite(v)) else None

    def _seed_arrays(self, x, y):
        d = self.seed(x, y)
        if not d:
            return None, None
        p0, bounds = [], []
        for nm in self.params:
            if nm in d:
                v, lo, hi = d[nm]
                p0.append(float(v))
                bounds.append((float(lo), float(hi)))
            else:
                p0.append(1.0)
                bounds.append((-np.inf, np.inf))
        # Partially-infinite bounds are kept. The solvers skip the global (DE)
        # stage unless every bound is finite, but the local trf solve honours
        # mixed bounds directly (see ``solve_weighted_nlls``), which is how a
        # seeder's positivity guard on one parameter survives an otherwise
        # unbounded seed. A fully unbounded seed constrains nothing and maps to
        # ``None``, keeping the unconstrained LM solver; forcing the bounded
        # path there measurably degrades tanh_step accuracy.
        if all(np.isneginf(lo) and np.isposinf(hi) for lo, hi in bounds):
            return p0, None
        return p0, bounds

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        method: str = "auto",
        p0=None,
        bounds=None,
    ) -> FittingResult:
        """Fit this model to ``(x, y)``.

        ``method="auto"`` (default) routes by :attr:`shape` through
        :func:`dtfit.auto_estimate`; ``"lsi"`` and ``"eac"`` force a
        specific engine. Seeds and bounds come from the model's seeder
        unless overridden. A callable model is passed straight through to the
        fitters, which resolve it via :func:`dtfit.models.resolve_model`.
        """
        sp0, sb = self._seed_arrays(x, y)
        p0 = sp0 if p0 is None else p0
        bounds = sb if bounds is None else bounds
        model: ModelExpr = self.expr if self.is_symbolic else self.func  # type: ignore[assignment]
        # The fitter re-resolves a callable, and re-introspection would either
        # fail on an ``f(x, *params)`` signature or, for a renamed callable,
        # return the raw signature names rather than ``self.params``. Passing
        # the names explicitly holds the order this Model committed to. A
        # symbolic model re-parses to the same sorted names and needs nothing.
        pnames: tuple[str, ...] | None = None if self.is_symbolic else self.params
        if method == "auto":
            # A composite such as trend + sine fits as 'bulk' LSI while still
            # carrying its freq_param. The cycle is pinned by the tight FFT
            # seed the composed seeder takes off the detrended residual, which
            # empirically beats the full oscillatory recipe here: its raised
            # order tends to over-fit a trend-plus-cycle spectrum.
            shape = self.shape if self.shape != "composite" else "bulk"
            return auto_estimate(x, y, model, self.var, shape=shape,
                                 freq_param=self.freq_param, p0=p0, bounds=bounds,
                                 param_names=pnames)
        if method == "lsi":
            return fit_lsi(x, y, model, self.var, p0=p0, bounds=bounds,
                           freq_param=self.freq_param, param_names=pnames)
        if method == "eac":
            # Bounds go as a pair list, fit_eac's canonical form. Converting to
            # a scipy 2-tuple would be ambiguous for a 2-parameter model and
            # lossy for partially-infinite bounds.
            return fit_eac(x, y, model, self.var,
                           p0=p0, bounds=bounds, param_names=pnames)
        raise ValueError(
            f"unknown method {method!r}; expected auto/lsi/eac"
        )

    def __add__(self, other: "Model") -> "Model":
        """Compose two models additively, as in ``trend + seasonal``.

        Colliding parameter names in ``other`` are renamed, and the seeders
        compose too, leaving the combined model self-seeding.

        Both operands must be symbolic. A callable carries no expression to
        rename or detrend against, and composing one raises :class:`TypeError`.
        """
        if not self.is_symbolic or not other.is_symbolic:
            raise TypeError(
                "cannot compose a callable model with '+': symbolic composition "
                "requires both operands to be SymPy-expression models (rename / "
                "detrend needs a manipulable expression). Compose the symbolic "
                "forms, or fit the callable model on its own."
            )
        assert self.expr is not None and other.expr is not None
        if other.var != self.var:
            raise ValueError(
                f"cannot add models on different variables: {self.var!r} vs {other.var!r}"
            )
        rename: dict[str, str] = {}
        used = set(self.params)
        for p in other.params:
            new = p
            i = 2
            while new in used:
                new = f"{p}_{i}"
                i += 1
            if new != p:
                rename[p] = new
            used.add(new)
        other_expr = sp.sympify(other.expr)
        if rename:
            other_expr = other_expr.subs({sp.Symbol(k): sp.Symbol(v)
                                          for k, v in rename.items()})
        combined = f"({self.expr}) + ({sp.sstr(other_expr)})"
        freq = self.freq_param or (
            rename.get(other.freq_param, other.freq_param) if other.freq_param else None
        )

        def seeder(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            s_seed = self.seed(x, y)
            d = dict(s_seed)
            # Seed the second component on the residual left after removing
            # the first's seed approximation. A cycle then reads its frequency
            # and amplitude off detrended data instead of off the raw trend.
            resid = y
            approx = self._eval_seed(x, s_seed)
            if approx is not None:
                resid = y - approx
            for k, v in other.seed(x, resid).items():
                d[rename.get(k, k)] = v
            return d

        return Model(combined, self.var, name=f"{self.name}+{other.name}",
                     shape="composite", category="composite", freq_param=freq,
                     seeder=seeder)
