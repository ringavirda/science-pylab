"""Public data types shared across the fitting methods."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Literal, overload

import numpy as np

from dtfit._pandas import (
    as_series,
    capture_index,
    is_dataframe,
    is_series,
    to_1d_array,
)

# Initial parameter guess. Coerced with np.asarray; a plain list is fine.
InitialGuess = Sequence[float] | np.ndarray | None


@functools.lru_cache(maxsize=256)
def _parse_model(expr: str, var: str):
    """Parse ``(expr, var)`` into ``(sympy, t, f, params)``, cached.

    Re-parsing the same expression is pure overhead. Every ``.model`` access
    hits it, as does every result in a batch fit that shares one model. SymPy
    expressions are immutable, which makes memoizing on ``(expr, var)`` safe.
    ``params`` comes back as a tuple to keep the result hashable.
    """
    import sympy as sp

    t = sp.Symbol(var)
    f = sp.sympify(expr)
    params = tuple(sorted((s for s in f.free_symbols if s != t), key=str))
    return sp, t, f, params


class FittingResult:
    """A fitted nonlinear model: parameters, their uncertainty, and the model.

    It carries the model expression, variable and parameter names alongside
    the coefficients. That is enough to name its parameters, report
    uncertainty from the covariance, predict with error bands, and round-trip
    to a plain dict.

    Attributes:
        coeffs: Fitted coefficients, ordered by parameter name.
        cov: Parameter covariance (``n_params x n_params``), or ``None`` when
            the method cannot produce one. Square roots of its diagonal are
            the standard errors.
        expr: The model expression, e.g. ``"a*exp(b*t)"``, when known. Needed
            for :meth:`to_dict` and for prediction error bands.
        var: The model's main variable name.
        names: Parameter names aligned with ``coeffs``.
        model: Precomputed NumPy callable of the fitted model. Lambdified
            lazily from ``expr`` and ``coeffs`` when omitted.
        label: Tag carried through batch and parallel fits, such as a channel
            name or a grid cell; ``None`` for a single fit.
        error: A message set in place of coefficients when a fit failed inside
            a batch (:func:`dtfit.fit_many`); ``None`` on success.
        converged: Whether the optimizer reported convergence. ``None`` means
            the method does not report it. A successful call with
            ``converged is False`` is the silent-failure case to watch for:
            you get a result, but the optimizer never settled.
        message: The optimizer's termination message, when available.
        x_range: ``(min, max)`` of the training ``x``, recorded for
            :meth:`predict`'s extrapolation warning; ``None`` when unknown.
        param_model: A parameters-explicit evaluator ``f(x, coeffs) -> y``, as
            produced by :meth:`dtfit.methods.ModelSpec.eval`. Set in place of
            ``expr`` for a callable-only model, in which case :meth:`predict`
            finite-differences it for the std band and :attr:`model` falls
            back to it. ``None`` for a symbolic model.
        n_obs: Number of observations; enables :attr:`aic` and :attr:`bic`.
        rss: Residual sum of squares; enables :attr:`rsquared`, :attr:`aic`
            and :attr:`bic`.
        tss: Total sum of squares of the data; enables :attr:`rsquared`.
        nfev: Model evaluations the optimizer used, when it reports them.
        cost: Final optimizer cost, typically ``0.5 * rss``.
        rss_source: Where ``rss`` was computed: ``"samples"``, ``"image"``
            or ``None``.
        image_order: The order of the image the fit ran on, or ``None``.
        basis_name: The basis of the image the fit ran on, or ``None``.
    """

    def __init__(
        self,
        coeffs: np.ndarray,
        cov: np.ndarray | None = None,
        expr: str | None = None,
        var: str | None = None,
        names: tuple[str, ...] | Sequence[str] = (),
        model: Callable[..., Any] | None = None,
        *,
        label: Any = None,
        error: str | None = None,
        converged: bool | None = None,
        message: str | None = None,
        x_range: tuple[float, float] | None = None,
        param_model: Callable[..., Any] | None = None,
        n_obs: int | None = None,
        rss: float | None = None,
        tss: float | None = None,
        nfev: int | None = None,
        cost: float | None = None,
    ) -> None:
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.cov = cov
        self.expr = expr
        self.var = var
        self.names: tuple[str, ...] = tuple(names)
        self._model = model
        self.param_model = param_model
        self.label = label
        self.error = error
        self.converged = converged
        self.message = message
        self.x_range = (
            None if x_range is None
            else (float(x_range[0]), float(x_range[1]))
        )
        self.n_obs = None if n_obs is None else int(n_obs)
        self.rss = None if rss is None else float(rss)
        self.tss = None if tss is None else float(tss)
        self.nfev = None if nfev is None else int(nfev)
        self.cost = None if cost is None else float(cost)
        self.rss_source: str | None = None
        self.image_order: int | None = None
        self.basis_name: str | None = None

    def __repr__(self) -> str:
        if self.error is not None:
            return f"FittingResult(error={self.error!r}, label={self.label!r})"
        conv = "" if self.converged is not False else ", converged=False"
        return (f"FittingResult(expr={self.expr!r}, "
                f"params={self.params!r}{conv})")

    # The lazily-built `_model` is a lambdified closure and does not survive a
    # process-pool round trip. Drop it on pickling; it rebuilds from
    # expr/coeffs on first access. That is what lets a worker return a fitted
    # result across the fit_many boundary.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @property
    def model(self) -> Callable[..., Any]:
        """The fitted model as a NumPy callable ``f(x) -> y``.

        Precomputed when supplied, otherwise rebuilt lazily from ``expr`` and
        ``coeffs``, or from ``param_model`` with the coefficients frozen.
        """
        if self._model is None:
            if self.expr is not None and self.var is not None:
                self._model = self._lambdify(self.coeffs)
            elif self.param_model is not None:
                pm = self.param_model
                coeffs = self.coeffs
                self._model = lambda x: pm(x, coeffs)
            else:
                raise ValueError(
                    "this FittingResult has no model: pass a model callable, an "
                    "expr, or a param_model."
                )
        return self._model

    def _sympy(self):
        if self.expr is None or self.var is None:
            raise ValueError(
                "this FittingResult has no expr/var; the operation needs the "
                "model expression (only the precomputed callable is available)."
            )
        sp, t, f, params = _parse_model(self.expr, self.var)
        return sp, t, f, list(params)

    def _lambdify(self, coeffs: np.ndarray) -> Callable[..., Any]:
        sp, t, f, params = self._sympy()
        return sp.lambdify(t, f.subs(dict(zip(params, coeffs))), "numpy")

    @property
    def params(self) -> dict[str, float]:
        """Fitted parameters as a ``{name: value}`` mapping."""
        names = self.names or tuple(f"p{i}" for i in range(self.coeffs.size))
        return {n: float(c) for n, c in zip(names, self.coeffs)}

    def stderr(self) -> dict[str, float]:
        """Per-parameter standard errors, from the covariance diagonal."""
        if self.cov is None:
            raise ValueError("no covariance available for this fit (cov is None).")
        se = np.sqrt(np.clip(np.diag(np.asarray(self.cov, float)), 0.0, None))
        names = self.names or tuple(f"p{i}" for i in range(self.coeffs.size))
        return {n: float(s) for n, s in zip(names, se)}

    def confidence_intervals(self, level: float = 0.95) -> dict[str, tuple[float, float]]:
        """Normal-approximation confidence intervals at ``level``."""
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2.0))
        se = self.stderr()
        return {n: (float(v) - z * se[n], float(v) + z * se[n])
                for n, v in self.params.items()}

    @property
    def rsquared(self) -> float | None:
        """Coefficient of determination ``1 - rss/tss``.

        ``None`` when the fit recorded neither ``rss`` nor ``tss``, or when
        the data is constant and ``tss == 0``.
        """
        if self.rss is None or self.tss is None or self.tss == 0.0:
            return None
        return 1.0 - self.rss / self.tss

    def _information_criteria(self) -> tuple[float, float] | None:
        if self.rss is None or self.n_obs is None:
            return None
        # Lazy: dtfit.types is imported before dtfit.methods, and importing
        # the criteria at module scope would form a partial-import cycle. By
        # call time the package is loaded.
        from dtfit.methods._common import information_criteria

        k = len(self.names) if self.names else self.coeffs.size
        return information_criteria(self.rss, self.n_obs, int(k))

    @property
    def aic(self) -> float | None:
        """Akaike information criterion, given ``rss`` and ``n_obs``."""
        ic = self._information_criteria()
        return None if ic is None else ic[0]

    @property
    def bic(self) -> float | None:
        """Bayesian information criterion, given ``rss`` and ``n_obs``."""
        ic = self._information_criteria()
        return None if ic is None else ic[1]

    def residuals(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit residuals ``y - model(x)`` at the given samples."""
        y = np.asarray(y, dtype=float)
        return y - self.predict(x)

    @overload
    def predict(self, x: np.ndarray, *,
                return_std: Literal[False] = False,
                warn_extrapolation: bool = ...) -> np.ndarray: ...
    @overload
    def predict(self, x: np.ndarray, *,
                return_std: Literal[True],
                warn_extrapolation: bool = ...) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(self, x: np.ndarray, *, return_std: bool = False,
                warn_extrapolation: bool = False):
        """Evaluate the fitted model at ``x``.

        With ``return_std=True`` this also returns the 1-sigma prediction
        standard deviation, propagated from the parameter covariance by the
        delta method. That needs ``cov`` and the model ``expr``.

        With ``warn_extrapolation=True`` a :class:`UserWarning` is issued for
        any ``x`` outside the fitted training range (:attr:`x_range`). A
        nonlinear model can extrapolate to nonsense, so this catches the most
        common curve-fitting mistake. No-op when the range is unknown.

        pandas in -> pandas out: a pandas ``Series`` (or single-column
        ``DataFrame``) ``x`` comes back as a ``Series`` on ``x``'s index, or
        as a ``(Series, Series)`` pair with ``return_std=True``. An ndarray or
        list returns an ndarray.
        """
        # pandas in -> pandas out: capture the index before coercing, then
        # realign the prediction to it. A non-pandas x leaves x_index None and
        # as_series hands back the plain ndarray.
        x_index = None
        if is_series(x) or is_dataframe(x):
            x_index = capture_index(x)
            x = to_1d_array(x, "x")
        else:
            x = np.asarray(x, dtype=float)
        if warn_extrapolation and self.x_range is not None and x.size:
            lo, hi = self.x_range
            xmin, xmax = float(np.min(x)), float(np.max(x))
            if xmin < lo or xmax > hi:
                warnings.warn(
                    f"predict() called outside the fitted range "
                    f"[{lo:.6g}, {hi:.6g}] (got [{xmin:.6g}, {xmax:.6g}]); "
                    "the nonlinear model is extrapolating.",
                    UserWarning,
                    stacklevel=2,
                )
        y = np.asarray(self.model(x), dtype=float)
        if np.ndim(y) == 0:
            y = np.full_like(x, float(y))
        if not return_std:
            return as_series(y, x_index)
        if self.cov is None:
            raise ValueError("prediction std needs a covariance (cov is None).")
        base = self.coeffs.astype(float)
        # Finite-difference a params-explicit evaluator g(coeffs) -> y(x), one
        # parameter at a time (delta method). The symbolic path lambdifies
        # once over (t, *params) rather than f.subs(...) per parameter. Either
        # way the baseline is g's own output, not self.model(x): an explicit
        # model= that is not exactly f(coeffs) cannot skew the band. Normally
        # y0 == y to machine precision.
        if self.expr is not None and self.var is not None:
            sp, t, f, params = self._sympy()
            f_lam = sp.lambdify((t, *params), f, "numpy")

            def g(c: np.ndarray) -> np.ndarray:
                vk = np.asarray(f_lam(x, *c), dtype=float)
                return np.full_like(x, float(vk)) if np.ndim(vk) == 0 else vk

            n_p = len(params)
        elif self.param_model is not None:
            pm = self.param_model

            def g(c: np.ndarray) -> np.ndarray:
                vk = np.asarray(pm(x, c), dtype=float)
                return np.full_like(x, float(vk)) if np.ndim(vk) == 0 else vk

            n_p = base.size
        else:
            raise ValueError(
                "prediction std needs the model expr or a param_model "
                "(only a precomputed callable is available)."
            )
        y0 = g(base)
        jac = np.empty((x.size, n_p))
        for k in range(n_p):
            step = 1e-6 * max(1.0, abs(base[k]))
            cp = base.copy()
            cp[k] += step
            jac[:, k] = (g(cp) - y0) / step
        var = np.einsum("ij,jk,ik->i", jac, np.asarray(self.cov, float), jac)
        std = np.sqrt(np.clip(var, 0.0, None))
        return as_series(y, x_index), as_series(std, x_index)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly dict for storage or shipping.

        Captures the expression, variable, names, coefficients and covariance,
        which is everything :meth:`from_dict` needs. Requires ``expr`` and
        ``var``: a fit carrying only a precomputed callable cannot be
        serialized.
        """
        if self.expr is None or self.var is None:
            raise ValueError(
                "cannot serialize a FittingResult without expr/var "
                "(only a precomputed model callable is available)."
            )
        out: dict[str, Any] = {
            "expr": self.expr,
            "var": self.var,
            "names": list(self.names),
            "coeffs": self.coeffs.tolist(),
            "cov": None if self.cov is None else np.asarray(self.cov, float).tolist(),
            "x_range": None if self.x_range is None else list(self.x_range),
        }
        # Round-trip only the diagnostics the fitter actually recorded.
        for key, val in (
            ("n_obs", self.n_obs), ("rss", self.rss), ("tss", self.tss),
            ("nfev", self.nfev), ("cost", self.cost),
        ):
            if val is not None:
                out[key] = val
        # The image diagnostics, unlike the ones above, are always
        # written to the dict and may be None.
        out["rss_source"] = self.rss_source
        out["image_order"] = self.image_order
        out["basis_name"] = self.basis_name
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FittingResult":
        """Rebuild a :class:`FittingResult` from :meth:`to_dict` output."""
        cov = d.get("cov")
        xr = d.get("x_range")
        result = cls(
            coeffs=np.asarray(d["coeffs"], dtype=float),
            cov=None if cov is None else np.asarray(cov, dtype=float),
            expr=d["expr"],
            var=d["var"],
            names=tuple(d.get("names", ())),
            x_range=None if xr is None else (float(xr[0]), float(xr[1])),
            n_obs=d.get("n_obs"),
            rss=d.get("rss"),
            tss=d.get("tss"),
            nfev=d.get("nfev"),
            cost=d.get("cost"),
        )
        result.rss_source = d.get("rss_source")
        result.image_order = d.get("image_order")
        result.basis_name = d.get("basis_name")
        return result

    def summary(self) -> str:
        """A short text summary: parameters +/- their standard errors."""
        lines = [f"FittingResult: {self.expr or '<callable>'}"]
        se = self.stderr() if self.cov is not None else None
        for n, v in self.params.items():
            if se is not None:
                lines.append(f"  {n} = {v:.6g} +/- {se[n]:.3g}")
            else:
                lines.append(f"  {n} = {v:.6g}")
        r2 = self.rsquared
        if r2 is not None:
            lines.append(f"  R^2 = {r2:.6g}")
        if self.converged is False:
            msg = f" ({self.message})" if self.message else ""
            lines.append(f"  [warning] optimizer did not converge{msg}")
        return "\n".join(lines)
