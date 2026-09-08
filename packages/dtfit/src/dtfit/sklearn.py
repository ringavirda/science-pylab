"""scikit-learn compatible estimator over the projected fit.

``NonlineRegressor`` exposes :func:`dtfit.fit` through the standard
``fit`` / ``predict`` / ``score`` API with ``get_params`` / ``set_params``
from ``BaseEstimator``, so it composes with ``sklearn.pipeline.Pipeline``,
``GridSearchCV`` and ``cross_val_score``. ``basis`` and ``order`` are
estimator parameters, which puts the basis and the resolution of the image
inside a grid search.

Importing this module imports scikit-learn; the rest of dtfit does not.
"""

from typing import Any, cast

import numpy as np
from scipy import sparse as sp_sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    _check_sample_weight,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from dtfit._pandas import as_series, capture_index, is_dataframe, is_series
from dtfit.image import Original, fit
from dtfit.types import FittingResult

__all__ = ["NonlineRegressor"]


class NonlineRegressor(RegressorMixin, BaseEstimator):
    """Fit a model that is nonlinear in its parameters to 1-D data.

    Args:
        expr: The model, in any of three equivalent forms resolved by
            :func:`dtfit.models.resolve_model`: a SymPy-expression string
            such as ``"a0 + a1*x + a2*exp(a3*x)"``, a :class:`sympy.Expr`, or
            a plain Python callable ``f(x, *params)``. The default is a
            simple affine string, because the scikit-learn contract requires
            ``NonlineRegressor()`` to construct with no arguments for
            ``clone`` and meta-estimator introspection.
        var: Main variable name in ``expr``, the single input feature. For a
            callable model it is a label only.
        param_names: Parameter names for a callable model, in signature order
            (those following the leading ``x``). Introspected from the
            callable when omitted, and required only where the signature
            cannot be introspected, as with a ``*args`` model. For a symbolic
            model it is optional and checked against the names parsed from
            the expression. Stored verbatim, since the sklearn contract
            forbids validating in ``__init__``, and forwarded to the fitter.
        basis: The basis to image the samples in: ``"auto"`` (default) fits
            the candidates and keeps the one with the lowest sample residual
            sum of squares, ``"legendre"`` and ``"block"`` fix it. See
            :func:`dtfit.fit`.
        order: The basis order, the polynomial degree for ``"legendre"`` and
            the window count for ``"block"``. ``None`` (default) takes the
            order rule of :func:`dtfit.fit`, which is required by
            ``basis="auto"``.
        bounds: Optional parameter bounds, either a per-parameter
            ``(min, max)`` pair list in canonical parameter order or a
            partial ``{name: (min, max)}`` dict. Passed to the fitter
            untouched. Fully finite bounds enable a global search.
        p0: Optional initial guess for the parameters, a sequence in
            canonical parameter order or a ``{name: value}`` dict. Passed
            through untouched.
        random_state: Seed for the deterministic global
            (differential-evolution) search that runs when ``bounds`` are
            given, keeping a bounded fit reproducible under ``GridSearchCV``
            and ``clone``. ``None`` uses the global RNG.
        robust: Huber-reweight the image built from the samples, the
            model-free outlier defence of :meth:`dtfit.Original.image`.
        nan_policy: ``"raise"`` (default) rejects non-finite samples;
            ``"omit"`` drops NaN/inf ``(x, y)`` pairs before fitting.

    Fitted attributes:
        coef_: Fitted coefficients, in canonical parameter order.
        model_: Callable model evaluated at the fitted coefficients.
        result_: The full :class:`dtfit.FittingResult`, exposing ``cov``,
            ``stderr()``, ``confidence_intervals()``, ``converged`` and the
            fit-quality stats (``rsquared``, ``aic`` / ``bic``).
        n_features_in_: Number of input features (always 1).
    """

    # Declared for type checkers only. ``fit`` sets these; sklearn's
    # ``validate_data`` fills the ``*_in_`` pair during validation.
    coef_: np.ndarray
    model_: Any
    result_: FittingResult
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        expr="a0 + a1*x",
        var="x",
        param_names=None,
        basis="auto",
        order=None,
        bounds=None,
        p0=None,
        random_state=0,
        robust=False,
        nan_policy="raise",
    ):
        self.expr = expr
        self.var = var
        self.param_names = param_names
        self.basis = basis
        self.order = order
        self.bounds = bounds
        self.p0 = p0
        self.random_state = random_state
        self.robust = robust
        self.nan_policy = nan_policy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True  # y is mandatory for fit
        # A bare 1-D x vector is the natural input and a single-column 2-D X
        # is accepted for Pipeline use, so both input-type tags hold.
        # ``one_d_array`` also tells the sklearn check suite to exercise the
        # estimator with 1-D X, as it does for ``IsotonicRegression``.
        tags.input_tags.one_d_array = True
        # With nan_policy="omit" the Original drops non-finite pairs, making
        # NaN input legitimate.
        tags.input_tags.allow_nan = self.nan_policy == "omit"
        # ``poor_score`` refers specifically to the check suite's shared
        # regression dataset: its informative feature is not column 0,
        # putting the R^2 > 0.5 the suite asserts out of reach for any
        # single-feature estimator.
        tags.regressor_tags.poor_score = True
        return tags

    def __getstate__(self):
        # Copy: BaseEstimator may hand back the live ``__dict__``.
        state = dict(super().__getstate__())
        # ``model_`` is a lambdified closure and does not pickle. Drop it and
        # rebuild from ``result_`` on unpickle; :class:`dtfit.FittingResult`
        # pickles cleanly, re-lambdifying its model lazily from ``expr`` and
        # ``coeffs``.
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "result_"):
            self.model_ = self.result_.model

    @staticmethod
    def _to_2d(X):
        """Promote a bare 1-D feature vector (array, Series, plain list) to a
        single column. 2-D array-likes are left alone, DataFrames included,
        so their column names survive. Sparse input is handed on untouched
        for ``validate_data`` to reject with the standard scikit-learn sparse
        error."""
        if sp_sparse.issparse(X):
            return X
        ndim = getattr(X, "ndim", None)
        if ndim is None:  # plain sequences / duck-typed array wrappers
            X = np.asarray(X)
            ndim = X.ndim
        if ndim == 1:
            return np.asarray(X).reshape(-1, 1)
        return X

    def _reject_multifeature(self, X) -> None:
        n_features = np.shape(X)[1] if np.ndim(X) == 2 else 1
        if n_features != 1:
            raise ValueError(
                "NonlineRegressor supports a single input feature; got "
                f"{n_features}. dtfit's image is one-dimensional, so "
                "multivariate X (several predictors) is not supported. If "
                "instead you have a 1-D signal that is a sum of components "
                "along one axis (e.g. trend + cycle), compose 1-D models "
                "with `+` (see the 'Multivariate data' docs note)."
            )

    def _sample_weight_to_sigma(self, sample_weight, X, order):
        """Translate sklearn ``sample_weight`` into the image's ``sigma``.

        ``None`` in gives ``None`` out and an equal-weight fit. Otherwise the
        weights are validated against ``X`` by
        :func:`~sklearn.utils.validation._check_sample_weight`, reordered to
        match the x-sorted samples, and mapped to the relative measurement
        standard deviation ``sigma = 1 / sqrt(weight)``. A zero weight
        becomes a huge but finite ``sigma``, ignoring that sample while every
        sigma stays inside the image's strictly-positive contract. Negative
        and all-zero weights raise.
        """
        if sample_weight is None:
            return None
        if is_series(sample_weight):
            # A Series of weights becomes its float values, positional like
            # every other array-like sample_weight; _check_sample_weight then
            # length-checks it against X.
            sample_weight = np.asarray(sample_weight, dtype=float)
        sw = _check_sample_weight(sample_weight, X, dtype=np.float64)[order]
        if np.any(sw < 0.0):
            raise ValueError("sample_weight must be non-negative.")
        if not np.any(sw > 0.0):
            raise ValueError(
                "sample_weight must contain at least one non-zero (strictly "
                "positive) weight; all sample weights are zero."
            )
        with np.errstate(divide="ignore"):
            sigma = 1.0 / np.sqrt(sw)
        nonfinite = ~np.isfinite(sigma)
        if nonfinite.any():
            # weight == 0 -> 1/sqrt(0) = inf. A huge but finite sigma ignores
            # the sample and keeps every sigma finite and positive, as the
            # image requires.
            finite_max = (
                float(np.max(sigma[~nonfinite])) if (~nonfinite).any() else 1.0
            )
            sigma[nonfinite] = 1e6 * finite_max
        return sigma

    def fit(self, X, y, sample_weight=None) -> "NonlineRegressor":
        """Fit the model to ``(X, y)``.

        Args:
            X: The single input feature, a 1-D vector or a single-column 2-D
                array / DataFrame.
            y: The target values.
            sample_weight: Optional per-sample weights, the scikit-learn
                convention. Translated to the image's per-sample
                ``sigma = 1 / sqrt(sample_weight)`` and consumed under
                ``absolute_sigma=False``, which reads them as relative
                weights: a down-weighted sample pulls the projected fit less
                without being dropped. A weight of ``0`` effectively ignores
                its sample through a huge ``sigma``, and all-zero weights
                raise.

        Returns:
            ``self``, with ``coef_``, ``model_`` and ``result_`` set.

        Raises:
            ValueError: More than one input feature; a malformed
                ``sample_weight``; anything :func:`dtfit.fit` raises for the
                model, ``p0`` or ``bounds``.
        """
        X = self._to_2d(X)
        # cast: the validate_data stub mistypes its array argument as str.
        # Validation runs first, letting sparse, non-finite and empty inputs
        # raise the standard scikit-learn errors; the single-feature
        # constraint is applied only afterwards.
        if self.__sklearn_tags__().input_tags.allow_nan:
            # nan_policy="omit": the Original drops non-finite (x, y) pairs
            # itself. Validating X and y separately relaxes the finite check
            # that otherwise guards both at once.
            X, y = cast(Any, validate_data)(
                self,
                X,
                y,
                reset=True,
                validate_separately=(
                    {"dtype": np.float64, "ensure_all_finite": False},
                    {
                        "dtype": np.float64,
                        "ensure_2d": False,
                        "ensure_all_finite": False,
                    },
                ),
            )
            # Mirror the default path's y shape rules (1-D or single column).
            y = column_or_1d(y, warn=True)
            check_consistent_length(X, y)
        else:
            X, y = cast(Any, validate_data)(
                self, X, y, reset=True, dtype=np.float64, y_numeric=True
            )
        self._reject_multifeature(X)
        x = np.asarray(X, dtype=float)[:, 0]
        y = np.asarray(y, dtype=float).ravel()
        # scikit-learn contract: sample order must not matter. The image is
        # built over an interval and its grid descriptor wants monotone x,
        # hence the sort; it is a no-op for already-ordered curve data.
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]

        # Per-sample weights become sigma = 1/sqrt(weight) for the weighted
        # image, aligned to the x-sorted samples.
        sigma = self._sample_weight_to_sigma(sample_weight, X, order)

        data = Original(x, y, sigma=sigma, nan_policy=self.nan_policy)
        result = fit(
            self.expr,
            data,
            self.var,
            basis=self.basis,
            order=self.order,
            p0=self.p0,
            bounds=self.bounds,
            absolute_sigma=False,
            robust=self.robust,
            param_names=self.param_names,
            random_state=self.random_state,
        )

        self.result_ = result
        self.model_ = result.model
        self.coef_ = np.asarray(result.coeffs, dtype=float)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict ``y`` for the single input feature ``X``.

        pandas in, pandas out: a ``Series`` or single-column ``DataFrame``
        ``X`` gives back a ``Series`` aligned to ``X``'s index, and an
        ndarray or list input gives back an ndarray. Only the container
        differs; the predicted values are the same either way.
        """
        check_is_fitted(self, "model_")
        # Capture the sample index before validation coerces X to a plain
        # array; the prediction is realigned to it at the end. validate_data
        # below rejects a multi-column DataFrame, so the captured index only
        # ever belongs to a Series or a single-column frame.
        x_index = capture_index(X) if (is_series(X) or is_dataframe(X)) else None
        X = self._to_2d(X)
        # cast: the validate_data stub mistypes its array argument as str.
        X = cast(Any, validate_data)(self, X, reset=False, dtype=np.float64)
        x = np.asarray(X, dtype=float)[:, 0]
        out = np.asarray(self.model_(x), dtype=float)
        if out.ndim == 0:  # constant model -> broadcast
            out = np.full(x.shape, out.item())
        # as_series returns the plain ndarray unchanged when x_index is None
        # (non-pandas input) or pandas is absent.
        return as_series(out, x_index)
