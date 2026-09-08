"""The streaming filter on the window image, and its two basis aliases."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from dtfit.image.bases import Basis, make_basis, u_of
from dtfit.image.fit import fit
from dtfit.image.image import gram_whitener
from dtfit.image.original import Original
from dtfit.types import InitialGuess
from . import coast as _coast
from ._model import CompiledModel
from .detect import DriftDetector

if TYPE_CHECKING:
    from dtfit.types import FittingResult


class ImageFilter:
    """Recursive parameter tracker whose measurement is the window image.

    Each ingested sample joins a sliding window of the last ``W`` samples
    (``W`` fixed, or sized from the data with ``adaptive_window``). The
    window is imaged in ``basis`` at ``order`` on its own domain:
    ``S_w = Phi_w^T y``, ``G_w = Phi_w^T Phi_w`` with ``Phi_w`` the basis at
    the window's sample positions. The innovation ``S_w - Phi_w^T f(t; p)``
    and the Jacobian ``Phi_w^T df/dp`` are whitened by the Cholesky factor
    of ``G_w``, so the measurement noise is ``s2 I`` with ``s2`` the window
    residual variance, and the gain state ``(p, P)`` is updated in
    information form with random-walk process noise ``Q``; a step that
    raises the window's whitened misfit is halved, up to eight times.
    ``P`` is a gain state: the calibrated parameter covariance is
    :meth:`result`.

    Drift: every ``W`` samples once the window is full; while the adaptive
    window is still growing, the stride is the shorter, current
    ``_W_eff`` instead, so the detector tests sooner and more often than
    it will once the window has reached its cap. The whitened innovation,
    rotated so its first component is the window-mean channel, goes to a
    :class:`~dtfit.streaming.detect.DriftDetector` (a jump test on its
    energy and a two-sided CUSUM on that first component); a detection
    inflates or resets ``P`` and collapses the adaptive window. The
    filter's adaptation is bounded by ``Q``, so a change beyond it leaves
    a lagging estimate and a visible innovation.

    Args:
        model: A SymPy-expression string, a ``sympy.Expr`` or a callable
            ``f(t, *params)``. A symbolic model may reference external
            regressors (``regressors``). A callable has no time derivatives,
            so :meth:`coast` and :meth:`coast_cov` raise for it, and it
            accepts no regressors.
        var: The main variable name (a label only for a callable).
        basis: ``"legendre"`` or ``"block"``, or a
            :class:`~dtfit.image.Basis` instance (which fixes ``order``).
        order: The Legendre order, or the window count of the block basis,
            default 5; the image has ``order + 1`` or ``order`` coefficients
            and needs at least as many as the model has parameters. A
            :class:`Basis` instance carries its own order.
        window_size: The window cap ``W`` (samples), at least the image's
            coefficient count plus one.
        min_window: Samples from which the filter measures; defaults to
            ``order + 2`` (Legendre) or ``2 * order`` (block, two samples per
            window) and is clamped to ``[that floor, window_size]``.
        adaptive_window: Size the window from the data (the default): grow
            from ``min_window`` by one sample per update while the model
            fits the window, shrink by one while the model residual over
            the window is autocorrelated (EWMA of its lag-1 autocorrelation
            above 0.35, a fitted curve lagging or missing the dynamics),
            collapse to ``min_window`` on a drift. A static model grows to
            the cap; a manoeuvring signal settles where the model still
            fits. ``False`` keeps the window at ``window_size``.
        q_diag: Process-noise variances per parameter, default 0.01 each.
        noise_var: The measurement variance ``s2``; ``None`` (default)
            estimates it as an EWMA of the window's model residual
            variance, so a poor start damps the gain and a converged
            filter runs at the noise variance.
        p0: Initial parameters in canonical order (sorted names for a
            symbolic model, signature order for a callable), default ones.
        regressors: Names of external-regressor symbols in a symbolic
            model; every ``partial_fit`` and ``predict`` then takes their
            values.
        param_names: A callable's parameter names when its signature
            cannot be introspected.
        robust: Winsorize each sample's residual to the current model at
            ``huber_c`` MAD sigmas around the window's median residual
            before imaging, so a spike cannot carry into the image while a
            sustained shift passes through to the drift test.
        huber_c: The winsorization threshold in robust sigmas.
        drift_reset: ``"inflate"`` multiplies ``P`` by ``drift_inflation``
            and keeps the window; ``"full"`` resets ``P`` to its initial
            value and clears the window.
        drift_inflation: The factor of ``"inflate"`` and of :meth:`inflate`.
        alpha: Significance of the detector's jump test; ``1e-15`` disables
            it.
        cusum_k: CUSUM slack in standard deviations; ``inf`` disables the
            CUSUM.
        cusum_h: CUSUM decision threshold.
        stream: An :class:`~dtfit.image.ImageStream` that receives every
            ingested ``(t, y)`` as well, for a running whole-stream image.

    Attributes:
        p, P, Q: The parameter estimate, the gain state and the process
            noise.
        params_: ``{name: value}`` of the current estimate.
        innovation_: The last whitened innovation (NaN before the first
            measurement).
        nis_: The last normalized innovation squared
            ``e^T (H P H^T + s2 I)^-1 e``, chi-square with ``n_coef``
            degrees of freedom under the model; NaN before the first
            measurement. Summing it over several filters is a fused test.
        last_residual_: The one-step residual ``y - f(t; p)`` at the newest
            sample before the update; NaN before the first measurement.
        detector: The :class:`DriftDetector`. It sees the innovation
            rotated so that its first component is the window-mean
            innovation, the direction channel for every basis.
        n_drifts_, drift_flag_, last_drift_direction_: Detections so far,
            whether the last update detected one, and its direction.

    Raises:
        ValueError: the image has fewer coefficients than the model has
            parameters; ``window_size`` cannot hold the image; an invalid
            ``drift_reset``; a non-finite ``p0``; ``q_diag`` or ``p0`` of
            the wrong length; ``noise_var <= 0``; an ``order`` disagreeing
            with a :class:`Basis` instance; ``regressors`` on a callable
            model. :meth:`partial_fit` also raises it, out of
            :func:`~dtfit.image.bases.u_of`, for a window whose first and
            last timestamps coincide (a stalled clock).
        RuntimeError: the model has no free parameters.
    """

    _fixed_basis: str | None = None

    @classmethod
    def tracking(cls, model: Any, var: str, **overrides: Any) -> "ImageFilter":
        """The adaptive-window configuration, ``adaptive_window=True``
        (the default); ``overrides`` win."""
        return cls(model, var, **{"adaptive_window": True, **overrides})

    @classmethod
    def robust(cls, model: Any, var: str, **overrides: Any) -> "ImageFilter":
        """The outlier-resilient configuration: ``robust=True`` and
        ``drift_reset="inflate"``; ``overrides`` win."""
        return cls(
            model, var,
            **{"robust": True, "drift_reset": "inflate", **overrides}
        )

    def __init__(
        self,
        model: Any,
        var: str,
        *,
        basis: str | Basis = "legendre",
        order: int | None = None,
        window_size: int = 50,
        min_window: int | None = None,
        adaptive_window: bool = True,
        q_diag: Sequence[float] | None = None,
        noise_var: float | None = None,
        p0: InitialGuess = None,
        regressors: str | Sequence[str] | None = None,
        param_names: Sequence[str] | None = None,
        robust: bool = False,
        huber_c: float = 3.0,
        drift_reset: str = "inflate",
        drift_inflation: float = 100.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        stream: Any = None,
    ) -> None:
        if self._fixed_basis is not None:
            basis = self._fixed_basis
        self.model = CompiledModel(model, var, regressors, param_names)
        n = self.model.n
        if isinstance(basis, Basis):
            if order is not None and order != basis.order:
                raise ValueError(
                    f"order={order} disagrees with the basis's order "
                    f"{basis.order}"
                )
            b = basis
        else:
            b = make_basis(basis, 5 if order is None else order)
        if b.n_coef < n:
            raise ValueError(
                f"an image with {b.n_coef} coefficients cannot identify "
                f"{n} parameters; raise order"
            )
        self.basis = b
        self.order = b.order
        self.W = int(window_size)
        if self.W < b.n_coef + 1:
            raise ValueError(
                f"window_size={self.W} cannot hold an image with "
                f"{b.n_coef} coefficients; needs at least {b.n_coef + 1}"
            )
        floor = 2 * b.n_coef if b.name == "block" else b.n_coef + 1
        floor = min(floor, self.W)
        self.min_window = (
            floor
            if min_window is None
            else int(min(self.W, max(floor, min_window)))
        )
        self.adaptive_window = bool(adaptive_window)
        self.noise_var = None if noise_var is None else float(noise_var)
        if self.noise_var is not None and not self.noise_var > 0.0:
            raise ValueError("noise_var must be positive")
        if drift_reset not in ("full", "inflate"):
            raise ValueError(
                f"drift_reset must be 'full' or 'inflate', got "
                f"{drift_reset!r}"
            )
        self.drift_reset = drift_reset
        self.drift_inflation = float(drift_inflation)
        self._robust = bool(robust)
        self._huber_c = float(huber_c)
        self.stream = stream

        self.p = (
            np.ones(n) if p0 is None
            else np.asarray(p0, dtype=float).reshape(-1)
        )
        if self.p.size != n or not np.all(np.isfinite(self.p)):
            raise ValueError(f"p0 must hold {n} finite values")
        self._p_init = np.eye(n) * 10.0
        self.P: np.ndarray = self._p_init.copy()
        self.Q: np.ndarray = np.diag(
            np.full(n, 0.01) if q_diag is None
            else np.atleast_1d(np.asarray(q_diag, dtype=float))
        )
        if self.Q.shape != (n, n):
            raise ValueError(f"q_diag must hold {n} values")
        if np.any(np.diag(self.Q) < 0.0):
            raise ValueError("q_diag must be non-negative")

        self.detector = DriftDetector(
            b.n_coef, alpha=alpha, cusum_k=cusum_k, cusum_h=cusum_h, warmup=3,
        )
        self.n_drifts_ = 0
        self.drift_flag_ = False
        self.last_drift_direction_ = 0
        self.last_residual_ = float("nan")
        self.innovation_: np.ndarray = np.full(b.n_coef, np.nan)
        self.nis_ = float("nan")

        self._W_eff = self.min_window
        self._resid_corr = 0.0
        self._shrink_corr = 0.35
        self._v_est: float | None = None
        self._v_lambda = 0.05
        self._n_full = 0
        # Cached per window length: at steady state (fixed window, or an
        # adaptive one that has settled) every call repeats one length,
        # and while the window is still growing no length repeats anyway.
        # Two entries cover an adaptive window oscillating between W and
        # W-1 (reachable through the residual-autocorrelation rule);
        # capped so it does not grow without bound.
        self._cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._t: list[float] = []
        self._y: list[float] = []
        self._rbuf: list[tuple] = []

    @property
    def params_(self) -> dict[str, float]:
        """Current parameter estimate as a ``{name: value}`` mapping."""
        return {s: float(v) for s, v in zip(self.model.names, self.p)}

    def _ingest(self, t_new: Any, y_new: Any, regressors: Any) -> bool:
        """Validate one sample and append it to the window; a non-finite
        sample is skipped with a ``RuntimeWarning`` and leaves every state
        untouched. Returns whether the sample was appended.

        The attached ``stream`` (if any) is fed before the window is
        mutated, so a sample it rejects (outside its domain) leaves the
        window untouched too, instead of already holding it.
        """
        t_val = float(t_new)
        y_val = float(y_new)
        reg = (
            self.model.reg_tuple(regressors)
            if self.model.has_regressors else ()
        )
        if not (np.isfinite(t_val) and np.isfinite(y_val)
                and all(np.isfinite(v) for v in reg)):
            warnings.warn(
                "non-finite sample skipped", RuntimeWarning, stacklevel=3
            )
            return False
        if self.stream is not None:
            self.stream.update(np.array([t_val]), np.array([y_val]))
        self._t.append(t_val)
        self._y.append(y_val)
        if self.model.has_regressors:
            self._rbuf.append(reg)
        return True

    def _window_ops(
        self, t_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The basis at the window's positions, the Cholesky factor of its
        Gram and the rotation that maps the window-mean channel onto the
        first axis, cached against the last window length while the
        normalized positions repeat (uniform streaming). ``None`` when a
        block window holds no sample, which leaves the Gram singular."""
        k = t_arr.size
        u = u_of(t_arr, float(t_arr[0]), float(t_arr[-1]))
        hit = self._cache.get(k)
        if hit is not None and np.allclose(u, hit[0], rtol=0.0, atol=1e-9):
            return hit[1], hit[2], hit[3]
        Phi = self.basis.evaluate(u)
        G = Phi.T @ Phi
        if self.basis.name == "block" and np.any(np.diag(G) <= 0.0):
            return None
        L = gram_whitener(G)
        # The constant function in basis coordinates: ones for the block
        # indicators, the first function otherwise. Its whitened direction
        # q carries the window-mean innovation; a Householder reflection
        # puts it on the first axis without changing the energy.
        n_coef = Phi.shape[1]
        c = (
            np.ones(n_coef) if self.basis.name == "block"
            else np.eye(1, n_coef)[0]
        )
        q = L.T @ c
        q = q / np.linalg.norm(q)
        v = q.copy()
        v[0] -= 1.0
        vn = float(v @ v)
        Q = (
            np.eye(n_coef)
            if vn < 1e-24
            else np.eye(n_coef) - 2.0 * np.outer(v, v) / vn
        )
        self._cache[k] = (u, Phi, L, Q)
        if len(self._cache) > 2:
            del self._cache[next(iter(self._cache))]
        return Phi, L, Q

    def _grow_on_skip(self) -> None:
        """Grow the adaptive window past a skipped measurement. A window
        that keeps failing to measure (an empty block bin, a rejected
        step) does not get more likely to measure by staying the size it
        stalled at; growing it is what gives the next sample a wider
        window to land a bin in, or a better-conditioned step to try."""
        if self.adaptive_window and self._W_eff < self.W:
            self._W_eff += 1

    def partial_fit(
        self, t_new: Any, y_new: Any, regressors: Any = None
    ) -> "ImageFilter":
        """Ingest one ``(t, y[, regressors])`` sample and update in place.

        ``regressors`` (required for a regressor model) is a ``{name:
        value}`` mapping or a sequence ordered like ``regressors``. A
        non-finite sample is skipped with a ``RuntimeWarning``. A sample
        is ingested but not measured when the window is not yet full
        enough, a block window holds an empty bin, the step never lowers
        the window's whitened misfit, or the innovation or the Jacobian
        comes out entirely non-finite; a partly non-finite Jacobian is
        instead measured with its non-finite entries zeroed.
        """
        self.drift_flag_ = False
        if not self._ingest(t_new, y_new, regressors):
            return self
        cap = self._W_eff if self.adaptive_window else self.W
        while len(self._t) > cap:
            self._t.pop(0)
            self._y.pop(0)
            if self.model.has_regressors:
                self._rbuf.pop(0)
        k = len(self._t)
        if k < self.min_window:
            return self
        full = k >= cap

        t_arr = np.asarray(self._t, dtype=float)
        y_arr = np.asarray(self._y, dtype=float)
        reg_cols = None
        if self.model.has_regressors:
            rb = np.asarray(self._rbuf, dtype=float)
            reg_cols = [rb[:, c] for c in range(rb.shape[1])]
        ops = self._window_ops(t_arr)
        if ops is None:
            self._grow_on_skip()
            return self
        Phi, L, Q = ops

        f = self.model.eval(t_arr, reg_cols, self.p)
        resid = y_arr - f
        y_eff = y_arr
        if self._robust:
            med = float(np.median(resid))
            sigma = 1.4826 * float(np.median(np.abs(resid - med)))
            if sigma > 0.0:
                c = self._huber_c * sigma
                y_eff = f + (med + np.clip(resid - med, -c, c))

        S = Phi.T @ y_eff
        v_now = float(resid @ resid) / k
        if self.noise_var is not None:
            s2 = self.noise_var
        else:
            if self._v_est is None:
                self._v_est = v_now
            else:
                self._v_est = (
                    (1.0 - self._v_lambda) * self._v_est
                    + self._v_lambda * v_now
                )
            s2 = max(self._v_est, 1e-12)

        e = solve_triangular(L, S - Phi.T @ f, lower=True)
        J = self.model.jacobian(t_arr, reg_cols, self.p)
        J = np.where(np.isfinite(J), J, 0.0)
        H = solve_triangular(L, Phi.T @ J, lower=True)
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(H))):
            self._grow_on_skip()
            return self
        self.last_residual_ = float(resid[-1])
        self.innovation_ = e

        A = H.T @ H / s2
        b = H.T @ e / s2
        try:
            P_post = np.linalg.inv(np.linalg.inv(self.P) + A)
        except np.linalg.LinAlgError:
            self._grow_on_skip()
            return self
        # nis_ is set here, against the pre-update state, so it always
        # matches the innovation and residual just recorded above even
        # when the detector diverts or the damping loop below rejects.
        self.nis_ = float(e @ e) / s2 - float(b @ (P_post @ b))

        if full:
            self._n_full += 1
            if self._n_full % cap == 0 and self.detector.update(Q @ e):
                self._on_drift(self.detector.last_direction_ >= 0)
                return self

        step = P_post @ b
        # Damped update: the linearization can overshoot on a nonlinear
        # model, so the step is halved until the window's whitened misfit
        # does not rise, and skipped when no fraction of it helps.
        misfit = float(e @ e)
        alpha = 1.0
        p_new = self.p + step
        accepted = False
        for _ in range(9):
            if np.all(np.isfinite(p_new)):
                f_new = self.model.eval(t_arr, reg_cols, p_new)
                # A finite step can still evaluate to a non-finite model
                # (an exponent that overflows), which would make the solve
                # below raise; treat it as a rejected step and damp on.
                if np.all(np.isfinite(f_new)):
                    e_new = solve_triangular(
                        L, S - Phi.T @ f_new, lower=True)
                    if (np.all(np.isfinite(e_new))
                            and float(e_new @ e_new) <= misfit):
                        accepted = True
                        break
            alpha *= 0.5
            p_new = self.p + alpha * step
        if not accepted:
            self._grow_on_skip()
            return self
        P_new = P_post + self.Q
        P_new = 0.5 * (P_new + P_new.T)
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            self._grow_on_skip()
            return self
        self.p = p_new
        self.P = P_new

        if self.adaptive_window:
            # Model adequacy on the window: a curve the model cannot follow
            # leaves an autocorrelated residual whatever the gain does.
            r = y_arr - f_new
            rr = float(r @ r)
            rho = (
                float(r[1:] @ r[:-1]) / rr
                if rr > 0.0 and r.size > 2 else 0.0
            )
            self._resid_corr = 0.9 * self._resid_corr + 0.1 * rho
            if self._resid_corr > self._shrink_corr:
                if self._W_eff > self.min_window:
                    self._W_eff -= 1
            elif self._W_eff < self.W:
                self._W_eff += 1
        return self

    update = partial_fit

    def _on_drift(self, up: bool) -> None:
        """Re-arm after a detection: inflate or reset ``P``, collapse the
        adaptive window and its sizing state, restart the stride."""
        if self.drift_reset == "inflate":
            self.P = self.P * self.drift_inflation
        else:
            self.P = self._p_init.copy()
            self._t, self._y, self._rbuf = [], [], []
        self._W_eff = self.min_window
        self._resid_corr = 0.0
        self._n_full = 0
        self.n_drifts_ = self.detector.n_drifts_
        self.drift_flag_ = True
        self.last_drift_direction_ = 1 if up else -1

    def inflate(self, factor: float | None = None) -> None:
        """Multiply ``P`` by ``factor`` (default ``drift_inflation``) so new
        data dominates: the hook for an external change detector."""
        self.P = self.P * (
            self.drift_inflation if factor is None else float(factor)
        )

    @property
    def _anchor(self) -> float | None:
        return float(self._t[-1]) if self._t else None

    def predict(self, x: Any, regressors: Any = None) -> np.ndarray:
        """The model at the current estimate on ``x``; see
        :func:`dtfit.streaming.coast.predict`."""
        return _coast.predict(self.model, self.p, x, regressors)

    def predict_cov(self, x: Any, regressors: Any = None) -> np.ndarray:
        """The output variance ``P`` implies at ``x``; see
        :func:`dtfit.streaming.coast.predict_cov`. The calibrated
        parameter covariance is :meth:`result`."""
        return _coast.predict_cov(self.model, self.p, self.P, x, regressors)

    def coast(
        self, x: Any, *, order: int = 1, regressors: Any = None
    ) -> np.ndarray:
        """Dead-reckon past the window from its last sample; see
        :func:`dtfit.streaming.coast.coast`."""
        return _coast.coast(
            self.model, self.p, x, self._anchor, order=order,
            regressors=regressors
        )

    def coast_cov(self, x: Any, *, order: int = 1) -> np.ndarray:
        """The variance of :meth:`coast` from ``P``; see
        :func:`dtfit.streaming.coast.coast_cov`."""
        return _coast.coast_cov(
            self.model, self.p, self.P, x, self._anchor, order=order
        )

    def result(self, **kwargs: Any) -> "FittingResult":
        """A batch fit of the model on the current window, with the filter's
        basis and order, started from the current estimate: the window's
        parameters, covariance, standard errors and prediction band in the
        same :class:`~dtfit.types.FittingResult` a batch fit returns. The
        window is imaged robustly when the filter is robust. A regressor
        model is fitted as a callable closed over the window's regressor
        columns, interpolated linearly onto whatever grid a diagnostic
        evaluates and exact on the window grid the fit runs on. ``kwargs``
        go to :func:`dtfit.image.fit.fit` (``bounds``, ``solver_options``).

        Raises:
            ValueError: fewer than ``min_window`` samples ingested.
        """
        k = len(self._t)
        if k < self.min_window:
            raise ValueError(
                f"result() needs at least min_window={self.min_window} "
                f"samples; the window holds {k}"
            )
        t_arr = np.asarray(self._t, dtype=float)
        y_arr = np.asarray(self._y, dtype=float)
        original = Original(t_arr, y_arr)
        options: dict[str, Any] = {
            "basis": self.basis, "p0": self.p, "robust": self._robust,
        }
        options.update(kwargs)
        if self.model.has_regressors:
            rb = np.asarray(self._rbuf, dtype=float)
            cols = [rb[:, c] for c in range(rb.shape[1])]
            names = list(self.model.names)
            model_f = self.model.f

            def on_window(x: Any, *p: float) -> np.ndarray:
                xa = np.asarray(x, dtype=float)
                at_x = [np.interp(xa, t_arr, col) for col in cols]
                return np.asarray(model_f(xa, *at_x, *p), dtype=float)

            return fit(
                on_window, original, self.model.var, param_names=names,
                **options,
            )
        if self.model.symbolic:
            return fit(self.model.source, original, self.model.var,
                       **options)
        return fit(self.model.source, original, self.model.var,
                   param_names=list(self.model.names), **options)


class LSIFilter(ImageFilter):
    """:class:`ImageFilter` in the Legendre basis: the streaming twin of
    :func:`dtfit.fit_lsi`. Takes every ``ImageFilter`` keyword except
    ``basis``."""

    _fixed_basis = "legendre"

    def __init__(self, model: Any, var: str, **kwargs: Any) -> None:
        if "basis" in kwargs:
            raise TypeError(
                "LSIFilter fixes basis='legendre'; use ImageFilter to choose"
            )
        super().__init__(model, var, **kwargs)


class EACFilter(ImageFilter):
    """:class:`ImageFilter` in the block basis: the streaming twin of
    :func:`dtfit.fit_eac`; ``order`` is the window count. Takes every
    ``ImageFilter`` keyword except ``basis``."""

    _fixed_basis = "block"

    def __init__(self, model: Any, var: str, **kwargs: Any) -> None:
        if "basis" in kwargs:
            raise TypeError(
                "EACFilter fixes basis='block'; use ImageFilter to choose"
            )
        super().__init__(model, var, **kwargs)
