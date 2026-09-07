"""Recursive Legendre-spectrum filter, the streaming counterpart of LSI.

The online analogue of the batch integral-least-squares method
(:func:`dtfit.image.fit.fit_lsi`), and structurally a sibling of
:class:`dtfit.streaming._eac.EACFilter`: a Kalman-style recursive estimator
whose "measurement" is an innovation between an experimental quantity and the
model's prediction of it, with a measurement Jacobian of integrated parameter
sensitivities. What changes is the measurement itself.

``EACFilter`` measures areas: the signal projected onto piecewise indicator
functions, the zeroth moment over each sub-window. This filter measures the
Legendre spectrum instead, projecting the signal onto the first ``order + 1``
orthogonal Legendre polynomials over the window. Three things follow from that.
A single window yields ``order + 1`` independent equations rather than one or a
handful of correlated sub-areas, and coupled multi-parameter and oscillatory
models are identified faster; net signed area is a low-order moment and nearly
blind to frequency and phase; Legendre moments resolve shape. The basis
is orthogonal, leaving the measurement covariance diagonal with
``R_j ∝ (2j+1)``, the LSI orthonormal weight, whereas contiguous sub-areas
are correlated and shrink in magnitude. That diagonal ``R`` also makes a
proper multivariate Normalized Innovation Squared (NIS) a clean chi-squared
with ``order + 1`` degrees of freedom.

The empirical spectrum comes from a cached Legendre projection on a window
normalized to ``[-1, 1]``, an ``O(W·order)`` mat-vec assuming roughly uniform
streaming. The model spectrum comes from Gauss-Legendre quadrature, integrating
the model exactly as the batch LSI scheme does. The symbolic model and its
derivatives are compiled once in ``__init__``; the hot path contains no SymPy
and is ``O(W·order·params)`` per sample.
"""

from typing import Any, Callable, Sequence

import numpy as np
import sympy as sp
from numpy.polynomial import legendre as L
from scipy.stats import chi2

from dtfit._core._kernels import legendre_project
from dtfit.types import InitialGuess
from ._base import _RecursiveFilter


class LSIFilter(_RecursiveFilter):
    """Online integral-least-squares parameter tracker with drift detection.

    Drop-in sibling of :class:`EACFilter` with the same ``partial_fit`` /
    ``predict`` / ``params_`` API, swapping the area measurement for an
    orthogonal Legendre-spectrum measurement (streaming LSI).

    As with :class:`EACFilter`, the constructor exposes the full knob set. Most
    callers should start from a preset classmethod and pass only overrides:
    :meth:`tracking` (responsive auto-sized window) or :meth:`robust`
    (outlier-resilient gains).
    """

    @classmethod
    def tracking(
        cls, expr: str | sp.Expr | Callable[..., Any], var: str, **overrides: Any
    ) -> "LSIFilter":
        """Responsive preset: auto-sized window for tracking drifting parameters.

        Equivalent to ``LSIFilter(expr, var, adaptive_window=True, ...)``; any
        keyword in ``overrides`` wins over the preset.
        """
        return cls(expr, var, **{"adaptive_window": True, **overrides})

    @classmethod
    def robust(
        cls, expr: str | sp.Expr | Callable[..., Any], var: str, **overrides: Any
    ) -> "LSIFilter":
        """Outlier-resilient preset: innovation winsorizing, measurement noise
        taken from the data, gentle drift re-arm. Equivalent to
        ``LSIFilter(expr, var, robust=True, adapt_noise=True,
        drift_reset="inflate", ...)``; ``overrides`` win.
        """
        return cls(expr, var, **{
            "robust": True, "adapt_noise": True, "drift_reset": "inflate",
            **overrides,
        })

    def __init__(
        self,
        expr: str | sp.Expr | Callable[..., Any],
        var: str,
        *,
        regressors: str | Sequence[str] | None = None,
        param_names: Sequence[str] | None = None,
        p0: InitialGuess = None,
        window_size: int = 50,
        min_window: int | None = None,
        adaptive_window: bool = False,
        window_tol: float = 0.001,
        order: int = 5,
        q_diag: list[float] | None = None,
        r: float = 1.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        adapt_r: bool = False,
        adapt_noise: bool = False,
        robust: bool = False,
        huber_c: float = 3.0,
        drift_reset: str = "full",
        drift_inflation: float = 100.0,
    ) -> None:
        """
        Args:
            expr: Model, in any of three forms: a SymPy-expression string
                (e.g. ``"A * sin(w * t)"``), a :class:`sympy.Expr`, or a plain
                Python callable ``f(t, *params)``. A string or expression may
                also reference external regressors (see ``regressors``), as in
                ``"c0 + c1*t + S"`` with ``S`` a measured side-channel, making
                the model ``f(t, regressors, params)``. A callable is evaluated
                numerically and needs no symbolic form, but it has no
                closed-form time derivatives; :meth:`coast` / :meth:`coast_cov`
                are unavailable for it and external regressors are not
                supported.
            var: Main variable name in ``expr`` (a label only for a callable).
            regressors: Optional name(s) of external-regressor channels in
                ``expr``; everything else free is a parameter. When given, each
                ``partial_fit`` / ``predict`` call must supply the regressor
                value(s) for that sample. The same Legendre-spectrum
                measurement still scores the model; the model may now depend on
                exogenous signals (an IMU-derived motion basis, say) and not on
                ``t`` alone. Symbolic models only.
            param_names: For a callable model, the parameter names in signature
                order (those after the leading ``t``); introspected from the
                callable's signature when omitted. Ignored for a symbolic
                model, whose parameters come from the expression.
            p0: Initial parameter estimate (defaults to ones). Ordered like
                :attr:`params_`: sorted names for a symbolic model, signature
                order for a callable.
            window_size: Target (maximum) sliding-window length used for the
                spectral projection. The window grows from ``min_window`` up to
                this size as samples arrive, then slides. A larger window
                smooths more (a more rigid estimate), a smaller one is more
                responsive.
            min_window: Smallest window at which the filter starts producing an
                estimate. The measurement is accumulative: rather than idle
                until ``window_size`` samples have arrived, the filter projects
                whatever the growing window holds once it reaches this many
                points. Defaults to ``order + 2``, the fewest points that admit
                an order-``order`` Legendre projection, putting the estimate to
                work almost immediately, not after a full-window dead time.
                Clamped to ``[order + 2, window_size]``.
            adaptive_window: If True, size the window from the data instead of
                fixing it at ``window_size`` (which becomes the maximum). It
                grows from ``min_window`` while more data still moves the
                estimate (the EWMA of the relative update step exceeds
                ``window_tol``), the sign that the window is not yet wide
                enough to identify static parameters. It shrinks when the
                one-step forecast residual turns systematically autocorrelated,
                in runs of the same sign, the fingerprint of a fit lagging
                changing dynamics. A time-varying signal is therefore tracked
                with a short window and a static one with a wide window, no
                per-regime hand-tuning either way. A detected drift also
                collapses the window to ``min_window`` for the new regime.
            window_tol: Relative-movement threshold for ``adaptive_window``.
                The window stops growing once successive updates move the
                estimate by less than this (EWMA of ``|Δp|/|p|``, default
                0.1%). Smaller grows a wider window.
            order: Legendre spectral order; the measurement is the first
                ``order + 1`` Legendre coefficients of the window. More orders
                means richer observability and a larger measurement vector.
                Clamped so ``order + 1 <= window_size``.
            q_diag: Process-noise variances (per parameter); larger values let a
                parameter drift faster. Defaults to 0.01 each.
            r: Base measurement-noise variance. The per-coefficient variance is
                ``R_j = r * (2j + 1)``, the LSI orthonormal weighting, which
                down-weights the noisier high-order coefficients.
            alpha: Significance level for the multivariate NIS sudden-jump test.
            cusum_k: CUSUM slack (reference value) in innovation standard
                deviations, applied to the zeroth-coefficient (mean/area) arm.
                Set to ``inf`` to disable the CUSUM test.
            cusum_h: CUSUM decision threshold in accumulated standard deviations.
            adapt_r: If True, adapt ``r`` online from an EWMA of the normalized
                innovation power (Mehra-style).
            adapt_noise: If True, take the measurement-noise covariance wholly
                from the data: ``R_diag = v * diag(proj @ proj.T)``, with ``v``
                an online EWMA of the residual variance. The coefficient noise
                is exactly the per-sample noise pushed through the Legendre
                projection, making this self-tuning; it overrides any hand-set
                ``r``. The gain is damped for a smoother output when the stream
                is noisy and freed when it is clean. Pairs naturally with
                ``robust=True``: the winsorization shields the update direction
                while ``v``, estimated from the raw residual, still senses the
                noise.
            robust: If True, gate each Kalman update by the normalized
                innovation. A window whose per-dof Mahalanobis innovation
                exceeds ``huber_c`` has its diagonal measurement noise
                inflated, shrinking the gain, and an outlier-corrupted window
                cannot yank the estimate. The drift detector still sees the raw
                spectral innovation, and a genuine regime shift is detected and
                re-armed; the inflated covariance then disables the gate during
                re-adaptation.
            huber_c: Robust gate threshold in innovation standard deviations
                (per degree of freedom); ~3 keeps clean windows unweighted.
            drift_reset: On a detected drift, ``"full"`` resets the covariance to
                its large initial value and clears the window. ``"inflate"``
                instead multiplies the covariance by ``drift_inflation`` and
                keeps the current estimate and window, a gentler re-adaptation.
                Any other value raises ``ValueError``.
            drift_inflation: Covariance inflation factor for
                ``drift_reset="inflate"``.
        """
        # Resolve the model (string / sympy.Expr / callable), set up regressors,
        # determine the canonical parameter order and compile the fast-path
        # callables once, off the hot path. A callable model has no closed-form
        # time derivatives (coast() is unavailable for it) and no external
        # regressors; the spectral hot path itself needs neither.
        n = self._setup_model(expr, var, regressors, param_names)

        self.p = np.ones(n) if p0 is None else np.asarray(p0, dtype=float)
        self._p_init = np.eye(n) * 10.0
        self.P = self._p_init.copy()
        self.Q = np.diag(q_diag if q_diag is not None else [0.01] * n)
        self.R0 = float(r)
        self.W = int(window_size)
        self.order = max(1, min(int(order), self.W - 1))
        # Accumulative warm-up: start measuring once the growing window holds
        # at least this many points, the fewest an order-`order` projection
        # admits, rather than idling until the window is full.
        floor = self.order + 2
        self.min_window = (
            min(self.W, floor) if min_window is None
            else int(min(self.W, max(floor, min_window)))
        )
        # With adaptive sizing, ``window_size`` is only the maximum (a memory
        # cap) and the effective window is grown from ``min_window``; the two
        # sizing signals live in partial_fit.
        self.adaptive_window = bool(adaptive_window)
        self.window_tol = float(window_tol)
        self._W_eff = self.min_window           # current effective window (adaptive)
        self._W_ref = 2 * (self.order + 1)       # comfortable measurement size
        self._move_ewma = 1.0                    # EWMA of relative estimate movement
        # EWMA sign-autocorrelation of the one-step forecast residual, the
        # shrink signal for the adaptive window (see partial_fit).
        self._resid_sign = 0.0
        self._resid_corr = 0.0
        # Shrink threshold. White residuals give an EWMA(0.1) correlation of
        # mean 0 and std ~0.23; a continuously curving but static oscillation
        # sits around 0.2, a mild and legitimate lag. 0.35 clears both and
        # still sits well below the ~+1 a sustained maneuver-lag produces.
        self._shrink_corr = 0.35

        # Per-coefficient measurement variance: the LSI orthonormal weight.
        j = np.arange(self.order + 1)
        self._R_diag = self.R0 * (2.0 * j + 1.0)
        self._norm = (2.0 * j + 1.0) / 2.0  # standard-Legendre projection scale

        # Empirical spectrum: cached Legendre projection on a window normalized
        # to [-1, 1]. For roughly uniform streaming the in-window sample
        # positions are constant, so the projection is a single cached mat-vec.
        tau = np.linspace(-1.0, 1.0, self.W)
        vander = L.legvander(tau, self.order)          # (W, order+1)
        self._proj = np.linalg.pinv(vander)            # (order+1, W)
        # Below-cap lengths are cached too: the pinv is an SVD, and under
        # adaptive_window the below-cap branch is the common case. A projection
        # depends only on k and order, never on the data; the cache therefore
        # survives a drift. Bounded by (W - min_window + 1) entries.
        self._proj_cache: dict[int, np.ndarray] = {self.W: self._proj}

        # Model spectrum: Gauss-Legendre quadrature on [-1, 1], integrating the
        # model exactly as batch LSI does.
        n_quad = max(2 * (self.order + 1), 16)
        self._nodes, self._qw = L.leggauss(n_quad)
        self._legvander_q = L.legvander(self._nodes, self.order)  # (n_quad, order+1)

        # Ratio threshold for the self-normalizing energy test: how many times
        # its running mean the spectral-energy innovation must reach to flag a
        # jump. Mapped from ``alpha`` through a chi-squared on the energy's
        # effective degrees of freedom, with a margin for the heavier tails an
        # EWMA-estimated scale carries.
        dof = self.order + 1
        self._energy_ratio = 1.6 * chi2.ppf(1 - alpha, df=dof) / dof
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.adapt_r = bool(adapt_r)
        self.adapt_noise = bool(adapt_noise)
        self._v_est = self.R0          # online EWMA of residual variance (adapt_noise)
        self._vn_lambda = 0.05
        self._robust = bool(robust)
        self._huber_c = float(huber_c)
        self.drift_reset = self._validate_drift_reset(drift_reset)
        self.drift_inflation = float(drift_inflation)
        self._r_scale = 1.0  # adaptive multiplier on R when adapt_r

        # Drift detector state (decimated, non-overlapping windows plus a
        # warmup), mirroring EACFilter so the two are directly comparable. One
        # EWMA scale calibrates the spectral-energy jump test and one the
        # mean-coefficient CUSUM; a single scale per statistic, rather than one
        # per coefficient, keeps the test from inheriting the heavy tails of a
        # noisy variance estimate.
        self._ewma_lambda = 0.15
        self._warmup_tests = 5
        self._s_scale = 0.0   # EWMA of the spectral-energy innovation S
        self._e0_scale2 = 0.0  # EWMA of the mean-coefficient innovation power
        self._n_full = 0
        self._n_tests = 0
        self._g_hi = 0.0
        self._g_lo = 0.0
        self.n_drifts_ = 0
        self.drift_flag_ = False
        self.last_drift_direction_ = 0

        # Most recent one-step forecast residual (innovation); see
        # EACFilter.last_residual_. NaN until the window first fills.
        self.last_residual_ = float("nan")

        self._t: list[float] = []
        self._y: list[float] = []
        self._rbuf: list[tuple] = []   # external-regressor window (aligned with _t)

    def _eval(self, func, t_arr, reg_cols=None):
        """Evaluate a compiled callable on the window, broadcasting scalars.
        ``reg_cols`` supplies the external-regressor columns aligned with ``t_arr``
        (passed positionally before the parameters)."""
        if reg_cols is None:
            v = func(t_arr, *self.p)
        else:
            v = func(t_arr, *reg_cols, *self.p)
        if np.ndim(v) == 0:  # constant model/derivative -> broadcast
            v = np.full_like(t_arr, float(v), dtype=float)
        return np.asarray(v, dtype=float)

    def _model_spectrum(self, t0, tn, t_arr=None, reg_cols=None, proj=None):
        """Model Legendre spectrum and its parameter Jacobian over the window.

        Without external regressors the model is a closed-form ``f(t)`` and is
        integrated exactly by Gauss-Legendre quadrature, as in batch LSI.
        Regressors are measured signals with no closed form. A regressor model
        can only be evaluated at the in-window sample positions, and its
        spectrum is the discrete Legendre projection at those samples, through
        the same ``proj`` operator used for the data. Data and model stay on
        one footing for the innovation either way."""
        if self._has_reg:
            fv = self._eval(self._f, t_arr, reg_cols)
            spec = proj @ fv
            h_mat = np.empty((self.order + 1, len(self.p)))
            for jx in range(len(self.p)):
                dv = self._eval(self._jac[jx], t_arr, reg_cols)
                h_mat[:, jx] = proj @ dv
            return spec, h_mat
        t_quad = t0 + (tn - t0) * (self._nodes + 1.0) / 2.0
        fv = self._eval(self._f, t_quad)
        spec = legendre_project(fv, self._qw, self._legvander_q, self._norm)
        h_mat = np.empty((self.order + 1, len(self.p)))
        for jx in range(len(self.p)):
            dv = self._eval(self._jac[jx], t_quad)
            h_mat[:, jx] = legendre_project(
                dv, self._qw, self._legvander_q, self._norm
            )
        return spec, h_mat

    def partial_fit(self, t_new, y_new, regressors=None) -> "LSIFilter":
        """Ingest one ``(t, y[, regressors])`` sample and update in place.

        ``regressors`` (required iff the model declares external regressors) is a
        ``{name: value}`` mapping or a value sequence ordered like ``regressors``.

        A non-finite sample (NaN or inf in ``t``, ``y`` or a regressor) is
        skipped at entry with a ``RuntimeWarning``. It never enters the window,
        cannot poison the innovations of the following ``window_size`` updates,
        and leaves the whole filter state (window, estimate, covariance,
        ``last_residual_``) untouched.
        """
        self.drift_flag_ = False
        if not self._ingest(t_new, y_new, regressors):
            return self  # non-finite sample skipped; state untouched
        cap = self._W_eff if self.adaptive_window else self.W
        while len(self._t) > cap:
            self._t.pop(0)
            self._y.pop(0)
            if self._has_reg:
                self._rbuf.pop(0)
        k = len(self._t)
        if k < self.min_window:
            return self
        full = k >= cap

        t_arr = np.asarray(self._t)
        y_arr = np.asarray(self._y)
        reg_cols = None
        if self._has_reg:
            rb = np.asarray(self._rbuf, dtype=float)
            reg_cols = [rb[:, c] for c in range(rb.shape[1])]
        t0, tn = float(t_arr[0]), float(t_arr[-1])

        # Robust measurement: winsorize the model residual before projecting.
        # Clipping each sample's deviation beyond huber_c robust sigmas (MAD)
        # from the window's median residual de-weights outlier spikes at the
        # sample level, where they would otherwise dominate the high-order
        # Legendre coefficients. The median residual carries any genuine
        # sustained shift and passes through, keeping drift detection alive.
        y_eff = y_arr
        m_win: np.ndarray | None = None
        resid: np.ndarray | None = None
        if self._robust or self.adapt_noise:
            m_win = self._eval(self._f, t_arr, reg_cols)
            resid = y_arr - m_win
        if self._robust:
            # both were assigned above (the guard includes ``self._robust``)
            assert m_win is not None and resid is not None
            med = float(np.median(resid))
            sigma = 1.4826 * float(np.median(np.abs(resid - med)))
            if sigma > 0.0:
                c = self._huber_c * sigma
                y_eff = m_win + (med + np.clip(resid - med, -c, c))

        # Empirical spectrum against model spectrum (quadrature). A window
        # length not yet in the cache gets its projection computed here, once.
        proj = self._proj_cache.get(k)
        if proj is None:
            proj = np.linalg.pinv(L.legvander(np.linspace(-1.0, 1.0, k), self.order))
            self._proj_cache[k] = proj
        beta_data = proj @ y_eff
        beta_model, h_mat = self._model_spectrum(t0, tn, t_arr, reg_cols, proj)
        e_vec = beta_data - beta_model

        # Robustness guard: reject a non-finite innovation or Jacobian, as an
        # unbounded model can overflow into one. A single bad sample must not
        # permanently poison the parameter state with NaNs. Keep the last good
        # estimate.
        if not (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(h_mat))):
            return self

        # One-step forecast residual at the newest sample (pre-update params).
        last_reg = None if reg_cols is None else [c[-1:] for c in reg_cols]
        self.last_residual_ = float(
            y_arr[-1] - self._eval(self._f, t_arr[-1:], last_reg)[0]
        )

        # Drift test on the decimated full-window innovation vector. Only run
        # it once the window is full; a partial, still-growing window's
        # spectrum is not yet a calibrated baseline for the change detector.
        if full:
            self._n_full += 1
            if self._n_full % cap == 0 and self._drift_step(e_vec):
                return self  # reset happened; skip the update

        # A smaller window averages fewer samples and is therefore noisier, so
        # both branches inflate R below the comfortable size to trust it
        # proportionally less. The gain then ramps up smoothly and noisy early
        # windows cannot over-kick the estimate.
        if self.adapt_noise:
            # The spectral-coefficient covariance is the per-sample noise
            # variance propagated through the projection, giving
            # ``R_diag = v * diag(proj @ proj.T)``. The raw un-winsorized
            # residual feeds ``v`` on purpose: the estimate should sense the
            # energy anomalies leak past the gate, and the slow EWMA keeps
            # individual spikes from jerking it. This branch needs no warm-up
            # fudge, because a short window's projection rows have larger norm
            # and ``proj_diag`` inflates ``R`` on its own.
            assert resid is not None  # assigned above (the guard includes adapt_noise)
            v_now = float(np.mean(resid * resid))
            self._v_est = (1.0 - self._vn_lambda) * self._v_est + self._vn_lambda * v_now
            proj_diag = np.einsum("ij,ij->i", proj, proj)
            r_diag = max(self._v_est, 1e-9) * proj_diag
        else:
            r_diag = self._R_diag * self._r_scale
            if self.adaptive_window:
                if k < self._W_ref:
                    r_diag = r_diag * (self._W_ref / k)
            elif not full:
                r_diag = r_diag * (self.W / k)
        s_mat = h_mat @ self.P @ h_mat.T + np.diag(r_diag)
        try:
            gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return self

        step = gain @ e_vec
        p_new = self.p + step
        P_new = (np.eye(len(self.p)) - gain @ h_mat) @ self.P + self.Q
        # Keep the covariance symmetric. Over a long stream ``(I - K H) P``
        # drifts asymmetric, and therefore non-PD, surfacing as a negative
        # ``stderr_`` or ``predict_cov``. Project onto the symmetric part each
        # step; it is O(n^2).
        P_new = 0.5 * (P_new + P_new.T)
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            return self  # reject an ill-conditioned (non-finite) update
        self.p = p_new
        self.P = P_new
        # Calibrate the adaptive noise scale only on full windows, so the
        # non-converged warm-up innovations do not corrupt it.
        if self.adapt_r and not self.adapt_noise and full:
            nis = float(e_vec @ (e_vec / r_diag)) / (self.order + 1)
            self._r_scale = 0.95 * self._r_scale + 0.05 * max(nis, 1e-3)
        # Bidirectional window sizing, shrink taking priority over grow. The
        # shrink signal is the EWMA sign-autocorrelation of the one-step
        # residual: a lagging fit leaves runs of same-sign residuals, a
        # well-matched one leaves white residuals whose signs alternate. It is
        # scale-free, so it neither self-normalizes away a sustained maneuver
        # nor fires on static noise.
        if self.adaptive_window and np.isfinite(self.last_residual_):
            s = 1.0 if self.last_residual_ >= 0.0 else -1.0
            if self._resid_sign != 0.0:
                self._resid_corr = 0.9 * self._resid_corr + 0.1 * (s * self._resid_sign)
            self._resid_sign = s
            mv = float(np.max(np.abs(step) / (np.abs(self.p) + 1e-9)))
            self._move_ewma = 0.9 * self._move_ewma + 0.1 * mv
            if self._resid_corr > self._shrink_corr and self._W_eff > self.min_window:
                self._W_eff -= 1
            elif self._move_ewma > self.window_tol and self._W_eff < self.W:
                self._W_eff += 1
        return self

    def _drift_step(self, e_vec: np.ndarray) -> bool:
        """Multivariate NIS (sudden jump) plus a CUSUM on the mean coefficient
        (sustained drift). Resets and returns True on detection."""
        self._n_tests += 1

        # Two scalar statistics: S, the R-weighted spectral energy, an
        # omnidirectional jump detector across all coefficients; and e0, the
        # mean area-like coefficient, the signed CUSUM channel. Each
        # is standardized against the scale built from previous windows before
        # being folded into it, so a fresh jump shows up large instead of
        # inflating its own threshold.
        r_diag = self._R_diag * self._r_scale
        s_energy = float(e_vec @ (e_vec / r_diag))
        e0 = float(e_vec[0])
        s_ratio = s_energy / self._s_scale if self._s_scale > 0.0 else 0.0
        z0 = e0 / np.sqrt(self._e0_scale2) if self._e0_scale2 > 0.0 else 0.0

        lam = self._ewma_lambda
        self._s_scale = (1.0 - lam) * self._s_scale + lam * s_energy
        self._e0_scale2 = (1.0 - lam) * self._e0_scale2 + lam * (e0 * e0)

        if self._n_tests <= self._warmup_tests:
            return False  # build baselines and let the filter converge first

        self._g_hi = max(0.0, self._g_hi + z0 - self.cusum_k)
        self._g_lo = max(0.0, self._g_lo - z0 - self.cusum_k)
        nis_drift = s_ratio > self._energy_ratio
        cusum_up = self._g_hi > self.cusum_h
        cusum_down = self._g_lo > self.cusum_h
        if nis_drift or cusum_up or cusum_down:
            self._on_drift(up=cusum_up or (not cusum_down and z0 >= 0))
            return True
        return False

    def _on_drift(self, *, up: bool) -> None:
        """Re-arm the filter after a detected drift so it re-adapts quickly."""
        if self.drift_reset == "inflate":
            self.P = self.P * self.drift_inflation
        else:
            self.P = self._p_init.copy()
            self._t, self._y, self._rbuf = [], [], []
        # A wide adaptive window now straddles the change and holds stale
        # old-regime data. Collapse it back to min_window and let it re-grow
        # as the new parameters become identified. The sizing state is
        # recalibrated too: a stale residual sign must not shrink the fresh
        # window or block it from re-growing.
        self._W_eff = self.min_window
        self._move_ewma = 1.0
        self._resid_sign = 0.0
        self._resid_corr = 0.0
        self._g_hi = 0.0
        self._g_lo = 0.0
        self._s_scale = 0.0
        self._e0_scale2 = 0.0
        self._n_full = 0
        self._n_tests = 0
        self.n_drifts_ += 1
        self.drift_flag_ = True
        self.last_drift_direction_ = 1 if up else -1

    # Convenience alias matching the recursive-filter naming.
    update = partial_fit
