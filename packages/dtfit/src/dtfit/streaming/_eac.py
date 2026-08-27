"""Recursive equal-areas filter, the streaming counterpart of EAC.

A Kalman-style recursive estimator whose "measurement" is the area innovation,
experimental minus model area over a sliding window, and whose measurement
Jacobian is the vector of integrated parameter sensitivities. Splitting the
window into ``n_sub`` sub-areas turns that into a vector measurement, giving
more independent equations per step and better observability of coupled
multi-parameter models; ``n_sub=1`` is a single full-window area. The
measurement-noise variance ``R`` can be adapted online (``adapt_r``), which
also keeps the vector measurement well-scaled as the sub-areas shrink.

Concept drift is caught by two complementary tests on the vector sub-area
innovation. Either one resets the covariance and the filter re-adapts quickly.

* A Normalized Innovation Squared (NIS) test on the sub-area energy ``e @ e``,
  a chi^2(n_sub) statistic that catches a single large sudden shift across the
  sub-areas. More sub-areas give the detector several independent channels
  instead of one scalar area, and a jump that a single area would average over
  still lights up the energy.
* A two-sided CUSUM on the signed total area ``e.sum()``. It accumulates
  evidence for a sustained drift and flags it in either direction (upward via
  the high arm, downward via the low arm), which the instantaneous NIS test
  misses.

The symbolic model and its derivatives are compiled once in ``__init__``. Every
``partial_fit`` call is then O(window x params) and contains no SymPy, keeping
the hot path real-time safe.
"""

from typing import Any, Callable, Sequence

import numpy as np
import sympy as sp
from scipy.stats import chi2

from dtfit._core._kernels import simpson_windows, simpson_windows_rows
from dtfit.types import InitialGuess
from ._base import _RecursiveFilter


class EACFilter(_RecursiveFilter):
    """Online equal-areas parameter tracker with drift detection.

    The constructor exposes the full Kalman, window, drift and robustness knob
    set. Most callers should start from a preset classmethod and pass only
    overrides:

    * :meth:`tracking`, a responsive auto-sized window for time-varying
      parameters (the usual default);
    * :meth:`robust`, outlier-resilient gains for noisy streams.
    """

    @classmethod
    def tracking(
        cls, expr: str | sp.Expr | Callable[..., Any], var: str, **overrides: Any
    ) -> "EACFilter":
        """Responsive preset: auto-sized window for tracking drifting parameters.

        Equivalent to ``EACFilter(expr, var, adaptive_window=True, ...)``; any
        keyword in ``overrides`` wins over the preset.
        """
        return cls(expr, var, **{"adaptive_window": True, **overrides})

    @classmethod
    def robust(
        cls, expr: str | sp.Expr | Callable[..., Any], var: str, **overrides: Any
    ) -> "EACFilter":
        """Outlier-resilient preset: innovation-gated, self-adapting noise,
        gentle drift re-arm. Equivalent to ``EACFilter(expr, var, robust=True,
        adapt_r=True, drift_reset="inflate", ...)``; ``overrides`` win.
        """
        return cls(expr, var, **{
            "robust": True, "adapt_r": True, "drift_reset": "inflate",
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
        window_tol: float = 0.02,
        q_diag: list[float] | None = None,
        r: float = 1.0,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        n_sub: int = 1,
        adapt_r: bool = False,
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
                also reference external regressors (see ``regressors``), making
                the model ``f(t, regressors, params)``. A callable is evaluated
                numerically and needs no symbolic form, but it has no
                closed-form time derivatives; :meth:`coast` / :meth:`coast_cov`
                are unavailable for it and external regressors are not
                supported.
            var: Main variable name in ``expr`` (a label only for a callable).
            regressors: Optional name(s) of external-regressor channels in
                ``expr``; everything else free is a parameter. When given, each
                ``partial_fit`` / ``predict`` call supplies the regressor
                value(s) for that sample. The model may then depend on
                exogenous signals and not on ``t`` alone, while still being
                scored by the same integrated area measurement. Symbolic models
                only.
            param_names: For a callable model, the parameter names in signature
                order (those after the leading ``t``); introspected from the
                callable's signature when omitted. Ignored for a symbolic
                model, whose parameters come from the expression.
            p0: Initial parameter estimate (defaults to ones). Ordered like
                :attr:`params_`: sorted names for a symbolic model, signature
                order for a callable.
            window_size: Target (maximum) sliding-window length used for area
                integration. The window grows from ``min_window`` up to this
                size as samples arrive, then slides. A larger window smooths
                more (a more rigid estimate), a smaller one is more responsive.
            min_window: Smallest window at which the filter starts producing an
                estimate. The area measurement is accumulative, so instead of
                idling until ``window_size`` samples have arrived the filter
                integrates whatever the growing window holds once it reaches
                this many points. Defaults to half ``window_size``, a scalar
                area over a few noisy points being unreliable. Clamped to
                ``[2*n_sub, window_size]``. Under ``adaptive_window`` the
                default is a small floor instead, letting the window collapse
                to it on a drift.
            adaptive_window: If True, size the window from the data instead of
                fixing it at ``window_size`` (which becomes the maximum). The
                window grows from ``min_window`` while the state covariance is
                still shrinking (``_adaptive_grow``), settling at a finite
                width on oscillatory and monotone shapes, and collapses back to
                ``min_window`` on a drift to re-acquire the new regime.
                Auto-sizing here is best-effort: stable, but it cannot size a
                global-parameter model (a polynomial's intercept, say) well.
                For that use :class:`LSIFilter`, for which a wide window is
                always safe.
            window_tol: Covariance-reduction threshold for ``adaptive_window``.
                The window stops growing once successive updates shrink
                ``trace(P)`` by less than this fraction (default 2%).
            q_diag: Process-noise variances (per parameter); larger values let
                a parameter drift faster. Defaults to 0.01 each.
            r: Measurement-noise variance of the area innovation.
            alpha: Significance level for the NIS sudden-jump test (e.g. 0.001).
            cusum_k: CUSUM slack (reference value), in innovation standard
                deviations; the smallest sustained shift to ignore. ~0.5 detects
                a one-sigma drift. Set to ``inf`` to disable the CUSUM test.
            cusum_h: CUSUM decision threshold, in accumulated standard
                deviations; larger means fewer false alarms but slower
                detection.
            n_sub: Number of sub-window area measurements per step. ``1`` (the
                default) is a single full-window area. Values above 1 split the
                window into that many sub-areas for a vector measurement,
                giving more independent equations per sample and better
                observability of multi-parameter models.
            adapt_r: If True, adapt the measurement-noise variance ``R`` online
                from an EWMA of the squared innovation (Mehra-style) rather
                than trusting the fixed ``r``.
            robust: If True, gate each Kalman update by the normalized
                innovation. A window whose per-dof Mahalanobis innovation
                exceeds ``huber_c`` has its measurement noise inflated by a
                Huber weight, shrinking the gain, and an outlier-corrupted
                window cannot yank the estimate. The drift detector still sees
                the raw innovation, so a genuine regime shift is detected and
                re-armed; the inflated covariance then disables the gate during
                re-adaptation.
            huber_c: Robust gate threshold in innovation standard deviations
                (per degree of freedom); ~3 keeps clean windows unweighted.
            drift_reset: On a detected drift, ``"full"`` resets the covariance to
                its large initial value and clears the window. ``"inflate"``
                instead multiplies the covariance by ``drift_inflation`` and
                keeps the current estimate and window, a gentler re-adaptation
                that does not discard hard-won parameter information. Any other
                value raises ``ValueError``.
            drift_inflation: Covariance inflation factor for
                ``drift_reset="inflate"``.
        """
        # Resolve the model (string / sympy.Expr / callable), set up regressors,
        # determine the canonical parameter order and compile the fast-path
        # callables once, off the hot path.
        n = self._setup_model(expr, var, regressors, param_names)

        self.p = np.ones(n) if p0 is None else np.asarray(p0, dtype=float)
        self._p_init = np.eye(n) * 10.0
        self.P = self._p_init.copy()
        self.Q = np.diag(q_diag if q_diag is not None else [0.01] * n)
        self.R = float(r)
        self.W = int(window_size)
        # Accumulative warm-up minimum is finalised once n_sub is known (below).
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)

        self.n_sub = max(1, int(n_sub))
        self.adaptive_window = bool(adaptive_window)
        self.window_tol = float(window_tol)
        # Accumulative warm-up: start measuring once the growing window holds at
        # least this many points. The hard floor is 2*n_sub, the fewest that
        # admit n_sub sub-area integrals. An adaptive window takes that small
        # floor so it can collapse here on a regime change and re-grow; the
        # measurement-noise inflation below down-weights its early areas.
        floor = max(2 * self.n_sub, 3)
        if min_window is not None:
            self.min_window = int(min(self.W, max(floor, min_window)))
        elif self.adaptive_window:
            self.min_window = min(self.W, max(floor, 2 * self.n_sub))
        else:
            self.min_window = min(self.W, max(floor, self.W // 2))
        self._W_eff = self.min_window            # current effective window (adaptive)
        self._W_ref = max(4 * self.n_sub, 10)    # comfortable area-measurement size
        self._cov_prev: float | None = None      # previous trace(P) for the grow test
        self.adapt_r = bool(adapt_r)
        self._robust = bool(robust)
        self._huber_c = float(huber_c)
        self.drift_reset = self._validate_drift_reset(drift_reset)
        self.drift_inflation = float(drift_inflation)
        self._r_ewma = float(r)  # adaptive measurement-noise estimate

        # Drift test: the Kalman update runs every sample, but the drift
        # statistic is evaluated only on non-overlapping windows (stride W),
        # because consecutive sliding-window area innovations are heavily
        # autocorrelated and would make a CUSUM false-alarm. The theoretical
        # s_cov is hard to calibrate for an area measurement, so each statistic
        # is standardized by its own running EWMA scale and fires regardless of
        # the innovation's absolute magnitude. The first few tested windows
        # are a warmup that lets the filter converge before arming.
        self._ewma_lambda = 0.25
        self._warmup_tests = 4
        # Two baselines, both built from previous windows. _e_scale2 tracks the
        # energy (sum of squared sub-area innovations) for the omnidirectional
        # NIS spike detector; over n_sub sub-areas that energy is a
        # chi^2(n_sub) statistic. _sum_scale2 tracks the signed total area, the
        # directional channel the two-sided CUSUM accumulates.
        self._energy_ratio = 1.6 * chi2.ppf(1 - alpha, df=self.n_sub) / self.n_sub
        self._e_scale2 = 0.0
        self._sum_scale2 = 0.0
        self._n_full = 0   # full-window steps seen since the last reset
        self._n_tests = 0  # drift evaluations done since the last reset

        # Two-sided CUSUM accumulators (high arm: upward drift; low arm:
        # downward drift) and observable drift state.
        self._g_hi = 0.0
        self._g_lo = 0.0
        self.n_drifts_ = 0
        self.drift_flag_ = False
        self.last_drift_direction_ = 0  # +1 = up, -1 = down, 0 = none yet

        # Most recent one-step forecast residual y_new - f(t_new; p) from the
        # pre-update parameters, i.e. the filter's innovation. It is public
        # because a coordinating layer (a FilterBank fusing several axes) can
        # pool the per-stream innovations into a maneuver detector; they carry
        # a sharper transient than the integrated area statistic. NaN until the
        # window first fills.
        self.last_residual_ = float("nan")

        self._t: list[float] = []
        self._y: list[float] = []
        self._rbuf: list[tuple] = []   # external-regressor window (aligned with _t)

    def _area_jac(self, j: int, t_arr: np.ndarray, reg_cols=None) -> np.ndarray:
        if reg_cols is None:
            d = self._jac[j](t_arr, *self.p)
        else:
            d = self._jac[j](t_arr, *reg_cols, *self.p)
        if np.isscalar(d):
            d = np.full_like(t_arr, d, dtype=float)
        return d

    def partial_fit(self, t_new, y_new, regressors=None) -> "EACFilter":
        """Ingest one ``(t, y[, regressors])`` sample and update in place.

        ``regressors`` (required iff the model declares external regressors) is a
        ``{name: value}`` mapping or a value sequence ordered like ``regressors``.

        A non-finite sample (NaN or inf in ``t``, ``y`` or a regressor) is
        skipped at entry with a ``RuntimeWarning``. It never enters the window,
        cannot poison the innovations of the following ``window_size`` updates,
        and leaves the whole filter state (window, estimate, covariance,
        ``last_residual_``) untouched.
        """
        self.drift_flag_ = False  # true only on the exact step a drift fires
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

        t_arr = np.ascontiguousarray(self._t, dtype=float)
        y_arr = np.ascontiguousarray(self._y, dtype=float)
        reg_cols = None
        if self._has_reg:
            rb = np.ascontiguousarray(self._rbuf, dtype=float)
            reg_cols = [np.ascontiguousarray(rb[:, c]) for c in range(rb.shape[1])]

        # Vector measurement: split the window into n_sub sub-areas. For each,
        # the innovation is (data area - model area) and the measurement
        # Jacobian row is the vector of integrated parameter sensitivities. The
        # model and its sensitivities are evaluated once over the window and
        # integrated per sub-area by the compiled Simpson kernel.
        np_params = len(self.p)
        m_func = self._f(t_arr, *self.p) if reg_cols is None else \
            self._f(t_arr, *reg_cols, *self.p)
        if np.ndim(m_func) == 0:  # constant model -> broadcast to the window
            m_func = np.full_like(t_arr, float(m_func))
        m_func = np.ascontiguousarray(m_func, dtype=float)
        sub = self._subwindows(t_arr.size)
        starts = np.array([a for a, _ in sub], dtype=np.intp)
        stops = np.array([b for _, b in sub], dtype=np.intp)

        # Robust measurement: winsorize the model residual before integrating.
        # Clipping each sample's deviation beyond huber_c robust sigmas (MAD)
        # from the window's median residual de-weights outlier spikes at the
        # sample level, and the integral can no longer carry them. The median
        # residual carries any genuine sustained shift and passes through
        # untouched, keeping drift detection alive.
        y_eff = y_arr
        if self._robust:
            resid = y_arr - m_func
            med = float(np.median(resid))
            sigma = 1.4826 * float(np.median(np.abs(resid - med)))
            if sigma > 0.0:
                c = self._huber_c * sigma
                y_eff = m_func + (med + np.clip(resid - med, -c, c))

        e_vec = simpson_windows(y_eff, t_arr, starts, stops) - simpson_windows(
            m_func, t_arr, starts, stops
        )
        jac_rows = np.vstack(
            [self._area_jac(j, t_arr, reg_cols) for j in range(np_params)]
        )
        h_mat = simpson_windows_rows(jac_rows, t_arr, starts, stops).T

        # Robustness guard: an unbounded nonlinear model (a decaying
        # exponential whose time constant wanders toward its singular value)
        # can overflow and leave the innovation or Jacobian non-finite.
        # Committing that would poison the parameter state permanently and
        # every later predict() would return NaN. Reject the sample and keep
        # the last good estimate; the next clean sample re-adapts.
        if not (np.all(np.isfinite(e_vec)) and np.all(np.isfinite(h_mat))):
            return self

        # One-step forecast residual at the newest sample (pre-update params).
        self.last_residual_ = float(y_arr[-1] - m_func[-1])

        # Only run the drift detector once the window is full; a partial,
        # still-growing window's area is not yet a calibrated baseline for it.
        if full:
            self._n_full += 1
            if self._n_full % cap == 0 and self._drift_step(e_vec):
                return self  # reset happened; skip the update

        # A smaller window averages fewer samples and is therefore noisier.
        # Inflating the measurement noise trusts it proportionally less, so the
        # gain ramps up smoothly and noisy first windows cannot over-kick the
        # estimate. A fixed window damps the W/k growing warm-up; an adaptive
        # one damps only until the measurement size W_ref is reached.
        r_eff = self._r_ewma if self.adapt_r else self.R
        if self.adaptive_window:
            if k < self._W_ref:
                r_eff = r_eff * (self._W_ref / k)
        elif not full:
            r_eff = r_eff * (self.W / k)
        s_mat = h_mat @ self.P @ h_mat.T + r_eff * np.eye(len(sub))
        try:
            gain = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        except np.linalg.LinAlgError:
            return self

        step = gain @ e_vec
        p_new = self.p + step
        P_new = (np.eye(np_params) - gain @ h_mat) @ self.P + self.Q
        # Keep the covariance symmetric. ``(I - K H) P`` is the algebraically
        # minimal but numerically least stable update: over a long,
        # effectively infinite-horizon stream, rounding makes ``P`` drift
        # asymmetric and can push an eigenvalue negative, surfacing as a
        # negative ``stderr_`` or ``predict_cov``. Projecting onto the
        # symmetric part each step is O(n^2) and prevents that accumulation.
        P_new = 0.5 * (P_new + P_new.T)
        if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(P_new))):
            return self  # reject an ill-conditioned (non-finite) update
        self.p = p_new
        self.P = P_new
        # Calibrate the adaptive noise scale only on full windows, so the
        # non-converged warm-up innovations do not corrupt it.
        if self.adapt_r and full:
            # Mehra-style: track the per-measurement innovation power.
            self._r_ewma = 0.95 * self._r_ewma + 0.05 * float(
                (e_vec @ e_vec) / len(sub)
            )
            # Floor the adaptive noise so a very clean stream cannot drive
            # R -> 0. A vanishing R saturates the gain (S ~= H P H^T), handing
            # the next outlier full authority over the estimate.
            self._r_ewma = max(self._r_ewma, 1e-6 * self.R)
        # Widen while more data still moves the estimate, capped at the
        # maximum. A global-parameter model widens on its own; a local one
        # stops and the window slides.
        if self.adaptive_window and self._W_eff < self.W:
            self._adaptive_grow(step, e_vec)
        return self

    def _adaptive_grow(self, step: np.ndarray, e_vec: np.ndarray) -> None:
        """Decide whether to widen the adaptive window by one sample.

        The criterion is covariance reduction: grow while the state covariance
        is still shrinking meaningfully. :class:`LSIFilter`'s estimate-movement
        test does not transfer here, since the scalar area estimate jitters
        and, on a decaying signal, is non-stationary even with static
        parameters. A movement signal would widen the window without bound and
        the area measurement then fails on it. Covariance reduction settles at
        a finite width on oscillatory and monotone shapes and never diverges.
        It cannot size a global-parameter model (a polynomial's intercept)
        well, because the area covariance under-estimates that extrapolation
        bias; :class:`LSIFilter` is the better choice for those. Override this
        method to plug in a different criterion.
        """
        tr = float(np.trace(self.P))
        rel = (self._cov_prev - tr) / (self._cov_prev + 1e-12) \
            if self._cov_prev is not None else 1.0
        self._cov_prev = tr
        if rel > self.window_tol:
            self._W_eff += 1

    def _subwindows(self, size: int) -> list[tuple[int, int]]:
        """Contiguous ``[start, stop)`` index spans partitioning the window into
        ``n_sub`` sub-areas, each with at least 2 points for integration."""
        n = min(self.n_sub, max(1, size // 2))
        if n == 1:
            return [(0, size)]
        edges = np.linspace(0, size, n + 1).astype(int)
        return [(int(edges[k]), int(edges[k + 1])) for k in range(n)]

    def _drift_step(self, e_vec: np.ndarray) -> bool:
        """Vector NIS (sudden jump) plus two-sided CUSUM (sustained drift) on
        one decimated sub-area innovation. Resets and returns True on a
        detection.

        Two scalar statistics, each standardized against the scale built from
        previous windows before being folded into it, so a fresh jump shows up
        large instead of inflating its own threshold.

        * ``energy = e_vec @ e_vec``, the omnidirectional chi^2(n_sub) spike
          detector. More sub-areas raise the degrees of freedom and sharpen the
          signal, and a regime shift that a single scalar area would average
          over still lights up the energy.
        * ``e_sum = e_vec.sum()``, the signed total area: a directional channel
          the two-sided CUSUM accumulates to flag a sustained shift up or down.
        """
        self._n_tests += 1

        energy = float(e_vec @ e_vec)
        e_sum = float(e_vec.sum())
        s_ratio = energy / self._e_scale2 if self._e_scale2 > 0.0 else 0.0
        z = e_sum / np.sqrt(self._sum_scale2) if self._sum_scale2 > 0.0 else 0.0
        lam = self._ewma_lambda
        self._e_scale2 = (1.0 - lam) * self._e_scale2 + lam * energy
        self._sum_scale2 = (1.0 - lam) * self._sum_scale2 + lam * (e_sum * e_sum)

        if self._n_tests <= self._warmup_tests:
            return False  # build the baseline scales and let the filter converge

        # NIS: energy spike across the sub-areas (omnidirectional sudden shift);
        # CUSUM: a sustained signed shift up (high arm) or down (low arm).
        self._g_hi = max(0.0, self._g_hi + z - self.cusum_k)
        self._g_lo = max(0.0, self._g_lo - z - self.cusum_k)
        nis_drift = s_ratio > self._energy_ratio
        cusum_up = self._g_hi > self.cusum_h
        cusum_down = self._g_lo > self.cusum_h
        if nis_drift or cusum_up or cusum_down:
            self._on_drift(up=bool(cusum_up or (nis_drift and z > 0)))
            return True
        return False

    def _on_drift(self, *, up: bool) -> None:
        """Re-arm the filter after a detected drift so it re-adapts quickly.

        ``"full"`` discards the covariance and window. That is fast, but it
        throws away the parameter estimate's history. ``"inflate"`` keeps the
        current estimate and window and blows up the covariance by
        ``drift_inflation`` instead, letting new data dominate: the gentler
        re-adaptation.
        """
        if self.drift_reset == "inflate":
            self.P = self.P * self.drift_inflation
        else:
            self.P = self._p_init.copy()
            self._t, self._y, self._rbuf = [], [], []
        # Collapse the adaptive window back to min_window. A wide window now
        # straddles the change and holds stale old-regime data; re-acquire the
        # new regime from the freshest samples and re-grow as it is identified.
        self._W_eff = self.min_window
        self._cov_prev = None
        self._g_hi = 0.0
        self._g_lo = 0.0
        self._e_scale2 = 0.0
        self._sum_scale2 = 0.0
        self._n_full = 0
        self._n_tests = 0
        self.n_drifts_ += 1
        self.drift_flag_ = True
        self.last_drift_direction_ = 1 if up else -1

    # Convenience alias matching the recursive-filter naming.
    update = partial_fit
