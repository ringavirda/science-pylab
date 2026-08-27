"""The methods dtfit is scored against across the experiment suite.

Each one is an established method a practitioner would actually reach for,
wrapped behind a small uniform helper so the harnesses can call them alike:

* classical curve fitting: SciPy ``curve_fit`` (Levenberg-Marquardt NLLS) and
  ``numpy.polyfit``;
* the Western parameter-estimation lineage a signal-processing or system-ID
  reviewer asks about: Prony's method, its subspace successors Matrix Pencil
  and ESPRIT, Golub-Pereyra variable projection, and the classical method of
  moments (:func:`prony_fit`, :func:`matrix_pencil_fit`, :func:`varpro_fit`,
  :func:`moment_match_fit`). These do the same job as dtfit's EAC/LSI, and
  recover the same nonlinear parameters (rates, frequencies, amplitudes) by
  other routes: algebraic recurrences, separable least squares, integral
  moment matching;
* neural nets: scikit-learn ``MLPRegressor`` (batch or ``partial_fit``) and
  PyTorch MLP / LSTM sequence forecasters;
* classical time series: statsmodels ARIMA / SARIMAX;
* trajectory tracking: a constant-acceleration Kalman filter in plain numpy;
* the naive random-walk forecast benchmark.

The torch and statsmodels backends import lazily behind ``HAVE_TORCH`` /
``HAVE_STATSMODELS``, so a core install still runs the suite with those rows
skipped. The deep forecasting research methods (DLinear, TimesNet, Time-LLM)
are not re-implemented: ``experiments/cases/06_benchmark_ltsf`` compares
against their published benchmark numbers instead.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    HAVE_STATSMODELS = True
except Exception:  # pragma: no cover
    HAVE_STATSMODELS = False


# classical curve fitting
def scipy_curve_fit(x, y, func, p0, *, bounds=None, maxfev=20000):
    """Levenberg-Marquardt / trust-region NLLS via scipy.optimize.curve_fit."""
    from scipy.optimize import curve_fit

    kwargs = {"p0": p0, "maxfev": maxfev}
    if bounds is not None:
        kwargs["bounds"] = bounds
    popt, _ = curve_fit(func, x, y, **kwargs)
    return popt


def polyfit_predict(x, y, x_eval, deg=5):
    c = np.polyfit(x, y, deg)
    return np.polyval(c, x_eval)


# the established long-memory (Hurst) estimators
def hurst_rs(x, *, n_scales=12):
    """Rescaled-range (R/S) Hurst estimator (Mandelbrot).

    For each scale ``m`` the average rescaled range ``E[R/S]`` over the
    non-overlapping windows scales as ``m^H``; ``H`` is the log-log slope.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    scales = np.unique(np.round(np.geomspace(8, max(16, N // 4), n_scales)).astype(int))
    scales = scales[scales >= 8]
    rs, used = [], []
    for m in scales:
        nb = N // m
        if nb < 1:
            continue
        vals = []
        for b in range(nb):
            seg = x[b * m:(b + 1) * m]
            z = np.cumsum(seg - seg.mean())
            s = seg.std()
            if s > 0:
                vals.append((z.max() - z.min()) / s)
        if vals:
            rs.append(float(np.mean(vals)))
            used.append(m)
    if len(used) < 3:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(rs), 1)[0])


def hurst_dfa(x, *, n_scales=12, order=1):
    """Detrended-fluctuation-analysis (DFA) Hurst estimator (Peng et al.).

    The integrated profile is split into windows of size ``m``; the RMS of the
    order-``order`` polynomial-detrended fluctuation scales as ``m^H`` (for the
    cumulative profile the DFA exponent equals the Hurst exponent). ``H`` is the
    log-log slope.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.round(np.geomspace(8, max(16, N // 4), n_scales)).astype(int))
    scales = scales[scales >= 8]
    F, used = [], []
    for m in scales:
        nb = N // m
        if nb < 1:
            continue
        rms = []
        tt = np.arange(m)
        for b in range(nb):
            seg = y[b * m:(b + 1) * m]
            fit = np.polyval(np.polyfit(tt, seg, order), tt)
            rms.append(float(np.mean((seg - fit) ** 2)))
        if rms:
            F.append(float(np.sqrt(np.mean(rms))))
            used.append(m)
    if len(used) < 3:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(F), 1)[0])


def mlp_curve(x, y, x_eval, *, hidden=(64, 64), max_iter=2000, seed=0):
    """A black-box neural-net regressor f(x)->y (1-D curve fitting)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    xs = StandardScaler().fit(x.reshape(-1, 1))
    Xtr = xs.transform(x.reshape(-1, 1))
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                       random_state=seed)
    net.fit(Xtr, y)
    return net.predict(xs.transform(np.asarray(x_eval).reshape(-1, 1)))


# forecasting baselines: train on a series, predict `horizon` steps ahead
def random_walk_forecast(train, horizon):
    """Persist the last observed value (the standard hard-to-beat benchmark)."""
    return np.full(horizon, float(train[-1]))


def _make_windows(series, lookback):
    X = np.array([series[i:i + lookback] for i in range(len(series) - lookback)])
    Y = np.array([series[i + lookback] for i in range(len(series) - lookback)])
    return X, Y


def mlp_forecast(train, horizon, *, lookback=24, hidden=(64, 64), max_iter=1500,
                 seed=0, incremental=False):
    """Autoregressive sklearn-MLP forecaster (recursive multi-step).

    ``incremental=True`` trains it by ``partial_fit`` over mini-batches instead
    of one batch ``fit``, the streaming-friendly form of the same baseline.
    """
    from sklearn.neural_network import MLPRegressor

    train = np.asarray(train, dtype=float)
    mu, sd = train.mean(), train.std() + 1e-12
    s = (train - mu) / sd
    X, Y = _make_windows(s, lookback)
    if len(X) < 5:
        return random_walk_forecast(train, horizon)
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter,
                       random_state=seed)
    if incremental:
        bs = max(32, len(X) // 20)
        for _ in range(10):
            for i in range(0, len(X), bs):
                net.partial_fit(X[i:i + bs], Y[i:i + bs])
    else:
        net.fit(X, Y)
    window = list(s[-lookback:])
    out = []
    for _ in range(horizon):
        nxt = float(net.predict(np.array(window[-lookback:]).reshape(1, -1))[0])
        out.append(nxt)
        window.append(nxt)
    return np.array(out) * sd + mu


def arima_forecast(train, horizon, *, order=(2, 1, 2), seasonal_order=None):
    """statsmodels ARIMA / SARIMAX point forecast."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    train = np.asarray(train, dtype=float)
    if seasonal_order is not None:
        model = sm.tsa.statespace.SARIMAX(
            train, order=order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(train, order=order)
    fit = model.fit() if seasonal_order is None else model.fit(disp=False)
    return np.asarray(fit.forecast(steps=horizon), dtype=float)


# torch sequence nets (small, CPU)
def _torch_seq_forecast(train, horizon, *, lookback, kind, epochs, seed):
    torch.manual_seed(seed)
    train = np.asarray(train, dtype=float)
    mu, sd = train.mean(), train.std() + 1e-12
    s = (train - mu) / sd
    X, Y = _make_windows(s, lookback)
    if len(X) < 5:
        return random_walk_forecast(train, horizon)
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

    if kind == "lstm":
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 32, batch_first=True)
                self.fc = nn.Linear(32, 1)

            def forward(self, x):
                o, _ = self.lstm(x.unsqueeze(-1))
                return self.fc(o[:, -1, :])
        net = Net()
    else:  # mlp
        net = nn.Sequential(nn.Linear(lookback, 64), nn.ReLU(),
                            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(Xt), Yt)
        loss.backward()
        opt.step()

    net.eval()
    window = list(s[-lookback:])
    out = []
    with torch.no_grad():
        for _ in range(horizon):
            xin = torch.tensor(window[-lookback:], dtype=torch.float32).reshape(1, -1)
            nxt = float(net(xin).item())
            out.append(nxt)
            window.append(nxt)
    return np.array(out) * sd + mu


def lstm_forecast(train, horizon, *, lookback=24, epochs=200, seed=0):
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    return _torch_seq_forecast(train, horizon, lookback=lookback, kind="lstm",
                               epochs=epochs, seed=seed)


def torch_mlp_forecast(train, horizon, *, lookback=24, epochs=300, seed=0):
    if not HAVE_TORCH:
        raise RuntimeError("torch not available")
    return _torch_seq_forecast(train, horizon, lookback=lookback, kind="mlp",
                               epochs=epochs, seed=seed)


# the constant-acceleration Kalman filter: trajectory tracking's gold standard
class KalmanCA:
    """Per-axis constant-acceleration Kalman filter (position measurements).

    State ``[p, v, a]`` per axis; the standard model used for tracking and
    short-horizon trajectory prediction. ``dim`` axes are tracked independently.
    """

    def __init__(self, dim=3, dt=1.0, q=1e-2, r=1.0):
        self.dim = dim
        self.dt = dt
        self.F = np.array([[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([[1.0, 0.0, 0.0]])
        self.Q = q * np.array([[dt**4 / 4, dt**3 / 2, dt**2 / 2],
                               [dt**3 / 2, dt**2, dt],
                               [dt**2 / 2, dt, 1.0]])
        self.R = np.array([[r]])
        self.x = [np.zeros((3, 1)) for _ in range(dim)]
        self.P = [np.eye(3) * 10.0 for _ in range(dim)]
        self._init = False
        # Per-axis one-step innovations of the last update and their fused
        # normalized-innovation-squared, ~chi-square(dim) while no maneuver is
        # under way. An external detector can then drive the Kalman baseline
        # with the identical adaptive re-arming it drives dtfit with, so the
        # maneuver-tracking comparison runs on one piece of machinery.
        self.last_residuals_ = np.zeros(dim)
        self.last_nis_ = 0.0

    def update(self, z):
        """Ingest one position measurement ``z`` (length ``dim``)."""
        z = np.asarray(z, dtype=float)
        if not self._init:
            for d in range(self.dim):
                self.x[d][0, 0] = z[d]
            self._init = True
            self.last_residuals_ = np.zeros(self.dim)
            self.last_nis_ = 0.0
            return self.position()
        nis = 0.0
        res = np.zeros(self.dim)
        for d in range(self.dim):
            xp = self.F @ self.x[d]
            Pp = self.F @ self.P[d] @ self.F.T + self.Q
            y = z[d] - (self.H @ xp)[0, 0]
            S = (self.H @ Pp @ self.H.T + self.R)[0, 0]
            res[d] = y
            nis += y * y / S
            K = (Pp @ self.H.T) / S
            self.x[d] = xp + K * y
            self.P[d] = (np.eye(3) - K @ self.H) @ Pp
        self.last_residuals_ = res
        self.last_nis_ = float(nis)
        return self.position()

    def inflate(self, factor):
        """Inflate every axis' covariance: the adaptive re-arming hook, mirror
        of the dtfit filter's ``inflate``."""
        for d in range(self.dim):
            self.P[d] = self.P[d] * float(factor)

    def position(self):
        return np.array([self.x[d][0, 0] for d in range(self.dim)])

    def forecast(self, horizon):
        """Roll the state forward ``horizon`` steps (no new measurements)."""
        xs = [self.x[d].copy() for d in range(self.dim)]
        out = np.zeros((horizon, self.dim))
        for k in range(horizon):
            for d in range(self.dim):
                xs[d] = self.F @ xs[d]
                out[k, d] = xs[d][0, 0]
        return out


# the rest of the standard classical forecasting toolkit
def seasonal_naive_forecast(train, horizon, *, period):
    """Repeat the last observed season: the standard seasonal benchmark."""
    train = np.asarray(train, dtype=float)
    if period <= 0 or train.size < period:
        return random_walk_forecast(train, horizon)
    last = train[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last, reps)[:horizon]


def drift_forecast(train, horizon):
    """Random walk with drift: extrapolate the average per-step change
    (Hyndman's "drift method")."""
    train = np.asarray(train, dtype=float)
    if train.size < 2:
        return random_walk_forecast(train, horizon)
    slope = (train[-1] - train[0]) / (train.size - 1)
    return train[-1] + slope * np.arange(1, horizon + 1)


def poly_extrap_forecast(train, horizon, *, deg=2):
    """Fit a global polynomial and extrapolate: the surrogate-fit baseline,
    carrying no parametric structure and extrapolating by curvature alone."""
    train = np.asarray(train, dtype=float)
    t = np.arange(train.size)
    c = np.polyfit(t, train, deg)
    return np.polyval(c, np.arange(train.size, train.size + horizon))


def ets_forecast(train, horizon, *, trend="add", seasonal=None, period=None,
                 damped=False):
    """Holt-Winters exponential smoothing (statsmodels
    ``ExponentialSmoothing``): the workhorse classical forecaster, a level plus
    an optional trend and season."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    train = np.asarray(train, dtype=float)
    kw = dict(trend=trend, damped_trend=damped and trend is not None)
    if seasonal is not None and period and train.size >= 2 * period:
        kw.update(seasonal=seasonal, seasonal_periods=period)
    fit = ExponentialSmoothing(train, **kw).fit()
    return np.asarray(fit.forecast(horizon), dtype=float)


def theta_forecast(train, horizon, *, period=None):
    """The Theta method (statsmodels ``ThetaModel``): the M3-competition
    winner, a robust and widely deployed decomposition forecaster."""
    if not HAVE_STATSMODELS:
        raise RuntimeError("statsmodels not available")
    from statsmodels.tsa.forecasting.theta import ThetaModel
    train = np.asarray(train, dtype=float)
    pr = period if (period and train.size >= 2 * period) else 1
    tm = ThetaModel(train, period=pr) if pr > 1 else ThetaModel(train, period=1,
                                                                deseasonalize=False)
    return np.asarray(tm.fit().forecast(horizon), dtype=float)


def sarima_forecast(train, horizon, *, order=(1, 1, 1), seasonal_order=None):
    """Seasonal ARIMA point forecast (statsmodels SARIMAX): ARIMA's seasonal
    extension, the standard statistical model for seasonal series."""
    return arima_forecast(train, horizon, order=order,
                          seasonal_order=seasonal_order)


# robust and nonparametric curve fitting, for the parameter-estimation studies
def robust_curve_fit(x, y, func, p0, *, bounds=None, loss="soft_l1",
                     f_scale=1.0, maxfev=20000):
    """NLLS under a robust loss (scipy ``least_squares``): the standard way to
    fit a known model when outliers are present. Huber and soft-L1 down-weight
    large residuals, making this the established robust analog of
    ``curve_fit``."""
    from scipy.optimize import least_squares
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def resid(p):
        return np.asarray(func(x, *p), dtype=float) - y

    kw = dict(loss=loss, f_scale=f_scale, max_nfev=maxfev)
    if bounds is not None:
        kw["bounds"] = bounds
    sol = least_squares(resid, np.asarray(p0, float), **kw)
    return np.asarray(sol.x, dtype=float)


def gp_curve(x, y, x_eval, *, seed=0):
    """Gaussian-process regression (sklearn): the standard nonparametric
    Bayesian smoother. It fits any smooth curve and recovers no physical
    parameters at all, which is the point of comparing it to a structured
    fit."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    x = np.asarray(x, float).reshape(-1, 1)
    y = np.asarray(y, float)
    span = float(x.max() - x.min()) or 1.0
    k = ConstantKernel(1.0) * RBF(span / 10) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                  random_state=seed, n_restarts_optimizer=1)
    gp.fit(x, y)
    return gp.predict(np.asarray(x_eval, float).reshape(-1, 1))


# The Western parameter-estimation lineage: the signal-processing / system-ID
# foils a reviewer outside the Pukhov school reaches for. Each recovers the
# same nonlinear parameters as dtfit's EAC/LSI (growth and decay rates,
# frequencies, amplitudes) by a different classical route:
#
#   * Prony / Matrix Pencil / ESPRIT, the algebraic and subspace route. An
#     exponential sum obeys a linear recurrence, so its modes are the roots of
#     a characteristic polynomial (Prony) or the eigenvalues of a shift on the
#     data's signal subspace (Matrix Pencil / ESPRIT). Closed form, no
#     iteration.
#   * Variable projection (Golub-Pereyra), the separable-NLLS route: eliminate
#     the linearly-appearing amplitudes in closed form, then optimise over the
#     nonlinear shape parameters alone.
#   * Method of moments / GMM, the integral-moment route: match the model's
#     integral moments to the data's. This is the unconditioned ancestor of
#     LSI, since monomial moments form an ill-conditioned Hilbert system, which
#     is the pathology LSI removes by switching to an orthogonal Legendre
#     basis.
#
# All are pure NumPy/SciPy and return a small uniform result, so the domain
# harnesses score their recovery and prediction just as they score dtfit.
class ExpSumModel:
    """A recovered sum of complex exponentials ``y(t) ~= Re sum_i amp_i e^{rate_i (t - t0)}``.

    The output of :func:`prony_fit` / :func:`matrix_pencil_fit`. ``rate`` are the
    continuous-time poles ``mu_i = log(z_i)/dt`` (real part = growth/decay, imag
    part = angular frequency); ``amp`` the complex amplitudes. For a real signal
    the conjugate-pair structure makes the reconstruction real.
    """

    def __init__(self, rate, amp, t0, dt):
        self.rate = np.asarray(rate, dtype=complex)
        self.amp = np.asarray(amp, dtype=complex)
        self.t0 = float(t0)
        self.dt = float(dt)

    @property
    def damping(self) -> np.ndarray:
        """Per-mode damping ``-Re(rate)`` (positive = decaying)."""
        return -self.rate.real

    @property
    def frequency(self) -> np.ndarray:
        """Per-mode angular frequency ``|Im(rate)|`` (rad per unit ``t``)."""
        return np.abs(self.rate.imag)

    def predict(self, t_eval) -> np.ndarray:
        t_eval = np.asarray(t_eval, dtype=float)
        basis = np.exp(np.outer(t_eval - self.t0, self.rate))  # (n, K)
        return np.real(basis @ self.amp)


def _exp_amplitudes(t, y, rate, t0) -> np.ndarray:
    """Least-squares complex amplitudes of an exponential sum with known poles."""
    vander = np.exp(np.outer(np.asarray(t, float) - t0, np.asarray(rate, complex)))
    amp, *_ = np.linalg.lstsq(vander, np.asarray(y, dtype=complex), rcond=None)
    return amp


def prony_fit(t, y, n_modes) -> ExpSumModel:
    """Prony's method (1795): fit a sum of ``n_modes`` exponentials.

    An exponential sum satisfies a linear recurrence, so the algorithm is two
    linear steps. Solve the overdetermined recurrence for its coefficients; the
    roots of the characteristic polynomial are then the discrete poles
    ``z_i = e^{mu_i dt}``, from which the rates ``mu_i`` follow and the
    amplitudes drop out of a Vandermonde least squares. Non-iterative and exact
    without noise, it is the origin of the whole exponential-fitting lineage
    and the algebraic counterpart to dtfit's integral EAC/LSI. It is
    noise-sensitive (:func:`matrix_pencil_fit` adds the SVD denoising step that
    fixes this) and assumes near-uniform sampling. ``y`` may be real or
    complex.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    k = int(n_modes)
    if n < 2 * k + 1:
        raise ValueError(f"need >= {2 * k + 1} samples for {k} Prony modes; got {n}")
    dt = float(np.mean(np.diff(t)))
    # Linear-prediction system: y[m] = sum_j a[j] y[m-j], m = k .. n-1.
    # Row m is [y[m-1], y[m-2], ..., y[m-k]] (the window before m, reversed).
    rows = np.array([y[m - k:m][::-1] for m in range(k, n)])
    rhs = y[k:n]
    a, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    # Characteristic polynomial z^k - a1 z^{k-1} - ... - ak ; its roots are z_i.
    z = np.roots(np.concatenate([[1.0], -a]))
    z = z[z != 0]
    rate = np.log(z.astype(complex)) / dt
    amp = _exp_amplitudes(t, y, rate, t[0])
    return ExpSumModel(rate, amp, t[0], dt)


def matrix_pencil_fit(t, y, n_modes, *, pencil=None) -> ExpSumModel:
    """Matrix Pencil (Hua & Sarkar 1990): the SVD-robust successor of Prony, in
    the same family as ESPRIT and MUSIC.

    Rather than root a noise-sensitive recurrence polynomial, it builds a
    Hankel matrix from the samples, denoises it by truncating the SVD to the
    ``n_modes`` dominant singular vectors, and recovers the discrete poles
    ``z_i`` as the eigenvalues of a one-row shift on the signal subspace. That
    subspace step is what makes it markedly more noise-robust than Prony, and
    it is the method a signal-processing reviewer names for exponential or
    sinusoid recovery. Assumes near-uniform sampling. ``pencil`` is the pencil
    parameter ``L``, defaulting to ``N/2``, the standard rank-robust choice.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = y.size
    k = int(n_modes)
    dt = float(np.mean(np.diff(t)))
    L = int(pencil) if pencil is not None else n // 2
    L = max(k + 1, min(L, n - k - 1))
    # Hankel Y, shape (n-L, L+1): Y[i, j] = y[i + j].
    hankel = np.array([y[i:i + L + 1] for i in range(n - L)])
    if min(hankel.shape) <= k:
        raise ValueError(f"too few samples ({n}) to resolve {k} matrix-pencil modes")
    _, _, vh = np.linalg.svd(hankel, full_matrices=False)
    vk = vh.conj().T[:, :k]                       # dominant right singular vectors
    # Rotational invariance: V2 = z V1 on the (denoised) signal subspace.
    psi = np.linalg.pinv(vk[:-1, :]) @ vk[1:, :]
    z = np.linalg.eigvals(psi)
    z = z[z != 0]
    rate = np.log(z.astype(complex)) / dt
    amp = _exp_amplitudes(t, y, rate, t[0])
    return ExpSumModel(rate, amp, t[0], dt)


class VarProModel:
    """A fitted separable model ``y ~= Phi(alpha; t) @ c`` (output of :func:`varpro_fit`).

    ``alpha`` are the recovered nonlinear shape parameters, ``linear`` the linear
    coefficients ``c`` solved in closed form at ``alpha``.
    """

    def __init__(self, alpha, linear, design):
        self.alpha = np.asarray(alpha, dtype=float)
        self.linear = np.asarray(linear, dtype=float)
        self._design = design

    def predict(self, t_eval) -> np.ndarray:
        phi = np.asarray(self._design(self.alpha, np.asarray(t_eval, float)),
                         dtype=float)
        return phi @ self.linear


def varpro_fit(t, y, design, alpha0, *, bounds=None, maxfev=20000) -> VarProModel:
    """Variable projection (Golub & Pereyra 1973) for a separable model.

    Many exponential, harmonic and basis models are linear in some parameters
    (the amplitudes ``c``) and nonlinear in the rest (rates or frequencies
    ``alpha``): ``y ~= Phi(alpha; t) @ c``. VarPro eliminates ``c``
    analytically, since for any ``alpha`` the optimal ``c`` is
    ``Phi(alpha)^+ y``, and optimises only over the smaller, better-conditioned
    nonlinear set by minimising the projection residual ``(I - Phi Phi^+) y``.
    It is the standard modern method for multi-exponential and Fourier fitting,
    converging far better than throwing every parameter at a joint NLLS. A
    structural cousin of dtfit's LSI, which is likewise linear in an amplitude
    once the model is projected onto its basis.

    Args:
        design: ``design(alpha, t) -> Phi`` building the ``(len(t), K)`` matrix of
            the ``K`` linearly-appearing basis columns at the nonlinear ``alpha``.
        alpha0: Initial nonlinear parameters.
        bounds: Optional ``(lo, hi)`` bounds on ``alpha`` (as ``least_squares`` takes).
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    def resid(alpha):
        phi = np.asarray(design(alpha, t), dtype=float)
        c, *_ = np.linalg.lstsq(phi, y, rcond=None)
        return phi @ c - y

    kw = {"max_nfev": maxfev}
    if bounds is not None:
        kw["bounds"] = bounds
    sol = least_squares(resid, np.asarray(alpha0, dtype=float), **kw)
    phi = np.asarray(design(sol.x, t), dtype=float)
    c, *_ = np.linalg.lstsq(phi, y, rcond=None)
    return VarProModel(sol.x, c, design)


def moment_match_fit(t, y, func, p0, *, bounds=None, n_moments=None,
                     maxfev=20000) -> np.ndarray:
    """The classical method of moments / GMM, on monomial integral moments.

    Identify ``theta`` by demanding the model reproduce the data's first ``m``
    integral moments, ``int t^k f(t; theta) dt = int t^k y dt`` for
    ``k = 0 .. m-1`` (``m`` defaults to the parameter count), solved as least
    squares over ``theta``. Like EAC/LSI it matches integral functionals of the
    model to the data, so noise averages out, but its test functions are
    monomials, which form an ill-conditioned Hilbert-like moment system. That
    is the conditioning pathology LSI removes by projecting onto an orthogonal
    Legendre basis, which makes this the fair foil for the question of what the
    reconditioning buys, and the deterministic-fit cousin of Pearson's method
    of moments (1894) and Hansen's GMM (1982). ``t`` is normalised to
    ``[0, 1]`` internally so the high-order moments do not overflow.
    """
    from scipy.optimize import least_squares
    from scipy.integrate import simpson

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(p0) if n_moments is None else int(n_moments)
    span = float(t[-1] - t[0]) or 1.0
    tn = (t - t[0]) / span
    weights = [tn ** k for k in range(m)]
    data_moments = np.array([simpson(y=y * w, x=t) for w in weights])

    def resid(p):
        fv = np.asarray(func(t, *p), dtype=float)
        model_moments = np.array([simpson(y=fv * w, x=t) for w in weights])
        return model_moments - data_moments

    kw = {"max_nfev": maxfev}
    if bounds is not None:
        kw["bounds"] = bounds
    sol = least_squares(resid, np.asarray(p0, dtype=float), **kw)
    return np.asarray(sol.x, dtype=float)


# the established real-time / streaming baselines
class RLSPredictor:
    """Recursive Least Squares one-step predictor on an AR(``order``) model.

    The classical online system-identification and adaptive-filtering
    algorithm. It tracks the signal's linear predictor coefficients online
    under a forgetting factor ``lam``, so it adapts to drift, and yields
    one-step predictions. Being a black-box AR model it recovers no physical
    parameters, which makes it the streaming counterpart of the MLP baseline.
    """

    def __init__(self, order=2, lam=0.99, delta=100.0):
        self.order = int(order)
        self.lam = float(lam)
        self.w = np.zeros(self.order)
        self.P = np.eye(self.order) * float(delta)
        self.hist: list[float] = []
        self.last_pred_ = float("nan")

    def update(self, y):
        y = float(y)
        if len(self.hist) >= self.order:
            xv = np.array(self.hist[-self.order:][::-1])
            self.last_pred_ = float(self.w @ xv)
            err = y - self.last_pred_
            Px = self.P @ xv
            k = Px / (self.lam + xv @ Px)
            self.w = self.w + k * err
            self.P = (self.P - np.outer(k, Px)) / self.lam
        self.hist.append(y)
        return self

    def predict_next(self):
        if len(self.hist) >= self.order:
            xv = np.array(self.hist[-self.order:][::-1])
            return float(self.w @ xv)
        return float(self.hist[-1]) if self.hist else 0.0


class EKFParam:
    """Extended Kalman Filter that estimates the parameters of a known
    nonlinear model online.

    The textbook method for online nonlinear parameter estimation: the
    parameters are a random-walk state, the measurement is ``y = f(t; p)``, and
    the EKF linearizes ``f`` about the current estimate through its
    parameter-Jacobian, compiled once with SymPy. It is the same-job baseline
    for dtfit's streaming equal-areas and Legendre filters, since both track
    the model parameters online; what differs is the measurement, a pointwise
    value here against an integrated area or spectrum for dtfit.
    """

    def __init__(self, expr, var, p0, *, q=1e-4, r=1.0, p_init=1.0):
        import sympy as sp
        t = sp.Symbol(var)
        model = sp.sympify(expr)
        self.params = sorted((s for s in model.free_symbols if s != t), key=str)
        n = len(self.params)
        self._f = sp.lambdify([t, *self.params], model, "numpy")
        self._jac = [sp.lambdify([t, *self.params], sp.diff(model, p), "numpy")
                     for p in self.params]
        self.p = np.asarray(p0, dtype=float)
        self.P = np.eye(n) * float(p_init)
        self.Q = np.eye(n) * float(q)
        self.R = float(r)
        self.last_residual_ = float("nan")

    def update(self, t, y):
        n = self.p.size
        # predict (random-walk dynamics): state unchanged, covariance grows
        self.P = self.P + self.Q
        yhat = float(self._f(t, *self.p))
        H = np.array([float(jk(t, *self.p)) for jk in self._jac])
        if not (np.isfinite(yhat) and np.all(np.isfinite(H))):
            return self
        S = float(H @ self.P @ H + self.R)
        if S <= 0:
            return self
        K = self.P @ H / S
        innov = float(y) - yhat
        self.p = self.p + K * innov
        self.P = (np.eye(n) - np.outer(K, H)) @ self.P
        self.last_residual_ = innov
        return self

    @property
    def params_(self):
        return {str(s): float(v) for s, v in zip(self.params, self.p)}

    def predict(self, x):
        return self._f(np.asarray(x, dtype=float), *self.p)


class CTEKFGyro:
    """Gyro-aided coordinated-turn EKF: the fair GPS+IMU recursive baseline.

    Planar state ``[x, vx, y, vy, omega]`` propagated by the coordinated-turn
    motion model, in which the velocity vector rotates at turn-rate ``omega``;
    the vertical channel ``z`` is a constant-acceleration sub-filter. GPS
    supplies the position ``(x, y, z)`` and the gyro measures ``omega``
    directly, so the filter fuses the same information as the windowed gyro
    dead-reckoning fit but recursively, and it is the EKF a tracking
    practitioner actually deploys. The transition is nonlinear in ``omega``
    because the rotation depends on the state, hence an EKF; its Jacobian comes
    from finite differences for robustness. It exposes the ``update``,
    ``forecast``, ``inflate`` and ``last_residuals_`` surface of
    :class:`KalmanCA`, so one adaptive maneuver detector drives both.
    """

    def __init__(self, dt=0.1, r_gps=2.25, r_gyro=9e-4, q_acc=3.0, q_w=0.8,
                 q_z=5e-2, r_z=2.25):
        self.dt = float(dt)
        self.r_gps = float(r_gps)
        self.r_gyro = float(r_gyro)
        self.x = np.zeros(5)
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1.0])
        qb = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * float(q_acc)
        Q = np.zeros((5, 5))
        Q[0:2, 0:2] = qb
        Q[2:4, 2:4] = qb
        Q[4, 4] = float(q_w) * dt
        self.Q = Q
        self._z = KalmanCA(dim=1, dt=dt, q=q_z, r=r_z)   # vertical CA sub-filter
        self._init = False
        self.last_residuals_ = np.zeros(3)
        self.last_nis_ = 0.0

    def _prop(self, s, dt=None):
        """Coordinated-turn propagation of ``[x, vx, y, vy, omega]`` over ``dt``."""
        dt = self.dt if dt is None else dt
        x, vx, y, vy, w = s
        th = w * dt
        if abs(w) < 1e-6:                                # constant-velocity limit
            return np.array([x + vx * dt, vx, y + vy * dt, vy, w])
        s_, c_ = np.sin(th), np.cos(th)
        a, b = s_ / w, (1.0 - c_) / w
        return np.array([x + a * vx - b * vy, c_ * vx - s_ * vy,
                         y + b * vx + a * vy, s_ * vx + c_ * vy, w])

    def _jac(self, s):
        """Finite-difference Jacobian d(prop)/ds at ``s`` (5x5)."""
        F = np.zeros((5, 5))
        f0 = self._prop(s)
        for j in range(5):
            sp_ = s.copy()
            h = 1e-6 * max(1.0, abs(s[j]))
            sp_[j] += h
            F[:, j] = (self._prop(sp_) - f0) / h
        return F

    def update(self, fix, omega):
        """Ingest a GPS fix ``[x, y, z]`` and a gyro yaw-rate ``omega``."""
        fix = np.asarray(fix, dtype=float)
        zpos = self._z.update(fix[2:3])[0]
        if not self._init:
            self.x = np.array([fix[0], 0.0, fix[1], 0.0, float(omega)])
            self._init = True
            self.last_residuals_ = np.zeros(3)
            return np.array([self.x[0], self.x[2], zpos])
        F = self._jac(self.x)
        xp = self._prop(self.x)
        Pp = F @ self.P @ F.T + self.Q
        # measure (x, y, omega): GPS position + direct gyro yaw-rate
        H = np.array([[1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0], [0, 0, 0, 0, 1.0]])
        R = np.diag([self.r_gps, self.r_gps, self.r_gyro])
        z = np.array([fix[0], fix[1], float(omega)])
        y = z - H @ xp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        self.x = xp + K @ y
        self.P = (np.eye(5) - K @ H) @ Pp
        # position innovations (x, y, z) for the shared maneuver detector
        zres = float(fix[2] - zpos)
        self.last_residuals_ = np.array([y[0], y[1], zres])
        self.last_nis_ = float(y @ np.linalg.solve(S, y))
        return np.array([self.x[0], self.x[2], zpos])

    def coast(self, omega):
        """Time-update on the gyro with no GPS, dead-reckoning one step through
        a dropout. The measured yaw-rate keeps steering the velocity and the
        vertical CA coasts on its own dynamics, which is the IMU-aided coasting
        a real INS/GPS rig does in a tunnel and is far better than holding
        position. Returns the new ``[x, y, z]``."""
        if not self._init:
            return self.position()
        self.x[4] = float(omega)               # gyro pins the turn-rate
        F = self._jac(self.x)
        self.x = self._prop(self.x)
        self.P = F @ self.P @ F.T + self.Q
        self._z.x[0] = self._z.F @ self._z.x[0]            # CA time-update of z
        self._z.P[0] = self._z.F @ self._z.P[0] @ self._z.F.T + self._z.Q
        return self.position()

    def inflate(self, factor):
        self.P = self.P * float(factor)
        self._z.inflate(factor)

    def position(self):
        return np.array([self.x[0], self.x[2], self._z.position()[0]])

    def forecast(self, horizon):
        """Roll the CT state (and the vertical CA) forward ``horizon`` steps."""
        s = self.x.copy()
        zf = self._z.forecast(horizon)
        out = np.zeros((horizon, 3))
        for k in range(horizon):
            s = self._prop(s)
            out[k] = [s[0], s[2], zf[k, 0]]
        return out
