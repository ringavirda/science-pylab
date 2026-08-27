"""Streaming counterpart of :func:`fit_stochastic`'s second-order stage.

:func:`~dtfit.stochastic.fit_stochastic` characterizes a whole record in one
batch pass. A series arriving as a stream instead needs its second-order
structure tracked as it goes, and the moment that structure shifts flagged.
:class:`StochasticFilter` keeps exponentially-weighted autocovariances of the
level and of the absolute deviations, updated in O(K) per input, and reads the
parameters off them in closed form:

* persistence (AR(1) ``phi``) and volatility persistence by the equal-areas
  criterion. Over two consecutive equal-width windows an exponentially
  decaying ACF has area ratio ``a2/a1 = exp(-g*h)``. That pins the decay rate
  ``g`` independently of the amplitude. It is the per-input form of
  ``fit_eac("exp(-g*k)")`` and tracks the batch fit to about 1e-2.
* the cycle from the AR(2) characteristic roots of the running
  autocovariances, i.e. the lag-1 and lag-2 quantities.

Nothing here optimizes per sample or re-runs a batch fit; time (~3 us/sample)
and memory both stay flat. A two-timescale fused statistic, the streaming
analogue of the fused chi-square detector, flags a structural break such as a
persistence jump or a volatility regime switch, once per change.

Scope is the second-order stage alone. Long memory (the spectral Hurst) and
the unit-root gate need a periodogram or a regression over the whole record,
so they are batch quantities and are not tracked here. The filter covers
persistence, cycle and volatility, the structure that can be maintained in
O(1).
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["StochasticFilter"]


class StochasticFilter:
    """Online second-order characterizer and regime-change detector.

    Feed samples one at a time with :meth:`update`. The running
    characterization (AR(1) phi, cycle, volatility persistence, each a
    closed-form estimate from the running autocovariances) is on
    :attr:`params_` and :meth:`snapshot`; a short forecast comes from
    :meth:`predict`; detected structural breaks are counted and timed on
    :attr:`n_flags_`, :attr:`flag_times_` and :attr:`last_flag_`.

    Args:
        nlags: Autocovariance lags maintained; sets the memory footprint.
        halflife: EWMA half-life in samples of the autocovariance tracker,
            i.e. how fast the characterization adapts.
        warmup: Samples to accumulate before the detector is active.
        settle: Samples after ``warmup`` spent calibrating the detector's
            in-control baseline before it may flag, letting the slow EWMA
            settle.
        z_thresh: Fused-statistic threshold in standard deviations for a flag.
    """

    def __init__(self, nlags: int = 24, halflife: float = 150.0,
                 warmup: int = 80, settle: int = 500, z_thresh: float = 4.0):
        self.nlags = int(nlags)
        self.alpha = 1.0 - 0.5 ** (1.0 / float(halflife))
        self.warmup = int(warmup)
        self.settle = int(settle)
        self.z_thresh = float(z_thresh)
        # online autocovariances of the level and of |level - mean|
        self._buf: deque[float] = deque(maxlen=self.nlags)
        self._mean: float | None = None
        self._acov = np.zeros(self.nlags + 1)
        self._vacov = np.zeros(self.nlags + 1)
        self._n = 0
        # Change detection: fast and slow EWMAs of the characterizing stats,
        # plus a variance of their gap that freezes while out of control. That
        # is the in-control-baseline control chart; a sustained break cannot
        # inflate its own normalizer.
        self._fast = np.zeros(2)
        self._slow = np.zeros(2)
        self._gvar = np.full(2, 1e-4)
        self._det_init = False
        self._afast = 1.0 - 0.5 ** (1.0 / 40.0)
        self._aslow = 1.0 - 0.5 ** (1.0 / 200.0)
        self._agvar = 1.0 - 0.5 ** (1.0 / 200.0)
        self._alarmed = False
        self.n_flags_ = 0
        # Capped ring of recent change-point times, so memory stays flat;
        # n_flags_ counts every flag but is a single int.
        self.flag_times_: deque[int] = deque(maxlen=512)
        self.last_flag_: int | None = None

    def update(self, x: float) -> "StochasticFilter":
        """Ingest one sample, updating the running autocovariances and the
        change detector."""
        x = float(x)
        a = self.alpha
        mean = x if self._mean is None else self._mean + a * (x - self._mean)
        self._mean = mean
        d = x - mean
        ad = abs(d)
        self._acov[0] += a * (d * d - self._acov[0])
        self._vacov[0] += a * (ad * ad - self._vacov[0])
        # EWMA autocovariance update across the whole lag buffer at once,
        # keeping the per-sample hot path free of a Python-level loop.
        m = len(self._buf)
        if m:
            buf = np.fromiter(self._buf, dtype=float, count=m)   # most-recent-first
            dl = buf - self._mean
            self._acov[1:m + 1] += a * (d * dl - self._acov[1:m + 1])
            self._vacov[1:m + 1] += a * (ad * np.abs(dl) - self._vacov[1:m + 1])
        self._buf.appendleft(x)
        self._n += 1
        self._detect()
        return self

    def partial_fit(self, xs) -> "StochasticFilter":
        """Ingest a batch of samples (house-style alias for a loop of update)."""
        for x in np.asarray(xs, dtype=float).ravel():
            self.update(x)
        return self

    # Closed-form estimators read per input off the running autocovariances.
    def _eac_decay(self, acov: np.ndarray) -> float:
        """Decay persistence by the equal-areas criterion, read per input off
        the running ACF.

        For an exponentially decaying ACF the areas of two consecutive
        equal-width windows stand in the ratio ``exp(-g*h)``. That pins the
        decay rate ``g``, and with it the persistence ``exp(-g)``, without
        reference to the amplitude: the streaming form of
        ``fit_eac("exp(-g*k)")``, integrating the ACF rather than reading a
        single lag of it.

        Only lags clearly above the white-noise band (``~2/sqrt(n_eff)``)
        count as signal. A weakly-persistent stream has a fast-decaying ACF
        whose tail hovers near zero, and a fixed tiny threshold would admit
        that noise into the area integration and leave the estimate jittering.
        Integration is therefore reserved for a slow decay with enough signal
        lags; below that, the lag-1 autocorrelation is the exact AR(1)
        persistence and is quieter than integrating a couple of near-noise
        lags."""
        c0 = acov[0]
        if c0 <= 1e-12:
            return float("nan")
        rho = acov / c0
        nlags = rho.size - 1
        neff = min(float(self._n), 1.0 / self.alpha)
        band = max(2.0 / np.sqrt(max(neff, 1.0)), 0.05)
        cut = nlags
        for k in range(1, nlags + 1):           # integrate the one-sided decay only
            if rho[k] <= band:
                cut = k
                break
        if cut < 6:                             # too few signal lags -> exact lag-1
            return float(np.clip(rho[1], 0.0, 0.999))
        h = cut // 2
        a1 = float(np.trapezoid(rho[:h + 1]))            # area over [0, h]
        a2 = float(np.trapezoid(rho[h:2 * h + 1]))       # area over [h, 2h]
        if a1 <= 0.0 or a2 <= 0.0 or a2 >= a1:
            return float(np.clip(rho[1], 0.0, 0.999))
        g = -np.log(a2 / a1) / h
        return float(np.clip(np.exp(-abs(g)), 0.0, 0.999))

    def _ar2_yule_walker(self) -> tuple[float, float]:
        """AR(2) coefficients ``(phi1, phi2)`` from the lag-1 and lag-2
        autocorrelations by Yule-Walker. The roots are complex, hence
        oscillatory, exactly when ``phi2 < 0`` and the discriminant is
        negative; that is the streaming cycle test."""
        c0 = self._acov[0]
        if c0 <= 1e-12:
            return 0.0, 0.0
        r1, r2 = self._acov[1] / c0, self._acov[2] / c0
        den = 1.0 - r1 * r1
        if abs(den) < 1e-9:
            return float(np.clip(r1, -0.999, 0.999)), 0.0
        return r1 * (1.0 - r2) / den, (r2 - r1 * r1) / den

    def _cycle_period(self) -> float:
        """Dominant cycle period from the AR(2) complex roots (or NaN if the
        lag-1/lag-2 structure is not oscillatory)."""
        phi1, phi2 = self._ar2_yule_walker()
        if phi2 >= 0.0:
            return float("nan")
        r = np.sqrt(-phi2)               # root modulus
        c = phi1 / (2.0 * r)             # cos(angular frequency)
        if abs(c) >= 1.0:
            return float("nan")
        per = float(2.0 * np.pi / np.arccos(c))
        # The resolvable-cycle window, shared with snapshot()'s "cyclical"
        # label: at least ~4 samples per period and at least one full period
        # inside the lag window. Outside it the period is unresolved and comes
        # back NaN, keeping params_ and the regime label in agreement.
        return per if 4.0 <= per <= self.nlags else float("nan")

    def _rho1(self) -> float:
        c0 = self._acov[0]
        return self._acov[1] / c0 if c0 > 1e-12 else 0.0

    def _detect(self) -> None:
        if self._n < self.warmup:
            return
        # Characterizing statistic: the lag-1 autocorrelation for persistence
        # and the log volatility level, fused into one descriptor.
        stat = np.array([self._rho1(), 0.5 * np.log(self._acov[0] + 1e-12)])
        if not self._det_init:
            self._fast[:] = stat
            self._slow[:] = stat
            self._det_init = True
            return
        self._fast += self._afast * (stat - self._fast)
        self._slow += self._aslow * (stat - self._slow)
        gap = self._fast - self._slow
        z2 = float(np.sum(gap ** 2 / (self._gvar + 1e-9)))
        calibrating = self._n < self.warmup + self.settle
        in_control = z2 < (0.5 * self.z_thresh) ** 2
        if calibrating or in_control:
            self._gvar += self._agvar * (gap ** 2 - self._gvar)
        if in_control:
            self._alarmed = False               # re-arm once back in-control
        if calibrating:
            return
        # Rising-edge latch: one flag per change, not one per out-of-control
        # sample.
        if z2 > self.z_thresh ** 2 and not self._alarmed:
            self.n_flags_ += 1
            self.flag_times_.append(self._n)
            self.last_flag_ = self._n
            self._alarmed = True

    @property
    def params_(self) -> dict[str, float]:
        """Current second-order characterization, computed on access from the
        running autocovariances: persistence and volatility by the EAC
        equal-areas criterion, the cycle by the AR(2) roots."""
        return {
            "n": float(self._n),
            "level": float(self._mean) if self._mean is not None else float("nan"),
            "sigma": float(np.sqrt(max(self._acov[0], 0.0))),
            "ar1_phi": self._eac_decay(self._acov),
            "cycle_period": self._cycle_period(),
            "vol_persistence": self._eac_decay(self._vacov),
        }

    def snapshot(self) -> dict[str, object]:
        """:attr:`params_` plus a coarse online ``regime`` label."""
        p = self.params_
        per = p["cycle_period"]
        if p["sigma"] < 1e-9:
            regime = "white noise"
        elif np.isfinite(per):  # _cycle_period already restricts to 4..nlags
            regime = "cyclical"
        elif p["ar1_phi"] > 0.2:
            regime = "mean-reverting"
        elif np.isfinite(p["vol_persistence"]) and p["vol_persistence"] > 0.3:
            regime = "vol-clustering"
        else:
            regime = "white noise"
        return {**p, "regime": regime}

    def predict(self, h: int) -> np.ndarray:
        """Forecast ``h`` steps by AR(1) mean reversion at the current snapshot."""
        p = self.params_
        phi = p["ar1_phi"] if np.isfinite(p["ar1_phi"]) else 0.0
        mu = p["level"]
        last = self._buf[0] if self._buf else mu
        return mu + phi ** np.arange(1, h + 1, dtype=float) * (last - mu)
