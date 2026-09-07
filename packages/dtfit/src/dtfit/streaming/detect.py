"""Change detection on a sequence of whitened innovation vectors, shared by
the block stream and the recursive filters."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2


class DriftDetector:
    """Two tests on successive whitened innovations ``e`` of dimension
    ``dim``: a jump test on the innovation energy ``e @ e`` against an
    exponentially weighted baseline of previous energies, and a two-sided
    CUSUM on the first component standardized by its own baseline.

    Args:
        dim: innovation length, at least 1.
        alpha: significance of the jump test; the energy ratio threshold is
            ``1.6 * chi2.ppf(1 - alpha, dim) / dim``.
        cusum_k: CUSUM slack in standard deviations, non-negative.
        cusum_h: CUSUM decision threshold in accumulated standard
            deviations, positive.
        ewma: forgetting weight of the baselines in ``(0, 1]``.
        warmup: number of innovations that only build the baselines after
            construction and after every detection; the baselines are
            bias-corrected exponential averages, so the first tests after
            the warmup use the average of the warmup innovations, not a
            fraction of it.

    Attributes:
        n_tests_: innovations seen since the last reset.
        n_drifts_: detections since construction.
        flag_: whether the last ``update`` detected a change.
        last_direction_: ``1`` for an upward, ``-1`` for a downward change
            of the first component at the last detection, ``0`` before any.
        threshold: the energy ratio threshold.

    Raises:
        ValueError: ``dim < 1``, ``alpha`` outside ``(0, 1)``,
            ``cusum_k < 0``, ``cusum_h <= 0``, ``ewma`` outside ``(0, 1]``
            or ``warmup < 0``.
    """

    def __init__(
        self,
        dim: int,
        *,
        alpha: float = 0.001,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        ewma: float = 0.15,
        warmup: int = 20,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be at least 1, got {dim}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if cusum_k < 0.0 or cusum_h <= 0.0:
            raise ValueError("cusum_k must be >= 0 and cusum_h > 0")
        if not 0.0 < ewma <= 1.0:
            raise ValueError(f"ewma must be in (0, 1], got {ewma}")
        if warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {warmup}")
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.cusum_k = float(cusum_k)
        self.cusum_h = float(cusum_h)
        self.ewma = float(ewma)
        self.warmup = int(warmup)
        self.threshold = 1.6 * float(chi2.ppf(1.0 - alpha, df=dim)) / dim
        self.n_drifts_ = 0
        self.last_direction_ = 0
        self.reset()

    def reset(self) -> None:
        """Forget the baselines and the CUSUM sums; the next ``warmup``
        innovations build them again."""
        self._s_scale = 0.0
        self._e0_scale2 = 0.0
        self._g_hi = 0.0
        self._g_lo = 0.0
        self.n_tests_ = 0
        self.flag_ = False

    def update(self, innovation: Any) -> bool:
        """Test one whitened innovation. Returns True on a detection, after
        which the detector has reset itself.

        Raises:
            ValueError: wrong length or a non-finite entry.
        """
        e = np.asarray(innovation, dtype=float).reshape(-1)
        if e.size != self.dim:
            raise ValueError(
                f"innovation has length {e.size}, expected {self.dim}"
            )
        if not np.all(np.isfinite(e)):
            raise ValueError("innovation must be finite")
        seen = self.n_tests_
        self.n_tests_ += 1
        self.flag_ = False
        energy = float(e @ e)
        e0 = float(e[0])
        lam = self.ewma
        corr = 1.0 - (1.0 - lam) ** seen if seen > 0 else 0.0
        s_base = self._s_scale / corr if corr > 0.0 else 0.0
        e0_base = self._e0_scale2 / corr if corr > 0.0 else 0.0
        s_ratio = energy / s_base if s_base > 0.0 else 0.0
        z0 = e0 / np.sqrt(e0_base) if e0_base > 0.0 else 0.0
        self._s_scale = (1.0 - lam) * self._s_scale + lam * energy
        self._e0_scale2 = (1.0 - lam) * self._e0_scale2 + lam * e0 * e0
        if self.n_tests_ <= self.warmup:
            return False
        self._g_hi = max(0.0, self._g_hi + z0 - self.cusum_k)
        self._g_lo = max(0.0, self._g_lo - z0 - self.cusum_k)
        jump = s_ratio > self.threshold
        up = self._g_hi > self.cusum_h
        down = self._g_lo > self.cusum_h
        if not (jump or up or down):
            return False
        self.n_drifts_ += 1
        self.last_direction_ = 1 if (up or (not down and z0 >= 0.0)) else -1
        self.reset()
        self.flag_ = True
        return True

    def state(self) -> dict[str, Any]:
        """The complete detector state as plain Python numbers."""
        return {
            "s_scale": self._s_scale,
            "e0_scale2": self._e0_scale2,
            "g_hi": self._g_hi,
            "g_lo": self._g_lo,
            "n_tests": self.n_tests_,
            "n_drifts": self.n_drifts_,
            "flag": self.flag_,
            "last_direction": self.last_direction_,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Load a state produced by :meth:`state` on a detector built with
        the same parameters."""
        self._s_scale = float(state["s_scale"])
        self._e0_scale2 = float(state["e0_scale2"])
        self._g_hi = float(state["g_hi"])
        self._g_lo = float(state["g_lo"])
        self.n_tests_ = int(state["n_tests"])
        self.n_drifts_ = int(state["n_drifts"])
        self.flag_ = bool(state["flag"])
        self.last_direction_ = int(state["last_direction"])
