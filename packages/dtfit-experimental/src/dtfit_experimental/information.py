"""Information-form (inverse-covariance) recursive estimator.

The covariance-form Kalman update the streaming filters run maintains ``P``
and, every step, inverts the innovation covariance ``S``, whose dimension is
the measurement size. The information form instead maintains the inverse
``Y = P^-1``, the information matrix, alongside ``yv = P^-1 p``, the
information vector. Three properties flip, all of them relevant to the
embedded / sensor-fusion target:

* the measurement update is purely additive, ``Y += Hᵀ R⁻¹ H`` and
  ``yv += Hᵀ R⁻¹ z``, so absorbing a measurement takes no matrix inverse at
  all; independent estimators fuse by adding information in an associative,
  order-independent reduce;
* the matrix you do invert is the smaller one. A readout solves the
  ``n_params × n_params`` system ``Y θ = yv`` once, against an ``m × m``
  innovation covariance inverted at every step: a win whenever the measurement
  dimension ``m`` exceeds the state dimension ``n``, the common LSI case of
  order+1 coefficients against 2-3 parameters;
* it is friendlier in fixed point. Accumulating information never meets the
  covariance-collapse conditioning that ``(I - K H) P`` does.

What lives here is the linear-Gaussian primitive: recursive least squares in
information form with an optional forgetting factor, the on-MCU building block.
It is the same update dtfit's nonlinear filters would take were they run in
information form rather than covariance form, offered standalone and
fusion-oriented. The covariance-form ``EACFilter`` / ``LSIFilter`` run the
covariance update directly and do not touch it; it stays in
``dtfit-experimental`` until a domain study exercises it.
"""

from __future__ import annotations

import numpy as np


class InformationFilter:
    """Recursive linear-Gaussian estimator in information (inverse-covariance) form.

    Estimates ``theta`` in the linear measurement model ``z = h . theta + noise``
    (noise variance ``r``) by accumulating the information matrix ``Y`` and vector
    ``yv``. Absorbing a measurement is an addition (no inverse); the estimate is
    read out by one small solve.

    Usage::

        from dtfit_experimental import InformationFilter
        f = InformationFilter(n_params=2)
        for h, z in stream:                 # h: (2,) row, z: scalar
            f.partial_fit(h, z, r=0.04)
        theta = f.theta_                    # current estimate
        cov = f.cov_                        # its covariance (P = Y^-1)

    Two independent estimators, one per sensor or per partition, fuse by adding
    their information, in whichever order they arrive::

        fused = a.fuse(b)                   # associative & commutative

    Args:
        n_params: State dimension.
        prior_precision: Diagonal of the initial information matrix ``Y0 =
            prior_precision * I``, a weak prior. ``0`` is the uninformative
            start, at the cost of leaving ``Y`` singular until enough
            measurements arrive.
        forgetting: Exponential forgetting factor in ``(0, 1]``, where ``1`` is
            no forgetting. Each step down-weights the accumulated information
            by this factor before adding the new measurement, so the estimator
            tracks slowly-varying parameters.
    """

    def __init__(
        self,
        n_params: int,
        *,
        prior_precision: float = 1e-6,
        forgetting: float = 1.0,
    ) -> None:
        self.n = int(n_params)
        if self.n < 1:
            raise ValueError("n_params must be >= 1")
        if not (0.0 < forgetting <= 1.0):
            raise ValueError("forgetting must be in (0, 1]")
        self.forgetting = float(forgetting)
        self._prior = float(prior_precision)
        # Bare ``np.ndarray`` on purpose. The 3.10 numpy stubs would otherwise
        # pin these to a 1-D ``tuple[int]`` shape, and the widening assignment
        # ``self.yv = self.yv + other.yv`` in ``fuse`` then fails mypy under
        # --python-version 3.10.
        self.Y: np.ndarray = np.eye(self.n) * self._prior   # information matrix (P^-1)
        self.yv: np.ndarray = np.zeros(self.n)              # information vector (P^-1 p)
        self.n_updates = 0

    def partial_fit(self, h, z, r: float = 1.0) -> "InformationFilter":
        """Absorb one measurement ``z = h . theta + noise`` (noise variance ``r``).

        ``h`` is a length-``n`` row for a scalar ``z``, or an ``(m, n)`` matrix
        for a vector measurement ``z`` of length ``m``; ``r`` is then either a
        scalar or a length-``m`` per-component variance. The update is
        additive, ``Y += Hᵀ R⁻¹ H`` and ``yv += Hᵀ R⁻¹ z``, with no inverse.
        """
        H = np.atleast_2d(np.asarray(h, dtype=float))
        zz = np.atleast_1d(np.asarray(z, dtype=float))
        if H.shape[1] != self.n:
            raise ValueError(
                f"h has {H.shape[1]} columns but the state has {self.n} params."
            )
        if H.shape[0] != zz.size:
            raise ValueError(
                f"got {H.shape[0]} measurement rows but {zz.size} values."
            )
        rr = np.atleast_1d(np.asarray(r, dtype=float))
        if rr.size == 1:
            rr = np.full(zz.size, float(rr.reshape(-1)[0]))
        r_inv = 1.0 / rr
        if self.forgetting < 1.0:
            self.Y *= self.forgetting
            self.yv *= self.forgetting
        # Hᵀ R⁻¹ H and Hᵀ R⁻¹ z, no inverse of an m x m innovation covariance.
        HtRinv = H.T * r_inv                     # (n, m)
        self.Y += HtRinv @ H
        self.yv += HtRinv @ zz
        self.n_updates += 1
        return self

    @property
    def cov_(self) -> np.ndarray:
        """Parameter covariance ``P = Y^-1`` (inverts the small ``n x n`` matrix)."""
        return np.linalg.inv(self.Y)

    @property
    def theta_(self) -> np.ndarray:
        """Current estimate, from the single small solve ``Y theta = yv``."""
        return np.linalg.solve(self.Y, self.yv)

    @property
    def p(self) -> np.ndarray:
        """Alias of :attr:`theta_` (covariance-filter naming)."""
        return self.theta_

    @property
    def P(self) -> np.ndarray:
        """Alias of :attr:`cov_` (covariance-filter naming)."""
        return self.cov_

    def fuse(self, other: "InformationFilter") -> "InformationFilter":
        """Combine another estimator's information into this one (in place).

        Information is additive. Fusing independent estimators is then an
        associative, commutative reduce, the property behind the information
        form's standing as the natural sensor-fusion / map-reduce state.
        Reordering the sum reproduces the single-pass estimate to
        floating-point rounding, not bit for bit. The shared prior is
        subtracted once here so it is not double-counted. Returns ``self``.

        Both operands are assumed to have run with ``forgetting == 1``, the
        usual partition/fusion case. Under forgetting < 1 the retained prior
        has been decayed per step and subtracting the full initial prior
        over-removes it, so fuse un-forgotten estimators.
        """
        if other.n != self.n:
            raise ValueError("cannot fuse filters with different state sizes")
        self.Y = self.Y + other.Y - np.eye(self.n) * self._prior
        self.yv = self.yv + other.yv
        self.n_updates += other.n_updates
        return self
