"""Map-reduce / partitioned LSI and EAC.

The empirical LSI spectrum coefficient is an integral ``∫ y·φ_j dx`` and an
EAC window area is an integral ``∫ y dx``. Integrals are additive over a
partition of the domain, meaning the data-side sufficient statistic can be
accumulated chunk by chunk and summed in an associative reduce. That turns
the batch methods into one-pass, distributable estimators: a stream of
arbitrary length is processed in fixed ``O(order)`` state, and a partitioned
dataset is fitted by reducing per-partition partial statistics with no second
pass over the data.

* :class:`PartitionedLSI` accumulates basis-projection integrals.
* :class:`PartitionedEAC` accumulates per-window areas.
* :class:`PartitionedBatchLSI` does both at once for many channels, fusing
  the volume partition with a GEMM over the channel axis.

All three need the global domain fixed up front, since every chunk has to
project onto the same basis; pass it to the constructor. Each class states
how exact its own reduce is.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import sympy as sp
from scipy.integrate import simpson
from scipy.optimize import least_squares

from dtfit.types import FittingResult, InitialGuess
from dtfit._core._backend import Backend, resolve_backend
from dtfit._core._spectral import make_basis, solve_spectral
from dtfit.methods._common import model_params, _covariance


class PartitionedLSI:
    """Streaming / distributed LSI via an additive basis-projection reduce.

    The reduce is exact relative to its own trapezoid projection. Parity with
    :func:`dtfit.fit_lsi`, which adds Savitzky-Golay pre-filtering,
    auto/robust order selection and a least-squares Legendre projection, is
    asymptotic on dense uniform data rather than bit-exact.

    Usage::

        acc = PartitionedLSI("a*exp(b*t)", "t", domain=(0, 10), order=6)
        for x_chunk, y_chunk in stream:      # one pass, fixed memory
            acc.update(x_chunk, y_chunk)
        result = acc.fit(p0=[1.0, 1.0])      # FittingResult

    Workers can each build their own accumulator and combine them with
    :meth:`merge`, the reduce step.
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        order: int = 6,
        basis: str = "legendre",
    ) -> None:
        self.expr = expr
        self.var = var
        self.basis = make_basis(basis, order, domain)
        self._s = np.zeros(self.basis.n_coef)  # accumulated ∫ y·φ_j integrals
        self._last: tuple[float, float] | None = None  # carried boundary sample
        self.n_samples = 0

    def update(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> "PartitionedLSI":
        """Fold one chunk's partial projection integrals into the accumulator.

        Consecutive ``update`` calls are exactly additive, equal to a single
        whole-domain projection, because the previous chunk's last sample is
        carried into the next one. The trapezoid rule then never drops the
        interval connecting two disjoint chunks. Feed chunks in domain order.
        """
        x = np.atleast_1d(np.asarray(x_chunk, dtype=float))
        y = np.atleast_1d(np.asarray(y_chunk, dtype=float))
        # count from the coerced array: a list/tuple chunk has no ``.shape``
        # and a 0-d scalar chunk stands for one sample. Must happen before
        # the boundary carry below extends x.
        n_orig = x.shape[0]
        if self._last is not None and x.size:
            x = np.concatenate([[self._last[0]], x])
            y = np.concatenate([[self._last[1]], y])
        if x.size >= 2:
            self._s += self.basis.project_integral(x, y)
            self.n_samples += n_orig
            self._last = (float(x[-1]), float(y[-1]))
        return self

    def merge(self, other: "PartitionedLSI") -> "PartitionedLSI":
        """Associative reduce: combine another accumulator's partial sums.

        Exact when the partitions share boundary samples, that is when each
        partition includes the sample where the next one begins and every
        connecting interval belongs to exactly one partition. Otherwise it is
        additive only up to one missing trapezoid interval per boundary.
        """
        self._s += other._s
        self.n_samples += other.n_samples
        return self

    def spectrum(self) -> np.ndarray:
        """The reduced empirical spectrum (whole-domain coefficients)."""
        return self.basis.integral_to_spectrum(self._s)

    def fit(self, *, p0: InitialGuess = None) -> FittingResult:
        """Solve the LSI spectral match against the accumulated spectrum."""
        guess = None if p0 is None else np.asarray(p0, float)
        return solve_spectral(self.expr, self.var, self.basis, self.spectrum(), p0=guess)


class PartitionedEAC:
    """Streaming / distributed EAC via additive per-window area accumulation.

    The domain is split into ``n_windows`` value-uniform windows. Each chunk
    adds the exact windowed integral of its piecewise-linear interpolant into
    the windows it overlaps, splitting any interval that straddles a window
    edge at that edge so no area is lost. The model is then matched to the
    reduced data areas with the same overdetermined least-squares solve as
    batch EAC, using a Simpson quadrature of the model over each window.
    Simpson converges to the true window integral; a single midpoint sample
    would bias on curved windows.

    The windows are placed by domain value, not by sample index as in batch
    :func:`dtfit.fit_eac`, making this a streaming approximation of the EAC
    criterion. It recovers the same parameters as batch EAC asymptotically on
    dense, roughly uniform data, not bit for bit.
    """

    #: model-side quadrature nodes per window (odd -> Simpson exact for cubics)
    _Q = 9

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        n_windows: int = 8,
    ) -> None:
        self.expr = expr
        self.var = var
        self.x0, self.xn = float(domain[0]), float(domain[1])
        self.m = int(n_windows)
        self.edges = np.linspace(self.x0, self.xn, self.m + 1)
        self._areas = np.zeros(self.m)
        self._first: tuple[float, float] | None = None  # accumulator's first sample
        self._last: tuple[float, float] | None = None    # ...and its last sample
        self.n_samples = 0
        # (m, _Q) grid of model-evaluation nodes, one Simpson panel per window.
        frac = np.linspace(0.0, 1.0, self._Q)
        self._qnodes = self.edges[:-1, None] + frac[None, :] * np.diff(self.edges)[:, None]

    def _accumulate(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add the exact windowed integral of the piecewise-linear
        interpolant of ``(x, y)`` into ``self._areas``, fully vectorized.

        Interior window edges falling inside the chunk are spliced into the
        grid as extra, linearly-interpolated nodes, leaving every resulting
        segment entirely within one window; each segment's trapezoid area is
        then binned to its window in a single ``np.add.at``. Splitting at the
        edges is what stops an interval straddling a window boundary from
        being dropped, and it makes the per-window areas sum to the exact
        trapezoid integral partitioned at the true edges.
        """
        edges = self.edges
        ie = edges[1:-1]
        ie = ie[(ie > x[0]) & (ie < x[-1])]  # interior edges inside this chunk
        if ie.size:
            ye = np.interp(ie, x, y)
            xa = np.concatenate([x, ie])
            ya = np.concatenate([y, ye])
            order = np.argsort(xa, kind="mergesort")  # stable; keeps edge-at-sample ties
            xa, ya = xa[order], ya[order]
        else:
            xa, ya = x, y
        seg = 0.5 * (ya[:-1] + ya[1:]) * np.diff(xa)  # trapezoid area per segment
        mid = 0.5 * (xa[:-1] + xa[1:])
        wk = np.clip(np.searchsorted(edges, mid, side="right") - 1, 0, self.m - 1)
        np.add.at(self._areas, wk, seg)

    def _add_interval(self, xa: float, ya: float, xb: float, yb: float) -> None:
        """Add the single connecting trapezoid interval ``[xa, xb]``, the
        merge twin of :meth:`update`'s boundary carry."""
        if xb > xa:
            self._accumulate(np.array([xa, xb]), np.array([ya, yb]))

    def update(self, x_chunk: np.ndarray, y_chunk: np.ndarray) -> "PartitionedEAC":
        """Fold one chunk's window areas into the accumulator.

        As in :class:`PartitionedLSI`, the previous chunk's last sample is
        carried into this one, keeping the connecting interval in the
        integral. The very first sample seen is kept too, for :meth:`merge`
        to stitch a neighbouring partition against. Feed chunks in domain
        order.
        """
        x = np.asarray(x_chunk, dtype=float)
        y = np.asarray(y_chunk, dtype=float)
        n_orig = x.size
        if self._first is None and x.size:
            self._first = (float(x[0]), float(y[0]))
        if self._last is not None and x.size:
            x = np.concatenate([[self._last[0]], x])
            y = np.concatenate([[self._last[1]], y])
        if x.size:
            self._last = (float(x[-1]), float(y[-1]))
        if x.size >= 2:
            self._accumulate(x, y)
        self.n_samples += n_orig
        return self

    def merge(self, other: "PartitionedEAC") -> "PartitionedEAC":
        """Associative reduce, made exact by stitching the partition boundary.

        Summing ``_areas`` alone would drop the trapezoid interval connecting
        one partition's last sample to the next's first sample, because
        neither accumulator ever integrated it, and the reduce would come out
        order-dependent. That interval is added explicitly, the reduce twin
        of :meth:`update`'s boundary carry. Merging disjoint domain-ordered
        partitions then equals processing them in one pass. Partitions must
        be disjoint; the order between the two is inferred.
        """
        left, right = self, other
        if (self._first is not None and other._last is not None
                and other._last[0] <= self._first[0]):
            left, right = other, self  # `other` lies before `self` on the domain
        if left._last is not None and right._first is not None:
            self._add_interval(left._last[0], left._last[1],
                               right._first[0], right._first[1])
        self._areas += other._areas
        self.n_samples += other.n_samples
        # extend this accumulator's boundary span to cover both partitions
        self._first = left._first if left._first is not None else self._first
        self._last = right._last if right._last is not None else self._last
        return self

    def fit(self, *, p0: InitialGuess = None) -> FittingResult:
        """Match the model's window areas to the reduced data areas."""
        t = sp.Symbol(self.var)
        f_sym = cast(sp.Expr, sp.sympify(self.expr))
        params = model_params(f_sym, t)
        n = len(params)
        f_func = sp.lambdify((t, *params), f_sym, "numpy")
        nodes = self._qnodes  # (m, _Q)

        def model_areas(c: np.ndarray) -> np.ndarray:
            # Simpson quadrature of the model over each window, to match the
            # trapezoid-integrated data areas.
            fv = np.asarray(f_func(nodes, *c), dtype=float)
            if fv.shape != nodes.shape:
                fv = np.broadcast_to(fv, nodes.shape)
            return simpson(fv, x=nodes, axis=1)

        def residual(c: np.ndarray) -> np.ndarray:
            return model_areas(c) - self._areas

        guess = np.ones(n) if p0 is None else np.asarray(p0, float)
        sol = least_squares(residual, guess, method="lm")
        coeffs = np.asarray(sol.x, dtype=np.float64)
        cov = _covariance(sol.jac, residual(coeffs), n)
        return FittingResult(coeffs=coeffs, cov=cov,
                             expr=self.expr, var=self.var,
                             names=tuple(str(p) for p in params),
                             converged=bool(sol.success), message=str(sol.message),
                             x_range=(self.x0, self.xn))


class PartitionedBatchLSI:
    """Fused map-reduce and GEMM-batched LSI for many channels in one pass.

    Combines the two big-data levers that :class:`PartitionedLSI` and
    :func:`dtfit.project_spectra` provide separately:

    * the volume partition of :class:`PartitionedLSI`, reducing a stream of
      arbitrary length in fixed ``O(channels x order)`` memory;
    * the channel batch of :func:`dtfit.project_spectra`, projecting the
      ``B`` channels that share the sampling grid in one GEMM
      ``S = Dᵀ·(w⊙Y)`` per chunk, through a pluggable array backend
      (NumPy/BLAS, or cupy/torch on a GPU).

    Fusing them stays exact because the projection is linear across channels
    and additive over the domain. Each chunk's ``(B, n_coef)`` partial
    integrals are folded into the accumulator, :meth:`merge` reduces
    accumulators across workers and partitions, and :meth:`fit` solves each
    channel's small spectral match.

    Usage::

        acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=(0, 10),
                                  n_channels=512, order=6, backend="auto")
        for x_chunk, Y_chunk in stream:        # Y_chunk is (n_chunk, 512)
            acc.update(x_chunk, Y_chunk)
        results = acc.fit(p0=[1.0, 1.0])       # list[FittingResult], one/channel
    """

    def __init__(
        self,
        expr: str,
        var: str,
        *,
        domain: tuple[float, float],
        n_channels: int,
        order: int = 6,
        basis: str = "legendre",
        backend: str | Backend = "auto",
        **basis_kwargs: object,
    ) -> None:
        self.expr = expr
        self.var = var
        self.basis = make_basis(basis, order, domain, **basis_kwargs)
        self.backend = (
            backend if isinstance(backend, Backend) else resolve_backend(backend)
        )
        self.n_channels = int(n_channels)
        # accumulated raw integrals s_{c,j} = ∫ y_c·φ_j  (B, n_coef)
        self._s = np.zeros((self.n_channels, self.basis.n_coef))
        self._last_x: float | None = None       # carried boundary sample (time)
        self._last_y: np.ndarray | None = None   # carried boundary row (B,)
        self.n_samples = 0

    def update(self, x_chunk: np.ndarray, Y_chunk: np.ndarray) -> "PartitionedBatchLSI":
        """Fold one chunk's ``B``-channel partial projection into the accumulator.

        ``Y_chunk`` is ``(n_chunk, B)``. As in :class:`PartitionedLSI`, the
        previous chunk's last row is carried into this one, keeping the
        connecting interval in the integral. Feed chunks in domain order.
        """
        x = np.atleast_1d(np.asarray(x_chunk, dtype=float))
        Y = np.atleast_1d(np.asarray(Y_chunk))
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[1] != self.n_channels:
            raise ValueError(
                f"Y_chunk has {Y.shape[1]} channels but accumulator holds "
                f"{self.n_channels}."
            )
        # count from the coerced array: a list/tuple chunk has no ``.shape``
        # and a 0-d scalar chunk stands for one sample. Must happen before
        # the boundary carry below extends x.
        n_orig = x.shape[0]
        if self._last_x is not None and self._last_y is not None and x.size:
            x = np.concatenate([[self._last_x], x])
            Y = np.concatenate([self._last_y[None, :], Y], axis=0)
        if x.size >= 2:
            self._s += self.basis.project_integral_batched(x, Y, self.backend)
            self.n_samples += n_orig
            self._last_x = float(x[-1])
            self._last_y = np.asarray(Y[-1], dtype=float)
        return self

    def merge(self, other: "PartitionedBatchLSI") -> "PartitionedBatchLSI":
        """Associative reduce: combine another worker's partial integrals."""
        self._s += other._s
        self.n_samples += other.n_samples
        return self

    def spectra(self) -> np.ndarray:
        """Reduced per-channel empirical spectra, shape ``(B, n_coef)``."""
        return self.basis.integral_to_spectrum(self._s)

    def fit(
        self,
        *,
        p0: InitialGuess = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> list[FittingResult]:
        """Solve each channel's LSI spectral match against its reduced spectrum."""
        specs = self.spectra()
        p0a = None if p0 is None else np.asarray(p0, float)
        return [
            solve_spectral(self.expr, self.var, self.basis, specs[i], p0=p0a, bounds=bounds)
            for i in range(specs.shape[0])
        ]
