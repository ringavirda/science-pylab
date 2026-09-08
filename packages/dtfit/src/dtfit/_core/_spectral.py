"""Shared spectral-match machinery for the LSI adaptations.

The LSI method (:func:`dtfit.fit_lsi`) fits a parameter-nonlinear model by
matching the model's spectrum to the data's on an orthogonal basis, where the
integral criterion ``∫(data-model)^2`` collapses to a diagonal sum of squared
coefficient residuals. ``fit_lsi`` hard-codes the Legendre basis; this module
generalizes it along two axes for the ``dtfit_experimental`` adaptations:

* a pluggable basis: Legendre, Chebyshev, Fourier for periodic signals,
  Laguerre for decay;
* an additive empirical spectrum. The data coefficients are the integrals
  ``∫ y·φ_j``, which sum across a partition of the domain; that additivity is
  what makes the map-reduce / streaming estimator exact.

A :class:`Basis` exposes what the solver needs: where to sample the model
(``nodes``), how those samples become model coefficients (``model_spectrum``),
the data coefficients (``empirical`` by least squares, or ``project`` of
arbitrary samples for the additive path), and the diagonal criterion weights
(``sqrt_w``). :func:`solve_spectral` runs the weighted NLLS on top.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import sympy as sp
from numpy.polynomial import (
    legendre as L,
    chebyshev as C,
    laguerre as Lag,
)
from scipy.optimize import least_squares, differential_evolution, minimize

from dtfit._symbolic import model_params
from dtfit._stats import nlls_covariance
from dtfit.types import FittingResult
from dtfit._core._backend import Backend


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    """Per-sample trapezoid weights ``w`` with ``∫y dx ≈ Σ w_i y_i``.

    Folding the rule into a weight vector turns every projection into one
    matrix product ``Dᵀ·(w⊙y)``, which runs as a BLAS/cuBLAS GEMM and batches
    over channels. Numerically identical to ``np.trapezoid``.
    """
    n = x.size
    w = np.zeros(n)
    if n < 2:
        return w
    w[0] = (x[1] - x[0]) * 0.5
    w[-1] = (x[-1] - x[-2]) * 0.5
    if n > 2:
        w[1:-1] = (x[2:] - x[:-2]) * 0.5
    return w


class Basis:
    """An orthogonal (or near-orthogonal) basis over a fixed domain.

    Maps the data domain ``[x0, xn]`` to the basis's natural variable, exposes
    quadrature ``nodes`` mapped back to ``x`` for evaluating the model, and the
    projection that turns node samples into spectral coefficients.
    """

    name = "base"

    # Set by the quadrature subclasses (Legendre / Chebyshev / Laguerre) and
    # read by _project_spectrum; declared here for the type checker.
    _w: np.ndarray
    _V: np.ndarray
    _norm: np.ndarray

    def __init__(self, order: int, domain: tuple[float, float]) -> None:
        self.order = int(order)
        self.x0, self.xn = float(domain[0]), float(domain[1])
        self.h = self.xn - self.x0
        self.n_coef = self.order + 1

    # the x samples at which the model is evaluated for projection
    def nodes(self) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    # node samples -> spectral coefficients
    def model_spectrum(self, fv: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _project_spectrum(self, fv: np.ndarray) -> np.ndarray:
        """``norm ⊙ ((w ⊙ fv) @ V)`` through the numpy kernel.

        Shared by every basis that carries ``(_w, _V, _norm)``. The kernel is
        basis-agnostic despite the ``legendre_project`` name.
        """
        from dtfit._core._kernels import legendre_project

        return legendre_project(
            np.ascontiguousarray(fv, float),
            np.ascontiguousarray(self._w, float),
            np.ascontiguousarray(self._V, float),
            np.ascontiguousarray(self._norm, float),
        )

    # data (x, y) -> empirical spectrum by least squares (best conditioned)
    def empirical(  # pragma: no cover
        self, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    # (design matrix D, quadrature weights w) with ∫ y·φ_j dx ≈ (Dᵀ (w⊙y))_j
    def _gemm_factors(  # pragma: no cover
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # ∫ y·φ_j over the given samples only; a partition's partial spectra
    # sum to the whole-domain spectrum.
    def project_integral(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        D, w = self._gemm_factors(x)
        return D.T @ (w * y)  # GEMV; no (n, k) temporary

    def project_integral_batched(
        self, x: np.ndarray, Y: np.ndarray, backend: Backend
    ) -> np.ndarray:
        """Additive integrals ``s_j = ∫ y·φ_j`` for many channels, one GEMM.

        ``Y`` is ``(n, B)``, one column per channel; returns ``(B, n_coef)`` of
        un-normalized integrals over the given samples. Unlike
        :meth:`empirical_batched` this skips the per-coefficient norm, which
        keeps a partition's partial sums additive so they can be accumulated
        chunk by chunk. ``S = Dᵀ·(w⊙Y)`` is one matrix product: a GPU backend
        runs it on cuBLAS, and the cost amortizes over all ``B`` channels.
        """
        x = np.asarray(x, float)
        Y = np.asarray(Y)  # keep dtype; the backend picks compute precision
        if Y.dtype.kind != "f":
            Y = Y.astype(float)
        if Y.ndim == 1:
            Y = Y[:, None]
        D, w = self._gemm_factors(x)
        # Fold the weights into the small (n, k) design, not the large (n, B)
        # data. (w⊙D)ᵀ·Y avoids an (n, B) temporary: Y is the only big array
        # touched, and the GEMM reads it exactly once.
        Dd = backend.asarray(w[:, None] * D)  # (n, k) on device
        Yd = backend.asarray(Y)               # (n, B) on device
        S = Dd.T @ Yd                         # (k, B) GEMM
        return backend.to_host(S).T           # (B, k) raw integrals

    def empirical_batched(
        self, x: np.ndarray, Y: np.ndarray, backend: Backend
    ) -> np.ndarray:
        """Empirical spectra of many channels sharing grid ``x``, in one GEMM.

        ``Y`` is ``(n, B)``, one column per channel; returns ``(B, n_coef)``.
        """
        s = self.project_integral_batched(x, Y, backend)
        return self.integral_to_spectrum(s)   # (B, k); the norm broadcasts

    def integral_to_spectrum(self, s: np.ndarray) -> np.ndarray:
        """Turn accumulated integrals ``s_j = ∫ y·φ_j`` into coefficients."""
        return s  # identity; overridden where a norm applies

    def sqrt_w(self) -> np.ndarray:  # diagonal criterion weights
        raise NotImplementedError


# The reference basis; mirrors what fit_lsi does directly.
class LegendreBasis(Basis):
    name = "legendre"

    def __init__(self, order: int, domain: tuple[float, float]) -> None:
        super().__init__(order, domain)
        n_quad = max(2 * (order + 1), 16)
        self._u, self._w = L.leggauss(n_quad)  # nodes/weights on [-1, 1]
        self._V = np.polynomial.legendre.legvander(self._u, order)  # (nq, k)
        self._norm = (2.0 * np.arange(order + 1) + 1.0) / 2.0

    def nodes(self) -> np.ndarray:
        return self.x0 + self.h * (self._u + 1.0) / 2.0

    def model_spectrum(self, fv: np.ndarray) -> np.ndarray:
        return self._project_spectrum(fv)

    def empirical(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return L.Legendre.fit(x, y, self.order, domain=[self.x0, self.xn]).coef

    def _gemm_factors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # s_j = ∫ y(x) P_j(u(x)) dx over these samples; trapezoid, additive.
        u = 2.0 * (x - self.x0) / self.h - 1.0
        return (np.polynomial.legendre.legvander(u, self.order),
                _trapz_weights(x))

    def integral_to_spectrum(self, s: np.ndarray) -> np.ndarray:
        # β_j = (2j+1)/h * ∫ y P_j dx, the continuous Legendre coefficient.
        return (2.0 * np.arange(self.n_coef) + 1.0) / self.h * s

    def sqrt_w(self) -> np.ndarray:
        j = np.arange(self.n_coef)
        return np.sqrt(self.h / (2.0 * j + 1.0))


class ChebyshevBasis(Basis):
    name = "chebyshev"

    def __init__(self, order: int, domain: tuple[float, float]) -> None:
        super().__init__(order, domain)
        n_quad = max(2 * (order + 1), 16)
        self._u, self._w = C.chebgauss(n_quad)
        self._V = np.polynomial.chebyshev.chebvander(self._u, order)
        # T_j orthogonality on [-1,1] under w(x)=1/sqrt(1-x^2):
        # ∫ T_i T_j w = pi (i=j=0), pi/2 (i=j>0).
        self._norm = np.full(order + 1, 2.0 / np.pi)
        self._norm[0] = 1.0 / np.pi

    def nodes(self) -> np.ndarray:
        return self.x0 + self.h * (self._u + 1.0) / 2.0

    def model_spectrum(self, fv: np.ndarray) -> np.ndarray:
        return self._project_spectrum(fv)

    def empirical(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return C.Chebyshev.fit(
            x, y, self.order, domain=[self.x0, self.xn]
        ).coef

    def _gemm_factors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = 2.0 * (x - self.x0) / self.h - 1.0
        return (np.polynomial.chebyshev.chebvander(u, self.order),
                _trapz_weights(x))

    def integral_to_spectrum(self, s: np.ndarray) -> np.ndarray:
        return self._norm * (2.0 / self.h) * s

    def sqrt_w(self) -> np.ndarray:
        return np.ones(self.n_coef)


class FourierBasis(Basis):
    """Real Fourier basis ``{1, cos(2πk t/P), sin(2πk t/P)}`` over the domain.

    For periodic or seasonal signals. ``order`` is the number of harmonics K,
    so the spectrum has ``2K+1`` coefficients ``[a0, a1..aK, b1..bK]``. The
    period defaults to the domain length ``P=h``, one fundamental cycle across
    the window.
    """

    name = "fourier"

    def __init__(
        self,
        order: int,
        domain: tuple[float, float],
        *,
        period: float | None = None,
    ) -> None:
        super().__init__(order, domain)
        self.K = int(order)
        self.P = self.h if period is None else float(period)
        self.n_coef = 2 * self.K + 1
        self._nq = max(8 * (self.K + 1), 64)
        self._tg = np.linspace(self.x0, self.xn, self._nq)  # dense model grid
        # The model-grid design does not depend on the coefficients. Factor it
        # once and model_spectrum becomes a GEMV rather than an lstsq (SVD)
        # per residual evaluation.
        # (n_coef, nq)
        self._model_pinv = np.linalg.pinv(self._design(self._tg))

    def _design(self, x: np.ndarray) -> np.ndarray:
        ph = 2.0 * np.pi * (np.asarray(x, float) - self.x0) / self.P
        cols = [np.ones_like(x, dtype=float)]
        cols += [np.cos(k * ph) for k in range(1, self.K + 1)]
        cols += [np.sin(k * ph) for k in range(1, self.K + 1)]
        return np.column_stack(cols)

    def nodes(self) -> np.ndarray:
        return self._tg

    def model_spectrum(self, fv: np.ndarray) -> np.ndarray:
        # least-squares coefficients of the model on the dense grid
        return self._model_pinv @ np.asarray(fv, dtype=float)

    def empirical(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(self._design(x), y, rcond=None)[0]

    def _gemm_factors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._design(x), _trapz_weights(x)

    def integral_to_spectrum(self, s: np.ndarray) -> np.ndarray:
        # trig orthogonality over a full period: ⟨1,1⟩=P,
        # ⟨cos_k,cos_k⟩=⟨sin_k,sin_k⟩=P/2.
        norm = np.full(self.n_coef, 2.0 / self.P)
        norm[0] = 1.0 / self.P
        return norm * s

    def sqrt_w(self) -> np.ndarray:
        return np.ones(self.n_coef)


# For decay / transient signals on [0, ∞), rescaled onto the domain.
class LaguerreBasis(Basis):
    name = "laguerre"

    def __init__(self, order: int, domain: tuple[float, float]) -> None:
        super().__init__(order, domain)
        n_quad = max(2 * (order + 1), 16)
        self._u, self._w = Lag.laggauss(n_quad)  # nodes on [0, ∞), weight e^-u
        self._V = np.polynomial.laguerre.lagvander(self._u, order)
        self._norm = np.ones(order + 1)  # ∫ L_i L_j e^-u = δ_ij

    def _to_u(self, x: np.ndarray) -> np.ndarray:
        # map domain -> [0, 5]
        return (np.asarray(x, float) - self.x0) / self.h * 5.0

    def nodes(self) -> np.ndarray:
        return self.x0 + self.h * self._u / 5.0

    def model_spectrum(self, fv: np.ndarray) -> np.ndarray:
        return self._project_spectrum(fv)

    def empirical(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return Lag.Laguerre.fit(self._to_u(x), y, self.order).coef

    def _gemm_factors(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # integrate over u, with the Laguerre weight e^-u folded into D
        u = self._to_u(x)
        D = (np.polynomial.laguerre.lagvander(u, self.order)
             * np.exp(-u)[:, None])
        return D, _trapz_weights(u)

    def integral_to_spectrum(self, s: np.ndarray) -> np.ndarray:
        return s

    def sqrt_w(self) -> np.ndarray:
        return np.ones(self.n_coef)


_BASES: dict[str, type[Basis]] = {
    "legendre": LegendreBasis,
    "chebyshev": ChebyshevBasis,
    "fourier": FourierBasis,
    "laguerre": LaguerreBasis,
}


def make_basis(
    name: str, order: int, domain: tuple[float, float], **kwargs: Any
) -> Basis:
    """Construct a basis by name: legendre, chebyshev, fourier or laguerre."""
    try:
        cls = _BASES[name]
    except KeyError:
        raise ValueError(
            f"unknown basis {name!r}; choose from {sorted(_BASES)}"
        ) from None
    return cls(order, domain, **kwargs)


def solve_spectral(
    expr: str,
    var: str,
    basis: Basis,
    beta_data: np.ndarray,
    *,
    p0: np.ndarray | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> FittingResult:
    """Match a model's spectrum to ``beta_data`` on ``basis`` (weighted NLLS).

    Evaluates the lambdified model at the basis nodes, projects it, then
    minimizes the diagonal-weighted coefficient residual. Fully finite
    ``bounds`` put a differential-evolution search ahead of the local refine,
    which is what makes a multimodal fit such as a free-frequency Fourier one
    insensitive to ``p0``. See :func:`solve_weighted_nlls`.
    """
    t = sp.Symbol(var)
    f_sym = cast(sp.Expr, sp.sympify(expr))
    params = model_params(f_sym, t)
    if not params:
        raise RuntimeError("Model expression has no free parameters to fit.")

    nodes = basis.nodes()
    f_func = sp.lambdify((t, *params), f_sym, "numpy")
    sqrt_w = basis.sqrt_w()

    def model_spectrum(c: np.ndarray) -> np.ndarray:
        fv = np.asarray(f_func(nodes, *c), dtype=float)
        if fv.ndim == 0:
            fv = np.full_like(nodes, float(fv))
        return basis.model_spectrum(fv)

    def residual(c: np.ndarray) -> np.ndarray:
        spec = model_spectrum(c)
        if not np.all(np.isfinite(spec)):
            return np.full(beta_data.size, 1e6)
        return sqrt_w * (beta_data - spec)

    guess = np.ones(len(params)) if p0 is None else np.asarray(p0, float)
    coeffs, jac, converged, message, nfev = solve_weighted_nlls(
        residual, sqrt_w, beta_data, guess, p0=p0, bounds=bounds
    )
    cov = nlls_covariance(jac, residual(coeffs), len(params))
    # No model= here: FittingResult lambdifies it lazily from expr and coeffs.
    # The optimizer status and the fitted domain go through so callers get a
    # `converged` signal and the predict(warn_extrapolation) guard.
    return FittingResult(
        coeffs=coeffs, cov=cov, expr=expr, var=var,
        names=tuple(str(p) for p in params),
        converged=converged, message=message, nfev=nfev,
        x_range=(basis.x0, basis.xn),
    )


def solve_weighted_nlls(
    residual,
    sqrt_w: np.ndarray,
    beta_data: np.ndarray,
    guess: np.ndarray,
    *,
    p0=None,
    bounds: list[tuple[float, float]] | None = None,
    seed: int | None = 0,
    solver_options: dict[str, Any] | None = None,
):
    """Bounded / unbounded weighted-NLLS driver for the LSI spectral match.

    Without bounds this is a weighted Levenberg-Marquardt from ``guess``. With
    bounds it first tries a bounded local (trf) solve from the supplied seed,
    accepting it only if the solver converged and the relative residual came
    in under 0.5; otherwise it falls back to a differential-evolution search
    refined by L-BFGS-B, the multimodal-safe path for something like a free
    frequency. ``seed`` seeds that search for reproducibility.

    The global stage needs a finite box and runs only when every bound is
    finite. Bounds reach the local trf solve either way (scipy handles
    ``+/-inf`` there); a partially bounded problem stays constrained.

    ``solver_options`` forwards solver tolerances where they apply: ``xtol`` /
    ``ftol`` / ``gtol`` / ``max_nfev`` to
    :func:`scipy.optimize.least_squares`, and ``ftol`` / ``gtol`` /
    ``max_nfev`` (as ``maxfun``) to the L-BFGS-B refine. Unknown keys are
    ignored.

    Returns ``(coeffs, jac, converged, message, nfev)``, where ``nfev`` counts
    the residual / cost evaluations the solver reported, summed across both
    stages on the differential-evolution path.
    """
    opts = solver_options or {}
    ls_opts = {
        k: opts[k] for k in ("xtol", "ftol", "gtol", "max_nfev") if k in opts
    }
    min_opts: dict[str, Any] = {}
    if "ftol" in opts:
        min_opts["ftol"] = opts["ftol"]
    if "gtol" in opts:
        min_opts["gtol"] = opts["gtol"]
    if "max_nfev" in opts:
        min_opts["maxfun"] = opts["max_nfev"]

    if bounds is not None:
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        # DE needs a finite search box; trf accepts +/-inf bounds directly.
        all_finite = bool(np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)))
        if p0 is not None or not all_finite:
            loc = least_squares(
                residual, np.clip(guess, lo, hi), bounds=(lo, hi),
                method="trf", **ls_opts,
            )
            denom = float(np.linalg.norm(sqrt_w * beta_data)) + 1e-30
            good = loc.success and float(np.linalg.norm(loc.fun)) / denom < 0.5
            if good or not all_finite:
                # With any infinite bound the global stage cannot run. This
                # local solve is the answer whether or not it looked good.
                return (np.asarray(loc.x, dtype=np.float64), loc.jac,
                        bool(loc.success), str(loc.message), int(loc.nfev))

        def cost(c: np.ndarray) -> float:
            r = residual(c)
            return float(r @ r)

        # cast: `seed` is the portable argument across scipy versions; newer
        # stubs only expose its `rng` successor.
        res_g = cast(Any, differential_evolution)(
            cost, bounds, strategy="best1bin", popsize=15, seed=seed
        )
        res = minimize(
            cost, res_g.x, method="L-BFGS-B", bounds=bounds,
            options=min_opts or None,
        )
        coeffs = np.asarray(res.x, dtype=np.float64)
        nfev = int(getattr(res_g, "nfev", 0)) + int(getattr(res, "nfev", 0))
        return (coeffs, _numeric_jac(residual, coeffs),
                bool(res.success), str(res.message), nfev)

    sol = least_squares(residual, guess, method="lm", **ls_opts)
    return (np.asarray(sol.x, dtype=np.float64), sol.jac,
            bool(sol.success), str(sol.message), int(sol.nfev))


def _numeric_jac(residual, c: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Forward-difference Jacobian of a residual vector at ``c``."""
    r0 = residual(c)
    jac = np.empty((r0.size, c.size))
    for k in range(c.size):
        step = eps * max(1.0, abs(c[k]))
        cp = c.copy()
        cp[k] += step
        jac[:, k] = (residual(cp) - r0) / step
    return jac
