"""The image of a signal: its discrete statistic in a basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import cholesky

from .bases import Basis, make_basis, u_of
from .grid import Grid
from .original import Original

if TYPE_CHECKING:
    from .analytics import ChiSquareTest, Decay


def gram_whitener(G: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor of ``G`` with a relative jitter of ``1e-14``
    times the mean diagonal, so a Gram that is singular to rounding still
    factors. ``G`` is a square symmetric matrix; the factor ``L`` satisfies
    ``L @ L.T = G + jitter * I``.

    Args:
        G: square symmetric matrix, shape ``(k, k)``.

    Returns:
        Lower-triangular ``L`` of the same shape.

    Raises:
        scipy.linalg.LinAlgError: ``G`` is not positive definite even
            after the jitter.
    """
    k = G.shape[0]
    jitter = 1e-14 * float(np.trace(G)) / k
    return cholesky(G + jitter * np.eye(k), lower=True)


def huber_weights(
    Phi: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    c: float = 1.345,
    passes: int = 5,
) -> np.ndarray:
    """Huber weights from an iteratively reweighted regression of ``y`` on the
    basis ``Phi``, starting from the base weights ``w``.

    Each pass fits ``y ~ Phi beta`` with the current weights, reads the
    residual and its MAD scale ``s`` and sets the weight of sample ``i`` to
    ``min(1, c s / |r_i|)``. Returns the multiplicative weights (ones for
    clean samples), not the product with ``w``.
    """
    mult = np.ones(y.size)
    for _ in range(passes):
        ww = w * mult
        beta = np.linalg.lstsq(
            Phi * np.sqrt(ww)[:, None], y * np.sqrt(ww), rcond=None
        )[0]
        r = y - Phi @ beta
        s = 1.4826 * float(np.median(np.abs(r - np.median(r)))) + 1e-12
        mult = np.minimum(1.0, c * s / np.maximum(np.abs(r), 1e-12))
    return mult


@dataclass(frozen=True, eq=False)
class Image:
    """The discrete image of a signal in a basis on a domain.

    ``S = Phi^T (w y)`` are the weighted projections, ``G = Phi^T diag(w)
    Phi`` the Gram matrix of the basis on the sample grid; with ``n``,
    ``sumsq = sum(w y^2)``, ``sumy = sum(w y)`` and ``wsum = sum(w)`` they
    are the sufficient statistic of the linear model in that basis. ``w``
    is the per-sample weights, stored when any weight differs from one or
    the image is robust; else None.

    Images with the same basis, order and domain merge by adding their sums
    whatever their sample sets (:meth:`merge`); a Legendre image is nested
    (:meth:`truncate`); the least-squares coefficients ``beta = G^-1 S`` are
    derived, never stored.
    """

    basis: Basis
    domain: tuple[float, float]
    S: np.ndarray
    G: np.ndarray
    n: int
    sumsq: float
    sumy: float
    wsum: float
    grid: Grid
    w: np.ndarray | None = None
    robust: bool = False

    @property
    def order(self) -> int:
        return self.basis.order

    @property
    def n_coef(self) -> int:
        return self.basis.n_coef

    @property
    def weighted(self) -> bool:
        """True when the image carries a weight vector."""
        return self.w is not None

    @cached_property
    def beta(self) -> np.ndarray:
        """Least-squares coefficients: the minimum-norm solution of
        ``G beta = S`` with singular values below 1e-15 of the largest
        dropped."""
        return np.linalg.pinv(self.G, hermitian=True) @ self.S

    def phi(self) -> np.ndarray:
        """The basis evaluated on the image's grid, ``(n, n_coef)``.

        Weights are not applied; a caller forming ``G`` from this must
        multiply by ``w`` itself.
        """
        return self.basis.evaluate(u_of(self.grid.positions(), *self.domain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        if (
            self.basis != other.basis
            or self.domain != other.domain
            or self.grid != other.grid
            or self.n != other.n
            or self.robust != other.robust
            or self.sumsq != other.sumsq
            or self.sumy != other.sumy
            or self.wsum != other.wsum
        ):
            return False
        if not (
            np.array_equal(self.S, other.S)
            and np.array_equal(self.G, other.G)
        ):
            return False
        if (self.w is None) != (other.w is None):
            return False
        if self.w is None or other.w is None:
            return True
        return np.array_equal(self.w, other.w)

    def __hash__(self) -> int:
        return hash(
            (self.basis, self.domain, self.grid, self.n, self.robust)
        )

    @classmethod
    def of(
        cls,
        original: Original,
        basis: str | Basis = "legendre",
        order: int | None = None,
        *,
        robust: bool = False,
        huber_c: float = 1.345,
    ) -> "Image":
        if not isinstance(basis, Basis) and order is None:
            raise ValueError("order is required to build an image")
        b = make_basis(basis, order)
        if original.n < b.n_coef + 1:
            raise ValueError(
                f"an image at order {b.order} needs at least "
                f"{b.n_coef + 1} samples; got {original.n}"
            )
        x0, x1 = original.domain
        Phi = b.evaluate(u_of(original.x, x0, x1))
        w = original.w
        if robust:
            w = w * huber_weights(Phi, original.y, w, huber_c)
        S = Phi.T @ (w * original.y)
        G = Phi.T @ (w[:, None] * Phi)
        store_w = w if (robust or original.weighted) else None
        return cls(
            b, (x0, x1), S, G, original.n,
            float(np.sum(w * original.y ** 2)),
            float(np.sum(w * original.y)), float(w.sum()),
            original.grid, store_w, robust,
        )

    @classmethod
    def of_model(
        cls,
        model: Any,
        params: Any,
        grid: Grid,
        basis: str | Basis = "legendre",
        order: int | None = None,
        *,
        var: str | None = None,
        domain: tuple[float, float] | None = None,
        w: np.ndarray | None = None,
        param_names: Any = None,
    ) -> "Image":
        """The image the model ``f(x; params)`` would have on ``grid``.

        ``param_names`` names the parameters of a callable model whose
        signature cannot be introspected; see
        :func:`~dtfit.methods.resolve_model`.
        """
        from dtfit.methods._modelinput import resolve_model

        spec = resolve_model(model, var, param_names=param_names)
        x = grid.positions()
        f = spec.eval(x, np.asarray(params, dtype=float))
        dom = domain if domain is not None else (float(x[0]), float(x[-1]))
        return cls.of(Original(x, f, w, domain=dom), basis, order)

    def merge(self, other: "Image") -> "Image":
        """The image of the two signals' samples pooled, whatever their
        sample sets; requires the same basis (including order) and domain.

        The sums (``S``, ``G``, ``n``, ``sumsq``, ``sumy``, ``wsum``) are
        additive regardless of sample order, so they are simply added; the
        grid is rebuilt from the sorted union of both position sets
        (:meth:`Grid.merge`).
        """
        if self.basis != other.basis or self.domain != other.domain:
            raise ValueError(
                "images to merge must share basis, order and domain"
            )
        positions = np.concatenate(
            [self.grid.positions(), other.grid.positions()]
        )
        idx = np.argsort(positions, kind="stable")
        grid = Grid.of(positions[idx])
        if self.weighted or other.weighted:
            wa = self.w if self.w is not None else np.ones(self.n)
            wb = other.w if other.w is not None else np.ones(other.n)
            w: np.ndarray | None = np.concatenate([wa, wb])[idx]
        else:
            w = None
        return Image(
            self.basis, self.domain, self.S + other.S, self.G + other.G,
            self.n + other.n, self.sumsq + other.sumsq,
            self.sumy + other.sumy, self.wsum + other.wsum,
            grid, w, self.robust or other.robust,
        )

    def truncate(self, order: int) -> "Image":
        """The image at a lower order; exact for nested bases (Legendre)."""
        if self.basis.name != "legendre":
            raise ValueError(
                f"the {self.basis.name} basis is not nested; "
                "cannot truncate"
            )
        if order > self.order:
            raise ValueError(
                f"cannot truncate order {self.order} to {order}"
            )
        k = order + 1
        return Image(
            make_basis("legendre", order), self.domain,
            self.S[:k].copy(), self.G[:k, :k].copy(),
            self.n, self.sumsq, self.sumy, self.wsum,
            self.grid, self.w.copy() if self.w is not None else None,
            self.robust,
        )

    def transfer(
        self, domain: tuple[float, float], order: int | None = None
    ) -> "Image":
        """This image expressed in the same basis on a coarser ``domain``
        that contains this one, at ``order`` (default this order).

        Exact for the Legendre basis when ``order`` is at most this order
        (``S_c = A^T S``, ``G_c = A^T G A`` with the change-of-basis matrix
        of :func:`~dtfit.image.transfer.legendre_transfer`); for the block
        basis every fine window must lie inside one coarse window, and a
        sample at this image's right edge belongs to the coarse window
        that starts there, not the one before it (see
        :func:`~dtfit.image.transfer.block_transfer`). The sample grid,
        weights, counts and sums are unchanged. Domain containment is
        checked with an absolute tolerance of ``1e-9 * (domain[1] -
        domain[0])``.

        Raises:
            ValueError: ``domain`` does not contain this domain; for the
                Legendre basis ``order`` above this order; for the block
                basis a coarse window that is not a union of fine windows
                (window counts are not compared, the domains differ); an
                unknown basis name.
        """
        from .transfer import block_transfer, legendre_transfer

        order = self.order if order is None else int(order)
        d0, d1 = float(domain[0]), float(domain[1])
        tol = 1e-9 * max(d1 - d0, 0.0)
        if self.domain[0] < d0 - tol or self.domain[1] > d1 + tol:
            raise ValueError(
                f"the coarse domain {domain} must contain this image's "
                f"domain {self.domain}"
            )
        if self.basis.name == "legendre":
            if order > self.order:
                raise ValueError(
                    f"coarse order {order} above this image's order "
                    f"{self.order}"
                )
            A = legendre_transfer(
                self.domain, (d0, d1), self.order, order
            )
        elif self.basis.name == "block":
            A = block_transfer(self.domain, (d0, d1), self.order, order)
        else:
            raise ValueError(
                f"transfer is not defined for the {self.basis.name} basis"
            )
        return Image(
            basis=make_basis(self.basis.name, order),
            domain=(d0, d1),
            S=A.T @ self.S,
            G=A.T @ self.G @ A,
            n=self.n,
            sumsq=self.sumsq,
            sumy=self.sumy,
            wsum=self.wsum,
            grid=self.grid,
            w=None if self.w is None else self.w.copy(),
            robust=self.robust,
        )

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """The least-squares reconstruction of the signal at ``x``.

        A position outside the domain is evaluated in the basis's own
        extension: the edge window for the block basis, the polynomial
        continuation for Legendre.
        """
        Phi = self.basis.evaluate(
            u_of(np.asarray(x, dtype=float), *self.domain)
        )
        return Phi @ self.beta

    def simulate(
        self,
        n: int | None = None,
        sigma: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """A synthetic record drawn from this image.

        The reconstruction (:meth:`reconstruct`) plus independent Gaussian
        noise, on the image's own sample grid or on a fresh uniform one.

        Args:
            n: Number of positions. ``None`` uses the image's own grid;
                an int at least 2 uses that many evenly spaced positions
                over the domain, endpoints included.
            sigma: Standard deviation of the added noise, at least 0.
                ``None`` takes :meth:`noise_sigma`.
            rng: A ``numpy.random.Generator``; ``None`` makes a fresh one,
                so the draw is not reproducible.

        Returns:
            ``(x, y)``, both of length ``n`` (or of the image's sample
            count).

        Raises:
            TypeError: ``n`` is not an integer; the noise level is
                ``sigma``, the second argument.
            ValueError: ``n`` below 2; ``sigma`` negative or not finite;
                or ``sigma=None`` with no readable noise level (a
                non-Legendre basis, or fewer than eight orders above the
                effective order).
        """
        from .analytics import _tail_variance

        rng = np.random.default_rng() if rng is None else rng
        if n is None:
            x = self.grid.positions()
        else:
            if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
                raise TypeError(
                    "simulate(n, sigma, rng): n is a sample count; pass the "
                    f"noise level as sigma= (got n={n!r})"
                )
            if int(n) < 2:
                raise ValueError(f"simulate needs n >= 2; got {int(n)}")
            x = np.linspace(self.domain[0], self.domain[1], int(n))
        if sigma is None:
            v = _tail_variance(self)
            if v is None:
                raise ValueError(
                    "simulate(sigma=None) reads the noise level from the "
                    "image's tail orders, which this image does not have; "
                    "pass sigma explicitly"
                )
            s = float(np.sqrt(v))
        else:
            s = float(sigma)
            if not np.isfinite(s) or s < 0.0:
                raise ValueError(
                    f"sigma must be finite and non-negative; got {sigma}"
                )
        return x, self.reconstruct(x) + s * rng.standard_normal(x.size)

    def noise_sigma(self) -> float | None:
        """The noise standard deviation read off the tail orders.

        Returns:
            The standard deviation, or ``None`` when fewer than eight
            orders stand above the effective order.

        Raises:
            ValueError: the image is not in the Legendre basis.

        Warns:
            RuntimeWarning: the return is ``None`` for lack of tail
                orders.

        See :func:`~dtfit.image.analytics.noise_sigma` for the formula.
        """
        from .analytics import noise_sigma

        return noise_sigma(self)

    def effective_order(self) -> int:
        """The highest order whose coefficient stands above the noise.

        Returns:
            An order in ``[0, image.order]``.

        Raises:
            ValueError: the image is not in the Legendre basis.

        See :func:`~dtfit.image.analytics.effective_order` for the
        formula.
        """
        from .analytics import effective_order

        return effective_order(self)

    def decay(self) -> "Decay":
        """Geometric and algebraic decay rates of the coefficients.

        Returns:
            A :class:`~dtfit.image.analytics.Decay` with both rates,
            both goodness-of-fit values and the name of the better law.

        Raises:
            ValueError: the image is not in the Legendre basis; or fewer
                than three non-zero coefficients lie between order 2 and
                the effective order.

        See :func:`~dtfit.image.analytics.decay` for the formula.
        """
        from .analytics import decay

        return decay(self)

    def test_equal(
        self, other: "Image", alpha: float = 0.05, *,
        sigma: float | None = None,
    ) -> "ChiSquareTest":
        """Chi-square test that ``other`` is an image of the same signal.

        Returns:
            A :class:`~dtfit.image.analytics.ChiSquareTest`; ``reject``
            is ``True`` when the images differ by more than noise.

        Raises:
            ValueError: the two images differ in basis, order or domain;
                ``alpha`` outside ``(0, 1)``; ``sigma`` not finite and
                positive; or, with ``sigma=None``, neither image has a
                residual degree of freedom to pool a noise scale from.

        See :func:`~dtfit.image.analytics.test_equal` for the formula.
        """
        from .analytics import test_equal

        return test_equal(self, other, alpha, sigma=sigma)

    def test_structure(
        self, model: Any, params: Any, var: str | None = None, *,
        alpha: float = 0.05, param_names: Any = None,
        sigma: float | None = None, fitted: bool = True,
    ) -> "ChiSquareTest":
        """Chi-square test that a model explains everything in the span.

        Returns:
            A :class:`~dtfit.image.analytics.ChiSquareTest`; ``reject``
            is ``True`` when the model leaves structure in the span
            unexplained.

        Raises:
            ValueError: ``alpha`` outside ``(0, 1)``; ``sigma`` not
                finite and positive; the model is not finite on the
                image's grid; the image has no degrees of freedom left
                after the model's parameters; the noise scale cannot be
                estimated.
            RuntimeError: the model has no free parameters.

        See :func:`~dtfit.image.analytics.test_structure` for the
        formula.
        """
        from .analytics import test_structure

        return test_structure(
            self, model, params, var, alpha=alpha, param_names=param_names,
            sigma=sigma, fitted=fitted,
        )

    def fit(self, model: Any, var: str | None = None, **kwargs: Any):
        from .fit import fit

        return fit(model, self, var, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis.to_dict(), "domain": list(self.domain),
            "S": self.S.tolist(), "G": self.G.tolist(), "n": self.n,
            "sumsq": self.sumsq, "sumy": self.sumy, "wsum": self.wsum,
            "grid": self.grid.to_dict(),
            "w": None if self.w is None else self.w.tolist(),
            "robust": self.robust,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Image":
        return cls(
            make_basis(d["basis"]["name"], d["basis"]["order"]),
            (float(d["domain"][0]), float(d["domain"][1])),
            np.asarray(d["S"], dtype=float), np.asarray(d["G"], dtype=float),
            int(d["n"]), float(d["sumsq"]), float(d["sumy"]),
            float(d["wsum"]), Grid.from_dict(d["grid"]),
            None if d.get("w") is None else np.asarray(d["w"], dtype=float),
            bool(d.get("robust", False)),
        )
