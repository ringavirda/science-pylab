"""The image of a signal: its discrete statistic in a basis."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .bases import Basis, make_basis, u_of
from .grid import Grid
from .original import Original


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

    Images with the same basis, order, domain and grid kind are additive
    (:meth:`merge`), a Legendre image is nested (:meth:`truncate`), and the
    least-squares coefficients ``beta = G^-1 S`` are derived, never stored.
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
        return self.w is None or np.array_equal(self.w, other.w)

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
    ) -> "Image":
        """The image the model ``f(x; params)`` would have on ``grid``."""
        from dtfit.methods._modelinput import resolve_model

        spec = resolve_model(model, var)
        x = grid.positions()
        f = spec.eval(x, np.asarray(params, dtype=float))
        dom = domain if domain is not None else (float(x[0]), float(x[-1]))
        return cls.of(Original(x, f, w, domain=dom), basis, order)

    def merge(self, other: "Image") -> "Image":
        """The image of the two signals concatenated; ``other`` follows
        ``self``."""
        if self.basis != other.basis or self.domain != other.domain:
            raise ValueError(
                "images to merge must share basis, order and domain"
            )
        if self.weighted or other.weighted:
            wa = self.w if self.w is not None else np.ones(self.n)
            wb = other.w if other.w is not None else np.ones(other.n)
            w: np.ndarray | None = np.concatenate([wa, wb])
        else:
            w = None
        return Image(
            self.basis, self.domain, self.S + other.S, self.G + other.G,
            self.n + other.n, self.sumsq + other.sumsq,
            self.sumy + other.sumy, self.wsum + other.wsum,
            self.grid.merge(other.grid), w, self.robust or other.robust,
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
            self.grid, self.w, self.robust,
        )

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """The least-squares reconstruction of the signal at ``x``."""
        Phi = self.basis.evaluate(
            u_of(np.asarray(x, dtype=float), *self.domain)
        )
        return Phi @ self.beta

    def simulate(
        self, sigma: float, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(x, y)`` on the image's grid: the reconstruction plus Gaussian
        noise of std ``sigma``."""
        rng = np.random.default_rng() if rng is None else rng
        x = self.grid.positions()
        return x, self.reconstruct(x) + float(sigma) * rng.standard_normal(
            x.size
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
