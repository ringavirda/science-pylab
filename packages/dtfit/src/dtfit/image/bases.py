"""Bases an image can be taken in.

A basis is any family of test functions on the unit interval; orthogonality
is not required because the image carries the Gram matrix of the family on
the sample grid. ``order`` is the polynomial degree for Legendre and the
number of windows for the block basis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.polynomial import legendre as L


def u_of(x: np.ndarray, x0: float, x1: float) -> np.ndarray:
    """Map positions on ``[x0, x1]`` to the basis variable on ``[-1, 1]``."""
    if x1 == x0:
        raise ValueError(f"degenerate domain: x0 == x1 == {x0}")
    return 2.0 * (np.asarray(x, dtype=float) - x0) / (x1 - x0) - 1.0


class Basis:
    """Interface of a basis: ``evaluate`` on the unit variable, ``n_coef``."""

    name = "base"

    def __init__(self, order: int) -> None:
        order = int(order)
        if order < 1:
            raise ValueError(
                f"{self.name} basis needs order >= 1, got {order}"
            )
        self.order = order

    @property
    def n_coef(self) -> int:
        raise NotImplementedError

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        """Basis functions at ``u`` in ``[-1, 1]``: shape
        ``(len(u), n_coef)``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "order": self.order}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Basis) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((self.name, self.order))


class LegendreBasis(Basis):
    """Legendre polynomials ``P_0 .. P_order``; the LSI basis."""

    name = "legendre"

    @property
    def n_coef(self) -> int:
        return self.order + 1

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        return L.legvander(np.asarray(u, dtype=float), self.order)


class BlockBasis(Basis):
    """Indicators of ``order`` equal windows on the unit interval; the
    EAC basis."""

    name = "block"

    @property
    def n_coef(self) -> int:
        return self.order

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        idx = np.clip(
            np.floor((u + 1.0) / 2.0 * self.order).astype(int),
            0,
            self.order - 1,
        )
        phi = np.zeros((u.size, self.order))
        phi[np.arange(u.size), idx] = 1.0
        return phi


_BASES: dict[str, type[Basis]] = {
    "legendre": LegendreBasis,
    "block": BlockBasis,
}


def make_basis(basis: str | Basis, order: int | None) -> Basis:
    """A basis by name at ``order``, or the given :class:`Basis` unchanged."""
    if isinstance(basis, Basis):
        return basis
    try:
        cls = _BASES[str(basis)]
    except KeyError:
        raise ValueError(
            f"unknown basis {basis!r}; choose from {sorted(_BASES)}"
        ) from None
    if order is None:
        raise ValueError(f"basis {basis!r} needs an order")
    return cls(order)
