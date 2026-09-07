"""Compact descriptor of the sample positions behind an image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, eq=False)
class Grid:
    """Sample positions of an :class:`Original`.

    ``kind`` is ``"uniform"`` when the spacing is constant to a relative
    tolerance of 1e-9, in which case the positions are regenerated from
    ``n``, ``x0`` and ``x1`` and ``x`` is ``None``; otherwise ``"explicit"``
    and the positions are stored. The model side of a fit is evaluated on
    these positions, so an image of a uniform grid needs only three numbers
    beyond its projections.
    """

    kind: str
    n: int
    x0: float
    x1: float
    x: np.ndarray | None = None

    @classmethod
    def of(cls, x: np.ndarray) -> "Grid":
        """Positions must be non-decreasing.

        :raises ValueError: if ``x`` is empty.
        """
        x = np.asarray(x, dtype=float)
        n = int(x.size)
        if n == 0:
            raise ValueError("grid needs at least one position")
        if n >= 2:
            d = np.diff(x)
            if np.allclose(
                d, d[0], rtol=1e-9, atol=1e-9 * abs(float(d[0]))
            ):
                return cls("uniform", n, float(x[0]), float(x[-1]))
        return cls("explicit", n, float(x[0]), float(x[-1]), x.copy())

    def positions(self) -> np.ndarray:
        if self.kind == "uniform":
            return np.linspace(self.x0, self.x1, self.n)
        if self.x is None:
            raise RuntimeError(
                "explicit grid has no stored positions"
            )
        return self.x.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        if (self.kind, self.n, self.x0, self.x1) != (
            other.kind,
            other.n,
            other.x0,
            other.x1,
        ):
            return False
        if self.kind == "explicit":
            return np.array_equal(self.positions(), other.positions())
        return True

    def __hash__(self) -> int:
        return hash((self.kind, self.n, self.x0, self.x1))

    def merge(self, other: "Grid") -> "Grid":
        """The union of ``self`` and ``other``'s positions, stably sorted.

        No ordering between the two is required, and ties (a position
        shared by both) are kept, not deduplicated. Uniform when the union
        is uniform to :meth:`of`'s tolerance, explicit otherwise.
        """
        positions = np.concatenate([self.positions(), other.positions()])
        order = np.argsort(positions, kind="stable")
        return Grid.of(positions[order])

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "n": self.n,
            "x0": self.x0,
            "x1": self.x1,
        }
        if self.kind == "explicit":
            d["x"] = self.positions().tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Grid":
        if d["kind"] == "explicit":
            return cls(
                "explicit",
                int(d["n"]),
                float(d["x0"]),
                float(d["x1"]),
                np.asarray(d["x"], dtype=float),
            )
        return cls("uniform", int(d["n"]), float(d["x0"]), float(d["x1"]))
