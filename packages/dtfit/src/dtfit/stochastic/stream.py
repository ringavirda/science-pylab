"""The block stream over second-order images: the accumulating and the
block form of the same additive statistic."""

from __future__ import annotations

from typing import Any

import numpy as np

from .image import SecondOrderImage

__all__ = ["SecondOrderStream"]

_STATE_VERSION = 1


class SecondOrderStream:
    """A running second-order image, or a stream of block images.

    The block form is the batch counterpart of
    :class:`~dtfit.stochastic.StochasticFilter`: blocks are exact and
    mergeable and resolve a change at block granularity, where the filter's
    exponentially weighted statistics resolve it within about a half-life.

    Args:
        block: samples per block, at least 2; ``None`` accumulates one
            running image instead.
        lag, nfreq, scales: the block images' budgets; see
            :class:`SecondOrderImage`.
        x0: position of global sample index 0.
        dx: sample spacing in position units, strictly positive.
        keep_fine: block images kept at block resolution, at least 1.
        fold: fine blocks merged into one coarse block once there are more
            than ``keep_fine`` of them, at least 1.

    Raises:
        ValueError: ``block`` below 2, ``keep_fine`` or ``fold`` below 1, or
            a budget :class:`SecondOrderImage` rejects.
    """

    def __init__(
        self,
        block: int | None = None,
        *,
        lag: int = 256,
        nfreq: int = 512,
        scales: int = 10,
        x0: float = 0.0,
        dx: float = 1.0,
        keep_fine: int = 64,
        fold: int = 16,
    ) -> None:
        if block is not None and int(block) < 2:
            raise ValueError(f"a block needs at least 2 samples, got {block}")
        if int(keep_fine) < 1:
            raise ValueError(f"keep_fine must be at least 1, got {keep_fine}")
        if int(fold) < 1:
            raise ValueError(f"fold must be at least 1, got {fold}")
        self.block = None if block is None else int(block)
        self.lag = int(lag)
        self.nfreq = int(nfreq)
        self.scales = int(scales)
        self.x0 = float(x0)
        self.dx = float(dx)
        self.keep_fine = int(keep_fine)
        self.fold = int(fold)
        self.n = 0
        self.fine_: list[SecondOrderImage] = []
        self.coarse_: list[SecondOrderImage] = []
        self._buf: list[np.ndarray] = []
        self._buf_n = 0
        self._current = self._new_image(0)

    def _new_image(self, t0: int) -> SecondOrderImage:
        return SecondOrderImage(
            self.lag, self.nfreq, self.scales,
            t0=t0, x0=self.x0, dx=self.dx,
        )

    def _config(self) -> dict[str, Any]:
        return {
            "block": self.block, "lag": self.lag, "nfreq": self.nfreq,
            "scales": self.scales, "x0": self.x0, "dx": self.dx,
            "keep_fine": self.keep_fine, "fold": self.fold,
        }

    def update(self, y: Any) -> list[SecondOrderImage]:
        """Add a chunk of samples and return the block images it finished.

        In accumulator mode the chunk goes into the running image and the
        list is empty. In block mode the samples fill the current block; each
        block that fills is imaged, retained and returned.

        Raises:
            ValueError: an empty chunk, a chunk that is not 1-D, or
                non-finite values.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"a chunk must be 1-D, got {y.ndim} dimensions")
        if y.size == 0:
            raise ValueError("a chunk needs at least one sample")
        if not np.all(np.isfinite(y)):
            raise ValueError("the series must be finite")
        if self.block is None:
            self._current.update(y)
            self.n += y.size
            return []
        finished: list[SecondOrderImage] = []
        start = 0
        while start < y.size:
            take = min(self.block - self._buf_n, y.size - start)
            self._buf.append(y[start:start + take])
            self._buf_n += take
            start += take
            if self._buf_n == self.block:
                finished.append(self._finish())
        self.n += y.size
        return finished

    def _finish(self) -> SecondOrderImage:
        img = self._new_image(self.n_blocked)
        img.update(np.concatenate(self._buf))
        self._buf, self._buf_n = [], 0
        self.fine_.append(img)
        while (len(self.fine_) > self.keep_fine
               and len(self.fine_) >= self.fold):
            old, self.fine_ = self.fine_[:self.fold], self.fine_[self.fold:]
            merged = old[0]
            for nxt in old[1:]:
                merged = merged.merge(nxt)
            self.coarse_.append(merged)
        return img

    @property
    def n_blocked(self) -> int:
        """Samples already closed into block images."""
        return sum(b.n for b in self.coarse_) + sum(b.n for b in self.fine_)

    def close(self) -> list[SecondOrderImage]:
        """Finish the partial block if it holds at least two samples and
        return it in a list, else return an empty list.

        Raises:
            ValueError: the stream is in accumulator mode.
        """
        if self.block is None:
            raise ValueError("close applies to block mode")
        if self._buf_n < 2:
            return []
        return [self._finish()]

    def image(self) -> SecondOrderImage:
        """The running image, accumulator mode only.

        Raises:
            ValueError: block mode (use :meth:`blocks` or :meth:`assemble`),
                or no samples yet.
        """
        if self.block is not None:
            raise ValueError(
                "image() applies to accumulator mode; use blocks() or "
                "assemble() in block mode"
            )
        if self._current.n == 0:
            raise ValueError("the stream holds no samples yet")
        return self._current

    def blocks(self, t0: float, t1: float) -> list[SecondOrderImage]:
        """The stored block images whose whole span lies inside the positions
        ``[t0, t1]``, coarse blocks first, each list in time order."""
        tol = 1e-9 * max(abs(t1 - t0), 1.0)
        return [
            b for b in [*self.coarse_, *self.fine_]
            if b.domain[0] >= t0 - tol and b.domain[1] <= t1 + tol
        ]

    def assemble(self, t0: float, t1: float) -> SecondOrderImage:
        """The image of the whole blocks inside ``[t0, t1]``, merged.

        Raises:
            ValueError: no whole block inside the range, or blocks that are
                not consecutive (a folded range with a gap).
        """
        found = self.blocks(t0, t1)
        if not found:
            raise ValueError(f"no blocks lie inside [{t0:g}, {t1:g}]")
        found = sorted(found, key=lambda b: b.t0)
        out = found[0]
        for nxt in found[1:]:
            out = out.merge(nxt)
        return out

    def checkpoint(self) -> dict[str, Any]:
        """The complete state as JSON-serializable data; :meth:`resume` on a
        stream built with the same arguments continues from it."""
        return {
            "version": _STATE_VERSION,
            "config": self._config(),
            "n": self.n,
            "partial": np.concatenate(self._buf).tolist() if self._buf else [],
            "current": self._current.state(),
            "fine": [b.state() for b in self.fine_],
            "coarse": [b.state() for b in self.coarse_],
        }

    def resume(self, state: dict[str, Any]) -> "SecondOrderStream":
        """Load a checkpoint into this stream and return it, discarding
        whatever it already held.

        Raises:
            ValueError: an unknown version or a different configuration.
        """
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unknown checkpoint version")
        if state["config"] != self._config():
            raise ValueError(
                "checkpoint configuration differs from this stream's")
        self.n = int(state["n"])
        part = np.asarray(state["partial"], dtype=float)
        self._buf = [part] if part.size else []
        self._buf_n = int(part.size)
        self._current = self._new_image(0).restore(state["current"])
        self.fine_ = [self._new_image(0).restore(d) for d in state["fine"]]
        self.coarse_ = [self._new_image(0).restore(d) for d in state["coarse"]]
        return self
