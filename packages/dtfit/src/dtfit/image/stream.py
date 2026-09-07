"""One stream over the image: a running image over a fixed domain, block
images with local domains, and channel batches over one shared Gram."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from dtfit._core._backend import Backend, resolve_backend
from dtfit.streaming.detect import DriftDetector
from .bases import Basis, make_basis, u_of
from .grid import Grid
from .image import Image, gram_whitener
from .original import Original
from .transfer import assemble as _assemble

_STATE_VERSION = 1


class _Sums:
    """Running sums of one image over one domain for ``channels`` signals
    sharing the sample positions. Positions are tracked as a uniform grid
    (endpoints, spacing, count) or kept explicitly."""

    def __init__(self, n_coef: int, channels: int, explicit: bool) -> None:
        self.S = np.zeros((channels, n_coef))
        self.G = np.zeros((n_coef, n_coef))
        self.n = 0
        self.sumsq = np.zeros(channels)
        self.sumy = np.zeros(channels)
        self.wsum = 0.0
        self.explicit = explicit
        self.x0 = 0.0
        self.x1 = 0.0
        self.dx: float | None = None
        self.xs: list[np.ndarray] = []
        self.ws: list[np.ndarray] = []
        self.weighted = False

    def _track_uniform(self, x: np.ndarray) -> None:
        m = x.size
        if m > 1:
            d = np.diff(x)
            dx = float(d[0])
            if dx <= 0.0 or not np.allclose(
                d, dx, rtol=1e-9, atol=1e-9 * abs(dx)
            ):
                raise ValueError(
                    "a uniform stream needs increasing, evenly spaced "
                    "positions; use grid='explicit' for other grids"
                )
            if self.dx is None:
                self.dx = dx
            elif not np.isclose(dx, self.dx, rtol=1e-9, atol=0.0):
                raise ValueError(
                    f"chunk spacing {dx:g} differs from the stream's "
                    f"uniform spacing {self.dx:g}"
                )
        if self.n == 0:
            self.x0 = float(x[0])
        else:
            if self.dx is None:
                self.dx = float(x[0]) - self.x1
                if self.dx <= 0.0:
                    raise ValueError("positions must increase")
            expected = self.x1 + self.dx
            if abs(float(x[0]) - expected) > 1e-9 * abs(self.dx):
                raise ValueError(
                    f"chunk starts at {float(x[0]):g}, the uniform grid "
                    f"expects {expected:g}"
                )
        self.x1 = float(x[-1])

    def add(
        self,
        x: np.ndarray,
        Y: np.ndarray,
        w: np.ndarray | None,
        Phi: np.ndarray,
        backend: Backend,
    ) -> None:
        m = x.size
        ww = np.ones(m) if w is None else w
        wy = ww[:, None] * Y
        if backend.name == "numpy":
            S_inc = Phi.T @ wy
        else:
            S_inc = backend.to_host(
                backend.asarray(Phi).T @ backend.asarray(wy)
            )
        if self.explicit:
            self.xs.append(np.array(x, dtype=float, copy=True))
            self.ws.append(np.array(ww, dtype=float, copy=True))
        else:
            self._track_uniform(x)
        self.S += np.asarray(S_inc).T
        self.G += Phi.T @ (ww[:, None] * Phi)
        self.n += m
        self.sumsq += ww @ (Y * Y)
        self.sumy += ww @ Y
        self.wsum += float(ww.sum())
        if w is not None and np.any(w != 1.0):
            self.weighted = True

    def grid_and_weights(self) -> tuple[Grid, np.ndarray | None]:
        if self.explicit:
            x = np.concatenate(self.xs) if self.xs else np.zeros(0)
            w = np.concatenate(self.ws) if self.ws else np.zeros(0)
            order = np.argsort(x, kind="stable")
            return Grid.of(x[order]), (w[order] if self.weighted else None)
        return Grid("uniform", self.n, self.x0, self.x1), None

    def image(
        self, basis: Basis, domain: tuple[float, float], channel: int
    ) -> Image:
        if self.n == 0:
            raise ValueError("the stream holds no samples yet")
        grid, w = self.grid_and_weights()
        return Image(
            basis=basis,
            domain=domain,
            S=self.S[channel].copy(),
            G=self.G.copy(),
            n=self.n,
            sumsq=float(self.sumsq[channel]),
            sumy=float(self.sumy[channel]),
            wsum=float(self.wsum),
            grid=grid,
            w=w,
            robust=False,
        )

    def state(self) -> dict[str, Any]:
        return {
            "S": self.S.tolist(),
            "G": self.G.tolist(),
            "n": self.n,
            "sumsq": self.sumsq.tolist(),
            "sumy": self.sumy.tolist(),
            "wsum": self.wsum,
            "x0": self.x0,
            "x1": self.x1,
            "dx": self.dx,
            "xs": [a.tolist() for a in self.xs],
            "ws": [a.tolist() for a in self.ws],
            "weighted": self.weighted,
        }

    def restore(self, d: dict[str, Any]) -> None:
        self.S = np.asarray(d["S"], dtype=float)
        self.G = np.asarray(d["G"], dtype=float)
        self.n = int(d["n"])
        self.sumsq = np.asarray(d["sumsq"], dtype=float)
        self.sumy = np.asarray(d["sumy"], dtype=float)
        self.wsum = float(d["wsum"])
        self.x0 = float(d["x0"])
        self.x1 = float(d["x1"])
        self.dx = None if d["dx"] is None else float(d["dx"])
        self.xs = [np.asarray(a, dtype=float) for a in d["xs"]]
        self.ws = [np.asarray(a, dtype=float) for a in d["ws"]]
        self.weighted = bool(d["weighted"])


class ImageStream:
    """A running image over a fixed domain, block images with local domains,
    or a channel batch over one shared Gram; the three uses of the same
    running statistic ``S = Phi^T w y``, ``G = Phi^T diag(w) Phi``.

    Args:
        basis: ``"legendre"``, ``"block"`` or a :class:`Basis` instance.
        order: basis order, at least 1 (with a ``Basis`` instance, its
            order).
        domain: ``(x0, x1)`` the images are taken over; required unless
            ``block`` gives a domain length.
        block: ``None`` for the accumulator; an ``int`` for blocks of that
            many samples, at least ``order + 2``; a ``float`` for blocks
            of that domain length, counted from the domain's start.
            Block mode needs ``channels == 1``.
        channels: number of signals sharing the sample positions; ``y``
            passed to :meth:`update` then has shape ``(n, channels)``.
        grid: ``"uniform"`` tracks the positions as endpoints and spacing
            and requires evenly spaced, increasing chunks that continue
            each other; ``"explicit"`` keeps every position (and weights).
        keep_fine, fold: block retention: at most ``keep_fine`` fine
            blocks are kept; when there are more, the oldest ``fold`` are
            folded into one coarse block at the stream's order.
        backend: ``"numpy"``, ``"cupy"`` or ``"torch"`` for the projection
            GEMM; accumulation is float64 on the host either way.
        detect: block-level drift detection, block mode only: ``None``
            for none, ``"previous"`` to compare each finished block to
            the one before it, or ``(model, params)`` / ``(model,
            params, var)`` to compare it to that model's image.

    Raises:
        ValueError: missing domain, ``order < 1``, ``channels < 1``, an
            unknown grid kind or backend; in block mode, ``channels !=
            1``, a ``block`` below ``order + 2`` samples or non-positive
            length, or an unrecognised ``detect``; ``detect`` given
            without ``block``.
    """

    def __init__(
        self,
        basis: str | Basis,
        order: int | None = None,
        *,
        domain: tuple[float, float] | None = None,
        block: int | float | None = None,
        channels: int = 1,
        grid: str = "uniform",
        keep_fine: int = 64,
        fold: int = 16,
        backend: str = "numpy",
        detect: Any = None,
    ) -> None:
        if domain is None:
            raise ValueError("domain is required")
        d0, d1 = float(domain[0]), float(domain[1])
        if not d1 > d0:
            raise ValueError(f"domain must satisfy x0 < x1, got {domain}")
        if channels < 1:
            raise ValueError(
                f"channels must be at least 1, got {channels}"
            )
        if grid not in ("uniform", "explicit"):
            raise ValueError(
                f"grid must be 'uniform' or 'explicit', got {grid!r}"
            )
        self.basis: Basis = make_basis(basis, order)
        self.order = self.basis.order
        self.domain = (d0, d1)
        self.channels = int(channels)
        self.grid_kind = grid
        self.keep_fine = int(keep_fine)
        self.fold = int(fold)
        self._backend = resolve_backend(backend)
        self.block = block
        self.detect = detect
        self._block_count: int | None = None
        self._block_len: float | None = None
        if block is not None:
            if self.channels != 1:
                raise ValueError("block mode needs channels == 1")
            if isinstance(block, (bool, np.bool_)):
                raise ValueError("block must be an int or a float")
            if isinstance(block, (int, np.integer)):
                if int(block) < self.order + 2:
                    raise ValueError(
                        f"a block needs at least order + 2 = "
                        f"{self.order + 2} samples, got {block}"
                    )
                self._block_count = int(block)
            else:
                if float(block) <= 0.0:
                    raise ValueError("a block length must be positive")
                self._block_len = float(block)
            self._detector = self._make_detector(detect)
        elif detect is not None:
            raise ValueError("detect needs block mode")
        self._sums = _Sums(
            self.basis.n_coef, self.channels, grid == "explicit"
        )
        self._buf_x: list[np.ndarray] = []
        self._buf_y: list[np.ndarray] = []
        self._buf_w: list[np.ndarray] = []
        self._buf_n = 0
        self._block_index = 0
        self.fine_: list[Image] = []
        self.coarse_: list[Image] = []
        self.flags_: list[tuple[int, tuple[float, float]]] = []
        self._prev: Image | None = None

    @property
    def n(self) -> int:
        """Samples accumulated so far."""
        return self._sums.n

    def _config(self) -> dict[str, Any]:
        return {
            "basis": self.basis.to_dict(),
            "domain": list(self.domain),
            "block": self.block,
            "channels": self.channels,
            "grid": self.grid_kind,
            "keep_fine": self.keep_fine,
            "fold": self.fold,
            "detect": (
                "previous" if self.detect == "previous"
                else ("model" if self.detect is not None else None)
            ),
        }

    def _make_detector(self, detect: Any) -> DriftDetector | None:
        if detect is None:
            return None
        if detect == "previous":
            return DriftDetector(self.basis.n_coef)
        if isinstance(detect, tuple) and len(detect) in (2, 3):
            return DriftDetector(self.basis.n_coef)
        raise ValueError(
            "detect must be None, 'previous', (model, params) or "
            "(model, params, var)"
        )

    def _current_domain(self) -> tuple[float, float]:
        k = self._block_index
        L = self._block_len
        assert L is not None
        return (self.domain[0] + k * L, self.domain[0] + (k + 1) * L)

    def _finish_block(self) -> Image:
        x = np.concatenate(self._buf_x)
        y = np.concatenate(self._buf_y)
        w = np.concatenate(self._buf_w) if self._buf_w else None
        if self._block_len is not None:
            dom = self._current_domain()
        else:
            dom = None
        img = Image.of(
            Original(x, y, w, domain=dom), self.basis, self.order
        )
        self._buf_x, self._buf_y, self._buf_w, self._buf_n = [], [], [], 0
        self._check_drift(img)
        self._block_index += 1
        self.fine_.append(img)
        while (
            len(self.fine_) > self.keep_fine and len(self.fine_) >= self.fold
        ):
            old, self.fine_ = self.fine_[:self.fold], self.fine_[self.fold:]
            self.coarse_.append(_assemble(old, order=self.order))
        return img

    def _check_drift(self, img: Image) -> None:
        if self._detector is None:
            return
        xg = img.grid.positions()
        if self.detect == "previous":
            if self._prev is None:
                self._prev = img
                return
            f = self._prev.reconstruct(xg)
        else:
            model, params = self.detect[0], self.detect[1]
            var = self.detect[2] if len(self.detect) == 3 else None
            f = Image.of_model(
                model, params, img.grid, self.basis, self.order, var=var,
                domain=img.domain, w=img.w,
            ).S
        if self.detect == "previous":
            w = img.w if img.w is not None else np.ones(img.n)
            S_f = img.phi().T @ (w * f)
        else:
            S_f = f
        e = solve_triangular(gram_whitener(img.G), img.S - S_f, lower=True)
        flagged = self._detector.update(e)
        self._prev = img
        if flagged:
            self.flags_.append((self._block_index, img.domain))

    def _validate(
        self, x: Any, y: Any, w: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        xa = np.asarray(x, dtype=float).reshape(-1)
        Y = np.asarray(y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.ndim != 2 or Y.shape != (xa.size, self.channels):
            raise ValueError(
                f"y must have shape ({xa.size}, {self.channels}) for this "
                f"stream, got {tuple(np.shape(y))}"
            )
        if xa.size == 0:
            raise ValueError("a chunk needs at least one sample")
        if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(Y))):
            raise ValueError("x and y must be finite")
        if np.any(xa < self.domain[0]) or np.any(xa > self.domain[1]):
            raise ValueError(
                f"positions must lie inside the domain {self.domain}"
            )
        wa = None
        if w is not None:
            if self.grid_kind != "explicit":
                raise ValueError("weights need grid='explicit'")
            wa = np.asarray(w, dtype=float).reshape(-1)
            if wa.size != xa.size:
                raise ValueError("w must have one weight per sample")
            if not np.all(np.isfinite(wa)) or np.any(wa <= 0.0):
                raise ValueError("weights must be finite and positive")
        return xa, Y, wa

    def update(self, x: Any, y: Any, w: Any = None) -> list[Image]:
        """Add a chunk of samples. ``x`` has shape ``(m,)``, ``y`` shape
        ``(m,)`` or ``(m, channels)``, ``w`` optional positive weights
        (explicit grids only). Returns the block images finished by this
        chunk, an empty list for the accumulator.

        Raises:
            ValueError: shape mismatch, non-finite input, positions outside
                the domain, weights on a uniform stream, or a chunk that
                breaks the uniform grid.
        """
        xa, Y, wa = self._validate(x, y, w)
        if self.block is None:
            Phi = self.basis.evaluate(u_of(xa, *self.domain))
            self._sums.add(xa, Y, wa, Phi, self._backend)
            return []
        finished: list[Image] = []
        y1 = Y[:, 0]
        start = 0
        while start < xa.size:
            if self._block_count is not None:
                take = min(self._block_count - self._buf_n, xa.size - start)
            else:
                _, hi = self._current_domain()
                last = hi >= self.domain[1] - 1e-9 * (
                    self.domain[1] - self.domain[0]
                )
                take = int(np.searchsorted(
                    xa[start:], hi, side="right" if last else "left"
                ))
                if take == 0:
                    if self._buf_n == 0:
                        self._block_index += 1
                        continue
                    finished.append(self._finish_block())
                    continue
            sl = slice(start, start + take)
            self._buf_x.append(xa[sl])
            self._buf_y.append(y1[sl])
            if wa is not None:
                self._buf_w.append(wa[sl])
            self._buf_n += take
            start += take
            if (
                self._block_count is not None
                and self._buf_n == self._block_count
            ):
                finished.append(self._finish_block())
            elif self._block_len is not None and start < xa.size:
                finished.append(self._finish_block())
        return finished

    def close(self) -> list[Image]:
        """Finish the partial block if it holds at least ``order + 2``
        samples and return it in a list, else return an empty list. A
        length block closed early keeps its fixed domain.

        Raises:
            ValueError: the stream is not in block mode.
        """
        if self.block is None:
            raise ValueError("close applies to block mode")
        if self._buf_n < self.order + 2:
            return []
        return [self._finish_block()]

    def blocks(self, t0: float, t1: float) -> list[Image]:
        """Stored block images whose domain lies inside ``[t0, t1]``,
        coarse blocks first, each list in time order."""
        tol = 1e-9 * (self.domain[1] - self.domain[0])
        return [
            b for b in [*self.coarse_, *self.fine_]
            if b.domain[0] >= t0 - tol and b.domain[1] <= t1 + tol
        ]

    def assemble(
        self, t0: float, t1: float, order: int | None = None
    ) -> Image:
        """The image of the whole blocks inside ``[t0, t1]`` on the hull of
        their domains at ``order`` (default the stream's order).

        Raises:
            ValueError: no whole block inside the range, or ``order`` above
                the stream's order.
        """
        found = self.blocks(t0, t1)
        if not found:
            raise ValueError(f"no blocks lie inside [{t0:g}, {t1:g}]")
        return _assemble(found, order=self.order if order is None else order)

    def image(self, channel: int = 0) -> Image:
        """The running image of one channel.

        Raises:
            ValueError: no samples yet, or ``channel`` out of range.
        """
        if not 0 <= channel < self.channels:
            raise ValueError(f"channel {channel} out of range")
        return self._sums.image(self.basis, self.domain, channel)

    def images(self) -> list[Image]:
        """The running images of every channel, in channel order."""
        return [self.image(c) for c in range(self.channels)]

    def merge(self, other: "ImageStream") -> "ImageStream":
        """A new stream holding both sample sets. Both streams need the
        same configuration; uniform streams must be contiguous (one starts
        one spacing after the other ends).

        Raises:
            ValueError: different configuration, or uniform streams that
                are not contiguous.
        """
        if self._config() != other._config():
            raise ValueError("streams must have the same configuration")
        out = ImageStream(
            self.basis, domain=self.domain, channels=self.channels,
            grid=self.grid_kind, keep_fine=self.keep_fine, fold=self.fold,
            backend=self._backend.name,
        )
        a, b = self._sums, other._sums
        if a.n == 0 or b.n == 0:
            src = b if a.n == 0 else a
            out._sums.restore(src.state())
            return out
        if not a.explicit:
            first, second = (a, b) if a.x0 <= b.x0 else (b, a)
            dx = first.dx if first.dx is not None else second.dx
            if dx is None or abs(second.x0 - (first.x1 + dx)) > 1e-9 * dx:
                raise ValueError(
                    "uniform streams merge only when contiguous; use "
                    "grid='explicit' for other sample sets"
                )
            out._sums.x0 = first.x0
            out._sums.x1 = second.x1
            out._sums.dx = dx
        else:
            out._sums.xs = [*a.xs, *b.xs]
            out._sums.ws = [*a.ws, *b.ws]
        out._sums.S = a.S + b.S
        out._sums.G = a.G + b.G
        out._sums.n = a.n + b.n
        out._sums.sumsq = a.sumsq + b.sumsq
        out._sums.sumy = a.sumy + b.sumy
        out._sums.wsum = a.wsum + b.wsum
        out._sums.weighted = a.weighted or b.weighted
        return out

    def checkpoint(self) -> dict[str, Any]:
        """The complete state as JSON-serializable data; ``resume`` on a
        stream built with the same arguments continues from it. In block
        mode a checkpoint from ``detect=(model, params[, var])`` needs the
        same ``detect`` argument again on resume; only the accumulated
        detector state travels with the checkpoint."""
        state: dict[str, Any] = {
            "version": _STATE_VERSION,
            "config": self._config(),
            "sums": self._sums.state(),
        }
        if self.block is not None:
            x = np.concatenate(self._buf_x) if self._buf_x else np.zeros(0)
            y = np.concatenate(self._buf_y) if self._buf_y else np.zeros(0)
            w = np.concatenate(self._buf_w) if self._buf_w else None
            state["partial"] = {
                "x": x.tolist(),
                "y": y.tolist(),
                "w": None if w is None else w.tolist(),
            }
            state["block_index"] = self._block_index
            state["fine"] = [img.to_dict() for img in self.fine_]
            state["coarse"] = [img.to_dict() for img in self.coarse_]
            state["flags"] = [[i, list(dom)] for i, dom in self.flags_]
            state["prev"] = (
                None if self._prev is None else self._prev.to_dict()
            )
            state["detector"] = (
                None if self._detector is None else self._detector.state()
            )
        return state

    def resume(self, state: dict[str, Any]) -> "ImageStream":
        """Load a checkpoint into this stream and return it.

        Raises:
            ValueError: the checkpoint was taken from a stream with a
                different configuration or an unknown version.
        """
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unknown checkpoint version")
        if state["config"] != self._config():
            raise ValueError(
                "checkpoint configuration differs from this stream's"
            )
        self._sums.restore(state["sums"])
        if self.block is not None:
            partial = state["partial"]
            x = np.asarray(partial["x"], dtype=float)
            self._buf_x = [x] if x.size else []
            y = np.asarray(partial["y"], dtype=float)
            self._buf_y = [y] if y.size else []
            if partial["w"] is None:
                self._buf_w = []
            else:
                self._buf_w = [np.asarray(partial["w"], dtype=float)]
            self._buf_n = int(x.size)
            self._block_index = int(state["block_index"])
            self.fine_ = [Image.from_dict(d) for d in state["fine"]]
            self.coarse_ = [Image.from_dict(d) for d in state["coarse"]]
            self.flags_ = [
                (int(i), (float(dom[0]), float(dom[1])))
                for i, dom in state["flags"]
            ]
            self._prev = (
                None if state["prev"] is None
                else Image.from_dict(state["prev"])
            )
            if self._detector is not None and state["detector"] is not None:
                self._detector.restore(state["detector"])
        return self
