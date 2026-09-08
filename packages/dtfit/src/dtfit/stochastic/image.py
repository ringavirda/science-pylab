"""The stochastic tier's image: the additive second-order statistic of a
uniformly sampled series, and the read-outs every gate and estimator of the
tier takes its numbers from."""

from __future__ import annotations

import warnings
from typing import Any, Literal, overload

import numpy as np

from dtfit.image.original import Original

__all__ = ["SecondOrderImage"]

_STATE_VERSION = 1

# Largest frequency grid SecondOrderImage.of takes on its own: 2 * this many
# residues, the image's largest field.
_MAX_AUTO_NFREQ = 4096


def _fft_len(n: int) -> int:
    """Smallest power of two at least ``n`` (``n >= 1``)."""
    return 1 << (max(1, int(n)) - 1).bit_length()


def _index_moments(start: int, n: int) -> tuple[int, int]:
    """``(sum t, sum t^2)`` over the ``n`` consecutive indices from ``start``,
    in Python integers (``start >= 0``, ``n >= 1``)."""
    lo, hi = int(start), int(start) + int(n) - 1

    def sq(x: int) -> int:
        return x * (x + 1) * (2 * x + 1) // 6

    return (lo + hi) * n // 2, sq(hi) - sq(lo - 1)


def _lag_sums(y: np.ndarray, lag: int) -> np.ndarray:
    """Raw lagged sums ``sum_{t >= k} y_t y_{t-k}``, ``k = 0..lag``, inside one
    sequence, by one FFT autocorrelation."""
    out = np.zeros(lag + 1)
    n = y.size
    if n == 0:
        return out
    m = _fft_len(2 * n)
    f = np.fft.rfft(y, m)
    ac = np.fft.irfft(f * np.conj(f), m)
    k = min(lag + 1, n)
    out[:k] = ac[:k]
    return out


def _cross_sums(left: np.ndarray, right: np.ndarray, lag: int) -> np.ndarray:
    """Lagged sums of the pairs that straddle the join of ``left`` and
    ``right``: ``sum_j right_j left_{j + m - k}`` at lag ``k = 1..lag``, ``m``
    the length of ``left``. One FFT cross-correlation, zero at ``k = 0``."""
    out = np.zeros(lag + 1)
    m, n = left.size, right.size
    if m == 0 or n == 0:
        return out
    nfft = _fft_len(m + n)
    c = np.fft.irfft(
        np.conj(np.fft.rfft(left, nfft)) * np.fft.rfft(right, nfft), nfft
    )
    k = np.arange(1, lag + 1)
    live = k <= m + n - 1
    out[1:][live] = c[(k[live] - m) % nfft]
    return out


def _centred_acov(
    c: np.ndarray, s: float, n: int, head: np.ndarray, tail: np.ndarray
) -> np.ndarray:
    """Centred autocovariance ``(1/n) sum_{t>=k} (y_t - m)(y_{t-k} - m)`` from
    the raw lagged sums ``c``, the total ``s = sum y``, the count ``n`` and the
    head and tail carries."""
    lag = c.size - 1
    if n <= 0:
        return np.zeros(lag + 1)
    m = s / n
    hs = np.concatenate([[0.0], np.cumsum(head)])
    ts = np.concatenate([[0.0], np.cumsum(tail[::-1])])
    k = np.arange(lag + 1)
    s_t = s - hs[np.minimum(k, head.size)]
    s_tk = s - ts[np.minimum(k, tail.size)]
    cnt = np.maximum(n - k, 0)
    return (c - m * (s_t + s_tk) + cnt * m * m) / n


class SecondOrderImage:
    """The second-order image of a uniformly sampled series.

    Every field is a sum over samples, so two images of consecutive stretches
    merge into the image of their union: the fields add and the lagged sums
    gain the cross terms between the first image's tail carry and the second's
    head carry.

    Fields, for a series ``y`` at global sample indices ``t = t0 .. t0+n-1``
    and positions ``x = x0 + dx * t``:

    * ``n``, ``sum_y``, and the time cross-sums ``sum_t``, ``sum_ty``,
      ``sum_t2`` (the trend);
    * raw lagged sums to lag ``lag`` of ``y``, of the increments
      ``dy_t = y_t - y_{t-1}`` and of the squared increments;
    * squared block sums and block counts at the scales ``2^j``,
      ``j = 0..scales``, aligned to the global index grid, with the partial
      block at each end;
    * the residue accumulator of the DFT on the fixed grid
      ``f_m = m / (2 nfreq)``, ``m = 0..nfreq-1``;
    * the head and tail carries, the first and last ``lag + 1`` samples.

    Args:
        lag: lag budget ``L`` in samples, at least 1. The autocovariance,
            the Blackman-Tukey spectrum and the unit-root statistic read at
            most this many lags.
        nfreq: frequency grid size ``M``, at least 1; the DFT bins are
            ``f_m = m / (2M)`` cycles per sample, ``m = 0..M-1``.
        scales: scale budget ``J``, at least 0; block sums are kept at the
            scales ``2^0 .. 2^J`` samples.
        t0: global sample index of the first sample, at least 0.
        x0: position of global index 0.
        dx: sample spacing in position units, strictly positive.

    Raises:
        ValueError: ``lag < 1``, ``nfreq < 1``, ``scales < 0``, ``t0 < 0`` or
            ``dx <= 0``.
    """

    def __init__(
        self,
        lag: int = 256,
        nfreq: int = 512,
        scales: int = 10,
        *,
        t0: int = 0,
        x0: float = 0.0,
        dx: float = 1.0,
    ) -> None:
        if int(lag) < 1:
            raise ValueError(f"lag must be at least 1, got {lag}")
        if int(nfreq) < 1:
            raise ValueError(f"nfreq must be at least 1, got {nfreq}")
        if int(scales) < 0:
            raise ValueError(f"scales must be at least 0, got {scales}")
        if int(t0) < 0:
            raise ValueError(f"t0 must be at least 0, got {t0}")
        if not float(dx) > 0.0:
            raise ValueError(f"dx must be positive, got {dx}")
        self.lag = int(lag)
        self.nfreq = int(nfreq)
        self.scales = int(scales)
        self.t0 = int(t0)
        self.x0 = float(x0)
        self.dx = float(dx)
        self.n = 0
        self.sum_y = 0.0
        self.sum_t = 0
        self.sum_t2 = 0
        self.sum_ty = 0.0
        self.c_y = np.zeros(self.lag + 1)
        self.c_dy = np.zeros(self.lag + 1)
        self.c_d2 = np.zeros(self.lag + 1)
        self.c_sq = np.zeros(self.lag + 1)
        j = self.scales + 1
        self.ss = np.zeros(j)
        self.nb = np.zeros(j, dtype=np.int64)
        self.lead_sum = np.zeros(j)
        self.lead_len = np.zeros(j, dtype=np.int64)
        self.part_sum = np.zeros(j)
        self.part_len = np.zeros(j, dtype=np.int64)
        self.residues = np.zeros(2 * self.nfreq)
        self.head = np.zeros(0)
        self.tail = np.zeros(0)

    # ------------------------------------------------------------ building
    @classmethod
    def of(
        cls,
        data: Original | np.ndarray,
        *,
        lag: int = 256,
        nfreq: int = 512,
        scales: int | None = None,
    ) -> "SecondOrderImage":
        """The image of a whole record.

        Args:
            data: an :class:`~dtfit.image.Original` on a uniform grid, or a
                1-D array of values at unit spacing from position 0.
            lag: lag budget in samples, capped at ``n - 1``.
            nfreq: smallest frequency grid to take. The grid is raised to
                ``2^ceil(log2 n) / 2``, the size that puts at most one
                frequency of the record in each bin so the seasonal read-out
                resolves it, and capped at 4096 bins; a record above 8192
                samples is under-resolved and :meth:`seasonal` says so.
            scales: scale budget; ``None`` takes ``floor(log2(n)) - 3``,
                floored at 0, the largest scale with at least eight blocks.

        Raises:
            ValueError: fewer than two samples, or an Original whose grid is
                not uniform (the tier's statistics assume a constant spacing).
        """
        if isinstance(data, Original):
            if data.grid.kind != "uniform":
                raise ValueError(
                    "the second-order image needs a uniformly sampled "
                    "series; this Original's grid is explicit"
                )
            y = data.y
            n = y.size
            x0 = float(data.grid.x0)
            dx = (float(data.grid.x1) - x0) / (n - 1) if n > 1 else 1.0
        else:
            y = np.asarray(data, dtype=float).reshape(-1)
            n = y.size
            x0, dx = 0.0, 1.0
        if n < 2:
            raise ValueError(f"need at least 2 samples; got {n}")
        if scales is None:
            scales = max(0, int(np.floor(np.log2(n))) - 3)
        img = cls(
            lag=max(1, min(int(lag), n - 1)),
            nfreq=min(max(int(nfreq), _fft_len(n) // 2), _MAX_AUTO_NFREQ),
            scales=scales,
            x0=x0,
            dx=dx,
        )
        return img.update(y)

    def update(self, y: Any) -> "SecondOrderImage":
        """Add the next chunk of samples in place and return self.

        The chunk continues the series: its first sample sits at global index
        ``t0 + n``. Cost is one FFT autocorrelation and one FFT
        cross-correlation per lagged-sum series, plus O(chunk) work.

        Args:
            y: the next 1-D array of finite sample values, at unit spacing
                continuing the record.

        Raises:
            ValueError: an empty chunk, a chunk that is not 1-D, or non-finite
                values.
        """
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError(f"a chunk must be 1-D, got {y.ndim} dimensions")
        if y.size == 0:
            raise ValueError("a chunk needs at least one sample")
        if not np.all(np.isfinite(y)):
            raise ValueError("the series must be finite")
        lag = self.lag
        n = y.size
        start = self.t0 + self.n
        t = start + np.arange(n)

        # first moments and time cross-sums; the index moments are closed
        # forms in Python integers, exact at any record length.
        self.sum_y += float(y.sum())
        self.sum_ty += float(np.einsum("i,i->", t.astype(float), y))
        m1, m2 = _index_moments(start, n)
        self.sum_t += m1
        self.sum_t2 += m2

        # lagged sums of the level, with the cross terms against the tail
        self.c_y += _lag_sums(y, lag) + _cross_sums(self.tail, y[:lag], lag)

        # lagged sums of the increments and of the squared increments; the
        # increment at the chunk's first sample needs the previous sample.
        prev = self.tail[-1:] if self.tail.size else np.zeros(0)
        d_new = np.diff(np.concatenate([prev, y]))
        d_tail = np.diff(self.tail)
        self.c_dy += _lag_sums(d_new, lag) + _cross_sums(
            d_tail, d_new[:lag], lag
        )
        self.c_d2 += _lag_sums(d_new * d_new, lag) + _cross_sums(
            d_tail * d_tail, (d_new * d_new)[:lag], lag
        )

        ysq = y * y
        self.c_sq += _lag_sums(ysq, lag) + _cross_sums(
            self.tail ** 2, ysq[:lag], lag
        )

        # dyadic block sums aligned to the global index grid
        for j in range(self.scales + 1):
            self._add_blocks(j, y)

        # the DFT on f_m = m / (2 nfreq) is the DFT of the sample sums folded
        # modulo 2 nfreq
        two_m = 2 * self.nfreq
        self.residues += np.bincount(t % two_m, weights=y, minlength=two_m)

        if self.head.size < lag + 1:
            self.head = np.concatenate([self.head, y])[: lag + 1]
        self.tail = np.concatenate([self.tail, y])[-(lag + 1):]
        self.n += n
        return self

    def _add_blocks(self, j: int, y: np.ndarray) -> None:
        """Fold one chunk into the block sums at scale ``2^j``."""
        m = 1 << j
        need = (-self.t0) % m
        pos = self.n
        arr = y
        if pos < need:
            k = min(arr.size, need - pos)
            self.lead_sum[j] += float(arr[:k].sum())
            self.lead_len[j] += k
            arr = arr[k:]
        elif self.part_len[j]:
            k = min(arr.size, m - int(self.part_len[j]))
            self.part_sum[j] += float(arr[:k].sum())
            self.part_len[j] += k
            if self.part_len[j] == m:
                self.ss[j] += float(self.part_sum[j]) ** 2
                self.nb[j] += 1
                self.part_sum[j] = 0.0
                self.part_len[j] = 0
            arr = arr[k:]
        if arr.size == 0:
            return
        nfull = arr.size // m
        if nfull:
            blk = arr[: nfull * m].reshape(nfull, m).sum(axis=1)
            self.ss[j] += float(np.einsum("i,i->", blk, blk))
            self.nb[j] += nfull
        rest = arr[nfull * m:]
        if rest.size:
            self.part_sum[j] = float(rest.sum())
            self.part_len[j] = rest.size

    def merge(self, other: "SecondOrderImage") -> "SecondOrderImage":
        """A new image of both records, the other's samples following this
        one's. Exact: the lagged sums gain the cross terms between this
        image's tail carry and the other's head carry.

        Args:
            other: the image of the record immediately following this one's,
                on the same budgets and sample grid.

        Raises:
            ValueError: different budgets or sample grids, or records that
                are not consecutive (``other.t0 != self.t0 + self.n``).
        """
        if (self.lag, self.nfreq, self.scales) != (
            other.lag,
            other.nfreq,
            other.scales,
        ):
            raise ValueError("images must have the same lag, nfreq and scales")
        if not (
            np.isclose(self.dx, other.dx, rtol=1e-12, atol=0.0)
            and np.isclose(self.x0, other.x0, rtol=0.0,
                           atol=1e-12 * abs(self.dx))
        ):
            raise ValueError("images must share the same sample grid")
        if other.t0 != self.t0 + self.n:
            raise ValueError(
                f"images must be consecutive: the second starts at index "
                f"{other.t0}, the first ends at {self.t0 + self.n}"
            )
        if self.n == 0 or other.n == 0:
            src = other if self.n == 0 else self
            out = SecondOrderImage(
                self.lag, self.nfreq, self.scales,
                t0=self.t0, x0=self.x0, dx=self.dx,
            )
            out.restore(src.state())
            out.t0 = self.t0
            return out
        lag = self.lag
        out = SecondOrderImage(
            lag, self.nfreq, self.scales, t0=self.t0, x0=self.x0, dx=self.dx
        )
        out.n = self.n + other.n
        out.sum_y = self.sum_y + other.sum_y
        out.sum_ty = self.sum_ty + other.sum_ty
        out.sum_t = self.sum_t + other.sum_t
        out.sum_t2 = self.sum_t2 + other.sum_t2
        out.residues = self.residues + other.residues

        left, right = self.tail, other.head
        out.c_y = self.c_y + other.c_y + _cross_sums(left, right[:lag], lag)
        out.c_sq = (
            self.c_sq + other.c_sq
            + _cross_sums(left ** 2, right[:lag] ** 2, lag)
        )
        # the increment sequences join through the boundary increment
        b0 = float(right[0] - left[-1])
        a_d = np.diff(left)
        c_d = np.diff(right)[:lag]
        joined = np.concatenate([[b0], c_d])
        extra = np.zeros(lag + 1)
        extra[0] = b0 * b0
        out.c_dy = (
            self.c_dy
            + other.c_dy
            + extra
            + _cross_sums(a_d, joined, lag)
            + _cross_sums(np.array([b0]), c_d, lag)
        )
        extra2 = np.zeros(lag + 1)
        extra2[0] = b0 ** 4
        out.c_d2 = (
            self.c_d2
            + other.c_d2
            + extra2
            + _cross_sums(a_d * a_d, joined * joined, lag)
            + _cross_sums(np.array([b0 * b0]), c_d * c_d, lag)
        )

        out.ss = self.ss + other.ss
        out.nb = self.nb + other.nb
        out.lead_sum = self.lead_sum.copy()
        out.lead_len = self.lead_len.copy()
        out.part_sum = other.part_sum.copy()
        out.part_len = other.part_len.copy()
        for j in range(self.scales + 1):
            m = 1 << j
            need = (-self.t0) % m
            if self.n < need:
                # this image never reached a block boundary: its lead and the
                # other's lead are one open leading block.
                out.lead_sum[j] = self.lead_sum[j] + other.lead_sum[j]
                out.lead_len[j] = self.lead_len[j] + other.lead_len[j]
                continue
            jl = int(self.part_len[j]) + int(other.lead_len[j])
            js = float(self.part_sum[j]) + float(other.lead_sum[j])
            if jl == m:
                out.ss[j] += js * js
                out.nb[j] += 1
            elif jl > 0:
                out.part_sum[j] = js
                out.part_len[j] = jl

        out.head = self.head if self.head.size >= lag + 1 else np.concatenate(
            [self.head, other.head]
        )[: lag + 1]
        out.tail = np.concatenate([self.tail, other.tail])[-(lag + 1):]
        return out

    # ------------------------------------------------------------ carries
    def first(self) -> float:
        """The first sample of the record.

        Raises:
            ValueError: the image holds no samples.
        """
        if self.n == 0:
            raise ValueError("the image holds no samples")
        return float(self.head[0])

    def last(self) -> float:
        """The last sample of the record, the forecast anchor.

        Raises:
            ValueError: the image holds no samples.
        """
        if self.n == 0:
            raise ValueError("the image holds no samples")
        return float(self.tail[-1])

    def carry(self) -> np.ndarray:
        """The last ``lag + 1`` samples, oldest first; empty before the first
        :meth:`update`, shorter than ``lag + 1`` while the record is."""
        return self.tail.copy()

    @property
    def domain(self) -> tuple[float, float]:
        """``(x0, x1)``, the positions of the first and last samples."""
        return (
            self.x0 + self.dx * self.t0,
            self.x0 + self.dx * (self.t0 + max(self.n, 1) - 1),
        )

    # ------------------------------------------------------------ read-outs
    def mean(self) -> float:
        """The sample mean of the record.

        Raises:
            ValueError: the image holds no samples.
        """
        if self.n == 0:
            raise ValueError("the image holds no samples")
        return self.sum_y / self.n

    def acov(self) -> np.ndarray:
        """Centred sample autocovariance ``gamma[0..lag]``, divided by ``n``.

        The lag-``k`` entry is ``(1/n) sum_{t>=k} (y_t - m)(y_{t-k} - m)``,
        the biased (positive definite) form, exact through the carries.
        """
        return _centred_acov(
            self.c_y, self.sum_y, self.n, self.head, self.tail)

    def acov_increments(self) -> np.ndarray:
        """Centred autocovariance of the increments ``dy_t = y_t - y_{t-1}``,
        length ``lag + 1``, divided by the increment count ``n - 1``.

        Raises:
            ValueError: the image holds no samples.
        """
        d_head = np.diff(self.head)
        d_tail = np.diff(self.tail)
        s = self.last() - self.first()
        return _centred_acov(self.c_dy, s, self.n - 1, d_head, d_tail)

    def acov_volatility(self) -> np.ndarray:
        """Centred autocovariance of the squared increments, the volatility
        functional of the returns, divided by ``n - 1``."""
        d_head = np.diff(self.head) ** 2
        d_tail = np.diff(self.tail) ** 2
        return _centred_acov(
            self.c_d2, float(self.c_dy[0]), self.n - 1, d_head, d_tail
        )

    def acov_squares(self) -> np.ndarray:
        """Centred autocovariance of the squared level, the volatility
        functional of a stationary series, divided by ``n``."""
        return _centred_acov(
            self.c_sq, float(self.c_y[0]), self.n,
            self.head ** 2, self.tail ** 2,
        )

    def trend(self) -> tuple[float, float]:
        """Least-squares line ``y ~ intercept + slope * x`` over the record,
        in position units. Exact from the stored cross-sums.

        Raises:
            ValueError: the image holds no samples.
        """
        n = self.n
        d = n * self.sum_t2 - self.sum_t ** 2
        if d <= 0:
            return 0.0, self.mean()
        slope_i = (n * self.sum_ty - self.sum_t * self.sum_y) / d
        icpt_i = (self.sum_y - slope_i * self.sum_t) / n
        return slope_i / self.dx, icpt_i - slope_i * self.x0 / self.dx

    def detrended_var(self) -> float:
        """Residual variance of the record about its least-squares line,
        ``RSS / n``. Exact from the cross-sums.

        Raises:
            ValueError: the image holds no samples.
        """
        n = self.n
        slope_i, _ = self._trend_index()
        my, mt = self.sum_y / n, self.sum_t / n
        syy = self.c_y[0] - n * my * my
        rss = syy - slope_i * (self.sum_ty - n * mt * my)
        return max(rss, 0.0) / n

    def _trend_index(self) -> tuple[float, float]:
        """The trend in index units: ``y ~ icpt + slope * t``."""
        n = self.n
        d = n * self.sum_t2 - self.sum_t ** 2
        if d <= 0:
            return 0.0, self.mean()
        slope = (n * self.sum_ty - self.sum_t * self.sum_y) / d
        return slope, (self.sum_y - slope * self.sum_t) / n

    def aggregated_variance(self) -> tuple[np.ndarray, np.ndarray]:
        """Block-mean variance against block size.

        Returns ``(m, var)`` over the scales with at least eight complete
        blocks and a strictly positive variance, ``m`` the block size in
        samples and ``var`` the unbiased variance of the means of the complete
        blocks at that scale. Both arrays are empty when no scale qualifies.
        """
        ms, vs = [], []
        for j in range(self.scales + 1):
            m = 1 << j
            nb = int(self.nb[j])
            if nb < 8:
                continue
            # the mean of the complete blocks, leaving out the samples
            # outside them
            total = self.sum_y - self.lead_sum[j] - self.part_sum[j]
            var = (self.ss[j] / nb - (total / nb) ** 2) / (m * m)
            var *= nb / (nb - 1.0)
            if var > 0.0:
                ms.append(float(m))
                vs.append(float(var))
        return np.asarray(ms), np.asarray(vs)

    def dft(self) -> np.ndarray:
        """The DFT of the record on the fixed grid ``f_m = m / (2 nfreq)``,
        ``m = 0..nfreq-1``, with the absolute sample index as the phase
        reference, so block images add."""
        return np.fft.rfft(self.residues)[: self.nfreq]

    def spectrum(
        self, n_freq: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blackman-Tukey spectrum from the autocovariances under a Parzen lag
        window.

        Args:
            n_freq: number of frequencies, at ``f = 1/n .. n_freq/n`` cycles
                per sample; ``None`` takes ``min(n // 2, 4096)``. The read-out
                allocates a ``(lag, n_freq)`` cosine matrix, so a large
                ``n_freq`` costs more memory than the image itself.

        Returns:
            ``(f, S)``, the frequencies and the spectral density.
        """
        g = self.acov()
        lag = self.lag
        k = np.arange(lag + 1)
        u = k / lag
        w = np.where(u <= 0.5, 1 - 6 * u ** 2 + 6 * u ** 3, 2 * (1 - u) ** 3)
        nf = int(n_freq or min(self.n // 2, 4096))
        f = np.arange(1, nf + 1) / self.n
        s = g[0] + 2 * (w[1:] * g[1:]) @ np.cos(2 * np.pi * np.outer(k[1:], f))
        return f, s

    def detrended_acov(self) -> np.ndarray:
        """Centred autocovariance of the residual about the least-squares
        line, ``gamma_e[0..lag]`` divided by ``n``. Exact from the stored
        cross-sums and the carries.

        Raises:
            ValueError: the image holds no samples.
        """
        n, k = self.n, np.arange(self.lag + 1)
        sl, ic = self._trend_index()
        ic = ic + sl * self.t0    # local intercept: y ~ ic + sl*u, u = t - t0
        hy = np.concatenate([[0.0], np.cumsum(self.head)])
        ty = np.concatenate([[0.0], np.cumsum(self.tail[::-1])])
        hu = np.concatenate(
            [[0.0], np.cumsum(self.head * np.arange(self.head.size))]
        )
        upos = n - 1 - np.arange(self.tail.size)
        tu = np.concatenate([[0.0], np.cumsum(self.tail[::-1] * upos)])
        kh = np.minimum(k, self.head.size)
        kt = np.minimum(k, self.tail.size)
        sum_uy = self.sum_ty - self.t0 * self.sum_y   # sum u y_u, u = t - t0
        a = self.sum_y - hy[kh]                 # sum_{u>=k} y_u
        b = self.sum_y - ty[kt]                 # sum_{u<=n-1-k} y_u
        c = sum_uy - hu[kh]                     # sum_{u>=k} u y_u
        d = sum_uy - tu[kt]                     # sum_{u<=n-1-k} u y_u
        cnt = np.maximum(n - k, 0)
        # sum over the overlap of (ic + sl u)(ic + sl (u - k)); u = t - t0
        # keeps the index moments O(n^3) regardless of how far t0 sits
        # down the stream, so no cancellation occurs at large t0
        lo = k.astype(float)
        hi = float(n - 1)
        s1 = np.where(cnt > 0, (lo + hi) * cnt / 2.0, 0.0)
        s2 = np.where(
            cnt > 0,
            (hi * (hi + 1.0) * (2.0 * hi + 1.0)
             - (lo - 1.0) * lo * (2.0 * lo - 1.0)) / 6.0,
            0.0,
        )
        quad = (
            ic * ic * cnt
            + ic * sl * (2 * s1 - k * cnt)
            + sl * sl * (s2 - k * s1)
        )
        out = (
            self.c_y
            - (ic * a + sl * (c - k * a))
            - (ic * b + sl * (d + k * b))
            + quad
        )
        return out / n

    def residual_acov(
        self, freq: float | None = None, coef: np.ndarray | None = None
    ) -> np.ndarray:
        """Autocovariance of the residual about the deterministic mean: the
        least-squares line, and the harmonics of ``freq`` with coefficients
        ``coef`` when both are given.

        Args:
            freq: fundamental frequency in cycles per sample, or ``None`` for
                no seasonal part.
            coef: ``(2K,)`` cosine/sine pairs at ``freq, 2*freq, ..``, the
                layout :func:`seasonal` returns.

        The trend part is exact; the seasonal part subtracts the harmonics'
        own autocovariance ``sum_j (A_j^2 / 2) cos(2 pi j f k)`` and neglects
        the sample cross-covariance between the harmonics and the residual,
        which is ``O(1/sqrt(n))``.

        Raises:
            ValueError: the image holds no samples.
        """
        g = self.detrended_acov()
        if freq is None or coef is None or len(coef) == 0:
            return g
        k = np.arange(self.lag + 1)
        coef = np.asarray(coef, dtype=float)
        for j in range(coef.size // 2):
            amp2 = coef[2 * j] ** 2 + coef[2 * j + 1] ** 2
            g = g - 0.5 * amp2 * np.cos(2 * np.pi * (j + 1) * freq * k)
        return g

    def _line_dft(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """DFTs of the sequences ``1`` and ``t`` over the record's indices at
        the frequencies ``f`` (cycles per sample)."""
        n, t0 = self.n, self.t0
        w = np.exp(-2j * np.pi * f)
        one = np.isclose(np.abs(w - 1.0), 0.0, atol=1e-14)
        wn = w ** n
        den = np.where(one, 1.0, 1.0 - w)
        d0 = np.where(one, float(n), w ** t0 * (1.0 - wn) / den)
        s1 = np.where(
            one,
            n * (n - 1) / 2.0,
            w * (1.0 - n * w ** (n - 1) + (n - 1) * wn) / (den * den),
        )
        d1 = t0 * d0 + np.where(one, s1, w ** t0 * s1)
        return d0, d1

    def _dirichlet(self, f: float, fj: np.ndarray) -> np.ndarray:
        """DFT at the grid frequencies ``fj`` of ``exp(2 pi i f t)`` over the
        record's indices."""
        d = fj - f
        n, t0 = self.n, self.t0
        w = np.exp(-2j * np.pi * d)
        one = np.abs(w - 1.0) < 1e-12
        den = np.where(one, 1.0, 1.0 - w)
        return np.where(one, float(n), (1.0 - w ** n) / den) * np.exp(
            -2j * np.pi * d * t0
        )

    def _harmonic_fit(
        self, f: float, n_harm: int, bins: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Real-constrained least squares of ``n_harm`` harmonics of ``f`` to
        the DFT values ``target`` at the grid bins ``bins``; returns the
        cosine/sine coefficients and the residual sum of squares."""
        fs = bins
        cols = []
        for j in range(1, n_harm + 1):
            dp = self._dirichlet(j * f, fs)
            dm = self._dirichlet(-j * f, fs)
            cols.append(0.5 * (dp + dm))
            cols.append(-0.5j * (dp - dm))
        mat = np.column_stack(cols)
        design = np.vstack([mat.real, mat.imag])
        rhs = np.concatenate([target.real, target.imag])
        coef, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        rss = float(np.sum((rhs - design @ coef) ** 2))
        return coef, rss

    def seasonal(
        self, max_harmonics: int = 1, *, freq: float | None = None,
        halfwidth: int = 4
    ) -> dict[str, Any]:
        """The seasonal component from the fixed-grid DFT.

        The trend's own DFT is subtracted in closed form, the largest
        remaining bin locates the fundamental, and the frequency and the
        harmonic coefficients are fitted to the bins around each harmonic by
        the Dirichlet-kernel (leakage-model) real-constrained least squares.
        The harmonic count is chosen by BIC up to ``max_harmonics``.

        Args:
            max_harmonics: largest number of harmonics to consider, at
                least 1.
            freq: fundamental frequency in cycles per sample; ``None``
                locates and refines it from the largest bin.
            halfwidth: bins on each side of a harmonic that enter the fit.

        Returns:
            ``{"period", "freq", "amp", "phase", "coef", "n_harmonics",
            "strength"}``: the fundamental period in samples, its frequency
            in cycles per sample, the fundamental amplitude and phase of
            ``A sin(2 pi f t + phase)``, the ``(2K,)`` cosine/sine
            coefficients at ``f, 2f, ..``, the chosen ``K``, and the share
            ``A^2 / (2 var)`` of the detrended variance the fundamental
            carries. The phase reference is the global sample index ``t``,
            the one ``t0`` counts from, so ``coef`` and ``phase`` describe the
            record wherever it sits in the stream. The frequency grid resolves
            ``1 / (2 nfreq)``; a record longer than ``2 nfreq`` samples puts
            several of its own frequencies inside one grid bin, which widens
            the error of the refined frequency.

        Non-finite ``period`` and zero ``amp`` come back when the record is
        too short for an interior bin (``n < 8``).

        Raises:
            ValueError: the image holds no samples.

        Warns:
            UserWarning: the record is longer than ``2 nfreq`` samples, so
                the frequency grid is coarser than one bin per record
                frequency.
        """
        from scipy.optimize import minimize_scalar

        m = self.nfreq
        fj = np.arange(m) / (2.0 * m)
        sl, ic = self._trend_index()
        d0, d1 = self._line_dft(fj)
        fd = self.dft() - (ic * d0 + sl * d1)
        empty = {
            "period": float("inf"), "freq": 0.0, "amp": 0.0, "phase": 0.0,
            "coef": np.zeros(0), "n_harmonics": 0, "strength": 0.0,
        }
        if self.n < 8 or m < 3:
            return empty
        if self.n > 2 * m:
            warnings.warn(
                f"the frequency grid resolves 1/{2 * m} cycles per sample and "
                f"the record is {self.n} samples, so several of its "
                f"frequencies share a bin; the period and amplitude are "
                f"read at that resolution. Raise nfreq.",
                UserWarning, stacklevel=2)
        # the fundamental must complete at least two cycles inside the record
        lo_bin = max(1, int(np.ceil(4.0 * m / self.n)))
        if lo_bin >= m - 1:
            return empty
        if freq is None:
            peak = lo_bin + int(np.argmax(np.abs(fd[lo_bin:m - 1])))
        else:
            peak = int(np.clip(round(2.0 * m * freq), lo_bin, m - 2))
        var = self.detrended_var()
        if freq is None:
            sel = slice(max(0, peak - halfwidth), min(m, peak + halfwidth + 1))
            res = minimize_scalar(
                lambda f: self._harmonic_fit(f, 1, fj[sel], fd[sel])[1],
                bounds=(fj[max(1, peak - 1)], fj[min(m - 1, peak + 1)]),
                method="bounded",
            )
            freq = float(res.x)
        freq = float(freq)
        if not 0.0 < freq < 0.5:
            return empty
        best = None
        for k in range(1, int(max_harmonics) + 1):
            if (k + 1) * freq >= 0.5 and k > 1:
                break
            keep = np.zeros(m, dtype=bool)
            for j in range(1, k + 1):
                c = int(round(2.0 * m * j * freq))
                if c >= m:
                    break
                keep[max(0, c - halfwidth):min(m, c + halfwidth + 1)] = True
            bins = fj[keep]
            coef, _ = self._harmonic_fit(freq, k, bins, fd[keep])
            power = 0.5 * float(coef @ coef)
            if power > 2.0 * var:
                # harmonics carrying more energy than the record holds are
                # cancelling one another, not decomposing it
                continue
            rss = max(self.n * (var - power), 1e-12 * self.n)
            bic = self.n * np.log(rss / self.n) + 2 * k * np.log(self.n)
            if best is None or bic < best[0]:
                best = (bic, k, coef, power)
        if best is None:
            return empty
        _, n_harm, coef, power = best
        a, b = float(coef[0]), float(coef[1])
        amp = float(np.hypot(a, b))
        # the fundamental's energy share, off the leakage-corrected amplitude
        # rather than off the peak bin, which an off-grid line under-reads
        strength = 0.5 * amp * amp / var if var > 0 else 0.0
        return {
            "period": 1.0 / freq if freq > 0 else float("inf"),
            "freq": freq,
            "amp": amp,
            "phase": float(np.arctan2(a, b)),
            "coef": coef,
            "n_harmonics": int(n_harm),
            "strength": float(strength),
        }

    @overload
    def dickey_fuller(
        self, lags: int | None = None, *, return_lag: Literal[False] = False
    ) -> float: ...

    @overload
    def dickey_fuller(
        self, lags: int | None = None, *, return_lag: Literal[True]
    ) -> tuple[float, int]: ...

    def dickey_fuller(
        self, lags: int | None = None, *, return_lag: bool = False
    ) -> float | tuple[float, int]:
        """The augmented Dickey-Fuller ``tau`` statistic of the constant plus
        trend regression, computed from the autocovariances.

        The normal equations of the regression ``dy_t = rho y_{t-1} +
        sum_j d_j dy_{t-j} + c + b t`` are written in Toeplitz form, so the
        statistic reads off the image; the deterministic columns enter as the
        centring of the autocovariances. Every candidate lag count reads the
        same autocovariance sequence, so all of them see the same effective
        sample; there is no separate windowing to redo per candidate.

        Args:
            lags: difference lags in the regression; ``None`` selects the
                lag count by AIC (``n * log(rss / n) + 2 * k``, ``k =
                1 + p`` the level and difference regressors; the constant
                and trend are centered out of the autocovariances, not
                counted) over ``0..maxlag`` with ``maxlag =
                min(12 (n/100)^0.25, 12, n // 3, lag - 2)``. A value given
                fixes the lag instead, clamped into ``[0, lag - 2]``.
            return_lag: also return the lag count the regression used.

        Returns:
            The ``tau`` statistic, or ``(tau, p)`` when ``return_lag`` is
            set; ``tau`` is ``nan`` (``p`` the requested or largest
            candidate lag) when the normal equations are singular. The
            standard error divides the residual variance by ``n`` rather
            than the regression's usable sample ``n - p - 2``, so ``tau``
            reads about ``sqrt(n / (n - p - 2))`` too large in magnitude --
            22 percent at ``n = 40, p = 12``, the unit-root gate's floor.

        Raises:
            ValueError: the image holds no samples.
        """
        g = self.detrended_acov()
        n, lag = self.n, self.lag

        def gd(k: int) -> float:
            k = abs(k)
            if k + 1 > lag:
                return 0.0
            return float(2 * g[k] - g[abs(k - 1)] - g[k + 1])

        def solve(p: int) -> tuple[np.ndarray, float, float] | None:
            dim = 1 + p
            a = np.zeros((dim, dim))
            b = np.zeros(dim)
            a[0, 0] = g[0]
            b[0] = g[1] - g[0]
            for j in range(1, p + 1):
                a[0, j] = a[j, 0] = g[j - 1] - g[j] if j <= lag else 0.0
                b[j] = gd(j)
                for i in range(1, p + 1):
                    a[i, j] = gd(i - j)
            try:
                coef = np.linalg.solve(a, b)
                inv00 = float(np.linalg.inv(a)[0, 0])
            except np.linalg.LinAlgError:
                return None
            resid_var = gd(0) - float(coef @ b)
            return coef, resid_var, inv00

        if lags is not None:
            p = max(0, min(int(lags), lag - 2))
            best = solve(p)
        else:
            maxlag = max(
                0, int(min(12.0 * (n / 100.0) ** 0.25, 12, lag - 2, n // 3))
            )
            p, best, best_ic = maxlag, None, float("inf")
            for cand in range(0, maxlag + 1):
                got = solve(cand)
                # a non-positive closed-form residual variance is a
                # numerical degeneracy, not a genuine fit: its log
                # diverges and would always win the comparison
                if got is None or got[1] <= 0.0:
                    continue
                ic = n * np.log(got[1]) + 2 * (1 + cand)
                if ic < best_ic:
                    p, best, best_ic = cand, got, ic

        if best is None:
            result = (float("nan"), p)
        else:
            coef, resid_var, inv00 = best
            se = np.sqrt(max(resid_var, 1e-12) / n * max(inv00, 1e-30))
            tau = float(coef[0] / se) if se > 0 else float("nan")
            result = (tau, p)
        return result if return_lag else result[0]

    # ------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        """The complete state as JSON-serializable data."""
        return {
            "version": _STATE_VERSION,
            "lag": self.lag,
            "nfreq": self.nfreq,
            "scales": self.scales,
            "t0": self.t0,
            "x0": self.x0,
            "dx": self.dx,
            "n": self.n,
            "sum_y": self.sum_y,
            "sum_t": self.sum_t,
            "sum_t2": self.sum_t2,
            "sum_ty": self.sum_ty,
            "c_y": self.c_y.tolist(),
            "c_dy": self.c_dy.tolist(),
            "c_d2": self.c_d2.tolist(),
            "c_sq": self.c_sq.tolist(),
            "ss": self.ss.tolist(),
            "nb": self.nb.tolist(),
            "lead_sum": self.lead_sum.tolist(),
            "lead_len": self.lead_len.tolist(),
            "part_sum": self.part_sum.tolist(),
            "part_len": self.part_len.tolist(),
            "residues": self.residues.tolist(),
            "head": self.head.tolist(),
            "tail": self.tail.tolist(),
        }

    def restore(self, d: dict[str, Any]) -> "SecondOrderImage":
        """Load a :meth:`state` into this image and return it.

        Args:
            d: a dict as returned by :meth:`state`, on the same budgets.

        Raises:
            ValueError: an unknown state version or different budgets.
        """
        if d.get("version") != _STATE_VERSION:
            raise ValueError("unknown second-order image state version")
        if (d["lag"], d["nfreq"], d["scales"]) != (
            self.lag, self.nfreq, self.scales
        ):
            raise ValueError("state budgets differ from this image's")
        self.t0 = int(d["t0"])
        self.x0 = float(d["x0"])
        self.dx = float(d["dx"])
        self.n = int(d["n"])
        self.sum_y = float(d["sum_y"])
        self.sum_t = int(d["sum_t"])
        self.sum_t2 = int(d["sum_t2"])
        self.sum_ty = float(d["sum_ty"])
        self.c_y = np.asarray(d["c_y"], dtype=float)
        self.c_dy = np.asarray(d["c_dy"], dtype=float)
        self.c_d2 = np.asarray(d["c_d2"], dtype=float)
        self.c_sq = np.asarray(d["c_sq"], dtype=float)
        self.ss = np.asarray(d["ss"], dtype=float)
        self.nb = np.asarray(d["nb"], dtype=np.int64)
        self.lead_sum = np.asarray(d["lead_sum"], dtype=float)
        self.lead_len = np.asarray(d["lead_len"], dtype=np.int64)
        self.part_sum = np.asarray(d["part_sum"], dtype=float)
        self.part_len = np.asarray(d["part_len"], dtype=np.int64)
        self.residues = np.asarray(d["residues"], dtype=float)
        self.head = np.asarray(d["head"], dtype=float)
        self.tail = np.asarray(d["tail"], dtype=float)
        return self
