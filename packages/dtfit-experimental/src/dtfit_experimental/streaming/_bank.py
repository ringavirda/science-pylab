"""FilterBank: a parallel array of independent streaming filters.

Many real-time problems are not a single stream but a bank of them: the
pseudorange from each GPS satellite, the outputs of a MIMO plant, the channels
of a sensor array, the axes of a trajectory. Each is tracked by its own
:class:`~dtfit.streaming.EACFilter` or :class:`~dtfit.streaming.LSIFilter`, and
because the streams are independent the whole bank fans across CPU cores.

The composition is thin: a :class:`FilterBank` holds K filters and routes
samples to them. Any thread speed-up comes from SciPy's compiled linear
algebra (``solve_triangular``'s LAPACK call in the filter's hot loop)
releasing the GIL on large enough windows, so workers updating different
filters overlap that work without any pickling; small windows leave the
GIL held throughout and gain nothing from threads. This is the streaming
counterpart of :func:`dtfit.fit_many`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2

from dtfit.streaming import EACFilter

__all__ = ["FilterBank", "FusedChiSquareDetector"]


def _drive_streams(
    recipe: tuple[Any, str, str, dict[str, Any]],
    t_seq: np.ndarray,
    y_cols: list[np.ndarray],
    track: bool,
) -> tuple[list[np.ndarray], list[int], list[list[float]]]:
    """Process-pool worker: rebuild a filter per assigned stream from the
    picklable ``recipe``, drive it over its column, and return final params,
    drift counts and optional per-step tracks. The compiled SymPy callables
    are never pickled; the worker recompiles them. Module-level to stay
    picklable.

    Each worker holds the GIL of its own interpreter, so the Python-level
    per-sample work (model evaluation, Kalman update) of different streams
    runs genuinely concurrently; the thread backend shares one GIL.
    """
    filter_cls, expr, var, kwargs = recipe
    params: list[np.ndarray] = []
    drifts: list[int] = []
    tracks: list[list[float]] = []
    for col in y_cols:
        flt = filter_cls(expr, var, **kwargs)
        nd = 0
        th: list[float] = []
        for s in range(t_seq.size):
            flt.partial_fit(float(t_seq[s]), float(col[s]))
            if getattr(flt, "drift_flag_", False):
                nd += 1
            if track:  # full-length column, NaN where there is no prediction,
                # letting the parent assign it by position without index drift
                if len(getattr(flt, "_t", [1])) > 0:
                    th.append(float(flt.predict(np.array([t_seq[s]]))[0]))
                else:
                    th.append(float("nan"))
        params.append(np.asarray(flt.p, dtype=float))
        drifts.append(nd)
        if track:
            tracks.append(th)
    return params, drifts, tracks


class FilterBank:
    """A bank of K independent streaming filters, updated in lockstep.

    Construct from explicit filters or, more commonly, from a model via
    :meth:`from_model`. Sample routing: at each step every stream gets its own
    ``y`` (and, optionally, its own ``t``).
    """

    def __init__(self, filters: Sequence[Any]) -> None:
        self.filters: list[Any] = list(filters)
        if not self.filters:
            raise ValueError("FilterBank needs at least one filter.")
        # Construction recipe for the process backend, set by ``from_model``;
        # only there are all streams identically configured and picklable. It
        # stays ``None`` for a bank built from explicit, possibly heterogeneous
        # filter objects.
        self._recipe: tuple[Any, str, str, dict[str, Any]] | None = None

    @classmethod
    def from_model(
        cls,
        expr: str,
        var: str,
        n_streams: int,
        *,
        filter_cls: type = EACFilter,
        **kwargs: Any,
    ) -> "FilterBank":
        """Build ``n_streams`` identically-configured filters for one model.

        Args:
            expr, var: Model expression and main variable (per stream).
            n_streams: Number of parallel streams (filters) in the bank.
            filter_cls: ``EACFilter`` (default) or
                ``LSIFilter``.
            **kwargs: Forwarded to each filter's constructor.
        """
        bank = cls([filter_cls(expr, var, **kwargs) for _ in range(n_streams)])
        bank._recipe = (filter_cls, expr, var, dict(kwargs))
        return bank

    def __len__(self) -> int:
        return len(self.filters)

    def __getitem__(self, i: int) -> Any:
        return self.filters[i]

    def partial_fit(
        self,
        t: float | np.ndarray,
        y: np.ndarray,
        *,
        n_jobs: int = 1,
    ) -> "FilterBank":
        """Ingest one sample per stream and update every filter in place.

        Args:
            t: Shared scalar time, or a per-stream array of length K.
            y: Per-stream observations, length K.
            n_jobs: Threads to fan the K updates across (``1`` = serial). A
                thread pool lets the GIL-released LAPACK solves overlap. For
                cheap per-step work serial is usually fastest; threading wins
                when K and the window are large (see :meth:`run`).
        """
        y = np.asarray(y, dtype=float)
        tv = (np.full(len(self.filters), float(t)) if np.ndim(t) == 0
              else np.asarray(t, float))
        if n_jobs == 1:
            for flt, ti, yi in zip(self.filters, tv, y):
                flt.partial_fit(float(ti), float(yi))
            return self
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            list(ex.map(
                lambda k: self.filters[k].partial_fit(
                    float(tv[k]), float(y[k])),
                range(len(self.filters)),
            ))
        return self

    def run(
        self,
        t_seq: np.ndarray,
        Y: np.ndarray,
        *,
        n_jobs: int = 1,
        track: bool = False,
        backend: str = "thread",
    ) -> dict[str, Any]:
        """Drive every stream over a whole block of samples.

        Streams are independent, so each worker takes a disjoint subset of
        them and runs it to completion. No per-step synchronization is what
        makes the bank scale.

        Args:
            t_seq: Time stamps, shape ``(n_steps,)`` (shared by all streams).
            Y: Observations, shape ``(n_steps, K)``; column k feeds filter k.
            n_jobs: Worker count (``1`` = serial).
            track: If True, also return the per-stream, per-step prediction
                of the current sample (``(n_steps, K)``); costs
                O(n_steps*K) memory.
            backend: ``"thread"`` (default) fans the K streams across worker
                threads. That speeds up only the GIL-released LAPACK solve
                work, leaving the Python-level filter loop usually no faster
                than serial. ``"process"`` runs disjoint stream subsets in
                separate interpreters, each with its own GIL, so the per-sample
                Python work runs genuinely concurrently. It wins on many
                streams by long records, where per-process compute dwarfs the
                spawn and pickling overhead. It needs a bank built via
                :meth:`from_model` for the workers to reconstruct the filters,
                and falls back to threads otherwise.

        Returns:
            ``params``: ``(K, n_params)`` final estimates; ``n_drifts``:
            per-stream drift counts; ``track`` (optional): tracking history.
        """
        t_seq = np.asarray(t_seq, dtype=float)
        Y = np.asarray(Y, dtype=float)
        n_steps, K = Y.shape
        if K != len(self.filters):
            raise ValueError(
                f"Y has {K} columns but bank holds {len(self.filters)} "
                "filters."
            )

        if backend == "process" and n_jobs > 1 and self._recipe is not None:
            return self._run_process(t_seq, Y, n_jobs=n_jobs, track=track)

        track_hist = np.full((n_steps, K), np.nan) if track else None
        drifts = np.zeros(K, dtype=int)

        def drive(k: int) -> None:
            flt = self.filters[k]
            col = Y[:, k]
            for s in range(n_steps):
                flt.partial_fit(t_seq[s], col[s])
                if getattr(flt, "drift_flag_", False):
                    drifts[k] += 1
                if (track_hist is not None
                        and len(getattr(flt, "_t", [1])) > 0):
                    track_hist[s, k] = float(
                        flt.predict(np.array([t_seq[s]]))[0])

        if n_jobs == 1:
            for k in range(K):
                drive(k)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                list(ex.map(drive, range(K)))

        out: dict[str, Any] = {
            "params": self.params_array(),
            "n_drifts": drifts,
        }
        if track_hist is not None:
            out["track"] = track_hist
        return out

    def _run_process(
        self, t_seq: np.ndarray, Y: np.ndarray, *, n_jobs: int, track: bool
    ) -> dict[str, Any]:
        """Process-pool backend for :meth:`run` (see its ``backend`` docs)."""
        # Only reached from ``run`` when a recipe is set, that being what the
        # workers rebuild the filters from. Bind it locally to make it
        # non-Optional.
        recipe = self._recipe
        assert recipe is not None
        n_steps, K = Y.shape
        groups = [g.tolist() for g in np.array_split(np.arange(K), n_jobs)
                  if len(g)]
        n_params = self.filters[0].p.size
        params = np.zeros((K, n_params))
        drifts = np.zeros(K, dtype=int)
        track_hist = np.full((n_steps, K), np.nan) if track else None
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = {
                ex.submit(_drive_streams, recipe, t_seq,
                          [Y[:, k] for k in g], track): g
                for g in groups
            }
            for fut, g in futs.items():
                ps, ds, ths = fut.result()
                for local, k in enumerate(g):
                    params[k] = ps[local]
                    drifts[k] = ds[local]
                    # keep bank readout consistent
                    self.filters[k].p = ps[local]
                    if track_hist is not None:
                        track_hist[:, k] = ths[local]
        out: dict[str, Any] = {"params": params, "n_drifts": drifts}
        if track_hist is not None:
            out["track"] = track_hist
        return out

    def params_array(self) -> np.ndarray:
        """Current estimates as ``(K, n_params)``."""
        return np.array([flt.p for flt in self.filters], dtype=float)

    @property
    def params_(self) -> list[dict[str, float]]:
        """Per-stream ``{name: value}`` parameter mappings."""
        return [flt.params_ for flt in self.filters]

    @property
    def drift_flags_(self) -> np.ndarray:
        """Per-stream drift flag from the most recent update, shape
        ``(K,)``."""
        return np.array(
            [bool(getattr(flt, "drift_flag_", False)) for flt in self.filters]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Per-stream prediction. Returns ``(K,)`` for scalar-like ``x`` else
        ``(K, len(x))``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        preds = np.array(
            [np.asarray(flt.predict(x), dtype=float) for flt in self.filters]
        )
        return preds[:, 0] if x.size == 1 else preds

    def fused_detector(self, **kwargs: Any) -> "FusedChiSquareDetector":
        """A :class:`FusedChiSquareDetector` driving this bank (see its
        docs)."""
        return FusedChiSquareDetector(self, **kwargs)


class FusedChiSquareDetector:
    """Pool a :class:`FilterBank`'s one-step innovations into a fused fault
    test.

    A fault that moves every stream (a damping fault on all axes of an
    oscillator, a regime shift in a sensor array) leaves only a weak signature
    in any single stream's innovation but a strong one in the sum across
    streams. This detector sums each filter's own ``nis_`` (approximately a
    chi-square statistic with ``n_coef`` degrees of freedom under the
    model) into one fused ``chi2(sum n_coef)`` statistic. Passing the
    ``alpha``-level threshold flags a synchronized multi-axis fault that
    any single stream's innovation would miss, and optionally re-arms
    each filter via :meth:`~dtfit.EACFilter.inflate`.

    Usage::

        bank = FilterBank.from_model(model, "t", n_axes,
                                     filter_cls=LSIFilter, ...)
        det = bank.fused_detector(alpha=1e-4, inflate=4.0)
        for i, (t, y) in enumerate(stream):     # y is length-K
            if det.update(t, y):
                handle_fault(i, det.statistic_)

    Args:
        bank: The :class:`FilterBank` to drive. Its filters must expose
            ``nis_``, ``basis.n_coef``, ``W`` and :meth:`inflate`; both
            stock ones do.
        alpha: Per-step false-alarm probability; the threshold is
            ``chi2.ppf(1 - alpha, df=sum(n_coef))`` over the bank's filters.
        inflate: Covariance re-arm factor applied to every filter on a
            detection (``<= 1`` disables the re-arm; the flag is still
            raised).
        warmup: Steps to wait before detecting. Defaults to ``3 * window``,
            long enough for the filters to settle.
        cooldown: Steps to suppress detection after a flag. Defaults to one
            ``window``, so a single fault is not re-flagged every step.
    """

    def __init__(
        self,
        bank: FilterBank,
        *,
        alpha: float = 1e-4,
        inflate: float = 4.0,
        warmup: int | None = None,
        cooldown: int | None = None,
    ) -> None:
        self.bank = bank
        self.k = len(bank.filters)
        df = sum(f.basis.n_coef for f in bank.filters)
        self.threshold_ = float(chi2.ppf(1.0 - alpha, df=df))
        self.inflate_factor = float(inflate)
        # W is the filter's window cap. An adaptive-window filter starts at
        # min_window and grows, so the default ``3*W`` warmup is conservative:
        # it delays first detection but never causes a false positive. Pass an
        # explicit ``warmup`` sized to ``min_window`` to arm the detector
        # sooner on an adaptive bank.
        w = int(getattr(bank.filters[0], "W", 1))
        self._warmup = 3 * w if warmup is None else int(warmup)
        self._cooldown_len = w if cooldown is None else int(cooldown)
        self._step = -1   # raw stream index of the current sample
        self._seen = 0    # steps with a full (finite-nis_) window
        self._cool = 0
        self.statistic_ = float("nan")
        self.flag_ = False
        self.flags_: list[int] = []

    def update(self, t: float | np.ndarray, y: np.ndarray) -> bool:
        """Ingest one sample per stream and test for a fused fault.

        Updates the bank in place, then returns ``True`` iff this step raises a
        fault flag. ``flags_`` records the zero-based stream step indices that
        fired, counting every :meth:`update` call including the warm-up steps
        before the filters' windows fill.
        """
        self._step += 1
        self.bank.partial_fit(t, y)
        self.flag_ = False
        nis = np.array(
            [getattr(f, "nis_", np.nan) for f in self.bank.filters]
        )
        if not np.all(np.isfinite(nis)):
            return False  # windows not yet full
        idx = self._step
        self._seen += 1
        self.statistic_ = float(nis.sum())
        if self._cool > 0:
            self._cool -= 1
            return False
        if self._seen < self._warmup:
            return False
        if self.statistic_ > self.threshold_:
            if self.inflate_factor > 1.0:
                for f in self.bank.filters:
                    f.inflate(self.inflate_factor)
            self.flag_ = True
            self.flags_.append(idx)
            self._cool = self._cooldown_len
            return True
        return False

    @property
    def n_flags_(self) -> int:
        """Number of fault flags raised so far."""
        return len(self.flags_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Per-stream prediction from the underlying bank."""
        return self.bank.predict(x)
