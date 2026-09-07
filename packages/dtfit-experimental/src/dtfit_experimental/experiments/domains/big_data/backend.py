"""Compute for the big-data domain: batch, streaming and distributed.

``big_data.ipynb`` imports this module as ``B`` and does the presentation. Pure
compute here, with no ``matplotlib``: every function returns numbers, arrays or
dicts the notebook renders.

It tests dtfit's map-reduce estimators end to end against the big-data toolkit
a practitioner actually reaches for, on the concerns that decide whether an
estimator survives at scale:

* exactness and accuracy across execution routes and model families;
* throughput and memory scaling against the established batched and streaming
  methods;
* numerical stability of the additive streaming reduction, naive float32
  against float64 against compensated Kahan, the concern that bites at
  10^8-10^9 samples;
* robustness and mergeability: the order-independent, variable-chunk,
  missing-data reduces a distributed or fault-tolerant pipeline requires;
* the online cost of the streaming filter against the established online
  estimators, recursive least squares and an incremental SGD net.

The dtfit methods under test exploit two properties of the empirical LSI
spectrum ``int y*phi_j``: it is linear across channels, so B channels project
in one GEMM ``S = D^T*(w*Y)``, and additive over the domain, so a stream
reduces chunk by chunk.

* whole-array GEMM (:func:`fit_resident`, through ``fit_lsi_batched`` and
  ``project_spectra``): all B channels' spectra in one BLAS matmul, for maximal
  throughput at O(N*B) memory with the data resident.
* fused streaming map-reduce (:func:`fit_streaming`, through
  ``PartitionedBatchLSI``): folds each chunk's ``(B, n_coef)`` partial
  integrals into an accumulator. One pass, flat O(B*order) memory, exact, and
  it handles streams larger than RAM.
* distributed reduce (:func:`fit_distributed`, through
  ``PartitionedBatchLSI.merge``): per-partition accumulators combined by an
  associative, order-independent ``merge``.
* the streaming filter (``EACFilter``), the online twin: an O(1)-per-sample
  recursive update tracking a model's parameters in bounded memory.

The established baselines are per-channel SciPy NLLS
(:func:`fit_per_channel_nlls`), vectorised polynomial lstsq
(:func:`poly_lstsq_batched`), scikit-learn ``SGDRegressor.partial_fit``
(:func:`sgd_incremental`) and recursive least squares (``bl.RLSPredictor``).
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dtfit.streaming import EACFilter

from dtfit_experimental.scale import (
    fit_lsi_batched, PartitionedBatchLSI, project_spectra,
)
from dtfit_experimental.experiments.common import fmt, metrics
from dtfit_experimental.experiments.common import baselines as bl
from dtfit_experimental.experiments.common import datasets as ltsf
from dtfit_experimental.experiments.domains.common import peak_memory

__all__ = [
    "DOMAIN", "SCENARIOS", "Panel",
    "fit_resident", "fit_streaming", "fit_distributed",
    "fit_per_channel_nlls", "poly_lstsq_batched", "sgd_incremental",
    "numpy_projection_vectorised",
    "param_err", "scenario_func", "time_call", "peak_memory", "metrics", "fmt",
    "scaling_memory", "memory_wall", "surrogate_trap", "numerics", "numerics_both",
    "robustness", "online_filter", "load_electricity", "realdata_projection",
    "HAS_SKLEARN",
]

DOMAIN = (0.0, 1.5)

try:                                            # optional baseline dependency
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# Multi-channel scenarios: realistic big-data panels, each carrying a different
# model shape and standing for a different concern.
def _g_exp(x, a, b):
    return a[None, :] * np.exp(np.outer(x, b))


def _g_decay(x, a, b):
    return a[None, :] * np.exp(-np.outer(x, b))


def _g_power(x, a, b):
    return a[None, :] * (x[:, None] + 1.0) ** b[None, :]


def _g_logistic(x, a, b):                                  # a=K (level), b=r (rate)
    return a[None, :] / (1.0 + np.exp(-b[None, :] * (x[:, None] - 0.75)))


SCENARIOS = [
    dict(key="exp_growth", expr="a*exp(b*t)", gen=_g_exp, p0=[1.0, 1.0],
         arange=(0.5, 2.0), brange=(0.2, 1.2),
         concern="sensor-drift / epidemic panel (growth)"),
    dict(key="exp_decay", expr="a*exp(-b*t)", gen=_g_decay, p0=[1.0, 1.0],
         arange=(1.0, 3.0), brange=(0.3, 1.5),
         concern="RC-discharge / relaxation array (decay)"),
    dict(key="power_law", expr="a*(t+1)**b", gen=_g_power, p0=[1.0, 1.0],
         arange=(0.5, 2.0), brange=(0.3, 1.6),
         concern="scaling-law panel (monotone)"),
    dict(key="logistic", expr="a/(1+exp(-b*(t-0.75)))", gen=_g_logistic,
         p0=[1.5, 2.0], arange=(1.0, 5.0), brange=(2.0, 8.0),
         concern="adoption / saturation curves (sigmoid)"),
]


class Panel:
    """A B-channel panel for one scenario, generated and consumed chunk by
    chunk."""

    def __init__(self, sc, n, b_channels, n_chunks, *, noise=0.01, seed=0):
        self.sc = sc
        self.n, self.B, self.n_chunks = int(n), int(b_channels), int(n_chunks)
        self.noise = float(noise)
        rng = np.random.default_rng(seed)
        self.a = rng.uniform(*sc["arange"], self.B)
        self.b = rng.uniform(*sc["brange"], self.B)
        self.x = np.linspace(*DOMAIN, self.n)
        self._edges = np.linspace(0, self.n, self.n_chunks + 1).astype(int)

    def chunk(self, p):
        lo, hi = self._edges[p], self._edges[p + 1]
        xi = self.x[lo:hi]
        clean = self.sc["gen"](xi, self.a, self.b)
        rng = np.random.default_rng(1000 + p)
        return xi, clean + rng.normal(0, self.noise, clean.shape)

    def resident(self):
        return self.x, np.concatenate([self.chunk(p)[1]
                                       for p in range(self.n_chunks)], axis=0)

    def true(self):
        return np.stack([self.a, self.b], axis=1)


def _coeffs(results):
    return np.array([r.coeffs for r in results], dtype=float)


# the dtfit routes, generic over the scenario's model
def fit_resident(panel, *, order=6):
    x, Y = panel.resident()
    return _coeffs(fit_lsi_batched(x, Y, panel.sc["expr"], "t", order=order,
                                   p0=panel.sc["p0"]))


def fit_streaming(panel, *, order=6):
    acc = PartitionedBatchLSI(panel.sc["expr"], "t", domain=DOMAIN,
                              n_channels=panel.B, order=order)
    for p in range(panel.n_chunks):
        acc.update(*panel.chunk(p))
    return _coeffs(acc.fit(p0=panel.sc["p0"]))


def _build_partition(panel, chunk_ids, order):
    acc = PartitionedBatchLSI(panel.sc["expr"], "t", domain=DOMAIN,
                              n_channels=panel.B, order=order)
    for p in chunk_ids:
        acc.update(*panel.chunk(int(p)))
    return acc


def fit_distributed(panel, *, n_workers=4, order=6, n_threads=1, chunk_order=None):
    ids = np.arange(panel.n_chunks) if chunk_order is None else np.asarray(chunk_order)
    parts = np.array_split(ids, n_workers)
    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            accs = list(ex.map(lambda c: _build_partition(panel, c, order), parts))
    else:
        accs = [_build_partition(panel, c, order) for c in parts]
    root = accs[0]
    for acc in accs[1:]:
        root.merge(acc)
    return _coeffs(root.fit(p0=panel.sc["p0"]))


# the established big-data baselines
def fit_per_channel_nlls(panel):
    """The obvious baseline: loop SciPy curve_fit over the channels, at gold
    accuracy."""
    x, Y = panel.resident()
    f = scenario_func(panel.sc)
    out = np.full((panel.B, 2), np.nan)
    for c in range(panel.B):
        try:
            out[c] = bl.scipy_curve_fit(x, Y[:, c], f, panel.sc["p0"])
        except Exception:
            pass
    return out


def scenario_func(sc):
    g = sc["gen"]
    return lambda t, a, b: g(np.atleast_1d(t), np.array([a]), np.array([b]))[:, 0]


def poly_lstsq_batched(x, Y, deg=6):
    """The established batched surrogate: one vectorised polynomial
    least-squares over all channels through `np.linalg.lstsq`. Fast, but it
    fits a surrogate that carries no physical parameters and extrapolates
    poorly."""
    V = np.vander(x, deg + 1)
    C = np.linalg.lstsq(V, Y, rcond=None)[0]
    return V, C


def _legendre_projection_operator(x, order):
    """The exact linear operator ``M`` (n x n_coef) of dtfit's empirical
    Legendre projection, built once in plain NumPy as
    ``M = diag(w) @ P @ diag((2j+1)/n)``, where ``P`` is the raw-Legendre
    design on ``x`` mapped to [-1, 1] and ``w`` is the trapezoidal-rule weight.
    ``spectra = M.T @ Y`` then reproduces ``project_spectra(x, Y, order)`` to
    machine precision. A single matmul is therefore a fair vectorised
    per-channel comparator (see :func:`numpy_projection_vectorised`)."""
    from numpy.polynomial import legendre as L
    x = np.asarray(x, float)
    n = x.size
    u = 2.0 * (x - x[0]) / (x[-1] - x[0]) - 1.0
    P = np.column_stack([L.legval(u, [0] * j + [1]) for j in range(order + 1)])
    w = np.full(n, n / (n - 1.0))                    # trapezoidal interior weight
    w[0] *= 0.5
    w[-1] *= 0.5
    scal = np.array([(2 * j + 1) / n for j in range(order + 1)])
    return (P * w[:, None]) @ np.diag(scal)          # (n, n_coef)


def numpy_projection_vectorised(x, Y, order=6):
    """The fair vectorised-NumPy per-channel baseline for the projection: build
    the Legendre projection operator once in plain NumPy, then apply it to
    every channel in a single matmul. This is what a competent practitioner
    writes, and comparing against it narrows dtfit's contribution to its
    optimised batched-GEMM and backend dispatch, rather than crediting dtfit
    for beating a per-channel Python loop that needlessly rebuilds the basis
    every iteration. It is the honest comparator :func:`realdata_projection`
    uses, and it reproduces ``project_spectra`` to about 1e-15."""
    Y = np.asarray(Y)
    M = _legendre_projection_operator(x, order)      # (n, n_coef)
    single = Y.ndim == 1
    S = M.T @ (Y[:, None] if single else Y)          # (n_coef, B)
    return S[:, 0] if single else S.T                # (B, n_coef) to match project_spectra


def sgd_incremental(x, Y, deg=6, chunk=2048):
    """The established streaming surrogate: scikit-learn
    ``SGDRegressor.partial_fit`` on polynomial features, fed chunk by chunk,
    the canonical incremental-ML baseline. One model per channel, as it is
    normally used."""
    from sklearn.linear_model import SGDRegressor
    V = np.vander(x, deg + 1)
    Vn = (V - V.mean(0)) / (V.std(0) + 1e-12)
    models = [SGDRegressor(max_iter=1, tol=None, learning_rate="invscaling",
                           eta0=0.01) for _ in range(Y.shape[1])]
    for i in range(0, x.size, chunk):
        for c, m in enumerate(models):
            m.partial_fit(Vn[i:i + chunk], Y[i:i + chunk, c])
    pred = np.column_stack([models[c].predict(Vn) for c in range(Y.shape[1])])
    return pred


def param_err(coef, panel):
    true = panel.true()
    return float(np.nanmean(np.abs(coef - true) / np.abs(true)) * 100)


def time_call(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# 2. throughput and memory scaling: the figure's data
def scaling_memory(panel, *, n_levels=None):
    """Peak memory of the resident and streaming routes as N grows, for the
    scaling figure. Returns ``(Ns, resident_mb, streaming_mb)``."""
    if n_levels is None:
        n_levels = [panel.n // 4, panel.n // 2, panel.n]
    rm, sm = [], []
    for nn in n_levels:
        d2 = Panel(panel.sc, nn, panel.B,
                   max(4, int(panel.n_chunks * nn / panel.n)))
        rm.append(peak_memory(lambda d=d2: fit_resident(d))[1])
        sm.append(peak_memory(lambda d=d2: fit_streaming(d))[1])
    return list(n_levels), rm, sm


# 2b. the GB-scale memory wall: resident against streaming
def memory_wall(Ns, *, B=64, order=6, chunk=100_000, scenario=None):
    """The GB-scale memory wall. The resident whole-array route holds the full
    ``(N, B)`` matrix in memory, an O(N*B) that climbs into the gigabytes,
    while the streaming reduce runs the identical computation chunk by chunk in
    flat O(B*order) memory. Returns ``(rows, res_mb, str_mb)``, ``rows`` being
    a list of dicts ready for a DataFrame."""
    sc = SCENARIOS[0] if scenario is None else scenario
    rng = np.random.default_rng(0)
    a = rng.uniform(*sc["arange"], B)
    b = rng.uniform(*sc["brange"], B)
    span = DOMAIN[1] - DOMAIN[0]

    def _xY(lo, hi, N):                       # one fixed-size chunk, generated live
        xi = DOMAIN[0] + span * np.arange(lo, hi) / (N - 1)
        return xi, sc["gen"](xi, a, b) + 0.01

    def stream(N):                            # flat: never holds more than a chunk
        acc = PartitionedBatchLSI(sc["expr"], "t", domain=DOMAIN, n_channels=B,
                                  order=order)
        for lo in range(0, N, chunk):
            acc.update(*_xY(lo, min(lo + chunk, N), N))
        return acc.fit(p0=sc["p0"])

    def resident(N):                          # holds the whole (N, B) panel
        xi, Y = _xY(0, N, N)
        return fit_lsi_batched(xi, Y, sc["expr"], "t", order=order, p0=sc["p0"])

    rows, res_mb, str_mb = [], [], []
    for N in Ns:
        _, m_str = peak_memory(lambda N=N: stream(N))
        _, m_res = peak_memory(lambda N=N: resident(N))
        res_mb.append(m_res); str_mb.append(m_str)
        rows.append({
            "elements": f"{N * B:,}",
            "N x B": f"{N:,}x{B}",
            "resident array (GB)": N * B * 8 / 1e9,
            "resident peak (MiB)": m_res,
            "streaming peak (MiB)": m_str,
            "memory ratio": m_res / m_str,
        })
    return rows, res_mb, str_mb


# 3. structured against surrogate: the extrapolation trap
def surrogate_trap(panel):
    """Fit each channel on the first half of the domain and predict the
    held-out second half. Only that extrapolation separates what each
    approach actually buys. Returns ``(results, fit_idx, ext_idx)``, where
    ``results`` maps a
    method label to a dict carrying ``in_window`` and ``extrapolation`` mean-R2
    along with ``kind`` and ``recovers``. The SGD surrogate row is skipped when
    scikit-learn is unavailable."""
    x, Y = panel.resident()
    split = int(0.5 * x.size)
    xf, Yf = x[:split], Y[:split]
    fit_idx, ext_idx = np.arange(split), np.arange(split, x.size)
    f = scenario_func(panel.sc)

    def r2win(Yhat, idx):
        return float(np.mean([metrics(Y[idx, c], Yhat[idx, c])["R2"]
                              for c in range(panel.B)]))

    # dtfit fused streaming on the fit region
    accf = PartitionedBatchLSI(panel.sc["expr"], "t",
                               domain=(float(xf[0]), float(xf[-1])),
                               n_channels=panel.B, order=6)
    for i in range(0, xf.size, 4096):
        accf.update(xf[i:i + 4096], Yf[i:i + 4096])
    cf = _coeffs(accf.fit(p0=panel.sc["p0"]))
    Yh_dt = panel.sc["gen"](x, cf[:, 0], cf[:, 1])

    # per-channel NLLS (recovers physics, no batch/stream)
    nl = np.full((panel.B, 2), np.nan)
    for c in range(panel.B):
        try:
            nl[c] = bl.scipy_curve_fit(xf, Yf[:, c], f, panel.sc["p0"])
        except Exception:
            pass
    Yh_nl = panel.sc["gen"](x, nl[:, 0], nl[:, 1])

    # established batched surrogate: polynomial lstsq
    V, C = poly_lstsq_batched(xf, Yf, deg=6)
    Yh_poly = np.vander(x, 7) @ C

    results = {
        "dtfit fused streaming": dict(
            kind="structured + streaming", recovers="physical a,b",
            in_window=r2win(Yh_dt, fit_idx), extrapolation=r2win(Yh_dt, ext_idx),
            pred=Yh_dt),
        "per-channel NLLS": dict(
            kind="structured, batch only", recovers="physical a,b",
            in_window=r2win(Yh_nl, fit_idx), extrapolation=r2win(Yh_nl, ext_idx),
            pred=Yh_nl),
        "polynomial lstsq (deg 6)": dict(
            kind="surrogate, batch", recovers="no params",
            in_window=r2win(Yh_poly, fit_idx), extrapolation=r2win(Yh_poly, ext_idx),
            pred=Yh_poly),
    }

    # The established streaming surrogate: SGD partial_fit on poly features,
    # tuned for a fair in-window fit with a standardized target, an adaptive
    # learning rate and several incremental epochs, so that its extrapolation
    # collapse is the honest point rather than under-fitting. It needs
    # sklearn, and is skipped without it.
    if HAS_SKLEARN:
        from sklearn.linear_model import SGDRegressor
        Vfull = np.vander(x, 7)
        mu, sd = Vfull[:split].mean(0), Vfull[:split].std(0) + 1e-12
        Vn_fit, Vn_all = (Vfull[:split] - mu) / sd, (Vfull - mu) / sd
        Yh_sgd = np.zeros_like(Y)
        for c in range(panel.B):
            ym, ys = Yf[:, c].mean(), Yf[:, c].std() + 1e-12
            m = SGDRegressor(max_iter=1, tol=None, eta0=0.01,
                             learning_rate="adaptive")
            for _ in range(15):
                for i in range(0, xf.size, 2048):
                    m.partial_fit(Vn_fit[i:i + 2048],
                                  (Yf[i:i + 2048, c] - ym) / ys)
            Yh_sgd[:, c] = m.predict(Vn_all) * ys + ym
        results["sklearn SGD partial_fit"] = dict(
            kind="surrogate, streaming", recovers="no params",
            in_window=r2win(Yh_sgd, fit_idx), extrapolation=r2win(Yh_sgd, ext_idx),
            pred=Yh_sgd)

    return results, fit_idx, ext_idx


# 4. numerical stability of the streaming reduction
def _reduce_errors(integrand, N, chunk_counts):
    """Chunked additive reduce of ``integrand`` in naive float32, float64 and
    Kahan, against an exact ``math.fsum`` reference. Returns
    ``(rows_partial, f32, f64, kahan)``, the three lists holding the relative
    error at each chunk-count."""
    exact = math.fsum(integrand.tolist())
    rows, f32c, f64c, kahc = [], [], [], []
    for k in chunk_counts:
        edges = np.linspace(0, N, k + 1).astype(int)
        s32 = np.float32(0.0)
        s64 = 0.0
        ks, kc = 0.0, 0.0                       # Kahan sum + compensation
        for j in range(k):
            seg = integrand[edges[j]:edges[j + 1]]
            s32 = np.float32(s32 + np.float32(seg.astype(np.float32).sum()))
            s64 += float(seg.sum())
            yk = float(seg.sum()) - kc
            tk = ks + yk
            kc = (tk - ks) - yk
            ks = tk
        e32 = abs(float(s32) - exact) / abs(exact)
        e64 = abs(s64 - exact) / abs(exact)
        ek = abs(ks - exact) / abs(exact)
        f32c.append(e32); f64c.append(e64); kahc.append(ek)
        rows.append({"# chunks": f"{k:,}", "naive float32": e32,
                     "dtfit float64": e64, "Kahan (compensated)": ek})
    return rows, f32c, f64c, kahc


def _integrand(N, kind):
    """The reduce-test integrand ``y*phi`` on ``N`` samples over :data:`DOMAIN`.

    ``kind="adversarial"`` gives the high-dynamic-range
    ``exp(2.5 x) cos(12 x)``, whose values span about e^3.75: the pathological
    case that stresses the additive reduce. ``kind="realistic"`` gives an
    O(1)-magnitude ``cos(12 x) + 1.5``, the well-scaled operating point of the
    streaming and embedded routes, whose signals are ENU metres or
    normalised."""
    x = np.linspace(*DOMAIN, N)
    phi = (2 * (x - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) - 1)        # P1 Legendre
    if kind == "realistic":
        return (np.cos(12 * x) + 1.5) * phi
    return (np.exp(2.5 * x) * np.cos(12 * x)) * phi                  # adversarial


def numerics(N, chunk_counts, *, kind="adversarial"):
    """Accumulate the projection integral ``int y*phi`` over a growing number
    of chunks, comparing naive float32, dtfit's float64 additive reduce and a
    compensated Kahan sum against an exact ``math.fsum`` reference. Returns
    ``(rows, f32, f64, kahan)``, the relative error at each chunk-count.

    ``kind`` selects the integrand (see :func:`_integrand`): the default
    adversarial high-dynamic-range case, where naive float32 visibly degrades,
    or the realistic O(1)-magnitude case. Pairing them through
    :func:`numerics_both` scopes the float32 claim honestly and keeps it
    consistent with the embedded domain deploying float32: float32 is unsafe
    for pathological dynamic range and safe for the well-scaled magnitudes the
    on-MCU filter actually runs on."""
    integrand = _integrand(N, kind)
    rows, f32c, f64c, kahc = _reduce_errors(integrand, N, chunk_counts)
    header = f"# chunks (over {N:,} samples)"
    rows = [{header: r["# chunks"], **{k: v for k, v in r.items() if k != "# chunks"}}
            for r in rows]
    return rows, f32c, f64c, kahc


def numerics_both(N, chunk_counts):
    """Run :func:`numerics` on both the adversarial and the realistic
    integrand, so the notebook can put the float32-instability claim beside its
    honest scope. Returns one :func:`numerics` tuple per kind, keyed
    ``"adversarial"`` and ``"realistic"``. The realistic curve shows naive
    float32 staying accurate on well-scaled data; that is why the embedded
    domain can ship float32 at all."""
    return {kind: numerics(N, chunk_counts, kind=kind)
            for kind in ("adversarial", "realistic")}


# 5. robustness and mergeability at scale
def robustness(panel):
    """Test the three distributed-pipeline guarantees the additive structure
    provides against the in-order whole-array reference: merge-order
    independence, uneven contiguous shards, and 20% missing data. Returns a
    list of DataFrame-ready dicts of condition, max |delta| against the
    reference and parameter error in percent."""
    c_ref = fit_resident(panel)
    rng = np.random.default_rng(7)
    ids = np.arange(panel.n_chunks)

    # (a) merge-order independence: 4 contiguous partitions merged in two
    # different orders must come out identical, the associative-merge guarantee
    # a distributed pipeline rests on.
    def merge_in(order):
        parts = [_build_partition(panel, c, 6) for c in np.array_split(ids, 4)]
        root = parts[order[0]]
        for k in order[1:]:
            root.merge(parts[k])
        return _coeffs(root.fit(p0=panel.sc["p0"]))
    c_o1 = merge_in([0, 1, 2, 3])
    c_o2 = merge_in([3, 1, 0, 2])

    # (b) uneven contiguous shards instead of equal splits.
    accs = [_build_partition(panel, ids[:1], 6),
            _build_partition(panel, ids[1:1 + panel.n_chunks // 4], 6),
            _build_partition(panel, ids[1 + panel.n_chunks // 4:], 6)]
    root = accs[0]
    for a in accs[1:]:
        root.merge(a)
    c_uneven = _coeffs(root.fit(p0=panel.sc["p0"]))

    # (c) missing data: drop 20% of samples uniformly at random (order preserved).
    acc = PartitionedBatchLSI(panel.sc["expr"], "t", domain=DOMAIN,
                              n_channels=panel.B, order=6)
    for p in range(panel.n_chunks):
        xc, yc = panel.chunk(p)
        keep = np.sort(rng.choice(xc.size, int(xc.size * 0.8), replace=False))
        acc.update(xc[keep], yc[keep])
    c_miss = _coeffs(acc.fit(p0=panel.sc["p0"]))

    return [
        {"condition": "merge partitions in a different order",
         "max |delta| vs reference": float(np.max(np.abs(c_o1 - c_o2))),
         "param err % vs true": param_err(c_o2, panel)},
        {"condition": "uneven contiguous shards (1 / 1/4 / rest)",
         "max |delta| vs reference": float(np.max(np.abs(c_ref - c_uneven))),
         "param err % vs true": param_err(c_uneven, panel)},
        {"condition": "20% of samples missing (uniform)",
         "max |delta| vs reference": float(np.max(np.abs(c_ref - c_miss))),
         "param err % vs true": param_err(c_miss, panel)},
    ]


# 6. the online streaming filter against the established online estimators
def online_filter(n):
    """Track a sinusoid through a mid-stream frequency jump one sample at a
    time, comparing dtfit's ``EACFilter`` against the established online
    toolkit (recursive least squares, plus an incremental SGD net where sklearn
    is present) on per-sample cost, memory, one-step prediction error and
    whether the physical frequency is recovered at all. Returns
    ``(rows, t, w_hist, half)``: ``rows`` is DataFrame-ready and the rest feed
    the tracking figure."""
    import tracemalloc
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, n)
    half = n // 2
    w_seq = np.where(np.arange(n) < half, 1.0, 1.6)
    dt_ = np.diff(t, prepend=t[0])
    phase = np.cumsum(w_seq * dt_)
    y = 3.0 * np.sin(phase) + rng.normal(0, 0.3, n)

    # dtfit EACFilter, tracking the physical model A*sin(w*t)
    flt = EACFilter("A*sin(w*t)", "t", p0=[2.0, 1.0], window_size=50,
                    q_diag=[1e-3, 5e-4], order=2)
    costs, w_hist, pred_eaf = [], [], np.full(n, np.nan)
    tracemalloc.start()
    for i in range(n):
        t0 = time.perf_counter()
        flt.partial_fit(t[i], y[i])
        costs.append((time.perf_counter() - t0) * 1e6)
        if len(flt._t):
            pred_eaf[i] = float(flt.predict(np.array([t[i]]))[0])
        w_hist.append(flt.params_["w"])
    peak_eaf = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    us_eaf = float(np.mean(costs[200:]))
    w_hist = np.array(w_hist)

    # established: an RLS one-step predictor, AR(6) with no forgetting, since a
    # forgetting factor below 1 winds the covariance up and blows the filter
    # out on a signal this oversampled.
    rls = bl.RLSPredictor(order=6, lam=1.0, delta=1000.0)
    pred_rls = np.full(n, np.nan)
    t0 = time.perf_counter()
    for i in range(n):
        pred_rls[i] = rls.predict_next()
        rls.update(y[i])
    us_rls = (time.perf_counter() - t0) / n * 1e6

    def osr(pred):  # one-step RMSE on the second half (post-jump), valid preds
        seg = slice(half + 100, n)
        p, a = pred[seg], y[seg]
        m = np.isfinite(p)
        return float(np.sqrt(np.mean((p[m] - a[m]) ** 2)))

    w_err = abs(w_hist[-1] - 1.6) / 1.6 * 100
    rows = [
        {"online method": "dtfit EACFilter", "us / sample": us_eaf,
         "memory": f"bounded ({peak_eaf:.1f} MB)",
         "one-step RMSE (post-jump)": osr(pred_eaf),
         "recovers physics": f"yes (w err {w_err:.1f}%)"},
        {"online method": "recursive least squares (AR6)", "us / sample": us_rls,
         "memory": "bounded", "one-step RMSE (post-jump)": osr(pred_rls),
         "recovers physics": "no (black-box AR)"},
    ]

    # established: an incremental SGD net on lagged features (optional)
    if HAS_SKLEARN:
        from sklearn.linear_model import SGDRegressor
        lag = 8
        sgd = SGDRegressor(max_iter=1, tol=None, eta0=0.005,
                           learning_rate="invscaling")
        pred_sgd = np.full(n, np.nan)
        t0 = time.perf_counter()
        buf = list(y[:lag])
        sgd.partial_fit(np.array(y[:lag]).reshape(1, -1), [y[lag]])
        for i in range(lag, n):
            xv = np.array(buf[-lag:]).reshape(1, -1)
            pred_sgd[i] = float(sgd.predict(xv)[0])
            sgd.partial_fit(xv, [y[i]])
            buf.append(y[i])
        us_sgd = (time.perf_counter() - t0) / n * 1e6
        rows.append(
            {"online method": "sklearn SGD partial_fit (lag-8)", "us / sample": us_sgd,
             "memory": "bounded", "one-step RMSE (post-jump)": osr(pred_sgd),
             "recovers physics": "no (black-box)"})

    return rows, t, w_hist, half


# 7. real data: the 321-channel electricity load from LTSF
def load_electricity(rows):
    """Load the last ``rows`` timesteps of the 321-channel electricity LTSF
    panel. Returns ``(x, Y, full)``, ``full`` being the un-sliced array the
    example plots want, or raises when the data is unavailable."""
    data = ltsf.load("electricity")
    Y = np.ascontiguousarray(data[-rows:, :], dtype=float)
    x = np.linspace(*DOMAIN, Y.shape[0])
    return x, Y, data


def realdata_projection(x, Y, *, order=6, n_chunks=12):
    """Project all the real channels several ways, by batched GEMM, by the fair
    vectorised-NumPy per-channel baseline and through the streaming
    accumulator, plus the polynomial lstsq surrogate for scale. They must all
    agree. Returns a list of DataFrame-ready dicts and the per-method timings
    dict the bar figure draws.

    The headline speed-up is dtfit's batched GEMM against the fair vectorised
    baseline, which builds the projection operator once and does one matmul, as
    a competent practitioner would. It is deliberately not reported against a
    naive per-channel Python loop that rebuilds the basis every channel: that
    comparator inflates the speed-up with loop overhead any vectorised
    implementation would remove, and it does not change the verdict. Should
    dtfit's optimised GEMM and the fair vectorised baseline land at
    near-parity, that is the honest result, because dtfit's win here is
    exactness across routes rather than beating a competently vectorised
    matmul."""
    B = Y.shape[1]

    spec_res, t_res = time_call(lambda: project_spectra(x, Y, order=order))
    _, mem_res = peak_memory(lambda: project_spectra(x, Y, order=order))

    # the fair comparator: one vectorised NumPy projection, all channels
    spec_vec, t_vec = time_call(lambda: numpy_projection_vectorised(x, Y, order=order))
    _, mem_vec = peak_memory(lambda: numpy_projection_vectorised(x, Y, order=order))

    _, t_poly = time_call(lambda: poly_lstsq_batched(x, Y, deg=order))

    edges = np.linspace(0, Y.shape[0], n_chunks + 1).astype(int)

    def stream_spectra():
        acc = PartitionedBatchLSI("a*exp(b*t)", "t", domain=DOMAIN,
                                  n_channels=B, order=order)
        for k in range(n_chunks):
            acc.update(x[edges[k]:edges[k + 1]], Y[edges[k]:edges[k + 1]])
        return acc.spectra()
    spec_str, t_str = time_call(stream_spectra)
    _, mem_str = peak_memory(stream_spectra)

    d_vec = float(np.max(np.abs(spec_res - spec_vec)))
    d_str = float(np.max(np.abs(spec_res - spec_str)))
    rows = [
        {"method": "dtfit batched GEMM (project_spectra)", "time (s)": t_res,
         "peak mem (MiB)": fmt(mem_res, "{:.1f}"), "max |delta| vs batched": "0 (ref)",
         "speed-up": "1x (ref)"},
        {"method": "vectorised NumPy projection (FAIR baseline)", "time (s)": t_vec,
         "peak mem (MiB)": fmt(mem_vec, "{:.1f}"),
         "max |delta| vs batched": fmt(d_vec, "{:.1e}"),
         "speed-up": fmt(t_vec / t_res, "{:.2f}x")},
        {"method": "dtfit streaming accumulator", "time (s)": t_str,
         "peak mem (MiB)": fmt(mem_str, "{:.1f}"),
         "max |delta| vs batched": fmt(d_str, "{:.1e}"), "speed-up": "flat memory"},
        {"method": "polynomial lstsq (established)", "time (s)": t_poly,
         "peak mem (MiB)": "-", "max |delta| vs batched": "n/a (surrogate)",
         "speed-up": fmt(t_poly / t_res, "{:.2f}x")},
    ]
    timings = dict(batched=t_res, vectorised=t_vec, streaming=t_str,
                   poly=t_poly, B=B,
                   # the headline: fair NumPy over batched GEMM
                   ratio_fair=t_vec / t_res)
    return rows, timings
