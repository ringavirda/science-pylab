"""A catalog of named nonlinear model families with data-driven seeders.

Each factory returns a fresh :class:`Model` whose seeder reads sensible
initial values and bounds off the data. You pick the structure and the model
handles the rest::

    from dtfit import models
    result = models.logistic().fit(x, y)        # no p0/bounds to hand-write

The families cover the shapes that keep recurring: trends and growth or decay
laws, sigmoids, saturating and rational responses, spectral peaks,
oscillations. Each carries a ``category`` tag the recommender uses to
shortlist candidates. Seeders are heuristics, a starting point for the fitter
to refine rather than a fit in their own right.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from dtfit.image.fit import fft_frequency_seed
from ._model import Model

INF = float("inf")


def _span(x: np.ndarray) -> float:
    return float(x[-1] - x[0]) or 1.0


def _level_crossing(x: np.ndarray, y: np.ndarray, frac: float) -> float:
    """x at which y first reaches ``frac`` of the way from its min to max."""
    lo, hi = float(np.min(y)), float(np.max(y))
    target = lo + frac * (hi - lo)
    idx = (np.where(y >= target)[0] if y[-1] >= y[0]
           else np.where(y <= target)[0])
    return float(x[idx[0]]) if idx.size else float(np.median(x))


def _peak_index(y: np.ndarray) -> int:
    return int(np.argmax(np.abs(y - np.median(y))))


# trends (polynomial / algebraic)
def linear() -> Model:
    def seed(x, y):
        a1, a0 = np.polyfit(x, y, 1)
        return {"a0": (float(a0), -INF, INF), "a1": (float(a1), -INF, INF)}
    return Model("a0 + a1*x", name="linear", shape="bulk", category="trend",
                 seeder=seed)


def quadratic() -> Model:
    def seed(x, y):
        a2, a1, a0 = np.polyfit(x, y, 2)
        return {"a0": (float(a0), -INF, INF), "a1": (float(a1), -INF, INF),
                "a2": (float(a2), -INF, INF)}
    return Model("a0 + a1*x + a2*x**2", name="quadratic", shape="bulk",
                 category="trend", seeder=seed)


def cubic() -> Model:
    def seed(x, y):
        a3, a2, a1, a0 = np.polyfit(x, y, 3)
        return {"a0": (float(a0), -INF, INF), "a1": (float(a1), -INF, INF),
                "a2": (float(a2), -INF, INF), "a3": (float(a3), -INF, INF)}
    return Model("a0 + a1*x + a2*x**2 + a3*x**3", name="cubic", shape="bulk",
                 category="trend", seeder=seed)


def power_law() -> Model:
    """A scaling law ``a*(x + 1)**b``; the +1 keeps it finite at x=0.

    Requires ``x > -1``.
    """
    def seed(x, y):
        x = np.asarray(x, float)
        if np.any(x <= -1.0):
            raise ValueError(
                "power_law() requires x > -1 (the model uses (x + 1)**b)"
            )
        return {"a": (float(np.clip(y[0], 1e-6, None)), -INF, INF),
                "b": (1.0, -INF, INF)}
    return Model("a*(x + 1)**b", name="power_law", shape="bulk",
                 category="trend", seeder=seed)


def logarithmic() -> Model:
    """A learning / log-growth curve ``a + b*log(x + 1)`` (requires x > -1)."""
    def seed(x, y):
        x = np.asarray(x, float)
        if np.any(x <= -1.0):
            raise ValueError(
                "logarithmic() requires x > -1 (the model uses log(x + 1))"
            )
        # Fit on the same basis the expression itself uses. Shifting x first
        # gives a p0 for a different model unless x starts near 0.
        b, a = np.polyfit(np.log(x + 1.0), y, 1)
        return {"a": (float(a), -INF, INF), "b": (float(b), -INF, INF)}
    return Model("a + b*log(x + 1)", name="logarithmic", shape="bulk",
                 category="trend", seeder=seed)


def sqrt_law() -> Model:
    """Diffusion-like ``a + b*sqrt(x)``, distance vs time. Needs ``x >= 0``."""
    def seed(x, y):
        x = np.asarray(x, float)
        if np.any(x < 0.0):
            raise ValueError(
                "sqrt_law() requires x >= 0 (the model uses sqrt(x))"
            )
        # Fit on the model's own basis, sqrt(x). A shifted one gives a p0 for
        # a different model unless x is already anchored at 0.
        b, a = np.polyfit(np.sqrt(x), y, 1)
        return {"a": (float(a), -INF, INF), "b": (float(b), -INF, INF)}
    return Model("a + b*sqrt(x)", name="sqrt_law", shape="bulk",
                 category="trend", seeder=seed)


# growth
def _log_rate(x: np.ndarray, y: np.ndarray) -> float:
    yy = np.clip(np.abs(y), 1e-9, None)
    return float(np.polyfit(x, np.log(yy), 1)[0])


def exponential() -> Model:
    def seed(x, y):
        a = float(y[0]) if y[0] != 0 else float(np.sign(np.mean(y)) or 1.0)
        return {"a": (a, -INF, INF), "b": (_log_rate(x, y), -INF, INF)}
    return Model("a*exp(b*x)", name="exponential", shape="bulk",
                 category="growth", seeder=seed)


def exp_growth_offset() -> Model:
    """Compounding growth above a baseline ``c + a*exp(b*x)``."""
    def seed(x, y):
        c = float(np.min(y))
        a = float(y[0] - c) or 1.0
        b = abs(_log_rate(x, y)) or 1.0 / _span(x)
        return {"a": (a, -INF, INF), "b": (b, -INF, INF), "c": (c, -INF, INF)}
    return Model("c + a*exp(b*x)", name="exp_growth_offset", shape="bulk",
                 category="growth", seeder=seed)


# decay / relaxation
def exp_decay() -> Model:
    def seed(x, y):
        a = float(y[0]) if y[0] != 0 else 1.0
        b = abs(_log_rate(x, y)) or 1.0 / _span(x)
        return {"a": (a, -INF, INF), "b": (b, 1e-9, INF)}
    return Model("a*exp(-b*x)", name="exp_decay", shape="bulk",
                 category="decay", seeder=seed)


def exp_decay_offset() -> Model:
    """Newton cooling / RC discharge to a floor: ``c + a*exp(-b*x)``."""
    def seed(x, y):
        c = float(np.mean(y[-max(3, y.size // 10):]))  # tail level
        a = float(y[0] - c)
        return {"a": (a, -INF, INF), "b": (3.0 / _span(x), 1e-6, INF),
                "c": (c, -INF, INF)}
    return Model("c + a*exp(-b*x)", name="exp_decay_offset", shape="bulk",
                 category="decay", seeder=seed)


def first_order() -> Model:
    """A first-order step response ``K*(1 - exp(-x/tau))`` (RC, DC motor)."""
    def seed(x, y):
        K = float(np.max(y) * 1.05)
        tau = max(_level_crossing(x, y, 0.63) - float(x[0]), _span(x) / 50.0)
        return {"K": (K, 0.0, float(np.max(y) * 5 + 1e-9)),
                "tau": (tau, _span(x) / 100.0, _span(x) * 5)}
    return Model("K*(1 - exp(-x/tau))", name="first_order", shape="bulk",
                 category="decay", seeder=seed)


def biexponential() -> Model:
    """Two-rate decay ``a*exp(-b*x) + c*exp(-d*x)`` (pharmacokinetics)."""
    def seed(x, y):
        y0 = float(y[0]) if y[0] != 0 else 1.0
        fast, slow = 3.0 / _span(x), 0.5 / _span(x)
        return {"a": (0.6 * y0, -INF, INF), "b": (fast, 1e-9, INF),
                "c": (0.4 * y0, -INF, INF), "d": (slow, 1e-9, INF)}
    return Model("a*exp(-b*x) + c*exp(-d*x)", name="biexponential",
                 shape="bulk", category="decay", seeder=seed)


def stretched_exponential() -> Model:
    """KWW relaxation ``A*exp(-(x/tau)**beta)`` (disordered/glassy systems)."""
    def seed(x, y):
        A = float(y[0]) if y[0] != 0 else 1.0
        return {"A": (A, -2 * abs(A) - 1e-9, 2 * abs(A) + 1e-9),
                "tau": (_span(x) / 3.0, _span(x) / 100.0, _span(x) * 5),
                "q": (1.0, 0.1, 2.5)}  # stretch exp (sympy reserves 'beta')
    return Model("A*exp(-(x/tau)**q)", name="stretched_exponential",
                 shape="bulk", category="decay", seeder=seed)


# sigmoids
def logistic() -> Model:
    """Saturating growth ``L/(1 + exp(-k*(x - x0)))`` (epidemics, adoption)."""
    def seed(x, y):
        L = float(np.max(y) * 1.05)
        x0 = _level_crossing(x, y, 0.5)
        return {"L": (L, float(np.max(y) * 0.8), float(np.max(y) * 12 + 1e-9)),
                "k": (4.0 / _span(x), 1e-3, 60.0),
                "x0": (x0, float(x[0]), float(x[0] + 2.5 * _span(x)))}
    return Model("L/(1 + exp(-k*(x - x0)))", name="logistic", shape="bulk",
                 category="sigmoid", seeder=seed)


def gompertz() -> Model:
    """Asymmetric sigmoid ``A*exp(-b*exp(-c*x))`` (tumour / population)."""
    def seed(x, y):
        A = float(np.max(y) * 1.05)
        return {"A": (A, float(np.max(y) * 0.5), float(np.max(y) * 5 + 1e-9)),
                "b": (1.0, 1e-3, 50.0), "c": (1.0 / _span(x), 1e-3, 60.0)}
    return Model("A*exp(-b*exp(-c*x))", name="gompertz", shape="bulk",
                 category="sigmoid", seeder=seed)


def weibull_cdf() -> Model:
    """Reliability failure CDF ``K*(1 - exp(-(x/lam)**k))``."""
    def seed(x, y):
        K = float(np.max(y) * 1.05)
        lam = max(_level_crossing(x, y, 0.63) - float(x[0]), _span(x) / 50.0)
        return {"K": (K, 0.0, float(np.max(y) * 5 + 1e-9)),
                "lam": (lam, _span(x) / 100.0, _span(x) * 5),
                "k": (1.5, 0.2, 10.0)}
    return Model("K*(1 - exp(-(x/lam)**k))", name="weibull_cdf", shape="bulk",
                 category="sigmoid", seeder=seed)


def tanh_step() -> Model:
    """A smooth step ``a + b*tanh(c*(x - d))`` (transitions, switching)."""
    def seed(x, y):
        a = 0.5 * (float(np.max(y)) + float(np.min(y)))
        b = 0.5 * (float(np.max(y)) - float(np.min(y))) + 1e-9
        return {"a": (a, -INF, INF), "b": (b, -INF, INF),
                "c": (4.0 / _span(x), -INF, INF),
                "d": (_level_crossing(x, y, 0.5), -INF, INF)}
    return Model("a + b*tanh(c*(x - d))", name="tanh_step", shape="bulk",
                 category="sigmoid", seeder=seed)


# saturating / rational; the curvature on the early rise routes these to EAC
def michaelis_menten() -> Model:
    """Enzyme-kinetics saturation ``Vmax*x/(K + x)``."""
    def seed(x, y):
        Vmax = float(np.max(y) * 1.1)
        idx = np.where(y >= 0.5 * Vmax)[0]
        K = float(x[idx[0]]) if idx.size else float(np.median(x))
        return {"Vmax": (Vmax, 1e-6, float(np.max(y) * 5 + 1e-9)),
                "K": (max(K, 1e-6), 1e-6, float(x[-1] * 5 + 1e-9))}
    return Model("Vmax*x/(K + x)", name="michaelis_menten", shape="peak",
                 category="saturating", seeder=seed)


def hill() -> Model:
    """Cooperative dose-response ``Vmax*x**n/(K**n + x**n)``."""
    def seed(x, y):
        Vmax = float(np.max(y) * 1.1)
        idx = np.where(y >= 0.5 * Vmax)[0]
        K = float(x[idx[0]]) if idx.size else float(np.median(x))
        return {"Vmax": (Vmax, 1e-6, float(np.max(y) * 5 + 1e-9)),
                "K": (max(K, 1e-6), 1e-6, float(x[-1] * 5 + 1e-9)),
                "n": (1.5, 0.2, 8.0)}
    return Model("Vmax*x**n/(K**n + x**n)", name="hill", shape="peak",
                 category="saturating", seeder=seed)


# peaks
def gaussian() -> Model:
    """A spectral peak ``A*exp(-(x-mu)**2/(2*s**2))``."""
    def seed(x, y):
        i = _peak_index(y)
        A = float(y[i])
        return {"A": (A, -2 * abs(A) - 1e-9, 2 * abs(A) + 1e-9),
                "mu": (float(x[i]), float(x[0]), float(x[-1])),
                "s": (max(_span(x) / 10.0, 1e-3), 1e-3, _span(x))}
    return Model("A*exp(-(x - mu)**2/(2*s**2))", name="gaussian", shape="peak",
                 category="peak", seeder=seed)


def lorentzian() -> Model:
    """A resonance peak ``A/(1 + ((x-mu)/g)**2)`` (heavy tails)."""
    def seed(x, y):
        i = _peak_index(y)
        A = float(y[i])
        return {"A": (A, -2 * abs(A) - 1e-9, 2 * abs(A) + 1e-9),
                "g": (max(_span(x) / 10.0, 1e-3), 1e-3, _span(x)),
                "mu": (float(x[i]), float(x[0]), float(x[-1]))}
    return Model("A/(1 + ((x - mu)/g)**2)", name="lorentzian", shape="peak",
                 category="peak", seeder=seed)


def double_gaussian() -> Model:
    """Two overlapping peaks (chromatography / spectroscopy)."""
    def seed(x, y):
        yc = np.asarray(y, float) - np.median(y)
        i1 = int(np.argmax(yc))
        masked = yc.copy()
        w = max(y.size // 10, 1)
        masked[max(0, i1 - w):i1 + w] = -np.inf
        i2 = int(np.argmax(masked)) if np.isfinite(masked).any() else i1
        s = max(_span(x) / 12.0, 1e-3)
        return {
            "A1": (float(y[i1]), 0.0, 2 * abs(float(y[i1])) + 1e-9),
            "m1": (float(x[i1]), float(x[0]), float(x[-1])),
            "s1": (s, 1e-3, _span(x)),
            "A2": (float(y[i2]), 0.0, 2 * abs(float(y[i2])) + 1e-9),
            "m2": (float(x[i2]), float(x[0]), float(x[-1])),
            "s2": (s, 1e-3, _span(x)),
        }
    return Model(
        "A1*exp(-(x - m1)**2/(2*s1**2)) + A2*exp(-(x - m2)**2/(2*s2**2))",
        name="double_gaussian", shape="peak", category="peak", seeder=seed)


# oscillatory: FFT-seeded frequency, oscillatory recipe
def sine() -> Model:
    """A sustained cycle ``c + A*sin(w*x + p)`` (signals, seasonality)."""
    def seed(x, y):
        w = fft_frequency_seed(x, y) or (2 * np.pi / _span(x))
        amp = float(np.std(y) * np.sqrt(2.0)) + 1e-3
        return {"A": (amp, 1e-3, 5 * amp),
                "c": (float(np.mean(y)), float(np.min(y) - amp),
                      float(np.max(y) + amp)),
                "p": (0.0, -np.pi, np.pi), "w": (w, 0.3 * w, 3 * w)}
    return Model("c + A*sin(w*x + p)", name="sine", shape="oscillatory",
                 category="oscillatory", freq_param="w", seeder=seed)


def damped_oscillation() -> Model:
    """A ring-down ``A*exp(-z*w*x)*sin(w*sqrt(1-z**2)*x)`` (RLC, vibration)."""
    def seed(x, y):
        w = fft_frequency_seed(x, y) or (2 * np.pi / _span(x))
        amp = float(np.max(np.abs(y))) + 1e-3
        return {"A": (amp, 0.1 * amp, 5 * amp), "w": (w, 0.3 * w, 3 * w),
                "z": (0.05, 1e-3, 0.9)}
    return Model("A*exp(-z*w*x)*sin(w*sqrt(1 - z**2)*x)",
                 name="damped_oscillation", shape="oscillatory",
                 category="oscillatory", freq_param="w", seeder=seed)


def fourier_series(n_harmonics: int = 3) -> Model:
    """A periodic waveform: fundamental plus ``n_harmonics`` harmonics.

    ``c + sum_k a_k*sin(k*w*x) + b_k*cos(k*w*x)``, for a distorted periodic
    signal (AC mains, gear meshing) that a single sine cannot represent.
    """
    terms = ["c"] + [f"a{k}*sin({k}*w*x) + b{k}*cos({k}*w*x)"
                     for k in range(1, n_harmonics + 1)]
    expr = " + ".join(terms)

    def seed(x, y):
        w = fft_frequency_seed(x, y) or (2 * np.pi / _span(x))
        amp = float(np.max(np.abs(y - np.mean(y)))) + 1e-3
        d = {"c": (float(np.mean(y)), float(np.min(y)), float(np.max(y))),
             "w": (w, 0.7 * w, 1.3 * w)}
        for k in range(1, n_harmonics + 1):
            d[f"a{k}"] = (0.0, -2 * amp, 2 * amp)
            d[f"b{k}"] = (0.0, -2 * amp, 2 * amp)
        return d
    return Model(expr, name=f"fourier_series({n_harmonics})",
                 shape="oscillatory", category="oscillatory", freq_param="w",
                 seeder=seed)


# The default candidate set for ``suggest_models``. fourier_series takes a
# parameter and is offered separately rather than in the default sweep.
# ``register`` and ``unregister`` mutate this dict in place: the recommender
# imported the name, and rebinding it here would leave that import stale.
CATALOG: dict[str, Callable[[], Model]] = {
    # trend
    "linear": linear, "quadratic": quadratic, "cubic": cubic,
    "power_law": power_law, "logarithmic": logarithmic, "sqrt_law": sqrt_law,
    # growth
    "exponential": exponential, "exp_growth_offset": exp_growth_offset,
    # decay
    "exp_decay": exp_decay, "exp_decay_offset": exp_decay_offset,
    "first_order": first_order, "biexponential": biexponential,
    "stretched_exponential": stretched_exponential,
    # sigmoid
    "logistic": logistic, "gompertz": gompertz, "weibull_cdf": weibull_cdf,
    "tanh_step": tanh_step,
    # saturating
    "michaelis_menten": michaelis_menten, "hill": hill,
    # peak
    "gaussian": gaussian, "lorentzian": lorentzian,
    "double_gaussian": double_gaussian,
    # oscillatory
    "sine": sine, "damped_oscillation": damped_oscillation,
}


def all_models() -> list[Model]:
    """Fresh instances of every catalogued family."""
    return [factory() for factory in CATALOG.values()]


# The shipped family names, snapshotted at import: ``register`` flags an
# overwrite of a builtin and ``unregister`` refuses to remove one.
_BUILTINS = frozenset(CATALOG)


def register(
    name: str,
    factory: Callable[[], Model],
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom model family into the catalog.

    Once registered, the family is visible everywhere the builtins are:
    :func:`all_models` returns it, :data:`CATALOG` holds it under its ``name``,
    and :func:`~dtfit.models.suggest_models` considers it. A family whose
    ``category`` falls outside the recommender's known shape categories, the
    default ``"general"`` among them, stays in every default shortlist.

    Parameters
    ----------
    name : str
        The catalog key. Must be a non-empty string.
    factory : Callable[[], Model]
        A zero-argument callable returning a fresh :class:`Model`, matching
        the builtin factories: ``lambda: Model("a*x", name="myline")``. It is
        called once here to validate it and again on every use; return a new
        instance each call rather than a shared one.
    overwrite : bool, optional
        A collision with an existing family, builtin or previously registered,
        raises :class:`ValueError` by default. Pass ``overwrite=True`` to
        replace it; replacing a builtin also emits a :class:`UserWarning`.

    Raises
    ------
    ValueError
        If ``name`` is empty or blank, if it collides with an existing family
        and ``overwrite`` is false, or if ``factory`` raises when called.
    TypeError
        If ``factory`` is not callable or does not return a :class:`Model`.

    See Also
    --------
    unregister : Remove a family added by :func:`register`.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a non-empty string")
    if not callable(factory):
        raise TypeError(
            f"factory for model {name!r} must be a zero-argument callable "
            f"returning a Model, got {type(factory).__name__}"
        )
    if name in CATALOG and not overwrite:
        kind = "builtin" if name in _BUILTINS else "registered"
        raise ValueError(
            f"a {kind} model named {name!r} already exists; "
            f"pass overwrite=True to replace it"
        )
    # Validate eagerly. A broken factory should fail here, not later inside
    # all_models() or suggest_models() where the traceback points elsewhere.
    try:
        probe = factory()
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ValueError(
            f"factory for model {name!r} raised when called: {exc}"
        ) from exc
    if not isinstance(probe, Model):
        raise TypeError(
            f"factory for model {name!r} must return a Model, got "
            f"{type(probe).__name__}"
        )
    if name in _BUILTINS:
        warnings.warn(
            f"overwriting the builtin model {name!r}", UserWarning,
            stacklevel=2
        )
    CATALOG[name] = factory


def unregister(name: str) -> None:
    """Remove a custom family previously added by :func:`register`.

    Builtin families cannot be removed; :func:`register` with
    ``overwrite=True`` shadows one instead.

    Raises
    ------
    ValueError
        If ``name`` is a builtin family.
    KeyError
        If no family named ``name`` is registered.
    """
    if name in _BUILTINS:
        raise ValueError(f"cannot unregister the builtin model {name!r}")
    if name not in CATALOG:
        raise KeyError(f"no model named {name!r} is registered")
    del CATALOG[name]
