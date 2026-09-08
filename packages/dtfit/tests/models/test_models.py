"""dtfit.models: self-seeding catalog families, composition, and suggest_models."""

import warnings

import numpy as np
import pytest

from dtfit import Original, models, suggest_models
from dtfit.models import Model


def test_logistic_self_seeds_and_recovers():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 12, 200)
    y = 1000.0 / (1 + np.exp(-0.7 * (t - 6))) + rng.normal(0, 8, t.size)
    r = models.logistic().fit(t, y)  # no p0/bounds supplied
    assert r.params["L"] == pytest.approx(1000, rel=0.05)
    assert r.params["k"] == pytest.approx(0.7, rel=0.1)
    assert r.params["x0"] == pytest.approx(6.0, abs=0.3)


def test_oscillatory_family_recovers_frequency():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 4 * np.pi, 300)
    y = 2.0 * np.sin(1.5 * t) + rng.normal(0, 0.05, t.size)
    r = models.sine().fit(t, y)
    assert r.params["w"] == pytest.approx(1.5, abs=0.1)


def test_model_carries_params_and_shape():
    m = models.damped_oscillation()
    assert m.shape == "oscillatory" and m.freq_param == "w"
    assert set(m.params) == {"A", "w", "z"}


def test_composition_builds_combined_expr():
    m = models.linear() + models.sine()
    # combined params are the union (linear a0,a1 + sine A,c,p,w), no collision
    assert {"a0", "a1", "A", "c", "p", "w"} <= set(m.params)
    assert m.shape == "composite"
    rng = np.random.default_rng(2)
    x = np.linspace(0, 20, 300)
    y = 0.5 * x + 3 * np.sin(2 * np.pi * x / 4) + rng.normal(0, 0.2, x.size)
    r = m.fit(x, y)
    # the cycle is seeded on the detrended residual; that is why 0.95 holds
    from dtfit.diagnostics import fit_report
    assert fit_report(r, x, y)["r2"] > 0.95


def test_composition_renames_colliding_params():
    m = models.gaussian() + models.gaussian()  # both have A, mu, s
    # second component's params are renamed, so all are distinct
    assert len(m.params) == 6


def test_suggest_models_ranks_true_family_first():
    rng = np.random.default_rng(3)
    t = np.linspace(0.5, 6, 200)
    y = 2.0 * np.exp(0.5 * t) + rng.normal(0, 1.0, t.size)
    ranked = suggest_models(t, y, top=3)
    assert ranked, "no suggestions returned"
    names = [s.name for s in ranked]
    # the exponential is the most parsimonious good fit for this data
    assert "exponential" in names[:2]
    assert ranked[0].r2 > 0.95


def test_suggest_models_custom_candidates():
    rng = np.random.default_rng(4)
    t = np.linspace(0, 10, 150)
    y = 3.0 / (1 + np.exp(-0.8 * (t - 5))) + rng.normal(0, 0.05, t.size)
    ranked = suggest_models(t, y, candidates=[models.logistic(), models.linear()])
    assert ranked[0].name == "logistic"


def test_suggest_models_include_exclude_filter():
    rng = np.random.default_rng(3)
    t = np.linspace(0.5, 6, 200)
    y = 2.0 * np.exp(0.5 * t) + rng.normal(0, 1.0, t.size)
    # exclude by category prunes the whole 'growth' family from the shortlist
    names = [s.name for s in suggest_models(t, y, exclude=["growth"])]
    assert "exponential" not in names and "exp_growth_offset" not in names
    # exclude by name drops just that family
    names = [s.name for s in suggest_models(t, y, exclude=["exponential"])]
    assert "exponential" not in names
    # include restricts the search to the named families only
    names = [s.name for s in suggest_models(t, y, include=["linear", "quadratic"])]
    assert set(names) <= {"linear", "quadratic"}


@pytest.mark.parametrize("name,fn,kw", [
    ("first_order", lambda t: 3.0 * (1 - np.exp(-t / 1.2)), {}),
    ("hill", lambda t: 5.0 * t**2 / (3.0**2 + t**2), {}),
    ("weibull_cdf", lambda t: 2.0 * (1 - np.exp(-(t / 3.0) ** 1.5)), {}),
    ("logarithmic", lambda t: 1.0 + 2.0 * np.log(t + 1), {}),
])
def test_expanded_families_self_seed_and_fit(name, fn, kw):
    rng = np.random.default_rng(7)
    t = np.linspace(0.1, 10, 200)
    y = fn(t) + rng.normal(0, 0.02 * (np.ptp(fn(t)) + 1e-9), t.size)
    r = getattr(models, name)().fit(t, y, **kw)
    from dtfit.diagnostics import fit_report
    assert fit_report(r, t, y)["r2"] > 0.97


def test_fourier_series_fits_harmonics():
    rng = np.random.default_rng(8)
    t = np.linspace(0, 4 * np.pi, 400)
    w = 1.0
    y = (np.sin(w * t) + 0.4 * np.sin(3 * w * t) + 0.2 * np.cos(2 * w * t)
         + rng.normal(0, 0.03, t.size))
    r = models.fourier_series(3).fit(t, y)
    from dtfit.diagnostics import fit_report
    assert fit_report(r, t, y)["r2"] > 0.97


def test_catalog_is_comprehensive():
    # every catalogued family parses and carries a category tag
    cats = {m.category for m in models.all_models()}
    assert {"trend", "growth", "decay", "sigmoid", "saturating",
            "peak", "oscillatory"} <= cats
    assert len(models.CATALOG) >= 20


def test_suggest_shortlist_prunes_by_shape():
    from dtfit.models._suggest import _shortlist
    rng = np.random.default_rng(9)
    t = np.linspace(0, 6 * np.pi, 300)
    y = np.sin(2 * t) + rng.normal(0, 0.05, t.size)
    short = {m.name for m in _shortlist(t, y)}
    # oscillatory data: keep the cycles, drop the peak/sigmoid families
    assert "sine" in short
    assert "gaussian" not in short and "logistic" not in short
    assert len(short) < len(models.CATALOG)


def test_model_from_raw_expr():
    m = Model("a*exp(b*x)", "x", name="exp", shape="bulk")
    rng = np.random.default_rng(5)
    x = np.linspace(0, 3, 200)
    y = 1.5 * np.exp(0.6 * x) + rng.normal(0, 0.02, x.size)
    r = m.fit(x, y, p0=[1.0, 1.0])
    assert r.params["b"] == pytest.approx(0.6, abs=0.1)


def test_seed_arrays_keeps_partial_bounds():
    # A seeder that bounds one parameter and leaves another unbounded must keep
    # both pairs. The solvers skip the global (DE) stage on infinite bounds,
    # and the local solve still honours the finite ones.
    def seed(x, y):
        return {"a": (2.0, -np.inf, np.inf), "s": (0.5, 1e-3, np.inf)}

    m = Model("a*exp(-(x/s)**2)", name="bump", shape="bulk", seeder=seed)
    x = np.linspace(0, 1, 20)
    p0, bounds = m._seed_arrays(x, np.exp(-x))
    assert p0 == [2.0, 0.5]
    assert bounds == [(-np.inf, np.inf), (1e-3, np.inf)]


def test_seed_arrays_fully_unbounded_maps_to_none():
    # An all-(-inf, inf) seed carries no constraint. Mapping it to None keeps
    # the solver on the unconstrained (LM) path instead of the bounded one.
    def seed(x, y):
        return {"a": (2.0, -np.inf, np.inf)}

    m = Model("a*x", name="lin0", shape="bulk", seeder=seed)
    p0, bounds = m._seed_arrays(np.linspace(0, 1, 5), np.linspace(0, 2, 5))
    assert p0 == [2.0]
    assert bounds is None


def test_partial_bounds_survive_to_solver(monkeypatch):
    # Seeded bounds, partly infinite or not, reach the estimator untouched
    # through Model.fit's route.
    captured = {}

    def fake_fit(model, data, var=None, **kw):
        captured.update(kw)
        return "sentinel"

    monkeypatch.setattr("dtfit.models._model.fit", fake_fit)

    def seed(x, y):
        return {"a": (1.0, -np.inf, np.inf), "s": (0.5, 1e-3, np.inf)}

    m = Model("a*exp(-(x/s)**2)", name="bump", shape="bulk", seeder=seed)
    x = np.linspace(0, 1, 20)
    assert m.fit(x, np.exp(-x)) == "sentinel"
    assert captured["p0"] == [1.0, 0.5]
    assert captured["bounds"] == [(-np.inf, np.inf), (1e-3, np.inf)]


def test_partial_bounds_positivity_guard_respected():
    # Growing data would drive an unconstrained decay rate negative; the
    # seeder's b > 0 guard must survive even though 'a' is unbounded.
    def seed(x, y):
        return {"a": (1.0, -np.inf, np.inf), "b": (0.3, 1e-9, np.inf)}

    m = Model("a*exp(-b*x)", name="decay", shape="bulk", seeder=seed)
    x = np.linspace(0.0, 3.0, 120)
    y = 2.0 * np.exp(0.5 * x)
    r = m.fit(x, y, basis="legendre")
    assert r.params["b"] >= 0.0


def test_suggest_models_warns_on_failed_candidate():
    # A candidate whose fit errors is skipped with a UserWarning naming it,
    # never silently. The surviving families still get ranked.
    def bad_seed(x, y):
        raise RuntimeError("boom")

    bad = Model("a*exp(b*x)", name="bad", shape="bulk", seeder=bad_seed)
    rng = np.random.default_rng(6)
    t = np.linspace(0, 5, 100)
    y = 2.0 * t + 1.0 + rng.normal(0, 0.05, t.size)
    with pytest.warns(UserWarning, match="candidate model 'bad' failed: boom"):
        ranked = suggest_models(t, y, candidates=[bad, models.linear()])
    names = [s.name for s in ranked]
    assert "bad" not in names
    assert "linear" in names


# callable models
def _decay(x, a, b, c):
    """f(x) = a*exp(-b*x) + c, signature order (a, b, c)."""
    return a * np.exp(-b * x) + c


def _line_ba(x, b, a):
    """f(x) = a*x + b, signature order (b, a), not sorted (a, b)."""
    return a * x + b


def test_callable_model_attributes_and_signature_order():
    m = Model.from_callable(_decay, name="cdecay", shape="bulk")
    assert m.is_symbolic is False
    assert m.expr is None
    assert m.func is _decay
    assert m.params == ("a", "b", "c")
    assert "callable" in repr(m).lower() or "<callable>" in repr(m)

    # signature order survives even where it differs from sorted
    m2 = Model.from_callable(_line_ba, name="cline")
    assert m2.params == ("b", "a")


def test_callable_model_recovers_params_lsi_and_auto():
    m = Model.from_callable(_decay, name="cdecay", shape="bulk")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 5, 200)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0, 0.02, x.size)
    for basis in ("legendre", "auto"):
        r = m.fit(x, y, basis=basis)
        assert r.params["a"] == pytest.approx(3.0, abs=0.1)
        assert r.params["b"] == pytest.approx(0.8, abs=0.1)
        assert r.params["c"] == pytest.approx(1.0, abs=0.1)
        # coefficients and names line up with the callable's signature order
        assert tuple(r.names) == ("a", "b", "c")


def test_callable_model_fits_via_eac():
    m = Model.from_callable(_decay, name="cdecay", shape="bulk")
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 200)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0, 0.02, x.size)
    r = m.fit(x, y, basis="block", p0=[2.5, 0.6, 0.8])
    assert r.params["a"] == pytest.approx(3.0, abs=0.2)
    assert r.params["b"] == pytest.approx(0.8, abs=0.15)
    assert r.params["c"] == pytest.approx(1.0, abs=0.2)


def test_callable_model_self_seeds_through_fit(monkeypatch):
    # A callable Model's seeder feeds p0/bounds in signature order. The raw
    # callable reaches the fitter for it to resolve.
    captured: dict = {}

    def fake_fit(model, data, var=None, **kw):
        captured["model"] = model
        captured.update(kw)
        return "sentinel"

    monkeypatch.setattr("dtfit.models._model.fit", fake_fit)

    def seed(x, y):
        return {"a": (3.0, 0.0, 10.0), "b": (0.5, 0.0, 5.0), "c": (1.0, -5.0, 5.0)}

    m = Model.from_callable(_decay, name="cdecay", seeder=seed)
    out = m.fit(np.linspace(0, 5, 50), np.ones(50), basis="legendre")
    assert out == "sentinel"
    assert captured["model"] is _decay  # the callable itself, unresolved
    assert captured["p0"] == [3.0, 0.5, 1.0]
    assert captured["bounds"] == [(0.0, 10.0), (0.0, 5.0), (-5.0, 5.0)]


def test_callable_model_self_seed_end_to_end():
    # No explicit p0: the seeder alone drives a good fit.
    def seed(x, y):
        return {"a": (float(y.max()), 0.0, 100.0),
                "b": (0.5, 1e-3, 10.0),
                "c": (float(y.min()), -100.0, 100.0)}

    m = Model.from_callable(_decay, name="cdecay", shape="bulk", seeder=seed)
    rng = np.random.default_rng(2)
    x = np.linspace(0, 5, 200)
    y = 3.0 * np.exp(-0.8 * x) + 1.0 + rng.normal(0, 0.02, x.size)
    r = m.fit(x, y, basis="legendre")
    assert r.params["b"] == pytest.approx(0.8, abs=0.1)
    assert r.params["a"] == pytest.approx(3.0, abs=0.15)


def test_callable_model_param_names_for_varargs():
    # A callable with a *params signature is not introspectable: names must be
    # supplied, and they become the canonical order.
    def poly(x, *c):
        return c[0] + c[1] * x + c[2] * x**2

    m = Model.from_callable(poly, names=["a0", "a1", "a2"], shape="bulk")
    assert m.params == ("a0", "a1", "a2")
    x = np.linspace(-2, 2, 120)
    y = 1.0 + 0.5 * x + 2.0 * x**2
    r = m.fit(x, y, basis="legendre", p0=[0.0, 0.0, 0.0])
    assert r.params["a0"] == pytest.approx(1.0, abs=0.05)
    assert r.params["a1"] == pytest.approx(0.5, abs=0.05)
    assert r.params["a2"] == pytest.approx(2.0, abs=0.05)


def test_callable_model_constructor_accepts_callable_with_param_names():
    # The constructor takes a callable and a param_names override too, not
    # only from_callable.
    m = Model(_decay, "t", name="cdecay", param_names=("a", "b", "c"))
    assert m.is_symbolic is False
    assert m.var == "t"
    assert m.params == ("a", "b", "c")

    # A param_names count that disagrees with the signature is rejected.
    with pytest.raises(ValueError):
        Model.from_callable(_decay, names=["a", "b"])  # signature has 3


def test_callable_model_add_raises():
    # Composition requires symbolic operands. A callable in either position,
    # or on both, raises a clear TypeError.
    m = Model.from_callable(_decay, name="cdecay")
    sym = Model("q*x", "x", name="lin")
    with pytest.raises(TypeError, match="cannot compose a callable model"):
        _ = m + sym
    with pytest.raises(TypeError, match="cannot compose a callable model"):
        _ = sym + m
    with pytest.raises(TypeError, match="cannot compose a callable model"):
        _ = m + m


def test_symbolic_model_behavior_unchanged():
    # A symbolic (string) Model: is_symbolic set, the expression string kept,
    # no callable, and sorted parameter order.
    m = Model("a*exp(b*x)", "x", name="exp", shape="bulk")
    assert m.is_symbolic is True
    assert m.expr == "a*exp(b*x)"
    assert m.func is None
    assert m.params == ("a", "b")  # sorted
    # composition works between two symbolic models
    combined = m + Model("c*x", "x", name="lin")
    assert combined.is_symbolic is True
    assert "c" in combined.params


# custom model registration
@pytest.fixture
def clean_catalog():
    # Registration mutates the module-global CATALOG in place. Snapshot and
    # restore it so one test's registrations never leak into another.
    from dtfit.models import _catalog
    snapshot = dict(_catalog.CATALOG)
    try:
        yield
    finally:
        _catalog.CATALOG.clear()
        _catalog.CATALOG.update(snapshot)


def _make_myline():
    # A zero-arg factory returning a fresh custom Model. Its category defaults
    # to 'general', outside the recommender's known shape vocabulary.
    def seed(x, y):
        a1, a0 = np.polyfit(x, y, 1)
        return {"a0": (float(a0), -np.inf, np.inf),
                "a1": (float(a1), -np.inf, np.inf)}
    return Model("a0 + a1*x", name="myline", shape="bulk", seeder=seed)


def test_register_adds_to_catalog_and_all_models(clean_catalog):
    assert "myline" not in models.CATALOG
    models.register("myline", _make_myline)
    assert "myline" in models.CATALOG
    # all_models() covers custom families as well as builtins
    assert any(m.name == "myline" for m in models.all_models())
    # a fresh instance every call, since the catalog stores factories
    a, b = models.CATALOG["myline"](), models.CATALOG["myline"]()
    assert a is not b and a.name == b.name == "myline"


def test_register_makes_model_visible_to_suggest(clean_catalog):
    models.register("myline", _make_myline)
    rng = np.random.default_rng(11)
    t = np.linspace(0, 10, 150)
    y = 2.0 * t + 1.0 + rng.normal(0, 0.05, t.size)
    ranked = suggest_models(t, y)
    names = [s.name for s in ranked]
    # the registered family is considered, and it is the true structure here
    assert "myline" in names
    top = next(s for s in ranked if s.name == "myline")
    assert top.r2 > 0.99


def test_custom_category_model_kept_in_default_shortlist(clean_catalog):
    from dtfit.models._suggest import _shortlist
    models.register("myline", _make_myline)
    rng = np.random.default_rng(9)
    t = np.linspace(0, 6 * np.pi, 300)
    y = np.sin(2 * t) + rng.normal(0, 0.05, t.size)  # strongly oscillatory
    short = {m.name for m in _shortlist(t, y)}
    # oscillatory data prunes the peak/sigmoid builtins, but the custom
    # 'general'-category family has no shape signal and must always survive.
    assert "myline" in short
    assert "gaussian" not in short and "logistic" not in short


def test_register_collision_raises_and_overwrite_replaces(clean_catalog):
    models.register("myfam", _make_myline)
    # a second registration under the same name collides
    with pytest.raises(ValueError, match="already exists"):
        models.register("myfam", _make_myline)
    # a builtin name also collides (without overwrite)
    with pytest.raises(ValueError, match="already exists"):
        models.register("linear", _make_myline)

    def other():
        return Model("a*x", name="myfam2", shape="bulk")

    models.register("myfam", other, overwrite=True)
    assert models.CATALOG["myfam"] is other


def test_register_overwrite_builtin_warns(clean_catalog):
    # Shadowing a shipped family is allowed with overwrite=True, but flagged.
    with pytest.warns(UserWarning, match="builtin"):
        models.register("linear", _make_myline, overwrite=True)
    assert models.CATALOG["linear"] is _make_myline


def test_register_rejects_bad_inputs(clean_catalog):
    with pytest.raises(ValueError, match="non-empty"):
        models.register("", _make_myline)
    with pytest.raises(ValueError, match="non-empty"):
        models.register("   ", _make_myline)
    with pytest.raises(TypeError):
        models.register("notcallable", 123)  # factory not callable
    with pytest.raises(TypeError, match="must return a Model"):
        models.register("badret", lambda: 42)  # returns a non-Model
    with pytest.raises(ValueError, match="raised when called"):
        models.register("boom", _boom_factory)  # factory raises
    # none of the rejected names leaked into the catalog
    for nm in ("", "   ", "notcallable", "badret", "boom"):
        assert nm not in models.CATALOG


def _boom_factory():
    raise RuntimeError("nope")


def test_unregister_removes_custom_and_refuses_builtin(clean_catalog):
    models.register("myfam", _make_myline)
    assert "myfam" in models.CATALOG
    models.unregister("myfam")
    assert "myfam" not in models.CATALOG
    # re-registration after unregister works (name is free again)
    models.register("myfam", _make_myline)
    assert "myfam" in models.CATALOG
    # builtins are protected, and unknown names error
    with pytest.raises(ValueError, match="builtin"):
        models.unregister("linear")
    with pytest.raises(KeyError):
        models.unregister("does_not_exist")


# either core type as the fit input
def test_model_fit_accepts_an_original_and_an_image():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 300)
    y = 8.0 / (1.0 + np.exp(-0.9 * (x - 5.0))) + rng.normal(0, 0.05, x.size)
    orig = Original(x, y)
    m = models.logistic()
    from_pair = m.fit(x, y)
    from_original = m.fit(orig)
    np.testing.assert_allclose(from_pair.coeffs, from_original.coeffs)
    img = orig.image("legendre", 16)
    # basis and order are ignored once data is already an Image: the
    # image's own basis and order are what the fit runs on regardless.
    from_image = m.fit(img, basis="block", order=99)
    assert from_image.basis_name == "legendre"
    assert from_image.image_order == 16
    for name, truth in (("L", 8.0), ("k", 0.9), ("x0", 5.0)):
        assert from_image.params[name] == pytest.approx(truth, rel=0.05)


def test_model_fit_rejects_a_second_argument_with_a_core_type():
    x = np.linspace(0.0, 5.0, 60)
    y = np.exp(0.3 * x)
    orig = Original(x, y)
    with pytest.raises(TypeError, match="carries its own values"):
        models.exponential().fit(orig, y)
    with pytest.raises(TypeError, match="carries its own values"):
        models.exponential().fit(orig.image("legendre", 8), y)


def test_model_fit_needs_values_with_a_bare_array():
    with pytest.raises(TypeError, match="needs the values"):
        models.exponential().fit(np.linspace(0.0, 1.0, 20))


def test_model_fit_on_an_image_rejects_the_auto_basis():
    x = np.linspace(0.0, 5.0, 120)
    img = Original(x, np.exp(0.4 * x)).image("legendre", 10)
    with pytest.raises(TypeError, match="needs the samples"):
        models.exponential().fit(img)


def test_suggest_models_accepts_an_original_and_an_image():
    """The default basis has to work on an Image too: the routing needs the
    samples an Image does not carry, so the image's own basis is what every
    candidate is fitted in. The property under test is the image path
    reproducing the sample-path ranking, not which family comes out on
    top: with two near-nested candidates in the catalog (a logistic and
    its rounded-corner tanh_step twin here), the winning name itself flips
    with the noise draw."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 300)
    y = 8.0 / (1.0 + np.exp(-0.9 * (x - 5.0))) + rng.normal(0, 0.08, x.size)
    orig = Original(x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_pair = [s.name for s in suggest_models(x, y, top=3)]
        from_original = [s.name for s in suggest_models(orig, top=3)]
        from_image = [
            s.name for s in suggest_models(orig.image("legendre", 24), top=3)
        ]
    assert from_pair == from_original == from_image


@pytest.mark.parametrize("basis", ["auto", "legendre", "block"])
def test_model_fit_forwards_the_seeded_bounds(monkeypatch, basis):
    import dtfit.models._model as _mod
    from dtfit.models._catalog import CATALOG

    seen = {}
    orig = _mod.fit

    def spy(*args, **kwargs):
        seen.setdefault("bounds", []).append(kwargs.get("bounds"))
        return orig(*args, **kwargs)

    monkeypatch.setattr(_mod, "fit", spy)
    # a logistic self-seeds bounds; every basis must forward them
    m = CATALOG["logistic"]()
    x = np.linspace(0.0, 10.0, 160)
    y = 5.0 / (1.0 + np.exp(-0.8 * (x - 5.0)))
    m.fit(x, y, basis=basis)
    assert seen["bounds"][0] is not None, "Model.fit lost the seeded bounds"
