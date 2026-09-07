"""Parallel batch fitting with ``dtfit.fit_many``: correctness per backend."""

import numpy as np

from dtfit import FittingProblem, fit_many, fit_eac


def _problems(n=6, method="eac"):
    rng = np.random.default_rng(0)
    probs = []
    for i in range(n):
        a, b = 1.0 + 0.1 * i, 0.6 + 0.05 * i
        x = np.linspace(0, 1.5, 80)
        y = a * np.exp(b * x) + rng.normal(0, 0.02, x.size)
        probs.append(FittingProblem(x=x, y=y, expr="a*exp(b*t)", var="t",
                                method=method, kwargs={"p0": [1.0, 1.0]},
                                label=f"ch{i}"))
    return probs


def test_fit_many_matches_serial():
    probs = _problems()
    par = fit_many(probs, n_jobs=2, backend="loky")
    ser = fit_many(probs, n_jobs=1)
    assert len(par) == len(ser) == len(probs)
    for p, s in zip(par, ser):
        assert p.error is None and s.error is None
        np.testing.assert_allclose(p.coeffs, s.coeffs, rtol=1e-6, atol=1e-6)


def test_fit_many_matches_direct_fit():
    probs = _problems(n=3)
    res = fit_many(probs, n_jobs=1)
    for p, r in zip(probs, res):
        direct = fit_eac(p.x, p.y, p.expr, p.var, **p.kwargs)
        np.testing.assert_allclose(r.coeffs, direct.coeffs,
                                    rtol=1e-8, atol=1e-8)


def test_fit_many_threading_backend():
    probs = _problems()
    res = fit_many(probs, n_jobs=2, backend="threading")
    assert all(r.error is None for r in res)
    assert [r.label for r in res] == [p.label for p in probs]


def test_batch_result_lazy_model_predicts():
    probs = _problems(n=2)
    res = fit_many(probs, n_jobs=1)
    x = probs[0].x
    yhat = res[0].predict(x)
    assert yhat.shape == x.shape and np.all(np.isfinite(yhat))


def test_fit_many_captures_per_problem_error():
    bad = FittingProblem(x=np.linspace(0, 1, 50), y=np.ones(50),
                     expr="a*exp(b*t)", var="t", method="nope")
    res = fit_many([bad], n_jobs=1)
    assert res[0].error is not None and res[0].coeffs.size == 0


def test_fit_many_empty():
    assert fit_many([]) == []
