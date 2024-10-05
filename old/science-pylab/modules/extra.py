import numpy as np
import scipy as sc
import sympy as sp

from modules.classes import Model, Polynomial
from modules.utils import rse, r_sq


def poly_fit(
    y_data: np.ndarray,
    rank: int,
    pred_start: int = 0,
    pred_length: int = 0,
) -> np.ndarray:
    # Determine bounds
    if pred_length == 0:
        pred_length = y_data.size
    if pred_length < 0:
        raise ValueError("Bounds specified are invalid.")

    # Calculate coefficients
    x_data: np.ndarray = np.arange(y_data.size)
    poly_c: np.ndarray = np.polyfit(x_data, y_data, rank)
    poly = np.poly1d(poly_c)

    # Calculate values
    poly_y: np.ndarray = np.arange(
        pred_start, pred_length + pred_start, dtype=np.float64
    )
    with np.nditer(poly_y, op_flags=["readwrite"]) as it:
        for value in it:
            value[...] = poly(value)
    return poly_y


def nonline_fit(
    raw_expr: str,
    main_var: str,
    y_data: np.ndarray,
    rank: int = 0,
    pred_start: int = 0,
    pred_length: int = 0,
    raise_rank: bool = False,
    offset: bool = False,
    numeric: bool = False,
    as_model: bool = False,
) -> sp.FiniteSet | tuple[Model, Model]:
    # Determine bounds
    if pred_length == 0:
        pred_length = y_data.size
    if pred_length < 0:
        raise ValueError("Bounds specified are invalid.")

    # Construct models
    nonline: Model = Model(raw_expr, main_var)
    if rank == 0:
        rank = get_poly_rank(nonline.rank, y_data) if raise_rank else nonline.rank
        if rank != nonline.rank:
            if any(d == 0 for d in nonline.spectrum(rank)):
                rank = nonline.rank
            else:
                print(f"Model rank was raised to {rank}")

    poly: Model = Polynomial(rank, main_var)
    poly_s: list[sp.Expr] = poly.spectrum(rank, offset)

    # Generate train x
    x_train: np.ndarray = np.arange(y_data.size)
    # Calculate polynomial coeffs
    poly_c: np.ndarray = np.polyfit(x_train, y_data, rank - 1)[::-1]
    poly.apply_trained(poly_c)

    # Solve balance
    balance: list[sp.Expr] = []
    nonline_s = nonline.spectrum(rank, offset)
    if any(d == 0 for d in nonline_s):
        raise RuntimeError("Cannot solve underdetermined system.")
    for i, a in enumerate(nonline_s):
        balance.append((a - poly_s[i]).subs(poly.coeffs[i], poly_c[i]))
    if rank == len(nonline.coeffs):
        solution = sp.nonlinsolve(balance, nonline.coeffs).args[0]
    else:
        solution0 = sp.nonlinsolve(balance[: len(nonline.coeffs)], nonline.coeffs).args[
            0
        ]
        solution0 = np.array(solution0).astype(np.float64)

        for i, eq in enumerate(balance):
            balance[i] = eq.subs(sp.abc.H, 1)
        func = sp.lambdify(nonline.coeffs, balance, "scipy")

        def wrapper(args: np.ndarray) -> np.float64:
            return func(*args)

        solution = sc.optimize.least_squares(wrapper, solution0, method="lm")
        solution = solution.x

    if numeric:
        factors = nonline.coeffs.copy()
        factors.insert(0, nonline.var)
        solution = np.array(solution).astype(np.float64)
        solution = sc.optimize.curve_fit(
            sp.lambdify(factors, nonline.sym),
            x_train,
            y_data,
            p0=solution,
            maxfev=100000,
        )[0]

    nonline.apply_trained(solution)
    if as_model:
        return (nonline, poly)

    # Calculate new fitted values
    y_pred = np.arange(pred_start, pred_start + pred_length, dtype=np.float64)
    with np.nditer(y_pred, op_flags=["readwrite"]) as it:
        for value in it:
            value[...] = nonline.collapse(value)
    return y_pred


def get_poly_rank(rank: int, data: np.ndarray, rank_range: int = 12) -> int:
    rank_rse = np.zeros(rank_range)
    rank_rsq = np.zeros(rank_range)

    for i in range(rank_range):
        data_y = poly_fit(data, rank + i)
        rank_rse[i] = rse(data, data_y)
        rank_rsq[i] = r_sq(data, data_y)

    rank_rr = []
    for i in range(rank_range - 1):
        rank_rr.append(
            np.abs(rank_rse[i + 1] - rank_rse[i])
            * np.abs(rank_rsq[i + 1] - rank_rsq[i])
        )
    for i, r in enumerate(rank_rr):
        if r < 10**-6:
            return rank + i

    # data_y = poly_fit(data, rank)
    # p_rse = rse(data, data_y)
    # p_rsq = r_sq(data, data_y)
    # p_delta = 100
    # for i in range(rank_range):
    #     data_y = poly_fit(data, rank + i)
    #     c_rse = rse(data, data_y)
    #     c_rsq = r_sq(data, data_y)
    #     delta = np.abs(c_rse - p_rse) * np.abs(c_rsq - p_rsq)
    #     if delta > p_delta:
    #         return rank + i
    #     p_delta = delta

    return rank
