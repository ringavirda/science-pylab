import sympy as sp
import numpy as np
import pandas as pd


def dsb_fit(
    expression: str,
    var: str,
    data: np.ndarray,
    rank: int = 0,
    numeric: bool = True,
    maxfev: int = 100000,
):
    pass

if __name__ == "__main__":
    data = pd.read_excel("data/usd_2023-2024_1.xlsx", "USD")
    usd_data = np.array(data.iloc[0:161, 6].values)
    dsb_fit("a0 + a1*t + a2*sin(a3*t) + a4*cos(a5*t)", "t", usd_data)
