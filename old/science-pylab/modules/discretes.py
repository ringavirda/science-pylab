import sympy as sp
from sympy.abc import H, k, n, w


class Discretes:
    I = sp.Piecewise((w, sp.Eq(k, 0)), (0, True))
    C = sp.Piecewise(((H**k), sp.Eq(k, n)), (0, True))
    E = ((w * H) ** k) / sp.factorial(k)
    Sin = (((w * H) ** k) / sp.factorial(k)) * sp.sin((sp.pi * k) / 2)
    Cos = (((w * H) ** k) / sp.factorial(k)) * sp.cos((sp.pi * k) / 2)