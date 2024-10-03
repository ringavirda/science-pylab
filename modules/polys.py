import sympy as sp

# Poly vars
t, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = sp.symbols(
    "t, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9"
)

# Poly expressions
poly0 = c0
poly1 = c0 + c1 * t
poly2 = c0 + c1 * t + c2 * t**2
poly3 = c0 + c1 * t + c2 * t**2 + c3 * t**3
poly4 = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4
poly5 = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
poly6 = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5 + c6 * t**6
poly7 = (
    c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5 + c6 * t**6 + c7 * t**7
)
poly8 = (
    c0
    + c1 * t
    + c2 * t**2
    + c3 * t**3
    + c4 * t**4
    + c5 * t**5
    + c6 * t**6
    + c7 * t**7
    + c8 * t**8
)
poly9 = (
    c0
    + c1 * t
    + c2 * t**2
    + c3 * t**3
    + c4 * t**4
    + c5 * t**5
    + c6 * t**6
    + c7 * t**7
    + c8 * t**8
    + c9 * t**9
)

# Container
polys = (poly0, poly1, poly2, poly3, poly4, poly5, poly6, poly7, poly8, poly9)
