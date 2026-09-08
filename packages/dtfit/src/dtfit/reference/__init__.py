"""The exact-balance ancestor of the image methods.

DSB (differential spectra balance) recovers a model's parameters by equating
its Maclaurin spectrum to the data's, order by order, and solving the
resulting system symbolically. It is the method the image core descends from
and the one the text derives the others against: it needs no basis, no
projection and no least-squares criterion, and in exchange it needs a
polynomial pre-fit of the data and an exactly identified balance.

It is not part of :func:`dtfit.fit` and has no place in the routing: the
projected estimator of :mod:`dtfit.image` supersedes it on data with noise.

From samples the call is two lines, the pre-fit and the balance::

    deg = max(find_degree(x, y, method="bic"), n_params - 1, 1)
    res = fit_dsb(np.polyfit(x, y, deg)[::-1], "a0 + a1*exp(a2*x)", "x")

The degree floor matters: below ``n_params - 1`` polynomial coefficients the
balance is underdefined and :func:`fit_dsb` raises.
"""

from .dsb import fit_dsb, find_degree

__all__ = ["fit_dsb", "find_degree"]
