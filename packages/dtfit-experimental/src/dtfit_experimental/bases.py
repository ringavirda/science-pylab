"""Fourier, Chebyshev and Laguerre on dtfit's image interface.

These are dtfit.image.Basis subclasses, so any is fit through the promoted
core route ``dtfit.fit(model, data, var, basis=FourierBasis(K))``: the image
carries the Gram of the family on the sample grid, and the projected NLLS
runs on it unchanged. They are adaptations, not core; core knows only
Legendre and block. Each earns its place by a coefficient count and a Gram
conditioning on its matching signal class, not by beating Legendre on
accuracy (measured 2026-09-08, 30 seeds):

* Fourier on a 4-cycle sine: K=5 (11 coef) matches Legendre order 24
  (25 coef). A couple of harmonics carry a cycle many polynomial orders
  need.
* Chebyshev on a smooth exponential: equal recovery to Legendre at order
  16, cond(G) about half Legendre's at every order (165 vs 482 at 64) --
  a conditioning tool at high order, not an accuracy gain.
* Laguerre on a decaying exponential: order 8 (9 coef) matches Legendre
  order 24 (25 coef). The basis functions themselves decay.

The older spectral-criterion adaptation :func:`fit_lsi_basis` still exists
for its own reason (a diagonal-weighted spectral match with a period and a
pre-smoother); these are the image-projection form.

An image built in one of these bases does not round-trip through
``dtfit.image`` serialization, whose ``from_dict`` resolves the name
against core's ``_BASES``; these bases are the batch-fit path only and
carry no ``transfer``, since block transfer is a block/haar concept and
these are global families.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _C, laguerre as _Lag

from dtfit.image.bases import Basis


class FourierBasis(Basis):
    """Real Fourier basis ``{1, cos(k*ph), sin(k*ph)}`` over the domain.

    For periodic or seasonal signals. ``order`` is the number of harmonics
    K, so ``n_coef = 2K + 1``. The fundamental is one cycle across the
    domain: ``ph = pi*(u + 1)`` maps ``u`` in ``[-1, 1]`` to phase
    ``[0, 2 pi]``. A period shorter than the domain is carried by the
    higher harmonics.
    """

    name = "fourier"

    def __init__(self, order: int) -> None:
        super().__init__(order)
        self.K = int(order)

    @property
    def n_coef(self) -> int:
        return 2 * self.K + 1

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        ph = np.pi * (u + 1.0)
        cols = [np.ones_like(u)]
        cols += [np.cos(k * ph) for k in range(1, self.K + 1)]
        cols += [np.sin(k * ph) for k in range(1, self.K + 1)]
        return np.column_stack(cols)


class ChebyshevBasis(Basis):
    """Chebyshev polynomials ``T_0 .. T_order`` on ``[-1, 1]``.

    Same span as Legendre on a smooth signal; the Gram on the sample grid
    is better conditioned at high order (about half Legendre's). ``order``
    is the degree, ``n_coef = order + 1``.
    """

    name = "chebyshev"

    @property
    def n_coef(self) -> int:
        return self.order + 1

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        return _C.chebvander(np.asarray(u, dtype=float), self.order)


class LaguerreBasis(Basis):
    """Decaying Laguerre functions ``L_j(t) e^{-t/2}`` for transients.

    For decay or relaxation signals. ``order`` is the degree,
    ``n_coef = order + 1``. The domain maps to ``t = (u + 1)/2 * scale`` in
    ``[0, scale]``; ``scale`` (default 6.0) sets how many decay constants
    fill the window and trades resolution near ``t = 0`` against reach.
    """

    name = "laguerre"

    def __init__(self, order: int, scale: float = 6.0) -> None:
        super().__init__(order)
        self.scale = float(scale)

    @property
    def n_coef(self) -> int:
        return self.order + 1

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        t = (np.asarray(u, dtype=float) + 1.0) / 2.0 * self.scale
        V = _Lag.lagvander(t, self.order)
        return V * np.exp(-t / 2.0)[:, None]

    def to_dict(self) -> dict[str, object]:
        d = super().to_dict()
        d["scale"] = self.scale
        return d
