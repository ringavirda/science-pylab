"""Experimental structural adaptations of EAC and LSI.

The distribution has two tiers. The library tier is this package
(``import dtfit_experimental``): a small importable surface of adaptations
plus the backend helpers, staging code that may graduate into stable
``dtfit``. The study tier is :mod:`dtfit_experimental.experiments`, the
per-case and per-domain validation suite each adaptation is measured in.
That tier is a research tree rather than an API: exempt from the mypy gate,
ruff-relaxed, driven with ``python -m dtfit_experimental.experiments...``
instead of imported. Nothing in the library tier imports from it.
:mod:`dtfit_experimental.streaming` is a third, narrower surface:
``FilterBank`` and ``FusedChiSquareDetector``, experiment tooling over
``dtfit``'s filters rather than library API, moved here with the
experiments that use them.

The adaptations are new ways to compose the differential-transformation
fitting methods of :mod:`dtfit`, grounded in the methods' own math: linearity
of integration, orthogonal-basis projection, additive areas. They are
prototyped here and evaluated across the experiment suite; whatever holds up
on two or more domains is promoted into stable ``dtfit``, where it then lives
physically rather than being re-imported from here.

    from dtfit_experimental import (
        fit_lsi_basis,        # #2 pluggable orthogonal basis (Fourier/...)
        fit_joint,            # #4 joint shared-parameter multi-channel fit
        boosted_fit,          # #5 stage-wise residual boosting
        InformationFilter,    # inverse-covariance (info-form) fusion primitive
    )

These signatures may change until promotion. Each of the four is here for its
own reason. ``fit_lsi_basis`` buys vocabulary, not accuracy: a Fourier or
Laguerre basis makes periodic and decay models expressible, yet recovery did
not improve and the LTSF benchmark went against it. ``fit_lsi`` still
hard-codes Legendre for that reason. ``boosted_fit`` is a genuine win, but on
one domain only (additive trend plus season, CO2 and the like); a confirming
second domain would clear the promotion gate. ``fit_joint`` is the substantial
new solver, still under evaluation. ``InformationFilter``, the
inverse-covariance primitive whose fusion is a plain addition, is coherent and
tested but no domain study exercises it and the covariance-form ``EACFilter``
/ ``LSIFilter`` do not use it; it waits here for a sensor-fusion or embedded
domain. Measured verdicts live in ``experiments/cases/analysis``.
"""

from dtfit._core._backend import available_backends, resolve_backend, Backend
from .basis_lsi import fit_lsi_basis
from .joint import fit_joint, JointResult
from .boosting import boosted_fit, BoostedModel
from .information import InformationFilter

__all__ = [
    "fit_lsi_basis",
    "available_backends",
    "resolve_backend",
    "Backend",
    "fit_joint",
    "JointResult",
    "boosted_fit",
    "BoostedModel",
    "InformationFilter",
]
