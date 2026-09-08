"""The image core: the original / image pair and the projected estimator."""

from .grid import Grid
from .bases import Basis, LegendreBasis, BlockBasis, make_basis, u_of
from .original import Original
from .image import Image, huber_weights, gram_whitener
from .analytics import Decay, decay, effective_order, noise_sigma
from .transfer import assemble, legendre_transfer, block_transfer
from .stream import ImageStream
from .fit import (
    fit, order_for, coverage, osc_order, fft_frequency_seed, fit_lsi,
    fit_eac,
)
from .parallel import FittingProblem, fit_many

__all__ = [
    "Grid",
    "Basis",
    "LegendreBasis",
    "BlockBasis",
    "make_basis",
    "u_of",
    "Original",
    "Image",
    "huber_weights",
    "gram_whitener",
    "Decay",
    "noise_sigma",
    "effective_order",
    "decay",
    "ImageStream",
    "assemble",
    "legendre_transfer",
    "block_transfer",
    "fit",
    "order_for",
    "coverage",
    "osc_order",
    "fft_frequency_seed",
    "fit_lsi",
    "fit_eac",
    "FittingProblem",
    "fit_many",
]
