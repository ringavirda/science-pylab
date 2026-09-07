"""Online / real-time parameter tracking on the window image.

:class:`ImageFilter` ingests one sample at a time, images the sliding
window in a basis and updates the parameter estimate in information form,
with bounded per-update cost; :class:`LSIFilter` and :class:`EACFilter` fix
its basis to Legendre and block, the streaming twins of :func:`fit_lsi` and
:func:`fit_eac`. :class:`DriftDetector` is the shared change-detection logic
on a whitened innovation, usable on its own or as a building block for a
filter.
"""

from .filter import ImageFilter, LSIFilter, EACFilter
from .detect import DriftDetector

__all__ = [
    "ImageFilter",
    "LSIFilter",
    "EACFilter",
    "DriftDetector",
]
