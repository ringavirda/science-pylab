"""Experiment tooling over the streaming filters: a bank of independent
filters driven in lockstep, and a fused fault test over their innovations.
The fused test on the stable filters is a one-line reduction of their
``nis_`` attributes; these classes serve the experiment harnesses."""

from ._bank import FilterBank, FusedChiSquareDetector

__all__ = ["FilterBank", "FusedChiSquareDetector"]
