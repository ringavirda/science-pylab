"""The old map-reduce and GEMM-batched estimators on the integral operator.

Kept for the experiment notebooks until they rerun on ``ImageStream``,
then removed.
"""

from ._partitioned import PartitionedLSI, PartitionedEAC, PartitionedBatchLSI
from ._batched import fit_lsi_batched, project_spectra

__all__ = [
    "PartitionedLSI",
    "PartitionedEAC",
    "PartitionedBatchLSI",
    "fit_lsi_batched",
    "project_spectra",
]
