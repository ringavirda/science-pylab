"""Objects needed to configure fitting functions.
"""

from dataclasses import dataclass, field
from enum import Enum


class FittingModes(Enum):
    """Contains available modes for fitting, can be specified through `FitOptions`."""

    AUTO = 0  # System will try to guess fitting method.
    DSB = 1  # Nonlinear fitting method with Differential Spectra Balance.
    DSBI = 2  # Nonlinear fitting method with DSB in Integral form.


@dataclass
class FittingOptions:
    """Object used to configure fitting methods, can change a lot in terms of execution."""

    # Choose one of existing fitting methods.
    mode: FittingModes = field(default=FittingModes.AUTO)
    # Return result as a Model instance, instead of ModelLite.
    full_model: bool = field(default=True)
    # Perform additional numeric fitting if possible, may cause additional overhead.
    numeric: bool = field(default=False)
