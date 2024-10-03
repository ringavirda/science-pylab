"""FFitting main interface for external access.
"""

from .common import np
from .model import ModelLite, Model
from .options import FittingOptions


def nonline_fit(
    data: np.ndarray, expr: str, var: str, options: FittingOptions = FittingOptions()
) -> Model | ModelLite:
    """Primary fitting method that utilizes developed experimental methods."""
    # Prepare data
    model = Model(data)
    # Perform fitting
    # Additional optimization
    # Final
    return model if options.full_model else model.as_lite()


def poly_fit(data: np.ndarray, rank: int) -> ModelLite:
    """Wrapper for the default `polyfit` function from `numpy`, that returns
    common for `ffitting` model type.

    Arguments:
        data -- raw data used to train the LSM model.
        rank -- the amount of terms/coeffs the polynomial should have.

    Returns:
        ModelLite instance with fitted data.
    """

    # Create horizontal axis for calculations.
    x = np.arange(data.size)
    # Extract optimal coefficients for the chosen poly rank.
    coeffs = np.polyfit(x, data, rank - 1)
    # Construct standard polynomial model.
    polynomial = np.poly1d(coeffs)

    return ModelLite(
        data_raw=data, data_fit=polynomial(x), model=polynomial, coeffs=coeffs
    )
