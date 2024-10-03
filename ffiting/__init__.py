"""Fallen Fitting Library

This package provides a variety of methods for fitting different functions and
models to data. The main focus here are nonline processes that require something
more than a regular polynomial. Library contains implementations of methods for
fitting nonline models developed using non-taylor transformations, which a part
of authors PHD dissertation. Use with care.

Interface:
    nonline_fit() -- generic function that organizes lover level actions. 
    Provides most customization in terms of fitting options.
    poly_fit() -- performs classical polynomial fitting for the given data.
    ModelLite -- a dataclass encapsulating original data, fitted model data, 
    as well as a callable version of it.
    Model -- comprehensive class that not only contains modelling data but 
    also provides a variety of analytics options for the data.
    
Required packages:
    numpy, sympy, scipy
"""

# Main exports
from .facade import nonline_fit, poly_fit
from .model import Model, ModelLite

__all__ = ["nonline_fit", "poly_fit", "Model", "ModelLite"]
