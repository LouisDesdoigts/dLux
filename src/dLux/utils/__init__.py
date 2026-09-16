"""Public utility functions used across dLux modules."""

from .helpers import reexport

# Import as modules
from . import (
    coordinates,
    helpers,
    interpolation,
    math,
    optics,
    units,
    array_ops,
    polynomials,
    source,
    geometry,
    misc,
    norms,
    apertures,
    fourier,
    polarisation,
    source_estimation,
)

_modules = (
    coordinates,
    helpers,
    interpolation,
    units,
    optics,
    math,
    array_ops,
    polynomials,
    source,
    geometry,
    misc,
    norms,
    apertures,
    fourier,
    polarisation,
    source_estimation,
)

_module_names = [
    "coordinates",
    "helpers",
    "interpolation",
    "math",
    "optics",
    "units",
    "array_ops",
    "polynomials",
    "source",
    "geometry",
    "misc",
    "norms",
    "apertures",
    "fourier",
    "polarisation",
    "source_estimation",
]
__all__ = _module_names + reexport(_modules, globals())

# Numerical objects depend on the already available array utilities.
from . import grids, propagation, sequential
from .propagation import _legacy

__all__ += ["grids", "propagation"]
__all__ += reexport((grids, propagation, sequential), globals())

# Parametric expressions build on the array utilities and sampling grids above.
from . import parametrics

__all__ += ["parametrics"] + reexport((parametrics,), globals())

# Preserve the released array-call interfaces at the utility root. Prepared
# FFT/MFT objects are available explicitly in dlu.propagation.
__all__ += reexport((_legacy,), globals())
__all__ = list(dict.fromkeys(__all__))
