"""Public utility functions used across dLux modules."""

from .helpers import reexport

# Import as modules
from . import (
    abcd,
    propagation,
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
    abcd,
    propagation,
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
    "abcd",
    "propagation",
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

# Parametric expressions build on the array utilities and sampling grids.
from . import parametrics

__all__ += ["parametrics"] + reexport((parametrics,), globals())
