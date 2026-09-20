"""Numerical calculations and array-producing parameterisations."""

from .. import reexport

# Grid definitions also use these independent numerical building blocks.
from . import math, units, norms
from . import interpolation, polynomials
from . import gaussians, shapes, profiles

_modules = (
    math,
    units,
    norms,
    interpolation,
    polynomials,
    gaussians,
    shapes,
    profiles,
)
__all__ = reexport(_modules, globals())
