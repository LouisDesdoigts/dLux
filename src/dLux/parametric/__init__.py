"""Parametric bases, reparametrisations, and physical-property models."""

from ..utils.helpers import reexport
from . import (
    bases,
    parametrics,
    parametrisations,
    polynomials,
    shapes,
    refractive,
    spectral,
)

_modules = (
    parametrics,
    parametrisations,
    bases,
    polynomials,
    shapes,
    refractive,
    spectral,
)
_module_names = [
    "parametrics",
    "parametrisations",
    "bases",
    "polynomials",
    "shapes",
    "refractive",
    "spectral",
]

__all__ = _module_names + reexport(_modules, globals())
