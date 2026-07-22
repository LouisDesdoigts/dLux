"""Public package interface for dLux."""

import importlib.metadata
from ._exports import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    parametric,
    polynomials,
    layers,
    optical_systems,
    detectors,
    instruments,
    sources,
    spectra,
    wavefronts,
    psfs,
    coordinates,
)

_modules = (
    parametric,
    polynomials,
    optical_systems,
    detectors,
    sources,
    spectra,
    instruments,
    layers,
    wavefronts,
    psfs,
    coordinates,
)

_module_names = [
    "parametric",
    "polynomials",
    "layers",
    "utils",
    "optical_systems",
    "detectors",
    "instruments",
    "sources",
    "spectra",
    "wavefronts",
    "psfs",
    "coordinates",
]
__all__ = _module_names + reexport(_modules, globals())
