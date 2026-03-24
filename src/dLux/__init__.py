"""Public package interface for dLux."""

import importlib.metadata
from ._exports import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    layers,
    optical_systems,
    detectors,
    instruments,
    sources,
    spectra,
    wavefronts,
    psfs,
    transformations,
    coordinates,
)

_modules = (
    optical_systems,
    detectors,
    sources,
    spectra,
    instruments,
    layers,
    wavefronts,
    psfs,
    transformations,
    coordinates,
)

_module_names = [
    "layers",
    "utils",
    "optical_systems",
    "detectors",
    "instruments",
    "sources",
    "spectra",
    "wavefronts",
    "psfs",
    "transformations",
    "coordinates",
]
__all__ = _module_names + reexport(_modules, globals())
