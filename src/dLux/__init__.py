"""Public package interface for dLux."""

import importlib.metadata
from ._exports import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    parametric,
    layers,
    optical_systems,
    detectors,
    sources,
    spectra,
    wavefronts,
    psfs,
    coordinates,
    abcd,
)

_modules = (
    parametric,
    layers,
    optical_systems,
    detectors,
    sources,
    spectra,
    wavefronts,
    psfs,
    coordinates,
    abcd,
)

_module_names = [
    "parametric",
    "layers",
    "utils",
    "optical_systems",
    "detectors",
    "sources",
    "spectra",
    "wavefronts",
    "psfs",
    "coordinates",
    "abcd",
]
__all__ = _module_names + reexport(_modules, globals())
