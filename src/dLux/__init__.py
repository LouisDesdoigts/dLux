"""Public package interface for dLux."""

import importlib.metadata
from ._exports import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    parametric,
    layers,
    systems,
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
    systems,
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
    "systems",
    "sources",
    "spectra",
    "wavefronts",
    "psfs",
    "coordinates",
    "abcd",
]
__all__ = _module_names + reexport(_modules, globals())
