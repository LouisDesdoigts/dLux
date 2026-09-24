"""Released convenience constructors backed by current dLux objects.

Constructors translate sampling and numerical conventions; small subclasses
adapt old method return types. There is no separate released optical engine.
Unsupported structural contracts raise migration errors instead of recreating
old class hierarchies. Utilities retain released numerical functions where needed.
"""

from ..utils import compat as utils, reexport
from .warnings import warn_deprecated as warn_deprecated
from . import (
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

__version__ = "0.15.1"
_modules = (
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
__all__ = [
    "utils",
    "layers",
    "optical_systems",
    "detectors",
    "instruments",
    "sources",
    "spectra",
    "wavefronts",
    "psfs",
    "coordinates",
]
__all__ += reexport(_modules, globals())
