"""Explicit old import paths; current modules and classes retain precedence."""

import sys

from . import coordinates, detectors, instruments, optical_systems, psfs, spectra
from . import wavefronts, layers
from .warnings import warn_deprecated
from ..systems import sources as current_sources

MODULES = {
    "sources": current_sources,
    "coordinates": coordinates,
    "detectors": detectors,
    "instruments": instruments,
    "optical_systems": optical_systems,
    "psfs": psfs,
    "spectra": spectra,
    "wavefronts": wavefronts,
    "layers.optical_layers": layers.optical_layers,
    "layers.optics": layers.optics,
    "layers.detector_layers": layers.detector_layers,
    "layers.unified_layers": layers.unified_layers,
    "layers.propagators": layers.propagators,
    "layers.apertures": layers.apertures,
    "layers.aberrations": layers.aberrations,
    "layers.abcd_propagators": layers.abcd_propagators,
}


def register_modules():
    """Retain released module paths without replacing current package modules."""
    for name, module in MODULES.items():
        sys.modules.setdefault(f"dLux.{name}", module)


def legacy_attribute(name, retained, package):
    """Resolve a missing public name to its released implementation with a warning."""
    if name.startswith("_") or name not in retained.__all__:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    warn_deprecated(
        f"{package}.{name}",
        f"{retained.__name__}.{name}",
        "the released constructor now builds current objects; see its migration "
        "warning",
        stacklevel=3,
    )
    return getattr(retained, name)
