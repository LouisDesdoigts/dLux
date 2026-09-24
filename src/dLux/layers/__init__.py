"""Layer contracts and concrete optical, detector, and propagation layers."""

from ..utils import reexport
from . import (
    unified,
    detector,
    optical,
    sparse,
    coronagraphy,
)

_modules = (
    unified,
    detector,
    optical,
    sparse,
    coronagraphy,
)

_module_names = [
    "unified",
    "detector",
    "optical",
    "sparse",
    "coronagraphy",
]

__all__ = _module_names + reexport(_modules, globals())


def __getattr__(name):
    # Compatibility imports layers during startup; defer the reverse dependency.
    from ..compat import layers as retained
    from ..compat.registration import legacy_attribute

    return legacy_attribute(name, retained, __name__)
