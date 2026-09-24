"""Public package interface for dLux.

Utility classes and functions are available directly here or through ``dLux.utils``.
Current APIs take precedence over released compatibility constructors.
"""

import importlib.metadata
from . import utils, prebuilt
from . import fields, layers, systems, compat
from .systems import sources as sources

__version__ = importlib.metadata.version("dLux")

# Export utilities first so current core implementations take precedence over
# duplicate helpers retained by the utility compatibility facade.
_modules = (utils, layers, systems, prebuilt, fields)
__all__ = [
    "utils",
    "prebuilt",
    "fields",
    "layers",
    "systems",
    "sources",
    "compat",
]
__all__ += utils.reexport(_modules, globals())

# Keep module names unambiguous: optical utilities live here, optical layers in
# layers.optical, and the released coordinates import path retains its adapter.
optical = utils.optical
coordinates = compat.coordinates
__all__ = list(dict.fromkeys(__all__))

from .compat.registration import register_modules, legacy_attribute

register_modules()
__all__ += [name for name in compat.__all__ if name not in __all__]


def __getattr__(name):
    return legacy_attribute(name, compat, __name__)
