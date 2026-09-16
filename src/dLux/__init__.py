"""Public package interface for dLux."""

import importlib.metadata
from .utils import grids, parametrics
from .utils.propagation import abcd as _abcd
from .utils.helpers import reexport
from .utils import propagation as _propagation
from .utils import sequential as _sequential

__version__ = importlib.metadata.version("dLux")

from . import (
    fields,
    layers,
    systems,
    sources,
    prebuilt,
    compatibility,
)

_modules = (
    _abcd,
    _sequential,
    parametrics,
    layers,
    systems,
    sources,
    fields,
    grids,
    prebuilt,
    compatibility,
)

_module_names = [
    "propagation",
    "parametrics",
    "layers",
    "utils",
    "systems",
    "sources",
    "fields",
    "grids",
    "prebuilt",
    "compatibility",
]
__all__ = _module_names + reexport(_modules, globals())

# Keep the numerical facade distinct from the layers' propagation module export.
propagation = _propagation
del _propagation
