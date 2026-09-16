"""Public package interface for dLux."""

import importlib.metadata
from .utils import parametrics
from .utils.helpers import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    fields,
    layers,
    systems,
    sources,
    grids,
    prebuilt,
    compatibility,
)

_modules = (
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
