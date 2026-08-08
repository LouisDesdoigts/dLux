"""Public package interface for dLux."""

import importlib.metadata
from .utils.helpers import reexport

__version__ = importlib.metadata.version("dLux")

from . import (
    fields,
    parametric,
    layers,
    systems,
    sources,
    grids,
    builders,
    prebuilt,
    compatibility,
)

_modules = (
    parametric,
    layers,
    systems,
    sources,
    fields,
    grids,
    builders,
    prebuilt,
    compatibility,
)

_module_names = [
    "parametric",
    "layers",
    "utils",
    "systems",
    "sources",
    "fields",
    "grids",
    "builders",
    "prebuilt",
    "compatibility",
]
__all__ = _module_names + reexport(_modules, globals())
