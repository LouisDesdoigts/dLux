"""Shared optical calculations, sampling grids, and numerical parameterisations.

Classes and array functions are available through ``import dLux.utils as dlu``.
Released function interfaces and old module paths are retained through compat.
FFT and MFT are numerical classes. Their released array-function interfaces live
in ``utils.compat``.
"""


def reexport(modules: tuple[object, ...], namespace: dict[str, object]) -> list[str]:
    """Re-export the public symbols from a collection of modules."""
    exported = []
    seen = set()
    for module in modules:
        for name in getattr(module, "__all__", ()):
            namespace[name] = getattr(module, name)
            if name not in seen:
                exported.append(name)
                seen.add(name)
    return exported


from . import batching, sequential
from . import numerical, grids, bases, sampled
from .grids import array_ops

__all__ = [
    "batching",
    "sequential",
    "sampled",
    "numerical",
    "grids",
    "bases",
    "array_ops",
]
__all__ += reexport((batching, sequential, sampled, numerical, bases), globals())
__all__ += reexport(grids._modules, globals())

from . import optical, propagation, astrometry, photometry

__all__ += ["optical", "propagation", "astrometry", "photometry"]
__all__ += reexport((optical, propagation, astrometry, photometry), globals())

# Compatibility owns the released import paths and functional conveniences.
from .compat.registration import register

__all__ += register(globals())
__all__ = list(dict.fromkeys(__all__))
del register
