"""Released array-function contracts, backed by current kernels where equivalent.

Use ``import dLux.utils.compat as dlu`` for the complete 0.15.1 namespace.
Legacy coordinate, normalisation and propagation conventions live here rather
than changing the contracts of Grid, Interp or the numerical propagator classes.
"""

from .. import reexport
from . import (
    coordinates,
    helpers,
    math,
    units,
    norms,
    interpolation,
    propagation,
    geometry,
    fourier,
    apertures,
    zernikes,
    source,
    array_ops,
    optics,
    misc,
)

_modules = (
    coordinates,
    helpers,
    math,
    units,
    norms,
    interpolation,
    propagation,
    geometry,
    fourier,
    apertures,
    zernikes,
    source,
    array_ops,
    optics,
    misc,
)
__all__ = [module.__name__.rsplit(".", 1)[-1] for module in _modules]
__all__ += reexport(_modules, globals())
