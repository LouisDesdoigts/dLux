"""Keep released utility imports separate from canonical module definitions."""

import sys
from types import ModuleType
from functools import partial

from .. import reexport, grids, numerical, bases, sequential
from ..propagation import FFT, MFT
from . import coordinates, geometry, helpers, polynomials, fourier, polarisation
from . import (
    math,
    units,
    norms,
    interpolation,
    propagation,
    zernikes,
    source,
    apertures,
)


def register(namespace):
    """Expose retained functions and old paths without mutating canonical modules."""
    package = namespace["__name__"]
    retained = (
        geometry,
        coordinates,
        helpers,
        polynomials,
        fourier,
        math,
        units,
        norms,
        interpolation,
        propagation,
        polarisation,
        source,
        apertures,
    )
    names = reexport(retained, namespace)

    # Old module imports have their own facades. For unchanged names, current
    # kernels supply new names alongside the released functions.
    facades = {
        "geometry": geometry,
        "coordinates": coordinates,
        "helpers": helpers,
        "polynomials": polynomials,
        "fourier": fourier,
        "polarisation": polarisation,
        "interpolation": interpolation,
        "zernikes": zernikes,
        "source": source,
        "apertures": apertures,
        "misc": bases.climb,
        "optics": namespace["optical"].phase,
    }
    for name, current, legacy in (
        ("math", numerical.math, math),
        ("units", numerical.units, units),
        ("norms", numerical.norms, norms),
        ("array_ops", grids.array_ops, None),
    ):
        facade = ModuleType(f"{package}.{name}", current.__doc__)
        facade.__all__ = reexport((current, legacy), vars(facade))
        if legacy is not None:
            facade.__getattr__ = partial(getattr, legacy)
        facades[name] = facade

    facades["array_ops"].paste = grids.paste
    facades["array_ops"].__all__.append("paste")
    for name, module in facades.items():
        namespace[name] = module
        sys.modules[f"{package}.{name}"] = module

    # The current entry points keep general leading batches and collision-safe
    # layer naming. Their exact released contracts live in compat.
    for name in ("cart2polar", "polar2cart"):
        namespace[name] = getattr(grids, name)
    for name in ("list2dictionary", "insert_layer", "remove_layer"):
        namespace[name] = getattr(sequential, name)
    namespace["eval_basis"] = bases.eval_basis
    namespace["solve_basis"] = bases.solve_basis
    namespace["FFT"] = FFT
    namespace["MFT"] = MFT
    return list(dict.fromkeys([*names, *facades]))
