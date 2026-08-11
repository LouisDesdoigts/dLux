"""Convert between supported angular and physical units."""

from functools import lru_cache

import jax.numpy as np
from jax import Array

__all__ = [
    "canonical_unit",
    "unit_factor",
    "unit_factor_to_rad",
    "convert",
    "rad2arcsec",
    "rad2deg",
    "rad2arcmin",
    "deg2rad",
    "deg2arcmin",
    "deg2arcsec",
    "arcmin2rad",
    "arcmin2deg",
    "arcmin2arcsec",
    "arcsec2rad",
    "arcsec2deg",
    "arcsec2arcmin",
]


# Canonical units store their physical dimension and canonical-SI factor
_UNITS = {
    "rad": ("angle", 1.0),
    "deg": ("angle", np.pi / 180.0),
    "arcmin": ("angle", np.pi / (180.0 * 60.0)),
    "arcsec": ("angle", np.pi / (180.0 * 3600.0)),
    "mas": ("angle", np.pi / (180.0 * 3.6e6)),
    "uas": ("angle", np.pi / (180.0 * 3.6e9)),
    "m": ("length", 1.0),
    "angstrom": ("length", 1e-10),
    "photon": ("photon", 1.0),
}

# Aliases are resolved before any prefix parsing
_ALIASES = {
    "radian": "rad",
    "radians": "rad",
    "degree": "deg",
    "degrees": "deg",
    "am": "arcmin",
    "arcmins": "arcmin",
    "arcminute": "arcmin",
    "arcminutes": "arcmin",
    "as": "arcsec",
    "arcsecs": "arcsec",
    "arcsecond": "arcsec",
    "arcseconds": "arcsec",
    "metre": "m",
    "metres": "m",
    "meter": "m",
    "meters": "m",
    "a": "angstrom",
    "aa": "angstrom",
    "ångström": "angstrom",
    "angstroms": "angstrom",
    "photons": "photon",
}

_PREFIXES = {"G": 1e9, "M": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9}
_DIMENSIONS = ("angle", "length", "photon")


@lru_cache(maxsize=None)
def _unit_info(unit: str):
    """Resolve a unit into canonical spelling, dimension, and SI factor."""
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    canonical = unit.strip()
    if not canonical:
        raise ValueError("unit cannot be empty.")

    # Resolve direct canonical units and aliases
    canonical = _ALIASES.get(canonical.lower(), canonical)
    if canonical in _UNITS:
        dimension, factor = _UNITS[canonical]
        return canonical, dimension, factor

    # Resolve the supported short-form prefixes
    prefix = canonical[0]
    base = _ALIASES.get(canonical[1:].lower(), canonical[1:])
    if prefix in _PREFIXES and base in _UNITS:
        dimension, factor = _UNITS[base]
        return prefix + base, dimension, _PREFIXES[prefix] * factor
    raise ValueError(f"Unknown unit {unit!r}.")


def _validate_dimension(unit, resolved, dimension, name):
    """Validate a resolved unit dimension with contextual errors."""
    if dimension is not None and dimension not in _DIMENSIONS:
        raise ValueError(f"Unknown physical dimension {dimension!r}.")
    if dimension is not None and resolved != dimension:
        raise ValueError(
            f"{name} must have dimension {dimension!r}; received {unit!r} "
            f"({resolved})."
        )


def _unit_error(unit, dimension, name):
    """Build an accepted-value error for one physical dimension."""
    if dimension is None:
        accepted = ", ".join(_UNITS)
    else:
        accepted = ", ".join(
            key for key, (kind, _) in _UNITS.items() if kind == dimension
        )
    return ValueError(
        f"Unknown {name} {unit!r}. Accepted units include {accepted} and supported "
        "short-form prefixes."
    )


def _resolve_unit(unit, dimension, name):
    """Resolve and validate a unit with contextual public errors."""
    try:
        info = _unit_info(unit)
    except ValueError as error:
        raise _unit_error(unit, dimension, name) from error
    _validate_dimension(unit, info[1], dimension, name)
    return info


def canonical_unit(unit: str, *, dimension=None, name="unit") -> str:
    """Validate a physical unit and return its canonical spelling.

    Parameters
    ----------
    unit : str
        Unit name or supported alias.
    dimension : str or None
        Required physical dimension, such as ``"angle"`` or ``"length"``.
    name : str
        Input name used in error messages.

    Returns
    -------
    unit : str
        Canonical unit spelling.

    Raises
    ------
    ValueError
        If the unit is unknown or has the wrong physical dimension.
    """
    return _resolve_unit(unit, dimension, name)[0]


@lru_cache(maxsize=None)
def unit_factor_to_rad(unit: str):
    """Return the factor satisfying ``value_in_unit * factor = value_in_radians``."""
    return unit_factor(unit, dimension="angle")


def unit_factor(unit: str, *, dimension=None, name="unit"):
    """Return the factor converting a supported unit to canonical SI units.

    Parameters
    ----------
    unit : str
        Unit name or supported alias.
    dimension : str or None
        Required physical dimension, such as ``"angle"`` or ``"length"``.
    name : str
        Input name used in error messages.

    Returns
    -------
    factor : float
        Multiplicative factor converting angles to radians or lengths to metres.

    Raises
    ------
    ValueError
        If the unit is unknown or has the wrong physical dimension.
    """
    return _resolve_unit(unit, dimension, name)[2]


def convert(
    value,
    unit_in: str,
    unit_out: str,
    scale: float = 1.0,
    *,
    dimension=None,
    name="unit",
):
    """Convert compatible physical units with optional output scaling.

    Parameters
    ----------
    value : Array
        Values expressed in ``unit_in``.
    unit_in : str
        Input unit name or supported alias.
    unit_out : str
        Output unit name or supported alias.
    scale : float
        Additional scale applied to the output unit.
    dimension : str or None
        Required physical dimension, such as ``"angle"`` or ``"length"``.
    name : str
        Input name used in error messages.

    Returns
    -------
    values : Array
        Values expressed in ``scale * unit_out``.

    Raises
    ------
    ValueError
        If either unit is unknown or the units have incompatible dimensions.
    """
    _, dim_in, fin = _resolve_unit(unit_in, dimension, name)
    _, dim_out, fout = _resolve_unit(unit_out, dimension, name)
    if dim_in != dim_out:
        raise ValueError(f"Cannot convert {name} from {unit_in!r} to {unit_out!r}.")
    return value * (fin / fout) / scale


def rad2arcsec(values: Array) -> Array:
    """Convert values from radians to arcseconds."""
    return convert(values, "rad", "arcsec")


def rad2deg(values: Array) -> Array:
    """Convert values from radians to degrees."""
    return convert(values, "rad", "deg")


def rad2arcmin(values: Array) -> Array:
    """Convert values from radians to arcminutes."""
    return convert(values, "rad", "arcmin")


def deg2rad(values: Array) -> Array:
    """Convert values from degrees to radians."""
    return convert(values, "deg", "rad")


def deg2arcmin(values: Array) -> Array:
    """Convert values from degrees to arcminutes."""
    return convert(values, "deg", "arcmin")


def deg2arcsec(values: Array) -> Array:
    """Convert values from degrees to arcseconds."""
    return convert(values, "deg", "arcsec")


def arcmin2rad(values: Array) -> Array:
    """Convert values from arcminutes to radians."""
    return convert(values, "arcmin", "rad")


def arcmin2deg(values: Array) -> Array:
    """Convert values from arcminutes to degrees."""
    return convert(values, "arcmin", "deg")


def arcmin2arcsec(values: Array) -> Array:
    """Convert values from arcminutes to arcseconds."""
    return convert(values, "arcmin", "arcsec")


def arcsec2rad(values: Array) -> Array:
    """Convert values from arcseconds to radians."""
    return convert(values, "arcsec", "rad")


def arcsec2deg(values: Array) -> Array:
    """Convert values from arcseconds to degrees."""
    return convert(values, "arcsec", "deg")


def arcsec2arcmin(values: Array) -> Array:
    """Convert values from arcseconds to arcminutes."""
    return convert(values, "arcsec", "arcmin")
