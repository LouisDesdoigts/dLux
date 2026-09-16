"""Convert between supported angular and physical units."""

from functools import lru_cache

import zodiax as zdx
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


_DIMENSIONS = ("angle", "length", "spatial_frequency", "photon")
_ACCEPTED_UNITS = {
    None: "rad, deg, arcmin, arcsec, mas, uas, m, angstrom, photon",
    "angle": "rad, deg, arcmin, arcsec, mas, uas",
    "length": "m, angstrom",
    "spatial_frequency": "1/m, 1/mm, 1/um, 1/nm",
    "photon": "photon",
}


@lru_cache(maxsize=None)
def _unit_info(unit: str):
    """Adapt Zodiax unit information to dLux physical dimensions."""
    if isinstance(unit, zdx.Unit):
        unit = unit.unit
    info = zdx.unit_info(unit)

    if info.category == "angular" and info.power == 1:
        dimension = "angle"
    elif info.category == "cartesian" and info.power == 1:
        dimension = "length"
    elif info.category == "cartesian" and info.power == -1:
        dimension = "spatial_frequency"
    elif info.category == "photon" and info.power == 1:
        dimension = "photon"
    else:
        raise ValueError(f"Unknown unit {unit!r}.")

    return info.unit, dimension, info.factor


def _unit_dimension(unit: str) -> str:
    """Return the physical dimension of a supported unit."""
    return _unit_info(unit)[1]


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
    accepted = _ACCEPTED_UNITS.get(dimension, "")
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
        Multiplicative factor converting values to canonical SI units.

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
    source, dim_in, _ = _resolve_unit(unit_in, dimension, name)
    target, dim_out, _ = _resolve_unit(unit_out, dimension, name)
    if dim_in != dim_out:
        raise ValueError(f"Cannot convert {name} from {unit_in!r} to {unit_out!r}.")
    return zdx.convert(value, source, target) / scale


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
