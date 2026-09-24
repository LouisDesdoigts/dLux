"""Retained angular conversion functions; new code can use ``convert``."""

from jax import Array

from ..numerical.units import (
    canonical_unit,
    unit_factor,
    unit_factor_to_rad as _unit_factor_to_rad,
    convert,
)

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


def unit_factor_to_rad(unit: str):
    """Retain the released angular-only validation and error messages."""
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    if not unit.strip():
        raise ValueError("unit cannot be empty.")
    try:
        return _unit_factor_to_rad(unit)
    except ValueError as error:
        raise ValueError(f"Unknown angular unit '{unit}'.") from error
