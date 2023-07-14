import jax.numpy as np
from jax import Array

__all__ = [
    "rad_to_arcsec",
    "rad_to_deg",
    "rad_to_arcmin",
    "deg_to_rad",
    "deg_to_arcmin",
    "deg_to_arcsec",
    "arcmin_to_rad",
    "arcmin_to_deg",
    "arcmin_to_arcsec",
    "arcsec_to_rad",
    "arcsec_to_deg",
    "arcsec_to_arcmin",
]


# Radians to:
def rad_to_arcsec(values: Array) -> Array:
    """
    Converts the inputs values from radians to arcseconds.

    Can also be imported as r2s.

    Parameters
    ----------
    values : Array, radians
        The input values in units of radians to be converted into arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * 3600 * 180 / np.pi


def rad_to_deg(values: Array) -> Array:
    """
    Converts the inputs values from radians to degrees.

    Can also be imported as r2d.

    Parameters
    ----------
    values : Array, radians
        The input values in units of radians to be converted into degrees.

    Returns
    -------
    values : Array, degrees
        The input values converted into degrees.
    """
    return values * 180 / np.pi


def rad_to_arcmin(values: Array) -> Array:
    """
    Converts the inputs values from radians to arcminutes.

    Can also be imported as r2m.

    Parameters
    ----------
    values : Array, radians
        The input values in units of radians to be converted into arcminutes.

    Returns
    -------
    values : Array, arcminutes
        The input values converted into arcminutes.
    """
    return values * 60 * 180 / np.pi


# Degrees to:
def deg_to_rad(values: Array) -> Array:
    """
    Converts the inputs values from degrees to radians.

    Can also be imported as d2r.

    Parameters
    ----------
    values : Array, degrees
        The input values in units of degrees to be converted into radians.

    Returns
    -------
    values : Array, radians
        The input values converted into radians.
    """
    return values * np.pi / 180


def deg_to_arcmin(values: Array) -> Array:
    """
    Converts the inputs values from degrees to arcminutes.

    Can also be imported as d2m.

    Parameters
    ----------
    values : Array, degrees
        The input values in units of degrees to be converted into arcminutes.

    Returns
    -------
    values : Array, arcminutes
        The input values converted into arcminutes.
    """
    return values * 60


def deg_to_arcsec(values: Array) -> Array:
    """
    Converts the inputs values from degrees to arcseconds.

    Can also be imported as d2s.

    Parameters
    ----------
    values : Array, degrees
        The input values in units of degrees to be converted into arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * 3600


# Arcminutes to:
def arcmin_to_rad(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to radians.

    Can also be imported as m2r.

    Parameters
    ----------
    values : Array, arcminutes
        The input values in units of arcminutes to be converted into radians.

    Returns
    -------
    values : Array, radians
        The input values converted into radians.
    """
    return values * np.pi / (60 * 180)


def arcmin_to_deg(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to degrees.

    Can also be imported as m2d.

    Parameters
    ----------
    values : Array, arcminutes
        The input values in units of arcminutes to be converted into degrees.

    Returns
    -------
    values : Array, degrees
        The input values converted into degrees.
    """
    return values / 60


def arcmin_to_arcsec(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to arcseconds.

    Can also be imported as m2s.

    Parameters
    ----------
    values : Array, arcminutes
        The input values in units of arcminutes to be converted into
        arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * 60


# Arcseconds to:
def arcsec_to_rad(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to radians.

    Can also be imported as s2r.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into radians.

    Returns
    -------
    values : Array, radians
        The input values converted into radians.
    """
    return values * np.pi / (3600 * 180)


def arcsec_to_deg(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to degrees.

    Can also be imported as s2d.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into degrees.

    Returns
    -------
    values : Array, degrees
        The input values converted into degrees.
    """
    return values / 3600


def arcsec_to_arcmin(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to arcminutes.

    Can also be imported as s2m.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into
        arcminutes.

    Returns
    -------
    values : Array, arcminutes
        The input values converted into arcminutes.
    """
    return values / 60
