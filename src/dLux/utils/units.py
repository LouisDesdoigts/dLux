import jax.numpy as np
from jax import Array

__all__ = [
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


# Radians to:
def rad2arcsec(values: Array) -> Array:
    """
    Converts the inputs values from radians to arcseconds.

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


def rad2deg(values: Array) -> Array:
    """
    Converts the inputs values from radians to degrees.

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


def rad2arcmin(values: Array) -> Array:
    """
    Converts the inputs values from radians to arcminutes.

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
def deg2rad(values: Array) -> Array:
    """
    Converts the inputs values from degrees to radians.

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


def deg2arcmin(values: Array) -> Array:
    """
    Converts the inputs values from degrees to arcminutes.

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


def deg2arcsec(values: Array) -> Array:
    """
    Converts the inputs values from degrees to arcseconds.

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
def arcmin2rad(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to radians.

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


def arcmin2deg(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to degrees.

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


def arcmin2arcsec(values: Array) -> Array:
    """
    Converts the inputs values from arcminutes to arcseconds.

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
def arcsec2rad(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to radians.

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


def arcsec2deg(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to degrees.

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


def arcsec2arcmin(values: Array) -> Array:
    """
    Converts the inputs values from arcseconds to arcminutes.

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
