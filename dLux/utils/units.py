import jax.numpy as np
from jax import Array


__all__ = ["convert_cartesian", "convert_angular",
           "radians_to_arcseconds", "radians_to_degrees", 
           "radians_to_arcminutes", "degrees_to_radians", 
           "degrees_to_arcminutes", "degrees_to_arcseconds", 
           "arcminutes_to_radians", "arcminutes_to_degrees", 
           "arcminutes_to_arcseconds", "arcseconds_to_radians", 
           "arcseconds_to_degrees", "arcseconds_to_arcminutes",
           "r2s", "r2d", "r2m", "d2r", "d2m", "d2s", 
           "m2r", "m2d", "m2s", "s2r", "s2d", "s2m"]


# General conversion classes:
def convert_cartesian(values : Array, 
                      input  : str = 'meters', 
                      output : str = 'meters') -> Array:
    """
    Converts the input values from one unit to another.

    Parameters
    ----------
    values : Array
        The input values to be converted.
    input : str = 'meters'
        The input units. Must be one of 'meters', 'millimeters', or 'microns'.
    output : str = 'meters'
        The output units. Must be one of 'meters', 'millimeters', or 'microns'.
    
    Returns
    -------
    values : Array
        The input values converted into the output units.
    """
    if input not in ('meters', 'millimeters', 'microns'):
        raise ValueError("input must be 'meters', 'millimeters', or 'microns'.")
    if output not in ('meters', 'millimeters', 'microns'):
        raise ValueError("output must be 'meters', 'millimeters', or "
                         "'microns'.")
    
    if input == output:
        factor = 1
    elif input == 'meters':
        if output == 'millimeters':
            factor = 1e-3
        elif output == 'microns':
            factor = 1e-6
    elif input == 'millimeter':
        if output == 'meters':
            factor = 1e3
        elif output == 'microns':
            factor = 1e-3
    elif input == 'microns':
        if output == 'meters':
            factor = 1e6
        elif output == 'millimeters':
            factor = 1e3
    return values * factor


def convert_angular(values : Array,
                    input  : str = 'radians',
                    output : str = 'radians') -> Array:
    """
    Converts the input values from one unit to another.

    Parameters
    ----------
    values : Array
        The input values to be converted.
    input : str = 'radians'
        The input units. Must be one of 'radians', 'degrees', 'arcseconds', or
        'arcminutes'.
    output : str = 'radians'
        The output units. Must be one of 'radians', 'degrees', 'arcseconds', or
        'arcminutes'.
    
    Returns
    -------
    values : Array
        The input values converted into the output units.
    """
    if input not in ('radians', 'degrees', 'arcseconds', 'arcminutes'):
        raise ValueError(f"input must be one of 'radians', 'degrees', "
                         f"'arcseconds' or 'arcminutes'.")
    if output not in ('radians', 'degrees', 'arcseconds', 'arcminutes'):
        raise ValueError(f"output must be one of 'radians', 'degrees', "
                         f"'arcseconds' or 'arcminutes'.")

    if input == output:
        factor = 1
    elif input == 'radians':
        if output == 'degrees':
            factor = r2d(1)
        elif output == 'arcminutes':
            factor = r2m(1)
        elif output == 'arcseconds':
            factor = r2s(1)
    elif input == 'degrees':
        if output == 'radians':
            factor = d2r(1)
        elif output == 'arcminutes':
            factor = d2m(1)
        elif output == 'arcseconds':
            factor = d2s(1)
    elif input == 'arcminutes':
        if output == 'radians':
            factor = m2r(1)
        elif output == 'degrees':
            factor = m2d(1)
        elif output == 'arcseconds':
            factor = m2s(1)
    elif input == 'arcseconds':
        if output == 'radians':
            factor = s2r(1)
        elif output == 'degrees':
            factor = s2d(1)
        elif output == 'arcminutes':
            factor = s2m(1)
    return values * factor


# Radians to:
def radians_to_arcseconds(values : Array) -> Array:
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


def radians_to_degrees(values : Array) -> Array:
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


def radians_to_arcminutes(values : Array) -> Array:
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
def degrees_to_radians(values : Array) -> Array:
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


def degrees_to_arcminutes(values : Array) -> Array:
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


def degrees_to_arcseconds(values : Array) -> Array:
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
def arcminutes_to_radians(values : Array) -> Array:
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


def arcminutes_to_degrees(values : Array) -> Array:
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


def arcminutes_to_arcseconds(values : Array) -> Array:
    """
    Converts the inputs values from arcminutes to arcseconds.

    Can also be imported as m2s.

    Parameters
    ----------
    values : Array, arcminutes
        The input values in units of arcminutes to be converted into arcseconds.

    Returns
    -------
    values : Array, arcseconds
        The input values converted into arcseconds.
    """
    return values * 60


# Arcseconds to:
def arcseconds_to_radians(values : Array) -> Array:
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


def arcseconds_to_degrees(values : Array) -> Array:
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


def arcseconds_to_arcminutes(values : Array) -> Array:
    """
    Converts the inputs values from arcseconds to arcminutes.

    Can also be imported as s2m.

    Parameters
    ----------
    values : Array, arcseconds
        The input values in units of arcseconds to be converted into arcminutes.

    Returns
    -------
    values : Array, arcminutes
        The input values converted into arcminutes.
    """
    return values / 60

# Alias to simpler names
r2s = radians_to_arcseconds
r2d = radians_to_degrees
r2m = radians_to_arcminutes
d2r = degrees_to_radians
d2m = degrees_to_arcminutes
d2s = degrees_to_arcseconds
m2r = arcminutes_to_radians
m2d = arcminutes_to_degrees
m2s = arcminutes_to_arcseconds
s2r = arcseconds_to_radians
s2d = arcseconds_to_degrees
s2m = arcseconds_to_arcminutes