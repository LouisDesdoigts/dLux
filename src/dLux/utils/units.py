import jax.numpy as np
from jax import Array
from functools import lru_cache

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


# ---- base factors: (value in unit) * factor = value in radians ----
_BASE_TO_RAD = {
    "rad": 1.0,
    "deg": np.pi / 180.0,
    "arcmin": np.pi / (180.0 * 60.0),
    "arcsec": np.pi / (180.0 * 3600.0),
}

# ---- aliases (lowercase lookup only) ----
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
}

# ---- SI-like prefixes ----
_PREFIX = {
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "": 1.0,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
}


def _canon(unit: str) -> str:
    if not isinstance(unit, str):
        raise TypeError("unit must be a string.")
    u = unit.strip()
    if not u:
        raise ValueError("unit cannot be empty.")

    # resolve aliases using lowercase lookup
    return _ALIASES.get(u.lower(), u)


@lru_cache(maxsize=None)
def unit_factor_to_rad(unit: str):
    """
    Returns factor f such that:
        value_in_unit * f = value_in_radians
    """
    u = _canon(unit)

    # base unit
    if u in _BASE_TO_RAD:
        return _BASE_TO_RAD[u]

    # prefixed unit
    p = u[0]
    base = u[1:]

    if p in _PREFIX and base in _BASE_TO_RAD:
        return _PREFIX[p] * _BASE_TO_RAD[base]

    raise ValueError(f"Unknown angular unit '{unit}'.")


def convert(value, unit_in: str, unit_out: str, scale: float = 1.0):
    """
    Convert angles from unit_in to unit_out with output unit scaling.

    Returns value expressed in (scale * unit_out).

    Examples
    --------
    convert(0.2, "deg", "arcsec", 1e-3)  -> milliarcseconds
    convert(10.0, "mas", "rad")          -> radians
    convert(1.0, "rad", "mrad")          -> milliradians
    """
    fin = unit_factor_to_rad(unit_in)
    fout = unit_factor_to_rad(unit_out)
    return value * (fin / fout) / scale


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
    return convert(values, "rad", "arcsec")


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
    return convert(values, "rad", "deg")


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
    return convert(values, "rad", "arcmin")


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
    return convert(values, "deg", "rad")


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
    return convert(values, "deg", "arcmin")


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
    return convert(values, "deg", "arcsec")


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
    return convert(values, "arcmin", "rad")


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
    return convert(values, "arcmin", "deg")


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
    return convert(values, "arcmin", "arcsec")


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
    return convert(values, "arcsec", "rad")


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
    return convert(values, "arcsec", "deg")


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
    return convert(values, "arcsec", "arcmin")
