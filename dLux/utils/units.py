import jax.numpy as np

__all__ = ["radians_to_arcseconds", "arcseconds_to_radians"]


Array = np.ndarray


def radians_to_arcseconds(values : Array) -> Array:
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


def arcseconds_to_radians(values : Array) -> Array:
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



