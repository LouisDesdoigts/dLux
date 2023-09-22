import jax.numpy as np
from jax import Array

__all__ = [
    "fluxes_from_contrast",
    "positions_from_sep",
]


def fluxes_from_contrast(mean_flux: float, contrast: float) -> Array:
    """
    Computes the fluxes of a binary object given the mean flux and contrast.

    Parameters
    ----------
    mean_flux : float
        The mean flux of the binary object.
    contrast : float
        The contrast of the binary object.

    Returns
    -------
    fluxes : Array
        The flux (flux1, flux2) of the binary object.
    """
    return 2 * np.array([contrast * mean_flux, mean_flux]) / (1 + contrast)


def positions_from_sep(
    position: Array, separation: float, position_angle: float
) -> Array:
    """
    Computes the on-sky positions of a binary object given the separation and
    position angle.

    Parameters
    ----------
    position : Array, radians
        The on-sky position of the primary object.
    separation : float, radians
        The separation of the binary object.
    position_angle : float, radians
        The position angle of the binary object.

    Returns
    -------
    position : Array, radians
        The ((x, y), (x, y)) on-sky position of this object.
    """
    r, phi = separation / 2, position_angle
    sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])
    return np.array([position + sep_vec, position - sep_vec])
