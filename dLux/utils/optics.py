import jax.numpy as np
from jax import Array

__all__ = [
    "wavenumber",
    "opd2phase",
    "phase2opd",
    "fringe_size",
]


def wavenumber(wavelength: float) -> float:
    """
    Calculates the wavenumber of a given wavelength.

    Parameters
    ----------
    wavelength : Array, metres
        The wavelength to calculate the wavenumber for.

    Returns
    -------
    wavenumber : Array, radians/meter
        The wavenumber of the input wavelength.
    """
    return 2 * np.pi / wavelength


def opd2phase(opd: Array, wavelength: float) -> Array:
    """
    Converts the input Optical Path Difference (opd) in units of meters to phases in
    units of radians for the given wavelength.

    Parameters
    ----------
    opd : Array, metres
        The Optical Path Difference (opd) to be converted into phase.
    wavelength : Array, metres
        The wavelength at which to calculate the phase for.

    Returns
    -------
    phase : Array, radians
        The equivalent phase value for the given opd and wavelength.
    """
    return wavenumber(wavelength) * opd


def phase2opd(phase: Array, wavelength: float) -> Array:
    """
    Converts the input phase in units of radians to the equivalent Optical Path
    Difference (OPD) in metres for the given wavelength.

    Parameters
    ----------
    phase : Array, radians
        The phase to be converted into OPD
    wavelength : Array, metres
        The wavelength at which to calculate the OPD for.

    Returns
    -------
    opd : Array, metres
        The equivalent opd value for the given phase and wavelength.
    """
    return phase / wavenumber(wavelength)


def fringe_size(
    wavelength: float, diameter: float, focal_length: float = None
) -> Array:
    """
    Calculates the linear size of the diffraction fringes.

    Parameters
    ----------
    wavelength : Array, metres
        The wavelength at which to calculate the diffraction fringe for.
    diameter : Array, metres
        The diameter of the aperture.
    focal_length : Array, float = None
        The focal length of the optical system. If none is provided, the fringe
        size is given in units of radians, else it is given in units of metres.

    Returns
    -------
    fringe_size : Array, radians, meters
        The fringe size. Has units of radians of focal length is None, else meters.
    """
    if focal_length is None:
        return wavelength / diameter
    else:
        return wavelength * focal_length / diameter
