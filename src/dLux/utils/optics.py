import jax.numpy as np
from jax import Array
from .units import unit_factor_to_rad

__all__ = [
    "wavenumber",
    "opd2phase",
    "phase2opd",
    "fringe_size",
    "tilt_opd",
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


def tilt_opd(coordinates: Array, angles: Array, unit: str = "rad") -> Array:
    """Return the linear OPD ramp produced by a two-axis wavefront tilt."""
    angles = np.asarray(angles, dtype=float)
    if angles.shape != (2,):
        raise ValueError("angles must have shape (2,).")
    angles = angles * unit_factor_to_rad(unit)
    return np.einsum("i,...ijk->...jk", angles, coordinates)
