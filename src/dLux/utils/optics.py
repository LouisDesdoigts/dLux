"""Convert common scalar quantities used in physical optics."""

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

__all__ = ["wavenumber", "opd2phase", "phase2opd", "fringe_size", "tilt_opd", "tilt"]


def wavenumber(wavelength: float) -> float:
    """Return ``2π / wavelength`` in radians per metre."""
    return 2 * np.pi / wavelength


def opd2phase(opd: Array, wavelength: float) -> Array:
    """Convert optical path difference in metres to phase in radians."""
    return wavenumber(wavelength) * opd


def phase2opd(phase: Array, wavelength: float) -> Array:
    """Convert phase in radians to optical path difference in metres."""
    return phase / wavenumber(wavelength)


def fringe_size(
    wavelength: float, diameter: float, focal_length: float = None
) -> Array:
    """Return ``wavelength / diameter`` in radians or focal-plane metres.

    Supplying ``focal_length`` converts the angular fringe size into a physical
    focal-plane size. All physical inputs are in metres.
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
    angles = angles * dlu.unit_factor_to_rad(unit)
    return np.einsum("i,...ijk->...jk", angles, coordinates)


def tilt(
    phasor: Array,
    coordinates: Array,
    angles: Array,
    wavelength: float,
    unit: str = "rad",
) -> Array:
    """Apply a two-axis angular tilt to a complex field."""
    opd = tilt_opd(coordinates, angles, unit)
    return phasor * np.exp(1j * opd2phase(opd, wavelength))
