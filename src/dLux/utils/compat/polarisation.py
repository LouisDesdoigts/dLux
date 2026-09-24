"""Retained array interfaces backed by the Jones expression."""

from ..optical.jones import Jones, JonesMatrix

__all__ = [
    "horizontal_polariser",
    "vertical_polariser",
    "rhc_polariser",
    "lhc_polariser",
    "quarter_wave_plate",
    "half_wave_plate",
    "linear_polariser",
    "retarder",
    "apply_jones",
    "rotate_jones",
    "jones_to_stokes",
]


def horizontal_polariser():
    """Return an ideal horizontal polariser matrix."""
    return Jones.horizontal_polariser()


def vertical_polariser():
    """Return an ideal vertical polariser matrix."""
    return Jones.vertical_polariser()


def rhc_polariser():
    """Return an ideal right-circular polariser matrix."""
    return Jones.rhc_polariser()


def lhc_polariser():
    """Return an ideal left-circular polariser matrix."""
    return Jones.lhc_polariser()


def quarter_wave_plate(angle):
    """Return a quarter-wave plate matrix with fast-axis angle in radians."""
    return Jones.quarter_wave_plate(angle)


def half_wave_plate(angle):
    """Return a half-wave plate matrix with fast-axis angle in radians."""
    return Jones.half_wave_plate(angle)


def linear_polariser(angle):
    """Return a linear polariser matrix with transmission angle in radians."""
    return Jones.linear_polariser(angle)


def retarder(retardance, angle):
    """Return a retarder matrix with phase delay and fast-axis angle in radians."""
    return Jones.retarder(retardance, angle)


def apply_jones(jones, phasor):
    """Left-multiply a Jones response; both arrays have shape (..., 2, 2)."""
    return JonesMatrix(jones).apply(phasor)


def rotate_jones(jones, angle):
    """Rotate a matrix by angle in radians; None returns the original array."""
    return jones if angle is None else Jones.rotate_matrix(jones, angle)


def jones_to_stokes(jones, stokes=None):
    """Return output Stokes parameters; None uses unpolarised unit input."""
    return JonesMatrix(jones).to_stokes(stokes)
