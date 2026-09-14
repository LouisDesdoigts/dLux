"""
Polarisation utilities using Jones matrices with shape `(2, 2, ...)`.

The first two axes are the Jones matrix axes and trailing axes are broadcast
dimensions, usually spatial coordinates. Angles are measured counter-clockwise from
the horizontal x-axis towards the vertical y-axis.
"""

from __future__ import annotations
from jax import Array
import jax.numpy as np

__all__ = [
    "horizontal_polariser",
    "vertical_polariser",
    "rhc_polariser",
    "lhc_polariser",
    "quarter_wave_plate",
    "half_wave_plate",
    "apply_jones",
    "rotate_jones",
    "linear_polariser",
    "retarder",
    "jones_to_stokes",
]


def horizontal_polariser() -> Array:
    """
    Jones matrix for an ideal horizontal linear polariser.

    Returns
    -------
    jones : Array
        Horizontal polariser Jones matrix with shape `(2, 2)`.
    """
    return np.array([[1, 0], [0, 0]])


def vertical_polariser() -> Array:
    """
    Jones matrix for an ideal vertical linear polariser.

    Returns
    -------
    jones : Array
        Vertical polariser Jones matrix with shape `(2, 2)`.
    """
    return np.array([[0, 0], [0, 1]])


def rhc_polariser() -> Array:
    """
    Jones matrix for an ideal right-handed circular polariser.

    Returns
    -------
    jones : Array
        Right-handed circular polariser Jones matrix with shape `(2, 2)`, following
        the Stokes V sign convention used by `jones_to_stokes`.
    """
    return np.array([[0.5, -0.5j], [0.5j, 0.5]])


def lhc_polariser() -> Array:
    """
    Jones matrix for an ideal left-handed circular polariser.

    Returns
    -------
    jones : Array
        Left-handed circular polariser Jones matrix with shape `(2, 2)`, following
        the Stokes V sign convention used by `jones_to_stokes`.
    """
    return np.array([[0.5, 0.5j], [-0.5j, 0.5]])


def quarter_wave_plate(angle: Array) -> Array:
    """
    Jones matrix for a quarter-wave plate with fast axis rotated by `angle`.
    """
    jones = np.array([[1, 0], [0, 1j]])
    return rotate_jones(jones, angle)


def half_wave_plate(angle: Array) -> Array:
    """
    Jones matrix for a half-wave plate with fast axis rotated by `angle`.
    """
    jones = np.array([[1, 0], [0, -1]])
    return rotate_jones(jones, angle)


def linear_polariser(angle: Array) -> Array:
    """
    Jones matrix for an ideal linear polariser.

    Parameters
    ----------
    angle : Array, radians
        Transmission-axis angle measured counter-clockwise from horizontal. An input
        shape of `...` returns a Jones matrix of shape `(2, 2, ...)`.

    Returns
    -------
    jones : Array
        Linear polariser Jones matrix with shape `(2, 2, ...)`.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c**2, c * s], [c * s, s**2]])


def retarder(retardance: Array, angle: Array) -> Array:
    """
    Jones matrix for a retarder.

    Parameters
    ----------
    retardance : Array, radians
        Phase delay of the vertical component relative to horizontal.
    angle : Array, radians
        Fast-axis angle measured counter-clockwise from horizontal. Inputs broadcast
        together and return a Jones matrix of shape `(2, 2, ...)`.

    Returns
    -------
    jones : Array
        Retarder Jones matrix with shape `(2, 2, ...)`.
    """
    phase = np.exp(1j * retardance)
    jones = np.array(
        [
            [np.ones_like(phase), np.zeros_like(phase)],
            [np.zeros_like(phase), phase],
        ]
    )
    return rotate_jones(jones, angle)


def apply_jones(jones: Array, phasor: Array) -> Array:
    """
    Applies a Jones matrix to a Jones phasor.

    Parameters
    ----------
    jones : Array
        Jones matrix with shape `(2, 2, ...)`.
    phasor : Array
        Jones phasor with shape `(2, 2, ...)`.

    Returns
    -------
    phasor : Array
        Transformed Jones phasor with shape `(2, 2, ...)`.
    """
    return np.einsum("ij..., jk... -> ik...", jones, phasor)


def rotate_jones(jones: Array, angle: Array | None) -> Array:
    """
    Rotate a Jones matrix by a given angle.

    Parameters
    ----------
    jones : Array
        The initial Jones matrix to be rotated, of shape (2, 2, ...).
    angle : Array | None, radians
        The angle by which to rotate the Jones matrix, of shape (...). If None,
        the input Jones matrix is returned unchanged.

    Returns
    -------
    jones : Array
        The rotated Jones matrix, of shape (2, 2, ...).
    """
    if angle is None:
        return jones
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, s], [-s, c]])
    return np.einsum("mi..., mn..., nj... -> ij...", R, jones, R)


# Pre calculate the A and its inverse
A = np.array(
    [
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, -1j, 1j, 0],  # Swapped 1j and -1j to match IAU
    ]
)

A_inv = np.linalg.inv(A)


def jones_to_stokes(jones, stokes=None):
    """
    Convert a Jones matrix to the output Stokes parameters for an input Stokes vector.
    This follows the module convention strictly: the first two axes are Jones axes,
    and all remaining axes are trailing broadcast dimensions. Leading vectorisation
    dimensions should be handled by the caller.

    Parameters
    ----------
    jones : Array
        Jones matrix with shape `(2, 2, ...)`.
    stokes : Array | None
        Input Stokes vector `[I, Q, U, V]`. If None, unpolarised unit intensity is
        used.

    Returns
    -------
    stokes : Array
        Output Stokes parameters with shape `(4, ...)`.
    """
    stokes = np.array([1.0, 0.0, 0.0, 0.0]) if stokes is None else stokes

    J_kron = np.einsum("ik...,jl...->ijkl...", jones, np.conj(jones))
    J_kron = J_kron.reshape((4, 4) + jones.shape[2:])
    M = np.einsum("xy,yz...,zw->xw...", A, J_kron, A_inv)
    return np.einsum("ab...,b->a...", M.real, stokes)
