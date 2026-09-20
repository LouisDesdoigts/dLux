"""Coordinate-grid construction and Cartesian/polar conversion."""

import jax.numpy as np
from jax import Array

__all__ = ["coordinate_grid", "cart2polar", "polar2cart"]


def coordinate_grid(axes):
    """Expand physical-order axis vectors into (..., ndim, *sample_shape).

    Axis vectors may carry leading batches. Physical (x, y, z) axes correspond
    to final array dimensions (z, y, x).
    """
    ndim = len(axes)
    components = []
    for i, axis in enumerate(axes):
        shape = [1] * ndim
        shape[-1 - i] = axis.shape[-1]
        components.append(axis.reshape(axis.shape[:-1] + tuple(shape)))
    return np.stack(np.broadcast_arrays(*components), axis=-ndim - 1)


def cart2polar(coordinates: Array) -> Array:
    """Convert ``(..., 2, ny, nx)`` Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    coordinates : Array
        The (x, y) Cartesian coordinates to be converted into polar
        coordinates.

    Returns
    -------
    coordinates : Array
        The input Cartesian coordinates converted into (r, phi) polar
        coordinates.
    """
    x = coordinates[..., 0, :, :]
    y = coordinates[..., 1, :, :]
    return np.stack((np.hypot(x, y), np.arctan2(y, x)), axis=-3)


def polar2cart(coordinates: Array) -> Array:
    """Convert ``(..., 2, ny, nx)`` polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    coordinates : Array
        The (r, phi) polar coordinates to be converted into Cartesian
        coordinates.

    Returns
    -------
    coordinates : Array
        The input polar coordinates converted into (x, y) Cartesian
        coordinates.
    """
    r = coordinates[..., 0, :, :]
    phi = coordinates[..., 1, :, :]
    return np.stack((r * np.cos(phi), r * np.sin(phi)), axis=-3)
