"""
This code is copied directly from https://github.com/adam-coogan/jaxinterp2d/blob/master/src/jaxinterp2d/__init__.py as the package is NOT pip installable and so we have to include its code base in our in order to access it
"""


from typing import Iterable, Optional

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


__all__ = ["interp2d", "CartesianGrid"]
Array = jnp.ndarray


def interp2d(
    x: Array,
    y: Array,
    xp: Array,
    yp: Array,
    zp: Array,
    fill_value: Optional[Array] = None,
) -> Array:
    """
    Bilinear interpolation on a grid. ``CartesianGrid`` is much faster if the data
    lies on a regular grid.
    Args:
        x, y: 1D arrays of point at which to interpolate. Any out-of-bounds
            coordinates will be clamped to lie in-bounds.
        xp, yp: 1D arrays of points specifying grid points where function values
            are provided.
        zp: 2D array of function values. For a function `f(x, y)` this must
            satisfy `zp[i, j] = f(xp[i], yp[j])`
    Returns:
        1D array `z` satisfying `z[i] = f(x[i], y[i])`.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D arrays")
    if zp.shape != (xp.shape + yp.shape):
        raise ValueError("zp must be a 2D array with shape xp.shape + yp.shape")

    ix = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, len(xp) - 1)
    iy = jnp.clip(jnp.searchsorted(yp, y, side="right"), 1, len(yp) - 1)

    # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    z_11 = zp[ix - 1, iy - 1]
    z_21 = zp[ix, iy - 1]
    z_12 = zp[ix - 1, iy]
    z_22 = zp[ix, iy]

    z_xy1 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_11 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_21
    z_xy2 = (xp[ix] - x) / (xp[ix] - xp[ix - 1]) * z_12 + (x - xp[ix - 1]) / (
        xp[ix] - xp[ix - 1]
    ) * z_22

    z = (yp[iy] - y) / (yp[iy] - yp[iy - 1]) * z_xy1 + (y - yp[iy - 1]) / (
        yp[iy] - yp[iy - 1]
    ) * z_xy2

    if fill_value is not None:
        oob = jnp.logical_or(
            x < xp[0], jnp.logical_or(x > xp[-1], jnp.logical_or(y < yp[0], y > yp[-1]))
        )
        z = jnp.where(oob, fill_value, z)

    return z


class CartesianGrid:
    """
    Linear Multivariate Cartesian Grid interpolation in arbitrary dimensions. Based
    on ``map_coordinates``.
    Notes:
        Translated directly from https://github.com/JohannesBuchner/regulargrid/ to jax.
    """

    values: Array
    """
    Values to interpolate.
    """
    limits: Iterable[Iterable[float]]
    """
    Limits along each dimension of ``values``.
    """

    def __init__(
        self,
        limits: Iterable[Iterable[float]],
        values: Array,
        mode: str = "constant",
        cval: float = jnp.nan,
    ):
        """
        Initializer.
        Args:
            limits: collection of pairs specifying limits of input variables along
                each dimension of ``values``
            values: values to interpolate. These must be defined on a regular grid.
            mode: how to handle out of bounds arguments; see docs for ``map_coordinates``
            cval: constant fill value; see docs for ``map_coordinates``
        """
        super().__init__()
        self.values = values
        self.limits = limits
        self.mode = mode
        self.cval = cval

    def __call__(self, *coords) -> Array:
        """
        Perform interpolation.
        Args:
            coords: point at which to interpolate. These will be broadcasted if
                they are not the same shape.
        Returns:
            Interpolated values, with extrapolation handled according to ``mode``.
        """
        # transform coords into pixel values
        coords = jnp.broadcast_arrays(*coords)
        # coords = jnp.asarray(coords)
        coords = [
            (c - lo) * (n - 1) / (hi - lo)
            for (lo, hi), c, n in zip(self.limits, coords, self.values.shape)
        ]
        return map_coordinates(
            self.values, coords, mode=self.mode, cval=self.cval, order=1
        )