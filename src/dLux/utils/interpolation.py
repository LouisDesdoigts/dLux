import jax.numpy as np
from jax import Array
import interpax as ipx
import dLux.utils as dlu

__all__ = [
    "interp",
    "scale",
    "rotate",
]


def interp(
    image: Array,
    knot_coords: Array,
    sample_coords: Array,
    method: str = "linear",
    fill: float = 0.0,
):
    """
    General interpolation function, wrapping interpax.interp2d

    Parameters
    ----------
    image : Array
        The input 2D image to interpolate
    knot_coords : Array
        The coordinates of the points in the image
    sample_coords : Array
        The coordinates to interpolate onto
    method : str = "linear"
        The interpolation method.
    fill : float = 0.0
        Default value outside knot_coords

    Returns
    -------
    array: Array
        The interpolated array.
    """
    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    return ipx.interp2d(
        ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill
    ).reshape(sample_coords[0].shape)


def scale(
    array: Array, npixels: int, ratio: float, method: str = "linear"
) -> Array:
    """
    Paraxially interpolates a square array based on the sampling ratio, and npixels_out.

    Parameters
    ----------
    array : Array
        The input field to interpolate, either in amplitude and phase, or real
        and imaginary.
    npixels : int
        The number of pixel in the output array.
    ratio : float
        The scale of the input relative to the output
    method : str = "linear"
        The interpolation method.

    Returns
    -------
    array : Array
        The interpolated array.
    """
    # Get coords arrays
    npixels_in = array.shape[-1]
    coords_in = dlu.pixel_coords(npixels_in, 1)
    coords_out = dlu.compress_coords(
        dlu.pixel_coords(npixels, 1),
        np.array([ratio, ratio]) * npixels / npixels_in,
    )

    # Interpolate
    return interp(array, coords_in, coords_out, method)


def rotate(array: Array, angle: Array, method: str = "linear") -> Array:
    """
    Rotates a square array by the angle, using interpolation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.
    method : str = "linear"
        The interpolation method.

    Returns
    -------
    array : Array
        The rotated array.
    """
    # Get coordinates
    npixels = array.shape[0]
    coords_in = dlu.nd_coords((npixels, npixels))
    coords_out = dlu.rotate_coords(coords_in, angle)

    # Interpolate
    return interp(array, coords_in, coords_out, method)
