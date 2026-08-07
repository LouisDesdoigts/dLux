"""Interpolate and rotate regularly sampled arrays."""

import jax.numpy as np
from jax import Array, vmap
import interpax as ipx
import dLux.utils as dlu

__all__ = ["interp", "scale", "rotate"]


def interp(
    image: Array,
    knot_coords: Array,
    sample_coords: Array,
    method: str = "linear",
    fill: float = 0.0,
    complex: bool = True,
) -> Array:
    """General 2D interpolation wrapper around `interpax.interp2d`.

    Parameters
    ----------
    image : Array
        The input 2D image to interpolate.
    knot_coords : Array
        The coordinates of the sampled points in `image`.
    sample_coords : Array
        The coordinates to interpolate onto.
    method : str = "linear"
        The interpolation method.
    fill : float = 0.0
        Fill value used outside `knot_coords`.
    complex : bool = True
        If the input image is complex, interpolate the real and imaginary components
        when True, or the amplitude and phase components when False.

    Returns
    -------
    array: Array
        The interpolated array.
    """
    # In the complex case we recurse on the cartesian/polar decomposition
    if np.iscomplexobj(image):
        vals, return_fn = dlu.from_complex(image, complex=complex)
        interp_fn = vmap(lambda x: interp(x, knot_coords, sample_coords, method, fill))
        return return_fn(interp_fn(vals))

    # Get the input/output coordinates
    xs, ys = knot_coords
    xpts, ypts = sample_coords.reshape(2, -1)

    # Interpolate using interpax
    return ipx.interp2d(
        ypts, xpts, ys[:, 0], xs[0], image, method=method, extrap=fill
    ).reshape(sample_coords[0].shape)


def scale(
    array: Array,
    npixels: int | tuple[int, int],
    ratio: float | Array,
    method: str = "linear",
    complex: bool = True,
) -> Array:
    """Paraxially interpolate a 2D array using per-axis sampling ratios.

    Parameters
    ----------
    array : Array
        The input field to interpolate, either in amplitude and phase, or real
        and imaginary.
    npixels : int | tuple[int, int]
        Output size in physical ``(x, y)`` axis order.
    ratio : float | Array
        Output sampling relative to the input along each physical axis.
    method : str = "linear"
        The interpolation method.
    complex : bool = True
        If the input array is complex, interpolate the real and imaginary components
        when True, or the amplitude and phase components when False.

    Returns
    -------
    array : Array
        The interpolated array.
    """
    # Get coords arrays
    size_in = array.shape[-2:][::-1]
    size_out = dlu.as_size(npixels, 2, "npixels")
    ratio = dlu.as_axis(ratio, 2, "ratio")
    coords_in = dlu.nd_coords(size_in, 1 / np.asarray(size_in))
    coords_out = dlu.nd_coords(size_out, 1 / np.asarray(size_out))
    coords_out = dlu.compress_coords(
        coords_out, ratio * np.asarray(size_out) / np.asarray(size_in)
    )

    # Interpolate
    return interp(array, coords_in, coords_out, method, complex=complex)


def rotate(
    array: Array, angle: Array, method: str = "linear", complex: bool = True
) -> Array:
    """Rotates a 2D array by the angle, retaining its input shape.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : float | Array, radians
        The angle to rotate the array by.
    method : str = "linear"
        The interpolation method.
    complex : bool = True
        If the input array is complex, interpolate the real and imaginary components
        when True, or the amplitude and phase components when False.

    Returns
    -------
    array : Array
        The rotated array.
    """
    # Get coordinates
    coords_in = dlu.nd_coords(array.shape[-2:][::-1])
    coords_out = dlu.rotate_coords(coords_in, angle)

    # Interpolate
    return interp(array, coords_in, coords_out, method, complex=complex)
