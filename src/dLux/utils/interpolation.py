import jax.numpy as np
from jax import Array
from jax.scipy.ndimage import map_coordinates
import dLux.utils as dlu

__all__ = [
    "generate_coordinates",
    "scale",
    "rotate",
]


def generate_coordinates(
    npixels_in: int,
    npixels_out: int,
    sampling_ratio: Array,
    x_shift: Array = np.array(0.0),
    y_shift: Array = np.array(0.0),
) -> Array:
    """
    Generates a new set of paraxial coordinates which can be used for interpolation.

    Parameters
    ----------
    npixels_in : int
        The number of pixels in the original array.
    npixels_out : int
        The number of pixel in the output array.
    sampling_ratio : Array
        The ratio of pixel sizes in the input and output array,
        i.e. pixel_scale_out/pixel_scale_in.
    x_shift : Array, pixels = np.array(0.)
        How much to shift the x_coordinates in the output array, in the pixel
        units of the output array.
    y_shift : Array, pixels = np.array(0.)
        How much to shift the y_coordinates in the output array, in the pixel
        units of the output array.

    Returns
    -------
    coordinates : Array
        The output coordinates at which to interpolate onto.
    """
    old_centre = (npixels_in - 1) / 2
    new_centre = (npixels_out - 1) / 2
    pixels = (
        sampling_ratio * np.linspace(-new_centre, new_centre, npixels_out)
        + old_centre
    )
    x_pixels, y_pixels = np.meshgrid(pixels + x_shift, pixels + y_shift)
    return np.array([y_pixels, x_pixels])


def scale(array: Array, npixels: int, ratio: float, order: int = 1) -> Array:
    """
    Paraxially interpolates an array based on the sampling ratio, and npixels_out.

    # TODO: Check if a half-pixel offset is produced

    Parameters
    ----------
    array : Array
        The input field to interpolate, either in amplitude and phase, or real
        and imaginary.
    npixels : int
        The number of pixel in the output array.
    ratio : float
        The relative input to output scales, TODO: does 2 make it bigger or
        smaller? i.e. input scale/output scale. <- get this right.
    order : int = 1
        The interpolation order to use. Can be 0 or 1.

    Returns
    -------
    array : Array
        The interpolated array.
    """
    # Get coords arrays
    npixels_in = array.shape[-1]
    # TODO: Update with array_coordinates
    coordinates = generate_coordinates(npixels_in, npixels, ratio)
    return map_coordinates(array, coordinates, order=order)


def rotate(array: Array, angle: Array, order: int = 1) -> Array:
    """
    Rotates an array by the angle, using interpolation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.
    order : int = 1
        The interpolation order to use. Can be 0 or 1.

    Returns
    -------
    array : Array
        The rotated array.
    """

    # TODO: Use rotate_coords
    def _rotate(coordinates: Array, rotation: Array) -> Array:
        x, y = coordinates[0], coordinates[1]
        new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
        new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
        return np.array([new_x, new_y])

    # Get coordinates
    npixels = array.shape[0]
    centre = (npixels - 1) / 2
    coordinates = dlu.nd_coords((npixels, npixels), indexing="ij")
    coordinates_rotated = _rotate(coordinates, angle) + centre

    # Interpolate
    return map_coordinates(array, coordinates_rotated, order=order)
