import jax.numpy as np
from jax import Array
from jax.scipy.ndimage import map_coordinates
import dLux.utils as dlu

# TODO: Resolve scale and scale_array
__all__ = [
    "scale_array",
    "downsample",
    "generate_coordinates",
    "scale",
    "rotate",
    "fourier_rotate",
]


def scale_array(array: Array, size_out: int, order: int) -> Array:
    """
    Scales some input array to size_out using interpolation.

    Parameters
    ----------
    array : Array
        The array to scale.
    size_out : int
        The output size of the mask
    order : int
        The interpolation order. Supports 0 and 1.

    Returns
    -------
    array : Array
        The array scaled to size_out
    """
    xs = np.linspace(0, array.shape[0], size_out)
    xs, ys = np.meshgrid(xs, xs)
    return map_coordinates(array, np.array([ys, xs]), order=order)


def downsample(array: Array, n: int, method="mean") -> Array:
    """
    Downsamples the input array by n.

    Parameters
    ----------
    array : Array
        The input array to downsample.
    n : int
        The factor by which to downsample the array.
    method : str = 'mean'
        The method by which to downsample the array. Can be 'mean' or 'sum'.

    Returns
    -------
    array : Array
        The downsampled array.
    """
    if method == "sum":
        method = np.sum
    elif method == "mean":
        method = np.mean
    else:
        raise ValueError('Invalid method. Choose "mean" or "sum".')

    size_in = array.shape[0]
    size_out = size_in // n

    # Downsample first dimension
    array = method(array.reshape((size_in * size_out, n)), 1)
    array = array.reshape(size_in, size_out).T

    # Downsample second dimension
    array = method(array.reshape((size_out * size_out, n)), 1)
    array = array.reshape(size_out, size_out).T
    return array


def generate_coordinates(
    npixels_in: int,
    npixels_out: int,
    sampling_ratio: Array,
    x_shift: Array = np.array(0.0),
    y_shift: Array = np.array(0.0),
) -> Array:
    """
    Generates a new set of paraxial coordinates which can be used for
    interpolation.

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
    Paraxially interpolates a wavefront field (either in amplitude and phase,
    or real and imaginary) based on the sampling ratio, and npixels_out.

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
        The interpolation order to use.

    Returns
    -------
    field : Array
        The interpolated output amplitude and phase arrays.
    """
    # Get coords arrays
    npixels_in = array.shape[-1]
    # TODO: Update with utils.pixel_coordinates
    coordinates = generate_coordinates(npixels_in, npixels, ratio)
    return map_coordinates(array, coordinates, order=order)


def rotate(array: Array, angle: Array, order: int = 1) -> Array:
    """
    Rotates an array by the angle, using linear interpolation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.
    order : int = 1
        The interpolation order to use.

    Returns
    -------
    array : Array
        The rotated array.
    """

    def _rotate(coordinates: Array, rotation: Array) -> Array:
        x, y = coordinates[0], coordinates[1]
        new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
        new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
        return np.array([new_x, new_y])

    # Get coordinates
    npixels = array.shape[0]
    centre = (npixels - 1) / 2
    coordinates = dlu.pixel_coordinates((npixels, npixels), indexing="ij")
    coordinates_rotated = _rotate(coordinates, angle) + centre

    # Interpolate
    return map_coordinates(array, coordinates_rotated, order=order)


def fourier_rotate(
    array: Array, angle: Array, padding: int = 2
) -> Array:  # pragma: no cover
    """
    Rotates an array by the angle, using a fourier rotation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.
    padding : int = 2
        The amount of fourier padding to use.

    Returns
    -------
    array : Array
        The rotated array.
    """
    raise NotImplementedError("Fourier rotation is under development.")
    # TODO implement
    in_shape = array.shape
    array_shape = np.array(in_shape, dtype=int) + 3
    array = (
        np.full(array_shape, np.nan, dtype=float)
        .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]
        .set(array)
    )

    # FFT rotation only work in the -45:+45 range
    # So I need to work out how to determine the quadrant that
    # angle is in and hence the
    # number of required pi/2 rotations and angle in radians.
    half_pi_to_1st_quadrant = angle // (np.pi / 2)
    angle_in_1st_quadrant = -angle + (half_pi_to_1st_quadrant * np.pi / 2)

    array = np.rot90(array, half_pi_to_1st_quadrant).at[:-1, :-1].get()

    width, height = array.shape
    left_corner = int(((padding - 1) / 2.0) * width)
    right_corner = int(((padding + 1) / 2.0) * width)
    top_corner = int(((padding - 1) / 2.0) * height)
    bottom_corner = int(((padding + 1) / 2.0) * height)

    # Make the padded array
    out_shape = (width * padding, height * padding)
    padded_array = (
        np.full(out_shape, np.nan, dtype=float)
        .at[left_corner:right_corner, top_corner:bottom_corner]
        .set(array)
    )

    padded_mask = (
        np.ones(out_shape, dtype=bool)
        .at[left_corner:right_corner, top_corner:bottom_corner]
        .set(np.where(np.isnan(array), True, False))
    )

    # Rotate the mask, to know what part is actually the array
    padded_mask = rotate(padded_mask, -angle_in_1st_quadrant)

    # Replace part outside the array which are NaN by 0, and go into
    # Fourier space.
    padded_array = np.where(np.isnan(padded_array), 0.0, padded_array)

    uncentered_angular_displacement = np.tan(angle_in_1st_quadrant / 2.0)
    centered_angular_displacement = -np.sin(angle_in_1st_quadrant)

    uncentered_frequencies = np.fft.fftfreq(out_shape[0])
    centered_frequencies = np.arange(-out_shape[0] / 2.0, out_shape[0] / 2.0)

    pi_factor = -2.0j * np.pi * np.ones(out_shape, dtype=float)

    uncentered_phase = np.exp(
        uncentered_angular_displacement
        * ((pi_factor * uncentered_frequencies).T * centered_frequencies).T
    )

    centered_phase = np.exp(
        centered_angular_displacement
        * (pi_factor * centered_frequencies).T
        * uncentered_frequencies
    )

    f1 = np.fft.ifft(
        (np.fft.fft(padded_array, axis=0).T * uncentered_phase).T, axis=0
    )

    f2 = np.fft.ifft(np.fft.fft(f1, axis=1) * centered_phase, axis=1)

    rotated_array = (
        np.fft.ifft((np.fft.fft(f2, axis=0).T * uncentered_phase).T, axis=0)
        .at[padded_mask]
        .set(np.nan)
    )

    return np.real(
        rotated_array.at[
            left_corner + 1 : right_corner - 1,
            top_corner + 1 : bottom_corner - 1,
        ].get()
    ).copy()
