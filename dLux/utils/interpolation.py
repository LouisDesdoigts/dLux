import jax.numpy as np
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from functools import partial
from dLux.utils.coordinates import get_polar_positions, polar_to_cartesian


__all__ = ["scale_array", "generate_coordinates", "interpolate_field",
           "interpolate", "rotate_field", "rotate", "fourier_rotate"]


Array =  np.ndarray


def scale_array(array    : Array,
                size_out : int,
                order    : int) -> Array:
    """
    Scales some input array to size_out using interolation.

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
    assert order in (0, 1), ("order must be 0 or 1.")
    xs = np.linspace(0, array.shape[0], size_out)
    return map_coordinates(array, np.meshgrid(xs, xs), order=order)


def generate_coordinates(npixels_in     : int,
                         npixels_out    : int,
                         sampling_ratio : Array,
                         x_shift        : Array = np.array(0.),
                         y_shift        : Array = np.array(0.)) -> Array:
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
        ie pixel_scale_out/pixel_scale_in.
    x_shift : Array, pixles = np.array(0.)
        How much to shift the x_coordinates in the output array, in the pixel
        units of the output array.
    y_shift : Array, pixles = np.array(0.)
        How much to shift the y_coordinates in the output array, in the pixel
        units of the output array.

    Returns
    -------
    coordinates : Array
        The output coordinates at which to interpolate onto.
    """
    old_centre = (npixels_in  - 1) / 2
    new_centre = (npixels_out - 1) / 2
    pixels = sampling_ratio * np.linspace(-new_centre, new_centre,
                                          npixels_out) + old_centre
    x_pixels, y_pixels = np.meshgrid(pixels + x_shift, pixels + y_shift)
    return np.array([y_pixels, x_pixels])


def interpolate_field(field           : Array,
                      npixels_out     : int,
                      sampling_ratio  : Array,
                      real_imaginary  : bool = False,
                      x_shift         : Array = np.array(0.),
                      y_shift         : Array = np.array(0.)) -> Array:
    """
    Paraxially interpolates a wavefront field (either in ampltude and phase, or
    real and imaginiary) based on the sampling ratio, and npixels_out.

    Parameters
    ----------
    field : Array
        The input field to interpolate, either in amplitude and phase, or real
        and imaginary.
    npixels_out : int
        The number of pixel in the output array.
    sampling_ratio : Array
        The ratio of pixel sizes in the input and output array,
        ie pixel_scale_out/pixel_scale_in.
    real_imaginary : bool = False
        Is the input field given in amplitude and phase, or real and imagninary.
    x_shift : Array, pixles = np.array(0.)
        How much to shift the x_coordinates in the output array, in the pixel
        units of the output array.
    y_shift : Array, pixles = np.array(0.)
        How much to shift the y_coordinates in the output array, in the pixel
        units of the output array.

    Returns
    -------
    field : Array
        The interpolated output amplitude and phase arrays.
    """
    # Get coords arrays
    npixels_in = field.shape[-1]
    coordinates = generate_coordinates(npixels_in, npixels_out, sampling_ratio,
                                       x_shift, y_shift)

    # Interpolate
    interpolator = vmap(map_coordinates, in_axes=(0, None, None))
    new_field = interpolator(field, coordinates, 1)

    # Conserve energy
    if real_imaginary:
        amplitude = np.hypot(new_field[0], new_field[1])
        phase = np.arctan2(new_field[1], new_field[0])
    else:
        amplitude, phase = new_field

    return np.array([amplitude, phase])


def interpolate(array          : Array,
                npixels_out    : int,
                sampling_ratio : Array,
                x_shift        : Array = np.array(0.),
                y_shift        : Array = np.array(0.)) -> Array:
    """
    Paraxially interpolates an array based on the sampling ratio, and
    npixels_out.

    Parameters
    ----------
    array : Array
        The input array to interpolate.
    npixels_out : int
        The number of pixel in the output array.
    sampling_ratio : Array
        The ratio of pixel sizes in the input and output array,
        ie pixel_scale_out/pixel_scale_in.
    x_shift : Array, pixles = np.array(0.)
        How much to shift the x_coordinates in the output array, in the pixel
        units of the output array.
    y_shift : Array, pixles = np.array(0.)
        How much to shift the y_coordinates in the output array, in the pixel
        units of the output array.

    Returns
    -------
    field : Array
        The interpolated arrays.
    """
    # Get coords arrays
    npixels_in = array.shape[-1]
    coordinates = generate_coordinates(npixels_in, npixels_out, sampling_ratio,
                                       x_shift, y_shift)

    # Interpolate
    new_array = map_coordinates(array, order=1)

    # Conserve energy and return
    return new_array * sampling_ratio


def rotate_field(field          : Array,
                 angle          : Array,
                 real_imaginary : bool = False,
                 fourier        : bool = False,
                 padding        : int  = 2) -> Array:
    """
    Paraxially rotates a wavefront field (either in ampltude and phase, or
    real and imaginiary) in the {}wise direction by angle. Two methods are
    available, interpolation and fourier rotation. Interpolation is much faster
    with large arrays, and fourier rotation is information preserving.

    Parameters
    ----------
    field : Array
        The input field to rotate, either in amplitude and phase, or real
        and imaginary.
    angle : Array, radians
        The angle by which to rotate the wavefront in a {}wise direction.
    real_imaginary : bool = False
        Whether to rotate the real and imaginary representation of the
        wavefront as opposed to the the amplitude and phase representation.
    fourier : bool = False
        Should the fourier rotation method be used (True), or regular
        interpolation method be used (False).
    padding : int = 2
        The amount of fourier padding to use. Only applies if fourier is True.

    Returns
    -------
    field : Array
        The rotated output amplitude and phase arrays.
    """
    # Generate rotator function
    if fourier:
        padded_rotate = partial(fourier_rotate, padding=padding)
        rotator = vmap(padded_rotate, in_axes=(0, None))
    else:
        rotator = vmap(rotate, in_axes=(0, None))

    # Rotate
    rotated_field = rotator(field, angle)

    # Get amplitude phase
    if real_imaginary:
        amplitude = np.hypot(rotated_field[0], rotated_field[1])
        phase = np.arctan2(rotated_field[1], rotated_field[0])
    else:
        amplitude, phase = rotated_field

    return np.array([amplitude, phase])


def rotate(array : Array, angle : Array) -> Array:
    """
    Rotates an array by the angle, using linear interpolation.

    Parameters
    ----------
    array : Array
        The array to rotate.
    angle : Array, radians
        The angle to rotate the array by.

    Returns
    -------
    array : Array
        The rotated array.
    """
    # Get coordinates
    npixels = array.shape[0]
    centre = (npixels - 1) / 2
    coordinates = get_polar_positions(npixels)
    coordinates += np.array([0., angle])[:, None, None]
    coordinates_rotated = np.roll(polar_to_cartesian(coordinates) + centre,
                                  shift=1, axis=0)

    # Interpolate
    return map_coordinates(array, coordinates_rotated, order=1)


def fourier_rotate(array   : Array,
                   angle   : Array,
                   padding : int = 2) -> Array:
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

    in_shape = array.shape
    array_shape = np.array(in_shape, dtype=int) + 3
    array = np.full(array_shape, np.nan, dtype=float)\
        .at[1 : in_shape[0] + 1, 1 : in_shape[1] + 1]\
        .set(array)

    # FFT rotation only work in the -45:+45 range
    # So I need to work out how to determine the quadrant that
    # angle is in and hence the
    # number of required pi/2 rotations and angle in radians.
    half_pi_to_1st_quadrant = angle // (np.pi / 2)
    angle_in_1st_quadrant = - angle + (half_pi_to_1st_quadrant * np.pi / 2)

    array = np.rot90(array, half_pi_to_1st_quadrant)\
        .at[:-1, :-1]\
        .get()

    width, height = array.shape
    left_corner   = int(((padding - 1) / 2.) * width)
    right_corner  = int(((padding + 1) / 2.) * width)
    top_corner    = int(((padding - 1) / 2.) * height)
    bottom_corner = int(((padding + 1) / 2.) * height)

    # Make the padded array
    out_shape = (width * padding, height * padding)
    padded_array = np.full(out_shape, np.nan, dtype=float)\
        .at[left_corner : right_corner, top_corner : bottom_corner]\
        .set(array)

    padded_mask = np.ones(out_shape, dtype=bool)\
        .at[left_corner : right_corner, top_corner : bottom_corner]\
        .set(np.where(np.isnan(array), True, False))

    # Rotate the mask, to know what part is actually the array
    padded_mask = self.__rotate(padded_mask, -angle_in_1st_quadrant)

    # Replace part outside the array which are NaN by 0, and go into 
    # Fourier space.
    padded_array = np.where(np.isnan(padded_array), 0. , padded_array)

    uncentered_angular_displacement = np.tan(angle_in_1st_quadrant / 2.)
    centered_angular_displacement = -np.sin(angle_in_1st_quadrant)

    uncentered_frequencies = np.fft.fftfreq(out_shape[0])
    centered_frequencies = np.arange(-out_shape[0] / 2., out_shape[0] / 2.)

    pi_factor = -2.j * np.pi * np.ones(out_shape, dtype=float)

    uncentered_phase = np.exp(
        uncentered_angular_displacement *\
        ((pi_factor * uncentered_frequencies).T *\
        centered_frequencies).T)

    centered_phase = np.exp(
        centered_angular_displacement *\
        (pi_factor * centered_frequencies).T *\
        uncentered_frequencies)

    f1 = np.fft.ifft(
        (np.fft.fft(padded_array, axis=0).T * uncentered_phase).T, axis=0)

    f2 = np.fft.ifft(
        np.fft.fft(f1, axis=1) * centered_phase, axis=1)

    rotated_array = np.fft.ifft(
        (np.fft.fft(f2, axis=0).T * uncentered_phase).T, axis=0)\
        .at[padded_mask]\
        .set(np.nan)

    return np.real(rotated_array\
        .at[left_corner + 1 : right_corner - 1,
            top_corner + 1 : bottom_corner - 1]\
        .get()).copy()
