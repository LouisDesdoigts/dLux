import jax.numpy as np
from jax import Array
import dLux.utils as dlu


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
    padded_mask = dlu.rotate(padded_mask, -angle_in_1st_quadrant)

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
