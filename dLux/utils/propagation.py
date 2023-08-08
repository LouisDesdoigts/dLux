import jax.numpy as np
from jax import Array, vmap
import dLux.utils as dlu


__all__ = ["FFT", "MFT", "fresnel_MFT"]


def FFT(
    phasor: Array,
    wavelength: float,
    pixel_scale: float,
    focal_length: float = None,
    pad: int = 2,
    inverse: bool = False,
) -> Array:
    """

    Field -> Array of [amplitude, phase]
    Phasor -> Complex array of [amplitude * exp(1j * phase)]

    Propagates the wavefront by performing a Fast Fourier Transform.

    Parameters
    ----------
    pad : int = 2
        The padding factory to apply to the input wavefront before
        performing the FFT.
    focal_length : Array = None
        The focal length of the propagation. If None, the propagation is
        treated as an 'angular' propagation, else it is treated as a
        'Cartesian' propagation.

    Returns
    -------
    wavefront : Wavefront
        The propagated wavefront.
    """
    npixels = phasor.shape[-1]

    # Calculate the output pixel scale
    fringe_size = wavelength / (pixel_scale * npixels)
    new_pixel_scale = fringe_size / pad
    if focal_length is not None:
        new_pixel_scale *= focal_length

    # Pad the input array
    npixels = (npixels * (pad - 1)) // 2
    phasor = np.pad(phasor, npixels)

    # Perform the FFT
    if inverse:
        phasor = np.fft.fft2(np.fft.ifftshift(phasor)) / phasor.shape[-1]
    else:
        phasor = np.fft.fftshift(np.fft.ifft2(phasor)) * phasor.shape[-1]

    return phasor, new_pixel_scale


def transfer_matrix(
    wavelength,
    npixels_in: Array,
    pixel_scale_in,
    npixels_out: int,
    pixel_scale_out: Array,
    shift: Array = 0.0,
    focal_length: Array = None,
    focal_shift: Array = 0.0,
    inverse: bool = False,
) -> Array:
    """
    Calculates the transfer matrix for the MFT.

    Parameters
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array
        The pixel scale of the output plane.
    shift : Array = 0.
        The shift to apply to the output plane.
    focal_length : Array = None
        The focal length of the propagation. If None, the propagation is
        treated as an 'angular' propagation, else it is treated as a
        'Cartesian' propagation.
    inverse : bool = False
        Is this a forward or inverse MFT.

    Returns
    -------
    transfer_matrix : Array
        The transfer matrix for the MFT.
    """
    # Get parameters
    fringe_size = wavelength / (pixel_scale_in * npixels_in)

    # Input coordinates
    scale_in = 1.0 / npixels_in
    in_vec = dlu.pixel_coordinates(npixels_in, scale_in, shift * scale_in)

    # Output coordinates
    scale_out = pixel_scale_out / fringe_size
    if focal_length is not None:
        # scale_out /= focal_length
        scale_out /= focal_length + focal_shift
    out_vec = dlu.pixel_coordinates(npixels_out, scale_out, shift * scale_out)

    # Generate transfer matrix
    matrix = 2j * np.pi * np.outer(in_vec, out_vec)
    if inverse:
        matrix *= -1
    return np.exp(matrix)


def calc_nfringes(
    wavelength,
    npixels_in: int,
    pixel_scale_in,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    focal_shift: float = 0.0,
) -> Array:
    """
    Calculates the number of fringes in the output plane.

    Parameters
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array
        The pixel scale of the output plane.
    focal_length : Array = None
        The focal length of the propagation. If None, the propagation is
        treated as an 'angular' propagation, else it is treated as a
        'Cartesian' propagation.

    Returns
    -------
    nfringes : Array
        The number of fringes in the output plane.
    """
    # Fringe size
    diameter = npixels_in * pixel_scale_in
    fringe_size = wavelength / diameter

    # Output array size
    output_size = npixels_out * pixel_scale_out
    if focal_length is not None:
        output_size /= focal_length + focal_shift

    # Fringe size and number of fringes
    return output_size / fringe_size


def MFT(
    phasor,
    wavelength,
    pixel_scale_in,
    npixels_out: int,
    pixel_scale_out: Array,
    focal_length: Array = None,
    focal_shift: Array = 0.0,
    shift: Array = np.zeros(2),
    pixel: bool = True,
    inverse: bool = False,
) -> Array:
    """
    Performs the actual phasor propagation and normalises the output

    Parameters
    ----------
    npixels : int
        The number of pixels in the output wavefront.
    pixel_scale : Array
        The pixel scale of the output wavefront.
    focal_length : Array = None
        The focal length of the propagation. If None, the propagation is
        treated as an 'angular' propagation, else it is treated as a
        'Cartesian' propagation.
    shift : Array = np.zeros(2)
        The shift in the center of the output plane.
    inverse : bool = False
        Is this a forward or inverse MFT.

    Returns
    -------
    phasor : Array
        The propagated phasor.
    """
    # Get parameters
    npixels_in = phasor.shape[-1]
    if not pixel:
        shift /= pixel_scale_out

    # Alias the transfer matrix function
    get_tf_mat = lambda s: transfer_matrix(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        s,
        focal_length,
        focal_shift,
        inverse,
    )

    # Get transfer matrices and propagate
    x_mat, y_mat = vmap(get_tf_mat)(shift)
    phasor = (y_mat.T @ phasor) @ x_mat

    # Normalise
    nfringes = calc_nfringes(
        wavelength,
        npixels_in,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
    )
    phasor *= np.exp(
        np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out))
    )

    return phasor


# Move to utils as thinlens?
def quadratic_phase(
    wavelength: float,
    distance: float,
    coordinates: Array,
) -> Array:
    """
    A convenience function for calculating quadratic phase factors.

    Parameters
    ----------
    x_coordinates : Array, metres
        The x coordinates of the pixels. This will be different
        in the plane of propagation and the initial plane.
    y_coordinates : Array, metres
        The y coordinates of the pixels. This will be different
        in the plane of propagation and the initial plane.
    distance : Array, metres
        The distance that is to be propagated.

    Returns
    -------
    quadratic_phase : Array
        A set of phase factors that are useful in optical calculations.
    """
    return np.exp(
        1j * np.pi * np.hypot(*coordinates) ** 2 / (distance * wavelength)
    )


def fresnel_phase_factors(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float,
    focal_shift: float,
) -> tuple:
    """
    Calculates the phase factors for the Fresnel propagation.

    Parameters
    ----------
    npixels : int
        The number of pixels in the output plane.
    pixel_scale : Array, metres/pixel
        The physical dimensions of each square pixel.
    focal_length : Array, metres
        The focal length of the lens.
    focal_shift : Array, metres
        The distance the focal plane is shifted from the focal length.

    Returns
    -------
    first_factor : Array
        The first factor in the Fresnel propagation.
    second_factor : Array
        The second factor in the Fresnel propagation.
    third_factor : Array
        The third factor in the Fresnel propagation.
    fourth_factor : Array
        The fourth factor in the Fresnel propagation.
    """
    # Calculate parameters
    prop_dist = focal_length + focal_shift
    input_positions = dlu.pixel_coords(npixels_in, pixel_scale_in)
    output_positions = dlu.pixel_coords(npixels_out, pixel_scale_out)

    # Calculate factors
    first_factor = quadratic_phase(
        *input_positions, -focal_length
    ) * quadratic_phase(*input_positions, prop_dist)

    second_factor = np.exp(
        2j * np.pi * prop_dist / wavelength
    ) * quadratic_phase(*output_positions, prop_dist)
    return first_factor, second_factor


def fresnel_MFT(
    phasor,
    wavelength,
    pixel_scale_in,
    npixels_out: int,
    pixel_scale_out: Array,
    shift: Array,
    focal_length: Array,
    focal_shift: Array,
    pixel: bool = True,
    inverse: bool = False,
) -> Array:
    """
    Propagates the wavefront from the input plane to the output plane using
    a Fresnel Transform using a Matrix Fourier Transform with a shift in
    the center of the output plane.
    TODO: Add link to Soumer et al. 2007(?),

    NOTE: This does have an 'inverse' parameter, however behaviour is not
    guaranteed to be correct when `inverse=True`.
    """
    # Calculate phase factors
    first_factor, second_factor = fresnel_phase_factors(
        wavelength,
        phasor.shape[-1],
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
        focal_shift,
    )

    # Propagate
    phasor *= first_factor
    phasor = MFT(
        phasor,
        wavelength,
        pixel_scale_in,
        npixels_out,
        pixel_scale_out,
        focal_length,
        focal_shift,
        shift,
        pixel,
        inverse,
    )
    phasor *= second_factor
    return phasor
