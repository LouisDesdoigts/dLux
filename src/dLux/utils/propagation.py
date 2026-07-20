import jax.numpy as np
from jax import Array, vmap
import dLux.utils as dlu

__all__ = ["FFT", "MFT"]


def FFT_spec(npixels: int, pixel_scale: float, wavelength: float, focal_length=None):
    """
    Compute the native FFT output pixel scale and centre.

    Parameters
    ----------
    npixels : int
        Number of output pixels along one axis.
    pixel_scale : float
        Input pixel scale.
    wavelength : float
        Wavelength used for propagation.
    focal_length : float = None
        Effective focal length. If provided, the output sampling is converted from
        angular to linear units.

    Returns
    -------
    pixel_scale : float
        Output pixel scale.
    center : float
        Output coordinate centre.
    """
    pixel_scale = wavelength / (npixels * pixel_scale)
    if focal_length is not None:
        pixel_scale *= focal_length

    center = 0.0 if npixels % 2 else -0.5 * pixel_scale
    return pixel_scale, center


def FFT_ramp(
    xs: Array,
    wavelength: float,
    shift: float,
    focal_length: float = None,
    inverse: bool = False,
) -> Array:
    """
    Compute the 2D complex phase ramp for FFT coordinate shifting.

    Parameters
    ----------
    xs : Array
        One-dimensional coordinate vector.
    wavelength : float
        Wavelength used for propagation.
    shift : float
        Output coordinate shift to apply.
    focal_length : float = None
        Effective focal length. If provided, coordinates are interpreted in linear
        units; otherwise angular units are assumed.
    inverse : bool = False
        If False, use the forward-transform phase sign convention. If True, use the
        backward-transform phase sign convention.

    Returns
    -------
    ramp : Array
        Two-dimensional complex phase ramp.
    """
    sign = -1 if inverse else 1
    alpha = wavelength if focal_length is None else wavelength * focal_length
    ramp = np.exp(sign * 2j * np.pi * xs * shift / alpha)
    return ramp[None, :] * ramp[:, None]


def FFT(
    phasor: Array,
    wavelength: float,
    pixel_scale: float,
    focal_length: float = None,
    pad: int = 2,
    inverse: bool = False,
    center: float = 0.0,
    output_center: float = None,
) -> tuple[Array, Array, Array]:
    """
    Calculates the Fast Fourier Transform (FFT) of the input phasor.

    Parameters
    ----------
    phasor : Array[complex]
        The input phasor.
    wavelength : float, meters
        The wavelength of the input phasor.
    pixel_scale : float, meters/pixel
        The pixel scale of the input phasor.
    focal_length : float = None
        The focal length of the propagation. If None, the output pixel scale has units
        of radians, else meters.
    pad : int = 2
        The amount to pad the input array by before propagation. Note this function
        does not automatically crop the output.
    inverse : bool = False
        If False, apply the forward propagation transform. If True, apply the
        backward propagation transform.
    center : float = 0.0
        The coordinate centre of the input phasor.
    output_center : float = None
        The requested output coordinate centre. If None, the native FFT output centre
        is used.

    Returns
    -------
    phasor : Array[complex]
        The propagated phasor.
    new_pixel_scale : float
        The pixel scale of the output phasor.
    center : float
        The coordinate centre of the output phasor.
    """
    npixels = phasor.shape[-1]
    npixels_out = npixels * pad
    coords = lambda n, d, c: c + (np.arange(n) - (n - 1) / 2) * d

    # Calculate the native output coordinate system of the padded FFT.
    new_pixel_scale, native_center = FFT_spec(
        npixels_out, pixel_scale, wavelength, focal_length
    )

    # Shift the input phasor if a non-native output centre is requested.
    if output_center is None:
        in_ramp = out_ramp = 1.0
        output_center = native_center
    else:
        shift = native_center - output_center
        xs_in = coords(npixels, pixel_scale, center)
        in_ramp = FFT_ramp(xs_in, wavelength, shift, focal_length, inverse)

        # Shift the propagated phasor back onto the requested output coordinate grid.
        shift = FFT_spec(npixels_out, new_pixel_scale, wavelength, focal_length)[1]
        xs_out = coords(npixels_out, new_pixel_scale, output_center)
        out_ramp = FFT_ramp(xs_out, wavelength, shift, focal_length, inverse)

    # Pad the input array
    npixels = (npixels * (pad - 1)) // 2
    phasor = np.pad(phasor * in_ramp, npixels)

    # Perform the FFT
    if inverse:
        phasor = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(phasor)))
        phasor *= phasor.shape[-1]

    else:
        phasor = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(phasor)))
        phasor /= phasor.shape[-1]
    return phasor * out_ramp, new_pixel_scale, output_center


def transfer_matrix(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    shift: float = 0.0,
    focal_length: float = None,
    focal_shift: float = 0.0,
    inverse: bool = False,
) -> Array:
    """
    Calculates the transfer matrix for the MFT.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    shift : float = 0.0
        The shift in output-plane coordinates.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.
    inverse: bool = False
        If False, apply the forward propagation transform. If True, apply the
        backward propagation transform.

    Returns
    -------
    transfer_matrix : Array
        The transfer matrix for the MFT.
    """
    # Get parameters
    fringe_size = wavelength / (pixel_scale_in * npixels_in)

    # Input coordinates
    scale_in = 1.0 / npixels_in
    in_vec = dlu.nd_coords(npixels_in, scale_in, shift * scale_in)

    # Output coordinates
    scale_out = pixel_scale_out / fringe_size
    if focal_length is not None:
        # scale_out /= focal_length
        scale_out /= focal_length + focal_shift
    out_vec = dlu.nd_coords(npixels_out, scale_out, shift * scale_out)

    # Generate transfer matrix
    matrix = -2j * np.pi * np.outer(in_vec, out_vec)
    if inverse:
        matrix *= -1
    return np.exp(matrix)


def calc_nfringes(
    wavelength: float,
    npixels_in: int,
    pixel_scale_in: int,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    focal_shift: float = 0.0,
) -> Array:
    """
    Calculates the number of fringes in the output plane.

    Parameters
    ----------
    wavelength : float, meters
        The wavelength of the input phasor.
    npixels_in : int
        The number of pixels in the input plane.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    focal_shift: float, meters
        The shift from focus to propagate to. Used for fresnel propagation.

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
    phasor: Array,
    wavelength: float,
    pixel_scale_in: float,
    npixels_out: int,
    pixel_scale_out: float,
    focal_length: float = None,
    shift: Array = np.zeros(2),
    pixel: bool = True,
    inverse: bool = False,
) -> Array:
    """
    Propagates a phasor using a Matrix Fourier Transform (MFT), allowing for output
    pixel scale and a shift to be specified.

    Soummer et al. (2007) describes the MFT formulation: https://arxiv.org/pdf/0711.0368

    Parameters
    ----------
    phasor : Array
        The input phasor.
    wavelength : float, meters
        The wavelength of the input phasor.
    pixel_scale_in : float, meters/pixel, radians/pixel
        The pixel scale of the input plane.
    npixels_out : int
        The number of pixels in the output plane.
    pixel_scale_out : float, meters/pixel or radians/pixel
        The pixel scale of the output plane.
    focal_length : float = None
        The focal length of the propagation. If None, the propagation is angular and
        pixel_scale_out is taken in as radians/pixel, else meters/pixel.
    shift : Array = np.zeros(2)
        The shift in the center of the output plane.
    pixel : bool = True
        Should the shift be taken in units of pixels, or pixel scale.
    inverse : bool = False
        If False, apply the forward propagation transform. If True, apply the
        backward propagation transform.

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
        0.0,
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
    phasor *= np.exp(np.log(nfringes) - (np.log(npixels_in) + np.log(npixels_out)))

    return phasor
