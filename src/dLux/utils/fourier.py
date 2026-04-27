from jax import Array
import jax.numpy as np
from abcdLux.mft import mft_kernels

__all__ = [
    "fft_spec",
    "fft_phase_ramp",
    "fourier_kernel_1d",
    "fourier_kernels",
    "eval_fourier_basis",
]


def fft_spec(npixels_in, pixel_scale_in, wavelength, focal_length=None):
    """
    Compute the native FFT output sampling and center offset.

    Parameters
    ----------
    npixels_in : int
        Number of input pixels along one axis.
    pixel_scale_in : float
        Input pixel scale.
    wavelength : float
        Wavelength used for propagation.
    focal_length : float = None
        Effective focal length. If provided, the output sampling is converted from
        angular to linear units.

    Returns
    -------
    d_out : float
        Output pixel scale.
    c_out : float
        Output coordinate center offset.
    """
    # Output pixel scale - fl * lambda / D
    d_out = wavelength / (npixels_in * pixel_scale_in)
    if focal_length is not None:
        d_out = d_out * focal_length

    # Return the centered case
    if npixels_in % 2 != 0:
        return d_out, 0.0
    return d_out, -0.5 * d_out


def fft_phase_ramp(xs, wavelength, shift, focal_length=None, inverse=False):
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
    # Sign convention: positive for forward transform, negative for inverse transform
    sign = -1 if inverse else 1

    # Calculate the phase ramp to shift the FFT from native_out to spec_out
    alpha = wavelength if focal_length is None else wavelength * focal_length
    ramp = np.exp(sign * 2j * np.pi * xs * shift / alpha)
    return ramp[None, :] * ramp[:, None]


def _to_xy(value: int | tuple[int], name: str) -> tuple[int]:
    """
    Casts an integer or tuple input to an `(x, y)` tuple.

    Parameters
    ----------
    value : int | tuple[int]
        The input value.
    name : str
        The name of the input, used for error messages.

    Returns
    -------
    value : tuple[int]
        The cast input value.
    """
    if isinstance(value, int):
        value = (value, value)
    elif not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be an int or length-2 tuple, got {type(value)}.")

    value = (int(value[0]), int(value[1]))
    if value[0] <= 0 or value[1] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value}.")

    return value


def _fourier_mode_order(n_modes: int) -> Array:
    """
    Calculates the Fourier mode ordering for the complex exponential basis.

    Parameters
    ----------
    n_modes : int
        The number of Fourier modes.

    Returns
    -------
    modes : Array
        The ordered Fourier mode indices.
    """
    if n_modes % 2 == 1:
        kmax = (n_modes - 1) // 2
        modes = [0]
        for m in range(1, kmax + 1):
            modes.extend([m, -m])
    else:
        kmax = n_modes // 2
        modes = [0]
        for m in range(1, kmax):
            modes.extend([m, -m])
        modes.append(kmax)

    return np.asarray(modes)


def _map_to_real(kernel: Array) -> Array:
    """
    Maps a complex exponential Fourier kernel to the real Fourier basis.

    The input kernel is assumed to use the mode ordering
    `[0, +1, -1, +2, -2, ...]`. The returned kernel uses
    `[DC, cos1, sin1, cos2, sin2, ...]`.

    Parameters
    ----------
    kernel : Array
        The complex exponential Fourier kernel.

    Returns
    -------
    kernel : Array
        The real Fourier kernel.
    """
    n_modes = kernel.shape[-1]

    # Initialise the basis transform matrix and set the DC term
    T = np.zeros((n_modes, n_modes), dtype=complex)
    T = T.at[0, 0].set(1.0)

    # Determine number of harmonic pairs and Nyquist handling.
    if n_modes % 2 == 0:
        n_harm = n_modes // 2 - 1
        has_nyq = True
    else:
        n_harm = (n_modes - 1) // 2
        has_nyq = False

    if n_harm > 0:
        k = np.arange(1, n_harm + 1)
        half = np.array(0.5, dtype=complex)
        half_j = np.array(0.5j, dtype=complex)

        # Get the indicies
        ip = 2 * k - 1
        im = 2 * k

        # Set the transform for each harmonic pair
        T = T.at[ip, ip].set(half)
        T = T.at[ip, im].set(-half_j)
        T = T.at[im, ip].set(half)
        T = T.at[im, im].set(half_j)

    # Nyquist mode (even n): purely cosine, maps directly.
    if has_nyq:
        T = T.at[n_modes - 1, n_modes - 1].set(1.0)

    return (kernel @ T).real


def fourier_kernel_1d(n_modes: int, npix: int, scale: float = 1.0) -> Array:
    """
    Calculates a cached 1D Fourier basis evaluation kernel.

    A unit-amplitude mode with `scale=1` evaluates to values between `-1` and `+1`.

    Parameters
    ----------
    n_modes : int
        The number of Fourier modes.
    npix : int
        The number of output pixels.
    scale : float = 1.0
        The output amplitude scaling.

    Returns
    -------
    kernel : Array
        The cached Fourier kernel.
    """
    # Calculate the Fourier mode indices.
    modes = _fourier_mode_order(n_modes)

    # Calculate the output coordinates.
    coords = (np.arange(npix) - npix // 2) / npix

    # Calculate the complex exponential kernel.
    kernel, _ = mft_kernels(spec_in=modes, spec_out=coords, alpha=2 * np.pi, weight=1.0)

    # Map the kernel to the real Fourier basis.
    return scale * _map_to_real(kernel)


def fourier_kernels(
    n_modes: int | tuple[int],
    npix: int | tuple[int],
    scale: float = 1.0,
) -> tuple[Array, Array]:
    """
    Calculates the cached 2D Fourier basis evaluation kernels.

    The input `n_modes` and `npix` are interpreted in `(x, y)` order. If an
    integer is provided it is cast to `(value, value)`. The returned kernels satisfy

    `output = Kx @ coefficients @ Ky.T`

    where `coefficients` has shape `(n_modes_x, n_modes_y)` and `output` has shape
    `(npix_x, npix_y)`.

    Parameters
    ----------
    n_modes : int | tuple[int]
        The number of Fourier modes in `(x, y)`.
    npix : int | tuple[int]
        The output number of pixels in `(x, y)`.
    scale : float = 1.0
        The per-axis output amplitude scaling.

    Returns
    -------
    Kx : Array
        The x-axis Fourier kernel.
    Ky : Array
        The y-axis Fourier kernel.
    """
    # Unpack the number of modes and pixels for each dimension.
    n_modes_x, n_modes_y = _to_xy(n_modes, "n_modes")
    npix_x, npix_y = _to_xy(npix, "npix")

    # Calculate the Fourier kernels for each dimension.
    Kx = fourier_kernel_1d(n_modes_x, npix_x, scale)
    Ky = fourier_kernel_1d(n_modes_y, npix_y, scale)

    return Kx, Ky


def eval_fourier_basis(coefficients: Array, Kx: Array, Ky: Array) -> Array:
    """
    Evaluates a 2D real Fourier basis using cached kernels.

    The coefficient array is assumed to be ordered as `(x, y)`, with shape
    `(n_modes_x, n_modes_y)`. The returned output is also ordered as `(x, y)`,
    with shape `(npix_x, npix_y)`.

    Parameters
    ----------
    coefficients : Array
        The Fourier coefficients.
    Kx : Array
        The x-axis Fourier kernel.
    Ky : Array
        The y-axis Fourier kernel.

    Returns
    -------
    output : Array
        The evaluated Fourier basis.
    """
    return Kx @ coefficients @ Ky.T
