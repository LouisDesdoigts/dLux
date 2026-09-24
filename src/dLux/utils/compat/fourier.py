"""Historical Fourier array axes, retained for existing callers."""

from jax import Array
import jax.numpy as np

from ..propagation.fft import fourier_shift
from ..bases.fourier import (
    fourier_kernel_1d,
    fourier_kernels,
)

__all__ = [
    "eval_fourier_basis",
    "fourier_kernel_1d",
    "fourier_kernels",
    "fourier_shift",
]


def eval_fourier_basis(coeffs: Array, Kx: Array, Ky: Array) -> Array:
    """Evaluates a 2D real Fourier basis using cached kernels.

    The coefficient array is assumed to be ordered as `(x, y)`, with shape
    `(n_modes_x, n_modes_y)`. The returned output is also ordered as `(x, y)`,
    with shape `(npix_x, npix_y)`.

    Parameters
    ----------
    coeffs : Array
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
    return Kx @ coeffs @ Ky.T


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


__all__ += ["fft_spec", "fft_phase_ramp"]


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
