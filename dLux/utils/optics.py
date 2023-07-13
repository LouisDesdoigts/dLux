import jax.numpy as np
from jax import Array

__all__ = [
    "opd_to_phase",
    "phase_to_opd",
    "get_fringe_size",
    "get_pixels_per_fringe",
    "get_pixel_scale",
    "get_airy_pixel_scale",
]


def opd_to_phase(opd: Array, wavelength: Array) -> Array:
    """
    Converts the input Optical Path Difference (opd) in units of metres to
    phases in units of radians for the given wavelength.

    Parameters
    ----------
    opd : Array, metres
        The Optical Path Difference (opd) to be converted into phase.
    wavelength : Array, metres
        The wavelength at which to calculate the phase for.

    Returns
    -------
    phase : Array, radians
        The equivalent phase value for the given opd and wavelength.
    """
    return 2 * np.pi * opd / wavelength


def phase_to_opd(phase: Array, wavelength: Array) -> Array:
    """
    Converts the input phase in units of radians to the equivalent Optical Path
    Difference (opd) in metres for the given wavelength.

    Parameters
    ----------
    phase : Array, radians
        The phase to be converted into Optical Path Difference (opd)
    wavelength : Array, metres
        The wavelength at which to calculate the phase for.

    Returns
    -------
    opd : Array, metres
        The equivalent opd value for the given phase and wavelength.
    """
    return phase * wavelength / (2 * np.pi)


def get_fringe_size(wavelength: Array, aperture: Array) -> Array:
    """
    Calculates the angular size of the diffraction fringes.

    Parameters
    ----------
    wavelength : Array, metres
        The wavelength at which to calculate the diffraction fringe for.
    aperture : Array, metres
        The size of the aperture.

    Returns
    -------
    fringe_size : Array, radians
        The angular fringe size in units of radians.
    """
    return wavelength / aperture


def get_pixels_per_fringe(
    wavelength: Array,
    aperture: Array,
    pixel_scale: Array,
    focal_length: Array = None,
) -> Array:
    """
    Calculates the number of pixels per diffraction fringe, i.e. the fringe
    sampling rate.

    Parameters
    ----------
    wavelength : Array, metres
        The wavelength at which to calculate the diffraction fringe for.
    aperture : Array, metres
        The size of the aperture.
    pixel_scale : Array, metres/pixel or radians/pixel
        The size of each pixel. This is taken in units of radians per pixel if
        no focal length is provided, else it is taken in size of metres per
        pixel.
    focal_length : Array = None
        The focal length of the optical system. If none is provided, the pixel
        scale is taken in units of radians per pixel, else it is taken in
        metres per pixel.

    Returns
    -------
    sampling : Array
        The sampling rate of the fringes in units of pixels.
    """
    # Get fringe size
    fringe_size = get_fringe_size(wavelength, aperture)

    # Convert to metres if focal_length is provided
    if focal_length is not None:
        fringe_size *= focal_length

    # Return sampling
    return fringe_size / pixel_scale
    # return pixel_scale/fringe_size


def get_pixel_scale(
    sampling_rate: Array,
    wavelength: Array,
    aperture: Array,
    focal_length: Array = None,
) -> Array:
    """
    Calculates the pixel_size needed in order to sample the diffraction fringes
    at the given sampling rate.

    Parameters
    ----------
    sampling_rate : Array
        The rate at which to sample the diffraction fringes. A value of 2 will
        give nyquist sampled pixels.
    wavelength : Array, metres
        The observation wavelength.
    aperture : Array, metres
        The size of the aperture.
    focal_length : Array = None
        The focal length of the optical system. If none is provided, the pixel
        scale is given in units of radians per pixel, else it is given in
        metres per pixel.

    Returns
    -------
    pixel_scale : Array, radians per pixel or metres per pixel
        The pixel_size needed to sample the diffraction fringes at the input
        sampling rate, in units of radians per pixel if no focal length is
        provided, else in units of metres per pixel.
    """
    # Get fringe size
    fringe_size = get_fringe_size(wavelength, aperture)

    # Convert to metres if focal_length is provided
    if focal_length is not None:
        fringe_size *= focal_length

    # Get sampling rate
    return fringe_size / sampling_rate


def get_airy_pixel_scale(
    sampling_rate: Array,
    wavelength: Array,
    aperture: Array,
    focal_length: Array = None,
) -> Array:
    """
    Calculates the pixel_size needed to sample the diffraction fringes
    at the given sampling rate. Applies the 1.22 multiplier for Airy disk
    diffraction fringes given by a circular aperture.

    Parameters
    ----------
    sampling_rate : Array
        The rate at which to sample the diffraction fringes. A value of 2 will
        give nyquist sampled pixels.
    wavelength : Array, metres
        The observation wavelength.
    aperture : Array, metres
        The size of the aperture.
    focal_length : Array = None
        The focal length of the optical system. If none is provided, the pixel
        scale is given in units of radians per pixel, else it is given in
        metres per pixel.

    Returns
    -------
    pixel_scale : Array, radians per pixel or metres per pixel
        The pixel_size needed to sample the diffraction fringes at the input
        sampling rate, in units of radians per pixel if no focal length is
        provided, else in units of metres per pixel.
    """
    return get_pixel_scale(
        sampling_rate, 1.22 * wavelength, aperture, focal_length
    )
