"""Generate and transform Cartesian and polar coordinate grids."""

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

__all__ = [
    "cart2polar",
    "polar2cart",
    "pixel_coords",
    "nd_axes",
    "nd_coords",
    "translate_coords",
    "compress_coords",
    "shear_coords",
    "rotate_coords",
    "distort_coords",
]


def translate_coords(coords: Array, translation: Array) -> Array:
    """Translate n-dimensional coordinates into a new centre.

    Coordinates follow ``(..., ndim, *spatial_shape)`` and translations follow
    ``(..., ndim)``. Leading dimensions use paired JAX broadcasting.

    Parameters
    ----------
    coords : Array
        The input coordinates to translate.
    translation : Array
        The translation to apply to the coordinates.

    Returns
    -------
    coords : Array
        The translated coordinates.
    """
    translation = np.asarray(translation)
    ndim = translation.shape[-1]
    return coords - translation.reshape(translation.shape + (1,) * ndim)


def compress_coords(coords: Array, compress: Array) -> Array:
    """Compress n-dimensional coordinates by a per-axis factor.

    Coordinates follow ``(..., ndim, *spatial_shape)`` and factors follow
    ``(..., ndim)``. Leading dimensions use paired JAX broadcasting.

    Parameters
    ----------
    coords : Array
        The input coordinates to compress.
    compress : Array
        The compression to apply to the coordinates.

    Returns
    -------
    coords : Array
        The compressed coordinates.
    """
    compress = np.asarray(compress)
    ndim = compress.shape[-1]
    return coords * compress.reshape(compress.shape + (1,) * ndim)


def shear_coords(coords: Array, shear: Array) -> Array:
    """Shear 2D coordinates by a per-axis factor.

    Coordinates follow ``(..., 2, ny, nx)`` and shear values follow ``(..., 2)``.
    Leading dimensions use paired JAX broadcasting.

    Parameters
    ----------
    coords : Array
        The input coordinates to shear.
    shear : Array
        The shear to apply to the coordinates.

    Returns
    -------
    coords : Array
        The sheared coordinates.
    """
    # Extract coordinates and broadcast the shear over the spatial axes
    x, y = coords[..., 0, :, :], coords[..., 1, :, :]
    x_shear = shear[..., 0, None, None]
    y_shear = shear[..., 1, None, None]

    # Apply each shear and restore the coordinate axis
    new_x = x + x_shear * y
    new_y = y + y_shear * x
    return np.stack((new_x, new_y), axis=-3)


def rotate_coords(coords: Array, rotation: float) -> Array:
    """Rotate 2D coordinates by an angle in radians.

    Coordinates follow ``(..., 2, ny, nx)`` and rotations follow ``(...)``.
    Leading dimensions use paired JAX broadcasting.

    Parameters
    ----------
    coords : Array
        The input coordinates to rotate.
    rotation : float, radians
        The rotation to apply to the coordinates.

    Returns
    -------
    coords : Array
        The rotated coordinates.
    """
    # Extract coordinates and broadcast the rotation over the spatial axes
    x, y = coords[..., 0, :, :], coords[..., 1, :, :]
    rotation = np.asarray(rotation)
    angle = rotation[..., None, None]
    cosine, sine = np.cos(-angle), np.sin(-angle)

    # Rotate and restore the coordinate axis
    new_x = cosine * x + sine * y
    new_y = -sine * x + cosine * y
    return np.stack((new_x, new_y), axis=-3)


def distort_coords(coords: Array, coeffs: Array, pows: Array):
    """Apply a polynomial distortion to 2D coordinates.

    Coordinates follow ``(..., 2, ny, nx)``. Unbatched coefficients have shape
    ``(2, n_terms)``; batching is owned by :class:`dLux.DistortCoords`.

    Parameters
    ----------
    coords : Array
        Input coordinates to distort
    coeffs : Array
        Distortion polynomial coefficients
    pows : Array
        Distortion polynomial powers

    Returns
    -------
    distorted_coords : Array
        Coords with the distortion applied
    """
    variables = np.moveaxis(coords, -3, 0)
    pow_base = dlu.polynomial_basis(variables, pows)
    distortion = np.tensordot(coeffs, pow_base, axes=(-1, 0))
    distortion = np.moveaxis(distortion, 0, -3)
    return coords + distortion


def cart2polar(coordinates: Array) -> Array:
    """Convert ``(..., 2, ny, nx)`` Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    coordinates : Array
        The (x, y) Cartesian coordinates to be converted into polar
        coordinates.

    Returns
    -------
    coordinates : Array
        The input Cartesian coordinates converted into (r, phi) polar
        coordinates.
    """
    x = coordinates[..., 0, :, :]
    y = coordinates[..., 1, :, :]
    return np.stack((np.hypot(x, y), np.arctan2(y, x)), axis=-3)


def polar2cart(coordinates: Array) -> Array:
    """Convert ``(..., 2, ny, nx)`` polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    coordinates : Array
        The (r, phi) polar coordinates to be converted into Cartesian
        coordinates.

    Returns
    -------
    coordinates : Array
        The input polar coordinates converted into (x, y) Cartesian
        coordinates.
    """
    r = coordinates[..., 0, :, :]
    phi = coordinates[..., 1, :, :]
    return np.stack((r * np.cos(phi), r * np.sin(phi)), axis=-3)


def pixel_coords(
    npixels: int,
    diameter: float = None,
    radius: float = None,
    pixel_scale: float = None,
    polar: bool = False,
    fft_style: bool = False,
) -> Array:
    """Returns a paraxial set of 2d coordinates for each pixel centre.

    Parameters
    ----------
    npixels : int
        The output size of the coordinates array to generate.
    diameter : float = None
        The diameter of the coordinates array to generate.
    radius : float = None
        The radius of the coordinates array to generate.
    pixel_scale : float = None
        The pixel scale of the coordinates array to generate.
    polar : bool = False
        Output the coordinates in polar (r, phi) coordinates.
    fft_style : bool = False
        If True, use FFT-style centering. For even npixels this produces integer
        centred coordinates. For odd npixels this is identical to the default.

    Returns
    -------
    coordinates : Array
        The array of pixel-centre coordinates.
    """
    supplied = sum(value is not None for value in (diameter, radius, pixel_scale))
    if supplied != 1:
        raise ValueError(
            "Exactly one of diameter, radius, or pixel_scale must be provided."
        )

    if diameter is not None:
        pixscale = diameter / npixels
    elif radius is not None:
        pixscale = 2 * radius / npixels
    else:
        pixscale = pixel_scale

    # Default to symmetric pixel-centre coordinates, half-integer for even sizes
    offsets = (0.0, 0.0)

    # FFT-style: shift by +0.5 pixel for even N so coordinates become integer-centred
    if fft_style and (npixels % 2 == 0):
        offsets = (pixscale / 2.0, pixscale / 2.0)

    # Generate the coordinates
    coords = nd_coords((npixels,) * 2, (pixscale,) * 2, offsets=offsets, indexing="xy")

    # Convert to polar if requested
    if polar:
        return cart2polar(coords)
    return coords


def nd_axes(
    npixels: int | tuple[int, ...],
    pixel_scales: float | tuple[float, ...] = 1.0,
    offsets: float | tuple[float, ...] = 0.0,
) -> tuple[Array, ...]:
    """Return one regularly sampled coordinate vector per physical axis."""
    npixels = dlu.as_size(npixels, name="npixels")
    pixel_scales, offsets = dlu.as_axis(pixel_scales), dlu.as_axis(offsets)
    ndim = max(len(npixels), pixel_scales.shape[-1], offsets.shape[-1])
    npixels = dlu.as_size(npixels, ndim, "npixels")
    pixel_scales = dlu.as_axis(pixel_scales, ndim, "pixel_scales")
    offsets = dlu.as_axis(offsets, ndim, "offsets")

    def axis(n, offset, scale):
        start = -(n - 1) / 2 * scale - offset
        end = (n - 1) / 2 * scale - offset
        return np.linspace(start, end, n)

    return tuple(
        axis(n, offset, scale)
        for n, offset, scale in zip(npixels, offsets, pixel_scales)
    )


def nd_coords(
    npixels: int | tuple[int, ...],
    pixel_scales: float | tuple[float, ...] = 1.0,
    offsets: float | tuple[float, ...] = 0.0,
    indexing: str = "xy",
) -> Array:
    """Returns a set of nd pixel-centre coordinates, with an optional offset. Each
    dimension can have a different number of pixels, pixel scale and offset by passing
    in tuples of values: `nd_coords((10, 10), (1, 2), (0, 1))`. pixel scale and offset
    can also be passed in as floats to apply those values to all dimensions, i.e.:
    `nd_coords((10, 10), 1, 0)`.

    The indexing argument is the same as in numpy.meshgrid, i.e.: giving the
    string ‘ij’ returns a meshgrid with matrix indexing, while ‘xy’ returns a
    meshgrid with Cartesian indexing. In the 2-D case with inputs of length M
    and N, the outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for
    ‘ij’ indexing. In the 3-D case with inputs of length M, N and P, outputs
    are of shape (N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.

    Parameters
    ----------
    npixels : int | tuple[int, ...]
        The number of pixels in each dimension.
    pixel_scales : float | tuple[float, ...] = 1.0
        The pixel_scales of each dimension. If a tuple, the length
        of the tuple must match the number of dimensions. If a float, the same
        scale is applied to all dimensions.
    offsets : float | tuple[float, ...] = 0.0
        The offset of the pixel centers in each dimension. If a tuple, the
        length of the tuple must match the number of dimensions. If a float,
        the same offset is applied to all dimensions.
        set to 0.
    indexing : str = 'xy'
        The indexing of the output. Default is 'xy'. See numpy.meshgrid for
        more details.

    Returns
    -------
    coordinates : Array
        The positions of the pixel centers in the given dimensions.
    """
    if indexing not in ["xy", "ij"]:
        raise ValueError("indexing must be either 'xy' or 'ij'.")

    axes = nd_axes(npixels, pixel_scales, offsets)
    positions = np.array(np.meshgrid(*axes, indexing=indexing))

    # Squeeze the output in case of 1d input
    return np.squeeze(positions)
