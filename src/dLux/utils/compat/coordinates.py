"""Retained coordinate-array interfaces backed by Grid and transform kernels."""

import jax.numpy as np
from jax import Array

from ..batching import as_axis
from ..grids import (
    Grid,
    as_size,
)

__all__ = [
    "as_axis",
    "imshow_extent",
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
    """
    Shears the coordinates by a given factor. Shear must have shape (2,).

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
    trans_coords = np.transpose(coords, (0, 2, 1))
    return coords + trans_coords * shear[:, None, None]


def rotate_coords(coords: Array, rotation: float) -> Array:
    """
    Rotates the coordinates by a given angle.

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

    x, y = coords
    new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
    new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
    return np.array([new_x, new_y])


def distort_coords(coords: Array, coeffs: Array, pows: Array):
    """
    Apply a 2D polynomial distortion to some coordinates

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
    pow_base = np.multiply(*(coords[:, None, ...] ** pows[..., None, None]))
    distortion = np.sum(coeffs[..., None, None] * pow_base[None, ...], axis=1)
    return coords + distortion


def imshow_extent(size: float) -> Array:
    """Return square plot edges [xmin, xmax, ymin, ymax] for a full width."""
    return Grid(n=(1, 1), d=size).extent()


def pixel_coords(
    npixels: int,
    diameter=None,
    radius=None,
    pixel_scale=None,
    polar=False,
    fft_style=False,
) -> Array:
    """Return square pixel-centre coordinates using the retained array API.

    Exactly one of diameter, radius or pixel_scale sets the spacing. FFT-style
    centring places an even grid's centre half a pixel below zero. Prefer
    Grid(n=(n, n), diam=diameter).coordinates for new code.
    """
    supplied = sum(value is not None for value in (diameter, radius, pixel_scale))
    if supplied != 1:
        raise ValueError(
            "Exactly one of diameter, radius, or pixel_scale must be provided."
        )
    if diameter is not None:
        pixel_scale = diameter / npixels
    elif radius is not None:
        pixel_scale = 2 * radius / npixels

    offset = pixel_scale / 2 if fft_style and npixels % 2 == 0 else 0.0
    coordinates = nd_coords((npixels, npixels), pixel_scale, offset)
    return cart2polar(coordinates) if polar else coordinates


def nd_axes(npixels, pixel_scales=1.0, offsets=0.0) -> tuple[Array, ...]:
    """Return regular physical-axis vectors; legacy offsets subtract from centres."""
    counts = as_size(npixels, name="npixels")
    ndim = max(len(counts), as_axis(pixel_scales).shape[-1], as_axis(offsets).shape[-1])
    counts = as_size(counts, ndim, "npixels")
    spacing = as_axis(pixel_scales, len(counts), "pixel_scales")
    centers = -as_axis(offsets, len(counts), "offsets")
    axes = []
    for i, n in enumerate(counts):
        # Match the release's linspace arithmetic, including float32 rounding.
        half_width = (n - 1) / 2 * spacing[..., i]
        start, end = -half_width + centers[..., i], half_width + centers[..., i]
        axes.append(np.linspace(start, end, n, axis=-1))
    return tuple(axes)


def nd_coords(npixels, pixel_scales=1.0, offsets=0.0, indexing="xy") -> Array:
    """Return the retained squeezed meshgrid representation.

    Grid coordinates retain their component and singleton sample axes. This
    compatibility function preserves the original squeezing and NumPy meshgrid
    convention: xy swaps only the first two physical axes in higher dimensions.
    """
    if indexing not in ("xy", "ij"):
        raise ValueError("indexing must be either 'xy' or 'ij'.")
    axes = nd_axes(npixels, pixel_scales, offsets)
    ndim = len(axes)
    shape = tuple(axis.shape[-1] for axis in axes)
    batch = np.broadcast_shapes(*(axis.shape[:-1] for axis in axes))
    components = []
    for i, axis in enumerate(axes):
        local_shape = [1] * ndim
        local_shape[i] = shape[i]
        axis = axis.reshape(axis.shape[:-1] + tuple(local_shape))
        components.append(np.broadcast_to(axis, batch + shape))
    coordinates = np.stack(components, axis=-ndim - 1)
    if indexing == "xy" and ndim > 1:
        coordinates = np.swapaxes(coordinates, -ndim, 1 - ndim)
    return np.squeeze(coordinates)


def gen_powers(degree: int):
    """
    Generates the powers required for a 2d polynomial

    Parameters
    ----------
    degree : int
        Maximum power to generate

    Returns
    -------
    xpows : Array
        x axis powers
    ypows : Array
        y axis powers
    """
    n = triangular_number(degree)
    vals = np.arange(n)

    # Ypows
    tris = triangular_number(np.arange(degree))
    ydiffs = np.repeat(tris, np.arange(1, degree + 1))
    ypows = vals - ydiffs

    # Xpows
    tris = triangular_number(np.arange(1, degree + 1))
    xdiffs = np.repeat(n - np.flip(tris), np.arange(degree, 0, -1))
    xpows = np.flip(vals - xdiffs)

    return np.array([xpows, ypows])


from ..numerical.math import triangular_number

__all__ += ["gen_powers"]


def cart2polar(coordinates: Array) -> Array:
    """
    Converts the input (x, y) Cartesian coordinates into (r, phi) polar
    coordinates.

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
    x, y = coordinates
    return np.array([np.hypot(x, y), np.arctan2(y, x)])


def polar2cart(coordinates: Array) -> Array:
    """
    Converts the input (r, phi) polar coordinates into (x, y) Cartesian
    coordinates.

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
    r, phi = coordinates
    return np.array([r * np.cos(phi), r * np.sin(phi)])
