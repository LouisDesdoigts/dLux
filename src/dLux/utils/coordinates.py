import jax.numpy as np
import jax.tree as jtu
from jax import Array
from typing import Union

__all__ = [
    "cart2polar",
    "polar2cart",
    "pixel_coords",
    "nd_coords",
    "translate_coords",
    "compress_coords",
    "shear_coords",
    "rotate_coords",
]


def translate_coords(coords: Array, translation: Array) -> Array:
    """
    Translates the coordinates by to a new centre. Translation must have shape (2,).

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
    return coords - translation[:, None, None]


def compress_coords(coords: Array, compress: Array) -> Array:
    """
    Compresses the coordinates by a given factor. Compress must have shape (2,).

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
    return coords * compress[:, None, None]


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


# Coordinate conversions #
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


# Positions Calculations #
def pixel_coords(npixels: int, diameter: float, polar: bool = False) -> Array:
    """
    Returns a paraxial set of 2d coordinates for each pixel center.

    Parameters
    ----------
    npixels : int
        The output size of the coordinates array to generate.
    diameter : float
        The diameter of the coordinates array to generate.
    polar : bool = False
        Output the coordinates in polar (r, phi) coordinates.

    Returns
    -------
    coordinates : Array
        The array of pixel center coordinates.
    """
    coords = nd_coords((npixels,) * 2, (diameter / npixels,) * 2)
    if polar:
        return cart2polar(coords)
    return coords


def nd_coords(
    npixels: Union[int, tuple],
    pixel_scales: Union[tuple, float] = 1.0,
    offsets: Union[tuple, float] = 0.0,
    indexing: str = "xy",
) -> Array:
    """
    Returns a set of nd pixel center coordinates, with an optional offset. Each
    dimension can have a different number of pixels, pixel scale and offset by passing
    in tuples of values: `nd_coords((10, 10), (1, 2), (0, 1))`. pixel scale and offset
    can also be passed in as floats to apply those values to all dimensions, ie:
    `nd_coords((10, 10), 1, 0)`.

    The indexing argument is the same as in numpy.meshgrid., i.e.: Giving the
    string ‘ij’ returns a meshgrid with matrix indexing, while ‘xy’ returns a
    meshgrid with Cartesian indexing. In the 2-D case with inputs of length M
    and N, the outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for
    ‘ij’ indexing. In the 3-D case with inputs of length M, N and P, outputs
    are of shape (N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.

    Parameters
    ----------
    npixels : Union[int, tuple]
        The number of pixels in each dimension.
    pixel_scales : Union[tuple, float] = 1.
        The pixel_scales of each dimension. If a tuple, the length
        of the tuple must match the number of dimensions. If a float, the same
        scale is applied to all dimensions.
    offsets : Union[tuple, float] = 0.
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

    # Promote npixels to tuple to handle 1d case
    if not isinstance(npixels, tuple):
        npixels = (npixels,)

    # Assume equal pixel scales if not given
    if not isinstance(pixel_scales, tuple):
        pixel_scales = (pixel_scales,) * len(npixels)

    # Assume no offset if not given
    if not isinstance(offsets, tuple):
        offsets = (offsets,) * len(npixels)

    def pixel_fn(n, offset, scale):
        start = -(n - 1) / 2 * scale - offset
        end = (n - 1) / 2 * scale - offset
        return np.linspace(start, end, n)

    # Generate the linear edges of each axes
    # TODO: tree.flatten()[0] to avoid squeeze?
    lin_pixels = jtu.map(pixel_fn, npixels, offsets, pixel_scales)

    # output (x, y) for 2d, else in order
    positions = np.array(np.meshgrid(*lin_pixels, indexing=indexing))

    # Squeeze for empty axis removal in 1d case
    return np.squeeze(positions)
