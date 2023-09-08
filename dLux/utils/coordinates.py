import jax.numpy as np
from jax import Array
from jax.tree_util import tree_map
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


def translate_coords(coords, centre):
    return coords - centre[:, None, None]


def compress_coords(coords, compress):
    return coords * compress[:, None, None]


def shear_coords(coords, shear):
    trans_coords = np.transpose(coords, (0, 2, 1))
    return coords + trans_coords * shear[:, None, None]


def rotate_coords(coords, rotation):
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
def pixel_coords(
    npixels: int,
    diameter: float,
    polar=False,
) -> Array:
    """
    Calculates the coordinates of the pixel centers for the given input,
    assuming an equal size and pixel scale in all dimensions. All
    coordinates are output in units of metres. This function is essentially a
    reduced version of the full `pixel_coordinates` function that gives
    flexibility to have different dimension sizes and scales.

    Parameters
    ----------
    npixels : int
        The number of pixels in all dimensions.
    diameter : float
        The diameter of the coordinates array to generate.
    polar : bool = False
        If True, the output is in polar coordinates. If False, the output is in
        Cartesian coordinates. ndims must be 2 if polar is True.

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
    Calculates the coordinates of the pixel centers for the given input. All
    coordinates are output in units of metres.

    The indexing argument is the same as in numpy.meshgrid., i.e.: Giving the
    string ‘ij’ returns a meshgrid with matrix indexing, while ‘xy’ returns a
    meshgrid with Cartesian indexing. In the 2-D case with inputs of length M
    and N, the outputs are of shape (N, M) for ‘xy’ indexing and (M, N) for
    ‘ij’ indexing. In the 3-D case with inputs of length M, N and P, outputs
    are of shape (N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.
    If the output is in polar coordinates, indexing is set to 'xy' and the
    input must be 2d.

    Parameters
    ----------
    npixels : Union[int, tuple]
        The number of pixels in each dimension.
    pixel_scales : Union[tuple, float] = 1.
        The pixel_scales of each dimension. If a tuple, the length
        of the tuple must match the number of dimensions. If a float, the same
        scale is applied to all dimensions. If None, the scale is set to 1.
    offsets : Union[tuple, float] = 0.
        The offset of the pixel centers in each dimension. If a tuple, the
        length of the tuple must match the number of dimensions. If a float,
        the same offset is applied to all dimensions. If None, the offset is
        set to 0.
    indexing : str = 'xy'
        The indexing of the output. Default is 'xy'. See numpy.meshgrid for
        more details.

    Returns
    -------
    positions : Array
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
        # TODO: calculate the start and end points first and then use linspace
        # so that ops are done on floats not arrays
        # scale = diam / n
        pix = np.arange(n) - (n - 1) / 2.0
        pix *= scale
        pix -= offset
        return pix

    # Generate the linear edges of each axes
    # TODO: tree_flatten()[0] to avoid squeeze?
    lin_pixels = tree_map(pixel_fn, npixels, offsets, pixel_scales)

    # output (x, y) for 2d, else in order
    positions = np.array(np.meshgrid(*lin_pixels, indexing=indexing))

    # Squeeze for empty axis removal in 1d case
    return np.squeeze(positions)
