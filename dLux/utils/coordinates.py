import jax.numpy as np
from jax import Array
from jax.tree_util import tree_map
from typing import Union
import dLux

__all__ = ["cartesian_to_polar", "polar_to_cartesian",
           "get_pixel_positions", "rotate", "translate", "shear", "compress"]


### Coordinates convertions ###
def cartesian_to_polar(coordinates : Array) -> Array:
    """
    Converts the input (x, y) cartesian cordinates into (r, phi) polar
    coordinates.

    Parameters
    ----------
    coordinates : Array
        The (x, y) cartesian coordinates to be converted into polar cordinates.

    Returns
    -------
    coordinates : Array
        The input cartesian coordinates converted into (r, phi) polar
        cordinates.
    """
    x, y = coordinates
    return np.array([np.hypot(x, y), np.arctan2(y, x)])


def polar_to_cartesian(coordinates : Array) -> Array:
    """
    Converts the input (r, phi) polar coordinates into (x, y) cartesian
    cordinates.

    Parameters
    ----------
    coordinates : Array
        The (r, phi) polar coordinates to be converted into cartesian
        cordinates.

    Returns
    -------
    coordinates : Array
        The input polar coordinates converted into (x, y) cartesian cordinates.
    """
    r, phi = coordinates
    return np.array([r*np.cos(phi), r*np.sin(phi)])


### Positions Calculations ###
def get_pixel_positions(npixels      : Union[int, tuple], 
                        pixel_scales : Union[tuple, float, None] = None,
                        offsets      : Union[tuple, float, None] = None,
                        polar        : bool = False,
                        indexing     : str = 'xy') -> Array:
    """
    Calculates the positions of the pixel centers for the given input. All 
    coordinates are output in units of meters. 
    
    The indexing argument is the same
    as in numpy.meshgrid., ie:  Giving the string ‘ij’ returns a meshgrid with 
    matrix indexing, while ‘xy’ returns a meshgrid with Cartesian indexing. In 
    the 2-D case with inputs of length M and N, the outputs are of shape (N, M) 
    for ‘xy’ indexing and (M, N) for ‘ij’ indexing. In the 3-D case with inputs 
    of length M, N and P, outputs are of shape (N, M, P) for ‘xy’ indexing and 
    (M, N, P) for ‘ij’ indexing. If the output is in polar coordainates, 
    indexing is set to 'xy' and the input must be 2d
    
    Parameters
    ----------
    npixels : Union[int, tuple]
        The number of pixels in each dimension.
    pixel_scales : Union[tuple, float, None] = None
        The pixel scales in each dimension. If a tuple, the length
        of the tuple must match the number of dimensions. If a float, the same
        scale is applied to all dimensions. If None, the scale is set to 1.
    offsets : Union[tuple, float, None] = None
        The offset of the pixel centers in each dimension. If a tuple, the 
        length of the tuple must match the number of dimensions. If a float, 
        the same offset is applied to all dimensions. If None, the offset is 
        set to 0.
    polar : bool = False
        If True, the output is in polar coordinates. If False, the output is in
        cartesian coordinates. Default is False.
    indexing : str = 'xy'
        The indexing of the output. Default is 'xy'. See numpy.meshgrid for more
        details.
    
    Returns
    -------
    positions : Array
        The positions of the pixel centers in the given dimensions.
    """
    if indexing not in ['xy', 'ij']:
        raise ValueError("indexing must be either 'xy' or 'ij'.")
    
    if polar and indexing == 'ij':
        indexing = 'xy'

    # Turn inputs into tuples
    if isinstance(npixels, int):
        npixels = (npixels,)

        if offsets is None:
            offsets = (0.,)
        elif not isinstance(offsets, (float, Array)):
            raise ValueError("offset must be a float or Array if npixels "
                             "is an int.")
        else:
            offsets = (offsets,)

        if pixel_scales is None:
            pixel_scales = (1.,)
        elif not isinstance(pixel_scales, (float, Array)):
            raise ValueError("pixel_scales must be a float or Array if npixels "
                             "is an int.")
        else:
            pixel_scales = (pixel_scales,)
        
    # Check input 
    else:
        if offsets is None:
            offsets = tuple([0.]*len(npixels))
        elif not isinstance(offsets, tuple):
            raise ValueError("offset must be an be a float or Array if npixels "
                             "is an int.")
        else:
            if len(offsets) != len(npixels):
                raise ValueError("offset must have the same length as npixels.")
            
        if pixel_scales is None:
            pixel_scales = tuple([1.]*len(npixels))
        elif isinstance(pixel_scales, float):
            pixel_scales = tuple([pixel_scales]*len(npixels))
        elif not isinstance(pixel_scales, tuple):
            raise ValueError("pixel_scales must be a tuple if npixels is a tuple.")
        else:
            if len(pixel_scales) != len(npixels):
                raise ValueError("pixel_scales must have the same length as npixels.")
    
    def pixel_fn(n, offset, scale):
        pix = np.arange(n) - (n - 1) / 2.
        pix *= scale
        pix -= offset
        return pix
    
    pixels = tree_map(pixel_fn, npixels, offsets, pixel_scales)

    # ouput (x, y) for 2d, else in order
    positions = np.array(np.meshgrid(*pixels, indexing=indexing))

    if polar:
        if len(npixels) != 2:
            raise ValueError("polar coordinates are only defined for 2D arrays.")
        return cartesian_to_polar(positions)

    # Squeeze for empty axis removal with 1d
    return np.squeeze(positions)


### Coordinates Transformations ###
def rotate(coordinates: Array, rotation: Array) -> Array:
    """
    Rotate the coordinate system by a pre-specified amount.

    Parameters
    ----------
    coordinates : Array, meters
        A `(2, npix, npix)` representation of the coordinate 
        system. The leading dimensions specifies the x and then 
        the y coordinates in that order. 
    rotation : Array, radians
        The counter-clockwise rotation to apply.

    Returns
    -------
    coordinates : Array, meters
        The rotated coordinate system. 
    """
    x, y = coordinates[0], coordinates[1]
    new_x = np.cos(-rotation) * x + np.sin(-rotation) * y
    new_y = -np.sin(-rotation) * x + np.cos(-rotation) * y
    return np.array([new_x, new_y])


def translate(coordinates: Array, centre: Array) -> Array:
    """
    Move the center of the coordinate system by some 
    amount (centre). 

    Parameters
    ----------
    coordinates : Array, meters
        The (x, y) coordinates with the dimensions 
        (2, npix, npix).
    centre : Array, meters
        The (x, y) coordinates of the new centre 
        with dimensions (2,)

    Returns
    -------
    coordinates: Array, meters
        The translated coordinate system. 
    """
    return coordinates - centre[:, None, None]


def shear(coordinates: Array, shear: Array) -> Array:
    """
    Apply a shear to the coordinate system. 

    Parameters
    ----------
    coordinates : Array, meters
        The (x, y) coordinates with the dimensions 
        (2, npix, npix).
    shear : Array
        The (x, y) shear with dimensions (2,)

    Returns
    -------
    coordinates: Array, meters
        The sheared coordinate system. 
    """
    trans_coordinates: Array = np.transpose(coordinates, (0, 2, 1))
    return coordinates + trans_coordinates * shear[:, None, None]


def compress(coordinates: Array, compression: Array) -> Array:
    """
    Apply a compression to the coordinates.

    Parameters
    ----------
    coordinates : Array, meters
        The (x, y) coordinates with the dimensions 
        (2, npix, npix).
    compression : Array
        The (x, y) compression with dimensions (2,)

    Returns
    -------
    coordinates : Array, meters
        The compressed coordinates. 
    """
    return coordinates * compression[:, None, None]


