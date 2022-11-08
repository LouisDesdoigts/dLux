import jax.numpy as np
import dLux

__all__ = ["cartesian_to_polar", "polar_to_cartesian",
           "get_positions_vector",  "get_pixel_positions",
           "get_polar_positions",   "get_coordinates_vector",
           "get_pixel_coordinates", "get_polar_coordinates"]


Array = np.ndarray


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
def get_positions_vector(npixels      : int,
                         pixel_offset : Array = np.array(0.)) -> Array:
    """
    Generate the vector pixel positions relative to the center.

    Parameters
    ----------
    npixels : int
        The number of pixels in the vector.
    pixel_offset : Array = np.array(0.)
        The number of pixels to offset from the center.

    Returns
    -------
    positions : Array
        The vector of pixel positions.
    """
    return np.arange(npixels) - \
            (npixels - 1) / 2. - pixel_offset


def get_pixel_positions(npixels        : int,
                        x_pixel_offset : Array = np.array(0.),
                        y_pixel_offset : Array = np.array(0.)) -> Array:
    """
    Returns arrays of pixel positions relative to the (central) optical axis.

    Parameters
    ----------
    npixels : int
        The number of pixels along the side of the output array.
    x_pixel_offset : Array, pixels = np.array(0.)
        The x offset of the centre of the coordinate system in the
        square output array.
    y_pixel_offset : Array, pixels = np.array(0.)
        The y offset of the centre of the coordinate system in the
        square output array.

    Returns
    -------
    positions : Array
        The (x, y) array of pixel positions.
    """
    x = get_positions_vector(npixels, x_pixel_offset)
    y = get_positions_vector(npixels, y_pixel_offset)
    return np.array(np.meshgrid(x, y))


def get_polar_positions(npixels        : int,
                        x_pixel_offset : Array = np.array(0.),
                        y_pixel_offset : Array = np.array(0.)) -> Array:
    """
    Generate the polar positions of each pixel.

    Parameters
    ----------
    npixels : int
        The number of pixels along one edge of the square array.
    x_pixel_offset : Array = np.array(0.)
        The x offset of the centre of the coordinate system in the
        square output array.
    y_pixel_offset : Array = np.array(0.)
        The y offset of the centre of the coordinate system in the
        square output array.

    Returns
    -------
    positions : Array
        The (r, phi) array of the radial coordinates.
    """
    positions = get_pixel_positions(npixels, x_pixel_offset, y_pixel_offset)
    return cartesian_to_polar(positions)


### Coordinate Calculations ###
def get_coordinates_vector(npixels     : int,
                           pixel_scale : Array,
                           offset      : Array = np.array(0.)) -> Array:
    """
    Generate the vector pixel coordinates relative to the center.

    Parameters
    ----------
    npixels : int
        The number of pixels in the vector.
    offset : Array, meters = np.array(0.)
        The offset from the center.

    Returns
    -------
    x : Array
        The vector of pixel coordinates.
    """
    return pixel_scale * get_positions_vector(npixels, pixel_scale * offset)


def get_pixel_coordinates(npixels     : int,
                          pixel_scale : Array,
                          x_offset    : Array = np.array(0.),
                          y_offset    : Array = np.array(0.)) -> Array:
    """
    Returns the physical (x, y) coordinate array of pixel
    positions relative to the (central) optical axis.

    Parameters
    ----------
    npixels : int
        The number of pixels along the side of the output array.
    pixel_scale : Array, meters
        The physical size of each pixel
    x_offset : Array = np.array(0.), meters
        The x offset of the centre of the coordinate system in the
        square output array.
    y_offset : Array = np.array(0.), meters
        The y offset of the centre of the coordinate system in the
        square output array.

    Returns
    -------
    pixel_positions : Array
        The (x, y) coordinate arrays in the square output array with the
        correct offsets.
    """
    return pixel_scale * get_pixel_positions(npixels,
                                             pixel_scale * x_offset,
                                             pixel_scale * y_offset)


def get_polar_coordinates(npixels     : int,
                          pixel_scale : Array,
                          x_offset    : Array = np.array(0.),
                          y_offset    : Array = np.array(0.)) -> Array:
    """
    Generate the (r, phi) polar coordinates of each pixel.

    Parameters
    ----------
    npixels : int
        The number of pixels along one edge of the square array.
    pixel_scale : Array, meters
        The physical size of each pixel
    x_offset : Array = np.array(0.).
        The x offset of the centre of the coordinate system in the
        square output array.
    y_offset : Array = np.array(0.)
        The y offset of the centre of the coordinate system in the
        square output array.

    Returns
    -------
    positions : Array
        The (r, phi) coordinate arrays in the square output array with the
        correct offsets
    """
    return pixel_scale * get_polar_positions(npixels,
                                             pixel_scale * x_offset,
                                             pixel_scale * y_offset)


