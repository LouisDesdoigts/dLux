"""
A script containing jax based functios to handle paraxial coordinate
array generation, to be used all throughout the package

Notes: 
 - All positions and coordinates are defined relative to the center
   of the the arrays
 - Positions are defined with pixel sizes equal to 1
 - Coordinates are defined with pixel sizes equal to the inputs
   pixel scale, ie physical units 
"""
__author__ = "Jordan Dennis"
__author__ = "Louis Desdoigts"
__date__ = "16/08/2022"
__all__ = ["get_positions_vector",  "get_pixel_positions",
           "get_polar_positions",   "get_coordinates_vector", 
           "get_pixel_coordinates", "get_polar_coordinates"]


import jax.numpy as np
import typing

Scalar = typing.NewType("Scalar", np.ndarray) # 0d
Vector = typing.NewType("Vector", np.ndarray) # 1d
Array =  typing.NewType("Array",  np.ndarray) # 2d +


### Positions Calculations ###

def get_positions_vector(
        number_of_pixels : int,
        pixel_offset     : float = 0) -> Vector:
    """
    Generate the vector pixel positions relative to the center.

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels in the vector.
    pixel_offset : float
        The number of pixels to offset from the center.

    Returns
    -------
    x : Vector
        The vector of pixel positions
    """
    return np.arange(number_of_pixels) - \
            (number_of_pixels + 1) / 2. - pixel_offset


def get_pixel_positions(
        number_of_pixels : int, 
        x_pixel_offset   : float = 0,
        y_pixel_offset   : float = 0) -> Array:
    """
    Returns arrays of pixel positions relative to the 
    (central) optical axis.

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along the side of the output array.
    x_pixel_offset : float = 0, pixels
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_pixel_offset : float = 0, pixels
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns 
    -------
    pixel_positions : Array
        The (x, y) array of pixel positions
    """
    x = get_positions_vector(number_of_pixels, x_pixel_offset)
    y = get_positions_vector(number_of_pixels, y_pixel_offset)
    return np.array(np.meshgrid(x, y))


def get_polar_positions(
        number_of_pixels : int,
        x_pixel_offset   : float = 0,
        y_pixel_offset   : float = 0) -> Array:
    """
    Generate the polar positions of each pixel. 

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along one edge of the square array.
    x_pixel_offset : float = 0
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_pixel_offset : float = 0
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns
    -------
    positions : Matrix
        The (r, phi) array of the radial coordinates.
    """
    positions = get_pixel_positions(number_of_pixels, 
        x_pixel_offset, y_pixel_offset)
    return dLux.utils.helpers.cart2polar(positions[0], positions[1])



### Coordinate Calculations ###

def get_coordinates_vector(
        number_of_pixels : int,
        pixel_scale      : float,
        offset           : float = 0) -> Vector:
    """
    Generate the vector pixel coordinates relative to the center.

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels in the vector.
    offset : float, meters
        The offset from the center.

    Returns
    -------
    x : Vector
        The vector of pixel coordinates
    """
    return pixel_scale * get_positions_vector(number_of_pixels, 
                                              pixel_scale * offset)

def get_pixel_coordinates(
        number_of_pixels : int,
        pixel_scale      : float,
        x_offset         : float = 0,
        y_offset         : float = 0) -> Array:
    """
    Returns the physical (x, y) coordinate array of pixel 
    positions relative to the (central) optical axis.

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along the side of the output array.
    pixel_scale : float, meters
        The physical size of each pixel
    x_offset : float = 0, meters
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_offset : float = 0, meters
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns 
    -------
    pixel_positions : Array
        The (x, y) coordinate arrays in the square output array 
        with the correct offsets
    """
    return pixel_scale * get_pixel_positions(number_of_pixels,
                                        pixel_scale * x_offset,
                                        pixel_scale * y_offset)


def get_polar_coordinates(
        number_of_pixels : int,
        pixel_scale      : float,
        x_offset         : float = 0,
        y_offset         : float = 0) -> Array:
    """
    Generate the (r, phi) polar coordinates of each pixel. 

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along one edge of the square array.
    pixel_scale : float, meters
        The physical size of each pixel
    x_pixel : float = 0.
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_pixel : float = 0
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns
    -------
    positions : Array
        The (r, phi) coordinate arrays in the square output array 
        with the correct offsets
    """
    return pixel_scale * get_polar_positions(number_of_pixels,
                                        pixel_scale * x_offset,
                                        pixel_scale * y_offset)



def invert_x(array : Array):
    """
    Inverts the input array 
    """


