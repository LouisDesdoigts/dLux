"""
src/utilities/image.py
----------------------
A static class that implements centering conventions for dealing 
with para-axial arrays. 
"""
__author__ = "Jordan Dennis"
__date__ = "07/07/2022"


import jax.numpy as np
from typing import TypeVar


Vector = TypeVar("Vector")
Matrix = TypeVar("Matrix")
Tensor = TypeVar("Tensor")
Array = TypeVar("Array")


def get_pixel_vector(
        number_of_pixels : int,
        pixel_offset : int = 0) -> Array:
    """
    Generate the coordinates along the edge of the image.

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along the edge of the pixel plane.
    pixel_offset : int
        The number of pixels to offset the zero.

    Returns
    -------
    x : Array
        The pixel positions along the edge of the image.
    """
    return np.arange(number_of_pixels) - number_of_pixels / 2. \
        + 0.5 + pixel_offset
    

def get_pixel_positions(
        number_of_pixels : int, 
        x_pixel_offset : int = 0,
        y_pixel_offset : int = 0) -> Array:
    """
    Generates offset para-axial coordinates, defining the optical 
    axis. 

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along the side of the output array.
    x_pixel_offset : int = 0.
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_pixel_offset : int = 0
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns 
    -------
    pixel_positions : Array
        The pixel positions in the square output array with the 
        correct offsets
    """
    x = get_pixel_vector(number_of_pixels, x_pixel_offset)
    y = get_pixel_vector(number_of_pixels, y_pixel_offset)
    return np.array(np.meshgrid(x, y))


# TODO:     
def get_radial_positions(number_of_pixels : int,
        x_pixel_offset : int,
        y_pixel_offset : int) -> Matrix:
    """
    Generate the radial coordinates of each pixel. 

    Parameters
    ----------
    number_of_pixels : int
        The number of pixels along one edge of the square array.
    x_pixel_offset : int = 0.
        The x offset of the centre of the coordinate system in the 
        square output array.
    y_pixel_offset : int = 0
        The y offset of the centre of the coordinate system in the 
        square output array.

    Returns
    -------
    positions : Matrix
        A grid of the radial coordinates.
    """
    # NOTE: I think CircularAperture is Broken 
    positions = get_pixel_positions(number_of_pixels, 
        x_pixel_offset, y_pixel_offset)
    x, y = positions[0], positions[1]
    rho = np.hypot(x, y)
    theta = np.arctan2(y[::-1, :], x)[::-1, :]
    return np.array([rho, theta])
