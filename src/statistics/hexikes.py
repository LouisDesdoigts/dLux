"""
src/statistics/hexikes.py
-------------------------
An attempt at generating the hexike polynomials for optimising a 
hexagonal surface for aberations. 
"""

from dLux.statistics import zernikes
from dLux.layers import HexagonalAperture
from typing import TypeVar


# TODO: Work out how to stop this repetition of logic. 
def _hexagonal_aperture(
        number_of_pixels : int, 
        x_pixel_offset : int,
        y_pixel_offset : int,
        maximum_radius : float = 1.) -> Array:
    """
    Generate a binary mask representing the pixels occupied by 
    the aperture. 

    Parameters
    ----------
    number_of_pixels : int
        The ouput array will have the shape `aperture.shape == 
        (number_of_pixels, number_of_pixels)`. 
    x_pixel_offset : int
        The offset of the aperture in the square output array 
    maximum_radius : float 
        The radius of the smallest circle that can completely contain
        the entire aperture. 

    Returns
    -------
    aperture : Array
        The bitmask that represents the circular aperture.        
    """
    # NOTE: This is where I need to choose generate everything correctly 
    # I think. 
    x, y = get_pixel_positions(number_of_pixels)

    rectangle = (np.abs(x + ) <= maximum_radius / 2.) \
        & (np.abs(y) <= (x + 1) * np.sqrt(3))

    left_triangle = (x <= -0.5) \
        & (x >= -1) \
        & (np.abs(y) <= (x + 1) * np.sqrt(3))

    right_triangle = (x >= 0.5) \
        & (x <= 1) \
        & (np.abs(y) <= (1 - x) * np.sqrt(3))

    hexagon = rectangle | left_triangle | right_triangle
    return np.asarray(hexagon).astype(float)


def hexikes(
        number_of_hexikes : int = 15, 
        number_of_pixels : int = 512,
        maximum_radius : float = 1.)
    """
    The hexike polynomials up until `number_of_hexikes` on a square
    array that `number_of_pixels` by `number_of_pixels`. The 
    polynomials can be restricted to a smaller subset of the 
    array by passing an explicit `maximum_radius`. The polynomial
    will then be defined on the largest hexagon that fits with a 
    circle of radius `maximum_radius`. 
    
    Parameters
    ----------
    number_of_hexikes : int = 15
        The number of basis terms to generate. 
    number_of_pixels : int = 512
        The size of the array to compute the hexikes on.
    maximum_radius : float = 1.
        The radius of the the smallest circle that can contain the 
        hexagonal surface. 

    Returns
    -------
    hexikes : Tensor
        The hexike polynomials evaluated on the square arrays
        containing the hexagonal apertures until `maximum_radius`.
        The leading dimension is `number_of_hexikes` long and 
        each stacked array is a basis term. The final shape is:
        ```py
        hexikes.shape == (number_of_hexikes, number_of_pixels, number_of_pixels)
        ```
    """
    
