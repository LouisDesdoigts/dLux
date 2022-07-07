"""
src/statistics/hexikes.py
-------------------------
An attempt at generating the hexike polynomials for optimising a 
hexagonal surface for aberations. 
"""

import jax.numpy as np

from matplotlib import pyplot
from dLux.statistics import zernike_basis
from typing import TypeVar

Array = TypeVar("Array")
Tensor = TypeVar("Tensor")


def _get_pixel_vector(
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
    

def _get_pixel_positions(
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
    x = _get_pixel_vector(number_of_pixels, x_pixel_offset)
    y = _get_pixel_vector(number_of_pixels, y_pixel_offset)
    return np.meshgrid(x, y)



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
        The offset of the aperture in the square output array in the 
        x direction.
    y_pixel_offset : int
        The offset of the aperture in the square output array in the 
        y direction. 
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
    x, y = _get_pixel_positions(number_of_pixels, x_pixel_offset,
        y_pixel_offset)

    x *= 2 / number_of_pixels
    y *= 2 / number_of_pixels

    rectangle = (np.abs(x) <= maximum_radius / 2.) \
        & (np.abs(y) <= (maximum_radius * np.sqrt(3) / 2.))

    left_triangle = (x <= -0.5) \
        & (x >= -1) \
        & (np.abs(y) <= (x + 1) * np.sqrt(3))

    right_triangle = (x >= 0.5) \
        & (x <= 1) \
        & (np.abs(y) <= (1 - x) * np.sqrt(3))

    hexagon = rectangle | left_triangle | right_triangle
    return np.asarray(hexagon).astype(float)


def hexike_basis(
        number_of_hexikes : int = 15, 
        number_of_pixels : int = 512,
        x_pixel_offset : int = 0,
        y_pixel_offset : int = 0,
        maximum_radius : float = 1.) -> Tensor:
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
    x_pixel_offset : int
        The offset of the aperture in the square output array in the 
        x direction.
    y_pixel_offset : int
        The offset of the aperture in the square output array in the 
        y direction. 
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
    aperture = _hexagonal_aperture(number_of_pixels, x_pixel_offset,
        y_pixel_offset, maximum_radius)
    pixel_area = aperture.sum()

    centre = number_of_pixels // 2
    x_centre = centre + x_pixel_offset
    y_centre = centre + y_pixel_offset
    remainder = number_of_pixels % 2

    offset_zernikes = np.zeros((number_of_hexikes, 
            number_of_pixels, number_of_pixels))\
        .at[:, x_centre - centre : x_centre + centre + remainder,
            y_centre - centre : y_centre + centre + remainder]\
        .set(zernike_basis(number_of_hexikes, number_of_pixels))

    offset_hexikes = np.zeros((number_of_hexikes, 
            number_of_pixels, number_of_pixels))\
        .at[1].set(aperture)

    for j in np.arange(1, number_of_hexikes): # Index of the zernike
        intermediate = offset_zernikes[j + 1] * aperture
        for k in np.arange(1, j + 1): # Gram-Schmidt orthonormalisation
            coefficient = -1 / pixel_area * \
                (offset_zernikes[j + 1] * offset_hexikes[k] \
                    * aperture).sum()
            if coefficient != 0:
                intermediate += coefficient * offset_hexikes[k]

        # Normalisation of the intermediate. Final step in the 
        # Gram-Schmidt orthonormalisation.
        offset_hexikes = offset_hexikes\
            .at[j + 1]\
            .set(intermediate / \
                np.sqrt((intermediate ** 2).sum() / pixel_area))

    return offset_hexikes


#hexikes = hexike_basis(5)
number_of_pixels = 256
x_pixel_offset = 100
y_pixel_offset = 50
maximum_radius = 1.
number_of_hexikes = 5

# NOTE: The testing below confirms that the _get_pixel_positions is
# working properly
#positions = _get_pixel_positions(number_of_pixels, x_pixel_offset,
#    y_pixel_offset)
#positions = np.sqrt(np.sum(np.array(positions) ** 2, axis=0))
#
#pyplot.imshow(positions)
#pyplot.show()
#exit()

# NOTE: The aperture generation is correct.
aperture = _hexagonal_aperture(number_of_pixels, x_pixel_offset,
    y_pixel_offset, maximum_radius)

shape = (number_of_hexikes, number_of_pixels, number_of_pixels)
center = number_of_pixels // 2
remainder = number_of_pixels % 2
x_width = (number_of_pixels - x_pixel_offset) // 2
y_width = (number_of_pixels - y_pixel_offset) // 2

zernikes = zernike_basis(number_of_hexikes, number_of_pixels)

pyplot.imshow(zernikes[1])
pyplot.colorbar()
pyplot.show()
print("Hello!")

# NOTE: The assignment script has been fixed. 
# NOTE: Know it has not.
offset_zernikes = np.zeros(shape)\
    .at[:, y_pixel_offset :, x_pixel_offset :]\
    .set(zernikes)

#pyplot.imshow(offset_zernikes[0])
#pyplot.colorbar()
#pyplot.show()
#pyplot.imshow(offset_zernikes[1])
#pyplot.colorbar()
#pyplot.show()
#pyplot.imshow(offset_zernikes[2])
#pyplot.colorbar()
#pyplot.show()
exit()


offset_hexikes = np.zeros((number_of_hexikes, 
        number_of_pixels, number_of_pixels))\
    .at[1].set(aperture)

for j in np.arange(1, number_of_hexikes): # Index of the zernike
    intermediate = offset_zernikes[j + 1] * aperture
    for k in np.arange(1, j + 1): # Gram-Schmidt orthonormalisation
        coefficient = -1 / pixel_area * \
            (offset_zernikes[j + 1] * offset_hexikes[k] \
                * aperture).sum()
        if coefficient != 0:
            intermediate += coefficient * offset_hexikes[k]

    # Normalisation of the intermediate. Final step in the 
    # Gram-Schmidt orthonormalisation.
    offset_hexikes = offset_hexikes\
        .at[j + 1]\
        .set(intermediate / \
            np.sqrt((intermediate ** 2).sum() / pixel_area))


 
print(hexikes[1])

#for i in range(1, 6):
#    pyplot.figure(figsize=(5, 5))
#    pyplot.title(f"{i}th Hexike Polynomial")
#    pyplot.subplot(1, 5, i)
#    pyplot.imshow(hexikes[i])
#
#pyplot.show()
   
