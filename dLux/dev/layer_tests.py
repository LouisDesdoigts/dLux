from layers import *
from matplotlib import pyplot
from jax import numpy as np
from jax.config import config
from typing import TypeVar

#config.update("jax_debug_nans", True)
Vector = TypeVar("Vector")


hexagonal_basis = HexagonalBasis(0, 0, 1, 256, 9)

#vertices = hexagonal_basis._vertices()
#pyplot.plot(vertices[0], vertices[1])
#pyplot.show()

# NOTE: Happy with these working as is.
n, m = hexagonal_basis._noll_index(np.array(np.arange(1, 10)))

#radial = hexagonal_basis._radial_zernike(3, 3, np.zeros((256, 256)))
#print(radial)
#pyplot.imshow(radial)
#pyplot.colorbar()
#pyplot.show()

# hexagonal_basis._basis()

positions = get_radial_positions(hexagonal_basis._npix, 0, 0)
rho = 2 / 256 * positions[0]
theta = positions[1]

# The non-PEP8 variable names come from the standard 
# mathematical form y = mx + c. This can also be expressed
# as ay + bx = c. If we use the two point form for the 
# standard expression we get:
#
#           y_2 - y_1
# y - y_1 = ---------(x - x_1)
#           x_2 - x_1
#
# This can be rearranged into the form. 
#
# (x_2 - x_1) y - (y_2 - y_1) x = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
# 
# More simply the form ay + bx = c, where a, b and c are 
# functions of the verticies points; x_1, x_2, y_1 and y_2.
# In polar form this gives us:
# 
#                  c
# r = ---------------------------
#     a cos(theta) + b sin(theta)
#
# Polar coordinates are used because we can always use r >
# to specify the aperture, avoiding the use of logic.

vertices = hexagonal_basis._vertices()

i = np.arange(-1, vertices.shape[1])
a = (vertices[0, i + 1] - vertices[0, i])
b = (vertices[1, i + 1] - vertices[1, i])
c = (vertices[0, i + 1] - vertices[0, i]) * vertices[1, i] -\
    (vertices[1, i + 1] - vertices[1, i]) * vertices[0, i]
angle = np.arctan(vertices[1, i + 1] / vertices[0, i + 1])


@jax.vmap
def less_than(array : Matrix, comparator : Matrix) -> Matrix:
    """
    < comparator for Tensors.

    Parameters
    ----------
    array : Matrix
        The array on the left of the <.
    comparator : Matrix
        The array on the right of the <.

    Returns
    is_less_than : Matrix
        Elementwise comparison of the arrays. 
    """
    return array < comparator


@jax.vmap
def greater_than(array : Matrix, comparator : Matrix) -> Matrix:
    """
    < comparator for Tensors.

    Parameters
    ----------
    array : Matrix
        The array on the left of the <.
    comparator : Matrix
        The array on the right of the <.

    Returns
    is_less_than : Matrix
        Elementwise comparison of the arrays. 
    """
    return array > comparator


@jax.vmap
def affine(theta : Matrix, a : Vector, b : Vector, c : Vector) -> Matrix:
    """
    Calculate and edge of the polygonal aperture.

    Parameters
    ----------
    theta : Matrix
        The polar angle of all the points in the plane.
    a : Vector
        A vector of coefficients.
    b : Vector
        A vector of coefficients.
    c : Vector
        A vector of coefficients.

    Returns 
    -------
    affine : Matrix
        The affine transformation mapped over the space. 
    """
    return c / (a * np.sin(theta) + b * np.cos(theta))

# I also need to implement the angular check for this to work. 
rho = np.tile(rho, (a.shape[0], 1, 1))
theta = np.tile(theta, (a.shape[0], 1, 1))

print(angle)

radials = less_than(rho, affine(theta, a, b, c))
# The problem is that the angles are defined from -pi for some 
# fucking stupid reason. This can be left for tomorrow.
angular = np.logical_and(less_than(theta, angle[i]), 
    greater_than(theta, angle))
#sum(axis=0)

pyplot.imshow(angular[0])
pyplot.colorbar()
pyplot.show()
pyplot.imshow(angular[1])
pyplot.colorbar()
pyplot.show()
pyplot.imshow(angular[2])
pyplot.colorbar()
pyplot.show()
pyplot.imshow(angular[3])
pyplot.colorbar()
pyplot.show()
pyplot.imshow(angular[4])
pyplot.colorbar()
pyplot.show()
pyplot.imshow(angular[5])
pyplot.colorbar()
pyplot.show()

pyplot.imshow(aperture.sum(axis=0))
pyplot.colorbar()
pyplot.show()
#zernikes = hexagonal_basis._zernike_basis()
#for i in range(9):
#    pyplot.subplot(3, 3, i + 1)
#    pyplot.title(f"{n[i]}, {m[i]}")
#    pyplot.imshow(zernikes[i])
#pyplot.show()
