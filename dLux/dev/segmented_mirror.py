from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot
from typing import TypeVar
from jax import tree_util
from jax import vmap

Vector = TypeVar("Vector")
Matrix = TypeVar("Matrix")
Tensor = TypeVar("Tensor")

config.update("jax_enable_x64", True)


class JWSTPrimaryAperture(PolygonalAperture):
def _wrap(array : Vector, order : Vector) -> tuple:
    """
    Re-order an array and duplicate the first element as an additional
    final element. Satisfies the postcondition `wrapped.shape[0] ==
    array.shape[0] + 1`. This is just a helper method to simplify 
    external object and is not physically important (Only invoke 
    this method if you know what you are doing)

    Parameters
    ----------
    array : Vector
        The 1-dimensional vector to sort and append to. Must be one 
        dimesnional else unexpected behaviour can occur.
    order : Vector
        The new order for the elements of `array`. Will be accessed 
        by invoking `array.at[order]` hence `order` must be `int`
        `dtype`.

    Returns
    -------
    wrapped : Vector
        `array` with `array[0]` appended to the end. The dimensions
        of `array` are also expanded twofold so that the final
        shape is `wrapped.shape == (array.shape[0] + 1, 1, 1)`.
        This is just for the vectorisation demanded later in the 
        code.
    """
    _array = np.zeros((array.shape[0] + 1,))\
        .at[:-1]\
        .set(array.at[order].get())\
        .reshape(-1, 1, 1)
    return _array.at[-1].set(_array[0])
    

def _vertices(vertices : Matrix) -> tuple:
    """
    Generates the vertices that are compatible with the rest of 
    the transformations from the raw data vertices.

    Parameters
    ----------
    vertices : Matrix, meters
        The vertices loaded from the WebbPSF module. 

    Returns
    -------
    x, y, angles : tuple 
        The vertices in normalised positions and wrapped so that 
        they can be used in the generation of the compound aperture.
        The `x` is the x coordinates of the vertices, the `y` is the 
        the y coordinates and `angles` is the angle of the vertex. 
    """
    _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
    _y = (vertices[:, 1] - np.mean(vertices[:, 1]))

    _angles = np.arctan2(_y, _x)
    _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)

    # By default the `np.arctan2` function returns values within the 
    # range `(-np.pi, np.pi)` but comparisons are easiest over the 
    # range `(0, 2 * np.pi)`. This is where the logic implemented 
    # above comes from. 

    order = np.argsort(_angles)

    x = _wrap(_x, order)
    y = _wrap(_y, order)
    angles = _wrap(_angles, order).at[-1].add(2 * np.pi)

    # The final `2 * np.pi` is designed to make sure that the wrap
    # of the first angle is within the angular coordinate system 
    # associated with the aperture. By convention this is the
    # range `angle[0], angle[0] + 2 * np.pi` what this means in 
    # practice is that the first vertex appearing in the array 
    # is used to chose the coordinate system in angular units. 

    return x, y, angles


def _offset(vertices : Matrix, pixel_scale : float) -> tuple:
    """
    Get the offsets of the coordinate system.

    Parameters
    ----------
    vertices : Matrix 
        The unprocessed vertices loaded from the JWST data file.
        The correct shape for this array is `vertices.shape == 
        (2, number_of_vertices)`. 
    pixel_scale : float, meters
        The physical size of each pixel along one of its edges.

    Returns 
    -------
    x_offset, y_offset : float, meters
        The x and y offsets in physical units. 
    """
    x_offset = np.mean(vertices[:, 0]) / pixel_scale
    y_offset = np.mean(vertices[:, 1]) / pixel_scale
    return x_offset, y_offset


def _pixel_scale(vertices : Matrix, pixels : int) -> float:
    """
    The physical dimesnions of a pixel along one edge. 

    Parameters
    ----------
    vertices : Matrix
        The vertices of the aperture in a two dimensional array.
        The pixel scale is assumed to be the same in each dimension
        so only the first row of the vertices is used.
    pixels : int
        The number of pixels that this aperture is going to 
        occupy. 

    Returns
    -------
    pixel_scale : float, meters
        The physical length along one edge of a pixel. 
    """
    return vertices[:, 0].ptp() / pixels


def _coordinates(number_of_pixels : int, vertices : Matrix,
        aperture_pixels : int, phi_naught : float) -> tuple:
    """
    Generates the vectorised coordinate system associated with the 
    aperture.

    Parameters
    ----------
    number_of_pixels : int
        The total number of pixels to generate. This is typically 
        more than `aperture_pixels` as this is used in the padding 
        of the array for the generation of compound apertures.
    vertices : Matrix, meters
        The vertices loaded from the file.
    aperture_pixels : int
        The number of pixels across that the individual aperture.
    phi_naught : float 
        The angle substending the first vertex. 

    Returns 
    -------
    rho, theta : tuple[Tensor]
        The stacked coordinate systems that are typically passed to 
        `_segments` to generate the segments.
    """
    pixel_scale = _pixel_scale(vertices, aperture_pixels)
    x_offset, y_offset = _offset(vertices, pixel_scale)

    positions = get_radial_positions(number_of_pixels,
        -x_offset, -y_offset)

    rho = positions[0] * pixel_scale

    theta = positions[1] 
    theta += 2 * np.pi * (positions[1] < 0.)
    theta += 2 * np.pi * (theta < phi_naught)

    rho = np.tile(rho, (vertices.shape[0], 1, 1))
    theta = np.tile(theta, (vertices.shape[0], 1, 1))
    return rho, theta


def _edges(x : Vector, y : Vector, rho : Tensor, 
        theta : Tensor) -> Tensor:
    """
    Generate lines connecting adjacent vertices.

    Parameters
    ----------
    x : Vector
        The x positions of the vertices.
    y : Vector
        The y positions of the vertices.
    rho : Tensor, meters
        Represents the radial distance of every point from the 
        centre of __this__ aperture. 
    theta : Tensor, Radians
        The angle associated with every point in the final bitmap.

    Returns
    -------
    edges : Tensor
        The edges represented as a Bitmap with the points inside the 
        edge marked as 1. and outside 0. The leading axis contains 
        each unique edge and the corresponding matrix is the bitmap.
    """
    # This is derived from the two point form of the equation for 
    # a straight line (eq. 1)
    # 
    #           y_2 - y_1
    # y - y_1 = ---------(x - x_1)
    #           x_2 - x_1
    # 
    # This is rearranged to the form, ay - bx = c, where:
    # - a = (x_2 - x_1)
    # - b = (y_2 - y_1)
    # - c = (x_2 - x_1)y_1 - (y_2 - y_1)x_1
    # we can then drive the transformation to polar coordinates in 
    # the usual way through the substitutions; y = r sin(theta), and 
    # x = r cos(theta). The equation is now in the form 
    #
    #                  c
    # r = ---------------------------
    #     a sin(theta) - b cos(theta) 
    #
    a = (x[1:] - x[:-1])
    b = (y[1:] - y[:-1])
    c = (a * y[:-1] - b * x[:-1])

    linear = c / (a * np.sin(theta) - b * np.cos(theta))
    return rho < linear 
    

def _wedges(phi : Vector, theta : Tensor) -> Tensor:
    """
    The angular bounds of each segment of an individual hexagon.

    Parameters
    ----------
    phi : Vector
        The angles corresponding to each vertex in order.
    theta : Tensor, Radians
        The angle away from the positive x-axis of the coordinate
        system associated with this aperture. Please note that `theta`
        May not start at zero. 

    Returns 
    -------
    wedges : Tensor 
        The angular bounds associated with each pair of vertices in 
        order. The leading axis of the Tensor steps through the 
        wedges in order arround the circle. 
    """
    # A wedge simply represents the angular bounds of the aperture
    # I have demonstrated below with a hexagon but understand that
    # these bounds are _purely_ angular (see fig 1.)
    #
    #               +-------------------------+
    #               |                ^^^^^^^^^|
    #               |     +--------+^^^^^^^^^^|
    #               |    /        /^*^^^^^^^^^|
    #               |   /        /^^^*^^^^^^^^|
    #               |  /        /^^^^^*^^^^^^^|
    #               | +        +^^^^^^^+^^^^^^|
    #               |  *              /       |
    #               |   *            /        |
    #               |    *          /         |
    #               |     +--------+          |
    #               +-------------------------+
    #               figure 1: The angular bounds 
    #                   between the zeroth and the 
    #                   first vertices. 
    #
    return (phi[:-1] < theta) & (theta < phi[1:])


def _segments(x : Vector, y : Vector, phi : Vector, 
        theta : Tensor, rho : Tensor) -> Tensor:
    """
    Generate the segments as a stacked tensor. 

    Parameters
    ----------
    x : Vector
        The x coordinates of the vertices.
    y : Vector
        The y coordinates of the vertices.
    phi : Vector
        The angles associated with each of the vertices in the order. 
    theta : Tensor
        The angle of every pixel associated with the coordinate system 
        of this aperture. 
    rho : Tensor
        The radial positions associated with the coordinate system 
        of this aperture. 

    Returns 
    -------
    segments : Tensor 
        The bitmaps corresponding to each vertex pair in the vertices.
        The leading dimension contains the unique segments. 
    """
    edges = _edges(x, y, rho, theta)
    wedges = _wedges(phi, theta)
    return edges & wedges
    

def _aperture(vertices : Matrix, number_of_pixels : int, 
        aperture_pixels : int) -> Matrix:
    """
    Generate the BitMap representing the aperture described by the 
    vertices. 

    Parameters
    ----------
    vertices : Matrix 
        The vertices describing the approximately hexagonal aperture. 
    number_of_pixels : int
        The number of pixels that represent the compound aperture.
    aperture_pixels : int
        The number of pixels that represent the single aperture. 

    Returns
    -------
    aperture : Matrix 
        The Bit-Map that represents the aperture. 
    """
    x, y, phi = _vertices(vertices)
    rho, theta = _coordinates(number_of_pixels, vertices, 
        aperture_pixels, phi[0])
    segments = _segments(x, y, phi, theta, rho)
    return segments.sum(axis=0)


def _load(self : Layer):
    """
    """


vertices = np.stack(tree_util.tree_map(
    lambda leaf : leaf[1], 
    JWST_PRIMARY_SEGMENTS,
    is_leaf = lambda leaf : isinstance(leaf[0], str)))

aperture = vmap(_aperture, in_axes=(0, None, None))(vertices, 1008, 200)
pyplot.imshow(aperture.sum(axis=0))
pyplot.show()
