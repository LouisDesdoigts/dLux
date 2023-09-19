import jax.numpy as np
from jax import lax, vmap, Array
import dLux.utils as dlu

__all__ = [
    "combine",
    "soften",
    "circle",
    "square",
    "rectangle",
    "reg_polygon",
    "spider",
    "soft_circle",
    "soft_square",
    "soft_rectangle",
    "soft_reg_polygon",
    "soft_spider",
]


def combine(arrays: Array, oversample: int = 1) -> Array:
    """
    Combines multiple arrays by multiplying them together, and downsampling the output.

    Parameters
    ----------
    arrays : Array
        The arrays to be combined. Should have shape (n_arrays, npix, npix).
    oversample : int = 1
        The amount to downsample the output by.
    """
    array = np.array(arrays)
    if oversample == 1:
        return array.prod(0)
    return dlu.downsample(array.prod(0), oversample)


def shift_and_scale(array: Array) -> Array:
    """
    Shifts and scales the array to be between 0 and 1

    Parameters
    ----------
    array : Array
        The array to be shifted and scaled.

    Returns
    -------
    array : Array
        The shifted and scaled array.
    """
    return dlu.math.nandiv(
        array - array.min(), array.max() - array.min(), fill=0
    )


def soften(distances: Array, clip_dist: float, invert: bool = False) -> Array:
    """
    Softens the edges of a distances array by clipping the distances to a maximum
    value, and then shifting and scaling the array to be between 0 and 1.

    Parameters
    ----------
    distances : Array
        The array of distances to be softened.
    clip_dist : float
        The maximum distance to clip the distances to.
    invert : bool = False
        Whether to invert the distances before softening.

    Returns
    -------
    distances : Array
        The softened distances array.
    """
    # TODO: Possibly clip from -clip_dist:0 to ensure zernikes have full pupil support
    if invert:
        distances *= -1
    return shift_and_scale(np.clip(distances, -clip_dist, clip_dist))


#####################
def circle(coords: Array, diam: float, invert: bool = False) -> Array:
    """
    Calculates a soft-edged circle via downsampling. This function is not
    differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the circle on.
    diam : float
        The diameter of the circle.
    invert : bool = False
        Whether to invert the circle.

    Returns
    -------
    circle : Array
        The circle."""
    if invert:
        return circ_distance(coords, diam) > 0
    return circ_distance(coords, diam) < 0


def square(coords: Array, width: float, invert: bool = False) -> Array:
    """
    Calculates a soft-edged square via downsampling. This function is not
    differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the square on.
    width : float
        The width of the square.
    invert : bool = False
        Whether to invert the square.

    Returns
    -------
    square : Array
        The square."""
    if invert:
        return square_distance(coords, width) > 0
    return square_distance(coords, width) < 0


def rectangle(
    coords: Array, width: float, height: float, invert: bool = False
) -> Array:
    """
    Calculates a soft-edged rectangle via downsampling. This function is not
    differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the rectangle on.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.

    Returns
    -------
    rectangle : Array
        The rectangle.
    """
    if invert:
        return rectangle_distance(coords, width, height) > 0
    return rectangle_distance(coords, width, height) < 0


def reg_polygon(
    coords: Array, rmax: float, nsides: int, invert: bool = False
) -> Array:
    """
    Calculates a soft-edged regular polygon via downsampling. This function is not
    differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the polygon on.
    rmax : float
        The radius of the polygon.
    nsides : int
        The number of sides of the polygon.
    invert : bool = False
        Whether to invert the polygon.

    Returns
    -------
    polygon : Array
        The polygon.
    """
    if invert:
        return reg_polygon_distance(coords, nsides, rmax) > 0
    return reg_polygon_distance(coords, nsides, rmax) < 0


def spider(coords: Array, width: float, angles: Array) -> Array:
    """
    Calculates a soft-edged spider via downsampling. This function is not
    differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the spider on.
    width : float
        The width of the spider.
    angles : Array
        The angles of the spider.

    Returns
    -------
    spider : Array
        The spider.
    """
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    calc_fn = vmap(lambda angle: spider_distance(coords, width, angle) < 0)
    return ~lax.reduce(calc_fn(angles), np.array(False), lax.bitwise_or, (0,))


# TODO: This eventually
# def irreg_polygon(
# npix, diam, vertices, oversample=1, invert=False, shift=np.zeros(2)):
#     """Calcs an downsampled irregular polygon to gain soft edges"""
#     pass


################
### Softened ###
################
def soft_circle(
    coords: Array, radius: float, clip_dist: float = 0.1, invert: bool = False
) -> Array:
    """
    Calculates a soft-edged circle differentiably. The 'clip_dist' parameter defines
    the distance from the edge to 'soften' up to. A large clip_dist will result in a
    circle with a very soft edge, while a small clip_dist will result in a circle with
    a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the circle on.
    radius : float
        The radius of the circle.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.
        circle.
    invert : bool = False
        Whether to invert the circle.

    Returns
    -------
    circle : Array
        The softened circle.
    """
    distances = -circ_distance(coords, radius)
    return soften(distances, clip_dist, invert)


def soft_square(
    coords: Array, width: float, clip_dist: float = 0.1, invert: bool = False
) -> Array:
    """
    Calculates a soft-edged square differentiably. The 'clip_dist' parameter defines
    the distance from the edge to 'soften' up to. A large clip_dist will result in a
    square with a very soft edge, while a small clip_dist will result in a square with
    a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the square on.
    width : float
        The width of the square.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.
        square.
    invert : bool = False
        Whether to invert the square.

    Returns
    -------
    square : Array
        The softened square.
    """
    distances = -square_distance(coords, width)
    return soften(distances, clip_dist, invert)


def soft_rectangle(
    coords: Array,
    width: float,
    height: float,
    clip_dist: float = 0.1,
    invert: bool = False,
) -> Array:
    """
    Calculates a soft-edged rectangle differentiably. The 'clip_dist' parameter defines
    the distance from the edge to 'soften' up to. A large clip_dist will result in a
    rectangle with a very soft edge, while a small clip_dist will result in a rectangle
    with a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the rectangle on.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.

    Returns
    -------
    rectangle : Array
        The softened rectangle.
    """
    distances = -rectangle_distance(coords, width, height)
    return soften(distances, clip_dist, invert)


def soft_reg_polygon(
    coords: Array,
    radius: float,
    nsides: int,
    clip_dist: float = 0.1,
    invert: bool = False,
) -> Array:
    """
    Calculates a soft-edged regular polygon differentiably. The 'clip_dist' parameter
    defines the distance from the edge to 'soften' up to. A large clip_dist will result
    in a polygon with a very soft edge, while a small clip_dist will result in a polygon
    with a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the polygon on.
    radius : float
        The radius of the polygon.
    nsides : int
        The number of sides of the polygon.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.
    invert : bool = False
        Whether to invert the polygon.

    Returns
    -------
    polygon : Array
        The softened polygon.
    """
    distances = -reg_polygon_distance(coords, nsides, radius)
    return soften(distances, clip_dist, invert)


def soft_spider(
    coords: Array,
    width: float,
    angles: Array,
    clip_dist: float = 0.1,
    invert: bool = False,
) -> Array:
    """
    Calculates a soft-edged spider differentiably. The 'clip_dist' parameter defines
    the distance from the edge to 'soften' up to. A large clip_dist will result in a
    spider with a very soft edge, while a small clip_dist will result in a spider with
    a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the spider on.
    width : float
        The width of the spider.
    angles : Array
        The angles of the spider.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.
    invert : bool = False
        Whether to invert the spider.

    Returns
    -------
    spider : Array
        The softened spider.
    """
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    spider_fn = vmap(lambda angle: spider_distance(coords, width, angle))
    spiders = -spider_fn(angles).min(axis=0)
    return soften(spiders, clip_dist, invert)


# def soft_irreg_polygon(radius, coords, clip_dist=0.1, invert=False):
#     """Dynamically calculates a soft irregular polygon differentiably"""
#     pass


##########################
### Distance functions ###
##########################
def circ_distance(coords: Array, radius: float) -> Array:
    """
    Calculates the distance from the edge of a circle.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    distance : Array
        The distance from the edge of the circle.
    """
    return dlu.cart2polar(coords)[0] - radius


def square_distance(coords: Array, width: float) -> Array:
    """
    Calculates the distance from the edge of a square.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the square.
    width : float
        The width of the square.

    Returns
    -------
    distance : Array
        The distance from the edge of the square.
    """
    return np.max(np.abs(coords), axis=0) - width / 2


def rectangle_distance(coords: Array, width: float, height: float) -> Array:
    """
    Calculates the distance from the edge of a rectangle.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the rectangle.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.

    Returns
    -------
    distance : Array
        The distance from the edge of the rectangle.
    """
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = np.abs(coords[1]) - height / 2
    return np.maximum(dist_from_vert, dist_from_horz)


def spider_distance(coords: Array, width: float, angle: float) -> Array:
    """
    Calculates the distance from the edge of a spider.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the spider.
    width : float
        The width of the spider.
    angle : float
        The angle of the spider.

    Returns
    -------
    distance : Array
        The distance from the edge of the spider.
    """
    coords = dlu.rotate_coords(coords, dlu.deg2rad(angle))
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = coords[1]
    return np.maximum(dist_from_vert, dist_from_horz)


def line_distance(coords: Array, m: float, xy: Array) -> Array:
    """
    Calculates the distance from the edge of a line. This should work for the irregular
    polygon case too, when it is implemented.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the line.
    m : float
        The gradient of the line.
    xy : Array
        The point on the line.

    Returns
    -------
    distance : Array
        The distance from the edge of the line.
    """
    # Extract x and y coordinates from the input array
    x, y = coords

    # First determine the case: horizontal, vertical or regular
    # NOTE: We may be able to improve here my using np.where
    case = (m == 0) * 0 + (m == np.inf) * 1 + (m != 0) * (m != np.inf) * 2

    # Now we determine the sign on the y-intersection so we know when to flip
    # the sign of the distance
    horiz_fn = lambda: np.sign(xy[1])
    vert_fn = lambda: np.sign(xy[0])
    base_fn = lambda: np.sign(xy[1] - m * xy[0])
    sgn = lax.switch(case, [horiz_fn, vert_fn, base_fn])

    # Now we calculate the distance from the point to the line
    horiz_fn = lambda: xy[1] - y
    vert_fn = lambda: xy[0] - x
    base_fn = lambda: (m * x - y + xy[1] - m * xy[0]) / np.sqrt(m**2 + 1)
    dist = lax.switch(case, [horiz_fn, vert_fn, base_fn])

    # Finally, return the distances with the correct sign
    return -sgn * dist


def reg_polygon_edges(n: int, radius: float) -> Array:
    """
    Calculates the gradients and points on the edges of a regular polygon. Can probably
    be extended to irregular polygons too by taking in x, y.

    Parameters
    ----------
    n : int
        The number of sides of the polygon.
    radius : float
        The radius of the polygon.

    Returns
    -------
    m : Array
        The gradients of the edges of the polygon.
    """
    # Calculate the x and y coordinates of the vertices of the polygon
    x = radius * np.cos(np.linspace(-np.pi, np.pi, n, endpoint=False))
    y = radius * np.sin(np.linspace(-np.pi, np.pi, n, endpoint=False))

    # Generate the indexes of the vertices and the next vertex
    ivals = np.arange(n)
    jvals = np.roll(ivals, -1)

    # Calculate the gradient and point on each edge of the polygon
    calc_m = vmap(lambda i, j: dlu.nandiv(y[j] - y[i], x[j] - x[i]), (0, 0))
    calc_xy = vmap(lambda i, j: np.array([x[i], y[i]]), (0, 0))
    return calc_m(ivals, jvals), calc_xy(ivals, jvals)


def reg_polygon_distance(coords: Array, nsides: int, radius: float) -> Array:
    """
    Calculates the distance from the edge of a regular polygon.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the polygon.
    nsides : int
        The number of sides of the polygon.
    radius : float
        The radius of the polygon.

    Returns
    -------
    distance : Array
        The distance from the edge of the polygon.
    """
    m, xy = reg_polygon_edges(nsides, radius)
    return vmap(line_distance, (None, 0, 0))(coords, m, xy).max(0)
