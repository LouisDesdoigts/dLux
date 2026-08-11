"""Evaluate hard and softened geometric transmission functions."""

import jax.numpy as np
from jax import lax, vmap, Array
import dLux.utils as dlu

__all__ = [
    "combine",
    "circle",
    "square",
    "rectangle",
    "reg_polygon",
    "convex_polygon",
    "spider",
    "soft_circle",
    "soft_square",
    "soft_rectangle",
    "soft_reg_polygon",
    "soft_convex_polygon",
    "soft_spider",
    "validate_convex",
]


def combine(arrays: Array, oversample: int = 1, use_sum: bool = False) -> Array:
    """Combine arrays multiplicatively and downsample the result.

    Parameters
    ----------
    arrays : Array
        The arrays to be combined. Should have shape (n_arrays, npix, npix).
    oversample : int = 1
        The amount to downsample the output by.
    use_sum : bool = False
        Whether to sum the arrays instead of multiplying them.

    Returns
    -------
    array : Array
        The combined array, optionally downsampled by `oversample`.
    """
    method = np.sum if use_sum else np.prod
    array = np.array(arrays)
    if oversample == 1:
        return method(array, 0)
    return dlu.downsample(method(array, 0), oversample)


def shift_and_scale(array: Array) -> Array:
    """Shifts and scales the array to be between 0 and 1.

    Parameters
    ----------
    array : Array
        The array to be shifted and scaled.

    Returns
    -------
    array : Array
        The shifted and scaled array.
    """
    return dlu.math.nandiv(array - array.min(), array.max() - array.min(), fill=0)


def soften(distances: Array, clip_dist: float, invert: bool = False) -> Array:
    """Softens the edges of a distances array by clipping the distances to a maximum
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
    distances = np.clip(distances, -clip_dist, clip_dist)
    cond = distances.max() == distances.min()
    constant_support = (distances > 0).astype(distances.dtype)
    return np.where(cond, constant_support, shift_and_scale(distances))


#####################
def circle(coords: Array, diameter: float, invert: bool = False) -> Array:
    """Calculates a hard-edged circle. This function is not differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the circle on.
    diameter : float
        The diameter of the circle.
    invert : bool = False
        Whether to invert the circle.

    Returns
    -------
    circle : Array
        The circle.
    """
    if invert:
        return (circ_distance(coords, diameter / 2) > 0).astype(float)
    return (circ_distance(coords, diameter / 2) < 0).astype(float)


def square(coords: Array, width: float, invert: bool = False) -> Array:
    """Calculates a hard-edged square. This function is not differentiable.

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
        The square.
    """
    if invert:
        return (square_distance(coords, width) > 0).astype(float)
    return (square_distance(coords, width) < 0).astype(float)


def rectangle(
    coords: Array, width: float, height: float, invert: bool = False
) -> Array:
    """Calculates a hard-edged rectangle. This function is not differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the rectangle on.
    width : float
        The width of the rectangle.
    height : float
        The height of the rectangle.
    invert : bool = False
        Whether to invert the rectangle.

    Returns
    -------
    rectangle : Array
        The rectangle.
    """
    if invert:
        return (rectangle_distance(coords, width, height) > 0).astype(float)
    return (rectangle_distance(coords, width, height) < 0).astype(float)


def reg_polygon(
    coords: Array, diameter: float, nsides: int, invert: bool = False
) -> Array:
    """Calculates a hard-edged regular polygon. This function is not differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the polygon on.
    diameter : float
        The diameter of the polygon's circumscribed circle.
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
        return (reg_polygon_distance(coords, nsides, diameter / 2) > 0).astype(float)
    return (reg_polygon_distance(coords, nsides, diameter / 2) < 0).astype(float)


def spider(coords: Array, width: float, angles: Array) -> Array:
    """Calculates a hard-edged spider. This function is not differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the spider on.
    width : float
        The width of the spider.
    angles : Array
        The angles of the spider in degrees.

    Returns
    -------
    spider : Array
        The spider.
    """
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    calc_fn = vmap(lambda angle: spider_distance(coords, width, angle) < 0)
    spiders = ~lax.reduce(calc_fn(angles), np.array(False), lax.bitwise_or, (0,))
    return spiders.astype(float)


# TODO: This eventually
# def irreg_polygon(
# npix, diam, vertices, oversample=1, invert=False, shift=np.zeros(2)):
#     """Calculates a downsampled irregular polygon to gain soft edges"""
#     pass


################
### Softened ###
################
def soft_circle(
    coords: Array, diameter: float, clip_dist: float = 0.1, invert: bool = False
) -> Array:
    """Calculates a soft-edged circle differentiably. The 'clip_dist' parameter defines
    the distance from the edge to 'soften' up to. A large clip_dist will result in a
    circle with a very soft edge, while a small clip_dist will result in a circle with
    a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the circle on.
    diameter : float
        The diameter of the circle.
    clip_dist : float = 0.1
        The distance from the edge to 'soften' up to.
    invert : bool = False
        Whether to invert the circle.

    Returns
    -------
    circle : Array
        The softened circle.
    """
    distances = -circ_distance(coords, diameter / 2)
    return soften(distances, clip_dist, invert)


def soft_square(
    coords: Array, width: float, clip_dist: float = 0.1, invert: bool = False
) -> Array:
    """Calculates a soft-edged square differentiably. The 'clip_dist' parameter defines
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
    """Calculate a differentiable soft-edged rectangle.

    The 'clip_dist' parameter defines the distance from the edge to 'soften' up to. A
    large clip_dist will result in a rectangle with a very soft edge, while a small
    clip_dist will result in a rectangle with a very hard edge.

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
    invert : bool = False
        Whether to invert the rectangle.

    Returns
    -------
    rectangle : Array
        The softened rectangle.
    """
    distances = -rectangle_distance(coords, width, height)
    return soften(distances, clip_dist, invert)


def soft_reg_polygon(
    coords: Array,
    diameter: float,
    nsides: int,
    clip_dist: float = 0.1,
    invert: bool = False,
) -> Array:
    """Calculates a soft-edged regular polygon differentiably. The 'clip_dist' parameter
    defines the distance from the edge to 'soften' up to. A large clip_dist will result
    in a polygon with a very soft edge, while a small clip_dist will result in a polygon
    with a very hard edge.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the polygon on.
    diameter : float
        The diameter of the polygon's circumscribed circle.
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
    distances = -reg_polygon_distance(coords, nsides, diameter / 2)
    return soften(distances, clip_dist, invert)


def soft_spider(
    coords: Array,
    width: float,
    angles: Array,
    clip_dist: float = 0.1,
    invert: bool = False,
) -> Array:
    """Calculates a soft-edged spider differentiably. The 'clip_dist' parameter defines
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
    """Calculates the distance from the edge of a circle.

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
    """Calculates the distance from the edge of a square.

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
    """Calculates the distance from the edge of a rectangle.

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
    """Calculates the distance from the edge of a spider.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the distance from the edge of the spider.
    width : float
        The width of the spider.
    angle : float
        The angle of the spider in degrees.

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
    """Calculate the signed distance from a line.

    This should also support a future irregular-polygon implementation.

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
    """Calculate gradients and points on the edges of a regular polygon.

    This may be extended to irregular polygons by accepting explicit vertices.

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
    """Calculates the distance from the edge of a regular polygon.

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


def convex_polygon_distance(coords: Array, vertices: Array) -> Array:
    """Return the maximum signed perpendicular distance to polygon edges.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(..., 2, ny, nx)``.
    vertices : Array
        Sequential polygon vertices with shape ``(..., nvertices, 2)``.

    Returns
    -------
    distances : Array
        Maximum signed edge distance, negative inside and positive outside. This
        half-plane measure is not the exact Euclidean distance outside a vertex.
    """
    vertices = np.asarray(vertices)

    # Generate oriented edges and determine the polygon winding
    ends = np.roll(vertices, -1, axis=-2)
    edges = ends - vertices
    area = np.sum(vertices[..., 0] * ends[..., 1], -1)
    area -= np.sum(vertices[..., 1] * ends[..., 0], -1)
    winding = np.where(area >= 0, 1.0, -1.0)

    # Calculate the signed distance from every point to every oriented edge
    x, y = coords[..., 0, :, :], coords[..., 1, :, :]
    x0, y0 = vertices[..., 0], vertices[..., 1]
    dx, dy = edges[..., 0], edges[..., 1]
    cross = dx[..., :, None, None] * (y[..., None, :, :] - y0[..., :, None, None])
    cross -= dy[..., :, None, None] * (x[..., None, :, :] - x0[..., :, None, None])
    lengths = np.linalg.norm(edges, axis=-1)
    distances = -winding[..., None, None, None] * cross
    distances /= lengths[..., :, None, None]

    # A convex polygon is bounded by its most restrictive edge
    return distances.max(-3)


def convex_polygon(coords: Array, vertices: Array, invert: bool = False) -> Array:
    """Evaluate a hard convex polygon from sequential vertices.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(..., 2, ny, nx)``.
    vertices : Array
        Sequential polygon vertices with shape ``(..., nvertices, 2)``.
    invert : bool
        Whether to return the complement of the polygon transmission.

    Returns
    -------
    transmission : Array
        Binary polygon transmission with the boundary included.
    """
    transmission = convex_polygon_distance(coords, vertices) <= 0
    transmission = np.logical_not(transmission) if invert else transmission
    return transmission.astype(float)


def soft_convex_polygon(
    coords: Array, vertices: Array, clip_dist: float = 0.1, invert: bool = False
) -> Array:
    """Evaluate a softened convex polygon from sequential vertices.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(..., 2, ny, nx)``.
    vertices : Array
        Sequential polygon vertices with shape ``(..., nvertices, 2)``.
    clip_dist : float
        Physical distance over which to soften the polygon boundary.
    invert : bool
        Whether to return the complement of the polygon transmission.

    Returns
    -------
    transmission : Array
        Differentiably softened polygon transmission.
    """
    distances = -convex_polygon_distance(coords, vertices)
    return soften(distances, clip_dist, invert)


def validate_convex(vertices: Array) -> None:
    """Validate one or more sequential convex-polygon vertex arrays.

    Parameters
    ----------
    vertices : Array
        Vertices with shape ``(..., nvertices, 2)`` in sequential boundary order.
        Clockwise and anticlockwise winding are accepted.

    Raises
    ------
    ValueError
        If the vertices have an invalid shape, repeat, enclose zero area, or do not
        describe a convex polygon in boundary order.
    """
    vertices = np.asarray(vertices)
    if vertices.ndim < 2 or vertices.shape[-1] != 2:
        raise ValueError("vertices must have shape (..., nvertices, 2).")
    if vertices.shape[-2] < 3:
        raise ValueError("vertices must contain at least three points.")

    # Calculate vertex separations and the signed polygon area
    ends = np.roll(vertices, -1, axis=-2)
    offsets = vertices[..., :, None, :] - vertices[..., None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    distances += np.eye(vertices.shape[-2])
    area = np.sum(vertices[..., 0] * ends[..., 1], -1)
    area -= np.sum(vertices[..., 1] * ends[..., 0], -1)

    # Check that every consecutive edge turns in a consistent direction
    edges = ends - vertices
    following = np.roll(edges, -1, axis=-2)
    cross = edges[..., 0] * following[..., 1]
    cross -= edges[..., 1] * following[..., 0]
    if np.any(distances == 0) or np.any(area == 0):
        raise ValueError("vertices must be distinct and define a non-zero area.")
    if np.any(np.any(cross > 0, axis=-1) & np.any(cross < 0, axis=-1)):
        raise ValueError("vertices must define a convex polygon in boundary order.")
