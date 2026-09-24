"""Retained geometry functions, backed by the parametric shape kernels.

Hard circle, square, rectangle and regular-polygon functions exclude their boundary
for either inversion choice. Hard ``spider`` returns one away from its arms; the
historical soft version returns one on them. Shape classes consistently return one
on their support and invert by taking its complement.
"""

import jax.numpy as np
from jax import Array, lax, vmap
import dLux.utils as dlu

from ..numerical.shapes import (
    circ_distance,
    square_distance,
    rectangle_distance,
    spider_distance,
    line_distance as line_distance,
    reg_polygon_edges as reg_polygon_edges,
    reg_polygon_distance,
    convex_polygon_distance,
    validate_convex,
)

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


def circle(coords: Array, radius: float, invert: bool = False) -> Array:
    """
    Calculates a hard-edged circle. This function is not differentiable.

    Parameters
    ----------
    coords : Array
        The coordinates to calculate the circle on.
    radius : float
        The radius of the circle.
    invert : bool = False
        Whether to invert the circle.

    Returns
    -------
    circle : Array
        The circle.
    """

    if invert:
        return (circ_distance(coords, radius) > 0).astype(float)
    return (circ_distance(coords, radius) < 0).astype(float)


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


def reg_polygon(coords: Array, rmax: float, nsides: int, invert: bool = False) -> Array:
    """
    Calculates a hard-edged regular polygon. This function is not differentiable.

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
        return (reg_polygon_distance(coords, nsides, rmax) > 0).astype(float)
    return (reg_polygon_distance(coords, nsides, rmax) < 0).astype(float)


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
    return dlu.math.nandiv(array - array.min(), array.max() - array.min(), fill=0)


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
    distances = np.clip(distances, -clip_dist, clip_dist)
    cond = distances.max() == distances.min()
    constant_support = (distances > 0).astype(distances.dtype)
    return np.where(cond, constant_support, shift_and_scale(distances))
