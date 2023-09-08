import jax.numpy as np
from jax import lax, vmap
import dLux.utils as dlu

__all__ = [
    "gen_coords",
    "shift_and_scale",
    "circ_distance",
    "square_distance",
    "rectangle_distance",
    "reg_polygon_distance",
    "spider_distance",
    "circle",
    "annulus",
    "square",
    "rectangle",
    "reg_polygon",
    "spider",
    "soft_circle",
    "soft_annulus",
    "soft_square",
    "soft_rectangle",
    "soft_reg_polygon",
    "soft_spider",
]


################################
### General helper functions ###
################################
def gen_coords(npix, diameter, shifts):
    """Generates a set of pixel coordinates"""
    return dlu.nd_coords(
        (npix,) * 2, (diameter / npix,) * 2, offsets=tuple(shifts)
    )


def shift_and_scale(arr):
    """Shifts and scales the array to be between 0 and 1"""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


##########################
### Distance functions ###
##########################
def circ_distance(radius, coords):
    return dlu.cart2polar(coords)[0] - radius


def square_distance(width, coords):
    return np.max(np.abs(coords), axis=0) - width / 2


def rectangle_distance(width, height, coords):
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = np.abs(coords[1]) - height / 2
    return np.maximum(dist_from_vert, dist_from_horz)


def spider_distance(width, angle, coords):
    """Calcs an downsampled spider to gain soft edges"""
    coords = dlu.rotate_coords(coords, dlu.deg2rad(angle))
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = coords[1]
    return np.maximum(dist_from_vert, dist_from_horz)


def line_distance(coords, m, xy):
    """This should work for the irregular polygon case too"""
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


def reg_polygon_edges(n, radius, test=False):
    """Can probably be extended to irregular polygons too by taking in x, y"""
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


def reg_polygon_distance(nsides, radius, coords):
    m, xy = reg_polygon_edges(nsides, radius)
    return vmap(line_distance, (None, 0, 0))(coords, m, xy).max(0)


###############
### Aliased ###
###############
def alias(shape, oversample, invert=False):
    if invert:
        shape = np.logical_not(shape)
    return dlu.downsample(shape, oversample, mean=True)


def circle(
    npix, diameter, radius, oversample=1, invert=False, shift=np.zeros(2)
):
    """Calcs an downsampled circle to gain soft edges"""
    coords = gen_coords(npix * oversample, diameter, shift)
    circ = circ_distance(radius, coords) < 0
    return alias(circ, oversample, invert=invert)


def annulus(
    npix,
    diameter,
    inner_radius,
    outer_radius,
    oversample=1,
    invert=False,
    shift=np.zeros(2),
):
    """Calcs an downsampled annulus to gain soft edges"""
    coords = gen_coords(npix * oversample, diameter, shift)
    outer = circ_distance(outer_radius, coords) < 0
    inner = circ_distance(inner_radius, coords) < 0
    annulus = np.logical_and(outer, np.logical_not(inner))
    return alias(annulus, oversample, invert=invert)


def square(
    npix,
    diameter,
    width,
    rotation=0,
    oversample=1,
    invert=False,
    shift=np.zeros(2),
):
    """Calcs an downsampled square to gain soft edges"""
    coords = gen_coords(npix * oversample, diameter, shift)
    coords = dlu.rotate_coords(coords, dlu.deg2rad(rotation))
    square = square_distance(width, coords) < 0
    return alias(square, oversample, invert=invert)


def rectangle(
    npix,
    diameter,
    width,
    height,
    rotation=0,
    oversample=1,
    invert=False,
    shift=np.zeros(2),
):
    """Calcs an downsampled rectangle to gain soft edges"""
    coords = gen_coords(npix * oversample, diameter, shift)
    coords = dlu.rotate_coords(coords, dlu.deg2rad(rotation))
    rectangle = rectangle_distance(width, height, coords) < 0
    return alias(rectangle, oversample, invert=invert)


def reg_polygon(
    npix,
    diameter,
    radius,
    nsides,
    rotation=0.0,
    oversample=1,
    invert=False,
    shift=np.zeros(2),
):
    """Calcs an downsampled regular polygon to gain soft edges"""
    coords = dlu.gen_coords(npix * oversample, diameter, shift)
    coords = dlu.rotate_coords(coords, dlu.deg2rad(rotation))
    poly = reg_polygon_distance(nsides, radius, coords) < 0
    return alias(poly, oversample, invert=invert)


def spider(npix, diameter, width, angles, oversample=1, shift=np.zeros(2)):
    """Calcs an downsampled spider to gain soft edges"""
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    coords = gen_coords(npix * oversample, diameter, shift)
    spider_fn = vmap(lambda angle: spider_distance(width, angle, coords) < 0)
    spiders = lax.reduce(
        spider_fn(angles), np.array(False), lax.bitwise_or, (0,)
    )
    return alias(spiders, oversample)


# def irreg_polygon(
# npix, diameter, vertices, oversample=1, invert=False, shift=np.zeros(2)):
#     """Calcs an downsampled irregular polygon to gain soft edges"""
#     pass


################
### Softened ###
################
def soften(distances, clip_distance, invert=False):
    if invert:
        distances *= -1
    return shift_and_scale(np.clip(distances, -clip_distance, clip_distance))


def soft_circle(radius, coords, clip_distance=0.1, invert=False):
    """Dynamically calculates a soft circle differentiably"""
    distances = -circ_distance(radius, coords)
    return soften(distances, clip_distance, invert)


def soft_annulus(
    inner_radius, outer_radius, coords, clip_distance=0.1, invert=False
):
    """Dynamically calculates a soft annulus differentiably"""
    outer_distances = -circ_distance(outer_radius, coords)
    inner_distances = circ_distance(inner_radius, coords)
    distances = np.minimum(outer_distances, inner_distances)
    return soften(distances, clip_distance, invert)


def soft_square(width, coords, clip_distance=0.1, invert=False):
    """Dynamically calculates a soft square differentiably"""
    distances = -square_distance(width, coords)
    return soften(distances, clip_distance, invert)


def soft_rectangle(width, height, coords, clip_distance=0.1, invert=False):
    """Dynamically calculates a soft rectangle differentiably"""
    distances = -rectangle_distance(width, height, coords)
    return soften(distances, clip_distance, invert)


def soft_reg_polygon(radius, nsides, coords, clip_distance=0.1, invert=False):
    """Dynamically calculates a soft regular polygon differentiably"""
    distances = -reg_polygon_distance(nsides, radius, coords)
    return soften(distances, clip_distance, invert)


def soft_spider(width, angles, coords, clip_distance=0.1):
    """Dynamically calculates a soft regular polygon differentiably"""
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    spider_fn = vmap(lambda angle: spider_distance(width, angle, coords))
    spiders = -spider_fn(angles).min(axis=0)
    return soften(spiders, clip_distance)


# def soft_irreg_polygon(radius, coords, clip_distance=0.1, invert=False):
#     """Dynamically calculates a soft irregular polygon differentiably"""
#     pass
