import jax.numpy as np
from jax import lax, vmap
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


def combine(arrays, oversample=1):
    array = np.array(arrays)
    if oversample == 1:
        return array.prod(0)
    return dlu.downsample(array.prod(0), oversample)


def shift_and_scale(array):
    """Shifts and scales the array to be between 0 and 1"""
    return (array - array.min()) / (array.max() - array.min())


def soften(distances, clip_dist, invert=False):
    # TODO: Possibly clip from -clip_dist:0 to ensure zernikes have full
    # pupil support
    if invert:
        distances *= -1
    return shift_and_scale(np.clip(distances, -clip_dist, clip_dist))


#####################
def circle(coords, diam, invert=False):
    """Calcs an downsampled circle to gain soft edges"""
    if invert:
        return circ_distance(coords, diam) > 0
    return circ_distance(coords, diam) < 0


def square(coords, width, invert=False):
    """Calcs an downsampled square to gain soft edges"""
    if invert:
        return square_distance(coords, width) > 0
    return square_distance(coords, width) < 0


def rectangle(coords, width, height, invert=False):
    """Calcs an downsampled rectangle to gain soft edges"""
    if invert:
        return rectangle_distance(coords, width, height) > 0
    return rectangle_distance(coords, width, height) < 0


def reg_polygon(coords, rmax, nsides, invert=False):
    """Calcs an downsampled regular polygon to gain soft edges"""
    if invert:
        return reg_polygon_distance(coords, nsides, rmax) > 0
    return reg_polygon_distance(coords, nsides, rmax) < 0


def spider(coords, width, angles):
    """Calcs an downsampled spider to gain soft edges"""
    angles = np.array(angles) if not isinstance(angles, np.ndarray) else angles
    calc_fn = vmap(lambda angle: spider_distance(coords, width, angle) < 0)
    return ~lax.reduce(calc_fn(angles), np.array(False), lax.bitwise_or, (0,))


# def irreg_polygon(
# npix, diam, vertices, oversample=1, invert=False, shift=np.zeros(2)):
#     """Calcs an downsampled irregular polygon to gain soft edges"""
#     pass


################
### Softened ###
################
def soft_circle(coords, radius, clip_dist=0.1, invert=False):
    """Dynamically calculates a soft circle differentiably"""
    distances = -circ_distance(coords, radius)
    return soften(distances, clip_dist, invert)


def soft_square(coords, width, clip_dist=0.1, invert=False):
    """Dynamically calculates a soft square differentiably"""
    distances = -square_distance(coords, width)
    return soften(distances, clip_dist, invert)


def soft_rectangle(coords, width, height, clip_dist=0.1, invert=False):
    """Dynamically calculates a soft rectangle differentiably"""
    distances = -rectangle_distance(coords, width, height)
    return soften(distances, clip_dist, invert)


def soft_reg_polygon(coords, radius, nsides, clip_dist=0.1, invert=False):
    """Dynamically calculates a soft regular polygon differentiably"""
    distances = -reg_polygon_distance(coords, nsides, radius)
    return soften(distances, clip_dist, invert)


def soft_spider(coords, width, angles, clip_dist=0.1, invert=False):
    """Dynamically calculates a soft regular polygon differentiably"""
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
def circ_distance(coords, radius):
    return dlu.cart2polar(coords)[0] - radius


def square_distance(coords, width):
    return np.max(np.abs(coords), axis=0) - width / 2


def rectangle_distance(coords, width, height):
    dist_from_vert = np.abs(coords[0]) - width / 2
    dist_from_horz = np.abs(coords[1]) - height / 2
    return np.maximum(dist_from_vert, dist_from_horz)


def spider_distance(coords, width, angle):
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


def reg_polygon_distance(coords, nsides, radius):
    m, xy = reg_polygon_edges(nsides, radius)
    return vmap(line_distance, (None, 0, 0))(coords, m, xy).max(0)
