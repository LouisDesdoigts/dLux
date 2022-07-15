from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot

config.update("jax_enable_x64", True)

vertices = JWST_PRIMARY_SEGMENTS[1][1]

number_of_pixels = 256
number_of_vertices = 6

# So now I need to plan how I want to do this so that it is good.
# I could do something like process vertices.
# 

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
    

def _vertices(vertices : Array) -> tuple:
    """
    Generates the vertices that are compatible with the rest of 
    the transformations from the raw data vertices.

    Parameters
    ----------
    vertices : Array, meters
        The vertices loaded from the WebbPSF module. 

    Returns
    -------
    vertices : Array
        The vertices in normalised positions and wrapped so that 
        they can be used in the generation of the compound aperture. 
    """
    _x = (vertices[:, 0] - np.mean(vertices[:, 0]))
    _y = (vertices[:, 1] - np.mean(vertices[:, 1]))

    _angles = np.arctan2(_y, _x)
    _angles += 2 * np.pi * (np.arctan2(_y, _x) < 0.)

    order = np.argsort(_angles)

    x = _wrap(_x, order)
    y = _wrap(_y, order)
    angles = _wrap(_angles, order).at[-1].add(2 * np.pi)

    return x, y, angles


def _offset()
    pixel_scale = (np.max(_x) - np.min(_x)) / number_of_pixels
    x_offset = np.mean(vertices[:, 0]) / pixel_scale
    y_offset = np.mean(vertices[:, 1]) / pixel_scale
def _coordinates()
def _affine()
def _aperture(vertices : tuple)



a = (x[1:] - x[:-1])
b = (y[1:] - y[:-1])
c = (a * y[:-1] - b * x[:-1])

positions = get_radial_positions(2 * number_of_pixels, #0, 0)
    -x_offset, -y_offset)

rho = positions[0] * pixel_scale

theta = positions[1] 
theta += 2 * np.pi * (positions[1] < 0.)
theta += 2 * np.pi * (theta < angles[0])

rho = np.tile(rho, (number_of_vertices, 1, 1))
theta = np.tile(theta, (number_of_vertices, 1, 1))

linear = c / (a * np.sin(theta) - b * np.cos(theta))
angular = ((angles[:-1] < theta) & (theta < angles[1:])) 
segments = (rho < linear) & angular

pyplot.imshow((segments).sum(axis = 0))
pyplot.colorbar()
pyplot.show()
