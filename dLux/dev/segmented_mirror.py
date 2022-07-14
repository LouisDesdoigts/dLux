from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot

config.update("jax_enable_x64", True)

vertices = JWST_PRIMARY_SEGMENTS[0][1]

# So I guess that I must sort the vertices to go in ascending angular 
# order that way the method I have devised will work. 

number_of_pixels = 256
number_of_vertices = 6

# These are wrong with the duplication. 
_x = vertices[:, 0]
_y = vertices[:, 1]
_angles = (np.arctan2(_y - np.mean(_y), _x - np.mean(_x)) + np.pi)

pixel_scale = (np.max(_x) - np.min(_x)) / number_of_pixels
x_offset = np.mean(_x) / pixel_scale
y_offset = np.mean(_y) / pixel_scale

sorted_indices = np.argsort(_angles)

x = np.zeros((number_of_vertices + 1,))
y = np.zeros((number_of_vertices + 1,))
angles = np.zeros((number_of_vertices + 1,))

x = x\
    .at[:number_of_vertices]\
    .set(_x.at[sorted_indices].get())\
    .at[-1]\
    .set(_x.min())\
    .reshape(number_of_vertices + 1, 1, 1)

y = y\
    .at[:number_of_vertices]\
    .set(_y.at[sorted_indices].get())\
    .at[-1]\
    .set(_y.min())\
    .reshape(number_of_vertices + 1, 1, 1)

angles = angles\
    .at[:number_of_vertices]\
    .set(_angles.at[sorted_indices].get())\
    .at[-1]\
    .set(_angles.min() + 2 * np.pi)\
    .reshape(number_of_vertices + 1, 1, 1)

# NOTE: Simple hexagonal case. 
#angles = np.linspace(0, 2 * np.pi, 7, endpoint=True).reshape(7, 1, 1)
#x = np.cos(angles).reshape(7, 1, 1)
#y = np.sin(angles).reshape(7, 1, 1)

a = (x[1:] - x[:-1])
b = (y[1:] - y[:-1])
c = (a * y[:-1] - b * x[:-1])

# All that needs to be done is that the positions be translated to
# the correct place

# TODO: Test the get_radial_positions for offset again.
positions = get_radial_positions(2 * number_of_pixels, 
    x_offset, y_offset)
rho = (positions[0] * 2 / 256)
theta = (positions[1] + np.pi)[:, ::-1]
rho = np.tile(rho, (6, 1, 1))
theta = np.tile(theta, (6, 1, 1))

linear = (-c / (a * np.sin(theta) + b * np.cos(theta)))
angular = ((angles[:-1] < theta) & (theta < angles[1:]))[::-1, :, :] 
test = (rho < linear) & angular

pyplot.imshow(test.sum(axis = 0))
pyplot.colorbar()
pyplot.show()
