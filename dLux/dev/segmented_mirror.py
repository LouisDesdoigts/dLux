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
_x = (vertices[:, 0] - np.mean(vertices[:, 0]))
_y = (vertices[:, 1] - np.mean(vertices[:, 1]))
_angles = (np.arctan2(_y, _x) + np.pi)

pixel_scale = (np.max(_x) - np.min(_x)) / number_of_pixels
x_offset = np.mean(vertices[:, 0]) / pixel_scale
y_offset = np.mean(vertices[:, 1]) / pixel_scale

sorted_indices = np.argsort(_angles)

x = np.zeros((number_of_vertices + 1,))
y = np.zeros((number_of_vertices + 1,))
angles = np.zeros((number_of_vertices + 1,))

x = x\
    .at[:number_of_vertices]\
    .set(_x.at[sorted_indices].get())\
    .reshape(number_of_vertices + 1, 1, 1)
x = x\
    .at[number_of_vertices]\
    .set(x[0])

y = y\
    .at[:number_of_vertices]\
    .set(_y.at[sorted_indices].get())\
    .reshape(number_of_vertices + 1, 1, 1)
y = y\
    .at[number_of_vertices]\
    .set(y[0])

angles = angles\
    .at[:number_of_vertices]\
    .set(_angles.at[sorted_indices].get())\
    .reshape(number_of_vertices + 1, 1, 1)  
angles = angles\
    .at[number_of_vertices]\
    .set(angles[0] + 2 * np.pi)

a = (x[1:] - x[:-1])
b = (y[1:] - y[:-1])
c = (a * y[:-1] - b * x[:-1])

positions = get_radial_positions(2 * number_of_pixels, 0., 0.) 
#     x_offset, y_offset)
rho = (positions[0] * 1 / 256)
theta = (positions[1] + np.pi)[:, ::-1]
rho = np.tile(rho, (number_of_vertices, 1, 1))
theta = np.tile(theta, (number_of_vertices, 1, 1))

linear = c / (a * np.sin(theta) - b * np.cos(theta))
angular = ((angles[:-1] < theta) & (theta < angles[1:]))[:, ::-1, ::-1] 
test = (rho < linear) & angular

pyplot.imshow(test.sum(axis = 0))
pyplot.colorbar()
pyplot.show()
