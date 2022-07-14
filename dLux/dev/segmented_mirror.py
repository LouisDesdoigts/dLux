from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot

config.update("jax_enable_x64", True)

vertices = JWST_PRIMARY_SEGMENTS[0][1]

angles = np.arctan2(vertices[1], vertices[0])
angles += (angles < 0.) * (2 * np.pi + angles)

radial = np.hypot(vertices[0], vertices[1])

# NOTE: Trying to get back to it working for the simple hexagonal 
# case. 
vertices = np.array([np.cos(np.arange(0, 2 * np.pi, np.pi / 3)),
    np.sin(np.arange(0, 2 * np.pi, np.pi / 3))]).reshape(6, 2)

i = np.arange(-1, vertices.shape[0])
a = (vertices[i + 1, 0] - vertices[i, 0]).reshape(7, 1, 1)
b = (vertices[i + 1, 1] - vertices[i, 1]).reshape(7, 1, 1)
c = ((vertices[i + 1, 0] - vertices[i, 0]) * vertices[i, 1] -\
    (vertices[i + 1, 1] - vertices[i, 1]) * vertices[i, 0])\
    .reshape(7, 1, 1)

positions = get_radial_positions(256, 0, 0) * 2 / 256
rho = positions[0]
theta = positions[1]
rho = np.tile(rho, (7, 1, 1))
theta = np.tile(theta, (7, 1, 1))

print(rho.shape)
print(theta.shape)
print(a.shape)

linear = (c / (a * np.sin(theta) + b * np.cos(theta)))

test = rho < linear
 
print(test.shape)

pyplot.imshow(linear[0])
pyplot.colorbar()
pyplot.show()
#aperture = less_than(rho, 
#    c / (a * np.sin(theta) + b * np.cos(theta)))


