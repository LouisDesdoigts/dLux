from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot

config.update("jax_enable_x64", True)

vertices = JWST_PRIMARY_SEGMENTS[0][1]

angles = np.arctan2(vertices[1], vertices[0])
angles += (angles < 0.) * (2 * np.pi + angles)

radial = np.hypot(vertices[0], vertices[1])

# NOTE: Simple hexagonal case. 
angles = np.linspace(0, 2 * np.pi, 7, endpoint=True).reshape(7, 1, 1)
vertices = np.array([np.cos(angles), np.sin(angles)]).reshape(7, 2, 1)

a = (vertices[1:, 0] - vertices[:-1, 0]).reshape(6, 1, 1)
b = (vertices[1:, 1] - vertices[:-1, 1]).reshape(6, 1, 1)
c = ((vertices[1:, 0] - vertices[:-1, 0]) * vertices[:-1, 1] -\
    (vertices[1:, 1] - vertices[:-1, 1]) * vertices[:-1, 0])\
    .reshape(6, 1, 1)

positions = get_radial_positions(256, 0, 0)
rho = positions[0] * 2 / 256
theta = (positions[1] + np.pi)[:, ::-1]
rho = np.tile(rho, (6, 1, 1))
theta = np.tile(theta, (6, 1, 1))

linear = (c / (a * np.sin(theta) + b * np.cos(theta)))
angular = ((angles[:-1] < theta) & (theta < angles[1:])) 
test = (rho < linear) & (angular)

print(angular.shape)
 
pyplot.imshow(theta[0])
pyplot.colorbar()
pyplot.show()
