from constants import *
from dLux.utils import get_radial_positions
from matplotlib import pyplot

config.update("jax_enable_x64", True)

vertices = JWST_PRIMARY_SEGMENTS[0][1]

angles = np.arctan2(vertices[1], vertices[0])
angles += (angles < 0.) * (2 * np.pi + angles)

radial = np.hypot(vertices[0], vertices[1])

# Wrap required
x = vertices[np.arange(-1, 6), 0].reshape(7, 1, 1)
y = vertices[np.arange(-1, 6), 1].reshape(7, 1, 1)

print(x)
print(y)

# NOTE: Simple hexagonal case. 
#angles = np.linspace(0, 2 * np.pi, 7, endpoint=True).reshape(7, 1, 1)
#x = np.cos(angles).reshape(7, 1, 1)
#y = np.sin(angles).reshape(7, 1, 1)

a = (x[1:] - x[:-1])
b = (y[1:] - y[:-1])
c = (a * y[:-1] - b * x[:-1])

# All that needs to be done is that the positions be translated to
# the correct place
x_offset = np.mean(x)
y_offset = np.mean(y)

print(vertices)
print(x_offset)
# TODO: Test the get_radial_positions for offset again.
positions = get_radial_positions(256, x_offset, y_offset)
rho = (positions[0] * 2 / 256)
theta = (positions[1] + np.pi)[:, ::-1]
rho = np.tile(rho, (6, 1, 1))
theta = np.tile(theta, (6, 1, 1))

linear = (-c / (a * np.sin(theta) + b * np.cos(theta)))
angular = ((angles[:-1] < theta) & (theta < angles[1:]))[::-1, :, :] 
test = (rho < linear) & angular

# pyplot.imshow(test.sum(axis = 0))
pyplot.imshow(rho[0])
pyplot.colorbar()
pyplot.show()

