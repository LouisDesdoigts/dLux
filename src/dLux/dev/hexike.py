from dLux.statistics import hexike_basis as dlux_hexike
from poppy.zernike import hexike_basis as poppy_hexike
from matplotlib import pyplot
from jax.config import config
from jax.numpy import allclose
config.update("jax_enable_x64", True)

number_of_pixels = 256
number_of_hexikes = 5

dlux_hexikes = dlux_hexike(number_of_hexikes, number_of_pixels)
poppy_hexikes = poppy_hexike(number_of_hexikes, number_of_pixels, outside=0.)

resids = dlux_hexikes - poppy_hexikes

pyplot.imshow(resids[8])
pyplot.colorbar()
pyplot.show()
