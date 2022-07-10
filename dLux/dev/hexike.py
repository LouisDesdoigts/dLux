from dLux.statistics import hexike_basis as dlux_hexike
from matplotlib import pyplot
from jax.config import config
import jax.numpy as np

config.update("jax_enable_x64", True)

number_of_pixels = 256
number_of_hexikes = 5

dlux_hexikes = dlux_hexike(
    number_of_hexikes = number_of_hexikes, 
    number_of_pixels = number_of_pixels,
    x_pixel_offset = 50, 
    y_pixel_offset = 50, 
    maximum_radius = 0.5)

dlux_hexikes += dlux_hexike(
    number_of_hexikes = number_of_hexikes, 
    number_of_pixels = number_of_pixels,
    x_pixel_offset = -50, 
    y_pixel_offset = 50, 
    maximum_radius = 0.5)

dlux_hexikes += dlux_hexike(
    number_of_hexikes = number_of_hexikes, 
    number_of_pixels = number_of_pixels,
    x_pixel_offset = 50, 
    y_pixel_offset = -50, 
    maximum_radius = 0.5)

dlux_hexikes += dlux_hexike(
    number_of_hexikes = number_of_hexikes, 
    number_of_pixels = number_of_pixels,
    x_pixel_offset = -50, 
    y_pixel_offset = -50, 
    maximum_radius = 0.5)

pyplot.imshow(dlux_hexikes[5])
pyplot.colorbar()
pyplot.show()
