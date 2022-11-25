import matplotlib.pyplot as plt
import dLux as dl

coordinates = dl.utils.get_pixel_coordinates(24, 2. / 24.)

# Uniform Spider Testing
even_soft_unif_spider = UniformSpider(0., 0., 4., .1, 0., softening=True)
odd_soft_unif_spider = UniformSpider(0., 0., 3., .1, 0., softening=True)

even_soft_unif_spider._aperture(coordinates)
