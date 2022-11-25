import dLux as dl
import jax.numpy as np

coordinates = dl.utils.get_pixel_coordinates(24, 2. / 24.)

# Uniform Spider Testing
even_soft_unif_spider = dl.UniformSpider(0., 0., 4., .1, 0., softening=True)
odd_soft_unif_spider = dl.UniformSpider(0., 0., 3., .1, 0., softening=True)

fig, axes = plt.subplots(2, 2, figsize=(2*4, 1*3))
axes[0][0].imshow(even_soft_unif_spider._aperture(coordinates))
plt.show()
