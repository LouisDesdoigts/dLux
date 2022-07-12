from layers import *
from matplotlib import pyplot
from jax import numpy as np
from jax.config import config

config.update("jax_debug_nans", True)

hexagonal_basis = HexagonalBasis(0, 0, 1, 256, 9)

#vertices = hexagonal_basis._vertices()
#pyplot.plot(vertices[0], vertices[1])
#pyplot.show()

# NOTE: Happy with these working as is.
n, m = hexagonal_basis._noll_index(np.array(np.arange(1, 10)))

#radial = hexagonal_basis._radial_zernike(3, 3, np.zeros((256, 256)))
#print(radial)
#pyplot.imshow(radial)
#pyplot.colorbar()
#pyplot.show()

norm_coeff = (1 + np.sqrt(2) * (m != 0).astype(int)) * np.sqrt(n + 1)
print(norm_coeff)

zernikes = hexagonal_basis._zernike_basis()
for i in range(9):
    pyplot.subplot(3, 3, i + 1)
    pyplot.title(f"{n[i]}, {m[i]}")
    pyplot.imshow(zernikes[i])
pyplot.show()
