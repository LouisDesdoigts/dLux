import jax.numpy as np
import dLux as dl
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

from apertures import *

aperture: ApertureLayer = SquareAperture

pixel_area = aperture.sum()
shape = zernikes.shape
width = shape[-1]
basis = np.zeros(shape).at[0].set(aperture)

for j in np.arange(1, self.nterms):
    intermediate = zernikes[j] * aperture
    coefficient = np.zeros((self.nterms, 1, 1), dtype=float)
    mask = (np.arange(1, self.nterms) > j + 1).reshape((-1, 1, 1))

    coefficient = -1 / pixel_area * \
        (zernikes[j] * basis[1:] * aperture * mask)\
        .sum(axis = (1, 2))\
        .reshape(-1, 1, 1) 

    print(coefficient)

    intermediate += (coefficient * basis[1:] * mask).sum(axis = 0)
    
    basis = basis\
        .at[j]\
        .set(intermediate / \
            np.sqrt((intermediate ** 2).sum() / pixel_area))
