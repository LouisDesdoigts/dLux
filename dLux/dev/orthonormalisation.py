import jax.numpy as np
import dLux as dl
import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

from apertures import *


def plot_basis(basis: float):
    length: int = basis.shape[0]
        
    fig: object = plt.figure(figsize=(length*4, 3))
    axes: object = fig.subplots(1, length)

    for i in range(length):
        cmap = axes[i].imshow(zernikes[i])
        fig.colorbar(cmap, ax=axes[i])
    
    plt.show()


nolls: int = [i for i in range(3, 10)]
coeffs: float = np.ones((len(nolls),), float)

npix: int = 128
grid: float = np.linspace(-1., 1., npix)
coords: float = np.array(np.meshgrid(grid, grid))

aper: ApertureLayer = SquareAperture(1.)
basis: ApertureLayer = AberratedAperture(nolls, coeffs, aper)

aperture = aper._aperture(coords)
zernikes: float = np.stack([h(coords) for h in basis.basis_funcs])

plot_basis(zernikes)

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
