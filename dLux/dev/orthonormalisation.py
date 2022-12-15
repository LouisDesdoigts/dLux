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
        cmap = axes[i].imshow(basis[i])
        fig.colorbar(cmap, ax=axes[i])
    
    plt.show()


nolls: int = [i for i in range(0, 10)]
coeffs: float = np.ones((len(nolls),), float)
nterms: int = len(nolls)

npix: int = 128
grid: float = np.linspace(-1., 1., npix)
coords: float = np.array(np.meshgrid(grid, grid))

aper: ApertureLayer = SquareAperture(1.)
basis: ApertureLayer = AberratedAperture(nolls, coeffs, aper)

aperture: float = aper._aperture(coords)
zcoords: float = aper._normalised_coordinates(coords)
zernikes: float = np.stack([h(zcoords) for h in basis.basis_funcs])


def inner_product(f1: float, f2: float) -> float:
    return (f1 * f2).sum()


def projection(f1: float, f2: float) -> float:
    return inner_product(f1, f2) / inner_product(f2, f2) * f2


plot_basis(zernikes)


