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


nolls: int = [i for i in range(3, 10)]
coeffs: float = np.ones((len(nolls),), float)
nterms: int = len(nolls)

npix: int = 128
grid: float = np.linspace(-1., 1., npix)
coords: float = np.array(np.meshgrid(grid, grid))

aper: ApertureLayer = SquareAperture(1.)
basis: ApertureLayer = AberratedAperture(nolls, coeffs, aper)

aperture = aper._aperture(coords)
zernikes: float = np.stack([h(coords) for h in basis.basis_funcs])

# +
# for j in np.arange(1, nterms):
# -

pixel_area: float = aperture.sum()
shape: tuple = zernikes.shape
width: int = shape[-1]
_basis: float = np.zeros(shape).at[0].set(aperture)
j: int = 1

plot_basis(_basis)

intermediate = zernikes[j] * aperture
coefficient = np.zeros((nterms, 1, 1), dtype=float)
mask = (np.arange(1, nterms) > j + 1)[:, None, None].astype(float)

mask.flatten()

# So I have identified the problem. `_basis[1:]` is identically zero. 

coefficient = -1. / pixel_area * (zernikes[j] * _basis[1:] * aperture * mask)

plot_basis(coefficient)

coefficient = -1. / pixel_area * \
    (zernikes[j] * _basis[1:] * aperture * mask)\
    .sum(axis = (1, 2))\
    .reshape(-1, 1, 1) 

intermediate += (coefficient * _basis[1:] * mask).sum(axis = 0)

_basis = _basis\
    .at[j]\
    .set(intermediate / \
        np.sqrt((intermediate ** 2).sum() / pixel_area))

plot_basis(_basis)

# +
if aperture is None:
    aperture = hex_aperture(npix=npix, rho=rho, theta=theta, vertical=vertical, outside=0)

# any pixels with zero or NaN in the aperture are outside the area
apmask = (np.isfinite(aperture) & (aperture > 0))
apmask_float = np.asarray(apmask, float)
A = apmask.sum()

# precompute zernikes
Z = np.full((nterms + 1,) + shape, outside, dtype=float)
Z[1:] = zernike_basis(nterms=nterms, npix=npix, rho=rho, theta=theta, outside=0.0)

G = [np.zeros(shape), np.ones(shape)]  # array of G_i etc. intermediate fn
H = [np.zeros(shape), apmask_float.copy()]  # array of hexikes
c = {}  # coefficients hash

for j in np.arange(nterms - 1) + 1:  # can do one less since we already have the piston term
    _log.debug("  j = " + str(j))
    # Compute the j'th G, then H
    nextG = Z[j + 1] * apmask_float
    for k in np.arange(j) + 1:
        c[(j + 1, k)] = -1 / A * (Z[j + 1] * H[k] * apmask_float).sum()
        if c[(j + 1, k)] != 0:
            nextG += c[(j + 1, k)] * H[k]
        _log.debug("    c[%s] = %f", str((j + 1, k)), c[(j + 1, k)])

    nextH = nextG / np.sqrt((nextG ** 2).sum() / A)

    G.append(nextG)
    H.append(nextH)

    # TODO - contemplate whether the above algorithm is numerically stable
    # cf. modified gram-schmidt algorithm discussion on wikipedia.

basis = np.asarray(H[1:])  # drop the 0th null element
basis[:, ~apmask] = outside
return basis
