import jax.numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dLux as dl
import jax

jax.config.update("jax_debug_nans", True)

nolls: float = np.arange(3, 9, dtype = int)
coeffs: float = np.ones((6,), dtype = float)
aperture: object = dl.CircularAperture(1.)
basis: object = dl.AberratedAperture(nolls, coeffs, aperture)
npix: int = 128
width: float = 2.

basis.get_basis(npix, width)
