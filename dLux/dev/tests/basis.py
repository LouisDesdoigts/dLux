import dLux as dl
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

pixels = 128
nterms = 6

coordinates = dl.utils.get_pixel_coordinates(pixels, 2. / pixels)

num_ikes = 10
noll_inds = [i for i in range(num_ikes)]
circ_ap = dl.CircularAperture(0., 0., 1., False, False)
basis = dl.AberratedCircularAperture(noll_inds, np.ones((num_ikes,)), circ_ap)

_basis = basis._basis(coordinates)
_aperture = circ_ap._aperture(coordinates)

fig, axes = plt.subplots(2, num_ikes // 2, figsize=((num_ikes // 2)*4, 2*3))
for i in range(num_ikes):
    row = i // (num_ikes // 2)
    col = i % (num_ikes // 2)
    _map = axes[row][col].imshow(_basis[i] * _aperture)
    fig.colorbar(_map, ax=axes[row][col]) 

plt.show()

hex_ap = dl.HexagonalAperture(0., 0., 0., 1., False, False)
hex_basis = dl.AberratedHexagonalAperture(noll_inds, np.ones((num_ikes,)), hex_ap)

_basis = hex_basis._basis(coordinates)
_aperture = hex_ap._aperture(coordinates)

fig, axes = plt.subplots(2, num_ikes // 2, figsize=((num_ikes // 2)*4, 2*3))
for i in range(num_ikes):
    row = i // (num_ikes // 2)
    col = i % (num_ikes // 2)
    _map = axes[row][col].imshow(_basis[i] * _aperture)
    fig.colorbar(_map, ax=axes[row][col]) 

plt.show()
