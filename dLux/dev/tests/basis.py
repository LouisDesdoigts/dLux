import dLux as dl
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True

pixels = 128
nterms = 6

coordinates = dl.utils.get_pixel_coordinates(2., 2. / pixels)
sq_aperture = dl.SquareAperture(0., 0., 1., 0., False, False)
sq_basis = dl.Basis(nterms, sq_aperture)
sq_basis_vecs = sq_basis.basis(coordinates) 

fig, axes = plt.subplots(2, 3)
axes[0][0].set_title("$j = 0$")
_map = axes[0][0].imshow(sq_basis_vecs[0])
fig.colorbar(_map, ax=axes[0][0])
axes[0][1].set_title("$j = 1$")
_map = axes[0][1].imshow(sq_basis_vecs[1])
fig.colorbar(_map, ax=axes[0][1])
axes[0][2].set_title("$j = 2$")
_map = axes[0][2].imshow(sq_basis_vecs[2])
fig.colorbar(_map, ax=axes[0][2])
axes[1][0].set_title("$j = 3$")
_map = axes[1][0].imshow(sq_basis_vecs[3])
fig.colorbar(_map, ax=axes[1][0])
axes[1][1].set_title("$j = 4$")
_map = axes[1][1].imshow(sq_basis_vecs[4])
fig.colorbar(_map, ax=axes[1][1])
axes[1][2].set_title("$j = 5$")
_map = axes[1][2].imshow(sq_basis_vecs[5])
fig.colorbar(_map, ax=axes[1][2])
plt.show()
