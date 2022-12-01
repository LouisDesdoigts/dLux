import dLux as dl
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

pixels = 128
coordinates = dl.utils.get_pixel_coordinates(pixels, 2. / pixels)

# Uniform Spider Testing
occ_soft_circ_ap = dl.CircularAperture(0., 0., 1., True, True)
occ_hard_circ_ap = dl.CircularAperture(0., 0., 1., True, False)
soft_circ_ap = dl.CircularAperture(0., 0., 1., False, True)
hard_circ_ap = dl.CircularAperture(0., 0., 1., False, False)
x_trans_circ_ap = dl.CircularAperture(.5, 0., 1., False, False)
y_trans_circ_ap = dl.CircularAperture(0., .5, 1., False, False)

fig, axes = plt.subplots(2, 3, figsize=(3*4, 2*3))

axes[0][0].set_title("Occ. Soft Circ. Ap.")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
_map = axes[0][0].imshow(occ_soft_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][0])

axes[0][1].set_title("Occ. Hard Circ. Ap.")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
_map = axes[0][1].imshow(occ_hard_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][1])

axes[0][2].set_title("Soft Circ. Ap.")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
_map = axes[0][2].imshow(soft_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][2])


axes[1][0].set_title("Hard Circ. Ap.")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
_map = axes[1][0].imshow(hard_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][0])

axes[1][1].set_title("Trans. x Circ. Ap.")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
_map = axes[1][1].imshow(x_trans_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][1])

axes[1][2].set_title("Trans. y Circ. Ap.")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
_map = axes[1][2].imshow(y_trans_circ_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][2])
plt.show()


occ_soft_sq_ap = dl.SquareAperture(0., 0., 0., 1., True, True)
occ_hard_sq_ap = dl.SquareAperture(0., 0., 0., 1., True, False)
soft_sq_ap = dl.SquareAperture(0., 0., 0., 1., False, True)
hard_sq_ap = dl.SquareAperture(0., 0., 0., 1., False, False)
x_trans_sq_ap = dl.SquareAperture(.5, 0., 0., 1., False, False)
y_trans_sq_ap = dl.SquareAperture(0., .5, 0., 1., False, False)
rot_sq_ap = dl.SquareAperture(0., 0., np.pi / 4., 1., False, False)

fig, axes = plt.subplots(2, 4, figsize=(4*4, 2*3))

axes[0][0].set_title("Occ. Soft Sq. Ap.")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
_map = axes[0][0].imshow(occ_soft_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][0])

axes[0][1].set_title("Occ. Hard Sq. Ap.")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
_map = axes[0][1].imshow(occ_hard_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][1])

axes[0][2].set_title("Soft Sq. Ap.")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
_map = axes[0][2].imshow(soft_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][2])


axes[1][0].set_title("Hard Sq. Ap.")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
_map = axes[1][0].imshow(hard_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][0])

axes[1][1].set_title("Trans. x Sq. Ap.")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
_map = axes[1][1].imshow(x_trans_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][1])

axes[1][2].set_title("Trans. y Sq. Ap.")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
_map = axes[1][2].imshow(y_trans_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][2])

axes[0][3].set_title("Rot. Sq. Ap.")
axes[0][3].set_xticks([])
axes[0][3].set_yticks([])
_map = axes[0][3].imshow(rot_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][3])

axes[1][3].set_xticks([])
axes[1][3].set_yticks([])
axes[1][3].axis("off")
plt.show()

# Rectangular Aperture
occ_soft_sq_ap = dl.SquareAperture(0., 0., 0., 1., True, True)
occ_hard_sq_ap = dl.SquareAperture(0., 0., 0., 1., True, False)
soft_sq_ap = dl.SquareAperture(0., 0., 0., 1., False, True)
hard_sq_ap = dl.SquareAperture(0., 0., 0., 1., False, False)
x_trans_sq_ap = dl.SquareAperture(.5, 0., 0., 1., False, False)
y_trans_sq_ap = dl.SquareAperture(0., .5, 0., 1., False, False)
rot_sq_ap = dl.SquareAperture(0., 0., np.pi / 4., 1., False, False)

fig, axes = plt.subplots(2, 4, figsize=(4*4, 2*3))

axes[0][0].set_title("Occ. Soft Sq. Ap.")
axes[0][0].set_xticks([])
axes[0][0].set_yticks([])
_map = axes[0][0].imshow(occ_soft_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][0])

axes[0][1].set_title("Occ. Hard Sq. Ap.")
axes[0][1].set_xticks([])
axes[0][1].set_yticks([])
_map = axes[0][1].imshow(occ_hard_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][1])

axes[0][2].set_title("Soft Sq. Ap.")
axes[0][2].set_xticks([])
axes[0][2].set_yticks([])
_map = axes[0][2].imshow(soft_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][2])


axes[1][0].set_title("Hard Sq. Ap.")
axes[1][0].set_xticks([])
axes[1][0].set_yticks([])
_map = axes[1][0].imshow(hard_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][0])

axes[1][1].set_title("Trans. x Sq. Ap.")
axes[1][1].set_xticks([])
axes[1][1].set_yticks([])
_map = axes[1][1].imshow(x_trans_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][1])

axes[1][2].set_title("Trans. y Sq. Ap.")
axes[1][2].set_xticks([])
axes[1][2].set_yticks([])
_map = axes[1][2].imshow(y_trans_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[1][2])

axes[0][3].set_title("Rot. Sq. Ap.")
axes[0][3].set_xticks([])
axes[0][3].set_yticks([])
_map = axes[0][3].imshow(rot_sq_ap._aperture(coordinates))
fig.colorbar(_map, ax=axes[0][3])

axes[1][3].set_xticks([])
axes[1][3].set_yticks([])
axes[1][3].axis("off")
plt.show()
