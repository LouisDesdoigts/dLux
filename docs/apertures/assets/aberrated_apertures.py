import matplotlib as mpl
import matplotlib.pyplot as plt
import dLux
import jax.numpy as np

mpl.rcParams["image.cmap"] = "seismic"
mpl.rcParams["text.usetex"] = True

width: float = 2.
pixels: int = 128
coords: float = dLux.utils.get_pixel_coordinates(pixels, width / pixels)

nvecs: int = 6
nolls: int = np.arange(1, nvecs, dtype=int)
coeff: float = np.ones(nvecs, dtype=float)

aps = [
    dLux.AberratedAperture(nolls, coeff, dLux.CircularAperture(1.)),
    dLux.AberratedAperture(nolls, coeff, dLux.HexagonalAperture(1.)),
]

naps: int = len(aps)
fig: object = plt.figure()
figs: object = fig.subfigures(naps, 1)

for i in range(naps):
    axes: object = figs[i].subplots(1, nvecs)
    basis: float = aps[i]._basis(coords)

    for j in range(nvecs):
        axes[j].imshow(basis[j], vmin=basis.min(), vmax=basis.max())
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        axes[j].axis("off")

fig.savefig("docs/apertures/assets/aberrated_apertures.png")
plt.show()
