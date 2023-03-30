import matplotlib as mpl
import matplotlib.pyplot as plt
import dLux

mpl.rcParams["image.cmap"] = "Greys"
mpl.rcParams["text.usetex"] = True

width: float = 2.
pixels: int = 128
coords: float = dLux.utils.get_pixel_coordinates(pixels, width / pixels)

aps = [
    dLux.CircularAperture(1.),
    dLux.SquareAperture(1.),
    dLux.RectangularAperture(1., .5),
    dLux.HexagonalAperture(1.),
    dLux.AnnularAperture(1., .5),
    dLux.RegularPolygonalAperture(9, 1.),
]

shape: int = len(aps)
fig: object = plt.figure(figsize=(shape, 1))
axes: object = fig.subplots(1, shape)

for i in range(shape):
    axes[i].imshow(aps[i]._aperture(coords))
    axes[i].set_xticks([])
    axes[i].set_yticks([])

fig.savefig("docs/apertures/assets/apertures.png")
plt.show()


