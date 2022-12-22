import matplotlib as mpl 
import matplotlib.pyplot as plt 
import dLux

mpl.rcParams["image.cmap"] = "Greys"

width: float = 2.
pixels: int = 32
coords: float = dLux.utils.get_pixel_coordinates(pixels, width / pixels)

aps = [
    dLux.CircularAperture(1.),
    dLux.CircularAperture(1., centre=[.5, .5]),
    dLux.CircularAperture(1., strain=[.05, .05]),
    dLux.CircularAperture(1., compression=[1.05, .95]),
    dLux.CircularAperture(1., softening=True),
    dLux.CircularAperture(1., occulting=True)
]

shape: int = len(aps)
fig: object = plt.figure(figsize=(shape*4, 3))
axes: object = fig.subplots(1, shape)

for i in range(shape):
    axes[i].imshow(aps[i]._aperture(coords))
    axes[i].set_yticks([])
    axes[i].set_xticks([])

fig.savefig("docs/apertures/assets/circles.png")
plt.show()
    
