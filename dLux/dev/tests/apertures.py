import dLux as dl
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"

pixels = 128
coordinates = dl.utils.get_pixel_coordinates(pixels, 2. / pixels)

# Circular Apertures
circ_aps = {
    "Occ. Soft Circ. Ap.": dl.CircularAperture(0., 0., 1., True, True),
    "Occ. Hard Circ. Ap.": dl.CircularAperture(0., 0., 1., True, False),
    "Soft Circ. Ap.": dl.CircularAperture(0., 0., 1., False, True),
    "Hard Circ. Ap.": dl.CircularAperture(0., 0., 1., False, False),
    "Trans. X Circ. Ap.": dl.CircularAperture(.5, 0., 1., False, False),
    "Trans. Y Circ. Ap.": dl.CircularAperture(0., .5, 1., False, False)
}

# Annular Apertures
ann_aps = {
    "Occ. Soft. Circ Ap.": dl.AnnularAperture(0., 0., 1., .5, True, True),
    "Occ. Hard. Circ Ap.": dl.AnnularAperture(0., 0., 1., .5, True, False),
    "Soft Circ. Ap.": dl.AnnularAperture(0., 0., 1., .5, False, True),
    "Hard Circ. Ap.": dl.AnnularAperture(0., 0., 1., .5, False, False),
    "Trans. X Circ. Ap.": dl.AnnularAperture(.5, 0., 1., .5, False, False),
    "Trans. Y Circ. Ap.": dl.AnnularAperture(0., .5, 1., .5, False, False)
}

# Square Apertures
sq_aps = {
    "Occ. Soft Sq. Ap.": dl.SquareAperture(0., 0., 0., 1., True, True),
    "Occ. Hard Sq. Ap.": dl.SquareAperture(0., 0., 0., 1., True, False),
    "Soft Sq. Ap.": dl.SquareAperture(0., 0., 0., 1., False, True),
    "Hard Sq. Ap.": dl.SquareAperture(0., 0., 0., 1., False, False),
    "Trans. x Sq. Ap.": dl.SquareAperture(.5, 0., 0., 1., False, False),
    "Trans. y Sq. Ap.": dl.SquareAperture(0., .5, 0., 1., False, False),
    "Rot. Sq. Ap.": dl.SquareAperture(0., 0., np.pi / 4., 1., False, False)
}

# Rectangular Aperture
rect_aps = {
    "Occ. Soft. Rect. Ap.": dl.RectangularAperture(0., 0., 0., 1., .5, True, True),
    "Occ. Hard. Rect. Ap.": dl.RectangularAperture(0., 0., 0., 1., .5, True, False),
    "Soft Rect. Ap.": dl.RectangularAperture(0., 0., 0., 1., .5, False, True),
    "Hard Rect. Ap.": dl.RectangularAperture(0., 0., 0., 1., .5, False, False),
    "Trans x Rect. Ap.": dl.RectangularAperture(.5, 0., 0., 1., .5, False, False),
    "Trans y Rect. Ap.": dl.RectangularAperture(0., .5, 0., 1., .5, False, False),
    "Rot. Rect. Ap.": dl.RectangularAperture(0., 0., np.pi / 4., 1., .5, False, False)
}

# Hexagonal Apertures
hex_aps = {
    "Occ. Soft Sq. Ap.": dl.HexagonalAperture(0., 0., 0., 1., True, True),
    "Occ. Hard Sq. Ap.": dl.HexagonalAperture(0., 0., 0., 1., True, False),
    "Soft Sq. Ap.": dl.HexagonalAperture(0., 0., 0., 1., False, True),
    "Hard Sq. Ap.": dl.HexagonalAperture(0., 0., 0., 1., False, False),
    "Trans. x Sq. Ap.": dl.HexagonalAperture(.5, 0., 0., 1., False, False),
    "Trans. y Sq. Ap.": dl.HexagonalAperture(0., .5, 0., 1., False, False),
    "Rot. Sq. Ap.": dl.HexagonalAperture(0., 0., np.pi / 4., 1., False, False)
}

# Apertures
aps = {
    "Circ. Aps.": circ_aps,
    "Ann. Aps.": ann_aps,
    "Rect. Aps.": rect_aps,
    "Sq. Aps.": sq_aps,
    "Hex. Aps.": hex_aps
}

# The massive plotting code that goes into generating the figure. 
# So I need to make this more streamlined. I could do this by putting 
# the apertures in a list that I then iterated over. That sounds hella 
# lit. You know what? Mayber I do each one along a line as well with 
# subfigures they should all be centered. Alternatively I could 
# just do this with a massive grid of subplots. 
fig = plt.figure()
subfigs = fig.subfigures(5, 1)

for subfig, ap in zip(subfigs, aps):
    _aps = aps[ap]
    num_aps = len(_aps)

    axes = subfig.subplots(1, num_aps)
    
    for i, ap in enumerate(_aps):
        axes[i].set_title(ap)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        _map = axes[i].imshow(_aps[ap]._aperture(coordinates))
        subfig.colorbar(_map, ax=axes[i])

plt.show()
