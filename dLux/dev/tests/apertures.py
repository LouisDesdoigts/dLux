import dLux as dl
import dLux
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
from dLux.apertures import *

mpl.rcParams["text.usetex"] = True
mpl.rcParams["image.cmap"] = "inferno"


def test_plots_of_aps(aps: dict) -> None:
    """
    A formalisation of the common testing routine that I have
    been using. This will be removed from the production code. 

    Parameters:
    -----------
    aps: dict
        The apertures with descriptive titles.
    """
    npix = 128
    width = 2.
    coords = dLux.utils.get_pixel_coordinates(npix, width / npix)
    num = len(aps)
    fig, axes = plt.subplots(1, num, figsize=(4*num, 3))
    for i, ap in enumerate(aps):
        axes[i].set_title(ap)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        _map = axes[i].imshow(aps[ap]._aperture(coords))
        fig.colorbar(_map, ax=axes[i])

    plt.show()


def test_plots_of_aber_aps(aber_aps: dict):
    """
    A formalisation of the common testing routine that I have
    been using. This will be removed from the production code. 

    Parameters:
    -----------
    aber_aps: dict
        The apertures with descriptive titles.
    """
    pixels: float = 128
    nterms: float = 6
    coords: float = dLux.utils.get_pixel_coordinates(pixels, 3. / pixels)

    length: int = len(aber_aps.keys())
    fig = plt.figure()
    subfigs = fig.subfigures(length, 1)

    for i, aber_ap in enumerate(aber_aps):
        subfigs[i].suptitle(aber_ap)

        basis: float = aber_aps[aber_ap]._basis(coords)
        aper: float = aber_aps[aber_ap].aperture._aperture(coords)
        
        num: int = aber_aps[aber_ap].nterms
        axes = subfigs[i].subplots(1, num)

        for j in range(num):
            _map = axes[j].imshow(basis[j] * aper)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
            axes[j].axis("off")
            subfigs[i].colorbar(_map, ax=axes[j])
            
    plt.show()


def test_plots_of_stat_aps(aps: dict):
    """
    A formalisation of the common testing routine that I have
    been using. This will be removed from the production code. 

    Parameters:
    -----------
    aps: dict
        The apertures with descriptive titles.
    """
    length: int = len(aps.keys())
    fig: object = plt.figure(figsize=(length*4, 3))
    axes: object = fig.subplots(1, length)

    for i, ap in enumerate(aps):
        axes[i].set_title(aps)
        cmap = axes[i].imshow(aps[ap].aperture)
        fig.colorbar(cmap, ax=axes[i])

    plt.show()


def test_plots_of_stat_aber_aps(aber_aps: dict):
    """
    A formalisation of the common testing routine that I have
    been using. This will be removed from the production code. 

    Parameters:
    -----------
    aber_aps: dict
        The apertures with descriptive titles.
    """
    length: int = len(aber_aps.keys())
    fig = plt.figure()
    subfigs = fig.subfigures(length, 1)

    for i, aber_ap in enumerate(aber_aps):
        subfigs[i].suptitle(aber_ap)

        basis: float = aber_aps[aber_ap].basis
        aper: float = aber_aps[aber_ap].aperture
        
        num: int = basis.shape[0]
        axes = subfigs[i].subplots(1, num)

        for j in range(num):
            _map = axes[j].imshow(basis[j] * aper)
            axes[j].set_xticks([])
            axes[j].set_yticks([])
            axes[j].axis("off")
            subfigs[i].colorbar(_map, ax=axes[j])
            
    plt.show()


################################## tests ######################################
test_plots_of_aps({
    "Occ. Soft": RectangularAperture(1., .5, occulting=True, softening=True),
    "Occ. Hard": RectangularAperture(1., .5, occulting=True),
    "Soft": RectangularAperture(1., .5, softening=True),
    "Hard": RectangularAperture(1., .5),
    "Trans.": RectangularAperture(1., .5, centre=[.5, .5]),
    "Strain": RectangularAperture(1., .5, strain=[.5, 0.]),
    "Compr.": RectangularAperture(1., .5, compression=[.5, 1.]),
    "Rot.": RectangularAperture(1., .5, rotation=np.pi / 4.)
})


test_plots_of_aps({
    "Occ. Soft": CircularAperture(1., occulting=True, softening=True),
    "Occ. Hard": CircularAperture(1., occulting=True),
    "Soft": CircularAperture(1., softening=True),
    "Hard": CircularAperture(1.),
    "Trans.": CircularAperture(1., centre=[.5, .5]),
    "Strain": CircularAperture(1., strain=[.5, 0.]),
    "Compr.": CircularAperture(1., compression=[.5, 1.])
})


test_plots_of_aps({
    "Occ. Soft": AnnularAperture(1., .5, occulting=True, softening=True),
    "Occ. Hard": AnnularAperture(1., .5, occulting=True),
    "Soft": AnnularAperture(1., .5, softening=True),
    "Hard": AnnularAperture(1., .5),
    "Trans.": AnnularAperture(1., .5, centre=[.5, .5]),
    "Strain": AnnularAperture(1., .5, strain=[.5, 0.]),
    "Compr.": AnnularAperture(1., .5, compression=[.5, 1.])
})


test_plots_of_aps({
   "Occ. Soft": SquareAperture(1., occulting=True, softening=True),
   "Occ. Hard": SquareAperture(1., occulting=True),
   "Soft": SquareAperture(1., softening=True),
   "Hard": SquareAperture(1.),
   "Trans.": SquareAperture(1., centre=[.5, .5]),
   "Strain": SquareAperture(1., strain=[.5, 0.]),
   "Compr.": SquareAperture(1., compression=[.5, 1.]),
   "Rot.": SquareAperture(1., rotation=np.pi / 4.)
})


vert_angs: float = np.linspace(0., 2. * np.pi, 4, endpoint=False)
verts: float = np.array([np.cos(vert_angs), np.sin(vert_angs)])
trans_verts: float = np.transpose(verts)


test_plots_of_aps({
   "Occ. Soft": IrregularPolygonalAperture(trans_verts, occulting=True, softening=True),
   "Occ. Hard": IrregularPolygonalAperture(trans_verts, occulting=True),
   "Soft": IrregularPolygonalAperture(trans_verts, softening=True),
   "Hard": IrregularPolygonalAperture(trans_verts),
   "Trans.": IrregularPolygonalAperture(trans_verts, centre=[.5, .5]),
   "Strain": IrregularPolygonalAperture(trans_verts, strain=[.5, 0.]),
   "Compr.": IrregularPolygonalAperture(trans_verts, compression=[.5, 1.]),
   "Rot.": IrregularPolygonalAperture(trans_verts, rotation=np.pi / 4.)
})


test_plots_of_aps({
   "Occ. Soft": RegularPolygonalAperture(5, 1., occulting=True, softening=True),
   "Occ. Hard": RegularPolygonalAperture(5, 1., occulting=True),
   "Soft": RegularPolygonalAperture(5, 1., softening=True),
   "Hard": RegularPolygonalAperture(5, 1.),
   "Trans.": RegularPolygonalAperture(5, 1., centre=[.5, .5]),
   "Strain": RegularPolygonalAperture(5, 1., strain=[.5, 0.]),
   "Compr.": RegularPolygonalAperture(5, 1., compression=[.5, 1.]),
   "Rot.": RegularPolygonalAperture(5, 1., rotation=np.pi / 4.)
})


test_plots_of_aps({
   "Occ. Soft": HexagonalAperture(1., occulting=True, softening=True),
   "Occ. Hard": HexagonalAperture(1., occulting=True),
   "Soft": HexagonalAperture(1., softening=True),
   "Hard": HexagonalAperture(1.),
   "Trans.": HexagonalAperture(1., centre=[.5, .5]),
   "Strain": HexagonalAperture(1., strain=[.5, 0.]),
   "Compr.": HexagonalAperture(1., compression=[.5, 1.]),
   "Rot.": HexagonalAperture(1., rotation=np.pi / 4.)
})


test_plots_of_aps({
    "Comp. Trans.": CompoundAperture(
        centre = [.5, .5],
        apertures = {
            "pupil": CircularAperture(1.),
            "obstruction": SquareAperture(.5, occulting=True),
    }),
    "Circ. Trans.": CompoundAperture(
        apertures = {
            "pupil": CircularAperture(1., centre=[.1, .1]),
            "obstruction": SquareAperture(.5, occulting=True),
    }),
    "Comp. Rot.": CompoundAperture(
        rotation = np.pi / 4.,
        apertures = {
            "pupil": CircularAperture(1.),
            "obstruction": SquareAperture(.5, occulting=True),
    }),
    "Comp. Strain": CompoundAperture(
        strain = [.05, .05],
        apertures = {
            "pupil": CircularAperture(1.),
            "obstruction": SquareAperture(.5, occulting=True),
    }),
    "Comp. Compr.": CompoundAperture(
        compression = [1., .5],
        apertures = {
            "pupil": CircularAperture(1.),
            "obstruction": SquareAperture(.5, occulting=True),
    })
})


test_plots_of_aps({
    "Comp. Trans.": MultiAperture(
        centre = [.5, .5],
        apertures = {
            "pupil": CircularAperture(.5, centre=[-.5, 0.]),
            "obstruction": CircularAperture(.5, centre=[.5, 0.]),
        }
    ),
    "Circ. Trans.": MultiAperture(
        apertures = {
            "pupil": CircularAperture(.5, centre=[-.5, .5]),
            "obstruction": CircularAperture(.5, centre=[.5, 0.]),
        }
    ),
    "Comp. Rot.": MultiAperture(
        rotation = np.pi / 4.,
        apertures = {
            "pupil": CircularAperture(.5, centre=[-.5, 0.]),
            "obstruction": CircularAperture(.5, centre=[.5, 0.]),
        }
    ),
    "Comp. Strain": MultiAperture(
        strain = [.05, .05],
        apertures = {
            "pupil": CircularAperture(.5, centre=[-.5, 0.]),
            "obstruction": CircularAperture(.5, centre=[.5, 0.]),
        }
    ),
    "Comp. Compr.": MultiAperture(
        compression = [1., .5],
        apertures = {
            "pupil": CircularAperture(.5, centre=[-.5, 0.]),
            "obstruction": CircularAperture(.5, centre=[.5, 0.]),
        }
    )
})

test_plots_of_aps({
    "Van.": UniformSpider(3, .1),
    "Trans.": UniformSpider(3, .1, centre=[.25, .25]),
    "Strain.": UniformSpider(3, .1, strain=[.05, .05]),
    "Compr.": UniformSpider(3, .1, compression=[1., .5]),
    "Rot.": UniformSpider(3, .1, rotation=np.pi / 2.),
    "Soft": UniformSpider(3, .1, softening=True),
    "More": UniformSpider(6, .1)
})

nolls: int = [i for i in range(3, 10)]
coeffs: float = np.ones((len(nolls),), float)

test_plots_of_aber_aps({
   "Squarikes": AberratedAperture(nolls, coeffs, SquareAperture(np.sqrt(2.))),
   "Annikes": AberratedAperture(nolls, coeffs, AnnularAperture(1., .5)),
   "Rectikes": AberratedAperture(nolls, coeffs, RectangularAperture(1., 2.)),
   "Hexikes": AberratedAperture(nolls, coeffs, HexagonalAperture(1.)),
   "Reg. Pol. 5": AberratedAperture(nolls, coeffs, RegularPolygonalAperture(5, 1.)),
   "Reg. Pol. 4": AberratedAperture(nolls, coeffs, RegularPolygonalAperture(4, 1.)),
   "Circ.": AberratedAperture(nolls, coeffs, CircularAperture(1.))
})

test_plots_of_stat_aps({
   "Sq.": StaticAperture(SquareAperture(np.sqrt(2.)), 128, 2. / 128),
   "Ann.": StaticAperture(AnnularAperture(1., .5), 128, 2. / 128),
   "Rect.": StaticAperture(RectangularAperture(1., 2.), 128, 2. / 128),
   "Hex.": StaticAperture(HexagonalAperture(1.), 128, 2. / 128),
   "Reg. Pol. 5": StaticAperture(RegularPolygonalAperture(5, 1.), 128, 2. / 128),
   "Reg. Pol. 4": StaticAperture(RegularPolygonalAperture(4, 1.), 128, 2. / 128),
   "Circ.": StaticAperture(CircularAperture(1.), 128, 2. / 128)
})

nolls: int = [i for i in range(3, 10)]
coeffs: float = np.ones((len(nolls),), float)

test_plots_of_stat_aber_aps({
    "Sq.": StaticAberratedAperture(AberratedAperture(nolls, coeffs, SquareAperture(np.sqrt(2.))), 128, 2. / 128),
    "Ann.": StaticAberratedAperture(AberratedAperture(nolls, coeffs, AnnularAperture(1., .5)), 128, 2. / 128),
    "Rect.": StaticAberratedAperture(AberratedAperture(nolls, coeffs, RectangularAperture(1., 2.)), 128, 2. / 128),
    "Hex.": StaticAberratedAperture(AberratedAperture(nolls, coeffs, HexagonalAperture(1.)), 128, 2. / 128),
    "Reg. Pol. 5": StaticAberratedAperture(AberratedAperture(nolls, coeffs, RegularPolygonalAperture(5, 1.)), 128, 2. / 128),
    "Reg. Pol. 4": StaticAberratedAperture(AberratedAperture(nolls, coeffs, RegularPolygonalAperture(4, 1.)), 128, 2. / 128),
    "Circ.": StaticAberratedAperture(AberratedAperture(nolls, coeffs, CircularAperture(1.)), 128, 2. / 128)
})
