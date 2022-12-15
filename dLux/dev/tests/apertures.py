import dLux as dl
import jax.numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt

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
