import jax.numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt


__all__ = ["list_to_dictionary", "single_image_plot", "two_image_plot",
           "spectrum_plot"]


Array = np.ndarray


def list_to_dictionary(list_in : list, ordered : bool = True) -> dict:
    """
    Converts some input list of dLux layers and converts them into an
    OrderedDict with the correct structure, ensuring that all keys are unique.

    Parameters
    ----------
    list_in : list
        The list of dLux OpticalLayers or DetectorLayers to be converted into
        a dictionary.
    ordered : bool = True
        Whether to return an ordered or regular dictionary.

    Returns
    -------
    dictionary : dict
        The equivilent dictionary or ordered dictionary.
    """
    # Construct names list and identify repeats
    names, repeats = [], []
    for i in range(len(list_in)):

        # Check for name attribute
        if hasattr(list_in[i], 'name') and list_in[i].name is not None:
            name = list_in[i].name

        # Else take name from object
        else:
            name = str(list_in[i]).split('(')[0]

        # Check for Repeats
        if name in names:
            repeats.append(name)
        names.append(name)

    # Get list of unique repeats
    repeats = list(set(repeats))

    # Iterate over repeat names
    for i in range(len(repeats)):

        idx = 0
        # Iterate over names list and append index value to name
        for j in range(len(names)):
            if repeats[i] == names[j]:
                names[j] = names[j] + '_{}'.format(idx)
                idx += 1

    # Turn list into Dictionary
    dict_out = OrderedDict() if ordered else {}
    for i in range(len(names)):

        # Assert no spaces in the name in order to ensure the __getattrr__
        # method will work
        assert ' ' not in names[i], \
        ("names can not contain spaces, {} was supplied.".format(names[i]))
        # dict_out[names[i]] = list_in[i]
        # Throws an equinox error since name is defined as a static_field
        # dict_out[names[i]] = list_in[i].set_name(names[i])
        dict_out[names[i]] = list_in[i].set('name', names[i])


    return dict_out


def single_image_plot(array       : Array, 
                      figsize     : tuple = (5, 4), 
                      title       : str   = "Array",
                      cbar_label  : str   = None,
                      cmap        : str   = "inferno",
                      bound       : float = None,
                      dpi         : int   = 120) -> None:
    """
    Plots a  single image.

    Parameters
    ----------
    array : Array
        The first array to plot.
    figsize : tuple = (5, 4)
        The size of the figure to display.
    title : str = "Array"
        The title of the array.
    cmap : str = "inferno"
        The colour map to use.
    bound : float = None
        The bound of the colour map.
    dpi : int = 120
        The resolution of the figure.
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    if bound is not None:
        array = array % bound
    plt.imshow(array, cmap=cmap)
    cbar = plt.colorbar()
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    plt.show()


def two_image_plot(array1      : Array, 
                   array2      : Array, 
                   figsize     : tuple = (10, 4), 
                   titles      : tuple = ("Array 1", "Array 2"),
                   cbar_labels : tuple = (None, None),
                   cmaps       : tuple = ("inferno", "inferno"),
                   bounds      : tuple = (None, None),
                   dpi         : int   = 120) -> None:
    """
    Plots two images side by side.

    Parameters
    ----------
    array1 : Array
        The first array to plot.
    array2 : Array
        The second array to plot.
    figsize : tuple = (10, 4)
        The size of the figure to display.
    titles : tuple = ("Array 1", "Array 2")
        The titles of the arrays.
    cmaps : tuple = ("inferno", "inferno")
        The colour maps to use.
    bounds : tuple = (None, None)
        The bounds of the colour maps.
    dpi : int = 120
        The resolution of the figure.
    """
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.title(titles[0])
    if bounds[0] is not None:
        array1 = array1 % bounds[0]
    plt.imshow(array1, cmap=cmaps[0])
    cbar = plt.colorbar()
    if cbar_labels[0] is not None:
        cbar.set_label(cbar_labels[0])

    plt.subplot(1, 2, 2)
    plt.title(titles[1])
    if bounds[1] is not None:
        array2 = array2 % bounds[1]
    plt.imshow(array2, cmap=cmaps[1])
    cbar = plt.colorbar()
    if cbar_labels[1] is not None:
        cbar.set_label(cbar_labels[1])
    plt.show()


def spectrum_plot(wavelengths     : Array, 
                  weights         : Array, 
                  figsize         : tuple = (6, 3),
                  labels          : tuple = None, 
                  cartesian_units : str = 'meters',
                  dpi             : int = 120) -> None:
    """
    Plots a spectrum based on wavelgths and weights.

    Parameters
    ----------
    wavelengths : Array, meters
        The wavelengths of the spectrum.
    weights : Array
        The weights of the spectrum.
    figsize : tuple = (6, 3)
        The size of the figure to display.
    labels : tuple = None
        The labels of the spectra.
    cartesian_units : str = 'meters'
        The units of the wavelengths.
    dpi : int = 120
        The resolution of the figure.
    """
    nspectra = 1 if wavelengths.ndim == 1 else len(wavelengths)
    if labels is None:
        labels = ['Spectrum {}'.format(i) for i in range(nspectra)]
    
    plt.figure(figsize=figsize, dpi=dpi)
    if nspectra == 1:
        plt.scatter(wavelengths, weights, label=labels[0])
    else:
        for i in range(nspectra):
            plt.scatter(wavelengths[i], weights[i], label=labels[i])
    plt.legend()
    plt.xlabel(f"Wavelengths {cartesian_units}")
    plt.ylabel("Weights")
    plt.ylim(-0.01)
    plt.show()