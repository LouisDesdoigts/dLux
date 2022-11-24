import jax.numpy as np
from collections import OrderedDict


__all__ = ["list_to_dictionary"]


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
        dict_out[names[i]] = list_in[i]
        # Throws an equinox error since name is defined as a static_field
        # dict_out[names[i]] = list_in[i].set_name(names[i])

    return dict_out