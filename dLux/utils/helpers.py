import jax.numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from jax import Array


__all__ = ["list_to_dictionary"]


def list_to_dictionary(list_in : list, ordered : bool = True) -> dict:
    """
    Converts some input list of dLux layers and converts them into an
    OrderedDict with the correct structure, ensuring that all keys are unique.

    Parameters
    ----------
    list_in : list
        The list of dLux Layersto be converted into a dictionary.
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
        item = list_in[i]

        if isinstance(item, tuple):
            item = item[0]
            name = item[1]
        else:
            name = item.__class__.__name__

        # Check for Repeats
        if name in names:
            repeats.append(name)
        names.append(name)

    # Get list of unique repeats
    repeats = list(set(repeats))

    # Iterate over repeat names
    for i in range(len(repeats)):

        # Iterate over names list and append index value to name
        idx = 0
        for j in range(len(names)):
            if repeats[i] == names[j]:
                names[j] = names[j] + '_{}'.format(idx)
                idx += 1

    # Turn list into Dictionary
    dict_out = OrderedDict() if ordered else {}
    for i in range(len(names)):
        
        # Check for spaces in names
        if ' ' in names[i]:
            raise ValueError(f"Names can not contain spaces, got {names[i]}")
        
        # Add to dict
        if isinstance(list_in[i], tuple):
            item = list_in[i][0]
        else:
            item = list_in[i]
        dict_out[names[i]] = item
    return dict_out