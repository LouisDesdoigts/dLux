from collections import OrderedDict
from typing import Any, Callable
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_map

__all__ = ["map2array", "list2dictionary", "insert_layer", "remove_layer"]


def map2array(fn: Callable, tree: Any, leaf_fn: Callable = None):
    """
    Maps a function across a pytree, flattening it and turning it into an
    array.
    """
    if leaf_fn is not None:
        return np.array(tree_flatten(tree_map(fn, tree, is_leaf=leaf_fn))[0])
    else:
        return np.array(tree_flatten(tree_map(fn, tree))[0])


# TODO: Map to list to handle different output shapes?


def list2dictionary(
    list_in: list, ordered: bool, allowed_types: tuple = ()
) -> dict:
    """
    Converts some input list of dLux layers and converts them into an
    OrderedDict with the correct structure, ensuring that all keys are unique.

    Parameters
    ----------
    list_in : list
        The list of dLux Layers to be converted into a dictionary.
    ordered : bool
        Whether to return an ordered or regular dictionary.
    allowed_types : tuple
        The allowed types of layers to be included in the dictionary.

    Returns
    -------
    dictionary : dict
        The equivalent dictionary or ordered dictionary.
    """
    # Construct names list and identify repeats
    names, repeats = [], []
    for item in list_in:
        # Check for specified names
        if isinstance(item, tuple):
            # item, name = item
            name, item = item
        else:
            name = item.__class__.__name__

        # Check input types
        if allowed_types != () and not isinstance(item, allowed_types):
            raise TypeError(
                f"Item {name} is not an allowed type, got " f"{type(item)}"
            )

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
                names[j] = names[j] + "_{}".format(idx)
                idx += 1

    # Turn list into Dictionary
    dict_out = OrderedDict() if ordered else {}
    for i in range(len(names)):
        # Check for spaces in names
        if " " in names[i]:
            raise ValueError(f"Names can not contain spaces, got {names[i]}")

        # Add to dict
        if isinstance(list_in[i], tuple):
            # item = list_in[i][0]
            item = list_in[i][1]
        else:
            item = list_in[i]
        dict_out[names[i]] = item
    return dict_out


def insert_layer(layers: OrderedDict, layer, index: int, type):
    layers_list = list(zip(layers.keys(), layers.values()))
    layers_list.insert(index, layer)
    return list2dictionary(layers_list, True, type)


def remove_layer(layers: OrderedDict, key: str):
    layers.pop(key)
    return layers
