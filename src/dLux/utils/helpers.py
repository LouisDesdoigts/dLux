from collections import OrderedDict
from typing import Any, Callable
import jax.numpy as np
import jax.tree as jtu
from jax import Array

__all__ = [
    "map2array",
    "list2dictionary",
    "insert_layer",
    "remove_layer",
    "inherit_docstrings",
]


def inherit_docstrings(cls, method_names=None):
    """
    Inherit docstrings and annotations from parent classes for specified methods.

    This function walks the MRO to find the first parent class with a docstring
    or annotations for each method, and copies them to the child class if missing.

    Parameters
    ----------
    cls : type
        The class being created via __init_subclass__.
    method_names : list[str] | None
        List of method names to inherit docstrings/annotations for.
        If None, only '__call__' is checked.

    Returns
    -------
    None
        Modifies cls in place.
    """
    if method_names is None:
        method_names = ["__call__"]

    for method_name in method_names:
        # Only process if method is defined in this class
        if method_name in cls.__dict__:
            method = cls.__dict__[method_name]

            # Inherit docstring if missing
            if method.__doc__ is None:
                for base in cls.__mro__[1:]:
                    if (
                        hasattr(base, method_name)
                        and getattr(base, method_name).__doc__ is not None
                    ):
                        method.__doc__ = getattr(base, method_name).__doc__
                        break

            # Inherit annotations if missing
            if not hasattr(method, "__annotations__") or not method.__annotations__:
                for base in cls.__mro__[1:]:
                    if method_name in base.__dict__ and hasattr(
                        base.__dict__[method_name], "__annotations__"
                    ):
                        method.__annotations__ = base.__dict__[
                            method_name
                        ].__annotations__
                        break


def map2array(fn: Callable, tree: Any, leaf_fn: Callable = None) -> Array:
    """
    Maps a function across a pytree, flattening it and turning it into an
    array.

    Parameters
    ----------
    fn : Callable
        The function to be mapped across the pytree.
    tree : Any
        The pytree to be mapped across.
    leaf_fn : Callable = None
        The function to be used to determine whether a leaf is reached.

    Returns
    -------
    array : Array
        The flattened array of the pytree.
    """
    if leaf_fn is not None:
        return np.array(jtu.flatten(jtu.map(fn, tree, is_leaf=leaf_fn))[0])
    else:
        return np.array(jtu.flatten(jtu.map(fn, tree))[0])


def list2dictionary(list_in: list, ordered: bool, allowed_types: tuple = ()) -> dict:
    """
    Converts some input list to a dictionary. The input list entries can either be
    objects, in which case the keys are taken as the class name, else a (key, object)
    tuple can be used to specify a key.

    If any duplicate keys are found, the key is appended with an index value. i.e. if
    two of the list entries have the same key 'layer', they will be assigned 'layer_0'
    and 'layer_1' respectively, depending on their input order in the list.

    Parameters
    ----------
    list_in : list
        The list of objects to be converted into a dictionary.
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
            raise TypeError(f"Item {name} is not an allowed type, got " f"{type(item)}")

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
            raise ValueError(f"Names cannot contain spaces, got {names[i]}")

        # Add to dict
        if isinstance(list_in[i], tuple):
            # item = list_in[i][0]
            item = list_in[i][1]
        else:
            item = list_in[i]
        dict_out[names[i]] = item
    return dict_out


def insert_layer(
    layers: dict,
    layer: Any,
    index: int,
    allowed_type: Any,
) -> dict:
    """
    Inserts a layer into a dictionary of layers at a specified index. This function
    calls the list2dictionary function to ensure all keys remain unique. Note that this
    can result in some keys being modified if they are duplicates. The input 'layer'
    can be a tuple of (key, layer) to specify a key, else the key is taken as the
    class name of the layer.

    Parameters
    ----------
    layers : dict
        The dictionary of layers to insert the layer into.
    layer : Any
        The layer to be inserted.
    index : int
        The index at which to insert the layer.
    allowed_type : Any
        The type of layer to be inserted. Used for type-checking.

    Returns
    -------
    layers : dict
        The updated dictionary of layers.
    """
    layers_list = list(zip(layers.keys(), layers.values()))
    layers_list.insert(index, layer)
    return list2dictionary(layers_list, True, allowed_type)


def remove_layer(layers: dict, key: str) -> dict:
    """
    Removes a layer from a dictionary of layers, specified by its key.

    Parameters
    ----------
    layers : dict
        The dictionary of layers to remove the layer from.
    key : str
        The key of the layer to be removed.

    Returns
    -------
    layers : dict
        The updated dictionary of layers.
    """
    layers.pop(key)
    return layers
