from collections import OrderedDict
from typing import Any, Callable
import jax.numpy as np
import jax.tree as jtu
from jax import vmap, Array
from dLux.utils.interpolation import scale


__all__ = [
    "map2array",
    "list2dictionary",
    "insert_layer",
    "remove_layer",
    "scale_layer",
]


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


def list2dictionary(
    list_in: list, ordered: bool, allowed_types: tuple = ()
) -> dict:
    """
    Converts some input list to a dictionary. The input list entries can either be
    objects, in which case the keys are taken as the class name, else a (key, object)
    tuple can be used to specify a key.

    If any duplicate keys are found, the key is appended with an index value. ie if two
    of the list entries have the same key 'layer', they will be assigned 'layer_0' and
    layer_1' respectively, depending on their input order in the list.

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


def insert_layer(layers: dict, layer: Any, index: int, type: Any) -> dict:
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
    type : Any
        The type of layer to be inserted. Used for type-checking.

    Returns
    -------
    layers : dict
        The updated dictionary of layers.
    """
    layers_list = list(zip(layers.keys(), layers.values()))
    layers_list.insert(index, layer)
    return list2dictionary(layers_list, True, type)


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


def scale_layer(layer_in, pixel_scale_in, pixel_scale_out, npix_out):
    """
    Rescale an optical layer to a new pixel scale and array size.

    This helper function supports `TransmissiveLayer`, `AberratedLayer`,
    `BasisLayer`, and `Optic` objects. It rescales any defined 2D (or 3D
    for `BasisLayer`) arrays—such as transmission maps, OPDs, phases,
    or bases—using the interpolation routine from
    `dLux.utils.interpolation.scale()`.

    Parameters
    ----------
    layer_in : TransmissiveLayer, AberratedLayer, BasisLayer, or Optic
        The optical layer instance to be rescaled.
    pixel_scale_in : float
        The current pixel scale of the input layer.
    pixel_scale_out : float
        The target pixel scale for the rescaled layer.
    npix_out : int
        The desired output array size.

    Returns
    -------
    layer_out : same type as `layer_in`
        A new instance of the same layer type with rescaled internal
        arrays and updated resolution.

    Raises
    ------
    ValueError
        If an array on the layer has an unexpected number of dimensions.
    TypeError
        If the provided layer type is not one of the supported classes.

    Notes
    -----
    - Scaling is applied only to defined fields (e.g., `opd`, `phase`,
      `transmission`, or `basis`).
    - Unsupported layer types will raise a `TypeError`.
    - This function imports layer classes locally to avoid circular
      dependencies within the `dLux` module.
    """
    from dLux.layers import (
        TransmissiveLayer,
        AberratedLayer,
        BasisLayer,
        Optic,
    )

    scale_factor = pixel_scale_out / pixel_scale_in
    new_layer = layer_in

    # --- Transmissive Component ---
    if isinstance(layer_in, TransmissiveLayer):
        transmission = layer_in.transmission
        if transmission is not None:
            if transmission.ndim != 2:
                raise ValueError(
                    f"TransmissiveLayer transmission must be 2D "
                    f"(got shape {transmission.shape})"
                )
            scaled = scale(transmission, npix_out, scale_factor)
            new_layer = new_layer.set("transmission", scaled)

    # --- Aberrated Component ---
    if isinstance(layer_in, AberratedLayer):
        if layer_in.opd is not None:
            if layer_in.opd.ndim != 2:
                raise ValueError(
                    f"AberratedLayer opd must be 2D, got shape {layer_in.opd.shape}"
                )
            scaled_opd = scale(layer_in.opd, npix_out, scale_factor)
            new_layer = new_layer.set("opd", scaled_opd)

        if layer_in.phase is not None:
            if layer_in.phase.ndim != 2:
                raise ValueError(
                    f"AberratedLayer phase must be 2D, got shape {layer_in.phase.shape}"
                )
            scaled_phase = scale(layer_in.phase, npix_out, scale_factor)
            new_layer = new_layer.set("phase", scaled_phase)

    # --- Basis Component ---
    if isinstance(layer_in, BasisLayer):
        basis = layer_in.basis
        if basis is not None:
            if basis.ndim != 3:
                raise ValueError(
                    "BasisLayer basis must be a 3D array [n_modes, H, W]"
                )
            scale_fn = vmap(scale, (0, None, None))
            scaled_basis = scale_fn(basis, npix_out, scale_factor)
            new_layer = new_layer.set("basis", scaled_basis)

    if isinstance(layer_in, Optic):
        # Optic is an object with optional opd/phase/transmission.
        # We can safely recurse scaling through any defined arrays.
        fields = {}
        if layer_in.opd is not None:
            fields["opd"] = scale(layer_in.opd, npix_out, scale_factor)
        if layer_in.phase is not None:
            fields["phase"] = scale(layer_in.phase, npix_out, scale_factor)
        if layer_in.transmission is not None:
            fields["transmission"] = scale(
                layer_in.transmission, npix_out, scale_factor
            )
        return layer_in.set(list(fields.keys()), list(fields.values()))

    # --- Final Return ---
    if new_layer is not layer_in:
        return new_layer

    # If no recognized layers were scaled
    raise TypeError(f"Unsupported layer type: {type(layer_in)}")
