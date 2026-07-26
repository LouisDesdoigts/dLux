from collections import Counter, defaultdict, OrderedDict
from numbers import Integral
from typing import Any, Callable
import jax.numpy as np
import jax.tree as jtu
from jax import Array

__all__ = [
    "map2array",
    "list2dictionary",
    "insert_layer",
    "remove_layer",
    "imshow_extent",
    "missing_attribute_error",
    "from_complex",
    "as_size",
    "as_axis",
]


def as_size(value, ndim=None, name="size"):
    """Return positive integer sizes as a static tuple."""
    if isinstance(value, Integral):
        values = (int(value),)
    elif isinstance(value, (tuple, list)) and value:
        values = tuple(value)
    else:
        raise TypeError(f"{name} must be an integer or non-empty tuple.")
    if not all(isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain integers.")
    if any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive values.")
    values = tuple(map(int, values))
    if ndim is None or len(values) == ndim:
        return values
    if len(values) == 1:
        return values * ndim
    raise ValueError(f"{name} cannot be broadcast to {ndim} dimensions.")


def as_axis(value, ndim=None, name="axis"):
    """Return scalar or per-axis values with an explicit trailing axis."""
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    size = 1 if value.ndim == 0 else value.shape[-1]
    ndim = size if ndim is None else max(int(ndim), 1)
    if value.ndim == 0:
        return np.broadcast_to(value, (ndim,))
    if value.shape[-1] == ndim:
        return value
    if value.shape[-1] == 1:
        return np.broadcast_to(value, value.shape[:-1] + (ndim,))
    raise ValueError(f"{name} must be scalar or have one value per axis.")


def reexport(modules: tuple[object, ...], namespace: dict[str, object]) -> list[str]:
    """Re-export the public symbols from a collection of modules."""
    exported = []
    seen = set()
    for module in modules:
        for name in getattr(module, "__all__", ()):
            namespace[name] = getattr(module, name)
            if name not in seen:
                exported.append(name)
                seen.add(name)
    return exported


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
    entries = [
        item if isinstance(item, tuple) else (type(item).__name__, item)
        for item in list_in
    ]
    for name, item in entries:
        if allowed_types and not isinstance(item, allowed_types):
            raise TypeError(f"Item {name} is not an allowed type, got {type(item)}")
        if " " in name:
            raise ValueError(f"Names cannot contain spaces, got {name}")

    counts, seen = Counter(name for name, _ in entries), defaultdict(int)
    dict_out = OrderedDict() if ordered else {}
    for name, item in entries:
        key = f"{name}_{seen[name]}" if counts[name] > 1 else name
        seen[name] += 1
        dict_out[key] = item
    return dict_out


def insert_layer(layers: dict, layer: Any, index: int, allowed_type: Any) -> dict:
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
    layers = layers.copy()
    layers.pop(key)
    return layers


def imshow_extent(size: float) -> Array:
    """
    Returns a square imshow extent in [xmin, xmax, ymin, ymax] order.

    Parameters
    ----------
    size : float
        The total width of the image in the relevant physical units.

    Returns
    -------
    extent : Array
        The extent array to pass directly to matplotlib imshow.
    """
    half_size = np.asarray(size, dtype=float) / 2
    return np.array([-half_size, half_size, -half_size, half_size])


def missing_attribute_error(
    owner: Any, key: str, valid_attrs: list[str] = None, hint: str = None
) -> AttributeError:
    """
    Builds a consistent AttributeError message for missing attributes.

    Parameters
    ----------
    owner : Any
        The object raising the error.
    key : str
        The missing attribute name.
    valid_attrs : list[str] = None
        Optional list of valid attribute names to surface.
    hint : str = None
        Optional additional guidance appended to the message.

    Returns
    -------
    error : AttributeError
        The formatted AttributeError instance.
    """
    message = f"{owner.__class__.__name__} has no attribute '{key}'."
    if valid_attrs:
        attrs = sorted(valid_attrs)
        attrs_str = ", ".join(attrs[:6])
        ellipsis = "..." if len(attrs) > 6 else ""
        message += f" Valid attributes: {attrs_str}{ellipsis}"
    if hint:
        message += f" {hint}"
    return AttributeError(message)


def from_complex(array: Array, complex: bool = True) -> Array:
    """
    Maps a complex array to a 2-channel representation (real/imag or amplitude/phase).

    Parameters
    ----------
    array : Array
        The input complex array.
    complex : bool = True
        If True, map to real and imaginary components. If False, map to amplitude and
        phase.

    Returns
    -------
    vals : Array
        The 2-channel representation of the input array.
    return_fn : Callable
        Function to reconstruct the original complex array from `vals`.
    """
    if complex:
        vals = np.array([array.real, array.imag])
        return_fn = lambda x: x[0] + 1j * x[1]
    else:
        vals = np.array([np.abs(array), np.angle(array)])
        return_fn = lambda x: x[0] * np.exp(1j * x[1])
    return vals, return_fn
