"""Validate and standardise common dLux input values."""

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from difflib import get_close_matches
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
    "resolve_attr",
    "from_complex",
    "as_size",
    "as_axis",
    "to_value",
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


def to_value(value, dtype=float, optional=False, types=None):
    """Preserve allowed object types or convert a value to an array.

    ``None`` is accepted only when ``optional`` is true. Instances of
    ``types`` are returned unchanged; every other value is passed to
    ``jax.numpy.asarray`` with the requested dtype.
    """
    if value is None:
        if optional:
            return None
        raise TypeError("value cannot be None.")
    if types is not None and isinstance(value, types):
        return value
    return np.asarray(value, dtype=dtype)


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
    """Maps a function across a pytree, flattening it and turning it into an
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
    """Converts some input list to a dictionary. The input list entries can either be
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
    """Inserts a layer into a dictionary of layers at a specified index. This function
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
    """Removes a layer from a dictionary of layers, specified by its key.

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
    """Returns a square imshow extent in [xmin, xmax, ymin, ymax] order.

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
    """Builds a consistent AttributeError message for missing attributes.

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
        attrs = sorted(set(valid_attrs))
        matches = get_close_matches(key, attrs, n=3, cutoff=0.6)
        if matches:
            message += f" Did you mean {', '.join(repr(match) for match in matches)}?"
    if hint:
        message += f" {hint}"
    return AttributeError(message)


def _raised_attrs(children, seen=None):
    """Collect public field and mapping names reachable through child objects."""
    seen = set() if seen is None else seen
    attrs = []

    for child in children:
        if child is None or id(child) in seen:
            continue
        seen.add(id(child))

        if isinstance(child, Mapping):
            attrs.extend(str(key) for key in child)
            attrs.extend(_raised_attrs(child.values(), seen))
            continue
        if isinstance(child, (tuple, list)):
            attrs.extend(_raised_attrs(child, seen))
            continue

        fields = getattr(type(child), "__dataclass_fields__", {})
        names = [name for name in fields if not name.startswith("_")]
        attrs.extend(names)
        values = [getattr(child, name) for name in names]
        attrs.extend(_raised_attrs(values, seen))
    return attrs


def resolve_attr(owner: Any, key: str, *children: Any) -> Any:
    """Raise a named child or first matching descendant attribute.

    Mapping keys take precedence over attributes raised from their values. Child
    objects are searched in the supplied order, allowing progressively qualified
    Zodiax paths to resolve ambiguity naturally.
    """
    # Raise named mapping children before searching their values
    values = []
    for child in children:
        if child is None:
            continue
        if isinstance(child, Mapping):
            if key in child:
                return child[key]
            values.extend(child.values())
        elif isinstance(child, (tuple, list)):
            values.extend(child)
        else:
            values.append(child)

    # Return the first descendant match in stable child order
    for value in values:
        try:
            return getattr(value, key)
        except AttributeError:
            pass

    valid = _raised_attrs((owner, *children))
    raise missing_attribute_error(owner, key, valid)


def from_complex(array: Array, complex: bool = True) -> Array:
    """Map a complex array to a two-channel representation.

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
