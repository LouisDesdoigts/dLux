"""Shared layer contract, named sequences and independent reductions."""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter, OrderedDict, defaultdict
from typing import Any
from jax import Array
import zodiax as zdx
import equinox as eqx
import jax.numpy as np

__all__ = [
    "list2dictionary",
    "insert_layer",
    "remove_layer",
    "BaseLayer",
    "Sequential",
    "Series",
    "Combine",
]


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
    # Assign candidate names and validate entries before generating unique suffixes.
    entries = [
        item if isinstance(item, tuple) else (type(item).__name__, item)
        for item in list_in
    ]
    for name, item in entries:
        if allowed_types and not isinstance(item, allowed_types):
            raise TypeError(f"Item {name} is not an allowed type, got {type(item)}")
        if " " in name:
            raise ValueError(f"Names cannot contain spaces, got {name}")

    # Reserve supplied names so generated suffixes cannot overwrite another entry.
    counts = Counter(name for name, _ in entries)
    used_names = set(counts)
    next_index = defaultdict(int)
    dict_out = OrderedDict() if ordered else {}
    for name, item in entries:
        key = name
        if counts[name] > 1:
            # Skip supplied names and names already assigned to earlier entries.
            key = f"{name}_{next_index[name]}"
            while key in used_names:
                next_index[name] += 1
                key = f"{name}_{next_index[name]}"
            next_index[name] += 1
            used_names.add(key)
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
    layers_list = list(layers.items())
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


class BaseLayer(zdx.Module):
    """An immutable operation on an array, field, or other compatible target.

    Implement ``apply(target, **context)`` and return the transformed target.
    Each subclass defines its input and output types and resolves its parameters
    with the context it needs. Calling a layer uses the same operation as apply.
    """

    @abstractmethod
    def apply(self, target: Any, **context) -> Any:
        """Transform a compatible target without mutating the input."""

    def __call__(self, target: Any, **context) -> Any:
        """Apply the layer to an explicit target."""
        return self.apply(target, **context)


class Sequential(BaseLayer):
    """Apply named compatible components in insertion order.

    Each component receives the preceding result and explicit caller context.
    Arrays, wavefronts and intensities use the same composition machinery; callers
    select components with compatible input/output contracts. Nested sequences keep
    their names and structure. Shared links are populated at the common owner scope
    before application, while each component owns its local expression resolution.
    Components are stored in ``layers`` so existing structural parameter paths remain
    valid; ``steps`` provides a generic read-only view. An empty sequence is the
    identity.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    import zodiax as zdx

    sequence = dl.Sequential({"scale": zdx.Map(s=2), "offset": zdx.Map(b=1)})
    print(sequence(np.array([1.0, 2.0])))
    result, stages = sequence.debug(np.array([1.0, 2.0]))
    ```
    """

    layers: OrderedDict

    def __init__(self, layers=(), layer_type=None, *, alias=None):
        """Construct a named sequence.

        Parameters
        ----------
        layers : mapping or sequence
            Callable modules, optionally named with ``(name, module)`` pairs.
        layer_type : type or None
            Optional construction-time restriction on every nested component.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter aliases.
        """
        self.layers = self._named_steps(layers, layer_type)
        self.alias = alias

    @property
    def steps(self):
        """Named components; stored as layers to preserve structural parameter paths."""
        return self.layers

    def apply(self, target, **context):
        """Pass the target through every component, returning the final result."""
        for step in self.populate().steps.values():
            target = step(target, **context)
        return target

    def debug(self, target, **context):
        """Return the final result and named intermediate values, including input."""
        outputs = {"input": target}
        for name, step in self.populate().steps.items():
            target = step(target, **context)
            outputs[name] = target
        return target, outputs

    def insert(self, step, index, step_type=None):
        """Return a copy with a compatible component inserted at ``index``."""
        if step_type is None:
            return self.insert_layer(step, index)
        return self.insert_layer(step, index, step_type)

    def remove(self, key):
        """Return a copy without the named component."""
        return self.remove_layer(key)

    def insert_layer(self, layer, index, layer_type=None):
        """Insert a named or unnamed component, optionally checking its type."""
        layers = insert_layer(self.layers, layer, index, ())
        return self.set(layers=self._named_steps(layers, layer_type))

    def remove_layer(self, key):
        """Return a copy with the named component removed."""
        return self.set(layers=remove_layer(self.layers, key))

    @staticmethod
    def _named_steps(steps, step_type=None):
        """Name components and validate their callable contract recursively."""
        entries = list(steps.items()) if isinstance(steps, dict) else steps
        named = list2dictionary(entries, True)
        for step in named.values():
            if isinstance(step, Sequential):
                Sequential._named_steps(step.steps, step_type)
            elif not isinstance(step, zdx.Module) or not callable(step):
                raise TypeError("Sequential steps must be callable Zodiax Modules.")
            elif step_type is not None and not isinstance(step, step_type):
                raise TypeError(f"Expected {step_type}, got {type(step)}.")
        return named


class Series(Sequential, zdx.Expression):
    """A numerical sequence that resolves when an input is supplied.

    ``series(array)`` applies the steps in order. ``series.resolve(target=array)``
    exposes the same calculation through Expression resolution. Without an input,
    resolution retains the series as a reusable operator; no input leaf is stored.
    Subclasses may supply their own starting value in evaluate.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    import zodiax as zdx

    series = dl.Series({"scale": zdx.Map(s=2), "offset": zdx.Map(b=1)})
    print(series.resolve(target=np.array([1.0, 2.0])))
    ```
    """

    # Inherited fields
    layers: OrderedDict

    def evaluate(self, *, target, **context) -> Array:
        """Resolve the supplied input, then pass it through the numerical steps."""
        target = zdx.resolve(target, **context)
        return self.apply(target, **context)

    def __call__(self, target=None, **context) -> Array | Series:
        """Apply to an explicit array, or resolve the subclass's starting value."""
        if target is None:
            return self.resolve(**context)
        return self.apply(target, **context)


class Combine(zdx.Expression):
    """Resolve named expressions in shared context and combine their arrays.

    Sum and product are arithmetic reductions. Union clips the sum to [0, 1],
    while intersection multiplies transmissions. Child outputs broadcast together.

    Examples
    --------
    ```python
    import dLux as dl

    grid = dl.Grid(8, diam=1.0, unit="m")
    annulus = dl.Combine([dl.Circle(1.0), dl.Complement(dl.Circle(0.3))],
                         operation="product")
    print(annulus.resolve(grid=grid).shape)
    ```
    """

    parametrics: dict[str, zdx.Expression]
    operation: str = eqx.field(static=True)

    def __init__(self, parametrics, operation="sum", *, alias=None):
        """Store named or unnamed expressions and their combination operation.

        Parameters
        ----------
        parametrics : mapping or sequence of zodiax.Expression
            At least one expression, optionally paired with a name.
        operation : str
            One of sum, product, union, or intersection.
        alias : mapping or sequence of pairs, optional
            Parameter paths into the stored expressions.
        """
        # Store a non-empty named set of expressions for independent evaluation.
        if isinstance(parametrics, dict):
            parametrics = list(parametrics.items())
        else:
            parametrics = list(parametrics)
        self.parametrics = list2dictionary(parametrics, True, zdx.Expression)
        if not self.parametrics:
            raise ValueError("Combine requires at least one parameterisation.")

        # Choose the reduction applied after all child expressions are evaluated.
        operation = str(operation).lower()
        if operation not in ("sum", "product", "union", "intersection"):
            raise ValueError("operation must be sum, product, union, or intersection.")
        self.operation = operation
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Resolve child values and combine broadcast-compatible arrays.

        Inputs required by each child are forwarded through context. Sum and product
        are arithmetic reductions; union clips a sum to [0, 1], while intersection
        multiplies transmissions. The output has the common broadcast shape. Children
        must produce arrays with scientifically compatible units for the operation.
        """
        parametrics = zdx.resolve(self.parametrics, **context)
        values = np.stack(np.broadcast_arrays(*parametrics.values()))
        if self.operation in ("product", "intersection"):
            return values.prod(0)
        output = values.sum(0)
        return np.clip(output, 0.0, 1.0) if self.operation == "union" else output
