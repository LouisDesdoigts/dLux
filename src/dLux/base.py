"""Shared dLux object behaviour built on Zodiax pytrees."""

from collections.abc import Mapping

import zodiax as zdx

import dLux.utils as dlu
from dLux.serialisation import load as _deserialise
from dLux.serialisation import save as _serialise

__all__ = ["Base"]


def _paths(parameters):
    """Yield path strings from a nested Zodiax parameter selector."""
    if isinstance(parameters, str):
        yield parameters
        return
    if isinstance(parameters, Mapping):
        parameters = parameters.keys()
    for parameter in parameters:
        yield from _paths(parameter)


def _resolve_path(obj, path):
    """Replay a parameter path through standard attribute resolution."""
    for key in path.split("."):
        if isinstance(obj, Mapping):
            if key not in obj:
                raise dlu.missing_attribute_error(obj, key, list(obj))
            obj = obj[key]
        elif isinstance(obj, (tuple, list)):
            obj = obj[int(key)]
        else:
            obj = getattr(obj, key)
    return obj


class Base(zdx.Base):
    """Base object for immutable dLux models with raised parameter paths.

    dLux objects are Equinox pytrees extended with Zodiax path operations. Nested
    public leaves can therefore be inspected and updated from a containing source,
    system, or layer using concise attribute paths. Failed attribute and path lookups
    share the same diagnostic behaviour.
    """

    def __getattr__(self, key):
        """Raise the first matching attribute from stored child objects."""
        fields = getattr(type(self), "__dataclass_fields__", {})
        children = []
        for name in fields:
            try:
                children.append(object.__getattribute__(self, name))
            except AttributeError:
                pass
        return dlu.resolve_attr(self, key, *children)

    def save(self, file_or_path) -> None:
        """Save this object to a validated dLux archive.

        Parameters
        ----------
        file_or_path : str, pathlib.Path, or binary file
            Destination for the archive. Paths without a suffix receive ``.dlux``.
            Binary files remain open after saving.

        Raises
        ------
        TypeError
            If any leaf or static value cannot be represented losslessly.

        Notes
        -----
        The archive records a generated JSON object definition alongside the JAX
        array payload, including diagnostic Python and package versions. Python
        numerical values, literals, static values, and array metadata live in the
        definition so `dLux.load` can rebuild a template without a caller-supplied
        object.

        NumPy array and scalar leaves are intentionally outside this initial
        contract. Callables, opaque objects, weakly typed JAX arrays, and unsupported
        JAX dtypes cannot be saved losslessly. Python integers must fit in signed 64
        bits, and non-finite Python float and complex values are unsupported.
        """
        _serialise(file_or_path, self)

    def load(self, file_or_path):
        """Load a dLux archive using this object as its structural template.

        Parameters
        ----------
        file_or_path : str, pathlib.Path, or binary file
            Archive to read. Paths without a suffix receive ``.dlux``. Binary files
            must be readable and seekable and remain open after loading.

        Returns
        -------
        object : Base
            Reconstructed object whose class and structure match this template. The
            template itself is not modified, and archived values replace its values.

        Notes
        -----
        This is equivalent to ``dLux.load(file_or_path, like=self)`` and is useful
        for local or custom classes that cannot be resolved from an imported module.
        """
        return _deserialise(file_or_path, like=self)

    def _check_paths(self, parameters, updates=None):
        """Replay failed paths to recover their raised attribute errors."""
        parameters = updates if parameters is None else parameters
        for path in _paths(parameters):
            _resolve_path(self, path)

    def get(self, parameters, as_dict=False, to_array=True):
        """Get nested parameters while preserving raised-attribute diagnostics.

        Parameters
        ----------
        parameters : str or sequence[str]
            Zodiax paths, including attributes raised from nested dLux objects.
        as_dict : bool
            Return a path-keyed mapping rather than values in selection order.
        to_array : bool
            Stack compatible selected values using the Zodiax convention.
        """
        try:
            return super().get(parameters, as_dict, to_array)
        except KeyError:
            self._check_paths(parameters)
            raise

    def set(self, parameters=None, values=None, /, **updates):
        """Return a copy with selected parameter paths replaced.

        Paths may be supplied through the Zodiax ``parameters, values`` form or as
        keyword updates. Raised dLux attributes are resolved consistently with direct
        attribute access; the original object is not mutated.
        """
        try:
            return super().set(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def add(self, parameters=None, values=None, /, **updates):
        """Return a copy after adding values to selected parameter paths.

        Accepts the same path selection forms as `set`; the original object is not
        mutated and raised dLux attributes retain their normal diagnostics.
        """
        try:
            return super().add(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def multiply(self, parameters=None, values=None, /, **updates):
        """Return a copy after multiplying selected parameter paths by values.

        Accepts the same path selection forms as `set`; the original object is not
        mutated and raised dLux attributes retain their normal diagnostics.
        """
        try:
            return super().multiply(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def divide(self, parameters=None, values=None, /, **updates):
        """Return a copy after dividing selected parameter paths by values.

        Accepts the same path selection forms as `set`; the original object is not
        mutated and raised dLux attributes retain their normal diagnostics.
        """
        try:
            return super().divide(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def power(self, parameters=None, values=None, /, **updates):
        """Return a copy after raising selected parameter paths to powers.

        Accepts the same path selection forms as `set`; the original object is not
        mutated and raised dLux attributes retain their normal diagnostics.
        """
        try:
            return super().power(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def min(self, parameters=None, values=None, /, **updates):
        """Return a copy after applying elementwise upper limits to parameters.

        This delegates to the Zodiax ``min`` update: selected values become
        ``minimum(current, update)``. The original object is not mutated.
        """
        try:
            return super().min(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def max(self, parameters=None, values=None, /, **updates):
        """Return a copy after applying elementwise lower limits to parameters.

        This delegates to the Zodiax ``max`` update: selected values become
        ``maximum(current, update)``. The original object is not mutated.
        """
        try:
            return super().max(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise
