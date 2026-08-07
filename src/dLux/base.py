"""Shared dLux object behaviour built on Zodiax pytrees."""

from collections.abc import Mapping

import zodiax as zdx

import dLux.utils as dlu

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
    """Unify raised attribute and Zodiax path error behaviour."""

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

    def _check_paths(self, parameters, updates=None):
        """Replay failed paths to recover their raised attribute errors."""
        parameters = updates if parameters is None else parameters
        for path in _paths(parameters):
            _resolve_path(self, path)

    def get(self, parameters, as_dict=False, to_array=True):
        """Get parameters while preserving raised-attribute diagnostics."""
        try:
            return super().get(parameters, as_dict, to_array)
        except KeyError:
            self._check_paths(parameters)
            raise

    def set(self, parameters=None, values=None, /, **updates):
        """Set parameters while preserving raised-attribute diagnostics."""
        try:
            return super().set(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def add(self, parameters=None, values=None, /, **updates):
        """Add to parameters while preserving raised-attribute diagnostics."""
        try:
            return super().add(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def multiply(self, parameters=None, values=None, /, **updates):
        """Multiply parameters while preserving raised-attribute diagnostics."""
        try:
            return super().multiply(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def divide(self, parameters=None, values=None, /, **updates):
        """Divide parameters while preserving raised-attribute diagnostics."""
        try:
            return super().divide(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def power(self, parameters=None, values=None, /, **updates):
        """Exponentiate parameters while preserving raised-attribute diagnostics."""
        try:
            return super().power(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def min(self, parameters=None, values=None, /, **updates):
        """Limit parameters below while preserving raised-attribute diagnostics."""
        try:
            return super().min(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise

    def max(self, parameters=None, values=None, /, **updates):
        """Limit parameters above while preserving raised-attribute diagnostics."""
        try:
            return super().max(parameters, values, **updates)
        except KeyError:
            self._check_paths(parameters, updates)
            raise
