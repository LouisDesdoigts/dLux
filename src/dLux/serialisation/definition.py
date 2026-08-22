"""Automatic, inspectable definitions of serialisable Equinox models."""

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
import json
import math
from types import MemberDescriptorType

import equinox as eqx

from ._leaves import _NOT_PAYLOAD, _payload_definition
from ._reconstruction import _build_template
from ._schema import _MAX_INTEGER, _MIN_INTEGER, _validate_definition
from ._types import _type_identifier

__all__ = ["ObjectDefinition"]


def _extra_state_names(value, declared_fields):
    """Return populated instance state not declared as dataclass fields."""
    field_names = {field.name for field in declared_fields}
    try:
        stored_names = set(object.__getattribute__(value, "__dict__"))
    except AttributeError:
        stored_names = set()
    extra_names = stored_names.difference(field_names)

    for cls in type(value).__mro__:
        namespace = type.__getattribute__(cls, "__dict__")
        for name, descriptor in namespace.items():
            if name in field_names or not isinstance(descriptor, MemberDescriptorType):
                continue
            try:
                object.__getattribute__(value, name)
            except AttributeError:
                continue
            extra_names.add(name)
    return extra_names


def _literal_definition(value, path):
    """Encode one retained or static Python value as typed JSON."""
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        if not _MIN_INTEGER <= value <= _MAX_INTEGER:
            raise TypeError(f"{path} must fit in a signed 64-bit Python integer.")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(
                f"{path} is a non-finite Python float; use a JAX array to preserve it."
            )
        return value
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise TypeError(
                f"{path} is a non-finite Python complex value; use a JAX array to "
                "preserve it."
            )
        return {
            "kind": "complex",
            "value": [value.real, value.imag],
        }
    if type(value) is str:
        return value
    if isinstance(value, range):
        values = (value.start, value.stop, value.step)
        for name, item in zip(("start", "stop", "step"), values):
            if not _MIN_INTEGER <= item <= _MAX_INTEGER:
                raise TypeError(f"{path}.{name} must fit in a signed 64-bit integer.")
        return {
            "kind": "range",
            "start": value.start,
            "stop": value.stop,
            "step": value.step,
        }
    if isinstance(value, slice):
        return {
            "kind": "slice",
            "start": _describe(value.start, static=True, template=False, path=path),
            "stop": _describe(value.stop, static=True, template=False, path=path),
            "step": _describe(value.step, static=True, template=False, path=path),
        }
    if isinstance(value, type):
        return {"kind": "type", "type": _type_identifier(value)}
    if callable(value):
        raise TypeError(f"{path} contains an unsupported callable.")
    if eqx.is_array(value):
        raise TypeError(f"{path} contains an unsupported static array.")
    raise TypeError(f"{path} contains unsupported {type(value).__name__} metadata.")


def _mapping_key_definition(value, path):
    """Encode an inert mapping key without invoking custom object methods."""
    if value is None or type(value) in (bool, int, str):
        return _literal_definition(value, path)
    if type(value) is float and math.isfinite(value):
        return _literal_definition(value, path)
    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _mapping_key_definition(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    raise TypeError(f"{path} has unsupported mapping key type {type(value).__name__}.")


def _describe(value, *, static, template, path):
    """Recursively describe one realised value using public object state."""
    if isinstance(value, eqx.Module):
        declared_fields = fields(value)
        extra_names = _extra_state_names(value, declared_fields)
        if extra_names:
            names = ", ".join(sorted(extra_names))
            raise TypeError(f"{path} contains non-field state: {names}.")

        described_fields = []
        for field in declared_fields:
            field_path = f"{path}.{field.name}"
            try:
                field_value = getattr(value, field.name)
            except AttributeError as error:
                raise TypeError(f"{field_path} is not initialised.") from error
            field_static = bool(field.metadata.get("static", False))
            described_fields.append(
                {
                    "name": field.name,
                    "static": field_static,
                    "value": _describe(
                        field_value,
                        static=static or field_static,
                        template=template,
                        path=field_path,
                    ),
                }
            )
        return {
            "kind": "module",
            "type": _type_identifier(type(value)),
            "fields": described_fields,
        }

    if isinstance(value, Mapping):
        if type(value) not in (dict, OrderedDict):
            raise TypeError(
                f"{path} has unsupported mapping type {type(value).__name__}."
            )
        items = [
            (key, item, _mapping_key_definition(key, f"{path}.keys[{index}]"))
            for index, (key, item) in enumerate(value.items())
        ]
        if type(value) is dict:
            try:
                items = sorted(items, key=lambda entry: entry[0])
            except TypeError as error:
                raise TypeError(
                    f"{path} has plain-dict keys without a stable JAX ordering."
                ) from error
        entries = []
        for index, (_, item, key_definition) in enumerate(items):
            entries.append(
                {
                    "key": key_definition,
                    "value": _describe(
                        item,
                        static=static,
                        template=template,
                        path=f"{path}.values[{index}]",
                    ),
                }
            )
        return {
            "kind": "mapping",
            "type": _type_identifier(type(value)),
            "entries": entries,
        }

    if type(value) is tuple:
        return {
            "kind": "tuple",
            "items": [
                _describe(
                    item,
                    static=static,
                    template=template,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        }

    if isinstance(value, tuple):
        raise TypeError(f"{path} has unsupported tuple type {type(value).__name__}.")

    if type(value) is list:
        return {
            "kind": "list",
            "items": [
                _describe(
                    item,
                    static=static,
                    template=template,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        }

    if not static:
        definition = _payload_definition(value, template=template, path=path)
        if definition is not _NOT_PAYLOAD:
            return definition
    return _literal_definition(value, path)


def _describe_root(value, *, template):
    """Describe a root while translating recursive object graphs cleanly."""
    try:
        return _describe(value, static=False, template=template, path="root")
    except RecursionError as error:
        raise TypeError("The object graph is cyclic or too deeply nested.") from error


def _first_difference(saved, current, path="root"):
    """Return the first differing JSON path and values."""
    if type(saved) is not type(current):
        return path, saved, current
    if isinstance(saved, dict):
        if saved.keys() != current.keys():
            return path, sorted(saved), sorted(current)
        for key in saved:
            difference = _first_difference(saved[key], current[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(saved, list):
        if len(saved) != len(current):
            return path, f"{len(saved)} items", f"{len(current)} items"
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            difference = _first_difference(
                saved_item,
                current_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if type(saved) is float:
        if saved.hex() != current.hex():
            return path, saved, current
        return None
    if saved != current:
        return path, saved, current
    return None


def _structural_view(definition):
    """Remove values restored from the archive while retaining topology."""
    if definition is None:
        return {"kind": "literal", "type": "none"}
    if type(definition) in (bool, int, float, str):
        return {"kind": "literal", "type": type(definition).__name__}

    kind = definition["kind"]
    if kind == "module":
        return {
            "kind": kind,
            "type": definition["type"],
            "fields": [
                {
                    "name": field["name"],
                    "static": field["static"],
                    "value": _structural_view(field["value"]),
                }
                for field in definition["fields"]
            ],
        }
    if kind in ("list", "tuple"):
        return {
            "kind": kind,
            "items": [_structural_view(item) for item in definition["items"]],
        }
    if kind == "mapping":
        return {
            "kind": kind,
            "type": definition["type"],
            "entries": [
                {
                    "key": entry["key"],
                    "value": _structural_view(entry["value"]),
                }
                for entry in definition["entries"]
            ],
        }
    if kind in ("jax_array", "jax_prng_key"):
        return definition
    if kind == "slice":
        return {
            "kind": kind,
            "start": _structural_view(definition["start"]),
            "stop": _structural_view(definition["stop"]),
            "step": _structural_view(definition["step"]),
        }
    return {"kind": kind}


def _collect_types(value, registry=None, ancestors=None):
    """Collect concrete classes already present in a trusted template."""
    registry = {} if registry is None else registry
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, type):
        registry[_type_identifier(value)] = value
        return registry
    if not isinstance(value, (eqx.Module, Mapping, tuple, list)):
        return registry
    identity = id(value)
    if identity in ancestors:
        raise ValueError("like contains a cyclic object graph.")
    ancestors.add(identity)
    try:
        if isinstance(value, eqx.Module):
            identifier = _type_identifier(type(value))
            existing = registry.setdefault(identifier, type(value))
            if existing is not type(value):
                raise ValueError(f"like contains ambiguous type {identifier!r}.")
            children = [getattr(value, field.name) for field in fields(value)]
        elif isinstance(value, Mapping):
            children = [item for pair in value.items() for item in pair]
        else:
            children = value
        for child in children:
            _collect_types(child, registry, ancestors)
        return registry
    finally:
        ancestors.remove(identity)


class ObjectDefinition:
    """Automatically generated JSON definition of a serialisable object.

    The definition records module classes and fields, built-in container topology,
    Python scalar and literal values, static values, and JAX array metadata. Concrete
    JAX array values are deliberately absent and live in the archive payload.

    Definitions are generated from realised objects; downstream Equinox and Zodiax
    classes do not need to declare a separate schema. All instance state must be held
    in dataclass fields containing values supported by the generic definition format.

    Examples
    --------
    Inspect an optic definition and build its abstract loading template:

    ```python
    import jax.numpy as jnp

    import dLux as dl

    optic = dl.Optic(transmission=jnp.ones((2, 2)))
    definition = dl.ObjectDefinition.from_object(optic)
    print(definition)

    template = definition.build_template()
    ```
    """

    __slots__ = ("_root",)

    def __init__(self, definition):
        """Validate and retain an independent definition tree."""
        _validate_definition(definition)
        self._root = deepcopy(definition)

    @property
    def root(self):
        """Return an independent JSON-compatible definition tree."""
        return deepcopy(self._root)

    @classmethod
    def from_object(cls, obj):
        """Generate a definition from one realised object."""
        return cls(_describe_root(obj, template=False))

    @classmethod
    def from_dict(cls, definition):
        """Construct a definition from decoded JSON data."""
        return cls(definition)

    def to_dict(self):
        """Return an independent JSON-compatible representation."""
        return deepcopy(self._root)

    def to_json(self, *, indent=2):
        """Return a human-readable JSON representation."""
        return json.dumps(self._root, indent=indent, sort_keys=False, allow_nan=False)

    def _difference(self, obj, *, structure_only):
        """Return the first difference from an object definition."""
        current = _describe_root(obj, template=True)
        saved = _structural_view(self._root) if structure_only else self._root
        current = _structural_view(current) if structure_only else current
        return _first_difference(saved, current)

    def _validate_like(self, like) -> None:
        """Validate a structural template while allowing archived values to differ."""
        difference = self._difference(like, structure_only=True)
        if difference is not None:
            path, saved, actual = difference
            raise ValueError(
                f"Object structure mismatch at {path}: saved {saved!r}, "
                f"received {actual!r}."
            )

    def validate(self, obj) -> None:
        """Raise if an object's generated definition differs from this one.

        Every JSON-held value is compared. JAX array values cannot be compared
        because a definition records only their shape, dtype, and weak-type metadata.
        """
        difference = self._difference(obj, structure_only=False)
        if difference is not None:
            path, saved, actual = difference
            raise ValueError(
                f"Object definition mismatch at {path}: saved {saved!r}, "
                f"received {actual!r}."
            )

    def build_template(self, *, like=None, custom_types=None):
        """Build a deserialisation template without running module constructors.

        Stored literals are realised exactly, while JAX arrays become
        `jax.ShapeDtypeStruct` placeholders. If supplied, ``like`` resolves classes
        and checks structure; its stored values are not copied into the template.
        """
        if custom_types is not None and not isinstance(custom_types, Mapping):
            raise TypeError("custom_types must be a mapping from identifiers to types.")
        if like is not None:
            if custom_types is not None:
                raise TypeError("custom_types cannot be supplied together with like.")
            self._validate_like(like)
            custom_types = _collect_types(like)
        obj = _build_template(self._root, custom_types)
        self.validate(obj)
        return obj

    def __str__(self):
        """Render this definition as readable JSON."""
        return self.to_json()
