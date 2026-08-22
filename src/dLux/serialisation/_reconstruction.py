"""Constructor-free templates built from stored object definitions."""

from collections import OrderedDict
from dataclasses import fields

import equinox as eqx

from ._leaves import _NOT_PAYLOAD, _payload_placeholder
from ._types import _resolve_type, _type_identifier


def _decode_literal(definition, custom_types, path):
    """Decode one retained or static literal node."""
    if definition is None or type(definition) in (bool, int, float, str):
        return definition
    if not isinstance(definition, dict):
        return _NOT_PAYLOAD

    kind = definition.get("kind")
    try:
        if kind == "complex":
            return complex(*definition["value"])
        if kind == "range":
            values = (definition["start"], definition["stop"], definition["step"])
            if not all(type(value) is int for value in values):
                raise TypeError
            return range(*values)
        if kind == "slice":
            return slice(
                _build_template(definition["start"], custom_types, f"{path}.start"),
                _build_template(definition["stop"], custom_types, f"{path}.stop"),
                _build_template(definition["step"], custom_types, f"{path}.step"),
            )
        if kind == "type":
            return _resolve_type(definition["type"], custom_types, path)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {kind!r} literal definition at {path}.") from error
    return _NOT_PAYLOAD


def _build_template(definition, custom_types, path="root"):
    """Recursively build an Equinox deserialisation template."""
    literal = _decode_literal(definition, custom_types, path)
    if literal is not _NOT_PAYLOAD:
        return literal
    if not isinstance(definition, dict):
        raise ValueError(f"Object definition node at {path} must be a mapping.")
    kind = definition.get("kind")

    placeholder = _payload_placeholder(definition, path)
    if placeholder is not _NOT_PAYLOAD:
        return placeholder

    if kind == "module":
        cls = _resolve_type(definition.get("type"), custom_types, path)
        if not issubclass(cls, eqx.Module):
            raise ValueError(
                f"Type {_type_identifier(cls)!r} at {path} is not a Module."
            )
        stored_fields = definition.get("fields")
        if not isinstance(stored_fields, list):
            raise ValueError(f"Module fields at {path} must be a list.")
        declared_fields = fields(cls)
        stored_schema = [
            (field.get("name"), field.get("static"))
            for field in stored_fields
            if isinstance(field, dict)
        ]
        declared_schema = [
            (field.name, bool(field.metadata.get("static", False)))
            for field in declared_fields
        ]
        if stored_schema != declared_schema:
            raise ValueError(f"The installed class schema for {path} has changed.")

        value = object.__new__(cls)
        for field in stored_fields:
            name = field["name"]
            child = _build_template(field["value"], custom_types, f"{path}.{name}")
            object.__setattr__(value, name, child)
        return value

    if kind in ("list", "tuple"):
        items = definition.get("items")
        if not isinstance(items, list):
            raise ValueError(f"Sequence items at {path} must be a list.")
        values = [
            _build_template(item, custom_types, f"{path}[{index}]")
            for index, item in enumerate(items)
        ]
        if kind == "list":
            return values
        return tuple(values)

    if kind == "mapping":
        cls = _resolve_type(definition.get("type"), custom_types, path)
        if cls not in (dict, OrderedDict):
            raise ValueError(
                f"Unsupported mapping type {_type_identifier(cls)!r} at {path}."
            )
        entries = definition.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Mapping entries at {path} must be a list.")
        values = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid mapping entry at {path}[{index}].")
            key = _build_template(
                entry.get("key"), custom_types, f"{path}.keys[{index}]"
            )
            item = _build_template(
                entry.get("value"), custom_types, f"{path}.values[{index}]"
            )
            values.append((key, item))
        return cls(values)

    raise ValueError(f"Unsupported object definition kind {kind!r} at {path}.")
