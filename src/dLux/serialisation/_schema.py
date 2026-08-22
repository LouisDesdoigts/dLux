"""Pure validation of untrusted object-definition data."""

import math

from ._leaves import _SUPPORTED_DTYPES

_MAX_DEPTH = 128
_MAX_NODES = 100_000
_MAX_ITEMS = 100_000
_MAX_TYPE_IDENTIFIER = 1_024
_MAX_PRNG_WORDS = 1_000_000
_MIN_INTEGER = -(2**63)
_MAX_INTEGER = 2**63 - 1


def _require_keys(node, expected, path):
    """Require one definition node to have exactly the expected keys."""
    if set(node) != expected:
        raise ValueError(f"Invalid fields for {node.get('kind')!r} node at {path}.")


def _validate_identifier(identifier, path):
    """Validate the syntax of a recorded module-qualified type name."""
    if type(identifier) is not str or len(identifier) > _MAX_TYPE_IDENTIFIER:
        raise ValueError(f"Invalid type identifier at {path}.")
    module, separator, qualname = identifier.partition(":")
    module_parts = module.split(".")
    qualname_parts = qualname.split(".")
    if (
        separator != ":"
        or not module
        or not qualname
        or not all(part.isidentifier() for part in module_parts)
        or not all(part.isidentifier() or part == "<locals>" for part in qualname_parts)
    ):
        raise ValueError(f"Invalid type identifier {identifier!r} at {path}.")


def _validate_integer(value, path):
    """Validate one signed 64-bit Python integer literal."""
    if type(value) is not int or not _MIN_INTEGER <= value <= _MAX_INTEGER:
        raise ValueError(f"Integer literal at {path} must fit in signed 64 bits.")


def _validate_shape(shape, path):
    """Validate one bounded concrete JAX array shape."""
    if (
        not isinstance(shape, list)
        or len(shape) > 64
        or not all(type(size) is int and 0 <= size < 2**63 for size in shape)
    ):
        raise ValueError(f"Invalid JAX array shape at {path}.")


def _decode_mapping_key(node, path):
    """Decode a validated inert key for canonical-order checks."""
    if node is None or type(node) in (bool, int, float, str):
        return node
    if isinstance(node, dict) and node.get("kind") == "tuple":
        return tuple(
            _decode_mapping_key(item, f"{path}.items[{index}]")
            for index, item in enumerate(node["items"])
        )
    raise ValueError(f"Unsupported mapping key definition at {path}.")


def _validate_mapping_key(node, path, depth, state, ancestors):
    """Validate and decode one inert built-in mapping key."""
    _validate_node(node, path, depth, state, ancestors, retained=True)
    if node is None or type(node) in (bool, int, float, str):
        return _decode_mapping_key(node, path)
    if isinstance(node, dict) and node.get("kind") == "tuple":
        return _decode_mapping_key(node, path)
    raise ValueError(f"Unsupported mapping key definition at {path}.")


def _validate_node(node, path, depth, state, ancestors, *, retained):
    """Validate one recursively nested definition node."""
    if depth > _MAX_DEPTH:
        raise ValueError("The object definition exceeds the maximum nesting depth.")
    state[0] += 1
    if state[0] > _MAX_NODES:
        raise ValueError("The object definition contains too many nodes.")

    if node is None or type(node) in (bool, str):
        return
    if type(node) is int:
        _validate_integer(node, path)
        return
    if type(node) is float:
        if not math.isfinite(node):
            raise ValueError(f"Non-finite Python float at {path} is unsupported.")
        return
    if not isinstance(node, dict):
        raise ValueError(f"Object definition node at {path} must be a mapping.")
    identity = id(node)
    if identity in ancestors:
        raise ValueError(f"The object definition contains a cycle at {path}.")
    ancestors.add(identity)
    try:
        kind = node.get("kind")
        if type(kind) is not str:
            raise ValueError(f"Object definition node at {path} has no valid kind.")

        if kind == "module":
            _require_keys(node, {"kind", "type", "fields"}, path)
            _validate_identifier(node["type"], f"{path}.type")
            stored_fields = node["fields"]
            if not isinstance(stored_fields, list) or len(stored_fields) > _MAX_ITEMS:
                raise ValueError(f"Module fields at {path} must be a bounded list.")
            names = []
            for index, field in enumerate(stored_fields):
                field_path = f"{path}.fields[{index}]"
                if not isinstance(field, dict):
                    raise ValueError(f"Module field at {field_path} must be a mapping.")
                _require_keys(field, {"name", "static", "value"}, field_path)
                name = field["name"]
                static = field["static"]
                if type(name) is not str or not name.isidentifier():
                    raise ValueError(f"Invalid module field name at {field_path}.")
                if type(static) is not bool:
                    raise ValueError(f"Invalid static marker at {field_path}.")
                names.append(name)
                _validate_node(
                    field["value"],
                    f"{path}.{name}",
                    depth + 1,
                    state,
                    ancestors,
                    retained=retained or static,
                )
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate module fields at {path}.")
            return

        if kind in {"list", "tuple"}:
            _require_keys(node, {"kind", "items"}, path)
            items = node["items"]
            if not isinstance(items, list) or len(items) > _MAX_ITEMS:
                raise ValueError(f"Sequence items at {path} must be a bounded list.")
            for index, item in enumerate(items):
                _validate_node(
                    item,
                    f"{path}.items[{index}]",
                    depth + 1,
                    state,
                    ancestors,
                    retained=retained,
                )
            return

        if kind == "mapping":
            _require_keys(node, {"kind", "type", "entries"}, path)
            if node["type"] not in {"builtins:dict", "collections:OrderedDict"}:
                raise ValueError(f"Unsupported mapping type at {path}.")
            entries = node["entries"]
            if not isinstance(entries, list) or len(entries) > _MAX_ITEMS:
                raise ValueError(f"Mapping entries at {path} must be a bounded list.")
            keys = []
            for index, entry in enumerate(entries):
                entry_path = f"{path}.entries[{index}]"
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Mapping entry at {entry_path} must be a mapping."
                    )
                _require_keys(entry, {"key", "value"}, entry_path)
                keys.append(
                    _validate_mapping_key(
                        entry["key"],
                        f"{entry_path}.key",
                        depth + 1,
                        state,
                        ancestors,
                    )
                )
                _validate_node(
                    entry["value"],
                    f"{entry_path}.value",
                    depth + 1,
                    state,
                    ancestors,
                    retained=retained,
                )
            try:
                if len(set(keys)) != len(keys):
                    raise ValueError(f"Duplicate mapping keys at {path}.")
                if node["type"] == "builtins:dict" and keys != sorted(keys):
                    raise ValueError(f"Plain-dict keys at {path} are not canonical.")
            except TypeError as error:
                raise ValueError(
                    f"Mapping keys at {path} do not have a stable JAX ordering."
                ) from error
            return

        if kind == "jax_array":
            _require_keys(node, {"kind", "shape", "dtype", "weak_type"}, path)
            if retained:
                raise ValueError(f"Static array definition at {path} is unsupported.")
            _validate_shape(node["shape"], path)
            if node["dtype"] not in _SUPPORTED_DTYPES:
                raise ValueError(f"Invalid JAX array dtype at {path}.")
            if node["weak_type"] is not False:
                raise ValueError(f"Weak JAX arrays at {path} are unsupported.")
            return

        if kind == "jax_prng_key":
            _require_keys(node, {"kind", "shape", "data_shape", "impl"}, path)
            if retained:
                raise ValueError(
                    f"Static random-key definition at {path} is unsupported."
                )
            _validate_shape(node["shape"], path)
            _validate_shape(node["data_shape"], f"{path}.data_shape")
            if math.prod(node["data_shape"]) > _MAX_PRNG_WORDS:
                raise ValueError(f"JAX random-key data at {path} is too large.")
            impl = node["impl"]
            if type(impl) is not str or not 0 < len(impl) <= 128:
                raise ValueError(f"Invalid JAX random-key implementation at {path}.")
            return

        if kind == "complex":
            _require_keys(node, {"kind", "value"}, path)
            values = node["value"]
            if (
                not isinstance(values, list)
                or len(values) != 2
                or not all(type(value) is float for value in values)
                or not all(math.isfinite(value) for value in values)
            ):
                raise ValueError(f"Invalid complex literal at {path}.")
            return
        if kind == "range":
            _require_keys(node, {"kind", "start", "stop", "step"}, path)
            values = (node["start"], node["stop"], node["step"])
            for name, value in zip(("start", "stop", "step"), values):
                _validate_integer(value, f"{path}.{name}")
            if node["step"] == 0:
                raise ValueError(f"Invalid range literal at {path}.")
            return
        if kind == "slice":
            _require_keys(node, {"kind", "start", "stop", "step"}, path)
            for name in ("start", "stop", "step"):
                _validate_node(
                    node[name],
                    f"{path}.{name}",
                    depth + 1,
                    state,
                    ancestors,
                    retained=True,
                )
            return
        if kind == "type":
            _require_keys(node, {"kind", "type"}, path)
            _validate_identifier(node["type"], f"{path}.type")
            return
        raise ValueError(f"Unsupported object definition kind {kind!r} at {path}.")
    finally:
        ancestors.remove(identity)


def _validate_definition(definition):
    """Validate one complete definition before resolving any Python type."""
    _validate_node(definition, "root", 0, [0], set(), retained=False)
