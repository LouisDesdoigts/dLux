"""Numerical-leaf policy for dLux archives."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

_SUPPORTED_DTYPES = {
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
    "bfloat16",
}
_NOT_PAYLOAD = object()


def _is_prng_dtype(dtype):
    """Return whether a dtype represents a modern JAX random key."""
    prng_key = getattr(jax.dtypes, "prng_key", None)
    if prng_key is None:
        return False
    try:
        return jax.dtypes.issubdtype(dtype, prng_key)
    except TypeError:
        return False


def _is_prng_key(value):
    """Return whether a concrete value is a modern JAX random key."""
    return isinstance(value, jax.Array) and _is_prng_dtype(value.dtype)


def _validate_dtype(dtype, path):
    """Reject JAX dtypes that the Equinox codec cannot round-trip."""
    if str(dtype) not in _SUPPORTED_DTYPES:
        raise TypeError(f"{path} has unsupported dtype {dtype}.")


def _normalise_shape(shape, path):
    """Return a concrete array shape as JSON integers."""
    try:
        return [int(size) for size in shape]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{path} has a non-concrete shape.") from error


def _payload_definition(value, *, template, path):
    """Describe a JAX payload leaf, or return ``_NOT_PAYLOAD``."""
    # Python scalars are represented exactly in the JSON definition. Although
    # Equinox considers them array-like, they are intentionally not part of the
    # binary JAX-array payload.
    if type(value) in (bool, int, float, complex):
        return _NOT_PAYLOAD

    if isinstance(value, jax.ShapeDtypeStruct):
        if not template:
            raise TypeError(f"{path} is an abstract array and cannot be saved.")
        if _is_prng_dtype(value.dtype):
            raise TypeError(
                f"{path} is an abstract JAX random key. Supply a concrete key in "
                "like or load without like."
            )
        _validate_dtype(value.dtype, path)
        return {
            "kind": "jax_array",
            "shape": _normalise_shape(value.shape, path),
            "dtype": str(value.dtype),
            "weak_type": bool(value.weak_type),
        }

    if isinstance(value, jax.Array):
        if _is_prng_key(value):
            data = jr.key_data(value)
            return {
                "kind": "jax_prng_key",
                "shape": _normalise_shape(value.shape, path),
                "data_shape": _normalise_shape(data.shape, path),
                "impl": str(jr.key_impl(value)),
            }
        _validate_dtype(value.dtype, path)
        if value.weak_type and not template:
            raise TypeError(f"{path} is weakly typed and cannot be saved losslessly.")
        return {
            "kind": "jax_array",
            "shape": _normalise_shape(value.shape, path),
            "dtype": str(value.dtype),
            "weak_type": bool(value.weak_type),
        }

    if eqx.is_array(value):
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        raise TypeError(
            f"{path} has unsupported array type {name}; only JAX arrays are "
            "supported."
        )
    if eqx.is_array_like(value):
        raise TypeError(f"{path} has an unsupported custom array-like type.")
    return _NOT_PAYLOAD


def _payload_placeholder(definition, path):
    """Construct an Equinox deserialisation placeholder for a payload node."""
    kind = definition.get("kind")
    if kind == "jax_array":
        try:
            dtype = jnp.dtype(definition["dtype"])
            shape = tuple(definition["shape"])
            weak_type = definition["weak_type"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid JAX array definition at {path}.") from error
        _validate_dtype(dtype, path)
        if not isinstance(weak_type, bool):
            raise ValueError(f"Invalid JAX weak type at {path}.")
        return jax.ShapeDtypeStruct(shape, dtype, weak_type=weak_type)

    if kind == "jax_prng_key":
        try:
            shape = tuple(definition["shape"])
            data_shape = tuple(definition["data_shape"])
            impl = definition["impl"]
            key = jr.wrap_key_data(jnp.zeros(data_shape, dtype=jnp.uint32), impl=impl)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid JAX random-key definition at {path}.") from error
        if key.shape != shape:
            raise ValueError(f"Invalid JAX random-key shape at {path}.")
        return key

    return _NOT_PAYLOAD


def _serialise_leaf(file, value):
    """Write only concrete JAX array leaves to the Equinox payload."""
    if _is_prng_key(value):
        eqx.default_serialise_filter_spec(file, jr.key_data(value))
    elif isinstance(value, jax.Array):
        eqx.default_serialise_filter_spec(file, value)


def _deserialise_leaf(file, value):
    """Restore only JAX array leaves, retaining definition-built literals."""
    if _is_prng_key(value):
        data = jr.key_data(value)
        template = jax.ShapeDtypeStruct(data.shape, data.dtype)
        loaded = eqx.default_deserialise_filter_spec(file, template)
        return jr.wrap_key_data(loaded, impl=jr.key_impl(value))
    if isinstance(value, (jax.Array, jax.ShapeDtypeStruct)):
        return eqx.default_deserialise_filter_spec(file, value)
    return value
