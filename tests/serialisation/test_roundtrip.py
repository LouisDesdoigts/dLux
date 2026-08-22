"""Tests for public save/load workflows and supported object state."""

from collections import OrderedDict, defaultdict, namedtuple
from io import BytesIO, StringIO
import json
import math
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import pytest

import dLux as dl

from ._helpers import (
    CustomArrayLike,
    ForeignModule,
    StaticParametric,
    make_system,
    roundtrip,
)


def test_nested_system_path_roundtrip_without_like(tmp_path):
    original = make_system()
    path = tmp_path / "system"

    original.save(path)
    loaded = dl.load(path)

    archive_path = path.with_suffix(".dlux")
    assert archive_path.is_file()
    assert eqx.tree_equal(loaded, original, typematch=True)
    assert jnp.allclose(
        loaded.propagate_mono(1e-6),
        original.propagate_mono(1e-6),
    )

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "leaves.eqx"}
        manifest = json.loads(archive.read("manifest.json"))
    assert set(manifest) == {"format", "definition", "payload", "created_with"}
    assert manifest["format"] == {"name": "dlux", "version": 1}
    assert manifest["definition"] == dl.ObjectDefinition.from_object(original).root
    assert manifest["payload"]["member"] == "leaves.eqx"
    assert manifest["payload"]["codec"] == "equinox-jax-leaves"
    assert manifest["payload"]["version"] == 1
    assert manifest["payload"]["size"] > 0
    assert manifest["created_with"]["packages"]["dlux"] == dl.__version__


def test_functional_save_replaces_caller_owned_stream():
    first = dl.Optic(transmission=jnp.zeros((2, 2)))
    second = dl.Optic(transmission=jnp.ones((2, 2)))
    file = BytesIO(b"stale archive contents")

    dl.save(file, first)
    dl.save(file, second)
    file.seek(0)
    loaded = dl.load(file)

    assert not file.closed
    assert eqx.tree_equal(loaded, second, typematch=True)


def test_explicit_path_suffix_is_preserved(tmp_path):
    original = dl.Optic(transmission=jnp.ones((2, 2)))
    path = tmp_path / "optic.archive"

    dl.save(path, original)
    loaded = dl.load(path)

    assert path.is_file()
    assert not path.with_suffix(".dlux").exists()
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_structured_literals_and_containers_roundtrip_without_like():
    original = OrderedDict(
        [
            ("none", None),
            ("bool", True),
            ("limits", (-(2**63), 2**63 - 1)),
            ("negative_zero", -0.0),
            ("float", 1.25),
            ("complex", 1 + 2j),
            ("unicode", "lambda: λ"),
            ("range", range(1, 8, 2)),
            ("slice", slice(1, 8, 2)),
            ("type", float),
            ("list", [1, "two", None]),
            ("plain_dict", {"z": jnp.arange(2), "a": jnp.ones((2, 2))}),
        ]
    )

    loaded = roundtrip(original)

    assert list(loaded) == list(original)
    assert list(loaded["plain_dict"]) == ["a", "z"]
    assert math.copysign(1.0, loaded["negative_zero"]) == -1.0
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_structured_static_metadata_roundtrip_without_like():
    metadata = [
        {"complex": 1 + 2j, "range": range(1, 5, 2)},
        slice(1, 5, 2),
        float,
    ]
    original = StaticParametric(jnp.arange(3.0), metadata)

    loaded = roundtrip(original)

    assert eqx.tree_equal(loaded, original, typematch=True)


def test_base_load_uses_self_as_like_and_archived_values_win():
    class LocalParametric(dl.Parametric):
        value: object
        mode: str = eqx.field(static=True)

        def evaluate(self, **context):
            return self.value

    original = LocalParametric(jnp.arange(3.0), "archived")
    like = LocalParametric(jnp.zeros(3), "template")

    file = BytesIO()
    original.save(file)
    file.seek(0)

    loaded = like.load(file)

    assert eqx.tree_equal(loaded, original, typematch=True)


def test_shape_dtype_struct_is_accepted_as_like():
    original = dl.Interpolation([0.0, 1.0], [2.0, 3.0])
    like = eqx.filter_eval_shape(lambda: original)

    loaded = roundtrip(original, like=like)

    assert isinstance(like.knots, jax.ShapeDtypeStruct)
    assert isinstance(loaded.knots, jax.Array)
    assert eqx.tree_equal(loaded, original, typematch=True)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_like_rejects_mismatched_array_metadata(mismatch):
    original = dl.Optic(transmission=jnp.ones((2, 2)))
    if mismatch == "shape":
        like = dl.Optic(transmission=jnp.ones((3, 3)))
    else:
        like = original.set("transmission", jnp.ones((2, 2), dtype=jnp.complex64))
    file = BytesIO()
    dl.save(file, original)
    file.seek(0)

    with pytest.raises(ValueError, match=f"Object structure mismatch.*{mismatch}"):
        dl.load(file, like=like)


@pytest.mark.parametrize(
    "like",
    [
        dl.TransmissiveLayer(transmission=jnp.zeros((2, 2))),
        make_system(jnp.zeros((2, 2)), name="other"),
    ],
    ids=["root-type", "mapping-key"],
)
def test_like_rejects_mismatched_topology(like):
    original = (
        make_system()
        if isinstance(like, dl.OpticalSystem)
        else dl.Optic(transmission=jnp.ones((2, 2)))
    )
    file = BytesIO()
    dl.save(file, original)
    file.seek(0)

    with pytest.raises(ValueError, match="Object structure mismatch"):
        dl.load(file, like=like)


def test_module_scoped_foreign_module_roundtrip_without_configuration():
    original = ForeignModule(jnp.arange(3.0), 3, "science")

    loaded = roundtrip(original)

    assert type(loaded) is ForeignModule
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_local_class_can_be_resolved_by_like_or_custom_types():
    class LocalModule(eqx.Module):
        array: object
        label: str

    original = LocalModule(jnp.arange(3.0), "archived")
    like = LocalModule(jnp.zeros(3), "template")
    definition = dl.ObjectDefinition.from_object(original)
    identifier = definition.root["type"]
    file = BytesIO()
    dl.save(file, original)

    file.seek(0)
    with pytest.raises(ValueError, match="Local type.*Supply like"):
        dl.load(file)

    file.seek(0)
    loaded_like = dl.load(file, like=like)
    file.seek(0)
    loaded_registered = dl.load(file, custom_types={identifier: LocalModule})

    assert eqx.tree_equal(loaded_like, original, typematch=True)
    assert eqx.tree_equal(loaded_registered, original, typematch=True)


def test_custom_type_arguments_are_validated():
    original = ForeignModule(jnp.arange(2.0), 2, "science")
    identifier = dl.ObjectDefinition.from_object(original).root["type"]
    file = BytesIO()
    dl.save(file, original)

    file.seek(0)
    with pytest.raises(TypeError, match="custom_types must be a mapping"):
        dl.load(file, custom_types=[])
    file.seek(0)
    with pytest.raises(TypeError, match="is not a class"):
        dl.load(file, custom_types={identifier: object()})
    file.seek(0)
    with pytest.raises(ValueError, match="nominal identifier"):
        dl.load(file, custom_types={identifier: StaticParametric})
    file.seek(0)
    with pytest.raises(TypeError, match="cannot be supplied together"):
        dl.load(file, like=original, custom_types={})


@pytest.mark.parametrize(
    "dtype_name",
    [
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
    ],
)
def test_supported_jax_dtype_roundtrip(dtype_name):
    with jax.enable_x64():
        original = jnp.asarray([0, 1], dtype=getattr(jnp, dtype_name))
        loaded = roundtrip(original)

    assert isinstance(loaded, jax.Array)
    assert loaded.dtype == original.dtype
    assert jnp.array_equal(loaded, original)


def test_typed_prng_keys_roundtrip_without_like():
    key = jr.key(17)
    original = {"batched": jr.split(key, 3), "single": key}

    loaded = roundtrip(original)

    for name in original:
        assert str(jr.key_impl(loaded[name])) == str(jr.key_impl(original[name]))
        assert loaded[name].shape == original[name].shape
        assert jnp.array_equal(jr.key_data(loaded[name]), jr.key_data(original[name]))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (onp.array([1.0], dtype=onp.float32), "only JAX arrays"),
        (onp.float32(1.0), "only JAX arrays"),
        (object(), "unsupported object metadata"),
        (CustomArrayLike(), "custom array-like"),
        (jnp.array(1.0), "weakly typed"),
        (jnp.zeros(1, dtype=jnp.float8_e5m2), "unsupported dtype"),
        (jax.ShapeDtypeStruct((1,), jnp.float32), "abstract array"),
        (float("nan"), "non-finite Python float"),
        (complex(1, float("inf")), "non-finite Python complex"),
        (-(2**63) - 1, "signed 64-bit"),
        (2**63, "signed 64-bit"),
    ],
    ids=[
        "numpy-array",
        "numpy-scalar",
        "opaque",
        "custom-array-like",
        "weak-jax",
        "float8",
        "abstract-jax",
        "nonfinite-float",
        "nonfinite-complex",
        "integer-underflow",
        "integer-overflow",
    ],
)
def test_save_rejects_unsupported_leaf(value, message):
    with pytest.raises(TypeError, match=message):
        dl.save(BytesIO(), value)


def test_save_rejects_transform_callable():
    transformed = dl.Interpolation([0.0, 1.0], [2.0, 3.0]).map(lambda value: value)

    with pytest.raises(TypeError, match=r"transformation.*unsupported callable"):
        transformed.save(BytesIO())


def test_save_rejects_static_array():
    with pytest.warns(UserWarning, match="being set as static"):
        original = StaticParametric(jnp.ones(1), jnp.ones(1))

    with pytest.raises(TypeError, match="unsupported static array"):
        dl.save(BytesIO(), original)


@pytest.mark.parametrize("state_kind", ["dictionary", "slot"])
def test_save_rejects_non_field_instance_state(state_kind):
    if state_kind == "dictionary":
        original = ForeignModule(jnp.ones(1), 1, "science")
        object.__setattr__(original, "cache", "populated")
    else:

        class SlottedModule(eqx.Module):
            __slots__ = ("cache",)

            value: object

            def __init__(self, value):
                self.value = value
                object.__setattr__(self, "cache", "populated")

        original = SlottedModule(jnp.ones(1))

    with pytest.raises(TypeError, match="non-field state: cache"):
        dl.save(BytesIO(), original)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (defaultdict(int, value=1), "unsupported mapping type defaultdict"),
        (namedtuple("Pair", "left right")(1, 2), "unsupported tuple type Pair"),
        ({1, 2}, "unsupported set metadata"),
        (OrderedDict([(frozenset({1}), 2)]), "unsupported mapping key type"),
        ({1: "one", "two": 2}, "without a stable JAX ordering"),
    ],
    ids=["mapping-subclass", "namedtuple", "set", "mapping-key", "mixed-dict"],
)
def test_save_rejects_unsupported_container(value, message):
    with pytest.raises(TypeError, match=message):
        dl.save(BytesIO(), value)


@pytest.mark.parametrize("file", [StringIO(), object()], ids=["text", "not-file"])
def test_save_rejects_non_binary_destination(file):
    with pytest.raises(TypeError, match="binary file|path or writable"):
        dl.save(file, jnp.ones(1))
