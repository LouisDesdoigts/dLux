"""Tests for generated, inspectable object definitions."""

import json

import jax
import jax.numpy as jnp
import pytest

import dLux as dl

from ._helpers import ConstructorModule, ForeignModule


def test_public_exports():
    assert dl.ObjectDefinition is dl.serialisation.ObjectDefinition
    assert dl.save is dl.serialisation.save
    assert dl.load is dl.serialisation.load
    assert dl.serialisation.__all__ == ["ObjectDefinition", "save", "load"]


def test_object_definition_has_readable_generated_schema():
    obj = ForeignModule(jnp.arange(3.0, dtype=jnp.float32), 3, "science")
    definition = dl.ObjectDefinition.from_object(obj)

    expected = {
        "kind": "module",
        "type": f"{ForeignModule.__module__}:ForeignModule",
        "fields": [
            {
                "name": "array",
                "static": False,
                "value": {
                    "kind": "jax_array",
                    "shape": [3],
                    "dtype": "float32",
                    "weak_type": False,
                },
            },
            {"name": "count", "static": False, "value": 3},
            {"name": "mode", "static": True, "value": "science"},
        ],
    }
    assert definition.root == expected
    assert definition.to_dict() == expected
    assert json.loads(definition.to_json()) == expected
    assert str(definition) == definition.to_json()


def test_object_definition_owns_defensive_copies():
    source = dl.ObjectDefinition.from_object(
        ForeignModule(jnp.ones(2), 2, "science")
    ).to_dict()
    definition = dl.ObjectDefinition.from_dict(source)

    source["fields"][0]["value"]["shape"][0] = 99
    returned = definition.root
    returned["fields"][1]["value"] = 99

    assert definition.root["fields"][0]["value"]["shape"] == [2]
    assert definition.root["fields"][1]["value"] == 2


def test_object_definition_validation_compares_metadata_not_array_data():
    definition = dl.ObjectDefinition.from_object(
        ForeignModule(jnp.arange(3.0), 3, "science")
    )

    definition.validate(ForeignModule(jnp.zeros(3), 3, "science"))

    for changed in (
        ForeignModule(jnp.zeros(4), 3, "science"),
        ForeignModule(jnp.zeros(3, dtype=jnp.complex64), 3, "science"),
        ForeignModule(jnp.zeros(3), 4, "science"),
        ForeignModule(jnp.zeros(3), 3, "engineering"),
    ):
        with pytest.raises(ValueError, match="Object definition mismatch"):
            definition.validate(changed)


def test_object_definition_builds_template_without_constructor():
    ConstructorModule.calls = 0
    original = ConstructorModule(jnp.arange(3.0), "stored")
    definition = dl.ObjectDefinition.from_object(original)
    calls = ConstructorModule.calls

    template = definition.build_template()

    assert ConstructorModule.calls == calls
    assert isinstance(template.array, jax.ShapeDtypeStruct)
    assert template.array.shape == (3,)
    assert template.label == "stored"
    definition.validate(template)


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ([1, True], "Duplicate mapping keys"),
        ([2, 1], "not canonical"),
        ([1, "a"], "do not have a stable JAX ordering"),
    ],
    ids=["equal", "noncanonical", "unsortable"],
)
def test_object_definition_rejects_ambiguous_plain_dict_keys(keys, message):
    definition = {
        "kind": "mapping",
        "type": "builtins:dict",
        "entries": [{"key": key, "value": None} for key in keys],
    }

    with pytest.raises(ValueError, match=message):
        dl.ObjectDefinition.from_dict(definition)


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (2**63, "signed 64 bits"),
        (
            {
                "kind": "jax_prng_key",
                "shape": [],
                "data_shape": [1_000_001],
                "impl": "fry",
            },
            "random-key data.*too large",
        ),
    ],
    ids=["integer", "prng-size"],
)
def test_object_definition_rejects_bounded_schema_values(definition, message):
    with pytest.raises(ValueError, match=message):
        dl.ObjectDefinition.from_dict(definition)


def test_object_definition_rejects_excessive_nesting():
    definition = None
    for _ in range(130):
        definition = {"kind": "list", "items": [definition]}

    with pytest.raises(ValueError, match="maximum nesting depth"):
        dl.ObjectDefinition.from_dict(definition)


def test_object_definition_rejects_changed_installed_class_schema():
    definition = dl.ObjectDefinition.from_object(
        ForeignModule(jnp.ones(2), 2, "science")
    ).to_dict()
    definition["fields"].pop()

    changed = dl.ObjectDefinition.from_dict(definition)
    with pytest.raises(ValueError, match="installed class schema.*changed"):
        changed.build_template()
