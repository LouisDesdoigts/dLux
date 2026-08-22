"""Tests for archive framing, schema validation, and payload integrity."""

from io import BytesIO
import json
import zipfile

import jax.numpy as jnp
import pytest

import dLux as dl

from ._helpers import ForeignModule, read_archive, rewrite_archive, write_archive


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.pop("definition"), "missing required fields"),
        (
            lambda manifest: manifest.__setitem__("format", {}),
            "not a supported dLux archive",
        ),
        (
            lambda manifest: manifest["format"].__setitem__("version", 2),
            "Unsupported dLux archive version",
        ),
        (
            lambda manifest: manifest.__setitem__("definition", []),
            "Object definition node.*must be a mapping",
        ),
        (
            lambda manifest: manifest.__setitem__("created_with", []),
            "invalid provenance",
        ),
        (
            lambda manifest: manifest.__setitem__("payload", []),
            "invalid payload description",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("member", "other"),
            "unsupported payload member",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("codec", "other"),
            "Unsupported dLux payload codec",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("version", True),
            "Unsupported dLux payload version",
        ),
        (
            lambda manifest: manifest["payload"].__setitem__("size", -1),
            "payload size must be a non-negative integer",
        ),
    ],
    ids=[
        "missing-definition",
        "format",
        "format-version",
        "definition",
        "provenance",
        "payload",
        "payload-member",
        "payload-codec",
        "payload-version",
        "payload-size",
    ],
)
def test_load_rejects_invalid_manifest(mutate, message):
    original = dl.Optic(transmission=jnp.ones((2, 2)))
    file = BytesIO()
    dl.save(file, original)

    rewrite_archive(file, update_manifest=lambda manifest, payload: mutate(manifest))

    with pytest.raises(ValueError, match=message):
        dl.load(file)


def test_definition_schema_is_validated_before_types_or_payload():
    original = ForeignModule(jnp.ones(2), 2, "science")
    file = BytesIO()
    dl.save(file, original)

    def mutate(manifest, payload):
        manifest["definition"]["type"] = "missing.module:Ghost"
        manifest["definition"]["fields"][0]["value"]["shape"] = [True]
        manifest["payload"]["size"] = len(payload)

    rewrite_archive(file, update_manifest=mutate, update_payload=lambda payload: b"x")

    with pytest.raises(ValueError, match="Invalid JAX array shape"):
        dl.load(file)


@pytest.mark.parametrize(
    ("archive_change", "message"),
    [
        ("duplicate-member", "duplicate members"),
        ("extra-member", "must contain manifest.json and leaves.eqx"),
        ("compressed", "Compressed dLux archive members are unsupported"),
        ("duplicate-json", "manifest is not valid UTF-8 JSON"),
    ],
)
def test_load_rejects_ambiguous_archive_framing(archive_change, message):
    original = dl.Optic(transmission=jnp.ones((2, 2)))
    file = BytesIO()
    dl.save(file, original)
    manifest, payload = read_archive(file)

    if archive_change == "duplicate-member":
        with pytest.warns(UserWarning, match="Duplicate name"):
            write_archive(file, manifest, payload, duplicate_manifest=True)
    elif archive_change == "extra-member":
        write_archive(file, manifest, payload, extra_member="extra.bin")
    elif archive_change == "compressed":
        write_archive(file, manifest, payload, compression=zipfile.ZIP_DEFLATED)
    else:
        encoded = json.dumps(manifest)[1:]
        raw_manifest = ('{"definition": null,' + encoded).encode()
        write_archive(file, manifest, payload, raw_manifest=raw_manifest)

    with pytest.raises(ValueError, match=message):
        dl.load(file)


def test_load_rejects_payload_size_mismatch():
    file = BytesIO()
    dl.save(file, jnp.ones(2))

    def mutate(manifest, payload):
        manifest["payload"]["size"] += 1

    rewrite_archive(file, update_manifest=mutate)

    with pytest.raises(ValueError, match="payload size does not match"):
        dl.load(file)


def test_load_rejects_corrupt_payload():
    file = BytesIO()
    dl.save(file, jnp.ones(2))

    def corrupt(payload):
        return b"BAD" + payload[3:]

    rewrite_archive(file, update_payload=corrupt)

    with pytest.raises(ValueError, match="payload could not be decoded"):
        dl.load(file)


def test_load_rejects_unconsumed_payload():
    file = BytesIO()
    dl.save(file, jnp.ones(2))

    def update_size(manifest, payload):
        manifest["payload"]["size"] = len(payload)

    rewrite_archive(
        file,
        update_manifest=update_size,
        update_payload=lambda payload: payload + b"EXTRA",
    )

    with pytest.raises(ValueError, match="payload contains unconsumed data"):
        dl.load(file)


def test_load_rejects_non_archive():
    with pytest.raises(ValueError, match="not a valid dLux ZIP archive"):
        dl.load(BytesIO(b"not an archive"))
