"""Shared fixtures for serialisation contract tests."""

from io import BytesIO
import json
import zipfile

import equinox as eqx
import jax.numpy as jnp

import dLux as dl


class ForeignModule(eqx.Module):
    """Module outside dLux used to exercise generic object definitions."""

    array: object
    count: int
    mode: str = eqx.field(static=True)


class StaticParametric(dl.Parametric):
    """Small dLux subclass with structured static metadata."""

    value: object
    metadata: object = eqx.field(static=True)

    def evaluate(self, **context):
        return self.value


class ConstructorModule(eqx.Module):
    """Module whose constructor lets template construction be observed."""

    calls = 0
    array: object
    label: str

    def __init__(self, array, label):
        type(self).calls += 1
        self.array = array
        self.label = label


class CustomArrayLike:
    """Unsupported object implementing the JAX array protocol."""

    def __jax_array__(self):
        return jnp.ones(1)


def make_system(transmission=None, *, scale=0.1, normalise=True, name="pupil"):
    """Construct a small executable optical system."""
    if transmission is None:
        transmission = jnp.arange(4.0).reshape(2, 2) + 1
    grid = dl.GridSpec(n=(2, 2), d=scale, unit="m")
    pupil = dl.Optic(transmission=transmission, normalise=normalise)
    return dl.OpticalSystem([(name, pupil)], grid)


def roundtrip(original, **load_kwargs):
    """Round-trip an object through the public functional API."""
    file = BytesIO()
    dl.save(file, original)
    assert not file.closed
    file.seek(0)
    loaded = dl.load(file, **load_kwargs)
    assert not file.closed
    return loaded


def read_archive(file):
    """Return the decoded manifest and raw payload from an in-memory archive."""
    file.seek(0)
    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = archive.read("leaves.eqx")
    return manifest, payload


def write_archive(
    file,
    manifest,
    payload,
    *,
    compression=zipfile.ZIP_STORED,
    raw_manifest=None,
    extra_member=None,
    duplicate_manifest=False,
):
    """Replace an in-memory archive with selected valid ZIP framing."""
    data = json.dumps(manifest).encode() if raw_manifest is None else raw_manifest
    rebuilt = BytesIO()
    with zipfile.ZipFile(rebuilt, "w", compression=compression) as archive:
        archive.writestr("manifest.json", data)
        if duplicate_manifest:
            archive.writestr("manifest.json", data)
        archive.writestr("leaves.eqx", payload)
        if extra_member is not None:
            archive.writestr(extra_member, b"extra")
    file.seek(0)
    file.truncate(0)
    file.write(rebuilt.getvalue())
    file.seek(0)


def rewrite_archive(file, *, update_manifest=None, update_payload=None):
    """Rebuild an archive after applying selected mutations."""
    manifest, payload = read_archive(file)
    if update_payload is not None:
        payload = update_payload(payload)
    if update_manifest is not None:
        update_manifest(manifest, payload)
    write_archive(file, manifest, payload)
