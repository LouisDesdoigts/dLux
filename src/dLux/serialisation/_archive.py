"""JSON and ZIP archive I/O for dLux PyTrees."""

import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path

import equinox as eqx

from ._leaves import _serialise_leaf

_SUFFIX = ".dlux"
_MANIFEST_MEMBER = "manifest.json"
_PAYLOAD_MEMBER = "leaves.eqx"
_MEMBERS = {_MANIFEST_MEMBER, _PAYLOAD_MEMBER}
_FORMAT_NAME = "dlux"
_FORMAT_VERSION = 1
_PAYLOAD_CODEC = "equinox-jax-leaves"
_PAYLOAD_VERSION = 1
_MAX_MANIFEST_SIZE = 16 * 1024**2
_PACKAGE_NAMES = {
    "dlux": "dLux",
    "equinox": "equinox",
    "jax": "jax",
    "jaxlib": "jaxlib",
    "numpy": "numpy",
    "zodiax": "zodiax",
}


class _CountingWriter:
    """Forward writes while recording the total byte count."""

    def __init__(self, file):
        self.file = file
        self.size = 0

    def write(self, data):
        """Write one complete byte chunk to the wrapped archive member."""
        written = self.file.write(data)
        written = len(data) if written is None else written
        if written != len(data):
            raise OSError("Failed to write the complete dLux payload.")
        self.size += written
        return written


def _package_version(distribution):
    """Return an installed distribution version when it is available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _reject_json_constant(value):
    """Reject non-standard NaN and infinity tokens while decoding JSON."""
    raise ValueError(f"Invalid JSON constant {value}.")


def _reject_duplicate_keys(pairs):
    """Build a JSON object while rejecting duplicate member names."""
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _provenance():
    """Return focused diagnostic versions for the producing environment."""
    packages = {
        name: _package_version(distribution)
        for name, distribution in _PACKAGE_NAMES.items()
    }
    return {
        "python": platform.python_version(),
        "packages": packages,
    }


def _build_manifest(definition, payload_size):
    """Build the archive manifest after serialising its JAX array payload."""
    return {
        "format": {
            "name": _FORMAT_NAME,
            "version": _FORMAT_VERSION,
        },
        "definition": definition,
        "payload": {
            "member": _PAYLOAD_MEMBER,
            "codec": _PAYLOAD_CODEC,
            "version": _PAYLOAD_VERSION,
            "size": payload_size,
        },
        "created_with": _provenance(),
    }


def _zip_info(name):
    """Return deterministic metadata for one uncompressed archive member."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _write_archive(file, pytree, definition):
    """Write one complete dLux archive to a binary file object."""
    with zipfile.ZipFile(file, mode="w", allowZip64=True) as archive:
        payload_info = _zip_info(_PAYLOAD_MEMBER)
        with archive.open(payload_info, mode="w", force_zip64=True) as payload:
            writer = _CountingWriter(payload)
            eqx.tree_serialise_leaves(writer, pytree, filter_spec=_serialise_leaf)

        manifest = _build_manifest(definition, writer.size)
        data = (
            json.dumps(
                manifest,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if len(data) > _MAX_MANIFEST_SIZE:
            raise ValueError("The generated dLux manifest is too large.")
        archive.writestr(_zip_info(_MANIFEST_MEMBER), data)


def _normalise_path(file_or_path):
    """Return a path with the default dLux suffix, or None for file objects."""
    if not isinstance(file_or_path, (str, os.PathLike)):
        return None
    path = Path(file_or_path)
    return path if path.suffix else path.with_suffix(_SUFFIX)


def _save_file(file, pytree, definition):
    """Stage an archive before replacing a caller-owned binary stream."""
    if not hasattr(file, "write"):
        raise TypeError("file_or_path must be a path or writable binary file.")
    try:
        file.write(b"")
    except TypeError as error:
        raise TypeError("file_or_path must be a writable binary file.") from error

    with tempfile.SpooledTemporaryFile(max_size=64 * 1024**2, mode="w+b") as staged:
        _write_archive(staged, pytree, definition)
        staged.seek(0)
        try:
            seekable = file.seekable()
        except AttributeError:
            seekable = False
        if seekable:
            file.seek(0)
            file.truncate(0)
        shutil.copyfileobj(staged, file)
    if seekable:
        file.truncate()


def _save_path(path, pytree, definition):
    """Atomically replace a path with one completely written archive."""
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w+b") as file:
            _write_archive(file, pytree, definition)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_manifest(manifest):
    """Validate the fixed fields required to interpret a v1 dLux archive."""
    if not isinstance(manifest, dict):
        raise ValueError("The dLux manifest must be a JSON object.")
    required = {"format", "definition", "payload", "created_with"}
    if not required.issubset(manifest):
        raise ValueError("The dLux manifest is missing required fields.")

    format_spec = manifest["format"]
    if not isinstance(format_spec, dict) or format_spec.get("name") != _FORMAT_NAME:
        raise ValueError("This is not a supported dLux archive.")
    format_version = format_spec.get("version")
    if type(format_version) is not int or format_version != _FORMAT_VERSION:
        raise ValueError(f"Unsupported dLux archive version {format_version!r}.")

    if not isinstance(manifest["created_with"], dict):
        raise ValueError("The dLux manifest has invalid provenance.")

    payload = manifest["payload"]
    if not isinstance(payload, dict):
        raise ValueError("The dLux manifest has an invalid payload description.")
    if payload.get("member") != _PAYLOAD_MEMBER:
        raise ValueError("The dLux manifest identifies an unsupported payload member.")
    if payload.get("codec") != _PAYLOAD_CODEC:
        raise ValueError(f"Unsupported dLux payload codec {payload.get('codec')!r}.")
    payload_version = payload.get("version")
    if type(payload_version) is not int or payload_version != _PAYLOAD_VERSION:
        raise ValueError(f"Unsupported dLux payload version {payload_version!r}.")
    if type(payload.get("size")) is not int or payload["size"] < 0:
        raise ValueError("The dLux payload size must be a non-negative integer.")


def _read_manifest(archive):
    """Read and parse the bounded UTF-8 JSON manifest."""
    info = archive.getinfo(_MANIFEST_MEMBER)
    if info.file_size > _MAX_MANIFEST_SIZE:
        raise ValueError("The dLux manifest is too large.")
    try:
        data = archive.read(_MANIFEST_MEMBER)
        manifest = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as error:
        raise ValueError("The dLux manifest is not valid UTF-8 JSON.") from error
    _validate_manifest(manifest)
    return manifest


def _validate_members(archive):
    """Require exactly one manifest and one JAX array payload member."""
    names = [info.filename for info in archive.infolist()]
    if len(names) != len(set(names)):
        raise ValueError("The dLux archive contains duplicate members.")
    if set(names) != _MEMBERS:
        raise ValueError("The dLux archive must contain manifest.json and leaves.eqx.")
    for info in archive.infolist():
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted dLux archive members are unsupported.")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Compressed dLux archive members are unsupported.")


def _validate_payload_size(archive, manifest):
    """Verify the uncompressed payload size against its manifest."""
    expected = manifest["payload"]["size"]
    actual = archive.getinfo(_PAYLOAD_MEMBER).file_size
    if actual != expected:
        raise ValueError("The dLux payload size does not match its manifest.")
