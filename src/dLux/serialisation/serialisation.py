"""Public save and load workflows for validated dLux archives."""

from contextlib import nullcontext
import zipfile

import equinox as eqx

from ._archive import (
    _PAYLOAD_MEMBER,
    _normalise_path,
    _read_manifest,
    _save_file,
    _save_path,
    _validate_members,
    _validate_payload_size,
)
from ._leaves import _deserialise_leaf
from .definition import ObjectDefinition

__all__ = ["ObjectDefinition", "save", "load"]


def save(file_or_path, pytree) -> None:
    """Save supported Equinox modules and containers to a dLux archive.

    Parameters
    ----------
    file_or_path : str, pathlib.Path, or binary file
        Destination for the archive. Paths without a suffix receive ``.dlux``.
        Binary files remain open after saving.
    pytree : Equinox module or supported container
        Realised object containing Equinox modules, supported built-in containers,
        JAX arrays, Python numerical values, literals, and static values.

    Raises
    ------
    TypeError
        If any leaf or static value cannot be represented losslessly.

    Examples
    --------
    Save an optic through either the functional or object API:

    ```python
    import jax.numpy as jnp

    import dLux as dl

    optic = dl.Optic(transmission=jnp.ones((2, 2)))
    dl.save("optic", optic)
    optic.save("optic")
    ```

    Notes
    -----
    Concrete JAX arrays are written to the binary payload. Python numerical values,
    literals, static values, and JAX array metadata are written to the generated JSON
    definition.

    NumPy array and scalar leaves are intentionally outside this initial contract.
    Callables, opaque objects, weakly typed JAX arrays, and unsupported JAX dtypes
    cannot be saved losslessly. Python integers must fit in signed 64 bits.
    Non-finite Python float and complex values are unsupported and should be stored
    as strongly typed JAX arrays when required.
    """
    # Describe the complete realised object before writing any bytes.
    definition = ObjectDefinition.from_object(pytree)

    # Stage file objects or atomically replace path destinations.
    path = _normalise_path(file_or_path)
    if path is None:
        _save_file(file_or_path, pytree, definition.to_dict())
    else:
        _save_path(path, pytree, definition.to_dict())


def load(file_or_path, *, like=None, custom_types=None):
    """Load a validated dLux archive, optionally using an existing template.

    Parameters
    ----------
    file_or_path : str, pathlib.Path, or binary file
        dLux archive to read. Paths without a suffix receive ``.dlux``. Binary files
        must be readable and seekable and remain open after loading.
    like : Equinox module or supported container, optional
        Trusted object used to resolve classes and validate module fields, container
        topology, mapping keys, literal types, and JAX array shapes and dtypes. Its
        non-array values are ignored; archived values come from the stored definition.
        If omitted, classes are resolved from modules that are already imported.
    custom_types : mapping[str, type], optional
        Explicit mapping from stored ``"module:qualname"`` identifiers to classes.
        This can resolve local or otherwise unavailable classes without importing
        code named by the archive. The supplied class must retain the stored nominal
        identifier and field schema.

    Returns
    -------
    pytree : Equinox module or supported container
        Reconstructed object with its stored values restored from the archive.

    Raises
    ------
    ValueError
        If the archive is invalid, a required class is unavailable, or its structural
        definition is incompatible with ``like``.
    TypeError
        If ``like`` or ``custom_types`` contains an unsupported value.

    Examples
    --------
    Reconstruct an optic directly, or restore it into a matching template:

    ```python
    import jax.numpy as jnp

    import dLux as dl

    restored = dl.load("optic")

    like = dl.Optic(transmission=jnp.zeros((2, 2)))
    checked = dl.load("optic", like=like)
    ```

    Notes
    -----
    Archives contain an automatically generated JSON object definition and an
    Equinox JAX-array stream. The producing Python and core package versions are
    recorded as diagnostic provenance; compatibility is determined by the archive
    and payload format versions plus the generated definition.

    Loading never imports code named by an archive. For template-free loading, every
    stored class must already be imported or explicitly supplied through
    ``custom_types``. An archive can select an already imported Equinox class, so
    template-free loading should be limited to trusted archives and class providers.
    Constructors are not run when rebuilding Equinox modules, so validation checks
    stored structure and field representation rather than rerunning package-specific
    constructor invariants.
    """
    # Open a path while retaining ownership of caller-provided file objects.
    path = _normalise_path(file_or_path)
    source = nullcontext(file_or_path) if path is None else path.open("rb")

    with source as file:
        try:
            with zipfile.ZipFile(file, mode="r") as archive:
                # Validate the archive and template before decoding JAX arrays.
                _validate_members(archive)
                manifest = _read_manifest(archive)
                _validate_payload_size(archive, manifest)
                definition = ObjectDefinition.from_dict(manifest["definition"])
                template = definition.build_template(
                    like=like,
                    custom_types=custom_types,
                )

                # Restore every JAX array leaf and require complete payload use.
                with archive.open(_PAYLOAD_MEMBER) as payload:
                    try:
                        loaded = eqx.tree_deserialise_leaves(
                            payload,
                            template,
                            filter_spec=_deserialise_leaf,
                        )
                    except Exception as error:
                        raise ValueError(
                            "The dLux JAX array payload could not be decoded."
                        ) from error
                    if payload.read(1) != b"":
                        raise ValueError("The dLux payload contains unconsumed data.")

                # Confirm that decoding did not alter the declared object definition.
                definition.validate(loaded)
                return loaded
        except zipfile.BadZipFile as error:
            raise ValueError("The input is not a valid dLux ZIP archive.") from error
