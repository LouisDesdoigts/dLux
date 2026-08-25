"""Latent coefficient-space reparametrisations.

`Reparametrisation` represents a general subspace in reversible parameter coordinates,
while `Selection` varies a masked subset without materialising a dense matrix.
`ParameterTransform` implementations define nonlinear coordinates such as logarithmic
values and can be composed independently of either parametrisation. Both
parametrisations are designed to occupy the ``coeffs`` leaf of a `ParametricBasis`.
Their `latent` values remain real PyTree leaves for ordinary Zodiax path updates, and
the containing basis resolves them to physical coefficients only when evaluated.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..base import Base
from .parametrics import Parametric

__all__ = [
    "ParameterTransform",
    "IdentityTransform",
    "LogTransform",
    "CompositeTransform",
    "Reparametrisation",
    "Selection",
]


def _validate_origin(origin) -> Array:
    """Standardise a non-empty physical coefficient array."""
    origin = dlu.to_value(origin)
    if origin.ndim == 0 or origin.size == 0:
        raise ValueError("origin must be a non-empty coefficient array.")
    return origin


def _validate_latent(latent, size, dtype) -> Array:
    """Standardise latent coordinates with a fixed trailing dimension."""
    latent = dlu.to_value(latent, dtype)
    if latent.ndim == 0 or latent.shape[-1] != size:
        raise ValueError(
            f"latent must have trailing dimension {size}, got {latent.shape}."
        )
    return latent


def _validate_coeffs(coeffs, shape, dtype) -> Array:
    """Standardise coefficients ending in a fixed physical shape."""
    coeffs = dlu.to_value(coeffs, dtype)
    ndim = len(shape)
    if coeffs.ndim < ndim or coeffs.shape[-ndim:] != shape:
        raise ValueError(
            "coeffs trailing dimensions must match origin.shape. "
            f"Expected {shape}, got {coeffs.shape}."
        )
    return coeffs


def _validate_transform(transform, origin):
    """Standardise and validate a physical-coordinate transformation."""
    transform = IdentityTransform() if transform is None else transform
    if not isinstance(transform, ParameterTransform):
        raise TypeError("transform must be a ParameterTransform or None.")
    transform.validate(origin)
    return transform


class ParameterTransform(Base):
    """Extension contract for reversible physical-parameter coordinates.

    A parameter transform maps physical values to optimisation coordinates through
    `encode` and returns them to the physical domain through `decode`. Implementations
    must preserve every input axis, support arbitrary leading batch axes, and remain
    compatible with JAX transformations. `validate` is a setup-time diagnostic;
    evaluation deliberately performs no runtime domain checks after immutable model
    updates.
    """

    @abstractmethod
    def encode(self, value) -> Array:
        """Map a physical array to shape-preserving optimisation coordinates."""

    @abstractmethod
    def decode(self, value) -> Array:
        """Map an encoded array back to shape-preserving physical values."""

    def validate(self, value) -> None:
        """Validate shape preservation and finite round-trip coordinates.

        Parameters
        ----------
        value : Array
            Representative physical values in the transform domain. Validation is
            intended for concrete construction-time arrays, not traced evaluation.

        Raises
        ------
        ValueError
            If encoding or decoding changes the input shape or produces non-finite
            values, or decoding does not invert encoding within numerical tolerance.
        """
        value = dlu.to_value(value)
        encoded = np.asarray(self.encode(value))
        decoded = np.asarray(self.decode(encoded))

        if encoded.shape != value.shape:
            raise ValueError(
                "transform.encode must preserve shape. "
                f"Expected {value.shape}, got {encoded.shape}."
            )
        if decoded.shape != value.shape:
            raise ValueError(
                "transform.decode must preserve shape. "
                f"Expected {value.shape}, got {decoded.shape}."
            )
        if not bool(np.all(np.isfinite(encoded))):
            raise ValueError("transform.encode must produce finite coordinates.")
        if not bool(np.all(np.isfinite(decoded))):
            raise ValueError("transform.decode must produce finite physical values.")
        if not bool(np.allclose(decoded, value, rtol=1e-5, atol=0)):
            raise ValueError("transform.decode must invert transform.encode.")


class IdentityTransform(ParameterTransform):
    """Preserve physical values as their optimisation coordinates."""

    def __init__(self):
        """Initialise a parameter transform with identity encoding."""

    def encode(self, value) -> Array:
        """Return physical values unchanged as optimisation coordinates."""
        return np.asarray(value)

    def decode(self, value) -> Array:
        """Return optimisation coordinates unchanged as physical values."""
        return np.asarray(value)


class LogTransform(ParameterTransform):
    """Represent strictly positive physical values in logarithmic coordinates.

    Encoding applies the natural logarithm and decoding applies the exponential.
    Additive coordinate displacements therefore produce multiplicative physical
    changes. For an origin ``x0`` and coordinate displacement ``delta``, the realised
    value is ``x0 * exp(delta)``.

    Notes
    -----
    This class follows `jax.numpy.log`, so ``log`` denotes the natural logarithm.
    This is distinct from the `Source` unit string ``"log"``, which denotes base-10
    values; source units use ``"ln"`` for the natural logarithm.

    Examples
    --------
    Reparameterise positive coefficients with additive logarithmic coordinates:

    ```python
    import jax.numpy as np

    import dLux as dl

    origin = np.asarray([2.0, 4.0])
    coeffs = dl.Reparametrisation(
        origin,
        np.eye(2),
        latent=np.log(np.asarray([1.5, 0.5])),
        transform=dl.LogTransform(),
    )
    values = coeffs.evaluate()  # [3.0, 2.0]
    ```
    """

    def __init__(self):
        """Initialise a natural-logarithmic parameter transform."""

    def encode(self, value) -> Array:
        """Return the natural logarithm of positive physical values."""
        return np.log(value)

    def decode(self, value) -> Array:
        """Exponentiate logarithmic coordinates into positive physical values."""
        return np.exp(value)

    def validate(self, value) -> None:
        """Validate that representative physical values are strictly positive.

        Parameters
        ----------
        value : Array
            Concrete physical values required to be strictly greater than zero.

        Raises
        ------
        ValueError
            If any physical value is zero or negative, or the inherited round-trip
            validation fails.
        """
        value = dlu.to_value(value)
        if not bool(np.all(value > 0)):
            raise ValueError("LogTransform requires strictly positive values.")
        super().validate(value)


class CompositeTransform(ParameterTransform):
    """Compose reversible parameter transforms in a defined order.

    `encode` applies the stored transforms in insertion order. `decode` applies their
    inverses in reverse order, so the complete mapping remains reversible. Named
    transforms retain ordinary Zodiax paths, while unnamed transforms receive class
    names with numeric suffixes where required. An empty composition is the identity.
    Explicit names must be non-empty strings, cannot contain spaces or periods, and
    cannot shadow mapping attributes such as ``items`` or ``values``. These names
    would otherwise conflict with nested Zodiax paths.

    Examples
    --------
    Construct a named composition and round-trip positive values:

    ```python
    import jax.numpy as np

    import dLux as dl

    transform = dl.CompositeTransform(
        [("identity", dl.IdentityTransform()), ("log", dl.LogTransform())]
    )
    physical = np.asarray([2.0, 4.0])
    encoded = transform.encode(physical)
    recovered = transform.decode(encoded)
    ```

    Parameters
    ----------
    transforms : mapping or sequence
        Named mapping, ``(name, transform)`` sequence, or transform sequence in
        encoding order. Every value must derive from `ParameterTransform`.
    """

    transforms: dict

    def __init__(self, transforms=()):
        """Initialise an ordered composition of parameter transforms.

        Parameters
        ----------
        transforms : mapping or sequence
            Named or unnamed `ParameterTransform` objects in encoding order.

        Raises
        ------
        TypeError
            If any entry is not a `ParameterTransform`, or an explicit name is not a
            string.
        ValueError
            If a named entry is not a ``(name, transform)`` pair, or if its name is
            empty, contains spaces or periods, or shadows a mapping attribute.
        """
        if isinstance(transforms, Mapping):
            transforms = list(transforms.items())
        else:
            transforms = list(transforms)

        for item in transforms:
            if not isinstance(item, tuple):
                continue
            if len(item) != 2:
                raise ValueError("named transforms must be (name, transform) pairs.")
            name, _ = item
            if not isinstance(name, str):
                raise TypeError("transform names must be strings.")
            if not name:
                raise ValueError("transform names must be non-empty.")
            if "." in name:
                raise ValueError("transform names cannot contain periods.")
            if hasattr(OrderedDict, name):
                raise ValueError("transform names cannot shadow mapping attributes.")

        self.transforms = dlu.list2dictionary(transforms, True, ParameterTransform)

    def encode(self, value) -> Array:
        """Encode physical values through each transform in insertion order."""
        for transform in self.transforms.values():
            value = transform.encode(value)
        return np.asarray(value)

    def decode(self, value) -> Array:
        """Decode coordinates through each transform in reverse insertion order."""
        for transform in reversed(tuple(self.transforms.values())):
            value = transform.decode(value)
        return np.asarray(value)

    def validate(self, value) -> None:
        """Validate every transform against its corresponding encoded values.

        Parameters
        ----------
        value : Array
            Representative physical values supplied to the first transform.

        Raises
        ------
        ValueError
            If any transform rejects its input, changes shape, or produces non-finite
            values, or the complete composition fails its inverse round trip.
        """
        original = dlu.to_value(value)
        value = original
        for transform in self.transforms.values():
            transform.validate(value)
            value = transform.encode(value)

        # Component tolerances can accumulate, so also validate the complete chain
        super().validate(original)


class Reparametrisation(Parametric):
    """Map latent coordinates through a reversible coefficient subspace.

    Physical coefficients are evaluated by encoding `origin`, adding ``matrix @
    latent`` in that coordinate system, and decoding the result. The identity
    transform recovers the affine mapping ``origin + matrix @ latent``; a
    `LogTransform` instead produces multiplicative physical changes. The matrix has
    one row per encoded coefficient and one column per latent direction. Leading
    latent axes are preserved as batch axes by this mapping.

    A containing basis retains its own documented coefficient-vectorisation
    contract; Fourier and spectral bases naturally preserve arbitrary shared batches,
    while generic bases pair a leading coefficient batch with a matching leading
    basis axis.

    `to_latent` performs a least-squares inverse in encoded-coordinate space. It
    exactly recovers coordinates for independent matrix columns and returns
    minimum-norm coordinates otherwise. Coefficients outside the represented
    subspace are projected by `project` and `initialise`. With the identity transform,
    each matrix column has the physical units of the target coefficients per latent
    coordinate; nonlinear transforms instead define the matrix units through their
    encoded coordinates.

    When stored as ``basis.coeffs``, the attribute remains this parametrisation rather
    than pretending to be a computed array. Consequently ``basis.set("latent", value)``
    and ``basis.set("coeffs.latent", value)`` update the real nested leaf, while
    ``basis.set("coeffs", array)`` deliberately replaces the complete
    reparametrisation. Call `evaluate`, `to_coeffs`, or `resolve` to inspect the
    realised physical coefficients. ``origin``, ``matrix``, and array fields on custom
    transforms are also differentiable leaves; select the ``latent`` path explicitly
    when they should remain fixed.

    Examples
    --------
    Reparameterise a Fourier basis around fitted coefficients:

    ```python
    import jax.numpy as np

    import dLux as dl

    origin = np.zeros((4, 4))
    matrix = np.eye(origin.size)[:, :3]
    coeffs = dl.Reparametrisation(origin, matrix)
    basis = dl.FourierBasis(npix=64, n_modes=4, coeffs=coeffs)

    # Optimise the real nested leaf through the ordinary Zodiax path interface
    basis = basis.set("latent", np.asarray([0.1, -0.2, 0.3]))
    values = basis.evaluate()
    ```

    Parameters
    ----------
    origin : Array
        Physical coefficient anchor with arbitrary non-empty shape.
    matrix : Array
        Two-dimensional latent-to-encoded-coordinate map with shape
        ``(origin.size, n_latent)``.
    latent : Array or None
        Latent coordinates with trailing dimension ``n_latent``. Defaults to zero.
        Leading dimensions vectorise the shared matrix.
    transform : ParameterTransform or None
        Reversible mapping between physical and optimisation coordinates. Defaults to
        `IdentityTransform`.
    """

    origin: Array
    matrix: Array
    latent: Array
    transform: ParameterTransform

    def __init__(self, origin, matrix, latent=None, *, transform=None):
        """Initialise a coefficient-space reparametrisation.

        Parameters
        ----------
        origin : Array
            Physical coefficient anchor with arbitrary non-empty shape.
        matrix : Array
            Two-dimensional latent-to-encoded-coordinate map with one row per
            flattened physical coefficient.
        latent : Array or None
            Latent coordinates, optionally with leading batch axes. The trailing
            dimension must match the matrix column count.
        transform : ParameterTransform or None
            Shape-preserving physical-coordinate transform. `None` selects identity
            coordinates.

        Raises
        ------
        ValueError
            If ``origin`` is empty or scalar, ``matrix`` is not two-dimensional,
            its row count differs from ``origin.size``, it has no columns or more
            columns than rows, the latent trailing dimension does not match, or the
            origin is outside the transform domain.
        TypeError
            If ``transform`` is not a `ParameterTransform` or `None`.
        """
        # Standardise the physical anchor and coordinate transform
        origin = _validate_origin(origin)
        transform = _validate_transform(transform, origin)

        # Standardise the latent-to-coordinate matrix
        matrix = dlu.to_value(matrix)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if matrix.shape[0] != origin.size:
            raise ValueError(
                "matrix row count must equal origin.size. "
                f"Expected {origin.size}, got {matrix.shape[0]}."
            )
        if matrix.shape[1] < 1 or matrix.shape[1] > matrix.shape[0]:
            raise ValueError(
                "matrix must contain between one and origin.size latent columns."
            )

        # Initialise latent coordinates at the physical anchor when omitted
        dtype = np.result_type(transform.encode(origin), matrix)
        matrix = dlu.to_value(matrix, dtype)
        if latent is None:
            latent = np.zeros(matrix.shape[1], dtype=dtype)
        latent = _validate_latent(latent, matrix.shape[1], dtype)

        self.origin = origin
        self.matrix = matrix
        self.latent = latent
        self.transform = transform

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the native physical coefficient shape as a static tuple."""
        return self.origin.shape

    @property
    def size(self) -> int:
        """Return the number of flattened physical coefficients."""
        return self.origin.size

    @property
    def ndim(self) -> int:
        """Return the number of native physical coefficient axes."""
        return self.origin.ndim

    def to_coeffs(self, latent=None) -> Array:
        """Map latent coordinates to physical coefficients.

        Parameters
        ----------
        latent : Array or None
            Coordinates with trailing dimension ``matrix.shape[1]``. The stored
            coordinates are used when omitted.

        Returns
        -------
        coeffs : Array
            Physical coefficients with shape ``latent.shape[:-1] + shape``.

        Raises
        ------
        ValueError
            If the latent trailing dimension does not match the matrix.
        """
        if latent is None:
            latent = self.latent
        latent = _validate_latent(latent, self.matrix.shape[1], self.matrix.dtype)

        # Add the shared linear displacement in encoded-coordinate space
        delta = np.einsum("nk,...k->...n", self.matrix, latent)
        origin = self.transform.encode(self.origin).reshape(-1)
        encoded = (origin + delta).reshape(latent.shape[:-1] + self.shape)

        # Decode the native coefficient axes into physical values
        return self.transform.decode(encoded)

    def to_latent(self, coeffs) -> Array:
        """Return least-squares latent coordinates for physical coefficients.

        Physical inputs may have leading batch axes and must end in `shape`. Values
        outside the represented subspace are mapped to their least-squares projection
        in encoded-coordinate space; rank-deficient maps return minimum-norm
        coordinates.

        Parameters
        ----------
        coeffs : Array
            Physical values within the transform domain and with trailing dimensions
            equal to `shape`.

        Returns
        -------
        latent : Array
            Coordinates with the same leading batch axes and trailing dimension
            ``matrix.shape[1]``.

        Raises
        ------
        ValueError
            If the physical trailing dimensions do not match `shape`.
        """
        coeffs = _validate_coeffs(coeffs, self.shape, self.origin.dtype)
        batch_shape = coeffs.shape[: -len(self.shape)]

        # Arrange encoded batches as independent least-squares right-hand sides
        encoded = self.transform.encode(coeffs).reshape((-1, self.size))
        origin = self.transform.encode(self.origin).reshape((1, -1))
        delta = encoded - origin
        latent = np.linalg.lstsq(self.matrix, delta.T, rcond=None)[0].T
        return latent.reshape(batch_shape + (self.matrix.shape[1],))

    def initialise(self, coeffs) -> Reparametrisation:
        """Return a copy whose latent coordinates best represent ``coeffs``.

        Parameters
        ----------
        coeffs : Array
            Physical values within the transform domain and with trailing dimensions
            equal to `shape`.

        Returns
        -------
        parametrisation : Reparametrisation
            Immutable copy with least-squares latent coordinates. Its evaluated value
            equals `project(coeffs)` and may differ from off-subspace input according
            to encoded-coordinate distance.
        """
        return self.set(latent=self.to_latent(coeffs))

    def project(self, coeffs) -> Array:
        """Project physical coefficients onto the represented coordinate subspace.

        Parameters
        ----------
        coeffs : Array
            Physical values within the transform domain and with trailing dimensions
            equal to `shape`.

        Returns
        -------
        projected : Array
            Encoded-coordinate least-squares projection with the same shape as
            ``coeffs``.
        """
        return self.to_coeffs(self.to_latent(coeffs))

    def evaluate(self, **context: Any) -> Array:
        """Evaluate the stored latent coordinates as physical coefficients.

        Additional named context is accepted for Parametric interoperability and is
        not consumed. The return shape is ``latent.shape[:-1] + shape``.
        """
        return self.to_coeffs()


class Selection(Parametric):
    """Select physical coefficients to vary while fixing all others.

    ``True`` mask elements identify active coefficients in C-order flattening. Each
    latent coordinate is an additive displacement from the encoded active `origin`;
    inactive physical values remain exactly anchored and are never transformed. With
    identity coordinates, this is the sparse special case of `Reparametrisation` with
    columns selected from the identity matrix.

    Examples
    --------
    Vary only selected Fourier coefficients:

    ```python
    import jax.numpy as np

    import dLux as dl

    origin = np.zeros((4, 4))
    mask = np.zeros_like(origin, dtype=bool).at[1:3, 1:3].set(True)
    coeffs = dl.Selection(origin, mask)
    basis = dl.FourierBasis(npix=64, n_modes=4, coeffs=coeffs)

    # Latents follow the True entries of mask.ravel() in C order
    basis = basis.set("latent", np.asarray([0.1, -0.2, 0.3, -0.4]))
    values = basis.evaluate()
    ```

    Parameters
    ----------
    origin : Array
        Physical coefficient anchor with arbitrary non-empty shape.
    mask : Array
        Boolean array matching ``origin.shape``; ``True`` elements are active.
    latent : Array or None
        Active encoded-coordinate displacements, optionally with leading batch axes.
        Defaults to zero.
    transform : ParameterTransform or None
        Reversible mapping applied to the active coefficient vector. Defaults to
        `IdentityTransform`.
    """

    origin: Array
    indices: Array
    latent: Array
    transform: ParameterTransform

    def __init__(self, origin, mask, latent=None, *, transform=None):
        """Initialise a masked coefficient selection.

        Parameters
        ----------
        origin : Array
            Physical coefficient anchor with arbitrary non-empty shape.
        mask : Array
            Boolean mask matching ``origin.shape``. ``True`` elements are assigned
            latent displacements in C-order.
        latent : Array or None
            Active encoded-coordinate displacements with trailing dimension equal to
            ``mask.sum()``.
        transform : ParameterTransform or None
            Shape-preserving transform applied only to active physical values. `None`
            selects identity coordinates.

        Raises
        ------
        ValueError
            If ``origin`` is empty or scalar, ``mask`` has a different shape or no
            active elements, the latent trailing dimension does not equal the number
            of active coefficients, or an active origin is outside the transform
            domain.
        TypeError
            If ``transform`` is not a `ParameterTransform` or `None`.
        """
        # Resolve the physical anchor and static-size active integer indices
        origin = _validate_origin(origin)
        mask = dlu.to_value(mask, bool)
        if mask.shape != origin.shape:
            raise ValueError(
                "mask shape must match origin.shape. "
                f"Expected {origin.shape}, got {mask.shape}."
            )
        indices = np.flatnonzero(mask.reshape(-1))
        if indices.size == 0:
            raise ValueError("mask must select at least one coefficient.")

        # Validate the transform only over active physical coefficients
        active = origin.reshape(-1)[indices]
        transform = _validate_transform(transform, active)

        # Initialise encoded-coordinate displacements at zero when omitted
        dtype = transform.encode(active).dtype
        if latent is None:
            latent = np.zeros(indices.size, dtype=dtype)
        latent = _validate_latent(latent, indices.size, dtype)

        self.origin = origin
        self.indices = indices
        self.latent = latent
        self.transform = transform

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the native physical coefficient shape as a static tuple."""
        return self.origin.shape

    @property
    def size(self) -> int:
        """Return the number of flattened physical coefficients."""
        return self.origin.size

    @property
    def ndim(self) -> int:
        """Return the number of native physical coefficient axes."""
        return self.origin.ndim

    @property
    def mask(self) -> Array:
        """Return the boolean active-coefficient mask with physical `shape`."""
        mask = np.zeros(self.size, dtype=bool).at[self.indices].set(True)
        return mask.reshape(self.shape)

    @property
    def matrix(self) -> Array:
        """Return the equivalent dense latent-to-coordinate selection matrix.

        The returned shape is ``(size, latent.shape[-1])``. It is materialised only
        on access; evaluation uses sparse integer scatter operations.
        """
        return np.eye(self.size, dtype=self.latent.dtype)[:, self.indices]

    def to_coeffs(self, latent=None) -> Array:
        """Map selected coordinate displacements to physical coefficients.

        Parameters
        ----------
        latent : Array or None
            Encoded-coordinate displacements with trailing dimension ``indices.size``.
            The stored coordinates are used when omitted.

        Returns
        -------
        coeffs : Array
            Physical coefficients with shape ``latent.shape[:-1] + shape``.

        Raises
        ------
        ValueError
            If the latent trailing dimension does not match the active count.
        """
        if latent is None:
            latent = self.latent
        latent = _validate_latent(latent, self.indices.size, self.latent.dtype)

        # Encode and displace only the active physical coefficients
        active = self.origin.reshape(-1)[self.indices]
        encoded = self.transform.encode(active) + latent
        active = self.transform.decode(encoded)

        # Scatter active values into an otherwise untouched physical anchor
        batch_shape = latent.shape[:-1]
        coeffs = np.broadcast_to(self.origin.reshape(-1), batch_shape + (self.size,))
        coeffs = coeffs.at[..., self.indices].set(active)
        return coeffs.reshape(batch_shape + self.shape)

    def to_latent(self, coeffs) -> Array:
        """Extract active encoded-coordinate displacements from coefficients.

        Parameters
        ----------
        coeffs : Array
            Physical values whose active entries lie within the transform domain and
            whose trailing dimensions equal `shape`. Inactive entries are ignored.

        Returns
        -------
        latent : Array
            Active coordinate displacements in C-order mask order, retaining input
            batch axes.
        """
        coeffs = _validate_coeffs(coeffs, self.shape, self.origin.dtype)
        batch_shape = coeffs.shape[: -len(self.shape)]
        active = coeffs.reshape(batch_shape + (self.size,))[..., self.indices]
        origin = self.origin.reshape(-1)[self.indices]
        return self.transform.encode(active) - self.transform.encode(origin)

    def initialise(self, coeffs) -> Selection:
        """Return a copy initialised from the active values of ``coeffs``.

        Inactive input values are discarded. The immutable returned copy evaluates
        to `project(coeffs)`, while the original selection is unchanged.

        Parameters
        ----------
        coeffs : Array
            Physical values whose active entries lie within the transform domain and
            whose trailing dimensions equal `shape`. Inactive entries are discarded.

        Returns
        -------
        selection : Selection
            Immutable copy with active encoded-coordinate displacements extracted
            from ``coeffs``.
        """
        return self.set(latent=self.to_latent(coeffs))

    def project(self, coeffs) -> Array:
        """Keep active coefficients and reset inactive values to the origin.

        The returned array has the same shape as ``coeffs``; leading batch axes are
        preserved.

        Parameters
        ----------
        coeffs : Array
            Physical values whose active entries lie within the transform domain and
            whose trailing dimensions equal `shape`. Inactive entries are reset to
            `origin`.

        Returns
        -------
        projected : Array
            Values retaining active input entries and anchored inactive entries.
        """
        return self.to_coeffs(self.to_latent(coeffs))

    def evaluate(self, **context: Any) -> Array:
        """Evaluate stored active displacements as physical coefficients.

        Additional named context is accepted for Parametric interoperability and is
        not consumed. The return shape is ``latent.shape[:-1] + shape``.
        """
        return self.to_coeffs()
