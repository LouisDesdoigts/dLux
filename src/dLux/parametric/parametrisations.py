"""Latent coefficient-space reparametrisations.

`Reparametrisation` represents a general affine subspace, while `Selection` varies a
masked subset of coefficients without materialising a dense matrix. Both are designed
to occupy the ``coeffs`` leaf of a `ParametricBasis`. Their `latent` values remain real
PyTree leaves for ordinary Zodiax path updates, and the containing basis resolves them
to physical coefficients only when evaluated.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from .parametrics import Parametric

__all__ = ["Reparametrisation", "Selection"]


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


class Reparametrisation(Parametric):
    """Map latent coordinates into an affine coefficient subspace.

    Physical coefficients are evaluated as ``origin + matrix @ latent`` after
    flattening ``origin`` in C order. The matrix therefore has one row per physical
    coefficient and one column per latent direction. Leading latent axes are
    preserved as batch axes by this mapping. A containing basis retains its own
    documented coefficient-vectorisation contract; Fourier and spectral bases
    naturally preserve arbitrary shared batches, while generic bases pair a leading
    coefficient batch with a matching leading basis axis.

    `to_latent` performs a least-squares inverse. It exactly recovers coordinates for
    independent matrix columns and returns minimum-norm coordinates otherwise.
    Coefficients outside the represented subspace are projected by `project` and
    `initialise`. ``origin`` and each matrix column have the physical units of the
    target coefficients per unit of their corresponding latent coordinate.

    When stored as ``basis.coeffs``, the attribute remains this parametrisation rather
    than pretending to be a computed array. Consequently ``basis.set("latent", value)``
    and ``basis.set("coeffs.latent", value)`` update the real nested leaf, while
    ``basis.set("coeffs", array)`` deliberately replaces the complete
    reparametrisation. Call `evaluate`, `to_coeffs`, or `resolve` to inspect the
    realised physical coefficients. ``origin`` and ``matrix`` are also differentiable
    array leaves; select the ``latent`` path explicitly when they should remain fixed.

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
        Two-dimensional latent-to-physical map with shape
        ``(origin.size, n_latent)``.
    latent : Array or None
        Latent coordinates with trailing dimension ``n_latent``. Defaults to zero.
        Leading dimensions vectorise the shared matrix.
    """

    origin: Array
    matrix: Array
    latent: Array

    def __init__(self, origin, matrix, latent=None):
        """Initialise an affine coefficient reparametrisation.

        Parameters
        ----------
        origin : Array
            Physical coefficient anchor with arbitrary non-empty shape.
        matrix : Array
            Two-dimensional latent-to-physical map with one row per flattened
            physical coefficient.
        latent : Array or None
            Latent coordinates, optionally with leading batch axes. The trailing
            dimension must match the matrix column count.

        Raises
        ------
        ValueError
            If ``origin`` is empty or scalar, ``matrix`` is not two-dimensional,
            its row count differs from ``origin.size``, it has no columns or more
            columns than rows, or the latent trailing dimension does not match.
        """
        # Standardise the physical anchor and latent-to-physical matrix
        origin = _validate_origin(origin)
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
        dtype = np.result_type(origin, matrix)
        if latent is None:
            latent = np.zeros(matrix.shape[1], dtype=dtype)
        latent = _validate_latent(latent, matrix.shape[1], dtype)

        self.origin = origin
        self.matrix = matrix
        self.latent = latent

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

        # Apply the shared linear map and restore the physical coefficient axes
        delta = np.einsum("nk,...k->...n", self.matrix, latent)
        coeffs = self.origin.reshape(-1) + delta
        return coeffs.reshape(latent.shape[:-1] + self.shape)

    def to_latent(self, coeffs) -> Array:
        """Return least-squares latent coordinates for physical coefficients.

        Physical inputs may have leading batch axes and must end in `shape`. Values
        outside the represented affine subspace are mapped to their least-squares
        projection; rank-deficient maps return minimum-norm coordinates.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

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

        # Arrange every batch item as one least-squares right-hand side
        delta = coeffs.reshape((-1, self.size)) - self.origin.reshape((1, -1))
        latent = np.linalg.lstsq(self.matrix, delta.T, rcond=None)[0].T
        return latent.reshape(batch_shape + (self.matrix.shape[1],))

    def initialise(self, coeffs) -> Reparametrisation:
        """Return a copy whose latent coordinates best represent ``coeffs``.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

        Returns
        -------
        parametrisation : Reparametrisation
            Immutable copy with least-squares latent coordinates. Its evaluated
            value equals `project(coeffs)` and may differ from off-subspace input.
        """
        return self.set(latent=self.to_latent(coeffs))

    def project(self, coeffs) -> Array:
        """Project physical coefficients onto the represented affine subspace.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

        Returns
        -------
        projected : Array
            Least-squares projection with the same shape as ``coeffs``.
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
    latent coordinate is an additive displacement from `origin`; inactive values
    remain exactly anchored. This is the sparse special case of `Reparametrisation`
    with columns selected from the identity matrix.

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
        Active coefficient displacements, optionally with leading batch axes.
        Defaults to zero.
    """

    origin: Array
    indices: Array
    latent: Array

    def __init__(self, origin, mask, latent=None):
        """Initialise a masked coefficient selection.

        Parameters
        ----------
        origin : Array
            Physical coefficient anchor with arbitrary non-empty shape.
        mask : Array
            Boolean mask matching ``origin.shape``. ``True`` elements are assigned
            latent displacements in C-order.
        latent : Array or None
            Active displacements with trailing dimension equal to ``mask.sum()``.

        Raises
        ------
        ValueError
            If ``origin`` is empty or scalar, ``mask`` has a different shape or no
            active elements, or the latent trailing dimension does not equal the
            number of active coefficients.
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

        # Initialise selected coefficient displacements at zero when omitted
        if latent is None:
            latent = np.zeros(indices.size, dtype=origin.dtype)
        latent = _validate_latent(latent, indices.size, origin.dtype)

        self.origin = origin
        self.indices = indices
        self.latent = latent

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
        """Return the equivalent dense latent-to-physical selection matrix.

        The returned shape is ``(size, latent.shape[-1])``. It is materialised only
        on access; evaluation uses sparse integer scatter operations.
        """
        return np.eye(self.size, dtype=self.origin.dtype)[:, self.indices]

    def to_coeffs(self, latent=None) -> Array:
        """Map selected latent displacements to physical coefficients.

        Parameters
        ----------
        latent : Array or None
            Displacements with trailing dimension ``indices.size``. The stored
            coordinates are used when omitted.

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
        latent = _validate_latent(latent, self.indices.size, self.origin.dtype)

        # Broadcast the anchor over leading latent axes and scatter displacements
        batch_shape = latent.shape[:-1]
        coeffs = np.broadcast_to(self.origin.reshape(-1), batch_shape + (self.size,))
        coeffs = coeffs.at[..., self.indices].add(latent)
        return coeffs.reshape(batch_shape + self.shape)

    def to_latent(self, coeffs) -> Array:
        """Extract active displacements from physical coefficients.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

        Returns
        -------
        latent : Array
            Active displacements in C-order mask order, retaining input batch axes.
        """
        coeffs = _validate_coeffs(coeffs, self.shape, self.origin.dtype)
        batch_shape = coeffs.shape[: -len(self.shape)]
        delta = coeffs.reshape(batch_shape + (self.size,)) - self.origin.reshape(-1)
        return delta[..., self.indices]

    def initialise(self, coeffs) -> Selection:
        """Return a copy initialised from the active values of ``coeffs``.

        Inactive input values are discarded. The immutable returned copy evaluates
        to `project(coeffs)`, while the original selection is unchanged.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

        Returns
        -------
        selection : Selection
            Immutable copy with active displacements extracted from ``coeffs``.
        """
        return self.set(latent=self.to_latent(coeffs))

    def project(self, coeffs) -> Array:
        """Keep active coefficients and reset inactive values to the origin.

        The returned array has the same shape as ``coeffs``; leading batch axes are
        preserved.

        Parameters
        ----------
        coeffs : Array
            Physical values with trailing dimensions equal to `shape`.

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
