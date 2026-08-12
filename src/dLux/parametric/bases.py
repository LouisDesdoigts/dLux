"""Core parameterisation and non-polynomial basis implementations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..grids import BaseCoordTransform, PasteSpec
from .parametrics import Parametric

__all__ = [
    "ParametricBasis",
    "Basis",
    "PastedBasis",
    "ImplicitBasis",
    "CoordBasis",
    "CLIMBBasis",
    "FourierBasis",
    "SplineBasis",
]

# TODO: Add Gaussian


def _resolve_coeffs(coeffs, coefficients):
    """Resolve the deprecated ``coefficients`` constructor keyword."""
    if coefficients is None:
        return coeffs
    if coeffs is not None:
        raise ValueError("Provide only one of coeffs or coefficients.")
    # Keep compatibility lazy: it imports the parametric package for legacy aliases.
    from ..compatibility import warn_deprecated

    warn_deprecated(
        "coefficients argument",
        "coeffs",
        "`Class(coefficients=value)` -> `Class(coeffs=value)`",
        stacklevel=4,
    )
    return coefficients


class ParametricBasis(Parametric):
    """Base contract for coefficient-weighted basis parameterisations."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def coefficients(self: ParametricBasis) -> Array:
        """Return ``coeffs`` through the deprecated attribute alias.

        Access emits a migration warning and the alias will be removed in dLux 0.17.
        """
        # Keep compatibility lazy to avoid the core/legacy import cycle.
        from ..compatibility import warn_deprecated

        warn_deprecated(
            ".coefficients attribute",
            ".coeffs",
            "`basis.coefficients` -> `basis.coeffs`",
            stacklevel=3,
        )
        return self.coeffs

    @property
    def c(self: ParametricBasis) -> Array:
        """Return ``coeffs`` through the compact scientific alias ``c``."""
        return self.coeffs

    @property
    def alpha(self: ParametricBasis) -> Array:
        """Return ``coeffs`` through the conventional scientific alias ``alpha``."""
        return self.coeffs

    def _set_coeffs(
        self: ParametricBasis, coeffs: Array, shape: tuple[int, ...]
    ) -> None:
        """Validate and assign coefficients with a native basis shape."""
        # Standardise coefficients and their native shape
        coeffs = dlu.to_value(coeffs)
        shape = tuple(shape)

        # Validate trailing basis dimensions while allowing compact vectors
        compact = shape == (1,) and coeffs.ndim == 1
        if not compact and coeffs.shape[-len(shape) :] != shape:
            raise ValueError(
                "Coefficient shape trailing dimensions must match the basis "
                "dimensions. "
                f"Expected {shape}, got {coeffs.shape}."
            )

        # Store the coefficients and native contraction shape
        self.coeffs = coeffs
        self.shape = shape

    def evaluate_basis(self, basis: Array) -> Array:
        """Contract stored coefficients against explicit basis vectors.

        ``basis`` begins with the native coefficient ``shape`` and is followed by
        sampled output axes. Coefficients may be shared or contain one matching
        leading vectorisation axis; the returned array retains all uncontracted axes.
        """
        # Evaluate coefficients shared across the complete basis
        if self.coeffs.shape == self.shape:
            return dlu.eval_basis(basis, self.coeffs)

        # Vectorise a compact single-mode basis over coefficients
        if self.shape == (1,) and self.coeffs.ndim == 1:
            evaluate = lambda coefficient: dlu.eval_basis(basis, coefficient[None])
            return jax.vmap(evaluate)(self.coeffs)

        # Validate one leading coefficient axis
        if self.coeffs.ndim != len(self.shape) + 1:
            raise ValueError("Only one leading coefficient axis is supported.")
        axis = len(self.shape)
        if basis.shape[axis] != self.coeffs.shape[0]:
            raise ValueError(
                "The leading coefficient axis must match the leading basis axis."
            )

        # Align and vectorise matching coefficient and basis axes
        basis = np.moveaxis(basis, axis, 0)
        return jax.vmap(dlu.eval_basis)(basis, self.coeffs)

    @abstractmethod
    def solve_basis(self: ParametricBasis, value: Array, **kwargs: Any) -> Array:
        """Solve for coefficients representing ``value`` in this basis.

        Concrete bases define how basis vectors are obtained. ``value`` must match
        the sampled output shape and the return value has the native coefficient
        ``shape``.
        """


class Basis(ParametricBasis):
    """Parameterise an explicitly sampled basis array.

    Parameters
    ----------
    basis : Array
        Basis vectors stored along the leading ``shape`` dimensions.
    coeffs : Array or None
        Coefficients contracted against the basis dimensions.
    shape : tuple[int, ...] or None
        Native basis dimensions. Required when ``coeffs`` is omitted.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array

    def __init__(
        self: Basis,
        basis: Array,
        coeffs: Array = None,
        shape: tuple[int, ...] = None,
        *,
        coefficients: Array = None,
    ):
        """Initialise an explicitly sampled linear basis.

        Parameters
        ----------
        basis : Array
            Basis values with coefficient axes followed by sampled output axes.
        coeffs : Array or None
            Coefficients matching ``shape``; defaults to zeros.
        shape : tuple[int, ...] or None
            Number and sizes of coefficient axes, inferred when omitted.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        # Resolve coefficients and their native basis shape
        self.basis = dlu.to_value(basis)
        coeffs = _resolve_coeffs(coeffs, coefficients)
        if coeffs is None:
            if shape is None:
                raise ValueError("Provide either coeffs or shape.")
            coeffs = np.zeros(shape)
        else:
            coeffs = dlu.to_value(coeffs)
            if shape is None:
                shape = coeffs.shape

        # Validate and store the explicit basis contract
        if self.basis.shape[: len(shape)] != shape:
            raise ValueError(
                "The leading basis dimensions must match the coefficient shape."
            )
        self._set_coeffs(coeffs, shape)

    def evaluate(self: Basis, **kwargs: Any) -> Array:
        """Contract stored coefficients against the explicit basis.

        Native coefficient dimensions are removed and all sampled output dimensions
        are retained. Additional context is accepted for parametric interoperability
        but is not consumed.
        """
        return self.evaluate_basis(self.basis)

    def solve_basis(self: Basis, value: Array, **kwargs: Any) -> Array:
        """Solve the explicit basis for coefficients representing ``value``.

        ``value`` must match the basis sampled output shape. The returned least-squares
        coefficients have the native basis ``shape``.
        """
        return dlu.solve_basis(value, self.basis)


class PastedBasis(ParametricBasis):
    """Parameterise compact local bases pasted into a common output grid.

    Parameters
    ----------
    basis : Array
        Basis stamps with shape ``(n_stamps, *basis_shape, ny, nx)``.
    spec : PasteSpec
        Placement and sampling contract shared by every compact stamp.
    coeffs : Array or None
        Per-stamp coefficients with shape ``(n_stamps, *basis_shape)``. They default
        to zero.
    method : str
        ``"scan"`` for bounded-memory placement or ``"scatter"`` for parallel
        placement using flattened pixel indices.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    spec: PasteSpec
    method: str

    def __init__(self, basis, spec, coeffs=None, method="scan", *, coefficients=None):
        """Initialise compact basis stamps and their placement.

        Parameters
        ----------
        basis : Array
            Per-stamp basis values with final two local spatial axes.
        spec : PasteSpec
            Fixed placement geometry for the compact stamps.
        coeffs : Array or None
            Shared or per-stamp basis coefficients.
        method : str
            ``"scan"`` for lower memory or ``"scatter"`` for parallel placement.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        if not isinstance(spec, PasteSpec):
            raise TypeError("spec must be a PasteSpec.")

        basis = dlu.to_value(basis)
        if basis.ndim < 4:
            raise ValueError("basis must have shape (n_stamps, ..., ny, nx).")
        if basis.shape[0] != len(spec.starts):
            raise ValueError("basis and PasteSpec must contain the same stamp count.")
        if basis.shape[-2:] != spec.shape[::-1]:
            raise ValueError("basis spatial shape must match the PasteSpec stamps.")

        method = str(method).lower()
        if method not in ("scan", "scatter"):
            raise ValueError("method must be either 'scan' or 'scatter'.")

        shape = basis.shape[:-2]
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self._set_coeffs(coeffs, shape)
        self.basis = basis
        self.spec = spec
        self.method = method

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate compact bases and paste their local values globally.

        Coefficients are contracted independently per stamp. The result has the full
        output spatial shape from `PasteSpec`; overlapping stamp values are added.
        """
        local = jax.vmap(dlu.eval_basis)(self.basis, self.coeffs)
        return self.spec.paste(local, self.method)

    def solve_basis(self, value: Array, **kwargs: Any) -> Array:
        """Solve coefficients independently from every extracted local stamp.

        ``value`` must end with the global output shape. Returned coefficients have
        shape ``(n_stamps, *basis_shape)``.
        """
        values = self.spec.extract(value)
        solve = lambda value, basis: dlu.solve_basis(value, basis)
        return jax.vmap(solve)(values, self.basis)


class ImplicitBasis(ParametricBasis):
    """Base class for bases generated or evaluated indirectly at runtime."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @abstractmethod
    def calculate_basis(self: ImplicitBasis, **kwargs: Any) -> Array:
        """Materialise basis vectors from the supplied context.

        The result must begin with the native coefficient ``shape`` and be followed
        by the sampled output axes consumed by `evaluate_basis`.
        """

    def evaluate(self: ImplicitBasis, **kwargs: Any) -> Array:
        """Generate the implicit basis and contract its coefficient dimensions.

        Named context is passed to `calculate_basis`. The result retains all sampled
        output axes after the native basis dimensions are contracted.
        """
        return self.evaluate_basis(self.calculate_basis(**kwargs))

    def solve_basis(self: ImplicitBasis, value: Array, **kwargs: Any) -> Array:
        """Generate the implicit basis and solve it for ``value``.

        The returned least-squares coefficients have the native basis ``shape``.
        Named context is passed to `calculate_basis`.
        """
        return dlu.solve_basis(value, self.calculate_basis(**kwargs))


class CoordBasis(ImplicitBasis):
    """Base class for implicit bases evaluated at Cartesian coordinates."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @staticmethod
    def get_coordinates(*, wavefront: Any = None, coordinates: Array = None) -> Array:
        """Resolve explicit coordinates or coordinates from a wavefront.

        Provide exactly one usable source. Returned coordinates have component axis
        length two immediately before the final ``(y, x)`` spatial axes and are in
        canonical SI units when obtained from a wavefront.
        """
        if coordinates is not None:
            return BaseCoordTransform.get_coordinates(coordinates)
        if wavefront is None:
            raise ValueError("Provide either wavefront or coordinates.")
        return BaseCoordTransform.get_coordinates(wavefront.coordinates)


class CLIMBBasis(Basis):
    """A continuous latent basis mapped through the CLIMB binarisation."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    values: Array
    oversample: int = eqx.field(static=True)

    def __init__(
        self,
        basis,
        coeffs=None,
        shape=None,
        values=(0.0, 1.0),
        oversample=3,
        *,
        coefficients=None,
    ):
        """Initialise a CLIMB-softened explicit basis.

        Parameters
        ----------
        basis : Array
            Explicit sampled basis values.
        coeffs : Array or None
            Basis coefficients, defaulting to zeros.
        shape : tuple[int, ...] or None
            Coefficient-axis shape, inferred when omitted.
        values : tuple[float, float]
            Lower and upper values used by soft binarisation.
        oversample : int
            Internal soft-binarisation oversampling factor.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        super().__init__(basis, coeffs, shape)
        output_shape = self.basis.shape[len(self.shape) :]
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError("The CLIMB latent output must be a square 2D array.")
        values = dlu.to_value(values)
        if values.shape != (2,):
            raise ValueError("values must contain exactly two output values.")
        self.values = values
        self.oversample = int(oversample)
        if self.oversample < 1:
            raise ValueError("oversample must be a positive integer.")
        if output_shape[0] % self.oversample != 0:
            raise ValueError(
                "The CLIMB latent output size must be divisible by oversample."
            )

    def evaluate_latent(self) -> Array:
        """Return the continuous explicit-basis field before CLIMB binarisation.

        The result retains the oversampled latent spatial shape.
        """
        return super().evaluate()

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate and softly binarise the latent field.

        The result is downsampled by ``oversample`` and mapped between the configured
        lower and upper values.
        """
        binary = dlu.soft_binarise(self.evaluate_latent(), self.oversample)
        low, high = self.values
        return low + (high - low) * binary


class FourierBasis(ImplicitBasis):
    """A parameterisation over a separable real Fourier basis."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    kernels: tuple[Array, Array]

    def __init__(
        self, npix, n_modes, coeffs=None, scale: float = 1.0, *, coefficients=None
    ):
        """Initialise an implicit separable Fourier basis.

        Parameters
        ----------
        npix : int or tuple[int, int]
            Sampled output size.
        n_modes : int or tuple[int, int]
            Fourier-mode counts in physical-axis order.
        coeffs : Array or None
            Coefficient array matching the generated separable basis shape.
        scale : float
            Coordinate scale used to construct the Fourier kernels.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        self.kernels = dlu.fourier_kernels(n_modes, npix, scale)
        shape = tuple(kernel.shape[1] for kernel in self.kernels)
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self._set_coeffs(coeffs, shape)

    def calculate_basis(self, **kwargs: Any) -> Array:
        """Materialise the complete separable real Fourier basis.

        The result has shape ``(*shape, ny, nx)``. Additional context is accepted but
        not consumed because the sampling kernels are fixed at construction.
        """
        Kx, Ky = self.kernels
        return np.einsum("xi,yj->ijxy", Kx, Ky)

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the separable Fourier expansion without materialising its basis.

        Returns a sampled ``(ny, nx)`` field, preserving any supported leading
        coefficient axis.
        """
        return dlu.eval_fourier_basis(self.coeffs, *self.kernels)

    def resize(self, npix, scale: float = 1.0):
        """Return a copy sampled onto a resized Fourier grid.

        ``npix`` gives physical ``(x, y)`` sample counts and ``scale`` defines the
        dimensionless Fourier coordinate extent. Coefficients are retained while the
        separable kernels are regenerated.
        """
        kernels = dlu.fourier_kernels(self.shape, npix, scale)
        return self.set(kernels=kernels)


class SplineBasis(ImplicitBasis):
    """A fixed 2D array represented by a lower-resolution grid of spline knots."""

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    knot_coords: Array
    sample_coords: Array
    method: str = eqx.field(static=True)

    def __init__(
        self, npix, n_knots, coeffs=None, method="cubic", *, coefficients=None
    ):
        """Initialise an implicit two-dimensional spline basis.

        Parameters
        ----------
        npix : int or tuple[int, int]
            Sampled output size.
        n_knots : int or tuple[int, int]
            At least two spline knots along each physical axis.
        coeffs : Array or None
            Knot coefficients, defaulting to zeros.
        method : str
            Interpolation method used between knots.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        npix = dlu.as_size(npix, 2, "npix")
        n_knots = dlu.as_size(n_knots, 2, "n_knots")
        if any(n < 2 for n in n_knots):
            raise ValueError("n_knots must contain values greater than one.")
        knot_axes = [np.linspace(-1.0, 1.0, n) for n in n_knots]
        sample_axes = [np.linspace(-1.0, 1.0, n) for n in npix]
        self.knot_coords = np.array(np.meshgrid(*knot_axes, indexing="xy"))
        self.sample_coords = np.array(np.meshgrid(*sample_axes, indexing="xy"))
        shape = self.knot_coords.shape[1:]
        coeffs = np.zeros(shape) if coeffs is None else coeffs
        self._set_coeffs(coeffs, shape)
        self.method = str(method)

    def calculate_basis(self, **kwargs: Any) -> Array:
        """Materialise one interpolated basis vector per spline knot.

        The result has shape ``(*shape, ny, nx)`` on the configured sampled output
        coordinates.
        """
        # Generate impulses at every spline knot
        size = self.coeffs.size
        impulses = np.eye(size).reshape((size,) + self.shape)

        # Interpolate every impulse onto the sampled output grid
        interpolate = lambda values: dlu.interp(
            values, self.knot_coords, self.sample_coords, method=self.method
        )
        basis = jax.vmap(interpolate)(impulses)

        # Restore the native coefficient and sampled output dimensions
        return basis.reshape(self.shape + self.sample_coords.shape[1:])

    def evaluate(self, **kwargs: Any) -> Array:
        """Interpolate spline-knot coefficients onto the sampled output grid.

        Returns a real ``(ny, nx)`` field using the configured interpolation method.
        """
        return dlu.interp(
            self.coeffs, self.knot_coords, self.sample_coords, method=self.method
        )
