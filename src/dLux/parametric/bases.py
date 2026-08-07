"""Core parameterisation and non-polynomial basis implementations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array

import dLux.utils as dlu
from ..grids import CoordTransform
from .parametrics import Parametric

__all__ = [
    "ParametricBasis",
    "Basis",
    "ImplicitBasis",
    "CoordBasis",
    "CLIMBBasis",
    "FourierBasis",
    "SplineBasis",
]

# TODO: Add Gaussian


class ParametricBasis(Parametric):
    """Base class for coefficient-weighted basis parameterisations."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def coeffs(self: ParametricBasis) -> Array:
        return self.coefficients

    @property
    def c(self: ParametricBasis) -> Array:
        return self.coefficients

    @property
    def alpha(self: ParametricBasis) -> Array:
        return self.coefficients

    @property
    def coefficient_shape(self: ParametricBasis) -> tuple[int, ...]:
        return self.coefficients.shape

    def _set_coefficients(
        self: ParametricBasis, coefficients: Array, coefficient_shape: tuple[int, ...]
    ) -> None:
        coefficients = dlu.to_value(coefficients)
        coefficient_shape = tuple(coefficient_shape)
        compact = coefficient_shape == (1,) and coefficients.ndim == 1
        if (
            not compact
            and coefficients.shape[-len(coefficient_shape) :] != coefficient_shape
        ):
            raise ValueError(
                "Coefficient shape trailing dimensions must match the basis "
                "dimensions. "
                f"Expected {coefficient_shape}, got {coefficients.shape}."
            )
        self.coefficients = coefficients
        self.shape = coefficient_shape

    def evaluate_basis(self, basis: Array) -> Array:
        """Apply global or leading-axis-vectorised coefficients to a basis."""
        if self.coefficients.shape == self.shape:
            return dlu.eval_basis(basis, self.coefficients)
        if self.shape == (1,) and self.coefficients.ndim == 1:
            evaluate = lambda coefficient: dlu.eval_basis(basis, coefficient[None])
            return jax.vmap(evaluate)(self.coefficients)
        if self.coefficients.ndim != len(self.shape) + 1:
            raise ValueError("Only one leading coefficient axis is supported.")
        axis = len(self.shape)
        if basis.shape[axis] != self.coefficients.shape[0]:
            raise ValueError(
                "The leading coefficient axis must match the leading basis axis."
            )
        basis = np.moveaxis(basis, axis, 0)
        return jax.vmap(dlu.eval_basis)(basis, self.coefficients)

    @abstractmethod
    def solve_basis(self: ParametricBasis, value: Array, **kwargs: Any) -> Array:
        """Solve for coefficients representing a supplied value."""


class Basis(ParametricBasis):
    """A parameterisation over an explicitly sampled basis array."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array

    def __init__(
        self: Basis,
        basis: Array,
        coefficients: Array = None,
        coefficient_shape: tuple[int, ...] = None,
    ):
        self.basis = dlu.to_value(basis)
        if coefficients is None:
            if coefficient_shape is None:
                raise ValueError("Provide either coefficients or coefficient_shape.")
            coefficients = np.zeros(coefficient_shape)
        else:
            coefficients = dlu.to_value(coefficients)
            if coefficient_shape is None:
                coefficient_shape = coefficients.shape
        if self.basis.shape[: len(coefficient_shape)] != coefficient_shape:
            raise ValueError(
                "The leading basis dimensions must match the coefficient shape."
            )
        self._set_coefficients(coefficients, coefficient_shape)

    def evaluate(self: Basis, **kwargs: Any) -> Array:
        return self.evaluate_basis(self.basis)

    def solve_basis(self: Basis, value: Array, **kwargs: Any) -> Array:
        return dlu.solve_basis(value, self.basis)


class ImplicitBasis(ParametricBasis):
    """Base class for bases generated or evaluated indirectly at runtime."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @abstractmethod
    def calculate_basis(self: ImplicitBasis, **kwargs: Any) -> Array:
        """Calculate basis vectors from the supplied context."""

    def evaluate(self: ImplicitBasis, **kwargs: Any) -> Array:
        return self.evaluate_basis(self.calculate_basis(**kwargs))

    def solve_basis(self: ImplicitBasis, value: Array, **kwargs: Any) -> Array:
        return dlu.solve_basis(value, self.calculate_basis(**kwargs))


class CoordBasis(ImplicitBasis):
    """Base class for implicit bases evaluated at Cartesian coordinates."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    @staticmethod
    def get_coordinates(*, wavefront: Any = None, coordinates: Array = None) -> Array:
        if coordinates is not None:
            return CoordTransform.get_coordinates(coordinates)
        if wavefront is None:
            raise ValueError("Provide either wavefront or coordinates.")
        return CoordTransform.get_coordinates(wavefront.coordinates)


class CLIMBBasis(Basis):
    """A continuous latent basis mapped through the CLIMB binarisation."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    values: Array
    oversample: int = eqx.field(static=True)

    def __init__(
        self,
        basis,
        coefficients=None,
        coefficient_shape=None,
        values=(0.0, 1.0),
        oversample=3,
    ):
        super().__init__(basis, coefficients, coefficient_shape)
        output_shape = self.basis.shape[len(self.coefficient_shape) :]
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
        """Evaluate the continuous, pre-binarised latent basis."""
        return super().evaluate()

    def evaluate(self, **kwargs: Any) -> Array:
        binary = dlu.soft_binarise(self.evaluate_latent(), self.oversample)
        low, high = self.values
        return low + (high - low) * binary


class FourierBasis(ImplicitBasis):
    """A parameterisation over a separable real Fourier basis."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    kernels: tuple[Array, Array]

    def __init__(self, npix, n_modes, coefficients=None, scale: float = 1.0):
        self.kernels = dlu.fourier_kernels(n_modes, npix, scale)
        shape = tuple(kernel.shape[1] for kernel in self.kernels)
        coefficients = np.zeros(shape) if coefficients is None else coefficients
        self._set_coefficients(coefficients, shape)

    def calculate_basis(self, **kwargs: Any) -> Array:
        Kx, Ky = self.kernels
        return np.einsum("xi,yj->ijxy", Kx, Ky)

    def evaluate(self, **kwargs: Any) -> Array:
        return dlu.eval_fourier_basis(self.coefficients, *self.kernels)

    def resize(self, npix, scale: float = 1.0):
        kernels = dlu.fourier_kernels(self.coefficient_shape, npix, scale)
        return self.set(kernels=kernels)


class SplineBasis(ImplicitBasis):
    """A fixed 2D array represented by a lower-resolution grid of spline knots."""

    coefficients: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    knot_coords: Array
    sample_coords: Array
    method: str = eqx.field(static=True)

    def __init__(self, npix, n_knots, coefficients=None, method="cubic"):
        npix = dlu.as_size(npix, 2, "npix")
        n_knots = dlu.as_size(n_knots, 2, "n_knots")
        if any(n < 2 for n in n_knots):
            raise ValueError("n_knots must contain values greater than one.")
        knot_axes = [np.linspace(-1.0, 1.0, n) for n in n_knots]
        sample_axes = [np.linspace(-1.0, 1.0, n) for n in npix]
        self.knot_coords = np.array(np.meshgrid(*knot_axes, indexing="xy"))
        self.sample_coords = np.array(np.meshgrid(*sample_axes, indexing="xy"))
        shape = self.knot_coords.shape[1:]
        coefficients = np.zeros(shape) if coefficients is None else coefficients
        self._set_coefficients(coefficients, shape)
        self.method = str(method)

    def calculate_basis(self, **kwargs: Any) -> Array:
        size = self.coefficients.size
        impulses = np.eye(size).reshape((size,) + self.coefficient_shape)
        interpolate = lambda values: dlu.interp(
            values, self.knot_coords, self.sample_coords, method=self.method
        )
        basis = jax.vmap(interpolate)(impulses)
        return basis.reshape(self.coefficient_shape + self.sample_coords.shape[1:])

    def evaluate(self, **kwargs: Any) -> Array:
        return dlu.interp(
            self.coefficients, self.knot_coords, self.sample_coords, method=self.method
        )
