"""Core parameterisation and non-polynomial basis implementations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array

import dLux.utils as dlu
from .parametrics import BaseParametric

__all__ = [
    "ParametricBasis",
    "ExplicitBasis",
    "ImplicitBasis",
    "CoordBasis",
    "CLIMBBasis",
    "FourierBasis",
    "SplineBasis",
]


class ParametricBasis(BaseParametric):
    """Base class for coefficient-weighted basis parameterisations.

    ??? abstract "UML"
        ![UML](../assets/uml/ParametricBasis.png)
    """

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)

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
        self: ParametricBasis,
        coefficients: Array,
        coefficient_shape: tuple[int, ...],
    ) -> None:
        coefficients = np.asarray(coefficients, dtype=float)
        coefficient_shape = tuple(coefficient_shape)
        if coefficients.shape[-len(coefficient_shape) :] != coefficient_shape:
            raise ValueError(
                "Coefficient shape trailing dimensions must match the basis "
                "dimensions. "
                f"Expected {coefficient_shape}, got {coefficients.shape}."
            )
        self.coefficients = coefficients
        self.basis_shape = coefficient_shape

    def evaluate_basis(self, basis: Array) -> Array:
        """Apply global or leading-axis-vectorised coefficients to a basis."""
        if self.coefficients.ndim == len(self.basis_shape):
            return dlu.eval_basis(basis, self.coefficients)
        if self.coefficients.ndim != len(self.basis_shape) + 1:
            raise ValueError("Only one leading coefficient axis is supported.")
        axis = len(self.basis_shape)
        if basis.shape[axis] != self.coefficients.shape[0]:
            raise ValueError(
                "The leading coefficient axis must match the leading basis axis."
            )
        basis = np.moveaxis(basis, axis, 0)
        return jax.vmap(dlu.eval_basis)(basis, self.coefficients)

    @abstractmethod
    def solve_basis(
        self: ParametricBasis, value: Array, **kwargs: Any
    ) -> Array:  # pragma: no cover
        pass


class ExplicitBasis(ParametricBasis):
    """A parameterisation over an explicitly sampled basis array.

    ??? abstract "UML"
        ![UML](../assets/uml/ExplicitBasis.png)
    """

    basis: Array

    def __init__(
        self: ExplicitBasis,
        basis: Array,
        coefficients: Array = None,
        coefficient_shape: tuple[int, ...] = None,
    ):
        self.basis = np.asarray(basis, dtype=float)
        if coefficients is None:
            if coefficient_shape is None:
                raise ValueError("Provide either coefficients or coefficient_shape.")
            coefficients = np.zeros(coefficient_shape)
        else:
            coefficients = np.asarray(coefficients, dtype=float)
            if coefficient_shape is None:
                coefficient_shape = coefficients.shape
        if self.basis.shape[: len(coefficient_shape)] != coefficient_shape:
            raise ValueError(
                "The leading basis dimensions must match the coefficient shape."
            )
        self._set_coefficients(coefficients, coefficient_shape)

    def evaluate(self: ExplicitBasis, **kwargs: Any) -> Array:
        return self.evaluate_basis(self.basis)

    def solve_basis(self: ExplicitBasis, value: Array, **kwargs: Any) -> Array:
        return dlu.solve_basis(value, self.basis)


class ImplicitBasis(ParametricBasis):
    """Base class for bases generated or evaluated indirectly at runtime.

    ??? abstract "UML"
        ![UML](../assets/uml/ImplicitBasis.png)
    """

    @abstractmethod
    def calculate_basis(
        self: ImplicitBasis, **kwargs: Any
    ) -> Array:  # pragma: no cover
        pass

    def evaluate(self: ImplicitBasis, **kwargs: Any) -> Array:
        return self.evaluate_basis(self.calculate_basis(**kwargs))

    def solve_basis(self: ImplicitBasis, value: Array, **kwargs: Any) -> Array:
        return dlu.solve_basis(value, self.calculate_basis(**kwargs))


class CoordBasis(ImplicitBasis):
    """Base class for implicit bases evaluated at Cartesian coordinates.

    ??? abstract "UML"
        ![UML](../assets/uml/CoordBasis.png)
    """

    @staticmethod
    def get_coordinates(*, wavefront: Any = None, coordinates: Array = None) -> Array:
        if coordinates is not None:
            return coordinates
        if wavefront is None:
            raise ValueError("Provide either wavefront or coordinates.")
        return wavefront.coordinates


class CLIMBBasis(ExplicitBasis):
    """A continuous latent basis mapped through the CLIMB binarisation.

    ??? abstract "UML"
        ![UML](../assets/uml/CLIMBBasis.png)
    """

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
        values = np.asarray(values, dtype=float)
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
    """A parameterisation over a separable real Fourier basis.

    ??? abstract "UML"
        ![UML](../assets/uml/FourierBasis.png)
    """

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
    """A fixed 2D array represented by a lower-resolution grid of spline knots.

    ??? abstract "UML"
        ![UML](../assets/uml/SplineBasis.png)
    """

    knot_coords: Array
    sample_coords: Array
    method: str = eqx.field(static=True)

    def __init__(self, npix, n_knots, coefficients=None, method="cubic"):
        npix = (npix, npix) if isinstance(npix, int) else tuple(npix)
        n_knots = (n_knots, n_knots) if isinstance(n_knots, int) else tuple(n_knots)
        if len(npix) != 2 or len(n_knots) != 2:
            raise ValueError("npix and n_knots must be integers or length-two tuples.")
        if any(n < 1 for n in npix):
            raise ValueError("npix must contain positive integers.")
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
            self.coefficients,
            self.knot_coords,
            self.sample_coords,
            method=self.method,
        )
