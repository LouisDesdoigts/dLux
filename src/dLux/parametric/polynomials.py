"""Polynomial and Zernike basis parameterisations."""

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from ..grids import GridSpec
from .bases import Basis, CoordBasis, ParametricBasis
from .parametrics import resolve

__all__ = [
    "DynamicZernike",
    "ZernikeBasis",
    "DynamicZernikeBasis",
    "Polynomial",
    "ExplicitPolynomial",
    "CoordinatePolynomial",
]


def _poly_params(degree, coefficients, ndim, powers):
    """Validate polynomial powers and coefficients."""
    powers = dlu.polynomial_powers(degree, ndim) if powers is None else powers
    powers = np.asarray(powers, dtype=int)
    powers = powers[None, :] if powers.ndim == 1 else powers
    if powers.ndim != 2:
        raise ValueError("powers must have shape (n_variables, n_terms).")
    if np.any(powers < 0):
        raise ValueError("powers must be non-negative.")
    coefficients = np.zeros(powers.shape[1]) if coefficients is None else coefficients
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.shape[0] != powers.shape[1]:
        raise ValueError("coefficients must have shape (n_terms,).")
    return powers, coefficients


def _poly_coordinates(coordinates, ndim):
    """Resolve explicit polynomial coordinates and dimensionality."""
    if isinstance(coordinates, GridSpec):
        ndim = coordinates.ndim if ndim is None else int(ndim)
        if coordinates.ndim == 1 and ndim > 1:
            coordinates = coordinates.broadcast(ndim)
        if coordinates.ndim < ndim:
            raise ValueError(
                "GridSpec dimensionality must be greater than or equal to ndim."
            )
        if coordinates.n is None or coordinates.d is None:
            raise ValueError("GridSpec must define n and d.")
        coordinates = coordinates.coordinates
    else:
        coordinates = np.asarray(coordinates, dtype=float)
        ndim = 1 if ndim is None and coordinates.ndim == 1 else ndim
        ndim = coordinates.shape[0] if ndim is None else int(ndim)
    if ndim < 1:
        raise ValueError("ndim must be positive.")
    coordinates = coordinates[None, :] if coordinates.ndim == 1 else coordinates
    if coordinates.shape[0] < ndim:
        raise ValueError("coordinates must contain at least ndim coordinate arrays.")
    return coordinates[:ndim], ndim


class DynamicZernike(zdx.Base):
    """A dynamically evaluable Zernike polynomial."""

    j: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    _c: Array
    _k: Array

    def __init__(self, j: int):
        self.j = int(j)
        if self.j < 1:
            raise ValueError("The Zernike index must be greater than 0.")
        self.name = dlu.zernike_name(self.j)
        self.n, self.m = dlu.noll_indices(self.j)
        self._c, self._k = dlu.zernike_factors(self.j)

    def calculate(
        self, coordinates: Array, nsides: int = 0, diameter: float = 2.0
    ) -> Array:
        if nsides == 0:
            return dlu.zernike_fast(
                self.n, self.m, self._c, self._k, coordinates, diameter
            )
        return dlu.polike_fast(
            nsides, self.n, self.m, self._c, self._k, coordinates, diameter
        )


class _ZernikeBasis:
    @staticmethod
    def get_indices(js=None, radial_orders=None) -> list[int]:
        if (js is None) == (radial_orders is None):
            raise ValueError("Provide exactly one of js or radial_orders.")
        if js is not None:
            indices = [int(j) for j in js]
        else:
            indices = dlu.radial_orders_to_indices(radial_orders)
        if not indices:
            raise ValueError("At least one Zernike mode must be selected.")
        if any(j < 1 for j in indices):
            raise ValueError("Zernike indices must be greater than zero.")
        return indices


class ZernikeBasis(_ZernikeBasis, Basis):
    """An explicitly sampled Zernike basis."""

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array

    def __init__(
        self, coordinates, js=None, radial_orders=None, coefficients=None, diameter=2.0
    ):
        js = self.get_indices(js, radial_orders)
        basis = dlu.zernike_basis(js, coordinates, diameter)
        super().__init__(basis, coefficients, (len(js),))


class DynamicZernikeBasis(_ZernikeBasis, CoordBasis):
    """A Zernike basis evaluated dynamically from coordinate context."""

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)
    zernikes: list[DynamicZernike]
    nsides: int = eqx.field(static=True)
    diameter: Array | None

    def __init__(
        self, js=None, radial_orders=None, coefficients=None, nsides=0, diameter=None
    ):
        js = self.get_indices(js, radial_orders)
        self.zernikes = [DynamicZernike(j) for j in js]
        coefficients = np.zeros(len(js)) if coefficients is None else coefficients
        self._set_coefficients(coefficients, (len(js),))
        self.nsides = int(nsides)
        if self.nsides not in (0,) and self.nsides < 3:
            raise ValueError("nsides must be zero or greater than two.")
        self.diameter = None if diameter is None else np.asarray(diameter, dtype=float)
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    def calculate_basis(
        self, *, wavefront=None, coordinates=None, diameter=None, **kwargs
    ):
        infer_diameter = coordinates is None and wavefront is not None
        coordinates = self.get_coordinates(wavefront=wavefront, coordinates=coordinates)
        if diameter is None:
            diameter = self.diameter
        if diameter is None:
            diameter = wavefront.diameter if infer_diameter else 2.0
        is_zernike = lambda leaf: isinstance(leaf, DynamicZernike)
        calculate = lambda zernike: zernike.calculate(
            coordinates, self.nsides, diameter
        )
        return np.array(jtu.map(calculate, self.zernikes, is_leaf=is_zernike))


class Polynomial(ParametricBasis):
    """A general polynomial in one or more supplied variables."""

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)
    powers: Array

    def __init__(self, degree, coefficients=None, ndim=1, powers=None):
        powers, coefficients = _poly_params(degree, coefficients, ndim, powers)
        self.powers = powers
        self._set_coefficients(coefficients, (coefficients.size,))

    def calculate_basis(self, *, variables=None, **context):
        if variables is None:
            raise ValueError("variables must be provided.")
        variables = resolve(variables, float, **context)
        if self.powers.shape[0] == 1 and variables.ndim == 1:
            variables = variables[None, :]
        if variables.shape[0] != self.powers.shape[0]:
            raise ValueError(
                "variables leading axis must match the number of polynomial variables."
            )
        return dlu.polynomial_basis(variables, self.powers)

    def evaluate(self, *, variables=None, **context):
        if variables is None:
            basis = self.calculate_basis(**context)
        else:
            basis = self.calculate_basis(variables=variables, **context)
        return self.evaluate_basis(basis)

    def solve_basis(self, value, *, variables=None, **context):
        if variables is None:
            basis = self.calculate_basis(**context)
        else:
            basis = self.calculate_basis(variables=variables, **context)
        return dlu.solve_basis(value, basis)


class ExplicitPolynomial(Basis):
    """A polynomial represented by basis vectors sampled on fixed coordinates."""

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    powers: Array

    def __init__(
        self,
        coordinates: Array | GridSpec,
        degree,
        coefficients=None,
        ndim=None,
        powers=None,
    ):
        coordinates, ndim = _poly_coordinates(coordinates, ndim)
        powers, coefficients = _poly_params(degree, coefficients, ndim, powers)
        if coordinates.shape[0] != powers.shape[0]:
            raise ValueError(
                "coordinate dimensionality must match the polynomial powers."
            )
        self.powers = powers
        basis = dlu.polynomial_basis(coordinates, powers)
        super().__init__(basis, coefficients)


class CoordinatePolynomial(Polynomial):
    """A polynomial evaluated dynamically from Cartesian coordinate context."""

    coefficients: Array
    basis_shape: tuple[int, ...] = eqx.field(static=True)
    powers: Array
    ndim: int = eqx.field(static=True)

    def __init__(self, degree: int, coefficients=None, ndim: int = 2):
        self.ndim = int(ndim)
        super().__init__(degree, coefficients, ndim)

    def calculate_basis(self, *, wavefront=None, coordinates=None, **kwargs):
        if coordinates is None:
            if wavefront is None:
                raise ValueError("Provide either wavefront or coordinates.")
            coordinates = wavefront.coordinates
        if coordinates.shape[0] != self.ndim:
            raise ValueError(
                "coordinates leading axis must match the polynomial dimensionality."
            )
        return super().calculate_basis(variables=coordinates, **kwargs)
