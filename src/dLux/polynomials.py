"""Polynomial and Zernike basis parameterisations."""

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from .parametric import CoordBasis, ExplicitBasis

__all__ = [
    "DynamicZernike",
    "ZernikeBasis",
    "DynamicZernikeBasis",
    "PolynomialBasis",
]


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


class ZernikeBasis(_ZernikeBasis, ExplicitBasis):
    def __init__(
        self, coordinates, js=None, radial_orders=None, coefficients=None, diameter=2.0
    ):
        js = self.get_indices(js, radial_orders)
        basis = dlu.zernike_basis(js, coordinates, diameter)
        super().__init__(basis, coefficients, (len(js),))


class DynamicZernikeBasis(_ZernikeBasis, CoordBasis):
    zernikes: list[DynamicZernike]
    nsides: int = eqx.field(static=True)

    def __init__(self, js=None, radial_orders=None, coefficients=None, nsides=0):
        js = self.get_indices(js, radial_orders)
        self.zernikes = [DynamicZernike(j) for j in js]
        coefficients = np.zeros(len(js)) if coefficients is None else coefficients
        self._set_coefficients(coefficients, (len(js),))
        self.nsides = int(nsides)
        if self.nsides not in (0,) and self.nsides < 3:
            raise ValueError("nsides must be zero or greater than two.")

    def calculate_basis(
        self, *, wavefront=None, coordinates=None, diameter=None, **kwargs
    ):
        infer_diameter = coordinates is None and wavefront is not None
        coordinates = self.get_coordinates(wavefront=wavefront, coordinates=coordinates)
        if diameter is None:
            diameter = wavefront.diameter if infer_diameter else 2.0
        is_zernike = lambda leaf: isinstance(leaf, DynamicZernike)
        calculate = lambda zernike: zernike.calculate(
            coordinates, self.nsides, diameter
        )
        return np.array(jtu.map(calculate, self.zernikes, is_leaf=is_zernike))


class PolynomialBasis(CoordBasis):
    powers: Array

    def __init__(self, degree: int, coefficients=None):
        degree = int(degree)
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        self.powers = dlu.gen_powers(degree + 1)
        shape = (self.powers.shape[1],)
        coefficients = np.zeros(shape) if coefficients is None else coefficients
        self._set_coefficients(coefficients, shape)

    def calculate_basis(self, *, wavefront=None, coordinates=None, **kwargs):
        coordinates = self.get_coordinates(wavefront=wavefront, coordinates=coordinates)
        return dlu.polynomial_basis(coordinates, self.powers)
