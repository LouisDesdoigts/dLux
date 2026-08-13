"""Polynomial and Zernike basis parameterisations."""

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import GridSpec
from .bases import Basis, CoordBasis, ParametricBasis, _resolve_coeffs
from .parametrics import resolve

__all__ = [
    "DynamicZernike",
    "ZernikeBasis",
    "DynamicZernikeBasis",
    "Polynomial",
    "ExplicitPolynomial",
    "CoordinatePolynomial",
]


def _poly_params(degree, coeffs, ndim, powers, degrees=None):
    """Validate polynomial powers and coefficients."""
    # Validate the polynomial term selection
    if degree is not None and degrees is not None:
        raise ValueError("Provide only one of degree or degrees.")

    # Generate powers from a maximum or selected total degrees
    if powers is None:
        if degree is None and degrees is None:
            raise ValueError("Provide either degree, degrees, or powers.")
        if degrees is None:
            powers = dlu.polynomial_powers(degree, ndim)
        else:
            degrees = np.atleast_1d(dlu.to_value(degrees, int))
            if degrees.ndim != 1 or degrees.size == 0:
                raise ValueError("degrees must contain at least one degree.")
            if np.any(degrees < 0):
                raise ValueError("degrees must be non-negative.")
            powers = dlu.polynomial_powers(int(degrees.max()), ndim)
            powers = powers[:, np.isin(powers.sum(0), degrees)]
    elif degrees is not None:
        raise ValueError("degrees and powers are mutually exclusive.")

    # Standardize and validate the polynomial powers
    powers = dlu.to_value(powers, int)
    powers = powers[None, :] if powers.ndim == 1 else powers
    if powers.ndim != 2:
        raise ValueError("powers must have shape (n_variables, n_terms).")
    if np.any(powers < 0):
        raise ValueError("powers must be non-negative.")

    # Initialise and validate the polynomial coefficients
    coeffs = np.zeros(powers.shape[1]) if coeffs is None else coeffs
    coeffs = dlu.to_value(coeffs)
    if coeffs.ndim < 1 or coeffs.shape[-1] != powers.shape[1]:
        raise ValueError("coeffs must have trailing shape (n_terms,).")
    return powers, coeffs


class DynamicZernike(Base):
    """Store compact radial data required to evaluate one Zernike mode.

    The object is constructed from a Noll index and evaluates that mode on runtime
    coordinates. It is primarily a leaf of `DynamicZernikeBasis`, keeping modes
    JIT-compatible without retaining sampled basis arrays.
    """

    j: int = eqx.field(static=True)
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    _c: Array
    _k: Array

    def __init__(self, j: int):
        """Initialise one dynamically evaluated Zernike mode.

        Parameters
        ----------
        j : int
            Positive Noll index.
        """
        self.j = int(j)
        if self.j < 1:
            raise ValueError("The Zernike index must be greater than 0.")
        self.name = dlu.zernike_name(self.j)
        self.n, self.m = dlu.noll_indices(self.j)
        self._c, self._k = dlu.zernike_factors(self.j)

    def calculate(
        self, coordinates: Array, nsides: int = 0, diameter: float = 2.0
    ) -> Array:
        """Evaluate the mode on circular or regular-polygon coordinates.

        Parameters
        ----------
        coordinates : Array
            Cartesian coordinates with leading physical-axis dimension two.
        nsides : int
            Polygon side count; zero selects the circular Zernike definition.
        diameter : float
            Pupil diameter in the coordinate unit.
        """
        if nsides == 0:
            return dlu.zernike_fast(
                self.n, self.m, self._c, self._k, coordinates, diameter
            )
        return dlu.polike_fast(
            nsides, self.n, self.m, self._c, self._k, coordinates, diameter
        )


class _ZernikeBasis:
    @staticmethod
    def get_indices(js=None, order=None, orders=None) -> list[int]:
        """Resolve individual modes or complete radial orders."""
        if sum(value is not None for value in (js, order, orders)) != 1:
            raise ValueError("Provide exactly one of js, order, or orders.")

        if js is not None:
            indices = [int(j) for j in js]
        else:
            orders = range(int(order) + 1) if order is not None else orders
            indices = dlu.radial_orders_to_indices(orders)
        if not indices:
            raise ValueError("At least one Zernike mode must be selected.")
        if any(j < 1 for j in indices):
            raise ValueError("Zernike indices must be greater than zero.")
        return indices


class ZernikeBasis(_ZernikeBasis, Basis):
    """Sample and parameterise Zernike modes on fixed Cartesian coordinates.

    Modes can be selected by individual Noll indices, a maximum radial order, or
    explicit radial orders. Vectors are evaluated once and retained explicitly; use
    `DynamicZernikeBasis` when coordinates change at runtime.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array

    def __init__(
        self,
        coordinates,
        js=None,
        order=None,
        orders=None,
        coeffs=None,
        diameter=2.0,
        *,
        coefficients=None,
    ):
        """Initialise an explicitly sampled Zernike expansion.

        Parameters
        ----------
        coordinates : Array or GridSpec
            Cartesian sampling used to evaluate the modes.
        js : ArrayLike or None
            Noll indices, mutually exclusive with ``order`` and ``orders``.
        order : int or None
            Maximum radial order, including every order from zero through this value.
        orders : ArrayLike or None
            Selected radial orders expanded to complete Noll sequences.
        coeffs : Array or None
            Mode coefficients, defaulting to zeros.
        diameter : float or Array
            Zernike pupil diameter in the coordinate unit.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        js = self.get_indices(js, order, orders)
        basis = dlu.zernike_basis(js, coordinates, diameter)
        super().__init__(basis, coeffs, (len(js),))


class DynamicZernikeBasis(_ZernikeBasis, CoordBasis):
    """Evaluate a Zernike expansion dynamically from coordinate context.

    Modes are selected by Noll index or radial order and evaluated on explicit or
    wavefront coordinates. `diameter` defines the physical pupil scale and may be
    inferred from wavefront context when omitted.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    zernikes: list[DynamicZernike]
    nsides: int = eqx.field(static=True)
    diameter: Array | None

    def __init__(
        self,
        js=None,
        order=None,
        orders=None,
        coeffs=None,
        nsides=0,
        diameter=None,
        *,
        coefficients=None,
    ):
        """Initialise a coordinate-dependent Zernike expansion.

        Parameters
        ----------
        js : ArrayLike or None
            Noll indices, mutually exclusive with ``order`` and ``orders``.
        order : int or None
            Maximum radial order, including every order from zero through this value.
        orders : ArrayLike or None
            Selected radial orders expanded to complete Noll sequences.
        coeffs : Array or None
            Mode coefficients, defaulting to zeros.
        nsides : int
            Polygonal support side count; zero uses the circular definition.
        diameter : float, Array, or None
            Pupil diameter supplied directly or through evaluation context.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        js = self.get_indices(js, order, orders)
        self.zernikes = [DynamicZernike(j) for j in js]
        coeffs = np.zeros(len(js)) if coeffs is None else coeffs
        self._set_coeffs(coeffs, (len(js),))
        self.nsides = int(nsides)
        if self.nsides not in (0,) and self.nsides < 3:
            raise ValueError("nsides must be zero or greater than two.")
        self.diameter = dlu.to_value(diameter, optional=True, name="diameter")
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    def calculate_basis(
        self, *, wavefront=None, coordinates=None, diameter=None, **kwargs
    ):
        """Evaluate every configured Zernike mode in coordinate context.

        Parameters
        ----------
        wavefront : Wavefront or None
            Optional source of coordinates and pupil diameter.
        coordinates : Array or None
            Explicit Cartesian coordinates overriding the wavefront grid.
        diameter : float, Array, or None
            Explicit diameter overriding the stored or inferred value.
        **kwargs
            Additional context accepted for composable parametric evaluation.
        """
        # Resolve coordinates and the aperture diameter
        infer_diameter = coordinates is None and wavefront is not None
        coordinates = self.get_coordinates(wavefront=wavefront, coordinates=coordinates)
        if diameter is None:
            diameter = self.diameter
        if diameter is None:
            diameter = wavefront.diameter if infer_diameter else 2.0

        # Evaluate every dynamic Zernike mode
        is_zernike = lambda leaf: isinstance(leaf, DynamicZernike)
        calculate = lambda zernike: zernike.calculate(
            coordinates, self.nsides, diameter
        )
        return np.array(jtu.map(calculate, self.zernikes, is_leaf=is_zernike))


class Polynomial(ParametricBasis):
    """A general polynomial in one or more supplied variables.

    Pass ``degree`` to include every total degree from zero through that value, or
    pass ``degrees`` to select total degrees explicitly. For example,
    ``degrees=[1]`` constructs only the linear terms and omits the constant term.

    Examples
    --------
    Construct and evaluate a quadratic polynomial:

    ```python
    import jax.numpy as np

    import dLux as dl

    polynomial = dl.Polynomial(degree=2, coeffs=[1.0, 0.5, -0.2])

    # Evaluate the polynomial
    variables = np.linspace(-1.0, 1.0, 100)
    values = polynomial.evaluate(variables=variables)

    # Recover the coefficients representing the evaluated values
    coeffs = polynomial.solve_basis(values, variables=variables)
    ```
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    powers: Array

    def __init__(
        self,
        degree=None,
        coeffs=None,
        ndim=1,
        powers=None,
        degrees=None,
        *,
        coefficients=None,
    ):
        """Initialise a general polynomial parameterisation.

        Parameters
        ----------
        degree : int or None
            Maximum total degree, mutually exclusive with ``degrees`` and ``powers``.
        coeffs : Array or None
            Coefficients matching the generated or supplied terms.
        ndim : int
            Number of polynomial variables.
        powers : Array or None
            Explicit exponents with shape ``(ndim, n_terms)``.
        degrees : int, sequence[int], or None
            Selected total degrees.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        powers, coeffs = _poly_params(degree, coeffs, ndim, powers, degrees)
        self.powers = powers
        self._set_coeffs(coeffs, (powers.shape[1],))

    def calculate_basis(self, *, variables=None, **context):
        """Evaluate polynomial terms at supplied variables.

        ``variables`` has leading variable axis ``ndim`` followed by arbitrary sample
        axes. The result has leading term axis followed by those sample axes.
        """
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
        """Evaluate the polynomial at supplied or contextual variables.

        Terms are contracted against ``coeffs`` and all variable sample axes are
        retained. ``variables`` has a leading polynomial-variable axis followed by
        arbitrary sample axes.
        """
        if variables is None:
            basis = self.calculate_basis(**context)
        else:
            basis = self.calculate_basis(variables=variables, **context)
        return self.evaluate_basis(basis)

    def solve_basis(self, value, *, variables=None, **context):
        """Solve for polynomial coefficients representing ``value``.

        ``variables`` has a leading polynomial-variable axis followed by arbitrary
        sample axes. ``value`` must match those sample axes. Returned least-squares
        coefficients have the polynomial term shape.
        """
        if variables is None:
            basis = self.calculate_basis(**context)
        else:
            basis = self.calculate_basis(variables=variables, **context)
        return dlu.solve_basis(value, basis)


class ExplicitPolynomial(Basis):
    """Represent a multivariate polynomial on fixed sampled coordinates.

    Polynomial terms are generated once and stored as an explicit basis. Coefficients
    remain differentiable while coordinates stay fixed; use `CoordinatePolynomial`
    when the coordinate field changes during evaluation.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    basis: Array
    powers: Array

    def __init__(
        self,
        coordinates: Array | GridSpec,
        degree=None,
        coeffs=None,
        ndim=None,
        powers=None,
        degrees=None,
        *,
        coefficients=None,
    ):
        """Initialise a polynomial on fixed coordinates.

        Parameters
        ----------
        coordinates : Array or GridSpec
            Fixed variables used for every evaluation.
        degree : int or None
            Maximum total degree.
        coeffs : Array or None
            Polynomial coefficients.
        ndim : int or None
            Variable count, inferred from coordinates when omitted.
        powers : Array or None
            Explicit term exponents.
        degrees : int, sequence[int], or None
            Selected total degrees.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        coordinates, ndim = self._coordinates(coordinates, ndim)
        powers, coeffs = _poly_params(degree, coeffs, ndim, powers, degrees)
        if coordinates.shape[0] != powers.shape[0]:
            raise ValueError(
                "coordinate dimensionality must match the polynomial powers."
            )
        self.powers = powers
        basis = dlu.polynomial_basis(coordinates, powers)
        super().__init__(basis, coeffs)

    @staticmethod
    def _coordinates(coordinates, ndim):
        """Resolve explicit polynomial coordinates and dimensionality."""
        # Resolve coordinates from a grid specification
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

        # Resolve explicit coordinate arrays
        else:
            coordinates = dlu.to_value(coordinates)
            ndim = 1 if ndim is None and coordinates.ndim == 1 else ndim
            ndim = coordinates.shape[0] if ndim is None else int(ndim)

        # Validate and return the requested coordinate dimensions
        if ndim < 1:
            raise ValueError("ndim must be positive.")
        coordinates = coordinates[None, :] if coordinates.ndim == 1 else coordinates
        if coordinates.shape[0] < ndim:
            raise ValueError(
                "coordinates must contain at least ndim coordinate arrays."
            )
        return coordinates[:ndim], ndim


class CoordinatePolynomial(Polynomial):
    """Evaluate a multivariate polynomial from Cartesian coordinate context.

    The leading coordinate component axis must match `ndim`. Coordinates may be
    supplied explicitly or resolved from a wavefront, so coefficients can follow
    coordinate transforms without rebuilding a sampled basis.
    """

    coeffs: Array
    shape: tuple[int, ...] = eqx.field(static=True)
    powers: Array
    ndim: int = eqx.field(static=True)

    def __init__(
        self,
        degree=None,
        coeffs=None,
        ndim: int = 2,
        degrees=None,
        *,
        coefficients=None,
    ):
        """Initialise a polynomial resolved from coordinate context.

        Parameters
        ----------
        degree : int or None
            Maximum total degree, mutually exclusive with ``degrees``.
        coeffs : Array or None
            Polynomial coefficients.
        ndim : int
            Number of coordinate variables.
        degrees : int, sequence[int], or None
            Selected total degrees.
        coefficients : Array or None
            Deprecated alias for ``coeffs``.
        """
        coeffs = _resolve_coeffs(coeffs, coefficients)
        self.ndim = int(ndim)
        super().__init__(degree, coeffs, ndim, degrees=degrees)

    def calculate_basis(self, *, wavefront=None, coordinates=None, **kwargs):
        """Evaluate polynomial terms on explicit or wavefront coordinates.

        Provide ``coordinates`` with leading component axis ``ndim`` or a wavefront
        whose SI coordinates are used. The returned leading axis enumerates terms.
        """
        if coordinates is None:
            if wavefront is None:
                raise ValueError("Provide either wavefront or coordinates.")
            coordinates = wavefront.coordinates
        if coordinates.shape[0] != self.ndim:
            raise ValueError(
                "coordinates leading axis must match the polynomial dimensionality."
            )
        return super().calculate_basis(variables=coordinates, **kwargs)
