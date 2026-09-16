"""Polynomial and Zernike parameterisations with explicit coefficient ownership."""

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..grids import GridSpec
from .bases import Basis, SpatialBasis, ParametricBasis, _coefficients
from .parametrics import SpatialExpression, _spatial_context

__all__ = [
    "Polynomial",
    "DynamicZernike",
    "ZernikeBasis",
    "DynamicZernikeBasis",
]


def _powers(degree=None, ndim=1, powers=None, degrees=None):
    """Resolve one static polynomial term definition."""
    if sum(value is not None for value in (degree, powers, degrees)) != 1:
        raise ValueError("Provide exactly one of degree, degrees, or powers.")
    if powers is not None:
        powers = np.asarray(powers)
        powers = powers[None, :] if powers.ndim == 1 else powers
        if powers.ndim != 2 or 0 in powers.shape:
            raise ValueError("powers must have shape (n_variables, n_terms).")
        if not np.issubdtype(powers.dtype, np.integer):
            raise TypeError("powers must have an integer dtype.")
        if np.any(powers < 0):
            raise ValueError("powers must be non-negative.")
        return powers
    if degree is not None:
        return dlu.polynomial_powers(degree, ndim)

    degrees = np.atleast_1d(np.asarray(degrees))
    if degrees.ndim != 1 or not degrees.size:
        raise ValueError("degrees must contain at least one total degree.")
    if not np.issubdtype(degrees.dtype, np.integer):
        raise TypeError("degrees must have an integer dtype.")
    if np.any(degrees < 0):
        raise ValueError("degrees must be non-negative.")
    powers = dlu.polynomial_powers(int(degrees.max()), ndim)
    return powers[:, np.isin(powers.sum(0), degrees)]


def _zernike_indices(js=None, order=None, orders=None) -> list[int]:
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


class Polynomial(ParametricBasis):
    """Expand coefficients over monomials at contextual coordinates.

    A D-variable coordinate field has shape ``(..., D, *spatial_shape)`` with D
    sampled axes. Leading coefficient and coordinate batches broadcast together.
    One-variable polynomials also accept a bare scalar or array of samples. The
    coefficient order is increasing total degree, with the first variable's power
    descending within each degree. Coordinates and coefficients use caller-defined
    units; no normalisation or centring is implicit.

    Local ``solve`` returns minimum-norm least-squares coefficients at the supplied
    samples. ``solve`` returns physical coefficients without reversing their
    transforms.
    Integer monomials retain finite derivatives at zero, including higher
    derivatives. Coefficients may be contextual Zodiax expressions.

    Examples
    --------
    ```python
    import dLux as dl

    polynomial = dl.Polynomial(degree=2, coeffs=[1., 2., 3.])
    values = polynomial.resolve(coordinates=[0., 1., 2.])
    ```
    """

    coeffs: Array | zdx.Expression | None
    powers: Array

    def __init__(
        self,
        degree=None,
        coeffs=None,
        *,
        ndim=1,
        degrees=None,
        powers=None,
        alias=None,
    ):
        """Initialise coefficients and fixed polynomial term topology.

        Parameters
        ----------
        degree : int or None
            Maximum non-negative total degree, including the constant term.
        coeffs : Array, zodiax.Expression, or None
            Coefficients ending in the selected term count, defaulting to zeros.
            Leading axes broadcast with leading coordinate batches.
        ndim : int
            Positive variable count, default one; explicit powers determine it
            directly when supplied.
        degrees : int or sequence[int], optional
            Selected non-negative total degrees, ordered canonically.
        powers : integer Array, optional
            Explicit non-negative exponents of shape ``(ndim, n_terms)``; a vector
            is accepted for one variable. Supply exactly one term definition.
        alias : mapping or None
            Optional parameter aliases.
        """
        self.powers = _powers(degree, ndim, powers, degrees)
        self.coeffs = _coefficients(coeffs, (self.powers.shape[1],))
        self.alias = alias

    def calculate_basis(self, *, grid=None, coordinates=None, **context):
        """Return sampled terms from a spatial grid or general variable coordinates."""
        if grid is not None:
            local = _spatial_context(grid, coordinates=coordinates, **context)
            coordinates = local["coordinates"]
        if coordinates is None:
            raise zdx.Unresolved("Polynomial requires grid or coordinates.")
        return dlu.polynomial_basis(
            zdx.resolve(coordinates, **context), self.powers, axis=None
        )

    def evaluate(self, *, grid=None, coordinates=None, **context) -> Array:
        """Expand stored coefficients on a grid or general variable coordinates.

        Grid and coordinates are mutually exclusive. A grid supplies two spatial
        variables; explicit coordinates can describe arbitrary polynomial variables.
        Missing both inputs defers during resolution.
        """
        if grid is not None:
            local = _spatial_context(grid, coordinates=coordinates, **context)
            coordinates = local["coordinates"]
        if coordinates is None:
            raise zdx.Unresolved("Polynomial requires grid or coordinates.")
        coordinates = zdx.resolve(coordinates, **context)
        coeffs = zdx.resolve(self.coeffs, coordinates=coordinates, **context)
        return dlu.eval_polynomial(coordinates, coeffs, self.powers)

    def solve(self, y: Array, *, grid=None, coordinates=None, **context) -> Array:
        """Recover coefficients on a grid or general variable coordinates."""
        if grid is not None:
            local = _spatial_context(grid, coordinates=coordinates, **context)
            coordinates = local["coordinates"]
        if coordinates is None:
            raise zdx.Unresolved("Polynomial requires grid or coordinates.")
        coordinates = zdx.resolve(coordinates, **context)
        return dlu.solve_polynomial(y, coordinates, self.powers)


class DynamicZernike(SpatialExpression):
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

    def __init__(self, j: int, *, alias=None):
        """Initialise one dynamically evaluated Zernike mode.

        Parameters
        ----------
        j : int
            Positive Noll index.
        alias : mapping or None
            Optional user-defined parameter aliases.
        """
        self.j = int(j)
        if self.j < 1:
            raise ValueError("The Zernike index must be greater than 0.")
        self.name = dlu.zernike_name(self.j)
        self.n, self.m = dlu.noll_indices(self.j)
        self._c, self._k = dlu.zernike_factors(self.j)

        self.alias = alias

    def evaluate(
        self,
        *,
        grid=None,
        coordinates=None,
        transform=None,
        nsides=0,
        diameter=2.0,
        **context,
    ):
        """Evaluate one Zernike or polygonal mode on explicit spatial sampling.

        Parameters
        ----------
        grid : GridSpec or None
            Sampling grid in SI units, mutually exclusive with coordinates.
        coordinates : Array or None
            One Cartesian grid with shape (2, ny, nx). Use explicit vmap for batches.
        transform : BaseCoordTransform or None
            Optional coordinate map; source pixel scale is retained.
        nsides : int
            Polygon side count, at least three; zero selects a circular Zernike.
        diameter : float or Array
            Positive pupil diameter in coordinate units, default 2.0.
        **context
            Context used for grid and transform preparation.

        Returns
        -------
        Array
            Dimensionless mode with shape (ny, nx). Missing both spatial inputs raises
            Unresolved; supplying both raises ValueError. Origin coordinate derivatives
            can be singular in the underlying radial kernels.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        if nsides == 0:
            return dlu.zernike_fast(
                self.n, self.m, self._c, self._k, coordinates, diameter
            )
        return dlu.polike_fast(
            nsides, self.n, self.m, self._c, self._k, coordinates, diameter
        )


class ZernikeBasis(Basis):
    """Sample and parameterise Zernike modes on fixed Cartesian coordinates.

    Modes can be selected by individual Noll indices, a maximum radial order, or
    explicit radial orders. Vectors are evaluated once and retained explicitly; use
    `DynamicZernikeBasis` when coordinates change at runtime. This is a
    ``Basis``: coefficients are stored in ``coeffs`` and sampled modes in ``basis``.
    Leading coefficient axes are independent batches. Use explicit ``jax.vmap``
    for paired coordinates and coefficients. Local ``solve`` returns minimum-norm
    least-squares coefficients;
    ``solve`` returns physical coefficients without reversing their transforms.
    """

    coeffs: Array | zdx.Expression | None
    basis: Array | zdx.Expression
    axes: int = eqx.field(static=True)

    def __init__(
        self,
        coordinates,
        js=None,
        order=None,
        orders=None,
        coeffs=None,
        diameter=2.0,
        *,
        alias=None,
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
        coeffs : Array, zodiax.Expression, or None
            Mode coefficients, defaulting to zeros.
        diameter : float or Array
            Zernike pupil diameter in the coordinate unit.
        alias : mapping or None
            Optional parameter aliases.
        """
        js = _zernike_indices(js, order, orders)
        if isinstance(coordinates, GridSpec):
            coordinates = coordinates.broadcast(2).coordinates
        basis = dlu.zernike_basis(js, coordinates, diameter)
        super().__init__(basis, _coefficients(coeffs, (len(js),)), alias=alias)


class DynamicZernikeBasis(SpatialBasis):
    """Evaluate a Zernike expansion dynamically from coordinate context.

    Modes are selected by Noll index or radial order and evaluated on a grid or
    explicit coordinates. The pupil diameter comes from explicit context, the
    stored value, or the default 2.0 in coordinate units; it is never inferred from
    a wavefront or the grid extent.
    """

    coeffs: Array | zdx.Expression | None
    shape: tuple[int, ...] = eqx.field(static=True)
    zernikes: list[DynamicZernike]
    nsides: int = eqx.field(static=True)
    diameter: Array | None = None

    def __init__(
        self,
        js=None,
        order=None,
        orders=None,
        coeffs=None,
        nsides=0,
        diameter=None,
        *,
        alias=None,
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
        coeffs : Array, zodiax.Expression, or None
            Mode coefficients, defaulting to zeros.
        nsides : int
            Polygonal support side count; zero uses the circular definition.
        diameter : float, Array, or None
            Stored pupil diameter in coordinate units. An evaluation-context
            diameter overrides it; if both are None, use 2.0.
        alias : mapping or None
            Optional parameter aliases.
        """
        js = _zernike_indices(js, order, orders)
        self.zernikes = [DynamicZernike(j) for j in js]
        coeffs = np.zeros(len(js)) if coeffs is None else coeffs
        super().__init__(coeffs, (len(js),), alias=alias)
        self.nsides = int(nsides)
        if self.nsides not in (0,) and self.nsides < 3:
            raise ValueError("nsides must be zero or greater than two.")
        self.diameter = zdx.as_array(diameter, dtype=float)
        if (
            self.diameter is not None
            and not isinstance(self.diameter, zdx.Expression)
            and self.diameter <= 0
        ):
            raise ValueError("diameter must be greater than zero.")

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Expand generated Zernike modes using the stored coefficients.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.

            A diameter in context overrides the stored diameter; if both are None,
            use 2.0 in coordinate units. No wavefront or grid-extent inference occurs.

        Returns
        -------
        Array
            Array with final shape (ny, nx), preceded by independent coefficient
            batches. Units follow the coefficients. Modes are not flux-normalised.
        """
        local = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        basis = self.calculate_basis(**local)
        coeffs = zdx.resolve(self.coeffs, **local)
        return dlu.eval_basis(basis, coeffs, len(self.shape))

    def calculate_basis(
        self, *, grid=None, coordinates=None, transform=None, diameter=None, **context
    ):
        """Evaluate every configured Zernike mode in coordinate context.

        Parameters
        ----------
        grid : GridSpec or None
            Sampling grid, used when explicit coordinates are absent.
        coordinates : Array or None
            Explicit Cartesian coordinates, mutually exclusive with grid.
        diameter : float, Array, or None
            Explicit diameter overriding the stored value; both None gives 2.0.
        transform : BaseCoordTransform or None
            Map sampling coordinates before generating modes.
        **context
            Additional context accepted for composable parametric evaluation.
        """
        # Use one explicit spatial input and one explicit or stored pupil scale.
        local = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = local["coordinates"]
        diameter = self.diameter if diameter is None else diameter
        diameter = 2.0 if diameter is None else zdx.resolve(diameter, **local)

        # Evaluate every dynamic Zernike mode
        is_zernike = lambda leaf: isinstance(leaf, DynamicZernike)
        calculate = lambda zernike: zernike.evaluate(
            coordinates=coordinates, nsides=self.nsides, diameter=diameter
        )
        return np.array(jtu.map(calculate, self.zernikes, is_leaf=is_zernike))
