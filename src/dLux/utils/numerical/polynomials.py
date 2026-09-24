"""Polynomial calculations and parameterisations with explicit coefficients."""

from collections.abc import Callable
from itertools import product
from operator import index

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array, core, custom_jvp, vmap

from ..grids.specs import UnitSpec
from ..bases.basis import ParametricBasis, initialise_coeffs, solve_basis
from ..batching import broadcast_batches, flatten_batch
from ..sampled import Sampled
from .units import unit_factor

__all__ = [
    "polynomial_powers",
    "polynomial_basis",
    "eval_polynomial",
    "solve_polynomial",
    "Polynomial",
]


def polynomial_powers(degree=None, ndim=1, *, degrees=None, powers=None) -> Array:
    """Generate or validate integer exponents with shape ``(ndim, n_terms)``.

    Supply one definition: ``degree`` includes every total degree from zero to
    that maximum; ``degrees`` selects total degrees; ``powers`` supplies explicit
    exponents and determines the variable count. Explicit powers retain their
    order, including an empty term axis for an identity coordinate distortion.

    Generated terms are ordered by increasing total degree, with the first
    variable's exponent descending within each degree. Degree selections and
    ndim are static construction inputs. Selection happens in Python before
    creating arrays, so fixed definitions also work when constructed under JIT.
    """
    # Choose one source of polynomial terms: explicit powers or total degrees.
    if sum(value is not None for value in (degree, degrees, powers)) != 1:
        raise ValueError("Provide exactly one of degree, degrees, or powers.")
    if powers is not None:
        powers = np.asarray(powers)
        powers = powers[None, :] if powers.ndim == 1 else powers
        if powers.ndim != 2 or powers.shape[0] == 0:
            raise ValueError("powers must have shape (n_variables, n_terms).")
        if not np.issubdtype(powers.dtype, np.integer):
            raise TypeError("powers must have an integer dtype.")
        # Values supplied during tracing are the caller's responsibility.
        negative = np.any(powers < 0)
        if not isinstance(negative, core.Tracer) and bool(negative):
            raise ValueError("powers must be non-negative.")
        return powers

    # Validate the variable count before generating exponent combinations.
    if isinstance(ndim, bool):
        raise TypeError("ndim must be an integer.")
    ndim = index(ndim)
    if ndim < 1:
        raise ValueError("ndim must be positive.")

    # Accept a maximum degree or an explicit selection of non-negative degrees.
    if degree is not None:
        if isinstance(degree, bool):
            raise TypeError("degree must be an integer.")
        degree = index(degree)
        if degree < 0:
            raise ValueError("degree must be non-negative.")
        selected_degrees = range(degree + 1)
    else:
        if isinstance(degrees, int) or getattr(degrees, "ndim", None) == 0:
            degrees = (degrees,)
        selected_degrees = []
        for value in degrees:
            if isinstance(value, bool):
                raise TypeError("degrees must contain integers.")
            selected_degrees.append(index(value))
        if not selected_degrees or any(value < 0 for value in selected_degrees):
            raise ValueError("degrees must contain non-negative total degrees.")
        selected_degrees = sorted(set(selected_degrees))

    # Enumerate terms in degree order, keeping only exponents with that total.
    selected_powers = []
    for total in selected_degrees:
        for exponents in product(range(total, -1, -1), repeat=ndim):
            if sum(exponents) == total:
                selected_powers.append(exponents)
    return np.asarray(selected_powers, dtype=int).T


@custom_jvp
def _monomial(value, power):
    """Evaluate integer powers without singular constant-term derivatives at zero."""
    return np.power(value, power)


@_monomial.defjvp
def _monomial_jvp(primals, tangents):
    value, power = primals
    tangent, _ = tangents
    slope = power * _monomial(value, np.maximum(power - 1, 0))
    return _monomial(value, power), slope * tangent


def _polynomial_layout(coordinates, powers):
    """Return monomial samples and their sampled-axis count."""
    coordinates, powers = np.asarray(coordinates), np.asarray(powers)
    powers = powers[None, :] if powers.ndim == 1 else powers
    if powers.ndim != 2 or 0 in powers.shape:
        raise ValueError("powers must have shape (n_variables, n_terms).")
    if not np.issubdtype(powers.dtype, np.integer):
        raise TypeError("powers must have an integer dtype.")
    ndim = powers.shape[0]

    # A scalar polynomial accepts arbitrary bare samples, including one scalar.
    if ndim == 1 and (coordinates.ndim < 2 or coordinates.shape[-2] != 1):
        return polynomial_basis(coordinates[None, ...], powers), coordinates.ndim

    # Sampled fields have a physical component axis before D sampled axes.
    component = coordinates.ndim - ndim - 1
    if component < 0 or coordinates.shape[component] != ndim:
        raise ValueError(f"coordinates must have shape (..., {ndim}, *spatial_shape).")
    return polynomial_basis(coordinates, powers, axis=component), ndim


def polynomial_basis(coordinates: Array, powers: Array, axis: int | None = 0) -> Array:
    """Evaluate monomials by replacing a coordinate component axis with a term axis.

    ``powers`` is a non-negative integer array of shape ``(D, n_terms)``.
    ``axis`` identifies the component axis of size D, defaulting to the first axis;
    all other axes are preserved. Variables can be sampled over any array shape,
    independently of their count. With ``axis=None``, use dLux's sampled field
    convention ``(..., D, *spatial_shape)`` with D final spatial axes. This mode
    also accepts bare scalar/array samples for one-variable polynomials, returning
    ``(n_terms, *samples.shape)``.
    Integer powers have finite derivatives at zero, including higher derivatives.
    """
    if axis is None:
        return _polynomial_layout(coordinates, powers)[0]

    # Match integer exponent rows to the chosen coordinate-component axis.
    coordinates, powers = np.asarray(coordinates), np.asarray(powers)
    powers = powers[None, :] if powers.ndim == 1 else powers
    if powers.ndim != 2 or 0 in powers.shape:
        raise ValueError("powers must have shape (n_variables, n_terms).")
    if not np.issubdtype(powers.dtype, np.integer):
        raise TypeError("powers must have an integer dtype.")
    axis = index(axis)
    if not -coordinates.ndim <= axis < coordinates.ndim:
        raise ValueError("axis must identify a coordinate component axis.")
    axis %= coordinates.ndim
    if coordinates.shape[axis] != powers.shape[0]:
        raise ValueError(
            "The component axis must match the number of polynomial variables."
        )

    # Introduce a term axis, then multiply variable powers into each monomial.
    shape = (1,) * axis + powers.shape + (1,) * (coordinates.ndim - axis - 1)
    terms = _monomial(np.expand_dims(coordinates, axis + 1), powers.reshape(shape))
    return np.prod(terms, axis=axis)


def eval_polynomial(coordinates: Array, coeffs: Array, powers: Array) -> Array:
    """Expand sampled polynomial terms with paired leading coefficient batches.

    ``coeffs`` ends in the term count defined by ``powers``. Leading coefficient
    and coordinate batches broadcast together, followed by the sampled output
    axes. Coordinates and powers follow :func:`polynomial_basis`.
    """
    terms, spatial_ndim = _polynomial_layout(coordinates, powers)
    coeffs = np.asarray(coeffs)
    term_axis = terms.ndim - spatial_ndim - 1
    n_terms = terms.shape[term_axis]
    if coeffs.ndim == 0 or coeffs.shape[-1] != n_terms:
        raise ValueError(f"coeffs must end in a term axis of size {n_terms}.")

    # Pair the leading batches, then expand coefficients over sampled coordinates.
    coeffs, terms = broadcast_batches(coeffs, terms, ndims=(1, spatial_ndim + 1))
    leading = coeffs.shape[:-1]
    coeffs = coeffs.reshape(coeffs.shape + (1,) * spatial_ndim)
    return np.sum(coeffs * terms, axis=len(leading))


def solve_polynomial(values: Array, coordinates: Array, powers: Array) -> Array:
    """Return minimum-norm polynomial coefficients for paired sampled batches.

    ``values`` ends in the coordinate sampling shape. Leading value and coordinate
    batches broadcast together; the output ends in a coefficient term axis. This
    solve represents the sampled values and is not necessarily a unique inverse.
    """
    terms, spatial_ndim = _polynomial_layout(coordinates, powers)
    values = np.asarray(values)
    term_axis = terms.ndim - spatial_ndim - 1
    n_terms = terms.shape[term_axis]
    spatial_shape = terms.shape[-spatial_ndim:] if spatial_ndim else ()
    if spatial_ndim and values.shape[-spatial_ndim:] != spatial_shape:
        raise ValueError(f"values must end in the coordinate shape {spatial_shape}.")

    dimensions = (spatial_ndim, spatial_ndim + 1)
    values, terms = broadcast_batches(values, terms, ndims=dimensions)
    # Broadcasting may add leading dimensions, so read the resulting term axis.
    leading = terms.shape[: terms.ndim - spatial_ndim - 1]
    terms = flatten_batch(terms, spatial_ndim + 1)
    values = flatten_batch(values, spatial_ndim)

    # solve_basis owns flattening the sampled dimensions for each linear solve.
    solve = lambda basis, value: solve_basis(value, basis, axes=1)
    coeffs = vmap(solve)(terms, values)
    return coeffs.reshape(leading + (n_terms,))


class Polynomial(Sampled, ParametricBasis):
    """Expand coefficients over monomials at contextual coordinates.

    A D-variable coordinate field has shape ``(..., D, *spatial_shape)`` with D
    sampled axes. Leading coefficient and coordinate batches broadcast together.
    One-variable polynomials also accept a bare scalar or array of samples. The
    coefficient order is increasing total degree, with the first variable's power
    descending within each degree. Coordinates and coefficients use caller-defined
    units; no normalisation or centring is implicit. Optional ``unit`` converts the
    complete output after contraction. Because raw coordinate powers differ,
    individual coefficients need not share the same physical dimension.

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

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)
    grid: UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

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
        grid=None,
        transform=None,
        strict=False,
        unit=None,
        empty=False,
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
        grid : UnitSpec or None
            Optional default sampling, with one dimension per variable.
        transform : callable or None
            Map the evaluation coordinates before expanding monomials.
        strict : bool
            Reject runtime replacement of stored sampling or a stored transform.
        empty : bool
            Retain omitted coefficients as None; otherwise initialise stored zeros.
        unit : str or None
            Output unit, converted after contraction. None leaves values unchanged.
        alias : mapping or None
            Optional parameter aliases.
        """
        super().__init__(grid, transform, strict, alias=alias)
        self.powers = polynomial_powers(degree, ndim, powers=powers, degrees=degrees)
        if self.powers.shape[1] == 0:
            raise ValueError("Polynomial requires at least one term.")
        self.coeffs = initialise_coeffs(coeffs, (self.powers.shape[1],), empty=empty)
        self.unit = self.validate_unit(unit)

    @property
    def coeffs_ndim(self) -> int:
        """Polynomial terms occupy one trailing coefficient axis."""
        return 1

    def sampling_context(self, grid=None, *, coordinates=None, **context):
        """Retain an Axis's coordinate component so its batches stay separate.

        Explicit bare coordinates keep their arbitrary sample shape. A sampling
        UnitSpec instead defines exactly one sampled axis per coordinate dimension.
        """
        local = super().sampling_context(grid, coordinates=coordinates, **context)
        if coordinates is None and self.powers.shape[0] == 1:
            local["coordinates"] = np.expand_dims(local["coordinates"], axis=-2)
        return local

    def calculate_basis(
        self, *, grid=None, coordinates=None, transform=None, **context
    ):
        """Return monomials on the stored or supplied, optionally mapped samples."""
        context = self.sampling_context(
            grid,
            coordinates=coordinates,
            transform=transform,
            ndim=self.powers.shape[0],
            **context,
        )
        return polynomial_basis(context["coordinates"], self.powers, axis=None)

    def evaluate(
        self, *, grid=None, coordinates=None, transform=None, **context
    ) -> Array:
        """Expand coefficients on the same samples used by calculate_basis and solve.

        Explicit coordinates retain scalar and general one-variable sampling.
        Stored sampling supplies the default; runtime inputs follow Sampled's
        override policy. Coordinate mappings are applied exactly once.
        """
        context = self.sampling_context(
            grid,
            coordinates=coordinates,
            transform=transform,
            ndim=self.powers.shape[0],
            **context,
        )
        coeffs = zdx.resolve(self.coeffs, **context)
        if coeffs is None or isinstance(coeffs, zdx.Expression):
            raise zdx.Unresolved("Polynomial coefficients need their required context.")
        values = eval_polynomial(context["coordinates"], coeffs, self.powers)
        return values * unit_factor(self.unit)

    def solve(
        self, y: Array, *, grid=None, coordinates=None, transform=None, **context
    ) -> Array:
        """Recover coefficients using the same sampling and coordinate mapping."""
        context = self.sampling_context(
            grid,
            coordinates=coordinates,
            transform=transform,
            ndim=self.powers.shape[0],
            **context,
        )
        return solve_polynomial(
            y / unit_factor(self.unit), context["coordinates"], self.powers
        )
