"""Utilities for ordinary, Zernike, and polygonal polynomials."""

from __future__ import annotations

from itertools import product
from math import prod
from operator import index

import jax.numpy as np
from jax import lax, vmap, Array, custom_jvp
import dLux.utils as dlu
import equinox as eqx

__all__ = [
    "polynomial_powers",
    "polynomial_basis",
    "eval_polynomial",
    "solve_polynomial",
    "zernike_name",
    "noll_indices",
    "radial_orders_to_indices",
    "zernike_factors",
    "eval_radial",
    "eval_azimuthal",
    "zernike",
    "zernike_fast",
    "zernike_basis",
    "polike",
    "polike_fast",
    "polike_basis",
]


def polynomial_powers(degree: int, ndim: int = 1) -> Array:
    """Generate integer exponents in ascending total-degree order.

    The result has shape ``(ndim, n_terms)``. Within each total degree the first
    variable's exponent decreases first, preserving dLux's coefficient ordering.
    ``degree`` is a non-negative integer and ``ndim`` is a positive integer.
    """
    if isinstance(degree, bool) or isinstance(ndim, bool):
        raise TypeError("degree and ndim must be integers.")
    degree = index(degree)
    ndim = index(ndim)
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if ndim < 1:
        raise ValueError("ndim must be positive.")
    powers = [
        values
        for total in range(degree + 1)
        for values in product(range(total, -1, -1), repeat=ndim)
        if sum(values) == total
    ]
    return np.asarray(powers, dtype=int).T


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

    leading = np.broadcast_shapes(coeffs.shape[:-1], terms.shape[:term_axis])
    spatial_shape = terms.shape[-spatial_ndim:] if spatial_ndim else ()
    coeffs = np.broadcast_to(coeffs, leading + (n_terms,))
    coeffs = coeffs.reshape(leading + (n_terms,) + (1,) * spatial_ndim)
    terms = np.broadcast_to(terms, leading + (n_terms,) + spatial_shape)
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

    leading = values.shape[:-spatial_ndim] if spatial_ndim else values.shape
    leading = np.broadcast_shapes(leading, terms.shape[:term_axis])
    terms = np.broadcast_to(terms, leading + (n_terms,) + spatial_shape)
    values = np.broadcast_to(values, leading + spatial_shape)
    dtype = np.result_type(values.dtype, terms.dtype, np.float32)
    matrices = terms.reshape((-1, n_terms, prod(spatial_shape))).astype(dtype)
    values = values.reshape((-1, prod(spatial_shape))).astype(dtype)

    # Solve each independent sampled system without mixing its batch axes.
    solve = lambda matrix, value: value @ np.linalg.pinv(matrix)
    coeffs = vmap(solve)(matrices, values)
    return coeffs.reshape(leading + (n_terms,))


zernike_names = {
    # 0th Radial
    1: "Piston",
    # 1st Radial
    2: "Tilt X",
    3: "Tilt Y",
    # Second Radial
    4: "Defocus",
    5: "Astig X",
    6: "Astig Y",
    # Third Radial
    7: "Coma X",
    8: "Coma Y",
    9: "Trefoil X",
    10: "Trefoil Y",
    # Fourth Radial
    11: "Spherical",
    12: "2nd Astig X",
    13: "2nd Astig Y",
    14: "Quadrafoil X",
    15: "Quadrafoil Y",
    # Fifth Radial
    16: "2nd Coma X",
    17: "2nd Coma Y",
    18: "2nd Trefoil X",
    19: "2nd Trefoil Y",
    20: "Pentafoil X",
    21: "Pentafoil Y",
    # Sixth Radial
    22: "2nd Spherical",
    23: "3rd Coma X",
    24: "3rd Coma Y",
    25: "3rd Astig X",
    26: "3rd Astig Y",
    27: "Hexafoil X",
    28: "Hexafoil Y",
    # Seventh Radial
    29: "4th Coma X",
    30: "4th Coma Y",
    31: "4th Astig X",
    32: "4th Astig Y",
    33: "3rd Trefoil X",
    34: "3rd Trefoil Y",
    35: "Heptafoil X",
    36: "Heptafoil Y",
}


def radial_orders_to_indices(radial_orders: list[int]) -> list[int]:
    """Convert radial Zernike orders to their complete Noll index sequence."""
    indices = []
    for order in radial_orders:
        order = int(order)
        if order < 0:
            raise ValueError("radial_orders must contain non-negative integers.")
        first = order * (order + 1) // 2 + 1
        indices.extend(range(first, first + order + 1))
    return indices


def zernike_name(j: int) -> str:
    """Gets the name of the jth Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    name : str
        The name of the Zernike polynomial.
    """
    return zernike_names[int(j)] if j >= 1 and j <= 36 else f"Zernike {int(j)}"


def noll_indices(j: int) -> tuple[int]:
    """Calculate the radial and azimuthal orders of the Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    n, m : tuple[int]
        The radial and azimuthal orders of the Zernike polynomial.
    """
    n = (np.ceil(-1 / 2 + np.sqrt(1 + 8 * j) / 2) - 1).astype(int)
    smallest_j_in_row = n * (n + 1) / 2 + 1
    number_of_shifts = (j - smallest_j_in_row + ~(n & 1) + 2) // 2
    sign_of_shift = -(j & 1) + ~(j & 1) + 2
    base_case = n & 1
    m = (sign_of_shift * (base_case + number_of_shifts * 2)).astype(int)
    return int(n), int(m)


def zernike_factors(j: int) -> tuple[Array]:
    """Calculates the normalisation coefficients and powers of the Zernike polynomial.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.

    Returns
    -------
    c, k : tuple[Array]
        The normalisation coefficients and powers of the Zernike polynomial.
    """
    if j < 1:
        raise ValueError("The Zernike index must be greater than 0.")
    n, m = noll_indices(j)

    # Calculate k
    k = np.arange(((n - m) // 2) + 1, dtype=float)

    # Calculate c
    sign = lax.pow(-1.0, k)
    _fact_1 = vmap(dlu.factorial)(np.abs(n - k))
    _fact_2 = vmap(dlu.factorial)(k)
    _fact_3 = vmap(dlu.factorial)(((n + m) // 2) - k)
    _fact_4 = vmap(dlu.factorial)(((n - m) // 2) - k)
    c = sign * _fact_1 / _fact_2 / _fact_3 / _fact_4
    return c, k


@eqx.filter_jit
def eval_radial(rho: Array, n: int, c: Array, k: Array) -> Array:
    """Calculates the radial component of the Zernike polynomial.

    Parameters
    ----------
    rho : Array
        The radial coordinate of the Zernike polynomial.
    n : int
        The radial order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.

    Returns
    -------
    radial : Array
        The radial component of the Zernike polynomial.
    """
    rads = lax.pow(rho[:, :, None], (np.abs(n) - 2 * k)[None, None, :])
    return (c * rads).sum(axis=2)


@eqx.filter_jit
def eval_azimuthal(theta: Array, n: int, m: int) -> Array:
    """Calculates the azimuthal component of the Zernike polynomial.

    Parameters
    ----------
    theta : Array
        The azimuthal coordinate of the Zernike polynomial.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.

    Returns
    -------
    azimuthal : Array
        The azimuthal component of the Zernike polynomial.
    """
    # Get the radial and angular factors
    norm_coeff = np.sqrt(n + 1) * np.where(m == 0, 1.0, np.sqrt(2))
    angle = np.abs(m) * theta

    # Evaluate the positive or negative azimuthal convention
    azimuthal = np.where(m >= 0, np.cos(angle), np.sin(angle))
    return norm_coeff * azimuthal


def scale_coords(coords: Array, rmax: float) -> Array:
    """Scale coordinates to a unit circle for polynomial evaluation.

    Parameters
    ----------
    coords : Array
        The coordinates to scale.
    rmax : float
        The radius to scale to.

    Returns
    -------
    coords : Array
        The scaled coordinates.
    """
    rmax = np.asarray(rmax)
    if rmax.ndim and rmax.shape[-1] == coords.shape[-3]:
        rmax = rmax.reshape(rmax.shape + (1,) * (coords.ndim - rmax.ndim))
    return coords / rmax


def zernike(j: int, coordinates: Array, diameter: float = 2) -> Array:
    """Calculates the Zernike polynomial. Note that this function is not-jittable as is
    has dynamic array shapes. To use this function in a jittable way, use the
    zernike_fast function, with the pre-calculated c and k parameters.

    Parameters
    ----------
    j : int
        The Zernike (noll) index.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike : Array
        The Zernike polynomial.
    """
    coordinates = scale_coords(coordinates, diameter / 2)
    polar_coordinates = dlu.cart2polar(coordinates)
    rho = polar_coordinates[0]
    theta = polar_coordinates[1]
    aperture = rho <= 1.0
    n, m = noll_indices(j)
    c, k = zernike_factors(j)
    return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


def zernike_fast(
    n: int, m: int, c: Array, k: Array, coordinates: Array, diameter: float = 2
) -> Array:
    """Calculate a JIT-compatible Zernike mode from precomputed factors.

    Parameters
    ----------
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike : Array
        The Zernike polynomial.
    """
    coordinates = scale_coords(coordinates, diameter / 2)
    polar_coordinates = dlu.cart2polar(coordinates)
    rho = polar_coordinates[0]
    theta = polar_coordinates[1]
    aperture = rho <= 1.0
    return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


def zernike_basis(js: list[int], coordinates: Array, diameter: float = 2) -> Array:
    """Calculates the Zernike polynomial basis. Note that this function is not-jittable.

    Parameters
    ----------
    js : list[int]
        The Zernike (noll) indices.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike_basis : Array
        The Zernike polynomial basis.
    """
    return np.array([zernike(j, coordinates, diameter) for j in js])


def polike(nsides: int, j: int, coordinates: Array, diameter: float = 2) -> Array:
    """Calculates the Zernike polynomial on an n-sided aperture. Note that this function
    is not-jittable as is has dynamic array shapes. To use this function in a jittable
    way, use the polike_fast function, with the pre-calculated c and k parameters.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    j : int
        The Zernike (noll) index.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the Zernike polynomial upon.

    Returns
    -------
    polike : Array
        The Zernike polynomial on an n-sided aperture.
    """
    if nsides < 3:
        raise ValueError(f"nsides must be >= 3, not {nsides}.")
    coordinates = scale_coords(coordinates, diameter / 2)
    theta = dlu.cart2polar(coordinates)[1]
    alpha = np.pi / nsides
    phi = theta + alpha
    wedge = np.floor((phi + alpha) / (2.0 * alpha))
    u_alpha = phi - wedge * (2 * alpha)
    r_alpha = np.cos(alpha) / np.cos(u_alpha)
    return 1 / r_alpha * zernike(j, coordinates / r_alpha)


def polike_fast(
    nsides: int,
    n: int,
    m: int,
    c: Array,
    k: Array,
    coordinates: Array,
    diameter: float = 2,
) -> Array:
    """Calculates the Zernike polynomial on an n-sided aperture using the pre-calculated
    c and k parameters, such that this function is jittable.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the polygonal polynomial upon.

    Returns
    -------
    polike : Array
        The Zernike polynomial on an n-sided aperture.
    """
    if nsides < 3:
        raise ValueError(f"nsides must be >= 3, not {nsides}.")
    coordinates = scale_coords(coordinates, diameter / 2)
    alpha = np.pi / nsides
    phi = dlu.cart2polar(coordinates)[1] + alpha
    wedge = np.floor((phi + alpha) / (2.0 * alpha))
    u_alpha = phi - wedge * (2 * alpha)
    r_alpha = np.cos(alpha) / np.cos(u_alpha)
    return 1 / r_alpha * zernike_fast(n, m, c, k, coordinates / r_alpha)


def polike_basis(nsides: int, js: list[int], coordinates: Array, diameter: float = 2):
    """Calculates the Zernike polynomial basis on an n-sided aperture. Note that this
    function is not-jittable.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    js : list[int]
        The Zernike (noll) indices.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.
    diameter : float = 2
        The diameter of the aperture to calculate the Zernike polynomial upon.

    Returns
    -------
    polike_basis : Array
        The Zernike polynomial basis on an n-sided aperture.
    """
    return np.array([polike(nsides, j, coordinates, diameter) for j in js])
