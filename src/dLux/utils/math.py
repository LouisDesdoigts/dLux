"""Provide general numerical operations used throughout dLux."""

from typing import Any
from math import prod
from operator import index

import jax.numpy as np
import jax.scipy.linalg as jspl
import jax.tree as jtu
from jax import Array, lax

import dLux.utils as dlu

__all__ = [
    "gaussian_profile",
    "mv_gaussian_profile",
    "gaussian",
    "mv_gaussian",
    "factorial",
    "triangular_number",
    "eval_basis",
    "solve_basis",
    "nandiv",
]


def _gaussian_profile(coordinates, mean, std, ndim):
    """Evaluate a separable relative Gaussian on Cartesian coordinates."""
    component_axis = -ndim - 1
    spatial_axes = tuple(range(-ndim, 0))
    shape = (1,) * ndim

    # Broadcast distribution parameters over the coordinate sampling
    mean = dlu.as_axis(mean, ndim, "mean")
    std = dlu.as_axis(std, ndim, "std")
    mean = mean.reshape(mean.shape + shape)
    std = std.reshape(std.shape + shape)
    offset = coordinates - mean

    # Build the quadratic exponent and nearest-sample support for zero widths
    safe = np.where(std == 0, 1.0, std)
    terms = np.where(std == 0, 0.0, np.square(offset / safe))
    exponent = 0.5 * terms.sum(component_axis)
    distance = np.abs(offset)
    minimum = distance.min(spatial_axes, keepdims=True)
    rtol = 4 * np.finfo(distance.dtype).eps
    delta = np.isclose(distance, minimum, rtol=rtol, atol=0.0)
    support = np.where(std == 0, delta, True).all(component_axis)

    # Shift by the sampled peak before exponentiating to retain narrow profiles
    reference = np.where(support, exponent, np.inf)
    reference = reference.min(spatial_axes, keepdims=True)
    return np.where(support, np.exp(reference - exponent), 0.0)


def _mv_gaussian_profile(coordinates, mean, cov, ndim):
    """Evaluate a covariance-defined relative Gaussian on Cartesian coordinates."""
    component_axis = -ndim - 1
    spatial_shape = (1,) * ndim

    # Align the distribution parameters with the coordinate component axis
    mean = dlu.as_axis(mean, ndim, "mean")
    mean = mean.reshape(mean.shape + spatial_shape)
    offset = coordinates - mean
    points = np.moveaxis(offset, component_axis, -1)

    # Broadcast and flatten the sampled coordinates as triangular-solve columns
    cholesky = np.linalg.cholesky(cov)
    spatial = points.shape[-ndim - 1 : -1]
    leading = np.broadcast_shapes(points.shape[: -ndim - 1], cov.shape[:-2])
    points = np.broadcast_to(points, leading + spatial + (ndim,))
    cholesky = np.broadcast_to(cholesky, leading + (ndim, ndim))
    rhs = np.moveaxis(points, -1, -ndim - 1).reshape(leading + (ndim, -1))

    # Evaluate stable sampled-peak-relative Mahalanobis distances
    whitened = jspl.solve_triangular(cholesky, rhs, lower=True)
    exponent = np.square(whitened).sum(-2).reshape(leading + spatial)
    spatial_axes = tuple(range(-ndim, 0))
    reference = exponent.min(spatial_axes, keepdims=True)
    return np.exp(0.5 * (reference - exponent))


def gaussian_profile(
    coordinates: Array,
    mean: float | Array = 0.0,
    std: float | Array = 1.0,
) -> Array:
    """Evaluate a relative axis-aligned Gaussian on two-dimensional coordinates.

    Parameters
    ----------
    coordinates : Array
        Cartesian coordinates with shape ``(..., 2, ny, nx)``. Components follow
        physical ``(x, y)`` order and final axes follow array ``(y, x)`` order.
    mean : float or Array
        Scalar or physical ``(x, y)`` centre in the coordinate unit. Leading axes
        broadcast against coordinate and width batches.
    std : float or Array
        Non-negative scalar or physical ``(x, y)`` standard deviation in the
        coordinate unit. Zero widths select the samples nearest to the corresponding
        mean component.

    Returns
    -------
    profile : Array
        Relative profile with final shape ``(ny, nx)`` and broadcast leading axes.
        Every sampled profile has peak one and is not sum-normalised over the grid.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
        raise ValueError("coordinates must have shape (..., 2, ny, nx).")
    return _gaussian_profile(coordinates, mean, std, 2)


def mv_gaussian_profile(coordinates: Array, mean: Array, cov: Array) -> Array:
    """Evaluate a relative covariance Gaussian on two-dimensional coordinates.

    Parameters
    ----------
    coordinates : Array
        Cartesian coordinates with shape ``(..., 2, ny, nx)``. Components follow
        physical ``(x, y)`` order and final axes follow array ``(y, x)`` order.
    mean : Array
        Scalar or physical ``(x, y)`` centre in the coordinate unit. Leading axes
        broadcast against coordinate and covariance batches.
    cov : Array
        Symmetric positive-definite covariance with trailing shape ``(2, 2)`` in
        physical ``(x, y)`` order. Leading axes vectorise covariance matrices.

    Returns
    -------
    profile : Array
        Relative profile with final shape ``(ny, nx)`` and broadcast leading axes.
        Every sampled profile has peak one and is not sum-normalised over the grid.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
        raise ValueError("coordinates must have shape (..., 2, ny, nx).")
    if cov.ndim < 2 or cov.shape[-2:] != (2, 2):
        raise ValueError("cov must have trailing shape (2, 2).")
    return _mv_gaussian_profile(coordinates, mean, cov, 2)


def gaussian(
    mean: float | Array = 0.0,
    std: float | Array = 1.0,
    npixels: int | tuple[int, ...] = 64,
    extent: float | Array = 5.0,
) -> Array:
    """Generates a normalised n-dimensional Gaussian function.

    Parameters
    ----------
    mean : float | Array = 0.0
        The centre position(s) of the Gaussian. Scalar for 1D, array for nD.
    std : float | Array = 1.0
        The standard deviation(s) of the Gaussian. Scalar for 1D, array for nD.
    npixels : int | tuple[int, ...] = 64
        The number of pixels along each axis. Scalar for 1D, tuple for nD.
    extent : float or Array = 5.0
        Per-axis coordinate extent on each side of the origin.

    Returns
    -------
    kernel : Array
        The normalised n-dimensional Gaussian kernel.
    """
    # Resolve dimensionality and broadcast each axis input
    npixels = dlu.as_size(npixels, name="npixels")
    mean, std = dlu.as_axis(mean), dlu.as_axis(std)
    extent = dlu.as_axis(extent)
    ndim = max(len(npixels), mean.shape[-1], std.shape[-1], extent.shape[-1])
    npixels = dlu.as_size(npixels, ndim, "npixels")
    mean, std = dlu.as_axis(mean, ndim, "mean"), dlu.as_axis(std, ndim, "std")
    extent = dlu.as_axis(extent, ndim, "extent")

    # Generate the Cartesian sampling coordinates
    linspaces = jtu.map(lambda n, e: np.linspace(-e, e, n), npixels, tuple(extent))
    coordinates = np.array(np.meshgrid(*linspaces, indexing="ij"))

    # Evaluate and independently normalise every leading profile
    kernel = _gaussian_profile(coordinates, mean, std, ndim)
    spatial_axes = tuple(range(-ndim, 0))
    return kernel / kernel.sum(spatial_axes, keepdims=True)


def mv_gaussian(
    mean: Array,
    cov: Array,
    npix: int | tuple[int, ...] = 64,
    extent: float | Array = 5.0,
) -> Array:
    """Generates a normalised multivariate Gaussian function.

    Parameters
    ----------
    mean : Array
        The mean vector of the multivariate Gaussian. Shape (ndim,).
    cov : Array
        The covariance matrix of the multivariate Gaussian. Shape (ndim, ndim).
    npix : int | Array = 64
        The number of pixels along each axis.
    extent : float or Array = 5.0
        Per-axis marginal standard-deviation multiplier.

    Returns
    -------
    kernel : Array
        The normalised multivariate Gaussian kernel.
    """
    # Validate the distribution and output dimensions
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("mean must be a non-empty one-dimensional array.")
    if cov.shape != (mean.size, mean.size):
        raise ValueError("cov shape must match (mean.size, mean.size).")

    ndim = mean.size
    npix = dlu.as_size(npix, ndim, "npix")
    extent = dlu.as_axis(extent, ndim, "extent")

    # Generate physical axes spanning the marginal standard deviations
    stds = np.sqrt(np.diag(cov))
    axis_fn = lambda m, s, n, e: np.linspace(m - e * s, m + e * s, n)
    axes = jtu.map(axis_fn, tuple(mean), tuple(stds), npix, tuple(extent))

    # Evaluate the distribution over the Cartesian product of the axes
    coordinates = np.array(np.meshgrid(*axes, indexing="ij"))
    kernel = _mv_gaussian_profile(coordinates, mean, cov, ndim)

    # Normalise the sampled kernel
    spatial_axes = tuple(range(-ndim, 0))
    return kernel / kernel.sum(spatial_axes, keepdims=True)


def factorial(n: float) -> float:
    """Calculate ``n!`` through the JAX-compatible gamma function."""
    n = np.asarray(n, float)
    return lax.cond(
        n == 0,
        lambda x: np.asarray(1.0, dtype=x.dtype),
        lambda x: lax.exp(lax.lgamma(x + 1.0)),
        n,
    )


def triangular_number(n: int) -> int:
    """Calculate the ``n``th triangular number."""
    return n * (n + 1) / 2


def eval_basis(basis: Array, coeffs: Array, axes: int | None = None) -> Array:
    """Contract trailing coefficient axes with leading sampled basis axes.

    Parameters
    ----------
    basis: Array
        The basis to use.
    coeffs: Array
        Coefficients ending in ``basis.shape[:axes]``. Leading axes are independent
        batches; basis axes after the contraction are retained as output axes.
    axes : int or None
        Positive contraction-axis count. ``None`` contracts every coefficient axis,
        preserving the original unbatched utility convention.
    """
    basis, coeffs = np.asarray(basis), np.asarray(coeffs)
    axes = coeffs.ndim if axes is None else index(axes)
    if axes < 1 or axes > basis.ndim or axes > coeffs.ndim:
        raise ValueError("axes must identify at least one coefficient and basis axis.")
    if basis.shape[:axes] != coeffs.shape[-axes:]:
        raise ValueError(
            "The leading basis dimensions must match the trailing coefficient "
            f"dimensions, received {basis.shape[:axes]} and {coeffs.shape[-axes:]}."
        )
    coeff_axes = tuple(range(coeffs.ndim - axes, coeffs.ndim))
    return np.tensordot(coeffs, basis, axes=(coeff_axes, tuple(range(axes))))


def solve_basis(array: Array, basis: Array, axes: int | None = None) -> Array:
    """Return minimum-norm least-squares coefficients over a sampled basis.

    Parameters
    ----------
    array : Array
        The array to solve for the basis coefficients of.
    basis : Array
        Basis with leading coefficient axes and trailing sampled output axes.
    axes : int or None
        Positive coefficient-axis count. ``None`` treats the complete ``array``
        shape as the output shape. An explicit count preserves leading value batches.

    Returns
    -------
    coeffs : Array
        Leading value batches followed by ``basis.shape[:axes]``. This is a
        representative solution, not an exact inverse for a rank-deficient basis.
    """
    array, basis = np.asarray(array), np.asarray(basis)
    axes = basis.ndim - array.ndim if axes is None else index(axes)
    if axes < 1 or axes > basis.ndim or 0 in basis.shape:
        raise ValueError("axes must identify non-empty leading coefficient axes.")
    coefficient_shape, output_shape = basis.shape[:axes], basis.shape[axes:]
    output_ndim = len(output_shape)
    if output_ndim and array.shape[-output_ndim:] != output_shape:
        raise ValueError(f"array must end in the sampled output shape {output_shape}.")

    # Flatten only the coefficient and sampled output axes for the linear solve.
    batch_shape = array.shape[:-output_ndim] if output_ndim else array.shape
    dtype = np.result_type(array.dtype, basis.dtype, np.float32)
    matrix = basis.reshape((prod(coefficient_shape), prod(output_shape))).astype(dtype)
    values = array.reshape((-1, prod(output_shape))).astype(dtype)
    # The pseudoinverse also gives zero coefficients for a zero-rank basis.
    coeffs = values @ np.linalg.pinv(matrix)
    return coeffs.reshape(batch_shape + coefficient_shape)


def nandiv(a: Array, b: Array, fill: Any = np.inf) -> Array:
    """Divides two arrays, replacing any NaNs with a fill value.

    Parameters
    ----------
    a : Array
        The numerator.
    b : Array
        The denominator.
    fill : Any = np.inf
        The value to replace NaNs with.

    Returns
    -------
    a / b : Array
        The result of the division.
    """
    # Avoid evaluating a/0 under jax_debug_nans by dividing through a safe denominator
    safe_b = np.where(b == 0, 1, b)
    out = a / safe_b
    return np.where(b == 0, fill, out)
