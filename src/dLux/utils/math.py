"""Provide general numerical operations used throughout dLux."""

from typing import Any

import jax.numpy as np
import jax.scipy as jsp
import jax.tree as jtu
from jax import Array, lax

import dLux.utils as dlu

__all__ = [
    "gaussian",
    "mv_gaussian",
    "factorial",
    "triangular_number",
    "eval_basis",
    "solve_basis",
    "nandiv",
]


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

    # Generate per-axis coordinates and corresponding 1D Gaussians
    def gauss_fn(axis, m, s):
        safe = np.where(s == 0, 1.0, s)
        gaussian = jsp.stats.norm.pdf(axis, loc=m, scale=safe)
        delta = np.abs(axis - m) == np.min(np.abs(axis - m))
        return np.where(s == 0, delta, gaussian)

    linspaces = jtu.map(lambda n, e: np.linspace(-e, e, n), npixels, tuple(extent))
    one_d_gauss = jtu.map(gauss_fn, linspaces, tuple(mean), tuple(std))

    # Construct nD separable Gaussian kernel from 1D marginals
    kernel = np.array(np.meshgrid(*one_d_gauss, indexing="ij")).prod(0)
    return kernel / np.sum(kernel)


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
    shape = tuple(len(axis) for axis in axes)
    grids = np.meshgrid(*axes, indexing="ij")
    points = np.stack(grids, axis=-1).reshape(-1, ndim)
    kernel = jsp.stats.multivariate_normal.pdf(points, mean=mean, cov=cov)
    kernel = kernel.reshape(shape)

    # Normalise the sampled kernel
    return kernel / np.sum(kernel)


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


def eval_basis(basis: Array, coeffs: Array) -> Array:
    """Performs an n-dimensional dot-product between the basis and coefficients arrays.

    Parameters
    ----------
    basis: Array
        The basis to use.
    coeffs: Array
        The Array of coefficients to be applied to each basis vector.
    """
    shape = coeffs.shape
    if basis.shape[: coeffs.ndim] != shape:
        raise ValueError(
            "The leading basis dimensions must match the coefficient shape, "
            f"received {basis.shape} and {shape}."
        )

    axes = tuple(range(coeffs.ndim))
    return np.tensordot(basis, coeffs, axes=(axes, axes))


def solve_basis(array: Array, basis: Array) -> Array:
    """Solves for the coefficients of an array over a basis using least squares.

    Parameters
    ----------
    array : Array
        The array to solve for the basis coefficients of.
    basis : Array
        The basis to solve over. Its trailing dimensions must match the shape of
        ``array``.

    Returns
    -------
    coeffs : Array
        The least-squares coefficients with shape ``basis.shape[:-array.ndim]``.
    """
    output_shape = basis.shape[-array.ndim :] if array.ndim else ()
    if output_shape != array.shape:
        raise ValueError(
            "The trailing dimensions of basis must match the shape of array, "
            f"received {output_shape} and {array.shape}."
        )

    matrix = basis.reshape((-1, array.size)).T
    out_shape = basis.shape[: -array.ndim] if array.ndim else basis.shape
    return np.linalg.lstsq(matrix, array.ravel(), rcond=None)[0].reshape(out_shape)


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
