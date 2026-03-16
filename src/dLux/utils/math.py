from jax import lax, Array
import jax.numpy as np
import jax.scipy as jsp
import jax.tree as jtu
from typing import Any

from .helpers import _cast_scalar, _cast_tuple, _input_len

__all__ = [
    "gaussian",
    "mv_gaussian",
    "factorial",
    "triangular_number",
    "eval_basis",
    "nandiv",
]


def gaussian(
    mean: float | Array = 0.0,
    std: float | Array = 1.0,
    npixels: int | tuple[int, ...] = 64,
    extent: float = 5.0,
) -> Array:
    """
    Generates a normalized n-dimensional Gaussian function.

    Parameters
    ----------
    mean : float | Array = 0.0
        The center position(s) of the Gaussian. Scalar for 1D, array for nD.
    std : float | Array = 1.0
        The standard deviation(s) of the Gaussian. Scalar for 1D, array for nD.
    npixels : int | tuple[int, ...] = 64
        The number of pixels along each axis. Scalar for 1D, tuple for nD.
    extent : float = 5.0
        The extent of the grid in units of standard deviation on each side.

    Returns
    -------
    kernel : Array
        The normalized n-dimensional Gaussian kernel.
    """
    # Check inputs and cast to tuples
    npixels = _cast_tuple(npixels, "npixels")
    ndim = max(len(npixels), _input_len(mean, "mean"), _input_len(std, "std"))
    mean = _cast_scalar(mean, ndim, "mean")
    std = _cast_scalar(std, ndim, "std")

    # Make sure npix is the right dimensionality
    if len(npixels) != ndim:
        npixels *= ndim

    # Generate per-axis coordinates and corresponding 1D Gaussians
    linspaces = jtu.map(
        lambda n: np.linspace(-extent, extent, n),
        npixels,
    )
    one_d_gauss = jtu.map(
        lambda axis, m, s: jsp.stats.norm.pdf(axis, loc=m, scale=s),
        linspaces,
        mean,
        std,
    )

    # Construct nD separable Gaussian kernel from 1D marginals
    kernel = np.array(np.meshgrid(*one_d_gauss, indexing="xy")).prod(0)
    return kernel / np.sum(kernel)


def mv_gaussian(
    mean: Array,
    cov: Array,
    npix: int | Array = 64,
    extent: float = 5.0,
) -> Array:
    """
    Generates a normalized multivariate Gaussian function.

    Parameters
    ----------
    mean : Array
        The mean vector of the multivariate Gaussian. Shape (ndim,).
    cov : Array
        The covariance matrix of the multivariate Gaussian. Shape (ndim, ndim).
    npix : int | Array = 64
        The number of pixels along each axis.
    extent : float = 5.0
        The extent of the grid in units of standard deviation on each side.

    Returns
    -------
    kernel : Array
        The normalized multivariate Gaussian kernel.
    """
    raise NotImplementedError("Self reminder to check input bullshit")

    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    npix_arr = np.atleast_1d(np.asarray(npix, dtype=int))
    ndim = mean.size

    # Get standard deviations from covariance matrix diagonal
    stds = np.sqrt(np.diag(cov))

    # Handle npix broadcasting
    if npix_arr.size == 1:
        npix_arr = np.repeat(npix_arr, ndim)

    # Create linspace function
    def make_axis(i):
        return np.linspace(
            mean[i] - extent * stds[i],
            mean[i] + extent * stds[i],
            npix_arr[i],
        )

    # Generate coordinate arrays for each dimension
    linspaces = jtu.map(make_axis, np.arange(ndim))

    # Create meshgrid
    grids = np.meshgrid(*linspaces, indexing="xy")

    # Stack grids into points array and reshape for computation
    grid_shape = tuple(len(ls) for ls in linspaces)
    points = np.stack(grids, axis=0).reshape(ndim, -1).T  # (n_points, ndim)

    # Compute multivariate Gaussian
    kernel = jsp.stats.multivariate_normal.pdf(points, mean=mean, cov=cov)
    kernel = kernel.reshape(grid_shape)

    # Normalize
    return kernel / np.sum(kernel)


def factorial(n: float) -> float:
    """
    Calculate n! in a JAX-friendly way.

    Parameters
    ----------
    n : float
        The value to calculate the factorial of.

    Returns
    -------
    n! : float
        The factorial of the value.
    """
    n = np.asarray(n, float)
    return lax.cond(
        n == 0,
        lambda x: np.asarray(1.0, dtype=x.dtype),
        lambda x: lax.exp(lax.lgamma(x + 1.0)),
        n,
    )


def triangular_number(n: int) -> int:
    """
    Calculate the nth triangular number.

    Parameters
    ----------
    n : int
        The nth triangular number to calculate.

    Returns
    -------
    n : int
        The nth triangular number.
    """
    return n * (n + 1) / 2


def eval_basis(basis: Array, coefficients: Array) -> Array:
    """
    Performs an n-dimensional dot-product between the basis and coefficients arrays.

    Parameters
    ----------
    basis: Array
        The basis to use.
    coefficients: Array
        The Array of coefficients to be applied to each basis vector.
    """
    ndim = coefficients.ndim
    return np.tensordot(basis, coefficients, axes=2 * (tuple(range(ndim)),))


def nandiv(a: Array, b: Array, fill: Any = np.inf) -> Array:
    """
    Divides two arrays, replacing any NaNs with a fill value.

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
    return np.where(b == 0, fill, a / b)
