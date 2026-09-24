"""Retained numerical helpers and unit-sum sampled Gaussian generators."""

import jax.numpy as np
import jax.tree as jtu
from jax import Array, lax
from jax.nn import softmax

from ..batching import as_axis
from ..grids.array_ops import as_size
from ..numerical.profiles import Profile
from ..bases.basis import eval_basis, solve_basis as _solve_basis

__all__ = ["factorial", "gaussian", "mv_gaussian", "eval_basis", "solve_basis"]


def factorial(n: float) -> float:
    """Retain the scalar factorial's float cast and explicit zero branch.

    New code should use jax.scipy.special.factorial. This compatibility function
    preserves the original zero derivative and log-gamma behaviour for negatives.
    """
    n = np.asarray(n, float)
    return lax.cond(
        n == 0,
        lambda x: np.asarray(1.0, dtype=x.dtype),
        lambda x: lax.exp(lax.lgamma(x + 1.0)),
        n,
    )


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
    npixels = as_size(npixels, name="npixels")
    mean, std = as_axis(mean), as_axis(std)
    extent = as_axis(extent)
    ndim = max(len(npixels), mean.shape[-1], std.shape[-1], extent.shape[-1])
    npixels = as_size(npixels, ndim, "npixels")
    mean, std = as_axis(mean, ndim, "mean"), as_axis(std, ndim, "std")
    extent = as_axis(extent, ndim, "extent")

    # Retain released xy indexing, including the first-two-axis swap in nD.
    linspaces = jtu.map(lambda n, e: np.linspace(-e, e, n), npixels, tuple(extent))
    coordinates = np.array(np.meshgrid(*linspaces, indexing="xy"))

    # Evaluate and independently normalise every leading profile
    logarithms = Profile._discrete_log_gaussian(coordinates, mean, std, ndim)
    spatial_axes = tuple(range(-ndim, 0))
    return softmax(logarithms, axis=spatial_axes)


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
    raise NotImplementedError("Multivariate Gaussian generation is under development.")


from ..numerical.math import triangular_number, nandiv

__all__ += ["triangular_number", "nandiv"]


def solve_basis(array, basis):
    """Retain the released trailing-output shape check and solve current kernels."""
    output_shape = basis.shape[-array.ndim :] if array.ndim else ()
    if output_shape != array.shape:
        raise ValueError("The trailing dimensions of basis must match the array.")
    return _solve_basis(array, basis)
