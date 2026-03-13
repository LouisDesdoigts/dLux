import jax.numpy as np
from jax import Array, vmap

__all__ = ["soft_binarise"]


def _lsq_matrix(n: int, m: int = None) -> Array:
    """
    Calculate the least-squares matrix for fitting a plane to an n x m patch on the
    unit square.

    Parameters
    ----------
    n: int
        Number of pixels along the x-axis.
    m: int, optional
        Number of pixels along the y-axis. If None, assumed to be equal to n.

    Returns
    -------
    M: Array
        The least-squares matrix of shape (3, n*m) that can be used to fit a plane to
        an n x m patch.
    """
    xs = np.linspace(0.0, 1.0, n)
    if m is None:
        yy, xx = np.meshgrid(xs, xs, indexing="ij")
        N = n**2
    else:
        ys = np.linspace(0.0, 1.0, m)
        yy, xx = np.meshgrid(xs, ys, indexing="ij")
        N = n * m
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(N)], axis=1)
    return np.linalg.pinv(A).astype(xx.dtype)


def _calc_area_fraction(coefficients: Array, epsilon: float = 1e-15) -> float:
    """
    Calculates the fractional area of a plane below zero using the a, b, c coefficients
    from a plane defined by `z = ax + by + c`. The area is calculated as the integral
    of the plane.

    Parameters
    ----------
    coefficients: Array
        The a, b, c coefficients of the plane, where z = ax + by + c.
    epsilon: float
        Small value to avoid division by zero when coefficients are zero.

    Returns
    -------
    frac: float
        The fraction of the plane below zero.
    """
    # Apply the stabilisation epsilon
    a, b, c = np.where(coefficients == 0, epsilon, coefficients)

    # Calculate the x-bounds
    x1 = (-b - c) / a
    x2 = -c / a
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    x1 = np.maximum(lo, 0.0)
    x2 = np.minimum(hi, 1.0)

    # Calculate the fractional area
    return (
        x1
        + (-c / b) * x2
        - (0.5 * a / b) * x2**2
        - (-c / b) * x1
        + (0.5 * a / b) * x1**2
    )


def soft_binarise(array: Array, oversample: int = 3) -> Array:
    """
    Applies the CLIMB algorithm originally introduced in Wong et al 2021 (TODO: add
    link). Assumes a square input array.

    Parameters
    ----------
    array: Array
        The continuous array to binarise
    oversample: int = 3
        The oversample of the input array

    Returns
    -------
    soft_binary: Array
        The binarised array with soft edges
    """
    # Make sure the array can be downsampled
    if array.shape[-1] % oversample != 0:
        raise ValueError("Array shape must be a multiple of oversample")
    n_tile = array.shape[-1] // oversample

    # Reshape the array into a vector of unit cells
    tiles = array.reshape(n_tile, oversample, n_tile, oversample).transpose(0, 2, 1, 3)
    flat_tiles = tiles.reshape(-1, oversample, oversample)  # (N,b,b)
    flat_pixels = flat_tiles.reshape(flat_tiles.shape[0], -1)  # (N,P)

    # Calculate the fractional area
    M = _lsq_matrix(oversample)  # (3,P)
    coeffs = flat_pixels @ M.T  # (N,3)
    frac = vmap(_calc_area_fraction)(coeffs)  # (N,)

    # Assign the sign of the area using the mean value of the patch
    means = np.mean(flat_tiles, axis=(1, 2))
    frac = np.maximum(frac, 1 - frac)
    frac = np.where(means >= 0, frac, 1 - frac)

    # Get a mask of the positive, negative and boundary regions
    all_pos = np.all(flat_tiles > 0, axis=(1, 2))
    all_nonpos = np.all(flat_tiles <= 0, axis=(1, 2))
    any_zero = np.any(flat_tiles == 0, axis=(1, 2))

    # Map the values to the different regions
    frac = np.where(all_pos, 1.0, frac)
    frac = np.where(all_nonpos, 0.0, frac)
    frac = np.where(any_zero, (frac > 0).astype(frac.dtype), frac)

    # Clip values between 0 and 1 and reshape
    return np.clip(frac, 0, 1).reshape(n_tile, n_tile)
