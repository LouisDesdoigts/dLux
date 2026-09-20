"""CLIMB soft binarisation and continuous basis parameterisation of binary masks."""

from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array, vmap
import zodiax as zdx

from .basis import Basis, initialise_coeffs

__all__ = ["soft_binarise", "CLIMBBasis"]


def _lsq_matrix(n: int, m: int = None) -> Array:
    """Calculate the least-squares matrix for fitting a plane to an n x m patch on the
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
    m = n if m is None else m
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, m)
    yy, xx = np.meshgrid(xs, ys, indexing="ij")
    design = np.stack([xx.ravel(), yy.ravel(), np.ones(n * m)], axis=1)
    return np.linalg.pinv(design).astype(xx.dtype)


def _calc_area_fraction(coeffs: Array, epsilon: float = 1e-15) -> float:
    """Calculate the fractional area of a plane below zero.

    The plane is defined by ``z = ax + by + c`` and integrated over the pixel.

    Parameters
    ----------
    coeffs: Array
        The a, b, c coefficients of the plane, where z = ax + by + c.
    epsilon: float
        Small value to avoid division by zero when coefficients are zero.

    Returns
    -------
    frac: float
        The fraction of the plane below zero.
    """
    # Apply the stabilisation epsilon
    a, b, c = np.where(coeffs == 0, epsilon, coeffs)

    # Calculate the x-bounds
    x1 = (-b - c) / a
    x2 = -c / a
    lo = np.minimum(x1, x2)
    hi = np.maximum(x1, x2)
    x1 = np.maximum(lo, 0.0)
    x2 = np.minimum(hi, 1.0)

    # Integrate the line -c/b - (a/b)*x between the clipped bounds.
    linear = -c / b
    quadratic = 0.5 * a / b
    return x1 + linear * x2 - quadratic * x2**2 - linear * x1 + quadratic * x1**2


def soft_binarise(array: Array, oversample: int = 3) -> Array:
    """Applies the CLIMB algorithm originally introduced in
    [Wong et al., 2021](https://ui.adsabs.harvard.edu/abs/2021JOSAB..38.2465W/abstract).
    Assumes a square input array.

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


class CLIMBBasis(zdx.Expression):
    """Map a continuous explicit-basis field through CLIMB soft binarisation.

    Coefficients first generate an oversampled latent two-dimensional field. It is
    softly binarised, downsampled by `oversample`, and mapped between two configured
    output values for differentiable binary-mask design. The nonlinear result has
    no coefficient inverse; ``expansion.solve(...)`` only reverses
    the latent linear field.
    """

    expansion: Basis
    values: Array
    oversample: int = eqx.field(static=True, default=3)

    def __init__(
        self,
        basis,
        coeffs=None,
        shape=None,
        values=(0.0, 1.0),
        oversample=3,
        *,
        alias=None,
    ):
        """Initialise a CLIMB-softened explicit basis.

        Parameters
        ----------
        basis : Array
            Explicit sampled basis values.
        coeffs : Array, zodiax.Expression, or None
            Basis coefficients, defaulting to zeros.
        shape : tuple[int, ...] or None
            Coefficient-axis shape, inferred from concrete coefficients. Required
            if coefficients are omitted or have no known shape.
        values : tuple[float, float]
            Lower and upper values used by soft binarisation.
        oversample : int
            Internal soft-binarisation oversampling factor.
        alias : mapping or None
            Optional parameter aliases.
        """
        # Construct the latent basis with a known, editable coefficient array.
        basis = zdx.as_array(basis, dtype=float)
        if coeffs is not None:
            coeffs = zdx.as_array(coeffs, dtype=float)
        if shape is None and isinstance(coeffs, Array):
            shape = coeffs.shape
        if shape is None:
            raise ValueError("shape is required when coeffs has no known shape.")
        shape = tuple(shape)
        self.expansion = Basis(basis, initialise_coeffs(coeffs, shape), axes=len(shape))

        # Require a square latent image with two output levels and divisible sampling.
        output_shape = basis.shape[len(shape) :]
        if len(output_shape) != 2 or output_shape[0] != output_shape[1]:
            raise ValueError("The CLIMB latent output must be a square 2D array.")
        values = zdx.as_array(values, dtype=float)
        if values.shape != (2,):
            raise ValueError("values must contain exactly two output values.")
        self.values = values
        self.oversample = int(oversample)
        if self.oversample < 1:
            raise ValueError("oversample must be a positive integer.")
        if output_shape[0] % self.oversample != 0:
            raise ValueError(
                "The CLIMB latent output size must be divisible by oversample."
            )

        self.alias = alias

    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate and softly binarise the latent field.

        The result is downsampled by ``oversample`` and mapped between the configured
        lower and upper values.
        """
        self = self.resolve_children(**kwargs)
        binary = soft_binarise(self.expansion, self.oversample)
        low, high = self.values
        return low + (high - low) * binary
