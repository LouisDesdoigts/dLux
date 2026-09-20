"""General scalar numerical operations."""

from collections.abc import Callable
from typing import Any

import jax.numpy as np
from jax import Array

__all__ = ["triangular_number", "nandiv", "from_complex", "to_complex"]


def triangular_number(n: int) -> int:
    """Calculate the ``n``th triangular number."""
    return n * (n + 1) / 2


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


def to_complex(array: Array, complex: bool = True, *, axis: int = 0) -> Array:
    """Reconstruct complex values from two channels along axis.

    complex=True interprets the channels as real/imaginary; False uses
    amplitude/phase. The channel axis is removed from the result.
    """
    first, second = np.moveaxis(array, axis, 0)
    if complex:
        return first + 1j * second
    return first * np.exp(1j * second)


def from_complex(
    array: Array, complex: bool = True, *, axis: int = 0
) -> tuple[Array, Callable[[Array], Array]]:
    """Map a complex array to a two-channel representation.

    Parameters
    ----------
    array : Array
        The input complex array.
    complex : bool = True
        If True, map to real and imaginary components. If False, map to amplitude and
        phase.
    axis : int = 0
        Where to insert the two-channel axis. Use -1 for trailing value channels.

    Returns
    -------
    vals : Array
        The 2-channel representation of the input array.
    return_fn : Callable
        Function to reconstruct the original complex array from `vals`.
    """
    if complex:
        vals = np.stack((array.real, array.imag), axis=axis)
    else:
        vals = np.stack((np.abs(array), np.angle(array)), axis=axis)
    return_fn = lambda values: to_complex(values, complex, axis=axis)
    return vals, return_fn
