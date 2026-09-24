"""Retained masked norm functions backed by the shared measurement kernel."""

import jax.numpy as np
from jax import Array

from ..numerical.norms import norm, normalise

__all__ = ["l1_norm", "l2_norm", "max_norm", "rms_norm", "p2v_norm", "normalise"]


def _resolve_mask(array, mask):
    """Align a compact mask without changing the legacy RMS denominator."""
    if mask is None:
        return np.ones(array.shape)
    mask = np.asarray(mask, dtype=float)
    if mask.ndim > array.ndim:
        raise ValueError("Mask has more dimensions than the input array.")
    return mask.reshape((1,) * (array.ndim - mask.ndim) + mask.shape)


def l1_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Calculate the optionally masked sum of absolute values.

    Axis and keepdims follow JAX reduction conventions. NaNs do not contribute
    to the numerator; RMS retains their mask weights in its denominator.
    """
    return norm(array, "l1", _resolve_mask(array, mask), axis, keepdims)


def l2_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Calculate the optionally masked square root of the summed absolute squares.

    Axis and keepdims follow JAX reduction conventions. NaNs do not contribute
    to the numerator; RMS retains their mask weights in its denominator.
    """
    return norm(array, "l2", _resolve_mask(array, mask), axis, keepdims)


def max_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Calculate the optionally masked maximum absolute value.

    Axis and keepdims follow JAX reduction conventions. NaNs do not contribute
    to the numerator; RMS retains their mask weights in its denominator.
    """
    return norm(array, "max", _resolve_mask(array, mask), axis, keepdims)


def rms_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Calculate the optionally masked root-mean-square value.

    Axis and keepdims follow JAX reduction conventions. NaNs do not contribute
    to the numerator; RMS retains their mask weights in its denominator.
    """
    return norm(array, "rms", _resolve_mask(array, mask), axis, keepdims)


def p2v_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    """Calculate the optionally masked peak-to-valley range.

    Axis and keepdims follow JAX reduction conventions. NaNs do not contribute
    to the numerator; RMS retains their mask weights in its denominator.
    """
    return norm(array, "p2v", _resolve_mask(array, mask), axis, keepdims)
