"""Calculate support-aware norms for sampled arrays."""

import jax.numpy as np
from jax import Array

__all__ = ["l1_norm", "l2_norm", "max_norm", "rms_norm", "p2v_norm"]


def _resolve_mask(array, mask):
    """Resolved the mask for a given array."""
    # If None, return an array of ones with the same shape as the input array
    if mask is None:
        return np.ones_like(array)

    # If the mask has less dimensions than the array, broadcast it
    if mask.ndim < array.ndim:
        mask = np.expand_dims(mask, np.arange(array.ndim - mask.ndim))

    # If the mask has more dimensions than the array, raise an error
    if mask.ndim > array.ndim:
        raise ValueError(
            f"Mask has more dimensions ({mask.ndim}) than input array ({array.ndim})."
        )

    # Finally, return the mask
    return mask.astype(float)


def l1_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """Calculate the optionally masked sum of absolute values.

    ``axis`` and ``keepdims`` follow the corresponding NumPy reduction conventions.
    """
    return np.nansum(
        _resolve_mask(array, mask) * np.abs(array), axis=axis, keepdims=keepdims
    )


def l2_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """Calculate the optionally masked square root of the summed squares.

    ``axis`` and ``keepdims`` follow the corresponding NumPy reduction conventions.
    """
    return np.sqrt(
        np.nansum(_resolve_mask(array, mask) * array**2, axis=axis, keepdims=keepdims)
    )


def max_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """Calculate the optionally masked maximum absolute value.

    ``axis`` and ``keepdims`` follow the corresponding NumPy reduction conventions.
    """
    resolved_mask = _resolve_mask(array, mask)
    return np.nanmax(
        np.where(resolved_mask.astype(bool), np.abs(array), -np.inf),
        axis=axis,
        keepdims=keepdims,
    )


def rms_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """Calculate the optionally masked root-mean-square value.

    ``axis`` and ``keepdims`` follow the corresponding NumPy reduction conventions.
    """
    mask = _resolve_mask(array, mask)
    n = np.sum(mask, axis=axis, keepdims=keepdims)
    return np.sqrt(np.nansum(mask * array**2, axis=axis, keepdims=keepdims) / n)


def p2v_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """Calculate the optionally masked peak-to-valley range.

    ``axis`` and ``keepdims`` follow the corresponding NumPy reduction conventions.
    """
    resolved_mask = _resolve_mask(array, mask)
    max_val = np.nanmax(
        np.where(resolved_mask.astype(bool), array, -np.inf),
        axis=axis,
        keepdims=keepdims,
    )
    min_val = np.nanmin(
        np.where(resolved_mask.astype(bool), array, np.inf),
        axis=axis,
        keepdims=keepdims,
    )
    return max_val - min_val
