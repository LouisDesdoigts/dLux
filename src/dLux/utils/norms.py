import jax.numpy as np
from jax import Array

__all__ = [
    "l1_norm",
    "l2_norm",
    "max_norm",
    "rms_norm",
]


def _resolve_mask(array, mask):
    """Resolved the mask for a given array"""
    if mask is None:
        return np.ones_like(array)
    return mask


def l1_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """
    Calculates the L1 norm of an array, optionally applying a mask.

    Parameters
    ----------
    array : Array
        The input array to calculate the L1 norm of.
    mask : Array | None = None
        An optional boolean mask to apply to the array before calculating the norm.
    axis : int | tuple[int, ...] | None = None
        Axis or axes along which the norm is computed. By default, all axes are used.
    keepdims : bool = False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    norm : float
        The L1 norm of the array, optionally masked.
    """
    return np.sum(
        _resolve_mask(array, mask) * np.abs(array),
        axis=axis,
        keepdims=keepdims,
    )


def l2_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """
    Calculates the L2 norm of an array, optionally applying a mask.

    Parameters
    ----------
    array : Array
        The input array to calculate the L2 norm of.
    mask : Array | None = None
        An optional boolean mask to apply to the array before calculating the norm.
    axis : int | tuple[int, ...] | None = None
        Axis or axes along which the norm is computed. By default, all axes are used.
    keepdims : bool = False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    norm : float
        The L2 norm of the array, optionally masked.
    """
    return np.sqrt(
        np.sum(_resolve_mask(array, mask) * array**2, axis=axis, keepdims=keepdims)
    )


def max_norm(
    array: Array,
    mask: Array | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float:
    """
    Calculates the maximum norm of an array, optionally applying a mask.

    Parameters
    ----------
    array : Array
        The input array to calculate the maximum norm of.
    mask : Array | None = None
        An optional boolean mask to apply to the array before calculating the norm.
    axis : int | tuple[int, ...] | None = None
        Axis or axes along which the norm is computed. By default, all axes are used.
    keepdims : bool = False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    norm : float
        The maximum norm of the array, optionally masked.
    """
    resolved_mask = _resolve_mask(array, mask)
    return np.max(
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
    """
    Calculates the root mean square (RMS) norm of an array, optionally applying a mask.

    Parameters
    ----------
    array : Array
        The input array to calculate the RMS norm of.
    mask : Array | None = None
        An optional boolean mask to apply to the array before calculating the norm.
    axis : int | tuple[int, ...] | None = None
        Axis or axes along which the norm is computed. By default, all axes are used.
    keepdims : bool = False
        If True, the reduced axes are left in the result as dimensions with size one.

    Returns
    -------
    norm : float
        The RMS norm of the array, optionally masked.

    """
    mask = _resolve_mask(array, mask)
    n = np.sum(mask, axis=axis, keepdims=keepdims)
    return np.sqrt(np.sum(mask * array**2, axis=axis, keepdims=keepdims) / n)
