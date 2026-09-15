"""Calculate support-aware norms for sampled arrays."""

import jax.numpy as np
from operator import index
from jax import Array, core

__all__ = ["l1_norm", "l2_norm", "max_norm", "rms_norm", "p2v_norm", "normalise"]


def normalise(array, mode="rms", scale=1.0, weights=None, axis=None):
    """Scale an array to a weighted sum, mean, RMS, or supported array norm.

    Parameters
    ----------
    array : Array
        Real or complex numerical values. Complex RMS and L2 use absolute squares;
        ``p2v`` requires real values.
    mode : str
        One of ``sum``, ``mean``, ``rms``, ``l1``, ``l2``, ``max``, or ``p2v``.
        Sum and mean retain sign/phase; L1 sums absolute values. Max and peak-to-
        valley treat positive weights as support rather than amplitude weights.
    scale : Array or float
        Target measure, default one. Must broadcast without changing array shape.
    weights : real Array or None
        Finite non-negative weights broadcastable to the input, or a support mask.
        Weights define the measure; they do not clip or multiply the returned array.
    axis : int, tuple[int, ...], or None
        Unique reduction axes. None measures the whole array; reduced axes retain
        size one during division, preserving every leading batch independently.

    Returns
    -------
    Array
        ``array * scale / measure``, with its original shape. Invalid weights and
        zero/nonfinite measures raise during eager evaluation. Under JIT these
        remain caller requirements; no runtime validation callbacks are added.
        Normalisation is differentiable for a finite non-zero measure.
    """
    if mode not in ("sum", "mean", "rms", "l1", "l2", "max", "p2v"):
        raise ValueError("mode must be sum, mean, rms, l1, l2, max, or p2v.")
    array = np.asarray(array)
    if axis is not None:
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        axis = tuple(index(value) for value in axes)
        if not axis or any(
            value < -array.ndim or value >= array.ndim for value in axis
        ):
            raise ValueError("axis must identify at least one valid array axis.")
        axis = tuple(value % array.ndim for value in axis)
        if len(set(axis)) != len(axis):
            raise ValueError("axis entries must identify unique dimensions.")
    weights = np.ones(array.shape) if weights is None else np.asarray(weights)
    if np.iscomplexobj(weights):
        raise TypeError("weights must be real.")
    weights = np.broadcast_to(weights, array.shape)
    # Check concrete inputs without adding callbacks to the compiled calculation.
    if not isinstance(weights, core.Tracer):
        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Normalisation weights must be finite and non-negative.")

    # Measure signed values or delegate to the shared array norm kernels.
    if mode in ("sum", "mean"):
        measure = np.sum(array * weights, axis=axis, keepdims=True)
    else:
        if mode == "p2v" and np.iscomplexobj(array):
            raise TypeError("p2v normalisation requires real values.")
        if not isinstance(array, core.Tracer) and not isinstance(weights, core.Tracer):
            if np.any((weights > 0) & ~np.isfinite(array)):
                raise ValueError("Values on the normalisation support must be finite.")
        norms = {
            "l1": l1_norm,
            "l2": l2_norm,
            "rms": rms_norm,
            "max": max_norm,
            "p2v": p2v_norm,
        }
        measure = norms[mode](array, weights, axis, keepdims=True)
    if mode == "mean":
        measure /= np.sum(weights, axis=axis, keepdims=True)
    if not isinstance(measure, core.Tracer):
        if np.any(~np.isfinite(measure)) or np.any(np.abs(measure) == 0):
            raise ValueError("Normalisation measure must be finite and non-zero.")

    output = array * np.asarray(scale) / measure
    if output.shape != array.shape:
        raise ValueError("scale must broadcast without changing the array shape.")
    return output


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
        np.nansum(
            _resolve_mask(array, mask) * np.abs(array) ** 2,
            axis=axis,
            keepdims=keepdims,
        )
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
    return np.sqrt(
        np.nansum(mask * np.abs(array) ** 2, axis=axis, keepdims=keepdims) / n
    )


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
