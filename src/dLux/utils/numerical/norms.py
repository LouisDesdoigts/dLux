"""Measure and normalise sampled arrays with optional weights."""

from operator import index

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

__all__ = ["Norm", "norm", "normalise"]

_MODES = ("sum", "mean", "rms", "l1", "l2", "max", "peak", "p2v")


def norm(array, mode="rms", weights=None, axis=None, keepdims=False):
    """Measure an array, optionally reducing selected axes with weights.

    sum/mean retain sign or phase; l1/l2/rms measure magnitude. max measures
    absolute peak, peak measures signed peak, and p2v measures signed range.
    Norm modes ignore NaNs; sum, mean and signed peak propagate them. Weights
    must be finite and non-negative. Invalid numerical inputs are not checked.
    """
    # Broadcast real weights onto the samples and check the selected reduction.
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}.")
    array = np.asarray(array)
    weights = np.ones(array.shape) if weights is None else np.asarray(weights)
    if np.iscomplexobj(weights):
        raise TypeError("weights must be real.")
    weights = np.broadcast_to(weights, array.shape)
    if mode in ("peak", "p2v") and np.iscomplexobj(array):
        raise TypeError(f"{mode} requires real values.")

    # Sums and means retain sign or phase; magnitude norms are handled below.
    if mode in ("sum", "mean"):
        total = np.sum(array * weights, axis=axis, keepdims=keepdims)
        if mode == "mean":
            total /= np.sum(weights, axis=axis, keepdims=keepdims)
        return total
    if mode == "l1":
        return np.nansum(weights * np.abs(array), axis=axis, keepdims=keepdims)
    if mode in ("l2", "rms"):
        squares = np.nansum(weights * np.abs(array) ** 2, axis=axis, keepdims=keepdims)
        if mode == "rms":
            squares /= np.sum(weights, axis=axis, keepdims=keepdims)
        return np.sqrt(squares)

    # Peak reductions use weights as a support mask, not as amplitude factors.
    if mode == "max":
        supported = np.where(weights.astype(bool), np.abs(array), -np.inf)
        return np.nanmax(supported, axis=axis, keepdims=keepdims)

    if mode == "peak":
        supported = np.where(weights.astype(bool), array, -np.inf)
        return np.max(supported, axis=axis, keepdims=keepdims)

    # Peak-to-valley measures the signed range over the support.
    maximum = np.nanmax(
        np.where(weights.astype(bool), array, -np.inf), axis=axis, keepdims=keepdims
    )
    minimum = np.nanmin(
        np.where(weights.astype(bool), array, np.inf), axis=axis, keepdims=keepdims
    )
    return maximum - minimum


def normalise(array, mode="rms", scale=1.0, weights=None, axis=None):
    """Scale an array to a weighted sum, mean, RMS, or supported array norm.

    Parameters
    ----------
    array : Array
        Real or complex numerical values. Complex RMS and L2 use absolute squares;
        ``p2v`` requires real values.
    mode : str
        One of sum, mean, rms, l1, l2, max, peak, or p2v.
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
        zero/nonfinite measures are caller errors, both eagerly and under JIT.
        No runtime checks or tracer-dependent behaviour are added.
        Normalisation is differentiable for a finite non-zero measure.
    """
    array = np.asarray(array)
    measure = norm(array, mode, weights, axis, keepdims=True)

    output = array * np.asarray(scale) / measure
    if output.shape != array.shape:
        raise ValueError("scale must broadcast without changing the array shape.")
    return output


class Norm(zdx.Expression):
    """A weighted measurement of an array, rather than a normalised array.

    ``sum`` and ``mean`` retain sign or phase; ``l1`` sums absolute values, while
    ``l2`` and ``rms`` use absolute squares. ``max`` and ``p2v`` use nonzero weights
    as support, without weighting their extrema. Peak-to-valley requires real
    values. Weights broadcast to the input shape before reduction.

    The norm modes ignore NaNs in the numerator, retaining their weights in the
    RMS denominator. Sum, mean and peak propagate NaNs. Empty support and zero measures
    are returned as calculated; a non-zero measure is required for normalisation.

    Examples
    --------
    ```python
    import dLux.utils as dlu

    error = dlu.Norm([1.0, -2.0, 3.0], mode="rms")
    rms = error.resolve()
    ```
    """

    x: Array | zdx.Expression
    weights: Array | zdx.Expression | None = None
    mode: str = eqx.field(static=True)
    axis: tuple[int, ...] | None = eqx.field(default=None, static=True)
    keepdims: bool = eqx.field(default=False, static=True)

    def __init__(
        self, x, mode="rms", weights=None, axis=None, keepdims=False, *, alias=None
    ):
        """Store the array and the reduction to perform.

        Parameters
        ----------
        x : Array or zodiax.Expression
            Real or complex values to measure. Peak-to-valley uses real values.
        mode : str
            One of sum, mean, rms (default), l1, l2, max, peak, or p2v.
        weights : Array, zodiax.Expression, or None
            Non-negative real weights broadcastable to x. None weights all samples
            equally. Weights and values remain differentiable numerical leaves.
        axis : int, tuple[int, ...], or None
            Axes to reduce; None reduces every axis. Other axes remain independent.
        keepdims : bool
            Retain reduced axes with size one, default False.
        alias : mapping or None
            Optional user-defined parameter aliases.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}.")
        if axis is not None:
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            axis = tuple(index(value) for value in axes)

        self.x = zdx.as_array(x)
        self.weights = zdx.as_array(weights)
        self.mode = mode
        self.axis = axis
        self.keepdims = bool(keepdims)
        self.alias = alias

    def evaluate(self, **context):
        """Resolve stored operands and return their measurement as a JAX array."""
        x, weights = zdx.resolve((self.x, self.weights), **context)
        return norm(x, self.mode, weights, self.axis, self.keepdims)
