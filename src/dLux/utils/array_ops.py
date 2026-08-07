"""Manipulate sampled arrays while preserving their spatial conventions."""

import jax.numpy as np
from jax import Array, lax

import dLux.utils as dlu

__all__ = ["pad_to", "crop_to", "resize", "downsample", "paste"]


def _size(npixels: int | tuple[int, ...]) -> tuple[int, ...]:
    """Cast a scalar or tuple to a physical-axis size tuple."""
    ndim = None if isinstance(npixels, (tuple, list)) else 2
    return dlu.as_size(npixels, ndim, "npixels")


def _spatial_sizes(array: Array, npixels) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return input and target sizes in physical-axis order."""
    sizes_out = _size(npixels)
    ndim = len(sizes_out)
    if array.ndim < ndim:
        raise ValueError(
            f"Input with {array.ndim} dimensions cannot have {ndim} spatial axes."
        )
    sizes_in = array.shape[-ndim:][::-1]
    return sizes_in, sizes_out


def _check_parity(sizes_in, sizes_out, operation):
    """Validate centre-preserving parity across every spatial axis."""
    for axis, (size_in, size_out) in enumerate(zip(sizes_in, sizes_out)):
        if size_in % 2 != size_out % 2:
            raise ValueError(
                f"Center-preserving {operation} requires matching parity on axis "
                f"{axis}: {size_in} cannot be changed to {size_out}."
            )


def pad_to(array: Array, npixels: int | tuple[int, ...], fill: float = 0.0) -> Array:
    """Centrally pad the final spatial axes to a target size.

    Parameters
    ----------
    array : Array
        The input array to pad.
    npixels : int or tuple[int, ...]
        Target size in physical-axis order. A scalar preserves the historical
        behaviour of padding the final two axes to a square.
    fill : float = 0.
        The value to fill the array with.

    Returns
    -------
    array : Array
        The padded array.
    """
    sizes_in, sizes_out = _spatial_sizes(array, npixels)
    if any(value_out < value_in for value_in, value_out in zip(sizes_in, sizes_out)):
        raise ValueError(
            f"Cannot pad input size {sizes_in} to smaller target {sizes_out}."
        )
    _check_parity(sizes_in, sizes_out, "padding")
    pads = tuple(
        (size_out - size_in) // 2 for size_in, size_out in zip(sizes_in, sizes_out)
    )
    width = (*[(0, 0)] * (array.ndim - len(pads)), *((pad, pad) for pad in pads[::-1]))
    return np.pad(array, width, constant_values=fill)


def crop_to(array: Array, npixels: int | tuple[int, ...]) -> Array:
    """Centrally crop the final spatial axes to a target size.

    Parameters
    ----------
    array : Array
        The input array to crop.
    npixels : int or tuple[int, ...]
        Target size in physical-axis order. A scalar preserves the historical
        behaviour of cropping the final two axes to a square.

    Returns
    -------
    array : Array
        The cropped array.
    """
    sizes_in, sizes_out = _spatial_sizes(array, npixels)
    if any(value_out > value_in for value_in, value_out in zip(sizes_in, sizes_out)):
        raise ValueError(
            f"Cannot crop input size {sizes_in} to larger target {sizes_out}."
        )
    _check_parity(sizes_in, sizes_out, "cropping")
    starts = tuple(
        (size_in - size_out) // 2 for size_in, size_out in zip(sizes_in, sizes_out)
    )
    leading = (slice(None),) * (array.ndim - len(starts))
    spatial = tuple(
        slice(start, start + size) for start, size in zip(starts[::-1], sizes_out[::-1])
    )
    return array[leading + spatial]


def resize(array: Array, npixels: int | tuple[int, ...], fill: float = 0.0) -> Array:
    """Centrally resize the final spatial axes using cropping and padding.

    Parameters
    ----------
    array : Array
        The input array to resize.
    npixels : int or tuple[int, ...]
        Target size in physical-axis order.
    fill : float = 0.
        The value to fill the array with if padding is required.

    Returns
    -------
    array : Array
        The resized array.
    """
    sizes_in, sizes_out = _spatial_sizes(array, npixels)
    _check_parity(sizes_in, sizes_out, "resizing")
    intermediate = tuple(min(a, b) for a, b in zip(sizes_in, sizes_out))
    if intermediate != sizes_in:
        array = crop_to(array, intermediate)
    if intermediate != sizes_out:
        array = pad_to(array, sizes_out, fill)
    return array


def downsample(array: Array, n: int | tuple[int, ...], mean: bool = True) -> Array:
    """Downsample the final spatial axes by integer factors.

    Parameters
    ----------
    array : Array
        The input array to downsample.
    n : int or tuple[int, ...]
        Factors in physical-axis order. A scalar applies to the final two axes.
    mean : bool = True
        Whether to downsample by taking the mean or sum of the array.

    Returns
    -------
    array : Array
        The downsampled array.
    """
    factors = _size(n)
    ndim = len(factors)
    if array.ndim < ndim:
        raise ValueError(
            f"Input with {array.ndim} dimensions cannot have {ndim} spatial axes."
        )
    spatial_shape = array.shape[-ndim:]
    array_factors = factors[::-1]
    if any(size % factor for size, factor in zip(spatial_shape, array_factors)):
        raise ValueError(
            f"Spatial shape {spatial_shape[::-1]} is not divisible by {factors}."
        )
    method = np.mean if mean else np.sum
    leading = array.shape[:-ndim]
    shape = tuple(
        value
        for size, factor in zip(spatial_shape, array_factors)
        for value in (size // factor, factor)
    )
    array = array.reshape(leading + shape)
    axes = tuple(len(leading) + 2 * axis + 1 for axis in range(ndim))
    return method(array, axis=axes)


def _paste_scan(arrays, starts, sizes):
    """Paste compact arrays sequentially with bounded working memory."""
    # Initialise the common output and fixed slice geometry
    nx, ny = sizes
    height, width = arrays.shape[-2:]
    leading = arrays.shape[1:-2]
    slice_shape = leading + (height, width)
    output = np.zeros(leading + (ny, nx), dtype=arrays.dtype)

    # Add each contiguous stamp without constructing per-pixel indices
    def add_stamp(output, data):
        array, start = data
        index = (0,) * len(leading) + (start[1], start[0])
        current = lax.dynamic_slice(output, index, slice_shape)
        output = lax.dynamic_update_slice(output, current + array, index)
        return output, None

    return lax.scan(add_stamp, output, (arrays, starts))[0]


def _paste_scatter(arrays, starts, sizes):
    """Paste compact arrays in parallel using flattened scatter indices."""
    # Generate flattened output indices for every stamp pixel
    nx, ny = sizes
    height, width = arrays.shape[-2:]
    x = starts[:, 0, None, None] + np.arange(width)[None, None, :]
    y = starts[:, 1, None, None] + np.arange(height)[None, :, None]
    indices = y * nx + x

    # Align the stamp axis behind any intermediate array dimensions
    leading = arrays.shape[1:-2]
    values = np.moveaxis(arrays, 0, len(leading)).reshape(leading + (-1,))

    # Add all stamp pixels into the flattened output in parallel
    output = np.zeros(leading + (nx * ny,), dtype=arrays.dtype)
    output = output.at[..., indices.reshape(-1)].add(values)
    return output.reshape(leading + (ny, nx))


def paste(
    arrays: Array,
    starts: Array,
    npixels: int | tuple[int, int],
    method: str = "scan",
) -> Array:
    """Add equally sized arrays into a common two-dimensional output.

    ``arrays`` has shape ``(n, ..., ny, nx)`` and ``starts`` has shape ``(n, 2)``
    in physical ``(x, y)`` pixel order. Intermediate axes are preserved. Every
    compact array must lie completely within the output. ``method="scan"`` limits
    working memory through sequential updates; ``method="scatter"`` constructs
    flattened indices to expose parallel placement.
    """
    # Standardise the pasted arrays and placement indices
    arrays = np.asarray(arrays)
    starts = np.asarray(starts, dtype=int)
    sizes = dlu.as_size(npixels, 2, "npixels")

    if arrays.ndim < 3:
        raise ValueError("arrays must have shape (n, ..., ny, nx).")
    if starts.shape != (arrays.shape[0], 2):
        raise ValueError("starts must have shape (n, 2).")
    if method not in ("scan", "scatter"):
        raise ValueError("method must be either 'scan' or 'scatter'.")

    # Apply the requested memory or parallelism strategy
    paste_fn = _paste_scan if method == "scan" else _paste_scatter
    return paste_fn(arrays, starts, sizes)
