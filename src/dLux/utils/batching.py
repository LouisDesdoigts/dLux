"""Array broadcasting and explicit component/batch dimensions."""

from math import prod
from operator import index

import jax.numpy as np

__all__ = ["as_axis", "broadcast_batch", "broadcast_batches", "flatten_batch"]


def as_axis(value, ndim=None, name="axis"):
    """Return float values with an explicit trailing component axis, or None.

    A scalar repeats over ``ndim`` components. Arrays already ending in ``ndim``
    are retained; a final singleton dimension expands to ``ndim``. All leading
    dimensions remain batches. Without ndim, infer the existing component count
    (one for a scalar). ``name`` labels incompatible shapes in error messages.

    For example, as_axis(2, 2) gives [2, 2], and shape (3, 1) becomes (3, 2).
    A batch of scalar values therefore needs a trailing singleton axis.
    """
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    size = 1 if value.ndim == 0 else value.shape[-1]
    ndim = size if ndim is None else max(int(ndim), 1)
    if value.ndim == 0:
        return np.broadcast_to(value, (ndim,))
    if value.shape[-1] == ndim:
        return value
    if value.shape[-1] == 1:
        return np.broadcast_to(value, value.shape[:-1] + (ndim,))
    raise ValueError(f"{name} must be scalar or have one value per axis.")


def broadcast_batch(array, batch_shape, ndim):
    """Broadcast to batch_shape, preserving the final ndim dimensions.

    ndim counts dimensions of one value: 0 for a scalar, 1 for a vector,
    2 for an image or matrix. It never assigns physical meaning to a batch.
    For example, an image (ny, nx) and batch_shape=(3,) produce (3, ny, nx).
    """
    array = np.asarray(array)
    ndim = index(ndim)
    if not 0 <= ndim <= array.ndim:
        raise ValueError("ndim must count existing trailing dimensions.")
    value_shape = array.shape[array.ndim - ndim :]
    return np.broadcast_to(array, tuple(batch_shape) + value_shape)


def broadcast_batches(*arrays, ndims):
    """Broadcast leading batches together, preserving each array's value shape.

    ndims is one trailing-dimension count per array, or an integer shared by all.
    For example, images (3, 1, ny, nx) and matrices (5, 2, 2), with ndims=(2, 2),
    acquire the common batch (3, 5). Use explicit singleton axes when batches
    should vary independently; matching batch axes are paired.
    """
    # Expand a shared trailing-axis count into one count per input array.
    arrays = tuple(np.asarray(array) for array in arrays)
    if isinstance(ndims, int):
        ndims = (ndims,) * len(arrays)
    if len(ndims) != len(arrays):
        raise ValueError("Supply one trailing-dimension count per array.")

    # Find a common leading batch without changing any intrinsic value axes.
    shapes = []
    for array, ndim in zip(arrays, ndims):
        ndim = index(ndim)
        if not 0 <= ndim <= array.ndim:
            raise ValueError("ndim must count existing trailing dimensions.")
        shapes.append(array.shape[: array.ndim - ndim])
    batch_shape = np.broadcast_shapes(*shapes)
    return tuple(
        broadcast_batch(array, batch_shape, ndim) for array, ndim in zip(arrays, ndims)
    )


def flatten_batch(array, ndim):
    """Collapse leading batches to one axis, retaining the final ndim dimensions.

    (2, 3, ny, nx) becomes (6, ny, nx) with ndim=2. An unbatched image becomes
    (1, ny, nx), so it can use the same vmap call. Restore the original batch
    shape with result.reshape(batch_shape + result.shape[1:]).
    """
    array = np.asarray(array)
    ndim = index(ndim)
    if not 0 <= ndim <= array.ndim:
        raise ValueError("ndim must count existing trailing dimensions.")
    split = array.ndim - ndim
    batch_size = prod(array.shape[:split])
    return array.reshape((batch_size,) + array.shape[split:])
