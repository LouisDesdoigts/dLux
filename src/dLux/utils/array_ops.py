import jax.numpy as np
from jax import Array

__all__ = [
    "pad_to",
    "crop_to",
    "resize",
    "downsample",
]


def pad_to(array: Array, npixels: int, fill: float = 0.0) -> Array:
    """
    Paraxially zero-pads the input array to the shape (npixels, npixels). Due to the
    paraxial requirement, the input array must be square and even arrays can only
    be padded to even shapes, and odd shaped arrays can only be padded to odd shapes.

    Parameters
    ----------
    array : Array
        The input array to pad.
    npixels : int
        The size to pad to the array to.
    fill : float = 0.
        The value to fill the array with.

    Returns
    -------
    array : Array
        The padded array.
    """
    npixels_in = array.shape[-1]
    if npixels_in % 2 != npixels % 2:
        parity_in = "even" if npixels_in % 2 == 0 else "odd"
        parity_out = "even" if npixels % 2 == 0 else "odd"
        suggested_npixels = npixels + 1
        raise ValueError(
            f"Center-preserving padding requires parity consistency, "
            f"i.e. even -> even or odd -> odd: "
            f"input array ({npixels_in} pixels, {parity_in}) cannot be padded "
            f"to {npixels} pixels ({parity_out} parity). "
            f"Try npixels={suggested_npixels} instead."
        )
    if npixels < npixels_in:
        raise ValueError(
            f"Cannot pad to smaller size: input has {npixels_in} pixels, "
            f"target is {npixels} pixels. Padding requires npixels >= {npixels_in}."
        )

    return np.pad(array, (npixels - npixels_in) // 2, constant_values=fill)


def crop_to(array: Array, npixels: int) -> Array:
    """
    Paraxially crops the input array to the shape (npixels, npixels). Due to the
    paraxial requirement, the input array must be square and even arrays can only
    be cropped to even shapes, and odd shaped arrays can only be cropped to odd shapes.

    Parameters
    ----------
    array : Array
        The input array to crop.
    npixels : int
        The size to crop the array to.

    Returns
    -------
    array : Array
        The cropped array.
    """
    npixels_in = array.shape[-1]
    if npixels_in % 2 != npixels % 2:
        parity_in = "even" if npixels_in % 2 == 0 else "odd"
        parity_out = "even" if npixels % 2 == 0 else "odd"
        suggested_npixels = npixels - 1
        raise ValueError(
            f"Center-preserving cropping requires parity consistency, "
            f"i.e. even -> even or odd -> odd: "
            f"input array ({npixels_in} pixels, {parity_in}) cannot be cropped "
            f"to {npixels} pixels ({parity_out} parity). "
            f"Try npixels={suggested_npixels} instead."
        )
    if npixels > npixels_in:
        raise ValueError(
            f"Cannot crop to larger size: input has {npixels_in} pixels, "
            f"target is {npixels} pixels. Cropping requires npixels <= {npixels_in}."
        )

    start, stop = (npixels_in - npixels) // 2, (npixels_in + npixels) // 2
    return array[start:stop, start:stop]


def resize(array: Array, npixels: int, fill: float = 0.0) -> Array:
    """
    Resizes the input array to the shape (npixels, npixels), using either a pad or crop
    depending on the input array size. Due to the paraxial requirement, the input array
    must be square and even arrays can only be resized to even shapes, and odd shaped
    arrays can only be resized to odd shapes.

    Parameters
    ----------
    array : Array
        The input array to resize.
    npixels : int
        The size to output the array.
    fill : float = 0.
        The value to fill the array with if padding is required.

    Returns
    -------
    array : Array
        The resized array.
    """
    npixels_in = array.shape[-1]

    if npixels == npixels_in:
        return array
    elif npixels < npixels_in:
        return crop_to(array, npixels)
    else:
        return pad_to(array, npixels, fill)


def downsample(array: Array, n: int, mean: bool = True) -> Array:
    """
    Downsamples the input array by a factor of n, either by taking the mean or sum
    of the array.

    Parameters
    ----------
    array : Array
        The input array to downsample.
    n : int
        The factor by which to downsample the array.
    mean : bool = True
        Whether to downsample by taking the mean or sum of the array.

    Returns
    -------
    array : Array
        The downsampled array.
    """
    if array.shape[0] != array.shape[1]:
        raise ValueError(f"Input array has shape {array.shape}, which is not square")
    if array.shape[0] % n != 0:
        raise ValueError(
            f"Input array has {array.shape[0]} pixels, which is not divisible "
            f"by {n}"
        )

    method = np.mean if mean else np.sum
    size_in = array.shape[0]
    size_out = size_in // n

    # Downsample first dimension
    array = method(array.reshape((size_in * size_out, n)), 1)
    array = array.reshape(size_in, size_out).T

    # Downsample second dimension
    array = method(array.reshape((size_out * size_out, n)), 1)
    array = array.reshape(size_out, size_out).T
    return array
