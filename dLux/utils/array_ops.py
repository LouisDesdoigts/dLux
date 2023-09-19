import jax.numpy as np
from jax import Array

__all__ = [
    "pad_to",
    "crop_to",
    "resize",
    "downsample",
]

# TODO: Add convolve?


def pad_to(array: Array, npixels: int) -> Array:
    """
    Paraxially zero-pads the input array to the size determined by
    npixels. To ensure that no output arrays are non-paraxial even shaped
    arrays can only be padded to even shapes, and odd shaped arrays can only
    be padded to odd shapes. i.e. 2 -> 4 or 3 -> 5.

    Parameters
    ----------
    npixels : int
        The size to pad to the array to.

    Returns
    -------
    array : Array
        The padded array.
    """
    npixels_in = array.shape[-1]
    if npixels_in % 2 != npixels % 2:
        raise ValueError(
            "Only supports even -> even or odd -> odd padding."
            f"Input array has {npixels_in} pixels, and requested padding to "
            f"{npixels} pixels"
        )
    if npixels < npixels_in:
        raise ValueError(
            "npixels must be larger than the current array, "
            f"npixels_in = {npixels_in} < npixels = {npixels}"
        )

    return np.pad(array, (npixels - npixels_in) // 2)


def crop_to(array: Array, npixels: int) -> Array:
    """
    Paraxially crops the input array to the size determined by
    npixels. To ensure that no output arrays are non-paraxial even shaped
    arrays can only be cropped to even shapes, and odd shaped arrays can only
    be cropped to odd shapes. i.e. 4 -> 2 or 5 -> 3.

    Parameters
    ----------
    npixels : int
        The size to crop the array to.

    Returns
    -------
    array : Array
        The cropped array.
    """
    npixels_in = array.shape[-1]
    if npixels_in % 2 != npixels % 2:
        raise ValueError(
            "Only supports even -> even or odd -> odd cropping."
            f"Input array has {npixels_in} pixels, and requested cropping to "
            f"{npixels} pixels"
        )
    if npixels > npixels_in:
        raise ValueError(
            "npixels must be smaller than the current array, "
            f"npixels_in = {npixels_in} > npixels = {npixels}"
        )

    start, stop = (npixels_in - npixels) // 2, (npixels_in + npixels) // 2
    return array[start:stop, start:stop]


def resize(array: Array, npixels: int) -> Array:
    """
    Paraxially resizes the input array to the size determined by
    npixels. To ensure that no output arrays are non-paraxial even shaped
    arrays can only be resized to even shapes, and odd shaped arrays can only
    be resized to odd shapes. i.e. 4 -> 2 or 5 -> 3.

    Parameters
    ----------
    npixels : int
        The size to output the array.

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
        return pad_to(array, npixels)


def downsample(array: Array, n: int, mean=True) -> Array:
    """
    Downsamples the input array by n.

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
        raise ValueError(
            f"Input array has shape {array.shape}, which is not square"
        )
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
