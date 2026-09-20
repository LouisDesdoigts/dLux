"""Manipulate sampled arrays while preserving their spatial conventions."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import jax.numpy as np
from jax import Array
import zodiax as zdx

import dLux.utils as dlu
from .specs import Spec, _check_parity

if TYPE_CHECKING:
    from .grid import Grid

__all__ = [
    "as_size",
    "pad_to",
    "crop_to",
    "resize",
    "downsample",
    "ResizeSpec",
    "FFTSpec",
]


def _spatial_counts(shape) -> tuple[int, int]:
    """Read physical (x, y) counts from final array axes (..., y, x)."""
    if len(shape) < 2:
        raise ValueError("Resizing requires two spatial array axes.")
    return tuple(shape[-2:][::-1])


def as_size(value, ndim=None, name="size"):
    """Return positive integer sizes as a static tuple."""
    if isinstance(value, Integral):
        values = (int(value),)
    elif isinstance(value, (tuple, list)) and value:
        values = tuple(value)
    else:
        raise TypeError(f"{name} must be an integer or non-empty tuple.")
    if not all(isinstance(item, Integral) for item in values):
        raise TypeError(f"{name} must contain integers.")
    if any(item < 1 for item in values):
        raise ValueError(f"{name} must contain positive values.")
    values = tuple(map(int, values))

    # Expand a shared scalar size only when a dimensionality was requested.
    if ndim is None or len(values) == ndim:
        return values
    if len(values) == 1:
        return values * ndim
    raise ValueError(f"{name} cannot be broadcast to {ndim} dimensions.")


def _size(npixels: int | tuple[int, ...]) -> tuple[int, ...]:
    """Cast a scalar or tuple to a physical-axis size tuple."""
    ndim = None if isinstance(npixels, (tuple, list)) else 2
    return as_size(npixels, ndim, "npixels")


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
    # Check that cropping preserves both the array bounds and its sample centre.
    sizes_in, sizes_out = _spatial_sizes(array, npixels)
    if any(value_out > value_in for value_in, value_out in zip(sizes_in, sizes_out)):
        raise ValueError(
            f"Cannot crop input size {sizes_in} to larger target {sizes_out}."
        )
    _check_parity(sizes_in, sizes_out, "cropping")

    # Translate physical-axis sizes into centred slices on the final array axes.
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
    # Match physical-axis bin factors to complete blocks on the sampled array axes.
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

    # Split each spatial axis into pixels and bins, then reduce only the bin axes.
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


class ResizeSpec(Spec):
    """Define two-dimensional output sampling by size or integer resize factors.

    Arrays have shape ``(..., ny, nx)``. Only the final two axes are resized;
    all preceding axes are batches. Sizes and factors follow physical ``(x, y)``
    order, and scalar inputs apply equally to both axes. Each padding or cropping
    stage must preserve odd/even parity on both axes so the centre stays fixed.

    Examples
    --------
    Apply factor-based or explicit resizing to an array:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Define padding and cropping around an operation
    resize = dl.ResizeSpec(pad=2, crop=2)
    array = np.ones((64, 64))

    # Apply the resizing operations
    padded = resize.pad_array(array)
    cropped = resize.crop_array(padded)
    output_size = resize.output_size(array.shape)

    # Alternatively, resize directly to an explicit output size
    resize = dl.ResizeSpec(n=96)
    resized = resize.resize(array)
    ```

    Parameters
    ----------
    n : int, tuple[int, int], or None
        Explicit output sizes in physical-axis order. Mutually exclusive with
        non-unit ``pad`` or ``crop``.
    pad, crop : int or tuple[int, int]
        Integer factors applied before and after an operation.
    """

    n: tuple[int, int] | None = None
    pad: tuple[int, int] = (1, 1)
    crop: tuple[int, int] = (1, 1)

    def __init__(self, n=None, pad=1, crop=1, *, alias=None):
        """Initialise an explicit or factor-based resize specification.

        Parameters
        ----------
        n : int, tuple[int, int], or None
            Explicit physical-axis output sizes, mutually exclusive with non-unit
            ``pad`` or ``crop``.
        pad, crop : int or tuple[int, int]
            Positive integer factors applied before and after an operation.
        alias : Mapping or sequence of pairs, optional
            Optional Zodiax parameter-path aliases.
        """
        # Store one value per physical axis; scalar inputs apply to both.
        self.n = None if n is None else as_size(n, 2, "n")
        self.pad = as_size(pad, 2, "pad")
        self.crop = as_size(crop, 2, "crop")
        if self.n is not None and (self.pad != (1, 1) or self.crop != (1, 1)):
            raise ValueError("Specify either n or pad/crop factors, not both.")
        self.alias = alias

    @property
    def ndim(self) -> int:
        """Resize requests always describe two spatial axes."""
        return 2

    @property
    def explicit(self) -> bool:
        """Return whether ``n`` defines an absolute output size.

        When false, output sampling is determined from the input shape and the
        configured integer pad and crop factors.
        """
        return self.n is not None

    def _pad_counts(self, n_in: tuple[int, int]) -> tuple[int, int]:
        """Calculate counts for the first stage while keeping the centre fixed."""
        if self.n is None:
            n_out = tuple(size * factor for size, factor in zip(n_in, self.pad))
        else:
            n_out = self.n
        _check_parity(n_in, n_out, "resizing")
        return n_out

    def _crop_counts(self, n_in: tuple[int, int]) -> tuple[int, int]:
        """Calculate counts for the final stage while keeping the centre fixed."""
        if self.n is None:
            n_out = tuple(size // factor for size, factor in zip(n_in, self.crop))
        else:
            n_out = self.n
        if any(size < 1 for size in n_out):
            raise ValueError("crop factors may not reduce an output count below one.")
        _check_parity(n_in, n_out, "cropping")
        return n_out

    def counts(self, n_in: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return working and output counts for input physical sizes ``(nx, ny)``.

        An explicit ``n`` defines both the working and returned counts. Otherwise,
        input counts are multiplied by ``pad`` and the resulting working counts are
        integer-divided by ``crop``. The established floor-division behaviour is
        preserved when the factors do not divide the working counts exactly.
        Both stages must retain the input parity, including intermediate working
        counts when the final counts happen to match the original input.
        """
        n_in = as_size(n_in, name="n_in")
        if len(n_in) != 2:
            raise ValueError("n_in must contain two physical-axis counts (nx, ny).")
        n_work = self._pad_counts(n_in)
        return n_work, self._crop_counts(n_work)

    def output_size(self, shape) -> tuple[int, int]:
        """Return final physical sizes ``(nx, ny)`` for array shape ``(..., ny, nx)``.

        Apply both padding and cropping factors, or use the explicit target ``n``.
        Leading batch axes do not affect the result.
        """
        return self.counts(_spatial_counts(shape))[1]

    def crop_size(self, shape) -> tuple[int, int]:
        """Return physical-axis sizes after applying only crop factors.

        ``shape`` follows array order ``(..., ny, nx)``. The input may already be
        padded; only its current final two sizes and the crop factors are used.
        """
        return self._crop_counts(_spatial_counts(shape))

    def pad_array(self, array: Array, fill: float = 0.0) -> Array:
        """Centrally pad the final two spatial axes.

        Explicit ``n`` sets the target size; otherwise each final axis is enlarged by
        its pad factor. ``fill`` supplies values outside the original array and
        leading axes are preserved.
        """
        target = self._pad_counts(_spatial_counts(array.shape))
        return pad_to(array, target, fill)

    def crop_array(self, array: Array) -> Array:
        """Centrally crop the final two spatial axes to the configured output size.

        Leading array axes are preserved and output spatial axes follow NumPy order.
        """
        return crop_to(array, self.crop_size(array.shape))

    def crop_axes(self, axes: tuple[Array, Array]) -> tuple[Array, Array]:
        """Centrally crop the two coordinate vectors ``(x, y)``.

        ``axes`` and the returned tuple follow physical-axis order. Each vector is
        cropped consistently with :meth:`crop_array`.
        """
        if len(axes) != 2:
            raise ValueError("axes must contain two coordinate vectors (x, y).")
        n_in = tuple(axis.shape[-1] for axis in axes)
        sizes = self._crop_counts(n_in)
        return tuple(crop_to(axis, (n,)) for axis, n in zip(axes, sizes))

    def resize(self, array: Array, fill: float = 0.0) -> Array:
        """Centrally pad or crop the final two axes to the represented output size.

        ``fill`` is used only for padded values. Leading axes are preserved.
        """
        return resize(array, self.output_size(array.shape), fill)


class FFTSpec(ResizeSpec):
    """Request native Fourier sampling through the canonical resize contract.

    ``n`` requests an explicit FFT count. Otherwise, ``pad`` determines the FFT
    working count and ``crop`` determines the returned central count. ``c=None``
    keeps the native centre and therefore requires no shift ramp; an explicit
    centre is interpreted in the natural canonical coordinate of the consuming
    propagator: radians for normalised Fraunhofer, metres for direct LCT, or
    reciprocal metres for spectral LCT.
    On an even-sized axis the native centre is minus half the output spacing,
    so ``c=None`` and ``c=0`` request different coordinates.

    Counts follow physical ``(x, y)`` order. Scalar values apply to both spatial
    axes, and centre arrays may carry leading batch axes. The centre is a physical
    coordinate, not an integer pixel offset. Padding and cropping preserve parity
    and sample positions; the propagator evaluates a shifted Fourier window using
    a phase ramp when a centre is supplied.

    Parameters
    ----------
    n : int, tuple[int, int], or None
        Explicit FFT working and output counts. Mutually exclusive with non-unit
        ``pad`` or ``crop``.
    pad, crop : int or tuple[int, int]
        Integer working-size and output-crop factors.
    c : ArrayLike, Expression, or None
        Optional final output centre in the consuming propagator's natural SI
        coordinate.
    """

    # Inherited fields
    n: tuple[int, int] | None = None
    pad: tuple[int, int] = (1, 1)
    crop: tuple[int, int] = (1, 1)

    c: Array | zdx.Expression | None = None

    def __init__(self, n=None, pad=1, crop=1, c=None, *, alias=None):
        """Initialise FFT counts and an optional physical output centre."""
        super().__init__(n=n, pad=pad, crop=crop, alias=alias)
        self.c = zdx.as_array(c, dtype=float)

    def working_grid(self, source: Grid) -> Grid:
        """Return padded input sampling, rejecting any loss of input samples."""
        n_work, _ = self.counts(source.n)
        if any(work < size for work, size in zip(n_work, source.n)):
            raise ValueError("FFT working counts may not crop the input grid.")
        return source.resize_samples(n_work)

    def output_grid(self, native: Grid, n_in) -> Grid:
        """Crop native Fourier sampling and apply an optional requested centre.

        ``native`` uses the consuming propagator's SI output unit. ``n_in`` is
        the original input count tuple used to resolve pad/crop factors.
        This describes the output coordinates; the propagator supplies the phase
        ramp that evaluates the field there. Array padding/cropping ignores ``c``.
        """
        self = self.resolve()
        _, n_out = self.counts(n_in)
        output = native.resize_samples(n_out)
        if self.c is not None:
            output = output.set(c=dlu.as_axis(self.c, native.ndim, "FFTSpec.c"))
        return output
