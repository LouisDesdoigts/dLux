"""Place and extract compact arrays on a common sampling grid."""

import jax.numpy as np
from jax import Array, lax, vmap
import zodiax as zdx

import dLux.utils as dlu

from .array_ops import as_size
from .grid import Grid
from .specs import Spec

__all__ = ["PasteSpec", "paste"]


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

    Overlapping values are added rather than overwritten. Every compact array must
    lie completely inside the output; clipping and boundary padding are deliberately
    left to the caller.

    Parameters
    ----------
    arrays : Array
        Compact arrays with shape ``(n_stamps, ..., ny, nx)``. Any intermediate
        dimensions are retained as leading output dimensions.
    starts : Array
        Integer start indices with shape ``(n_stamps, 2)`` in physical ``(x, y)``
        order. Array storage therefore applies them to the final axes as ``(y, x)``.
    npixels : int or tuple[int, int]
        Output size in physical ``(x, y)`` order. A scalar produces a square output.
    method : str
        ``"scan"`` applies sequential dynamic updates with bounded working memory.
        ``"scatter"`` constructs flattened indices and exposes parallel placement at
        the cost of larger intermediate arrays.

    Returns
    -------
    Array
        The summed output with shape ``(..., output_ny, output_nx)``.

    Notes
    -----
    Both methods are JIT-compatible and numerically equivalent apart from ordinary
    floating-point accumulation-order differences. Their relative performance is
    hardware- and problem-dependent.
    """
    # Standardise the pasted arrays and placement indices
    arrays = np.asarray(arrays)
    starts = np.asarray(starts, dtype=int)
    sizes = as_size(npixels, 2, "npixels")

    if arrays.ndim < 3:
        raise ValueError("arrays must have shape (n, ..., ny, nx).")
    if starts.shape != (arrays.shape[0], 2):
        raise ValueError("starts must have shape (n, 2).")
    if method not in ("scan", "scatter"):
        raise ValueError("method must be either 'scan' or 'scatter'.")

    # Apply the requested memory or parallelism strategy
    paste_fn = _paste_scan if method == "scan" else _paste_scatter
    return paste_fn(arrays, starts, sizes)


class PasteSpec(Spec):
    """Map compact local arrays into a larger regularly sampled grid.

    The specification stores integer ``(x, y)`` starts and residual physical offsets
    for equally sized stamps. Construction is deliberately separate from evaluation:
    stamp sizes and integer placements are chosen before JIT compilation. Stamp
    values and residual offsets remain differentiable. Changing offsets does not
    recalculate the integer placements.

    ``stamp_shape`` holds physical (x, y) stamp counts; inherited ``shape`` gives
    the full output shape in array (y, x) order.

    Every stamp must fit completely inside the output. Spatial arrays follow
    ``(..., ny, nx)``; dimensions between the stamp and spatial axes are batches.

    This class is normally constructed by segmented aperture builders. It remains
    public so custom builders can reuse compact JIT-compatible placement without
    materialising full-size intermediate arrays.
    """

    n: tuple[int, int]
    stamp_shape: tuple[int, int]
    starts: Array
    offsets: Array
    d: Array

    def __init__(self, n, stamp_shape, starts, offsets, d, *, alias=None):
        """Initialise fixed placement geometry for compact stamps.

        Parameters
        ----------
        n : int or tuple[int, int]
            Full output size in physical ``(x, y)`` order.
        stamp_shape : int or tuple[int, int]
            Shared stamp size in physical ``(x, y)`` order.
        starts : Array
            Integer lower-corner indices with shape ``(n_stamps, 2)``.
        offsets : Array
            Stamp-grid centre minus requested centre, in SI units, with the same
            shape as ``starts``.
        d : Array
            Output pixel scales in SI units and physical ``(x, y)`` order.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        # Store fixed stamp sizes and one placement/offset pair per aperture.
        self.n = dlu.as_size(n, 2, "n")
        self.stamp_shape = dlu.as_size(stamp_shape, 2, "stamp_shape")
        self.starts = zdx.as_array(starts, dtype=int)
        self.offsets = zdx.as_array(offsets, dtype=float)
        self.d = dlu.as_axis(d, 2, "d")

        if self.starts.ndim != 2 or self.starts.shape[-1] != 2:
            raise ValueError("starts must have shape (n_stamps, 2).")
        if self.offsets.shape != self.starts.shape:
            raise ValueError("offsets must match the shape of starts.")

        # Reject placements whose complete stamp would extend beyond the output grid.
        ends = self.starts + np.asarray(self.stamp_shape)
        if np.any(self.starts < 0) or np.any(ends > np.asarray(self.n)):
            raise ValueError("Every stamp must lie within the output grid.")

        self.alias = alias

    @property
    def ndim(self) -> int:
        """Pasting places two-dimensional stamps into a two-dimensional output."""
        return 2

    @classmethod
    def from_grid(cls, grid, centers, extent):
        """Construct compact stamp placement outside JIT from one resolved grid.

        Stamp half-widths are rounded up to whole pixels with one spare pixel on
        each side. The complete stamps, including this margin, must fit inside the
        output grid. Fractional positions are retained in the local coordinates.

        Parameters
        ----------
        grid : Grid
            Unbatched two-dimensional ``xy`` grid with defined size and sampling.
            Expressions are resolved before calculating the placement geometry.
        centers : Array
            Stamp centres in SI units with shape ``(n_stamps, 2)`` in ``(x, y)`` order.
        extent : float or Array
            Local half-width in SI units, as a scalar or physical ``(x, y)`` pair.
        """
        if not isinstance(grid, Grid):
            raise TypeError("grid must be a Grid.")
        grid = grid.resolve()
        if grid.n is None or grid.d is None or grid.ndim != 2:
            raise ValueError("grid must define two-dimensional n and d values.")
        if grid.batch_shape:
            raise ValueError("PasteSpec requires an unbatched grid.")

        centers = zdx.as_array(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n_stamps, 2).")

        # Resolve the concrete grid geometry in canonical SI units
        spacing = grid.d_si
        center = grid.c_si
        origin = center - (np.asarray(grid.n) - 1) * spacing / 2

        # Fix the compact stamp shape before numerical evaluation
        extent = dlu.as_axis(extent, 2, "extent")
        half = tuple(int(value) + 1 for value in np.ceil(extent / spacing))
        shape = tuple(2 * value + 1 for value in half)

        # Resolve integer placements and exact residual physical offsets
        pixels = (centers - origin) / spacing
        anchors = np.rint(pixels).astype(int)
        starts = anchors - np.asarray(half)
        offsets = origin + anchors * spacing - centers
        return cls(grid.n, shape, starts, offsets, spacing)

    @property
    def coordinates(self) -> Array:
        """Return one local SI coordinate grid per stamp.

        The result has shape ``(n_stamps, 2, stamp_y, stamp_x)``. Values are output
        sample positions relative to each requested stamp centre.
        """
        coordinates = Grid(n=self.stamp_shape, d=self.d).coordinates
        return coordinates[None] + self.offsets[:, :, None, None]

    def paste(self, arrays, method="scan") -> Array:
        """Add compact stamp arrays into the common output grid.

        Parameters
        ----------
        arrays : Array
            Values with shape ``(n_stamps, ..., stamp_y, stamp_x)``.
        method : {"scan", "scatter"}
            ``"scan"`` limits intermediate memory; ``"scatter"`` uses per-pixel
            indices for parallel placement and may use more memory.

        Returns
        -------
        array : Array
            Summed values with shape ``(..., output_y, output_x)``. Overlapping
            stamps are added.
        """
        arrays = np.asarray(arrays)
        if arrays.shape[-2:] != self.stamp_shape[::-1]:
            raise ValueError("array spatial shape must match the PasteSpec stamps.")
        return dlu.paste(arrays, self.starts, self.n, method)

    def extract(self, array) -> Array:
        """Extract one compact stamp at every configured start position.

        ``array`` must have shape ``(..., output_y, output_x)``. The returned array
        has shape ``(n_stamps, ..., stamp_y, stamp_x)``.

        Extraction reads the combined values in overlapping regions; it does not
        recover the original overlapping stamps independently.
        """
        # Validate the full spatial shape and resolve slice geometry
        array = np.asarray(array)
        if array.shape[-2:] != self.n[::-1]:
            raise ValueError("array spatial shape must match the PasteSpec output.")

        leading = array.shape[:-2]
        slice_shape = leading + self.stamp_shape[::-1]

        # Extract every compact stamp in parallel
        def extract_stamp(start):
            index = (0,) * len(leading) + (start[1], start[0])
            return lax.dynamic_slice(array, index, slice_shape)

        return vmap(extract_stamp)(self.starts)
