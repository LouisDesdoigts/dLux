"""Regular Cartesian sampling specifications."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array, core

import dLux.utils as dlu

from .specs import UnitSpec
from .transforms import BaseCoordTransform

__all__ = ["Grid"]


class Grid(UnitSpec):
    """A complete or partially specified two-dimensional Cartesian grid.

    Axis parameters are ordered physically as ``(x, y)``. Array dimensions
    are ordered in reverse, so a specification with
    ``n=(nx, ny)`` produces coordinate arrays with shape ``(2, ny, nx)``.

    Parameters
    ----------
    n : int, tuple[int, int], or None
        Number of samples along each physical axis.
    d : ArrayLike or None
        Pixel scales in ``unit`` along each physical axis.
    c : ArrayLike or None
        Grid centers in ``unit`` along each physical axis.
    unit : str or None
        Physical or angular unit associated with ``d`` and ``c``.
    diam : ArrayLike or None
        Alternative physical extent used to calculate ``d = diam / n``.

    Examples
    --------
    Construct and manipulate unit-aware sampling grids:

    ```python
    import jax.numpy as np

    import dLux as dl


    # Define a square grid from its total field of view
    grid = dl.Grid(n=128, diam=6.5, unit="m")

    # Equivalently, define the grid from its pixel scale
    grid = dl.Grid(n=128, d=6.5 / 128, unit="m")

    # Inspect its sampling
    fov = grid.fov  # Total sampled width in metres
    coordinates = grid.coordinates  # Shape (2, 128, 128), in SI units

    # Define a non-square, anisotropic, and off-centre grid
    grid = dl.Grid(
        n=(128, 64),
        d=(10, 15),
        c=(50, -30),
        unit="um",
    )
    coordinates = grid.coordinates  # Shape (2, 64, 128)

    # Generate transformed coordinates without changing the grid
    rotation = dl.Affine(rotation=np.deg2rad(10))
    coordinates = grid.transformed(rotation)

    # Generate plot-ready bounds in another unit
    extent = grid.extent(unit="mm")

    # Change the sampling while preserving the field of view
    fine_grid = grid.oversample(2)
    original_grid = fine_grid.downsample(2)
    ```

    ``diam`` is a constructor convenience for total sampled extent. ``fov`` is the
    corresponding general property and may contain a different value for each axis.
    Undefined sampling leaves are retained until a field or numerical operation
    needs them. Coordinates always use SI units, independently of Zodiax's global
    unit settings; ``d``, ``c``, ``fov`` and default ``extent`` use stored units.
    Scalar inputs apply to both physical axes. Use Axis for one-dimensional
    sampling; a Grid always has two dimensions, including when incomplete.
    Resolving expressions retains the stored sample counts. Resolved spacings and
    centres must be scalar or have one value per physical axis; they do not expand
    ``n`` implicitly.
    Coordinate components retain singleton sample axes and any leading metadata
    batches. Sample counts are fixed topology; spacings and centres
    remain differentiable numerical leaves.
    """

    # Inherited fields
    unit: str | None = eqx.field(kw_only=True, default=None)

    n: tuple[int, int] | None = None
    d: Array | zdx.Expression | None = None
    c: Array | zdx.Expression | None = None

    def __init__(self, n=None, d=None, c=None, unit=None, diam=None, *, alias=None):
        """Store regular sampling with physical (x, y) and array (y, x) axes."""
        super().__init__(n=n, d=d, c=c, unit=unit, diam=diam, ndim=2, alias=alias)

    @property
    def ndim(self) -> int:
        """A Grid always describes two physical coordinate dimensions."""
        return 2

    def frequency(self, n=None) -> Grid:
        """Return the centred FFT dual grid in reciprocal SI units.

        ``n`` optionally selects working counts. Spatial centres do not change
        the frequency sampling. Even FFT lengths place its centre half a bin below
        zero. This constructs metadata, not coordinate or phase arrays.
        """
        # Compute reciprocal spacing and the even-length FFT centre in SI units.
        self = self.resolve()
        grid = self if n is None else self.resize(n)
        if grid.n is None:
            raise ValueError("n must be specified to calculate FFT sampling.")
        d = 1 / (np.asarray(grid.n) * grid.d_si)
        c = np.where(np.asarray(grid.n) % 2 == 0, -d / 2, 0.0)

        # Choose reciprocal units; angular grids have no frequency grid here.
        unit = None
        if grid.unit is not None:
            info = zdx.unit_info(grid.unit)
            if info.category != "cartesian" or info.power not in (-1, 1):
                raise ValueError(
                    "FFT frequency grids require length or reciprocal-length units."
                )
            unit = "1/m" if info.power == 1 else "m"
        return type(self)(grid.n, d, c, unit)

    @classmethod
    def from_axes(cls, axes, unit=None, *, alias=None) -> Grid:
        """Construct a regular grid from physical coordinate axes.

        Parameters
        ----------
        axes : tuple[Array, ...]
            Regularly sampled coordinate axes in physical-axis order and SI units.
            Concrete inputs are checked for regular spacing. Under JIT, regularity
            is the caller's contract. Each axis needs at least two samples, and
            leading batch dimensions are preserved.
        unit : str or None
            Unit used to store the recovered pixel scales and centers.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter-path aliases for the recovered grid.
        """
        # Validate coordinate vectors and recover physical-axis sample counts
        axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        if len(axes) != 2:
            raise ValueError("Grid requires two coordinate axes.")
        if any(axis.ndim == 0 or axis.shape[-1] < 2 for axis in axes):
            raise ValueError("Every coordinate axis must contain at least two samples.")
        n = tuple(axis.shape[-1] for axis in axes)

        # Recover pixel scales and centres with shared metadata broadcasting
        spacing = [
            (axis[..., -1] - axis[..., 0]) / (axis.shape[-1] - 1) for axis in axes
        ]
        centers = [(axis[..., -1] + axis[..., 0]) / 2 for axis in axes]

        # Check that every supplied vector matches the reconstructed regular sampling.
        for axis, step, centre in zip(axes, spacing, centers):
            offsets = np.arange(axis.shape[-1]) - (axis.shape[-1] - 1) / 2
            expected = centre[..., None] + offsets * step[..., None]
            magnitude = np.max(np.abs(axis), axis=-1, keepdims=True)
            tolerance = 8 * np.finfo(axis.dtype).eps * magnitude
            invalid = np.any(~np.isfinite(axis) | (np.abs(axis - expected) > tolerance))
            if not isinstance(invalid, core.Tracer) and bool(invalid):
                raise ValueError("Grid axes must be finite and regularly spaced.")
        d = np.stack(np.broadcast_arrays(*spacing), -1)
        c = np.stack(np.broadcast_arrays(*centers), -1)

        # Convert from SI coordinates into the requested output unit
        scale = dlu.unit_factor(unit)
        return cls(n=n, d=d / scale, c=c / scale, unit=unit, alias=alias)

    def transformed(self, transform=None) -> Array:
        """Return 2D SI coordinates after applying an optional transform.

        The result has shape ``(..., 2, ny, nx)``. Coordinate components use
        physical ``(x, y)`` order and sampled axes use NumPy ``(y, x)`` order.
        """
        if transform is None:
            return self.coordinates
        if not isinstance(transform, BaseCoordTransform):
            raise TypeError("transform must be a BaseCoordTransform or None.")
        return transform(self.coordinates)

    def extent(self, unit=None):
        """Return plot-ready grid edges in the declared output unit.

        Edges follow ``(xmin, xmax, ymin, ymax)``, suitable for the ``extent``
        argument of ``matplotlib.pyplot.imshow``. Passing ``unit`` converts from
        the grid's declared unit into the requested unit.
        """
        self = self.resolve()
        half_width = self.fov / 2
        center = np.zeros(2) if self.c is None else dlu.as_axis(self.c, 2, "c")
        extent = np.stack((center - half_width, center + half_width), axis=-1)
        extent = extent.reshape(extent.shape[:-2] + (4,))
        if unit is None:
            return extent
        return extent * dlu.unit_factor(self.unit) / dlu.unit_factor(unit)
