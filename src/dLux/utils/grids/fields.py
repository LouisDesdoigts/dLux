"""Real and complex arrays coupled to coordinate sampling."""

from __future__ import annotations

import operator

import jax.numpy as np
import zodiax as zdx
from jax import Array, vmap
from jax.scipy.signal import convolve

import dLux.utils as dlu

from . import array_ops, transforms
from .specs import UnitSpec

__all__ = ["Field"]

_ops = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


class Field(zdx.Module):
    """Sampled real or complex arrays on a coordinate specification.

    A field couples an array to a `UnitSpec`. The final sampled axes are in
    NumPy order, while grid values use physical-axis order; all preceding axes are
    preserved as vectorisation axes. Field operations are immutable and return a new
    object carrying sampling consistent with the transformed data.

    Examples
    --------
    ```python
    import dLux as dl
    import jax.numpy as np

    # Sampled values retain their own coordinate grid.
    grid = dl.Grid(8, diam=1.0)
    field = dl.Field(dl.Gaussian(std=0.2, grid=grid).resolve(), grid)
    print(field.resample(dl.Grid(12, diam=1.0)))
    values = field.sample((np.array([0.0, 0.1]), np.array([0.0, 0.1])))
    ```
    """

    array: Array
    grid: UnitSpec

    def __init__(self, array: Array, grid: UnitSpec, *, alias=None):
        """Store an array and its coordinate sampling.

        Parameters
        ----------
        array : Array
            Real or complex samples with shape (..., *grid.shape). Leading axes are
            ordinary batches. The input numerical dtype is retained.
        grid : UnitSpec
            Coordinate sampling matched to the array's final grid.ndim axes.
        alias : mapping or sequence of pairs, optional
            Explicit Zodiax parameter-path aliases; None defines no aliases.
        """
        array = zdx.as_array(array)
        if not isinstance(grid, UnitSpec):
            raise TypeError("grid must be a sampling UnitSpec.")
        if array.ndim < grid.ndim:
            raise ValueError("array has fewer axes than its sampling UnitSpec.")
        self.array = array
        self.grid = grid.resolve().match_shape(array.shape[-grid.ndim :])
        self.alias = alias

    @property
    def real(self) -> Array:
        """Real component, retaining the array shape."""
        return np.real(self.array)

    @property
    def imaginary(self) -> Array:
        """Imaginary component; zero for real-valued arrays."""
        return np.imag(self.array)

    @property
    def imag(self) -> Array:
        """Alias the imaginary component."""
        return self.imaginary

    @property
    def amplitude(self) -> Array:
        """Elementwise non-negative modulus."""
        return np.abs(self.array)

    @property
    def phase(self) -> Array:
        """Elementwise wrapped phase in radians."""
        return np.angle(self.array)

    @property
    def intensity(self) -> Array:
        """Elementwise squared modulus of the stored samples."""
        return np.abs(self.array) ** 2

    @property
    def complex(self) -> Array:
        """Real and imaginary components on a new leading axis."""
        values, _ = dlu.from_complex(self.array)
        return values

    @property
    def polar(self) -> Array:
        """Amplitude and phase on a new leading axis."""
        values, _ = dlu.from_complex(self.array, complex=False)
        return values

    @property
    def field(self) -> Array:
        """Return the sampled array."""
        return self.array

    @property
    def batch_ndim(self) -> int:
        """Return the number of axes preceding the sampled dimensions."""
        return self.array.ndim - self.grid.ndim

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Return the sampled array shape in NumPy axis order."""
        return self.field.shape[-self.grid.ndim :]

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return SI coordinate axes using the field's realised spatial shape.

        Axes are returned in physical ``(x, y)`` order and may contain leading grid
        batch dimensions.
        """
        return self.grid.axes

    @property
    def coordinates(self) -> Array:
        """Return SI coordinates using the field's realised spatial shape.

        The result has shape ``(..., grid.ndim, *grid.shape)``. Coordinate
        components follow physical order and samples follow reversed array order.
        """
        return self.grid.coordinates

    @property
    def xs(self) -> tuple[Array, ...]:
        """Alias `axes` using the retained compact coordinate terminology."""
        return self.axes

    @property
    def npixels(self) -> int:
        """Return ``nx`` for compatibility with square-grid APIs.

        This does not assert that ``ny == nx``; use `spatial_shape` for rectangular
        fields.
        """
        return self.field.shape[-1]

    @property
    def pixel_scale(self) -> Array:
        """Return per-axis pixel scales in canonical SI units.

        Values follow physical ``(x, y)`` order and preserve leading grid batch axes.
        """
        return self.grid.d_si

    @property
    def center(self) -> Array:
        """Return the grid centre in canonical SI units and physical-axis order."""
        return self.grid.c_si

    @property
    def diameter(self) -> Array:
        """Return the sampled field width along each physical axis.

        This aliases the grid field of view in its declared unit, not SI units.
        """
        return self.grid.fov

    def _map_arrays(self, function, *others):
        """Map a numerical kernel over flattened leading array batches.

        Each argument retains its final sampling axes. Matching leading axes are
        broadcast before flattening and restored around the kernel's result.
        """
        ndim = self.grid.ndim
        arrays = (self.array,) + tuple(np.asarray(value) for value in others)
        arrays = dlu.broadcast_batches(*arrays, ndims=ndim)
        batch = arrays[0].shape[:-ndim]
        values = [dlu.flatten_batch(array, ndim) for array in arrays]
        result = vmap(function)(*values)
        return result.reshape(batch + result.shape[1:])

    def convolve(self, other: Array, mode="same", method="auto") -> Field:
        """Convolve sampled axes independently across broadcast leading batches.

        Kernels have odd sizes on every sampled axis, placing their origin on a
        sample. This preserves the field's centre in same, full and valid modes.
        Leading kernel axes broadcast against the field's numerical batches.
        """
        # Require centred odd-sized kernels, then convolve each independent field batch.
        other = np.asarray(other)
        ndim = self.grid.ndim
        if other.ndim < ndim:
            raise ValueError("The kernel must contain every sampled axis.")
        if any(size % 2 == 0 for size in other.shape[-ndim:]):
            raise ValueError("Centred convolution requires odd kernel sizes.")
        apply = lambda x, y: convolve(x, y, mode=mode, method=method)
        array = self._map_arrays(apply, other)
        shape = array.shape[-ndim:]
        grid = self.grid
        if shape != grid.shape:
            grid = grid.resize_samples(shape[::-1])
        return self.set(array=array, grid=grid)

    def _binary_op(self, other, op: str) -> Field:
        """Apply arithmetic to another compatible sampled field or array."""
        if other is None:
            return self
        if not isinstance(other, (Field, Array, float, int, complex)):
            raise TypeError(
                f"Unsupported type for {op}: {type(other)}. Must be an array, "
                "field, or None."
            )
        self, other = self._prepare_operand(other)
        return self.set(field=_ops[op](self.field, other))

    def _prepare_operand(self, other) -> tuple[Field, Array]:
        """Return the field and array used for arithmetic."""
        if isinstance(other, Field):
            return self, other.field
        return self, other

    def resize(self, npixels: int | tuple[int, ...]) -> Field:
        """Resize sampled axes by centred zero-padding or cropping.

        ``npixels`` follows physical ``(x, y)`` order. Pixel scales and centre are
        retained, so changing the sample count changes the field of view.

        Returns a new field of the same concrete type; leading axes are preserved.
        """
        fill = 0j if np.iscomplexobj(self.field) else 0.0
        field = array_ops.resize(
            self.field, array_ops.as_size(npixels, self.grid.ndim), fill
        )
        return self.set(field=field, grid=self.grid.resize_samples(npixels))

    def downsample(self, n: int | tuple[int, ...], mean: bool = True) -> Field:
        """Downsample sampled axes by integer factors.

        ``n`` follows physical-axis order. Each block is averaged by default for
        both real and complex arrays; ``mean=False`` sums instead. Sampling spacing
        increases to preserve the coordinate extent.

        Returns a new field of the same concrete type with leading axes preserved.
        """
        field = array_ops.downsample(
            self.field, array_ops.as_size(n, self.grid.ndim), mean
        )
        return self.set(field=field, grid=self.grid.downsample(n))

    def flip(self, axis: tuple[int, ...] | int) -> Field:
        """Return a copy flipped about one or more NumPy array axes.

        ``axis`` follows array-axis indexing and may include leading batch axes. Grid
        metadata are retained unchanged.
        """
        return self.set(field=np.flip(self.field, axis))

    def __add__(self, other) -> Field:
        return self._binary_op(other, "add")

    def __sub__(self, other) -> Field:
        return self._binary_op(other, "subtract")

    def __mul__(self, other) -> Field:
        return self._binary_op(other, "multiply")

    def __truediv__(self, other) -> Field:
        return self._binary_op(other, "divide")

    def __iadd__(self, other) -> Field:
        return self._binary_op(other, "add")

    def __isub__(self, other) -> Field:
        return self._binary_op(other, "subtract")

    def __imul__(self, other) -> Field:
        return self._binary_op(other, "multiply")

    def __itruediv__(self, other) -> Field:
        return self._binary_op(other, "divide")

    def _sample(self, knots, coordinates, method, complex, fill=None, **kwargs):
        """Delegate sampled array batches to the interpolation engine."""
        operation = dlu.Interp(
            ndim=self.grid.ndim,
            array=self.array,
            grid_in=knots,
            method=method,
            complex=complex,
            fill=fill,
        )
        array = operation.sample(coordinates)
        return self.set(array=array, grid=kwargs.get("grid_out") or self.grid)

    def sample(self, coordinates: Array, method="cubic2", *, fill=None) -> Array:
        """Sample SI coordinates; None extrapolates, otherwise fill outside values."""
        return self._sample(self.axes, coordinates, method, True, fill).array

    def resample(self, grid_out, method="cubic2", *, fill=None, **context) -> Field:
        """Sample a new UnitSpec or apply an Interp recipe, preserving sampled values.

        ``fill=None`` extrapolates; a supplied value fills outside the source.
        Context supplies local values for expressions in the sampling recipe.
        """
        if isinstance(grid_out, dlu.Interp):
            operation = grid_out
        else:
            operation = dlu.Interp(
                ndim=self.grid.ndim, grid_out=grid_out, method=method, fill=fill
            )
        return self._resample(operation, **context)

    def _resample(self, operation, **context):
        """Apply a sampling recipe and retain its realised output specification."""
        grid = self.grid.resolve(**context)
        output = operation.output_grid(self.spatial_shape, grid, **context)
        operation = operation.set(grid_out=output)
        array = operation(self.array, grid_in=grid, **context)
        return self.set(array=array, grid=output)

    def scale_to(self, npixels, pixel_scale, method="cubic2", complex=True):
        """Interpolate to physical-axis sizes and SI per-axis spacing."""
        n = array_ops.as_size(npixels, self.grid.ndim, "npixels")
        spacing = dlu.as_axis(pixel_scale, self.grid.ndim) / dlu.unit_factor(
            self.grid.unit
        )
        return self.resample(
            dlu.Interp(
                ndim=self.grid.ndim,
                grid_out=self.grid.resample(n, spacing),
                method=method,
                complex=complex,
            )
        )

    def interpolate(self, transformation, method="cubic2", complex=True, fill=None):
        """Sample through an output-to-input coordinate transformation."""
        return self.resample(
            dlu.Interp(
                ndim=self.grid.ndim,
                transform=transformation,
                method=method,
                complex=complex,
                fill=fill,
            )
        )

    def rotate(self, angle, method="cubic2", complex=True):
        """Rotate the sampled field clockwise by angle radians about the origin."""
        return self.interpolate(
            transforms.Affine(rotation=np.asarray(angle)), method, complex
        )
