"""Interpolate array values on one-, two- or three-dimensional sampling."""

from __future__ import annotations

import equinox as eqx
import interpax as ipx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array

from .. import grids
from ..grids.coordinates import coordinate_grid
from .math import from_complex, to_complex
from ..batching import broadcast_batch, broadcast_batches, flatten_batch

__all__ = ["Interp", "interp"]


def interp(image, knot_coords, sample_coords, method="linear", fill=None, complex=True):
    """Interpolate a 2D array using physical (x, y) coordinate grids.

    Coordinates have shape (..., 2, ny, nx); arrays have shape (..., ny, nx).
    ``fill=None`` extrapolates; a supplied value fills outside the knot range.
    Sample positions and interpolated values are otherwise unchanged.
    """
    knots = np.asarray(knot_coords)
    axes = (knots[..., 0, 0, :], knots[..., 1, :, 0])
    operation = Interp(image, axes, method=method, complex=complex, fill=fill)
    return operation.sample(sample_coords)


class Interp(zdx.Expression):
    """Interpolate values, optionally binding an array for repeated sampling.

    Final ``ndim`` array axes are sampled; leading axes are ordinary batches.
    Source axis vectors use physical (x, y, z) order. Grid and Axis supply SI
    coordinates; raw vectors and queries must use the same units.

    ``Interp(rotation=angle)(array)`` samples transformed pixel coordinates.
    Positive angles rotate coordinates anticlockwise, turning the sampled object
    clockwise in physical (x, y), as in the released interpolation API.
    ``Interp(array, grid_in).sample(coordinates)`` queries a bound source.
    ``prepare()`` stores an Interpax snapshot for reuse. Re-prepare after changing
    source values, sampling, method or complex mode. Prepare inside differentiated
    code when fitting those inputs.

    Interpolation only calculates sample values. It does not normalise arrays,
    conserve flux, or compensate for coordinate stretching.
    """

    array: Array | zdx.Expression | None = None
    grid_in: grids.UnitSpec | Array | tuple[Array, ...] | zdx.Expression | None = None
    grid_out: grids.UnitSpec | None = None
    transform: grids.BaseCoordTransform | None = None
    ndim: int = eqx.field(static=True, default=2)
    method: str
    complex: bool = eqx.field(static=True, default=True)
    fill: Array | zdx.Expression | None = None
    interpolator: ipx.AbstractInterpolator | None = None

    def __init__(
        self,
        array=None,
        grid_in=None,
        grid_out=None,
        *,
        ndim=2,
        method="cubic2",
        rotation=None,
        scale=None,
        shift=None,
        transform=None,
        complex=True,
        fill=None,
        alias=None,
    ):
        """Store samples, sampling and optional coordinate mapping.

        ``grid_in`` accepts a UnitSpec, a 1D vector, or a tuple of increasing axis
        vectors. None uses centred unit pixels. Each axis needs at least two
        samples. ``grid_out`` supplies the output sampling for apply/resolve.

        ``rotation``, ``scale`` and ``shift`` construct a 2D Affine; alternatively
        supply ``transform``. ``complex=True`` interpolates real/imaginary parts;
        False interpolates amplitude/phase. ``method`` is an Interpax method,
        defaulting to the C2 cubic spline ``cubic2``.

        ``fill=None`` extrapolates outside the source. A supplied value fills
        outside instead, including zero, complex values or NaN. Queries are never
        snapped to the knot endpoints.
        """
        if ndim not in (1, 2, 3):
            raise ValueError("ndim must be 1, 2 or 3.")
        if grid_out is not None and not isinstance(grid_out, grids.UnitSpec):
            raise TypeError("grid_out must be a UnitSpec or None.")

        # Motion keywords are a convenience for the same coordinate transform.
        motions = (rotation, scale, shift)
        if ndim != 2 and any(value is not None for value in (*motions, transform)):
            raise ValueError("Coordinate transforms require ndim=2.")
        if transform is not None and any(value is not None for value in motions):
            raise ValueError("Supply transform or rotation/scale/shift, not both.")
        if transform is None and any(value is not None for value in motions):
            transform = grids.Affine(translation=shift, rotation=rotation, scale=scale)

        # Store source sampling separately from the output grid and coordinate map.
        self.array = zdx.as_array(array)
        if isinstance(grid_in, grids.UnitSpec):
            self.grid_in = grid_in
        elif isinstance(grid_in, tuple):
            self.grid_in = tuple(zdx.as_array(axis) for axis in grid_in)
        else:
            self.grid_in = zdx.as_array(grid_in)
        self.grid_out = grid_out
        self.transform = transform
        self.ndim = int(ndim)
        self.method = method
        self.complex = bool(complex)
        self.fill = zdx.as_array(fill)
        self.interpolator = None
        self.alias = alias

    @property
    def axes(self):
        """Return the physical-order knot vectors of a prepared interpolation."""
        if self.interpolator is None:
            raise ValueError("Prepare the interpolation before accessing its axes.")
        names = "xyz"[: self.ndim]
        return tuple(getattr(self.interpolator, name) for name in reversed(names))

    def evaluate(self, *, grid=None, coordinates=None, **context):
        """Resolve only when both source values and an output mapping are available."""
        if self.array is None and self.interpolator is None:
            raise zdx.Unresolved("Interpolation requires source samples.")
        outputs = (grid, coordinates, self.grid_out, self.transform)
        if all(value is None for value in outputs):
            raise zdx.Unresolved(
                "Interpolation requires output sampling or a transform."
            )
        if coordinates is not None:
            return self.sample(coordinates, **context)
        operation = self if grid is None else self.set(grid_out=grid)
        return operation.apply(**context)

    def __call__(self, array=None, grid_in=None, **context):
        """Apply to supplied samples, or resolve the stored expression."""
        if array is None and grid_in is None:
            return self.resolve(**context)
        return self.apply(array, grid_in, **context)

    def apply(self, array=None, grid_in=None, **context):
        """Interpolate onto grid_out, or the source sample points if it is absent."""
        # Explicit inputs replace stored definitions and invalidate the snapshot.
        operation = self
        if array is not None:
            operation = operation.set(array=zdx.as_array(array), interpolator=None)
        if grid_in is not None:
            operation = operation.set(grid_in=grid_in, interpolator=None)
        if operation.interpolator is None:
            operation = operation.prepare(**context)

        # Select the destination independently of source preparation.
        shape = operation.interpolator.f.shape[-self.ndim - 1 : -1]
        output = operation.output_grid(shape, **context)
        if output is None:
            coordinates = coordinate_grid(operation.axes)
        else:
            coordinates = output.coordinates

        return operation.sample(coordinates, **dict(context, grid=output))

    def output_grid(self, shape, grid_in=None, **context):
        """Resolve output sampling, using the source shape for omitted counts.

        ``shape`` contains the sampled array axes in array order. A Field supplies
        its input grid as the fallback when grid_out is absent. Without either
        grid, return None so interpolation can use its original knot vectors.
        """
        output = grid_in if self.grid_out is None else self.grid_out
        output = zdx.resolve(output, **context)
        if output is None:
            return None
        if output.ndim != self.ndim:
            raise ValueError("Output sampling dimension must match ndim.")
        if output.n is None:
            output = output.match_shape(shape)
        return output

    def prepare(self, **context):
        """Prepare an Interpax snapshot from the stored source and its sampling."""
        # Source expressions must see source coordinates, never query coordinates.
        context.pop("grid", None)
        context.pop("coordinates", None)
        source = self.populate()
        grid = zdx.resolve(source.grid_in, **context)
        sampling_known = grid is not None and (
            not isinstance(grid, grids.UnitSpec) or grid.n is not None
        )
        if isinstance(source.array, zdx.Expression) and sampling_known:
            axes = grid.axes if isinstance(grid, grids.UnitSpec) else grid
            axes = axes if isinstance(axes, tuple) else (axes,)
            if self.ndim == 1 and axes[0].ndim == 1:
                context["coordinates"] = axes[0]
            else:
                context["coordinates"] = coordinate_grid(axes)
            if isinstance(grid, grids.UnitSpec):
                context["pixel_scale"] = grid.d_si

        # Validate only the numerical shapes needed by Interpax.
        array = zdx.resolve(source.array, **context)
        if array is None or isinstance(array, zdx.Expression):
            raise zdx.Unresolved("Interpolation requires resolved source samples.")
        array = np.asarray(array)
        if array.ndim < self.ndim or min(array.shape[-self.ndim :]) < 2:
            raise ValueError("Each interpolation axis requires at least two samples.")

        # Recover source knot vectors and check them against the sampled array axes.
        if isinstance(grid, grids.UnitSpec):
            if grid.ndim != self.ndim:
                raise ValueError("Source sampling dimension must match ndim.")
            axes = grid.match_shape(array.shape[-self.ndim :]).axes
        elif grid is None:
            axes = tuple(
                np.arange(n, dtype=float) - (n - 1) / 2
                for n in array.shape[-self.ndim :][::-1]
            )
        else:
            axes = grid if isinstance(grid, tuple) else (grid,)
        if any(np.iscomplexobj(axis) for axis in axes):
            raise TypeError("Interpolation coordinates must be real.")
        axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        if len(axes) != self.ndim:
            raise ValueError("Supply one source axis vector per sampled dimension.")
        for axis, size in zip(axes, array.shape[-self.ndim :][::-1]):
            if axis.ndim == 0 or axis.shape[-1] != size:
                raise ValueError("Source axis vectors must match the array shape.")

        interpolator = source._build(array, axes)
        return source.set(interpolator=interpolator)

    def sample(self, coordinates, **context):
        """Query a bound array, mapping coordinates before interpolation.

        Tuples (x, y, z) describe arbitrary point shapes. Stacked grid coordinates
        have shape (..., ndim, *Q), with exactly ndim query axes. Leading grid
        batches broadcast against source batches. In 1D, scalars/vectors also work.
        """
        if self.interpolator is None:
            operation = self.prepare(**context)
        else:
            operation = self.populate()
        coordinates = zdx.resolve(coordinates, **context)

        # Flatten arbitrary point tuples into a grid-shaped query temporarily.
        ndim = self.ndim
        point_shape = None
        if isinstance(coordinates, tuple):
            if len(coordinates) != ndim:
                raise ValueError("Supply one query component per physical axis.")
            components = np.broadcast_arrays(*coordinates)
            point_shape = components[0].shape
            coordinates = np.stack([value.reshape(-1) for value in components])
        else:
            coordinates = np.asarray(coordinates)
            if ndim == 1 and coordinates.ndim <= 1:
                point_shape = coordinates.shape

        if point_shape is not None:
            query_shape = (ndim,) + (1,) * (ndim - 1) + (-1,)
            coordinates = coordinates.reshape(query_shape)

        # Stacked grids have a coordinate component axis before the query axes.
        if coordinates.ndim < ndim + 1 or coordinates.shape[-ndim - 1] != ndim:
            raise ValueError(
                "Stacked queries need (..., ndim, *Q) with ndim query axes. "
                "Use tuples for points."
            )
        if np.iscomplexobj(coordinates):
            raise TypeError("Interpolation coordinates must be real.")
        coordinates = np.asarray(coordinates, dtype=float)

        # Transform query positions, then interpolate values without rescaling.
        local = dict(context, coordinates=coordinates)
        fill = zdx.resolve(operation.fill, **local)
        if operation.transform is not None:
            coordinates = operation.transform(coordinates, **context)
        values = operation._interpolate(coordinates, fill)

        if point_shape is not None:
            values = values.reshape(values.shape[: -self.ndim] + point_shape)
        return values

    def _build(self, array, axes):
        """Prepare one Interpax object, with complex components as value channels."""
        ndim = self.ndim
        constructors = (ipx.Interpolator1D, ipx.Interpolator2D, ipx.Interpolator3D)
        constructor = constructors[ndim - 1]

        # Interpax interpolates trailing value channels together.
        if np.iscomplexobj(array):
            values, _ = from_complex(array, self.complex, axis=-1)
        else:
            values = np.asarray(array, dtype=np.result_type(array, 1.0))[..., None]

        # One value contains the sampled axes and a final component channel.
        # Interpax takes knots in array order (z, y, x), opposite to physical order.
        value_ndim = ndim + 1
        dimensions = (value_ndim,) + (1,) * ndim
        values, *knots = broadcast_batches(values, *axes[::-1], ndims=dimensions)
        batch = values.shape[:-value_ndim]

        # Each vmap call constructs one interpolation table.
        values = flatten_batch(values, value_ndim)
        knots = tuple(flatten_batch(knot, 1) for knot in knots)

        build = lambda values, *knots: constructor(
            *knots, values, method=self.method, extrap=True
        )
        prepared = eqx.filter_vmap(build)(values, *knots)
        return jtu.map(
            lambda value: (
                value.reshape(batch + value.shape[1:]) if eqx.is_array(value) else value
            ),
            prepared,
        )

    def _interpolate(self, coordinates, fill):
        """Pair source/query batches and let Interpax calculate the sample values."""
        interpolator, ndim = self.interpolator, self.ndim
        source_batch = interpolator.f.shape[: -ndim - 1]
        query_shape = coordinates.shape[-ndim:]
        batch = np.broadcast_shapes(source_batch, coordinates.shape[: -ndim - 1])

        # Each mapped call sees one table and one coordinate grid. All interpolation
        # coefficients live in the Interpax object, including any prepared snapshot.
        def broadcast(value):
            if not eqx.is_array(value):
                return value
            ndim = value.ndim - len(source_batch)
            value = broadcast_batch(value, batch, ndim)
            return flatten_batch(value, ndim)

        prepared = jtu.map(broadcast, interpolator)
        queries = broadcast_batch(coordinates, batch, ndim + 1)
        queries = flatten_batch(queries, ndim + 1)
        if fill is not None:
            fill = broadcast_batch(fill, batch, ndim=0).reshape(-1)

        def sample(table, query, fill):
            # Interpax takes array-axis order (z, y, x). Only reshape/reorder the
            # components; evaluate the exact requested coordinates.
            points = query.reshape(ndim, -1)[::-1]
            values = table(*points)
            if values.shape[-1] == 2:
                result = to_complex(values, self.complex, axis=-1)
            else:
                result = values[..., 0]

            # None keeps the extrapolated values. Apply a supplied fill after complex
            # reconstruction so it also works with amplitude/phase interpolation.
            if fill is not None:
                knots = tuple(getattr(table, name) for name in "xyz"[:ndim])
                lower = np.stack([axis[0] for axis in knots])[:, None]
                upper = np.stack([axis[-1] for axis in knots])[:, None]
                inside = np.all((points >= lower) & (points <= upper), axis=0)
                result = np.where(inside, result, fill)
            return result.reshape(query_shape)

        # Batch the table's arrays; share its Python settings (axis, extrapolation).
        values = eqx.filter_vmap(sample)(prepared, queries, fill)
        return values.reshape(batch + query_shape)
