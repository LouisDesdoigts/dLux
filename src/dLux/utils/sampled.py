"""Coordinate-sampled numerical expressions."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as np
import zodiax as zdx

from . import grids

__all__ = ["Sampled"]


class Sampled(zdx.Expression):
    """An expression evaluated on optional stored or runtime sampling.

    Concrete classes implement ``evaluate`` and use ``sampling_context`` to select
    coordinates before calculating their values. A runtime grid or coordinates
    replaces stored sampling; a runtime transform replaces the stored mapping.
    Without sampling, resolution keeps the definition until inputs are available.

    ``strict=True`` rejects call-time replacement of a stored grid or transform.
    It does not change immutable ``set`` updates or require sampling at construction.
    Grid counts remain fixed topology; coordinate and transform parameters are
    ordinary differentiable leaves. This class does not resample stored arrays.
    """

    grid: grids.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    def __init__(self, grid=None, transform=None, strict=False, *, alias=None):
        """Store sampling and a map called as transform(coordinates, **context)."""
        if grid is not None and not isinstance(grid, grids.UnitSpec):
            raise TypeError("grid must be a sampling UnitSpec or None.")
        if transform is not None and not callable(transform):
            raise TypeError("transform must be callable or None.")
        self.grid = grid
        self.transform = transform
        self.strict = bool(strict)
        self.alias = alias

    def __call__(self, grid=None, *, transform=None, **context):
        """Resolve on stored sampling or an explicit grid/coordinate context."""
        return self.resolve(grid=grid, transform=transform, **context)

    def sampling_context(
        self,
        grid=None,
        *,
        coordinates=None,
        transform=None,
        ndim=None,
        pixel_scale=None,
        **context,
    ):
        """Select sampling, map coordinates once, and return local context.

        ``ndim`` is the coordinate dimension required by the calculation. Grid
        provides two spatial dimensions; Axis provides one coordinate dimension.
        One-dimensional sampling returns its coordinate vector directly. Higher
        dimensions use ``(..., ndim, *sample_shape)`` in physical-axis order.

        Coordinates are SI values. Pixel scale describes the original sampling,
        including when the transform changes evaluation coordinates; mapping is
        not a change of output grid or an interpolation of existing samples.
        """
        # Check whether runtime sampling or transforms may replace stored definitions.
        if grid is not None and coordinates is not None:
            raise ValueError("Supply exactly one of grid or coordinates, not both.")
        if self.strict:
            if self.grid is not None and (grid is not None or coordinates is not None):
                raise ValueError("strict=True prevents overriding the stored grid.")
            if self.transform is not None and transform is not None:
                raise ValueError(
                    "strict=True prevents overriding the stored transform."
                )

        # Explicit coordinates take priority over a stored grid. Only coordinate
        # generation uses the sampling object; children receive the local frame.
        if coordinates is None:
            grid = self.grid if grid is None else grid
            if grid is None:
                raise zdx.Unresolved("Sampled evaluation requires grid or coordinates.")
            if not isinstance(grid, grids.UnitSpec):
                raise TypeError("grid must be a sampling UnitSpec.")
            grid = grid.resolve(**context)
            if grid.n is None:
                raise zdx.Unresolved("Sampled evaluation requires sample counts.")
            needs_spacing = any(size > 1 for size in grid.shape)
            if needs_spacing and grid.d is None:
                raise zdx.Unresolved("Sampled evaluation requires sample spacing.")
            if ndim is not None and grid.ndim != ndim:
                raise ValueError(f"grid must define {ndim} coordinate dimensions.")
            ndim = grid.ndim
            coordinates = grid.axes[0] if ndim == 1 else grid.coordinates
            pixel_scale = grid.d_si if grid.d is not None else pixel_scale
        else:
            coordinates = zdx.resolve(coordinates, **context)
            coordinates = np.asarray(coordinates)

        # Validate the component axis, then pass children the transformed local frame.
        if ndim is not None and ndim > 1:
            if coordinates.ndim < ndim + 1 or coordinates.shape[-ndim - 1] != ndim:
                raise ValueError(
                    "coordinates must have shape (..., ndim, *sample_shape)."
                )
        local = dict(context, pixel_scale=pixel_scale)
        transform = self.transform if transform is None else transform
        if transform is not None:
            if not callable(transform):
                raise TypeError("transform must be callable.")
            coordinates = transform(coordinates, **local)
        local["coordinates"] = coordinates
        return local
