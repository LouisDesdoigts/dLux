"""Numerical sampling specifications and coordinate transformations."""

from __future__ import annotations

from abc import abstractmethod
import warnings

import jax.numpy as np
import numpy as onp
from jax import Array, core, lax, vmap
import equinox as eqx
import zodiax as zdx

import dLux.utils as dlu

__all__ = [
    "BaseGridSpec",
    "GridSpec",
    "Grid",
    "ResizeSpec",
    "FFTSpec",
    "PasteSpec",
    "BaseCoordTransform",
    "Affine",
    "AffineMap",
    "TransformChain",
    "Distortion",
]


def _distortion_powers(order, orders, powers, shift_invariant):
    """Resolve the polynomial powers used by a coordinate distortion."""
    # Validate mutually exclusive term specifications
    if sum(value is not None for value in (order, orders, powers)) > 1:
        raise ValueError("Provide only one of order, orders, or powers.")

    # Validate explicit powers or generate selected total orders
    if powers is not None:
        powers = np.asarray(powers)
        if powers.ndim != 2 or powers.shape[0] != 2:
            raise ValueError("powers must have shape (2, n_terms).")
        if not np.issubdtype(powers.dtype, np.integer) or np.any(powers < 0):
            raise ValueError("powers must contain non-negative integers.")
    else:
        if orders is None:
            order = 1 if order is None else int(order)
            orders = tuple(range(1, order + 1))
        else:
            orders = tuple(map(int, orders))
        if not orders or any(order < 1 for order in orders):
            raise ValueError("orders must contain positive integers.")
        powers = dlu.polynomial_powers(max(orders), 2)[:, 1:]
        powers = powers[:, np.isin(powers.sum(0), np.asarray(orders))]

    # Remove linear coordinate terms for shift-invariant distortions
    if shift_invariant:
        linear = np.logical_or(
            np.all(powers == np.array([[1], [0]]), axis=0),
            np.all(powers == np.array([[0], [1]]), axis=0),
        )
        powers = powers[:, ~linear]
    return powers


class _PixelSpacing(zdx.Expression):
    """Resolve scalar, per-axis or batched diameters into pixel spacing."""

    diameter: zdx.Expression
    n: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, diameter, n):
        self.diameter = diameter
        self.n = n
        self.alias = None

    def evaluate(self, **context) -> Array:
        diameter = zdx.resolve(self.diameter, **context)
        diameter = dlu.as_axis(diameter, len(self.n), "diam")
        return diameter / np.asarray(self.n)


class BaseGridSpec(zdx.Module):
    """Base contract for immutable coordinate and sampling specifications.

    Concrete specifications describe either a physical grid or a construction-time
    resize/placement operation. Tuple values follow physical-axis order, which is the
    reverse of the final NumPy array-axis order.
    """


class ResizeSpec(BaseGridSpec):
    """Define output sampling by an explicit size or integer resize factors.

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
    n : int, tuple[int, ...], or None
        Explicit output sizes in physical-axis order. Mutually exclusive with
        non-unit ``pad`` or ``crop``.
    pad, crop : int or tuple[int, ...]
        Integer factors applied before and after an operation.
    c : Array or None
        Optional output centre in the associated propagation unit.
    """

    n: tuple[int, ...] | None = None
    pad: tuple[int, ...]
    crop: tuple[int, ...]
    c: Array | None = None

    def __init__(self, n=None, pad=1, crop=1, c=None, *, alias=None):
        """Initialise an explicit or factor-based resize specification.

        Parameters
        ----------
        n : int, tuple[int, ...], or None
            Explicit physical-axis output sizes, mutually exclusive with non-unit
            ``pad`` or ``crop``.
        pad, crop : int or tuple[int, ...]
            Positive integer factors applied before and after an operation.
        c : Array or None
            Optional output centre in the propagation output unit.
        alias : Mapping or sequence of pairs, optional
            Optional Zodiax parameter-path aliases.
        """
        if n is not None and (pad != 1 or crop != 1):
            raise ValueError("Specify either n or pad/crop factors, not both.")
        self.n = None if n is None else dlu.as_size(n, name="n")
        self.pad = dlu.as_size(pad, name="pad")
        self.crop = dlu.as_size(crop, name="crop")
        self.c = zdx.as_array(c, dtype=float)

        self.alias = alias

    @property
    def ndim(self) -> int | None:
        """Return the fixed topology dimensionality, when one is defined.

        Scalar values retain their broadcast intent and therefore return ``None``.
        Non-scalar explicit sizes or factors fix the number of physical axes.
        """
        values = (self.n, self.pad, self.crop)
        lengths = [
            len(value) for value in values if value is not None and len(value) > 1
        ]
        if not lengths:
            return None
        if len(set(lengths)) > 1:
            raise ValueError("Resize values must have compatible dimensionality.")
        return lengths[0]

    def broadcast(self, ndim: int) -> BaseGridSpec:
        """Broadcast scalar resize values to a fixed dimensionality.

        ``ndim`` is the positive number of physical dimensions. The returned copy
        has ``n``, ``pad``, and ``crop`` tuples of that length; existing non-scalar
        tuples must already have the requested length.
        """
        ndim = int(ndim)
        if ndim < 1:
            raise ValueError("ndim must be a positive integer.")
        n = None if self.n is None else dlu.as_size(self.n, ndim, "n")
        return self.set(
            n=n,
            pad=dlu.as_size(self.pad, ndim, "pad"),
            crop=dlu.as_size(self.crop, ndim, "crop"),
        )

    @property
    def explicit(self) -> bool:
        """Return whether ``n`` defines an absolute output size.

        When false, output sampling is determined from the input shape and the
        configured integer pad and crop factors.
        """
        return self.n is not None

    @property
    def _padding(self) -> dict:
        """Return FFT-padding keywords represented by this specification.

        Explicit specifications return ``{"pad_to": n}``; factor specifications
        return a two-dimensional ``{"pad": pad}`` mapping.
        """
        if self.explicit:
            return {"pad_to": self.n}
        return {"pad": dlu.as_size(self.pad, 2, "pad")}

    def counts(self, n_in) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return FFT working and output counts for physical-order input sizes.

        An explicit ``n`` defines both the working and returned counts. Otherwise,
        input counts are multiplied by ``pad`` and the resulting working counts are
        integer-divided by ``crop``. The established floor-division behaviour is
        preserved when the factors do not divide the working counts exactly.
        """
        ndim = self.ndim
        n_in = dlu.as_size(n_in, ndim, "n_in")
        ndim = len(n_in)

        if self.explicit:
            n = dlu.as_size(self.n, ndim, "n")
            return n, n

        pad = dlu.as_size(self.pad, ndim, "pad")
        crop = dlu.as_size(self.crop, ndim, "crop")
        n_work = tuple(n * factor for n, factor in zip(n_in, pad))
        n_out = tuple(n // factor for n, factor in zip(n_work, crop))
        if any(n < 1 for n in n_out):
            raise ValueError("crop factors may not reduce an output count below one.")
        return n_work, n_out

    def output_size(self, shape) -> tuple[int, ...]:
        """Return final physical-axis sizes for an input NumPy ``shape``.

        The final spatial axes are reversed into physical order before applying the
        configured pad and crop factors. Explicit ``n`` is returned unchanged.
        """
        if self.explicit:
            return self.n
        ndim = max(len(self.pad), len(self.crop), 2)
        pad = dlu.as_size(self.pad, ndim, "pad")
        crop = dlu.as_size(self.crop, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(size * pad // crop for size, pad, crop in zip(sizes, pad, crop))

    def crop_size(self, shape) -> tuple[int, ...]:
        """Return physical-axis sizes after applying only crop factors.

        ``shape`` is supplied in NumPy array order. Its final relevant axes are
        reversed into physical order before division by ``crop``.
        """
        if self.explicit:
            return self.n
        ndim = max(len(self.crop), 2)
        factors = dlu.as_size(self.crop, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(size // factor for size, factor in zip(sizes, factors))

    def pad_array(self, array: Array, fill: float = 0.0) -> Array:
        """Centrally pad the final spatial axes.

        Explicit ``n`` sets the target size; otherwise each final axis is enlarged by
        its pad factor. ``fill`` supplies values outside the original array and
        leading axes are preserved.
        """
        if self.explicit:
            return dlu.pad_to(array, self.n, fill)
        ndim = max(len(self.pad), 2)
        factors = dlu.as_size(self.pad, ndim, "pad")
        sizes = tuple(array.shape[-ndim:][::-1])
        target = tuple(size * factor for size, factor in zip(sizes, factors))
        return dlu.pad_to(array, target, fill)

    def crop_array(self, array: Array) -> Array:
        """Centrally crop final spatial axes to the configured output size.

        Leading array axes are preserved and output spatial axes follow NumPy order.
        """
        if self.explicit:
            return dlu.crop_to(array, self.n)
        return dlu.crop_to(array, self.crop_size(array.shape))

    def crop_axes(self, axes: tuple[Array, ...]) -> tuple[Array, ...]:
        """Centrally crop one-dimensional coordinate axes.

        ``axes`` and the returned tuple follow physical-axis order. Each vector is
        cropped consistently with :meth:`crop_array`.
        """
        if self.explicit:
            sizes = self.n
        else:
            factors = dlu.as_size(self.crop, len(axes), "crop")
            sizes = tuple(a.shape[-1] // f for a, f in zip(axes, factors))

        crop_fn = lambda a, n: dlu.crop_to(a, (n,))
        return tuple(crop_fn(a, n) for a, n in zip(axes, sizes))

    def resize(self, array: Array, fill: float = 0.0) -> Array:
        """Centrally pad or crop final axes to the represented output size.

        ``fill`` is used only for padded values. Leading axes are preserved.
        """
        return dlu.resize(array, self.output_size(array.shape), fill)


class FFTSpec(ResizeSpec):
    """Request native Fourier sampling through the canonical resize contract.

    ``n`` requests an explicit FFT count. Otherwise, ``pad`` determines the FFT
    working count and ``crop`` determines the returned central count. ``c=None``
    keeps the native centre and therefore requires no shift ramp; an explicit
    centre is interpreted in the natural canonical coordinate of the consuming
    propagator: radians for normalised Fraunhofer, metres for direct LCT, or
    reciprocal metres for spectral LCT.

    All counts follow physical-axis order. Scalar values broadcast once the source
    dimensionality is known, and centre arrays may carry leading batch axes.

    Parameters
    ----------
    n : int, tuple[int, ...], or None
        Explicit FFT working and output counts. Mutually exclusive with non-unit
        ``pad`` or ``crop``.
    pad, crop : int or tuple[int, ...]
        Integer working-size and output-crop factors.
    c : ArrayLike or None
        Optional final output centre in the consuming propagator's natural SI
        coordinate.
    """

    n: tuple[int, ...] | None = None
    pad: tuple[int, ...]
    crop: tuple[int, ...]
    c: Array | None = None

    def working_grid(self, source: GridSpec) -> GridSpec:
        """Return padded input sampling, rejecting any loss of input samples."""
        n_work, _ = self.counts(source.n)
        if any(work < size for work, size in zip(n_work, source.n)):
            raise ValueError("FFT working counts may not crop the input grid.")
        return source.resize_samples(n_work)

    def output_grid(self, native: GridSpec, n_in) -> GridSpec:
        """Crop native Fourier sampling and apply an optional requested centre.

        ``native`` uses the consuming propagator's SI output unit. ``n_in`` is
        the original input count tuple used to resolve pad/crop factors.
        """
        _, n_out = self.counts(n_in)
        output = native.resize_samples(n_out)
        if self.c is not None:
            output = output.set(c=dlu.as_axis(self.c, native.ndim, "FFTSpec.c"))
        return output


class PasteSpec(BaseGridSpec):
    """Map compact local arrays into a larger regularly sampled grid.

    The specification stores integer ``(x, y)`` starts and residual physical offsets
    for equally sized stamps. Construction is deliberately separate from evaluation:
    stamp and output shapes are fixed before JIT compilation, while coordinates and
    pasted values remain ordinary JAX calculations.

    This class is normally constructed by segmented aperture builders. It remains
    public so custom builders can reuse compact JIT-compatible placement without
    materialising full-size intermediate arrays.
    """

    n: tuple[int, int]
    shape: tuple[int, int]
    starts: Array
    offsets: Array
    d: Array

    def __init__(self, n, shape, starts, offsets, d, *, alias=None):
        """Initialise fixed placement geometry for compact stamps.

        Parameters
        ----------
        n : int or tuple[int, int]
            Full output size in physical ``(x, y)`` order.
        shape : int or tuple[int, int]
            Shared stamp size in physical ``(x, y)`` order.
        starts : Array
            Integer lower-corner indices with shape ``(n_stamps, 2)``.
        offsets : Array
            Residual physical offsets matching ``starts``.
        d : Array
            Output pixel scales in physical ``(x, y)`` order.
        """
        self.n = dlu.as_size(n, 2, "n")
        self.shape = dlu.as_size(shape, 2, "shape")
        self.starts = zdx.as_array(starts, dtype=int)
        self.offsets = zdx.as_array(offsets, dtype=float)
        self.d = dlu.as_axis(d, 2, "d")

        if self.starts.ndim != 2 or self.starts.shape[-1] != 2:
            raise ValueError("starts must have shape (n_stamps, 2).")
        if self.offsets.shape != self.starts.shape:
            raise ValueError("offsets must match the shape of starts.")

        ends = self.starts + np.asarray(self.shape)
        if np.any(self.starts < 0) or np.any(ends > np.asarray(self.n)):
            raise ValueError("Every stamp must lie within the output grid.")

        self.alias = alias

    @classmethod
    def from_grid(cls, grid, centers, extent):
        """Construct compact stamp placement from a concrete grid.

        Parameters
        ----------
        grid : GridSpec
            Concrete two-dimensional grid with defined size and sampling.
        centers : Array
            Physical stamp centres with shape ``(n_stamps, 2)``.
        extent : float or Array
            Local half-width required around each centre.
        """
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        if grid.n is None or grid.d is None or grid.ndim != 2:
            raise ValueError("grid must define two-dimensional n and d values.")

        centers = zdx.as_array(centers, dtype=float)
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape (n_stamps, 2).")

        # Resolve the concrete grid geometry in canonical SI units
        spacing = grid.d * grid.scale
        center = np.zeros(2) if grid.c is None else grid.c * grid.scale
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

        The result has shape ``(n_stamps, 2, stamp_y, stamp_x)``. Each grid is centred
        on its requested physical centre through the stored residual offset.
        """
        coordinates = dlu.nd_coords(self.shape, self.d)
        return coordinates[None] + self.offsets[:, :, None, None]

    def paste(self, arrays, method="scan") -> Array:
        """Add compact stamp arrays into the common output grid.

        Parameters
        ----------
        arrays : Array
            Values with shape ``(n_stamps, ..., stamp_y, stamp_x)``.
        method : {"scan", "vmap"}
            ``"scan"`` limits intermediate memory; ``"vmap"`` evaluates placements
            in parallel and may use more memory.

        Returns
        -------
        array : Array
            Summed values with shape ``(..., output_y, output_x)``. Overlapping
            stamps are added.
        """
        return dlu.paste(arrays, self.starts, self.n, method)

    def extract(self, array) -> Array:
        """Extract one compact stamp at every configured start position.

        ``array`` must have shape ``(..., output_y, output_x)``. The returned array
        has shape ``(n_stamps, ..., stamp_y, stamp_x)``.
        """
        # Validate the full spatial shape and resolve slice geometry
        array = np.asarray(array)
        if array.shape[-2:] != self.n[::-1]:
            raise ValueError("array spatial shape must match the PasteSpec output.")

        leading = array.shape[:-2]
        slice_shape = leading + self.shape[::-1]

        # Extract every compact stamp in parallel
        def extract_stamp(start):
            index = (0,) * len(leading) + (start[1], start[0])
            return lax.dynamic_slice(array, index, slice_shape)

        return vmap(extract_stamp)(self.starts)


class GridSpec(BaseGridSpec):
    """A complete or partially specified regular Cartesian coordinate grid.

    Axis parameters are ordered physically as ``(x, y, z, ...)``. Array dimensions
    are ordered in reverse, so a two-dimensional specification with
    ``n=(nx, ny)`` produces coordinate arrays with shape ``(2, ny, nx)``.

    Parameters
    ----------
    n : int, tuple[int, ...], or None
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


    # Define a one-dimensional grid from its total field of view
    grid = dl.GridSpec(n=128, diam=6.5, unit="m")

    # Equivalently, define the grid from its pixel scale
    grid = dl.GridSpec(n=128, d=6.5 / 128, unit="m")

    # Inspect its sampling
    fov = grid.fov  # Total sampled width in metres
    coordinates = grid.coordinates  # Shape (1, 128), in SI units

    # Broadcast the same sampling to a square grid
    square = grid.broadcast(2)
    coordinates = square.coordinates  # Shape (2, 128, 128)

    # Define a non-square, anisotropic, and off-centre grid
    grid = dl.GridSpec(
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
    Scalar inputs define a one-dimensional grid. Fields promote these to their own
    spatial dimensionality; standalone grids require an explicit ``broadcast``.
    Coordinate components retain singleton sample axes and any leading metadata
    batches. Sample counts and indexing are fixed topology; spacings and centres
    remain differentiable numerical leaves.
    """

    n: tuple[int, ...] | None = None
    d: Array | zdx.Expression | None = None
    c: Array | zdx.Expression | None = None
    unit: str | None = None
    indexing: str = eqx.field(default="xy", static=True)

    def __init__(
        self, n=None, d=None, c=None, unit=None, diam=None, *, indexing="xy", alias=None
    ):
        """Initialise a regular Cartesian sampling specification.

        Parameters
        ----------
        n : int, tuple[int, ...], or None
            Sample counts in physical-axis order.
        d : ArrayLike or None
            Pixel scales in ``unit``. Mutually exclusive with ``diam``.
        c : ArrayLike or None
            Grid centres in ``unit``.
        unit : str or None
            Supported physical or angular unit for ``d`` and ``c``.
        diam : ArrayLike or None
            Physical extent used to derive ``d = diam / n``; requires ``n``.
        indexing : {"xy", "ij"}
            Defaults to dLux image ordering: reverse physical axes in arrays.
            ``ij`` retains physical-axis order for any number of dimensions.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter-path aliases. ``d`` and ``c`` may be expressions
            resolved later with :meth:`sampled`.
        """
        # Validate mutually dependent sampling inputs
        if d is not None and diam is not None:
            raise ValueError("Provide only one of d or diam.")
        if diam is not None and n is None:
            raise ValueError("n must be provided with diam.")
        if indexing not in ("xy", "ij"):
            raise ValueError("indexing must be 'xy' or 'ij'.")

        # Infer the physical dimensionality from non-scalar inputs
        values = [
            value
            for value in (n, d, c, diam)
            if value is not None and not isinstance(value, zdx.Expression)
        ]
        lengths = [
            np.asarray(value).shape[-1]
            for value in values
            if np.asarray(value).ndim >= 1
        ]
        ndim = max(lengths, default=1 if values else 0)

        # Resolve sizes and optional diameter-based sampling
        self.n = None if n is None else dlu.as_size(n, ndim, "n")

        if diam is not None:
            if isinstance(diam, zdx.Expression):
                d = _PixelSpacing(diam, self.n)
            else:
                diam = dlu.as_axis(diam, ndim, "diam")
                d = diam / np.asarray(self.n)

        # Standardise sampling and centre arrays
        self.d = d if isinstance(d, zdx.Expression) else dlu.as_axis(d, ndim, "d")
        self.c = c if isinstance(c, zdx.Expression) else dlu.as_axis(c, ndim, "c")

        # Validate concrete positive pixel scales
        if (
            self.d is not None
            and not isinstance(self.d, zdx.Expression)
            and not isinstance(self.d, core.Tracer)
            and onp.any(onp.asarray(self.d) <= 0)
        ):
            raise ValueError("d must contain positive values.")

        # Validate and store the declared coordinate unit
        self.unit = None if unit is None else self._validate_unit(unit)
        self.indexing = indexing
        self.alias = alias

    @staticmethod
    def _validate_unit(unit):
        return dlu.canonical_unit(unit, name="grid unit")

    @property
    def ndim(self) -> int:
        """Return the physical dimensionality inferred from defined grid leaves.

        Returns zero only when ``n``, ``d``, and ``c`` are all undefined.
        """
        for value in (self.n, self.d, self.c):
            if value is not None and not isinstance(value, zdx.Expression):
                return len(value) if isinstance(value, tuple) else value.shape[-1]
        return 0

    def broadcast(self, ndim: int) -> GridSpec:
        """Broadcast scalar grid values to a fixed physical dimensionality.

        ``ndim`` is the positive number of physical axes. The returned copy has
        defined ``n``, ``d``, and ``c`` values with final axis length ``ndim``;
        existing non-scalar values must already have that length.
        """
        ndim = int(ndim)
        if ndim < 1:
            raise ValueError("ndim must be a positive integer.")
        return self.set(
            n=None if self.n is None else dlu.as_size(self.n, ndim, "n"),
            d=(
                self.d
                if isinstance(self.d, zdx.Expression)
                else dlu.as_axis(self.d, ndim, "d")
            ),
            c=(
                self.c
                if isinstance(self.c, zdx.Expression)
                else dlu.as_axis(self.c, ndim, "c")
            ),
        )

    def match_shape(self, shape) -> GridSpec:
        """Return a grid matching a spatial shape in array-axis order.

        The grid is broadcast to the number of supplied spatial axes. An undefined
        ``n`` is populated from the reversed array shape; an existing ``n`` must
        already describe the same physical-axis pixel counts.
        """
        shape = dlu.as_size(shape, name="shape")
        grid = self.broadcast(len(shape))
        n = shape[::-1] if self.indexing == "xy" else shape

        if grid.n is None:
            return grid.set(n=n)
        if grid.n != n:
            raise ValueError("Spatial shape must match grid.n.")
        return grid

    def resize(self, n) -> GridSpec:
        """Change physical-axis sample counts while retaining ``d`` and ``c``.

        ``n`` is scalar or follows physical-axis order. The returned copy may have a
        different field of view because its pixel scales are unchanged.
        """
        return self.set(n=dlu.as_size(n, self.ndim, "n"))

    def sampled(self, **context) -> GridSpec:
        """Resolve sampling expressions without constructing coordinates.

        Missing leaves remain missing. A field may subsequently recover ``n``
        through :meth:`match_shape`; a propagator requires complete sampling.
        """
        d = zdx.resolve(self.d, **context)
        c = zdx.resolve(self.c, **context)
        return type(self)(
            self.n, d, c, self.unit, indexing=self.indexing, alias=self.alias
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Broadcast leading shape of resolved sampling and centre arrays."""
        values = [zdx.resolve(value) for value in (self.d, self.c) if value is not None]
        shapes = [dlu.as_axis(value, self.ndim).shape[:-1] for value in values]
        return np.broadcast_shapes(*shapes)

    @property
    def d_si(self) -> Array:
        """Pixel spacing in canonical SI units."""
        if self.d is None:
            raise ValueError("d must be specified to calculate spacing.")
        return dlu.as_axis(zdx.resolve(self.d), self.ndim, "d") * self.scale

    @property
    def c_si(self) -> Array:
        """Grid centre in SI units, with an omitted centre interpreted as zero."""
        if self.c is None:
            return np.zeros(self.ndim)
        return dlu.as_axis(zdx.resolve(self.c), self.ndim, "c") * self.scale

    def resize_samples(self, n) -> GridSpec:
        """Track central padding/cropping, retaining the original pixel positions.

        An unmatched sample is added or removed on the upper side. Unlike
        :meth:`resize`, changing parity therefore shifts the centre by half a pixel.
        """
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to resize sampled coordinates.")
        n = dlu.as_size(n, self.ndim, "n")
        shifts = tuple(
            (1 if new >= old else -1) * (abs(new - old) % 2) / 2
            for old, new in zip(self.n, n)
        )
        if not any(shifts):
            return self.resize(n)
        centre = 0 if self.c is None else self.c
        return self.set(n=n, c=centre + self.d * np.asarray(shifts))

    def frequency(self, n=None) -> GridSpec:
        """Return the centred FFT dual grid in reciprocal SI units.

        ``n`` optionally selects working counts. Spatial centres do not change
        the frequency sampling. Even FFT lengths place its centre half a bin below
        zero. This constructs metadata, not coordinate or phase arrays.
        """
        grid = self if n is None else self.resize(n)
        if grid.n is None:
            raise ValueError("n must be specified to calculate FFT sampling.")
        d = 1 / (np.asarray(grid.n) * grid.d_si)
        c = np.where(np.asarray(grid.n) % 2 == 0, -d / 2, 0.0)
        unit = None
        if grid.unit is not None:
            info = zdx.unit_info(grid.unit)
            if info.category != "cartesian" or info.power not in (-1, 1):
                raise ValueError(
                    "FFT frequency grids require length or reciprocal-length units."
                )
            unit = "1/m" if info.power == 1 else "m"
        return type(self)(grid.n, d, c, unit, indexing=self.indexing)

    def downsample(self, factors) -> GridSpec:
        """Return sampling metadata after integer spatial downsampling.

        Sample counts are divided by ``factors`` and pixel scales are multiplied by
        them, preserving the field of view and centre. Values follow physical-axis
        order.
        """
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to downsample a GridSpec.")
        factors = dlu.as_size(factors, self.ndim, "factors")
        n = tuple(n // factor for n, factor in zip(self.n, factors))
        return self.set(n=n, d=self.d * np.asarray(factors))

    def oversample(self, factors) -> GridSpec:
        """Return sampling metadata after integer spatial oversampling.

        Sample counts are multiplied by ``factors`` and pixel scales are divided by
        them, preserving the field of view and centre. Values follow physical-axis
        order.
        """
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to oversample a GridSpec.")
        factors = dlu.as_size(factors, self.ndim, "factors")
        n = tuple(n * factor for n, factor in zip(self.n, factors))
        return self.set(n=n, d=self.d / np.asarray(factors))

    def resample(self, n, d) -> GridSpec:
        """Set new sample counts and pixel scales in the grid's stored unit.

        Both values follow physical-axis order. The grid centre is retained, while
        the field of view may change.
        """
        n = dlu.as_size(n, self.ndim, "n")
        return self.set(n=n, d=dlu.as_axis(d, self.ndim, "d"))

    @classmethod
    def from_axes(cls, axes, unit=None, *, indexing="xy", alias=None) -> GridSpec:
        """Construct a regular grid from physical coordinate axes.

        Parameters
        ----------
        axes : tuple[Array, ...]
            Regularly sampled coordinate axes in physical-axis order and SI units.
            Regular sampling is required but not validated. Each axis must contain
            at least two samples. Leading batch dimensions are preserved.
        unit : str or None
            Unit used to store the recovered pixel scales and centers.
        indexing : {"xy", "ij"}
            Sample-axis ordering of the resulting coordinate mesh.
        alias : mapping or sequence of pairs, optional
            Zodiax parameter-path aliases for the recovered grid.
        """
        # Validate coordinate vectors and recover physical-axis sample counts
        axes = tuple(np.asarray(axis) for axis in axes)
        if not axes:
            raise ValueError("axes must contain at least one coordinate vector.")
        if any(axis.ndim == 0 or axis.shape[-1] < 2 for axis in axes):
            raise ValueError("Every coordinate axis must contain at least two samples.")
        n = tuple(axis.shape[-1] for axis in axes)

        # Recover pixel scales and centres with shared metadata broadcasting
        spacing = [axis[..., 1] - axis[..., 0] for axis in axes]
        centers = [(axis[..., -1] + axis[..., 0]) / 2 for axis in axes]
        d = np.stack(np.broadcast_arrays(*spacing), -1)
        c = np.stack(np.broadcast_arrays(*centers), -1)

        # Convert from SI coordinates into the requested output unit
        scale = 1.0 if unit is None else dlu.unit_factor(unit)
        return cls(
            n=n, d=d / scale, c=c / scale, unit=unit, indexing=indexing, alias=alias
        )

    def build(self, builder, **kwargs):
        """Evaluate a compatible builder on this sampling specification.

        Delegates to ``builder.build(self, **kwargs)`` and returns the builder-defined
        sampled arrays or objects.
        """
        return builder.build(self, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return sample counts in the declared array-axis order.

        A physical ``n=(nx, ny)`` grid returns ``(ny, nx)`` with default ``xy``
        indexing, or ``(nx, ny)`` with ``ij`` indexing.
        """
        if self.n is None:
            raise ValueError("n must be specified to calculate shape.")
        return self.n[::-1] if self.indexing == "xy" else self.n

    @property
    def scale(self) -> float:
        """Return the multiplicative conversion from the declared unit to SI."""
        return 1.0 if self.unit is None else dlu.unit_factor(self.unit)

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return one SI pixel-centre vector per physical axis.

        This is the descriptive alias of `xs`.
        """
        return self.xs

    @property
    def xs(self) -> tuple[Array, ...]:
        """Return one SI pixel-centre vector per physical axis.

        The tuple follows physical ``(x, y, ...)`` order and preserves leading batch
        axes carried by ``d`` or ``c``.
        """
        if self.n is None:
            raise ValueError("n must be specified to calculate xs.")
        return self._xs_for(self.n)

    def axes_for(self, n: tuple[int, ...]) -> tuple[Array, ...]:
        """Return SI-valued coordinate axes for explicit sample counts.

        ``n`` follows physical-axis order and must match the grid dimensionality. The
        returned tuple contains one regularly sampled vector per physical axis; the
        grid itself is unchanged.
        """
        return self._xs_for(n)

    def _xs_for(self, n: tuple[int, ...]) -> tuple[Array, ...]:
        """Generate SI coordinate vectors with broadcast sampling metadata."""
        if self.d is None:
            raise ValueError("d must be specified to generate coordinates.")
        n = dlu.as_size(n, name="n")
        if len(n) != self.ndim:
            raise ValueError("n dimensionality must match the coordinate grid.")

        # Share every leading metadata axis across the physical coordinate vectors
        d, c = np.broadcast_arrays(self.d_si, self.c_si)
        return tuple(
            c[..., i, None] + (np.arange(size) - (size - 1) / 2) * d[..., i, None]
            for i, size in enumerate(n)
        )

    @property
    def coordinates(self) -> Array:
        """Return SI coordinates with shape ``(..., ndim, *shape)``.

        The component axis follows physical ``(x, y, ...)`` order while final sampled
        axes follow the declared indexing. Default ``xy`` reverses their order;
        ``ij`` retains it. A one-dimensional grid has shape ``(..., 1, n)``.
        """
        if self.n is None:
            raise ValueError("n must be specified to calculate coordinates.")
        return self._coordinates_for(self.n)

    def transformed(self, transform=None) -> Array:
        """Return 2D SI coordinates after applying an optional transform.

        The result has shape ``(..., 2, ny, nx)``. Coordinate components use
        physical ``(x, y)`` order and sampled axes use NumPy ``(y, x)`` order.
        """
        if transform is None:
            return self.coordinates
        if not isinstance(transform, BaseCoordTransform):
            raise TypeError("transform must be a BaseCoordTransform or None.")
        if self.ndim != 2:
            raise ValueError(
                "BaseCoordTransform currently supports only 2D GridSpec objects."
            )
        return transform(self.coordinates)

    def _coordinates_for(self, n: tuple[int, ...]) -> Array:
        """Broadcast coordinate vectors into the declared sampled-axis order."""
        axes = self._xs_for(n)
        batch_shape = axes[0].shape[:-1]
        shape = tuple(n[::-1]) if self.indexing == "xy" else tuple(n)

        # Place each physical component along its corresponding sampled axis
        components = []
        for i, axis in enumerate(axes):
            position = self.ndim - 1 - i if self.indexing == "xy" else i
            spatial_shape = [1] * self.ndim
            spatial_shape[position] = n[i]
            component = axis.reshape(batch_shape + tuple(spatial_shape))
            components.append(np.broadcast_to(component, batch_shape + shape))
        return np.stack(components, axis=len(batch_shape))

    @property
    def fov(self):
        """Return ``n * d`` in the grid's declared unit and physical-axis order."""
        if self.n is None or self.d is None:
            raise ValueError("n and d must be specified to calculate fov.")
        return np.asarray(self.n) * self.d

    def extent(self, ndim=None, unit=None):
        """Return plot-ready grid edges in the declared output unit.

        The result may optionally be expanded to ``ndim`` axes. Passing ``ndim=2``
        makes a scalar square-grid specification directly compatible with the
        ``extent`` argument of ``matplotlib.pyplot.imshow``. Passing ``unit``
        converts the result from the grid's declared unit into the requested unit.
        """
        ndim = self.ndim if ndim is None else int(ndim)
        if ndim < self.ndim:
            raise ValueError("ndim cannot be smaller than the grid dimensionality.")
        grid = self if ndim == self.ndim else self.broadcast(ndim)
        half_width = grid.fov / 2
        center = np.zeros(ndim) if grid.c is None else grid.c
        extent = np.stack((center - half_width, center + half_width), axis=-1)
        extent = extent.reshape(extent.shape[:-2] + (2 * ndim,))
        if unit is None:
            return extent
        return extent * grid.scale / dlu.unit_factor(unit)


Grid = GridSpec


class BaseCoordTransform(zdx.Module):
    """Extension contract for transforms of ``(..., 2, ny, nx)`` coordinates.

    Leading transform and coordinate dimensions use paired JAX broadcasting. An
    unbatched coordinate field may therefore be expanded by batched transform
    parameters, while matching batches are transformed element-by-element. Subclasses
    preserve the physical ``(x, y)`` component axis and final spatial axes. A
    transform changes where an object is evaluated; it does not resample an array.
    """

    @staticmethod
    def get_coordinates(coordinates) -> Array:
        """Validate and return a Cartesian coordinate array.

        ``coordinates`` must have shape ``(..., 2, ny, nx)`` with its physical
        ``(x, y)`` component axis immediately before the spatial axes. Values are not
        converted or copied beyond standard JAX array coercion.
        """
        if coordinates is None:
            raise ValueError("Provide coordinates when calling the transformation.")

        coordinates = zdx.as_array(coordinates, dtype=float)
        if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
            raise ValueError("coordinates must have shape (..., 2, ny, nx).")

        return coordinates

    @abstractmethod
    def __call__(self, coordinates: Array) -> Array:
        """Transform Cartesian coordinates with shape ``(..., 2, ny, nx)``.

        Subclasses return the same coordinate convention. Leading transform axes are
        paired with matching leading coordinate axes; singleton axes broadcast by
        ordinary JAX rules.
        """

    def apply(self, coordinates: Array) -> Array:
        """Apply the transform through its deprecated method alias.

        ``coordinates`` and the returned array follow ``(..., 2, ny, nx)``. Use
        ``transform(coordinates)`` in new code; this alias is removed in dLux 0.17.
        """
        warnings.warn(
            "The `.apply()` method is deprecated and will be removed in dLux "
            "0.17.0. Use `transform(coordinates)` instead: "
            "`transform.apply(coordinates)` -> `transform(coordinates)`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(coordinates)


class TransformChain(BaseCoordTransform):
    """Apply an ordered collection of coordinate transformations.

    Examples
    --------
    Apply an ordered rotation and polynomial distortion to a grid:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct an ordered rotation and polynomial distortion
    transform = dl.TransformChain(
        [
            ("rotation", dl.Affine(rotation=np.deg2rad(15))),
            ("distortion", dl.Distortion(order=3)),
        ]
    )

    # Apply the complete transformation chain to a grid
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    coordinates = grid.transformed(transform)
    ```

    Parameters
    ----------
    transformations : sequence or dict
        Named or unnamed ``BaseCoordTransform`` objects in application order.
    """

    transformations: dict

    def __init__(self, transformations=(), *, alias=None):
        """Initialise an ordered coordinate-transformation chain.

        Parameters
        ----------
        transformations : mapping or sequence
            Named mapping, ``(name, transform)`` sequence, or transform sequence in
            application order. Every value must derive from `BaseCoordTransform`.
        """
        if isinstance(transformations, dict):
            transformations = list(transformations.items())
        else:
            transformations = list(transformations)
        self.transformations = dlu.list2dictionary(
            transformations, True, BaseCoordTransform
        )

        self.alias = alias

    def __call__(self, coords: Array) -> Array:
        """Apply each coordinate transformation in insertion order."""
        coords = self.get_coordinates(coords)
        for transformation in self.transformations.values():
            coords = transformation(coords)
        return coords


class Distortion(BaseCoordTransform):
    """Apply a polynomial distortion to Cartesian coordinates.

    Polynomial coefficients have shape ``(2, n_terms)`` for output ``x`` and
    ``y``. Leading coefficient axes vectorise independent distortions; matching
    coordinate batches are transformed element-by-element.

    Examples
    --------
    Construct and apply a shift-invariant polynomial distortion:

    ```python
    import dLux as dl

    # Construct a third-order shift-invariant polynomial distortion
    transform = dl.Distortion(order=3, shift_invariant=True)

    # Apply the distortion to a grid
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    coordinates = grid.transformed(transform)

    # Inspect the polynomial terms and their coefficients
    powers = transform.powers
    coeffs = transform.distortion
    ```

    Parameters
    ----------
    order : int or None
        Maximum total polynomial order.
    distortion : Array or None
        Distortion coefficients with trailing shape matching ``powers``.
    orders : sequence[int] or None
        Explicit total polynomial orders, mutually exclusive with ``order``.
    powers : Array or None
        Non-negative integer powers with shape ``(2, n_terms)``.
    shift_invariant : bool
        Remove the linear coordinate terms from the parameterisation.
    """

    powers: Array
    distortion: Array
    shift_invariant: bool

    def __init__(
        self,
        order: int | None = None,
        distortion: Array | None = None,
        *,
        orders: tuple[int, ...] | list[int] | None = None,
        powers: Array | None = None,
        shift_invariant: bool = False,
        alias=None,
    ):
        """Initialise a polynomial coordinate distortion.

        Parameters
        ----------
        order : int or None
            Maximum total order, mutually exclusive with ``orders`` and ``powers``.
        distortion : Array or None
            Coefficients with trailing shape ``(2, n_terms)``. Defaults to zeros.
        orders : sequence[int] or None
            Selected positive total orders.
        powers : Array or None
            Explicit exponents with shape ``(2, n_terms)``.
        shift_invariant : bool
            Remove linear coordinate terms from generated powers.
        """
        self.shift_invariant = bool(shift_invariant)
        self.powers = _distortion_powers(order, orders, powers, self.shift_invariant)
        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = zdx.as_array(distortion, dtype=float)
        if distortion.shape[-2:] != self.powers.shape:
            raise ValueError("distortion trailing dimensions must match powers shape.")
        self.distortion = distortion

        self.alias = alias

    def __call__(self, coords: Array) -> Array:
        """Apply the configured polynomial coordinate distortion."""
        coords = self.get_coordinates(coords)
        if self.distortion.ndim > 2:
            apply = lambda distortion, coordinates: dlu.distort_coords(
                coordinates, distortion, self.powers
            )
            if coords.ndim > 3 and coords.shape[0] == self.distortion.shape[0]:
                return vmap(apply)(self.distortion, coords)
            return vmap(lambda distortion: apply(distortion, coords))(self.distortion)
        return dlu.distort_coords(coords, self.distortion, self.powers)


class AffineMap(BaseCoordTransform):
    """Apply a direct affine coordinate map ``x' = matrix @ x + offset``.

    Matrix, offset, and coordinate leading dimensions use paired JAX broadcasting.
    This is the low-level direct representation; use `Affine` for semantic
    translation, rotation, scale, and shear parameters.

    Parameters
    ----------
    matrix : Array or None
        Matrix with trailing shape ``(2, 2)``. Defaults to identity.
    offset : Array or None
        Offset with trailing shape ``(2,)``. Defaults to zero.
    """

    matrix: Array
    offset: Array

    def __init__(self, matrix=None, offset=None, *, alias=None):
        """Initialise a direct affine map.

        Parameters
        ----------
        matrix : Array or None
            Matrix with trailing shape ``(2, 2)``; defaults to identity.
        offset : Array or None
            Translation with trailing shape ``(2,)``; defaults to zero.
        """
        matrix = np.eye(2) if matrix is None else zdx.as_array(matrix, dtype=float)
        offset = np.zeros(2) if offset is None else zdx.as_array(offset, dtype=float)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must have trailing shape (2, 2).")
        if offset.shape[-1:] != (2,):
            raise ValueError("offset must have trailing shape (2,).")
        self.matrix = matrix
        self.offset = offset

        self.alias = alias

    def __call__(self, coords: Array) -> Array:
        """Apply the direct affine matrix and offset to coordinates."""
        coords = self.get_coordinates(coords)
        shift = self.offset[..., :, None, None]
        return np.einsum("...ij,...jxy->...ixy", self.matrix, coords) + shift


class Affine(BaseCoordTransform):
    """An affine coordinate transform with semantic parameters.

    Translation, rotation, scale, and shear map coordinates into a transformed
    object's local frame. Operations are composed in the order supplied by ``order``.
    Parameter and coordinate leading dimensions use paired JAX broadcasting.

    Examples
    --------
    Transform a grid using semantic affine parameters:

    ```python
    import jax.numpy as np

    import dLux as dl

    # Construct a semantic affine transformation
    transform = dl.Affine(
        translation=[0.1, -0.1],
        rotation=np.deg2rad(15),
        scale=[1.0, 0.8],
    )

    # Apply the transformation to grid coordinates
    grid = dl.GridSpec(n=128, diam=1.0, unit="m")
    coordinates = grid.transformed(transform)

    # Access the equivalent affine matrix and offset
    matrix, offset = transform.coeffs
    ```

    Parameters
    ----------
    translation : ArrayLike or None
        Two-dimensional physical translation.
    rotation : ArrayLike or None
        Counter-clockwise rotation in radians.
    scale : ArrayLike or None
        Scalar or two-dimensional coordinate scale.
    shear : ArrayLike or None
        Two-dimensional shear coefficients.
    order : tuple[str, ...]
        Order in which the supplied transformations are composed.
    """

    translation: Array | None = None
    rotation: Array | None = None
    scale: Array | None = None
    shear: Array | None = None
    order: tuple[str, ...]

    def __init__(
        self,
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        order=("translation", "rotation", "scale", "shear"),
        *,
        alias=None,
    ):
        """Initialise a semantic affine coordinate transformation.

        Parameters
        ----------
        translation : ArrayLike or None
            Physical ``(x, y)`` translation into the object's local frame.
        rotation : ArrayLike or None
            Scalar or vectorised counter-clockwise angle in radians.
        scale : ArrayLike or None
            Non-zero scalar or ``(x, y)`` coordinate scale.
        shear : ArrayLike or None
            Two-component shear.
        order : tuple[str, ...]
            Unique composition order drawn from translation, rotation, scale, and
            shear.
        """
        self.translation = self._vector(translation, "translation")
        self.rotation = zdx.as_array(rotation, dtype=float)

        if self.rotation is not None and self.rotation.ndim > 1:
            raise ValueError("rotation must be scalar or one-dimensional.")

        self.scale = None

        if scale is not None:
            self.scale = dlu.as_axis(scale, 2, "scale")

            if np.any(self.scale == 0):
                raise ValueError("scale values must be non-zero.")

        self.shear = self._vector(shear, "shear")

        valid = ("translation", "rotation", "scale", "shear")
        self.order = tuple(order)
        if len(set(self.order)) != len(self.order) or not set(self.order) <= set(valid):
            raise ValueError(f"order entries must be unique values from {valid}.")

        self.alias = alias

    @staticmethod
    def _vector(value, name):
        if value is None:
            return None
        value = zdx.as_array(value, dtype=float)
        if value.shape[-1:] != (2,):
            raise ValueError(f"{name} must have trailing shape (2,).")
        return value

    def _matrices(self) -> Array:
        """Return all affine components as ordered homogeneous matrices."""
        # Resolve the shared parameter batch shape
        shapes = []
        for value in (self.translation, self.scale, self.shear):
            if value is not None:
                shapes.append(value.shape[:-1])
        if self.rotation is not None:
            shapes.append(self.rotation.shape)
        batch = np.broadcast_shapes(*shapes) if shapes else ()

        # Initialise every optional component to the batched identity
        identity = np.broadcast_to(np.eye(3), batch + (3, 3))

        translation = identity
        if self.translation is not None:
            value = np.broadcast_to(self.translation, batch + (2,))
            translation = identity.at[..., :2, 2].set(-value)

        # Construct the inverse rotation into the local coordinate frame
        rotation = identity
        if self.rotation is not None:
            angle = np.broadcast_to(self.rotation, batch)
            cosine, sine = np.cos(angle), np.sin(angle)
            rotation = rotation.at[..., 0, 0].set(cosine)
            rotation = rotation.at[..., 0, 1].set(-sine)
            rotation = rotation.at[..., 1, 0].set(sine)
            rotation = rotation.at[..., 1, 1].set(cosine)

        # Construct scale and shear components
        scale = identity
        if self.scale is not None:
            value = np.broadcast_to(self.scale, batch + (2,))
            scale = scale.at[..., 0, 0].set(1 / value[..., 0])
            scale = scale.at[..., 1, 1].set(1 / value[..., 1])

        shear = identity
        if self.shear is not None:
            value = np.broadcast_to(self.shear, batch + (2,))
            shear = shear.at[..., 0, 1].set(value[..., 0])
            shear = shear.at[..., 1, 0].set(value[..., 1])

        # Select the configured component order
        matrices = np.stack((translation, rotation, scale, shear))
        indices = np.array(
            tuple(
                ("translation", "rotation", "scale", "shear").index(name)
                for name in self.order
            )
        )
        return matrices[indices]

    @property
    def coeffs(self) -> tuple[Array, Array]:
        """Return the composed affine matrix and offset.

        Returns
        -------
        matrix : Array
            Forward coordinate matrix with shape ``(..., 2, 2)``.
        offset : Array
            Translation vector with shape ``(..., 2)``. Leading parameter axes are
            broadcast and preserved in both outputs.
        """
        combine = lambda cumulative, operation: (operation @ cumulative, None)
        matrices = self._matrices()
        identity = np.broadcast_to(np.eye(3), matrices.shape[1:])
        homogeneous, _ = lax.scan(combine, identity, matrices)
        return homogeneous[..., :2, :2], homogeneous[..., :2, 2]

    def __call__(self, coords: Array) -> Array:
        """Apply the composed semantic affine transformation."""
        coords = self.get_coordinates(coords)
        matrix, offset = self.coeffs
        shift = offset[..., :, None, None]
        return np.einsum("...ij,...jxy->...ixy", matrix, coords) + shift
