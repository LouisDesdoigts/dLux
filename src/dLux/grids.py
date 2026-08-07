"""Coordinate specifications, transformations, and ordered composition."""

from __future__ import annotations

from abc import abstractmethod
import jax.numpy as np
import zodiax as zdx
from jax import Array, core, lax, vmap

import dLux.utils as dlu

__all__ = [
    "GridSpec",
    "ResizeSpec",
    "PasteSpec",
    "CoordTransform",
    "Affine",
    "AffineMap",
    "TransformChain",
    "DistortCoords",
]


def _distortion_powers(order, orders, powers, shift_invariant):
    """Resolve the polynomial powers used by a coordinate distortion."""
    # Validate mutually exclusive term specifications
    if sum(value is not None for value in (order, orders, powers)) > 1:
        raise ValueError("Provide only one of order, orders, or powers.")

    # Validate explicit powers or generate selected total orders
    if powers is not None:
        powers = np.asarray(powers, dtype=float)
        if powers.ndim != 2 or powers.shape[0] != 2:
            raise ValueError("powers must have shape (2, n_terms).")
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


class BaseGridSpec(zdx.Base):
    """Base class for coordinate and sampling specifications."""


class ResizeSpec(BaseGridSpec):
    """Define output sampling by an explicit size or integer resize factors.

    Parameters
    ----------
    n : int, tuple[int, ...], or None
        Explicit output sizes in physical-axis order. Mutually exclusive with
        non-unit ``pad`` or ``crop``.
    pad, crop : int or tuple[int, ...]
        Integer factors applied before and after an operation.
    c : Array or None
        Optional output center in the associated propagation unit.
    """

    n: tuple[int, ...] | None
    pad: tuple[int, ...]
    crop: tuple[int, ...]
    c: Array | None

    def __init__(self, n=None, pad=1, crop=1, c=None):
        if n is not None and (pad != 1 or crop != 1):
            raise ValueError("Specify either n or pad/crop factors, not both.")
        self.n = None if n is None else dlu.as_size(n, name="n")
        self.pad = dlu.as_size(pad, name="pad")
        self.crop = dlu.as_size(crop, name="crop")
        self.c = dlu.to_value(c, optional=True)

    def broadcast(self, ndim: int) -> BaseGridSpec:
        """Broadcast sizes and factors to ``ndim`` dimensions."""
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
        """Whether this specification defines an absolute output size."""
        return self.n is not None

    @property
    def padding(self) -> dict:
        """Return keyword arguments for FFT propagation utilities."""
        if self.explicit:
            return {"pad_to": self.n}
        return {"pad": dlu.as_size(self.pad, 2, "pad")}

    def output_size(self, shape) -> tuple[int, ...]:
        """Return the requested physical-axis size for an input array shape."""
        if self.explicit:
            return self.n
        ndim = max(len(self.pad), len(self.crop), 2)
        pad = dlu.as_size(self.pad, ndim, "pad")
        crop = dlu.as_size(self.crop, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(size * pad // crop for size, pad, crop in zip(sizes, pad, crop))

    def crop_size(self, shape) -> tuple[int, ...]:
        """Return the size after applying only the crop factors."""
        if self.explicit:
            return self.n
        ndim = max(len(self.crop), 2)
        factors = dlu.as_size(self.crop, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(size // factor for size, factor in zip(sizes, factors))

    def pad_array(self, array: Array, fill: float = 0.0) -> Array:
        """Pad an array using this specification."""
        if self.explicit:
            return dlu.pad_to(array, self.n, fill)
        ndim = max(len(self.pad), 2)
        factors = dlu.as_size(self.pad, ndim, "pad")
        sizes = tuple(array.shape[-ndim:][::-1])
        target = tuple(size * factor for size, factor in zip(sizes, factors))
        return dlu.pad_to(array, target, fill)

    def crop_array(self, array: Array) -> Array:
        """Crop an array using this specification."""
        if self.explicit:
            return dlu.crop_to(array, self.n)
        return dlu.crop_to(array, self.crop_size(array.shape))

    def crop_axes(self, axes: tuple[Array, ...]) -> tuple[Array, ...]:
        """Centrally crop physical coordinate axes to this output sampling."""
        if self.explicit:
            sizes = self.n
        else:
            factors = dlu.as_size(self.crop, len(axes), "crop")
            sizes = tuple(a.shape[-1] // f for a, f in zip(axes, factors))

        crop_fn = lambda a, n: dlu.crop_to(a, (n,))
        return tuple(crop_fn(a, n) for a, n in zip(axes, sizes))

    def resize(self, array: Array, fill: float = 0.0) -> Array:
        """Resize an array to the final sampling represented by this object."""
        return dlu.resize(array, self.output_size(array.shape), fill)


class PasteSpec(BaseGridSpec):
    """Map compact local arrays into a larger regularly sampled grid.

    The specification stores integer ``(x, y)`` starts and residual physical offsets
    for equally sized stamps. Construction is deliberately separate from evaluation:
    stamp and output shapes are fixed before JIT compilation, while coordinates and
    pasted values remain ordinary JAX calculations.
    """

    n: tuple[int, int]
    shape: tuple[int, int]
    starts: Array
    offsets: Array
    d: Array

    def __init__(self, n, shape, starts, offsets, d):
        self.n = dlu.as_size(n, 2, "n")
        self.shape = dlu.as_size(shape, 2, "shape")
        self.starts = dlu.to_value(starts, int)
        self.offsets = dlu.to_value(offsets)
        self.d = dlu.as_axis(d, 2, "d")

        if self.starts.ndim != 2 or self.starts.shape[-1] != 2:
            raise ValueError("starts must have shape (n_stamps, 2).")
        if self.offsets.shape != self.starts.shape:
            raise ValueError("offsets must match the shape of starts.")

        ends = self.starts + np.asarray(self.shape)
        if np.any(self.starts < 0) or np.any(ends > np.asarray(self.n)):
            raise ValueError("Every stamp must lie within the output grid.")

    @classmethod
    def from_grid(cls, grid, centers, extent):
        """Construct compact stamp placement from a concrete two-dimensional grid."""
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        if grid.n is None or grid.d is None or grid.ndim != 2:
            raise ValueError("grid must define two-dimensional n and d values.")

        centers = dlu.to_value(centers)
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
        """Return one local coordinate grid per stamp."""
        coordinates = dlu.nd_coords(self.shape, self.d)
        return coordinates[None] + self.offsets[:, :, None, None]

    def paste(self, arrays, method="scan") -> Array:
        """Add compact stamp arrays into the common output grid."""
        return dlu.paste(arrays, self.starts, self.n, method)

    def extract(self, array) -> Array:
        """Extract one compact stamp from the output grid at every start."""
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
    """A complete regularly sampled Cartesian coordinate grid.

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
    """

    n: tuple[int, ...] | None
    d: Array | None
    c: Array | None
    unit: str | None

    def __init__(self, n=None, d=None, c=None, unit=None, diam=None):
        # Validate mutually dependent sampling inputs
        if d is not None and diam is not None:
            raise ValueError("Provide only one of d or diam.")
        if diam is not None and n is None:
            raise ValueError("n must be provided with diam.")

        # Infer the physical dimensionality from non-scalar inputs
        values = [value for value in (n, d, c, diam) if value is not None]
        lengths = [
            np.asarray(value).shape[-1]
            for value in values
            if np.asarray(value).ndim >= 1
        ]
        ndim = max(lengths, default=1 if values else 0)

        # Resolve sizes and optional diameter-based sampling
        self.n = None if n is None else dlu.as_size(n, ndim, "n")

        if diam is not None:
            diam = dlu.as_axis(diam, ndim, "diam")
            d = diam / np.asarray(self.n)

        # Standardize sampling and center arrays
        self.d = dlu.as_axis(d, ndim, "d")
        self.c = dlu.as_axis(c, ndim, "c")

        # Validate concrete positive pixel scales
        if (
            self.d is not None
            and not isinstance(self.d, core.Tracer)
            and np.any(self.d <= 0)
        ):
            raise ValueError("d must contain positive values.")

        # Validate and store the declared coordinate unit
        self.unit = None if unit is None else self._validate_unit(unit)

    @staticmethod
    def _validate_unit(unit):
        if not isinstance(unit, str):
            raise TypeError("unit must be a string.")
        unit = unit.strip()
        if not unit:
            raise ValueError("unit cannot be empty.")
        dlu.unit_factor(unit)
        return unit

    @property
    def ndim(self) -> int:
        """Return the number of coordinate dimensions."""
        for value in (self.n, self.d, self.c):
            if value is not None:
                return len(value) if isinstance(value, tuple) else value.shape[-1]
        return 0

    def broadcast(self, ndim: int) -> GridSpec:
        """Broadcast scalar or one-axis leaves to ``ndim`` dimensions."""
        ndim = int(ndim)
        if ndim < 1:
            raise ValueError("ndim must be a positive integer.")
        return self.set(
            n=None if self.n is None else dlu.as_size(self.n, ndim, "n"),
            d=dlu.as_axis(self.d, ndim, "d"),
            c=dlu.as_axis(self.c, ndim, "c"),
        )

    def resize(self, n) -> GridSpec:
        """Change the grid size while retaining its sampling."""
        return self.set(n=dlu.as_size(n, self.ndim, "n"))

    def downsample(self, factors) -> GridSpec:
        """Update sampling metadata after downsampling by integer factors."""
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to downsample a GridSpec.")
        factors = dlu.as_size(factors, self.ndim, "factors")
        n = tuple(n // factor for n, factor in zip(self.n, factors))
        return self.set(n=n, d=self.d * np.asarray(factors))

    def oversample(self, factors) -> GridSpec:
        """Increase pixel counts while preserving field of view and grid centre."""
        if self.n is None or self.d is None:
            raise ValueError("n and d are required to oversample a GridSpec.")
        factors = dlu.as_size(factors, self.ndim, "factors")
        n = tuple(n * factor for n, factor in zip(self.n, factors))
        return self.set(n=n, d=self.d / np.asarray(factors))

    def resample(self, n, d) -> GridSpec:
        """Set a new grid size and per-axis sampling."""
        n = dlu.as_size(n, self.ndim, "n")
        return self.set(n=n, d=dlu.as_axis(d, self.ndim, "d"))

    @classmethod
    def from_axes(cls, axes, unit=None) -> GridSpec:
        """Construct a regular grid from physical coordinate axes.

        Parameters
        ----------
        axes : tuple[Array, ...]
            Regularly sampled coordinate axes in physical-axis order and SI units.
            Regular sampling is required but not validated. Each axis must contain
            at least two samples. Leading batch dimensions are preserved.
        unit : str or None
            Unit used to store the recovered pixel scales and centers.
        """
        # Recover the pixel counts in physical-axis order
        axes = tuple(axes)
        n = tuple(axis.shape[-1] for axis in axes)

        # Recover the pixel scales and centers
        d = np.stack([axis[..., 1] - axis[..., 0] for axis in axes], -1)
        c = np.stack([(axis[..., -1] + axis[..., 0]) / 2 for axis in axes], -1)

        # Convert from SI coordinates into the requested output unit
        scale = 1.0 if unit is None else dlu.unit_factor(unit)
        return cls(n=n, d=d / scale, c=c / scale, unit=unit)

    def build(self, builder, **kwargs):
        """Evaluate a ``GridBuilder`` on this sampling specification."""
        from .builders import GridBuilder

        if not isinstance(builder, GridBuilder):
            raise TypeError("builder must be a GridBuilder.")
        return builder.build(self, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array shape associated with this coordinate grid."""
        if self.n is None:
            raise ValueError("n must be specified to calculate shape.")
        return self.n[::-1]

    @property
    def scale(self) -> float:
        """Return the factor converting coordinate values to canonical SI units."""
        return 1.0 if self.unit is None else dlu.unit_factor(self.unit)

    @property
    def axes(self) -> tuple[Array, ...]:
        """Alias for the one-dimensional coordinate axes."""
        return self.xs

    @property
    def xs(self) -> tuple[Array, ...]:
        """Return one pixel-centre coordinate vector per physical axis."""
        if self.n is None:
            raise ValueError("n must be specified to calculate xs.")
        return self.xs_for(self.n)

    def axes_for(self, n: tuple[int, ...]) -> tuple[Array, ...]:
        """Alias for coordinate axes at concrete physical-axis pixel counts."""
        return self.xs_for(n)

    def xs_for(self, n: tuple[int, ...]) -> tuple[Array, ...]:
        """Return coordinate axes for concrete physical-axis pixel counts."""
        # Validate the requested physical-axis sizes
        if self.d is None:
            raise ValueError("d must be specified to calculate xs.")
        if len(n) != self.ndim:
            raise ValueError("n dimensionality must match the coordinate spec.")

        # Broadcast sampling and centers over their leading dimensions
        batch = self.d.shape[:-1]
        if self.c is not None:
            batch = np.broadcast_shapes(batch, self.c.shape[:-1])
        spacing = np.broadcast_to(self.d, batch + (self.ndim,))
        center = (
            np.zeros(batch + (self.ndim,))
            if self.c is None
            else np.broadcast_to(self.c, batch + (self.ndim,))
        )

        # Generate one SI-valued coordinate vector per physical axis
        return tuple(
            (
                center[..., i, None]
                + (np.arange(size) - (size - 1) / 2) * spacing[..., i, None]
            )
            * self.scale
            for i, size in enumerate(n)
        )

    @property
    def coordinates(self) -> Array:
        """Return the full coordinate array with shape ``(ndim, *shape)``."""
        if self.n is None:
            raise ValueError("n must be specified to calculate coordinates.")
        return self.coordinates_for(self.n)

    def transformed(self, transform=None) -> Array:
        """Return coordinates after applying an optional coordinate transform."""
        if transform is None:
            return self.coordinates
        if not isinstance(transform, CoordTransform):
            raise TypeError("transform must be a CoordTransform or None.")
        return transform(self.coordinates)

    def coordinates_for(self, n: tuple[int, ...]) -> Array:
        """Return full coordinates for concrete physical-axis pixel counts."""
        # Resolve the broadcast batch and output spatial shapes
        batch = self.d.shape[:-1]
        if self.c is not None:
            batch = np.broadcast_shapes(batch, self.c.shape[:-1])
        spacing = np.broadcast_to(self.d, batch + (self.ndim,))
        shape = tuple(n[::-1])

        # Generate and broadcast each physical coordinate axis
        axes = []
        for i, size in enumerate(n):
            axis = (np.arange(size) - (size - 1) / 2) * spacing[..., i, None]
            spatial_shape = [1] * self.ndim
            spatial_shape[self.ndim - i - 1] = size
            axis = axis.reshape(batch + tuple(spatial_shape))
            axes.append(np.broadcast_to(axis, batch + shape))

        # Stack axes and apply the physical unit scale
        coordinates = np.stack(tuple(axes), axis=len(batch))
        if self.c is None:
            return coordinates * self.scale
        center = np.broadcast_to(self.c, batch + (self.ndim,))
        center = center.reshape(batch + (self.ndim,) + (1,) * self.ndim)
        return (coordinates + center) * self.scale

    @property
    def fov(self):
        """Return the field of view in the grid's declared output unit."""
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
        spec = self if ndim == self.ndim else self.broadcast(ndim)
        half_width = spec.fov / 2
        center = np.zeros(ndim) if spec.c is None else spec.c
        extent = np.stack((center - half_width, center + half_width), axis=-1).reshape(
            center.shape[:-1] + (2 * ndim,)
        )
        if unit is None:
            return extent
        return extent * spec.scale / dlu.unit_factor(unit)


class CoordTransform(zdx.Base):
    """Base class for transformations applied to coordinate fields."""

    @staticmethod
    def get_coordinates(coordinates) -> Array:
        """Validate and return a Cartesian coordinate array."""
        if coordinates is None:
            raise ValueError("Provide coordinates when calling the transformation.")

        coordinates = dlu.to_value(coordinates)
        if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
            raise ValueError("coordinates must have shape (..., 2, ny, nx).")

        return coordinates

    @abstractmethod
    def __call__(self, coordinates: Array) -> Array:
        """Transform an array of Cartesian coordinates."""

    def apply(self, coordinates: Array) -> Array:
        """Backwards-compatible alias for calling the transformation."""
        return self(coordinates)


class TransformChain(CoordTransform):
    """Apply an ordered collection of coordinate transformations.

    Parameters
    ----------
    transformations : sequence or dict
        Named or unnamed ``CoordTransform`` objects in application order.
    """

    transformations: dict

    def __init__(self, transformations=()):
        if isinstance(transformations, dict):
            transformations = list(transformations.items())
        else:
            transformations = list(transformations)
        self.transformations = dlu.list2dictionary(
            transformations, True, CoordTransform
        )

    def __call__(self, coords: Array) -> Array:
        """Apply each coordinate transformation in insertion order."""
        coords = self.get_coordinates(coords)
        for transformation in self.transformations.values():
            coords = transformation(coords)
        return coords

class DistortCoords(CoordTransform):
    """Apply a polynomial distortion to Cartesian coordinates.

    Polynomial coefficients have shape ``(2, n_terms)`` for output ``x`` and
    ``y``. A leading coefficient axis vectorises independent distortions.

    Parameters
    ----------
    order : int or None
        Maximum total polynomial order.
    distortion : Array or None
        Distortion coefficients with trailing shape matching ``powers``.
    orders : sequence[int] or None
        Explicit total polynomial orders, mutually exclusive with ``order``.
    powers : Array or None
        Explicit powers with shape ``(2, n_terms)``.
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
    ):
        self.shift_invariant = bool(shift_invariant)
        self.powers = _distortion_powers(order, orders, powers, self.shift_invariant)
        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = dlu.to_value(distortion)
        if distortion.shape[-2:] != self.powers.shape:
            raise ValueError("distortion trailing dimensions must match powers shape.")
        self.distortion = distortion

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


class AffineMap(CoordTransform):
    """Apply a direct affine coordinate map ``x' = matrix @ x + offset``.

    Parameters
    ----------
    matrix : Array or None
        Matrix with trailing shape ``(2, 2)``. Defaults to identity.
    offset : Array or None
        Offset with trailing shape ``(2,)``. Defaults to zero.
    """

    matrix: Array
    offset: Array

    def __init__(self, matrix=None, offset=None):
        matrix = np.eye(2) if matrix is None else dlu.to_value(matrix)
        offset = np.zeros(2) if offset is None else dlu.to_value(offset)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must have trailing shape (2, 2).")
        if offset.shape[-1:] != (2,):
            raise ValueError("offset must have trailing shape (2,).")
        self.matrix = matrix
        self.offset = offset

    def __call__(self, coords: Array) -> Array:
        """Apply the direct affine matrix and offset to coordinates."""
        coords = self.get_coordinates(coords)
        shift = self.offset[..., :, None, None]
        return np.einsum("...ij,...jxy->...ixy", self.matrix, coords) + shift


class Affine(CoordTransform):
    """An affine coordinate transform with semantic parameters.

    Translation, rotation, scale, and shear map coordinates into a transformed
    object's local frame. Operations are composed in the order supplied by ``order``.

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

    translation: Array | None
    rotation: Array | None
    scale: Array | None
    shear: Array | None
    order: tuple[str, ...]

    def __init__(
        self,
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        order=("translation", "rotation", "scale", "shear"),
    ):
        self.translation = self._vector(translation, "translation")
        self.rotation = dlu.to_value(rotation, optional=True)

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

    @staticmethod
    def _vector(value, name):
        if value is None:
            return None
        value = dlu.to_value(value)
        if value.shape[-1:] != (2,):
            raise ValueError(f"{name} must have trailing shape (2,).")
        return value

    def _matrices(self) -> Array:
        """Return all affine components as ordered homogeneous matrices."""
        # Initialize every optional component to the identity
        identity = np.eye(3)

        translation = identity
        if self.translation is not None:
            translation = identity.at[:2, 2].set(-self.translation)

        # Construct the inverse rotation into the local coordinate frame
        rotation = identity
        if self.rotation is not None:
            cosine, sine = np.cos(self.rotation), np.sin(self.rotation)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])

        # Construct scale and shear components
        scale = identity
        if self.scale is not None:
            scale = np.diag(np.concatenate((1 / self.scale, np.ones(1))))

        shear = identity
        if self.shear is not None:
            shear = shear.at[0, 1].set(self.shear[0])
            shear = shear.at[1, 0].set(self.shear[1])

        # Select the configured component order
        matrices = np.stack((translation, rotation, scale, shear))
        indices = np.array(
            tuple(
                ("translation", "rotation", "scale", "shear").index(name)
                for name in self.order
            )
        )
        return matrices[indices]

    def coefficients(self) -> tuple[Array, Array]:
        """Return the composed transformation matrix and offset."""
        combine = lambda cumulative, operation: (operation @ cumulative, None)
        homogeneous, _ = lax.scan(combine, np.eye(3), self._matrices())
        return homogeneous[:2, :2], homogeneous[:2, 2]

    def __call__(self, coords: Array) -> Array:
        """Apply the composed semantic affine transformation."""
        coords = self.get_coordinates(coords)
        matrix, offset = self.coefficients()
        shift = offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", matrix, coords) + shift
