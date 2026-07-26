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
    "CoordTransform",
    "Affine",
    "AffineMap",
    "TransformChain",
    "DistortCoords",
]


class BaseGridSpec(zdx.Base):
    """Base class for coordinate and sampling specifications."""


class ResizeSpec(BaseGridSpec):
    """Array sampling defined by an explicit size or pad/crop factors."""

    n: tuple[int, ...] | None
    pad_factor: tuple[int, ...]
    crop_factor: tuple[int, ...]
    c: Array | None

    def __init__(self, n=None, pad=1, crop=1, c=None):
        if n is not None and (pad != 1 or crop != 1):
            raise ValueError("Specify either n or pad/crop factors, not both.")
        self.n = None if n is None else dlu.as_size(n, name="n")
        self.pad_factor = dlu.as_size(pad, name="pad")
        self.crop_factor = dlu.as_size(crop, name="crop")
        self.c = None if c is None else np.asarray(c, float)

    def broadcast(self, ndim: int) -> BaseGridSpec:
        """Broadcast sizes and factors to ``ndim`` dimensions."""
        ndim = int(ndim)
        if ndim < 1:
            raise ValueError("ndim must be a positive integer.")
        n = None if self.n is None else dlu.as_size(self.n, ndim, "n")
        return self.set(
            n=n,
            pad_factor=dlu.as_size(self.pad_factor, ndim, "pad"),
            crop_factor=dlu.as_size(self.crop_factor, ndim, "crop"),
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
        return {"pad": dlu.as_size(self.pad_factor, 2, "pad")}

    def output_size(self, shape) -> tuple[int, ...]:
        """Return the requested physical-axis size for an input array shape."""
        if self.explicit:
            return self.n
        ndim = max(len(self.pad_factor), len(self.crop_factor), 2)
        pad_factor = dlu.as_size(self.pad_factor, ndim, "pad")
        crop_factor = dlu.as_size(self.crop_factor, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(
            size * pad // crop
            for size, pad, crop in zip(sizes, pad_factor, crop_factor)
        )

    def crop_size(self, shape) -> tuple[int, ...]:
        """Return the size after applying only the crop factors."""
        if self.explicit:
            return self.n
        ndim = max(len(self.crop_factor), 2)
        factors = dlu.as_size(self.crop_factor, ndim, "crop")
        sizes = tuple(shape[-ndim:][::-1])
        return tuple(size // factor for size, factor in zip(sizes, factors))

    def pad(self, array: Array, fill: float = 0.0) -> Array:
        """Pad an array using this specification."""
        if self.explicit:
            return dlu.pad_to(array, self.n, fill)
        ndim = max(len(self.pad_factor), 2)
        factors = dlu.as_size(self.pad_factor, ndim, "pad")
        sizes = tuple(array.shape[-ndim:][::-1])
        target = tuple(size * factor for size, factor in zip(sizes, factors))
        return dlu.pad_to(array, target, fill)

    def crop(self, array: Array) -> Array:
        """Crop an array using this specification."""
        if self.explicit:
            return dlu.crop_to(array, self.n)
        return dlu.crop_to(array, self.crop_size(array.shape))

    def resize(self, array: Array, fill: float = 0.0) -> Array:
        """Resize an array to the final sampling represented by this object."""
        return dlu.resize(array, self.output_size(array.shape), fill)


class GridSpec(BaseGridSpec):
    """A complete regularly sampled Cartesian coordinate grid.

    Axis parameters are ordered physically as ``(x, y, z, ...)``. Array dimensions
    are ordered in reverse, so a two-dimensional specification with
    ``n=(nx, ny)`` produces coordinate arrays with shape ``(2, ny, nx)``.
    """

    n: tuple[int, ...] | None
    d: Array | None
    c: Array | None
    unit: str | None

    def __init__(self, n=None, d=None, c=None, unit=None, diam=None):
        if d is not None and diam is not None:
            raise ValueError("Provide only one of d or diam.")
        if diam is not None and n is None:
            raise ValueError("n must be provided with diam.")

        values = [value for value in (n, d, c, diam) if value is not None]
        lengths = [
            np.asarray(value).shape[-1]
            for value in values
            if np.asarray(value).ndim >= 1
        ]
        ndim = max(lengths, default=1 if values else 0)

        self.n = None if n is None else dlu.as_size(n, ndim, "n")
        if diam is not None:
            diam = dlu.as_axis(diam, ndim, "diam")
            d = diam / np.asarray(self.n)
        self.d = dlu.as_axis(d, ndim, "d")
        self.c = dlu.as_axis(c, ndim, "c")
        if (
            self.d is not None
            and not isinstance(self.d, core.Tracer)
            and np.any(self.d <= 0)
        ):
            raise ValueError("d must contain positive values.")
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
        """Return one pixel-centre coordinate vector per physical axis."""
        if self.n is None:
            raise ValueError("n must be specified to calculate axes.")
        return self.axes_for(self.n)

    def axes_for(self, n: tuple[int, ...]) -> tuple[Array, ...]:
        """Return coordinate axes for concrete physical-axis pixel counts."""
        if self.d is None:
            raise ValueError("d must be specified to calculate axes.")
        if len(n) != self.ndim:
            raise ValueError("n dimensionality must match the coordinate spec.")
        batch = self.d.shape[:-1]
        if self.c is not None:
            batch = np.broadcast_shapes(batch, self.c.shape[:-1])
        spacing = np.broadcast_to(self.d, batch + (self.ndim,))
        center = (
            np.zeros(batch + (self.ndim,))
            if self.c is None
            else np.broadcast_to(self.c, batch + (self.ndim,))
        )
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

    def coordinates_for(self, n: tuple[int, ...]) -> Array:
        """Return full coordinates for concrete physical-axis pixel counts."""
        batch = self.d.shape[:-1]
        if self.c is not None:
            batch = np.broadcast_shapes(batch, self.c.shape[:-1])
        spacing = np.broadcast_to(self.d, batch + (self.ndim,))
        shape = tuple(n[::-1])
        axes = []
        for i, size in enumerate(n):
            axis = (np.arange(size) - (size - 1) / 2) * spacing[..., i, None]
            spatial_shape = [1] * self.ndim
            spatial_shape[self.ndim - i - 1] = size
            axis = axis.reshape(batch + tuple(spatial_shape))
            axes.append(np.broadcast_to(axis, batch + shape))
        coordinates = np.stack(tuple(axes), axis=len(batch))
        if self.c is None:
            return coordinates * self.scale
        center = np.broadcast_to(self.c, batch + (self.ndim,))
        center = center.reshape(batch + (self.ndim,) + (1,) * self.ndim)
        return (coordinates + center) * self.scale

    @property
    def xs(self):
        """Return all one-dimensional coordinate axes in one array."""
        if self.n is None:
            raise ValueError("n must be specified to calculate xs.")
        return self.xs_for(self.n)

    def xs_for(self, n: tuple[int, ...]):
        """Return stacked axes for concrete physical-axis pixel counts."""
        try:
            return np.stack(self.axes_for(n), axis=-2)
        except ValueError as error:
            raise ValueError(
                "xs requires equal axis lengths; use axes for a rectangular grid."
            ) from error

    @property
    def fov(self):
        """Return the field of view along every physical axis."""
        if self.n is None or self.d is None:
            raise ValueError("n and d must be specified to calculate fov.")
        return np.asarray(self.n) * self.d * self.scale

    @property
    def extent(self):
        """Return lower and upper grid-edge coordinates for every axis."""
        half_width = self.fov / 2
        center = np.zeros(self.ndim) if self.c is None else self.c * self.scale
        return center - half_width, center + half_width


class CoordTransform(zdx.Base):
    """Base class for transformations applied to coordinate fields."""

    @staticmethod
    def get_coordinates(coordinates) -> Array:
        """Validate and return a Cartesian coordinate array."""
        if coordinates is None:
            raise ValueError("Provide coordinates when calling the transformation.")
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim < 3 or coordinates.shape[-3] != 2:
            raise ValueError("coordinates must have shape (..., 2, ny, nx).")
        return np.asarray(coordinates, dtype=float)

    @abstractmethod
    def __call__(self, coordinates: Array) -> Array:  # pragma: no cover
        """Transform an array of Cartesian coordinates."""

    def apply(self, coordinates: Array) -> Array:
        """Backwards-compatible alias for calling the transformation."""
        return self(coordinates)


class TransformChain(CoordTransform):
    """Apply an ordered collection of coordinate transformations."""

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
        coords = self.get_coordinates(coords)
        for transformation in self.transformations.values():
            coords = transformation(coords)
        return coords


def _distortion_powers(order, orders, powers, shift_invariant):
    """Resolve the polynomial powers used by a coordinate distortion."""
    if sum(value is not None for value in (order, orders, powers)) > 1:
        raise ValueError("Provide only one of order, orders, or powers.")
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
    if shift_invariant:
        linear = np.logical_or(
            np.all(powers == np.array([[1], [0]]), axis=0),
            np.all(powers == np.array([[0], [1]]), axis=0),
        )
        powers = powers[:, ~linear]
    return powers


class DistortCoords(CoordTransform):
    """Polynomially distorted Cartesian coordinates."""

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
        distortion = np.asarray(distortion, dtype=float)
        if distortion.shape[-2:] != self.powers.shape:
            raise ValueError("distortion trailing dimensions must match powers shape.")
        self.distortion = distortion

    def __call__(self, coords: Array) -> Array:
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
    """A direct affine coordinate map ``x' = matrix @ x + offset``."""

    matrix: Array
    offset: Array

    def __init__(self, matrix=None, offset=None):
        matrix = np.eye(2) if matrix is None else np.asarray(matrix, dtype=float)
        offset = np.zeros(2) if offset is None else np.asarray(offset, dtype=float)
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("matrix must have trailing shape (2, 2).")
        if offset.shape[-1:] != (2,):
            raise ValueError("offset must have trailing shape (2,).")
        self.matrix = matrix
        self.offset = offset

    def __call__(self, coords: Array) -> Array:
        coords = self.get_coordinates(coords)
        shift = self.offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", self.matrix, coords) + shift


class Affine(CoordTransform):
    """An affine coordinate transform with semantic parameters.

    Translation, rotation, scale, and shear map coordinates into a transformed
    object's local frame. Operations are composed in the order supplied by ``order``.
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
        self.rotation = None if rotation is None else np.asarray(rotation, dtype=float)
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
        value = np.asarray(value, dtype=float)
        if value.shape[-1:] != (2,):
            raise ValueError(f"{name} must have trailing shape (2,).")
        return value

    def _matrices(self) -> Array:
        """Return all affine components as ordered homogeneous matrices."""
        identity = np.eye(3)

        translation = identity
        if self.translation is not None:
            translation = identity.at[:2, 2].set(-self.translation)

        rotation = identity
        if self.rotation is not None:
            cosine, sine = np.cos(self.rotation), np.sin(self.rotation)
            rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])

        scale = identity
        if self.scale is not None:
            scale = np.diag(np.concatenate((1 / self.scale, np.ones(1))))

        shear = identity
        if self.shear is not None:
            shear = shear.at[0, 1].set(self.shear[0])
            shear = shear.at[1, 0].set(self.shear[1])

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
        coords = self.get_coordinates(coords)
        matrix, offset = self.coefficients()
        shift = offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", matrix, coords) + shift
