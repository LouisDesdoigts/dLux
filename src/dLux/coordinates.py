"""Coordinate specifications, transformations, and ordered composition."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array, lax, vmap

import dLux.utils as dlu

__all__ = [
    "BaseSpec",
    "PadSpec",
    "CoordSpec",
    "CoordTransform",
    "Affine",
    "AffineMap",
    "TransformChain",
    "DistortedCoords",
]


class BaseSpec(zdx.Base):
    """Base class for coordinate and sampling specifications."""


class PadSpec(BaseSpec):
    """Sampling specification defined by padding and cropping factors."""

    pad: int
    crop: int
    c: float

    def __init__(self, pad=1, crop=1, c=0.0):
        self.pad = int(pad)
        self.crop = int(crop)
        self.c = np.asarray(c, float)


class CoordSpec(BaseSpec):
    """A complete regularly sampled Cartesian coordinate grid.

    Axis parameters are ordered physically as ``(x, y, z, ...)``. Array dimensions
    are ordered in reverse, so a two-dimensional specification with
    ``n=(nx, ny)`` produces coordinate arrays with shape ``(2, ny, nx)``.
    """

    n: Array | None
    d: Array | None
    c: Array | None
    unit: str | None

    def __init__(self, n=None, d=None, c=None, unit=None):
        values = [value for value in (n, d, c) if value is not None]
        lengths = [
            np.asarray(value).shape[0]
            for value in values
            if np.asarray(value).ndim == 1
        ]
        ndim = max(lengths, default=1 if values else 0)

        self.n = self._as_axes(n, ndim, int, "n")
        self.d = self._as_axes(d, ndim, float, "d")
        self.c = self._as_axes(c, ndim, float, "c")
        if self.n is not None and np.any(self.n < 1):
            raise ValueError("n must contain positive integers.")
        if self.d is not None and np.any(self.d <= 0):
            raise ValueError("d must contain positive values.")
        self.unit = None if unit is None else self._validate_unit(unit)

    @staticmethod
    def _as_axes(value, ndim, dtype, name):
        if value is None:
            return None
        value = np.asarray(value, dtype=dtype)
        if value.ndim > 1:
            raise ValueError(f"{name} must be scalar or one-dimensional.")
        if ndim == 0:
            ndim = 1
        try:
            return np.broadcast_to(value, (ndim,))
        except ValueError as error:
            raise ValueError(
                f"{name} must be scalar or have one value per axis."
            ) from error

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
                return value.shape[0]
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the array shape associated with this coordinate grid."""
        if self.n is None:
            raise ValueError("n must be specified to calculate shape.")
        return tuple(int(value) for value in self.n[::-1])

    @property
    def scale(self) -> float:
        """Return the factor converting coordinate values to canonical SI units."""
        return 1.0 if self.unit is None else dlu.unit_factor(self.unit)

    @property
    def axes(self) -> tuple[Array, ...]:
        """Return one pixel-centre coordinate vector per physical axis."""
        if self.n is None or self.d is None:
            raise ValueError("n and d must be specified to calculate axes.")
        center = np.zeros(self.ndim) if self.c is None else self.c
        return tuple(
            (center[i] + (np.arange(self.n[i]) - (self.n[i] - 1) / 2) * self.d[i])
            * self.scale
            for i in range(self.ndim)
        )

    @property
    def coordinates(self) -> Array:
        """Return the full coordinate array with shape ``(ndim, *shape)``."""
        coordinates = np.meshgrid(*self.axes, indexing="ij")
        spatial_axes = tuple(range(self.ndim - 1, -1, -1))
        return np.stack(
            tuple(np.transpose(axis, spatial_axes) for axis in coordinates),
            axis=0,
        )

    @property
    def xs(self):
        """Return all one-dimensional coordinate axes in one array."""
        try:
            return np.stack(self.axes)
        except ValueError as error:
            raise ValueError(
                "xs requires equal axis lengths; use axes for a rectangular grid."
            ) from error

    @property
    def fov(self):
        """Return the field of view along every physical axis."""
        if self.n is None or self.d is None:
            raise ValueError("n and d must be specified to calculate fov.")
        return self.n * self.d * self.scale

    @property
    def extent(self):
        """Return lower and upper grid-edge coordinates for every axis."""
        half_width = self.fov / 2
        center = np.zeros(self.ndim) if self.c is None else self.c * self.scale
        return center - half_width, center + half_width


class CoordTransform(zdx.Base):
    """Base coordinate transformation with an optional coordinate source.

    Coordinates supplied when calling the transform take precedence over the stored
    source. The stored array or ``CoordSpec`` is used when the transform is called
    without coordinates.
    """

    coordinates: Array | CoordSpec | None

    def __init__(self, coordinates=None):
        if coordinates is not None and not isinstance(coordinates, CoordSpec):
            coordinates = np.asarray(coordinates, dtype=float)
            if coordinates.shape[-3] != 2:
                raise ValueError("coordinates must have shape (..., 2, n, n).")
        self.coordinates = coordinates

    def __init_subclass__(cls, **kwargs):
        """Inherit the coordinate transformation interface documentation."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])

    def calculate(self, npix: int, diameter: float) -> Array:
        """Generate a coordinate grid and apply this transformation."""
        return self(dlu.pixel_coords(npix, diameter))

    @staticmethod
    def _from_spec(spec: CoordSpec) -> Array:
        return spec.coordinates

    def get_coordinates(self, coordinates=None) -> Array:
        """Resolve call-time coordinates, falling back to the stored source."""
        coordinates = self.coordinates if coordinates is None else coordinates
        if coordinates is None:
            raise ValueError("Provide coordinates when calling the transformation.")
        if isinstance(coordinates, CoordSpec):
            coordinates = self._from_spec(coordinates)
        return np.asarray(coordinates, dtype=float)

    @abstractmethod
    def __call__(self, coordinates: Array = None) -> Array:  # pragma: no cover
        """Transform an array of Cartesian coordinates."""

    def apply(self, coordinates: Array) -> Array:
        """Backwards-compatible alias for calling the transformation."""
        return self(coordinates)


class TransformChain(CoordTransform):
    """Apply an ordered collection of coordinate transformations."""

    transformations: dict

    def __init__(self, transformations=(), coordinates=None):
        super().__init__(coordinates)
        if isinstance(transformations, dict):
            transformations = list(transformations.items())
        else:
            transformations = list(transformations)
        self.transformations = dlu.list2dictionary(
            transformations, True, CoordTransform
        )

    def __call__(self, coords: Array = None) -> Array:
        coords = self.get_coordinates(coords)
        for transformation in self.transformations.values():
            coords = transformation(coords)
        return coords


class DistortedCoords(CoordTransform):
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
        coordinates=None,
    ):
        super().__init__(coordinates)
        supplied = sum(value is not None for value in (order, orders, powers))
        if supplied > 1:
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
                orders = tuple(int(value) for value in orders)
            if not orders or any(value < 1 for value in orders):
                raise ValueError("orders must contain positive integers.")
            powers = np.array(dlu.gen_powers(max(orders) + 1))[:, 1:]
            powers = powers[:, np.isin(powers.sum(0), np.asarray(orders))]

        self.shift_invariant = bool(shift_invariant)
        if self.shift_invariant:
            linear = np.logical_or(
                np.all(powers == np.array([[1], [0]]), axis=0),
                np.all(powers == np.array([[0], [1]]), axis=0),
            )
            powers = powers[:, ~linear]
        self.powers = powers
        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = np.asarray(distortion, dtype=float)
        if distortion.shape[-2:] != self.powers.shape:
            raise ValueError("distortion trailing dimensions must match powers shape.")
        self.distortion = distortion

    def __call__(self, coords: Array = None) -> Array:
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

    def __init__(self, matrix=None, offset=None, coordinates=None):
        super().__init__(coordinates)
        matrix = np.eye(2) if matrix is None else np.asarray(matrix, dtype=float)
        offset = np.zeros(2) if offset is None else np.asarray(offset, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2).")
        if offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")
        self.matrix = matrix
        self.offset = offset

    def __call__(self, coords: Array = None) -> Array:
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
        coordinates=None,
    ):
        super().__init__(coordinates)
        self.translation = self._vector(translation, "translation")
        self.rotation = None if rotation is None else np.asarray(rotation, dtype=float)
        if self.rotation is not None and self.rotation.shape != ():
            raise ValueError("rotation must be scalar.")
        self.scale = None
        if scale is not None:
            self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (2,))
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
        if value.shape != (2,):
            raise ValueError(f"{name} must have shape (2,).")
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

    def __call__(self, coords: Array = None) -> Array:
        coords = self.get_coordinates(coords)
        matrix, offset = self.coefficients()
        shift = offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", matrix, coords) + shift
