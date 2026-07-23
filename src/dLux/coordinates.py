"""Coordinate-transformation contracts and ordered composition."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array, lax

import dLux.utils as dlu

__all__ = [
    "BaseCoordTransform",
    "Affine",
    "TransformChain",
    "DistortedCoords",
    "Distort",
]


class BaseCoordTransform(zdx.Base):
    """Base class for coordinate transformations."""

    def __init_subclass__(cls, **kwargs):
        """Inherit the coordinate transformation interface documentation."""
        super().__init_subclass__(**kwargs)
        dlu.helpers.inherit_docstrings(cls, ["__call__"])

    def calculate(self, npix: int, diameter: float) -> Array:
        """Generate a coordinate grid and apply this transformation."""
        return self(dlu.pixel_coords(npix, diameter))

    @abstractmethod
    def __call__(self, coordinates: Array) -> Array:  # pragma: no cover
        """Transform an array of Cartesian coordinates."""

    def apply(self, coordinates: Array) -> Array:
        """Backwards-compatible alias for calling the transformation."""
        return self(coordinates)


class Affine(BaseCoordTransform):
    """A general affine coordinate transform with semantic parameters.

    Coordinates are mapped as ``x' = M x + t``. ``matrix`` and ``offset`` expose
    that low-level form directly, while translation, rotation, scale, and shear
    provide the common physical parameterisation. Operations are composed in the
    order supplied by ``order``.
    """

    translation: Array | None
    rotation: Array | None
    scale: Array | None
    shear: Array | None
    matrix: Array | None
    offset: Array | None
    order: tuple[str, ...]

    def __init__(
        self,
        translation=None,
        rotation=None,
        scale=None,
        shear=None,
        matrix=None,
        offset=None,
        order=("translation", "rotation", "scale", "shear", "matrix"),
    ):
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
        self.matrix = None if matrix is None else np.asarray(matrix, dtype=float)
        if self.matrix is not None and self.matrix.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2).")
        self.offset = self._vector(offset, "offset")

        valid = ("translation", "rotation", "scale", "shear", "matrix")
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

        matrix = identity
        if self.matrix is not None:
            matrix = matrix.at[:2, :2].set(self.matrix)
        if self.offset is not None:
            matrix = matrix.at[:2, 2].set(self.offset)

        matrices = np.stack((translation, rotation, scale, shear, matrix))
        indices = np.array(
            tuple(
                ("translation", "rotation", "scale", "shear", "matrix").index(name)
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
        matrix, offset = self.coefficients()
        shift = offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", matrix, coords) + shift


class TransformChain(BaseCoordTransform):
    """Apply an ordered collection of coordinate transformations."""

    transformations: dict

    def __init__(self, transformations=()):
        if isinstance(transformations, dict):
            transformations = list(transformations.items())
        else:
            transformations = list(transformations)
        self.transformations = dlu.list2dictionary(
            transformations, True, BaseCoordTransform
        )

    def __call__(self, coords: Array) -> Array:
        for transformation in self.transformations.values():
            coords = transformation(coords)
        return coords


class DistortedCoords(BaseCoordTransform):
    """Polynomially distorted Cartesian coordinates."""

    powers: Array
    distortion: Array

    def __init__(self, order: int = 1, distortion: Array | None = None):
        self.powers = np.array(dlu.gen_powers(order + 1))[:, 1:]
        if distortion is None:
            distortion = np.zeros_like(self.powers)
        distortion = np.asarray(distortion, dtype=float)
        if distortion.shape != self.powers.shape:
            raise ValueError("distortion shape must match powers shape.")
        self.distortion = distortion

    def __call__(self, coords: Array) -> Array:
        return dlu.distort_coords(coords, self.distortion, self.powers)


class Distort(BaseCoordTransform):
    """Apply a polynomial distortion as one coordinate transformation."""

    distortion: DistortedCoords

    def __init__(self, order=1, coefficients=None):
        self.distortion = DistortedCoords(order, coefficients)

    @property
    def coefficients(self) -> Array:
        return self.distortion.distortion

    def __call__(self, coords: Array) -> Array:
        return self.distortion(coords)
