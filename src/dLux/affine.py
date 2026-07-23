"""Semantic affine operations and ordered affine composition."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

import dLux.utils as dlu
from .coordinates import BaseCoordTransform

__all__ = [
    "BaseAffineOperation",
    "MatrixAffine",
    "Translation",
    "Rotation",
    "Scaling",
    "Shearing",
    "Affine",
]


class BaseAffineOperation(BaseCoordTransform):
    """Base class for one semantically parameterised affine operation."""

    matrix: Array
    offset: Array

    def __call__(self, coords: Array) -> Array:
        shift = self.offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", self.matrix, coords) + shift

    def __matmul__(self, other: BaseAffineOperation) -> Affine:
        """Compose operations, applying ``other`` before ``self``."""
        if not isinstance(other, BaseAffineOperation):
            return NotImplemented
        return Affine([other, self])


class MatrixAffine(BaseAffineOperation):
    """An affine operation parameterised directly by a matrix and offset."""

    matrix: Array
    offset: Array

    def __init__(self, matrix=None, offset=None):
        matrix = np.eye(2) if matrix is None else np.asarray(matrix, dtype=float)
        offset = np.zeros(2) if offset is None else np.asarray(offset, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("matrix must have shape (2, 2).")
        if offset.shape != (2,):
            raise ValueError("offset must have shape (2,).")
        self.matrix = matrix
        self.offset = offset


class Translation(BaseAffineOperation):
    """Map coordinates into a translated local frame."""

    translation: Array

    def __init__(self, translation):
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (2,):
            raise ValueError("translation must have shape (2,).")
        self.translation = translation

    @property
    def matrix(self) -> Array:
        return np.eye(2)

    @property
    def offset(self) -> Array:
        return -self.translation


class Rotation(BaseAffineOperation):
    """Map coordinates into a local frame rotated by an angle in radians."""

    angle: Array

    def __init__(self, angle):
        angle = np.asarray(angle, dtype=float)
        if angle.shape != ():
            raise ValueError("angle must be scalar.")
        self.angle = angle

    @property
    def matrix(self) -> Array:
        cosine = np.cos(self.angle)
        sine = np.sin(self.angle)
        return np.array([[cosine, -sine], [sine, cosine]])

    @property
    def offset(self) -> Array:
        return np.zeros(2)


class Scaling(BaseAffineOperation):
    """Actively scale an object by mapping coordinates into its local frame."""

    scale: Array

    def __init__(self, scale):
        scale = np.broadcast_to(np.asarray(scale, dtype=float), (2,))
        if np.any(scale == 0):
            raise ValueError("scale values must be non-zero.")
        self.scale = scale

    @property
    def matrix(self) -> Array:
        return np.diag(1 / self.scale)

    @property
    def offset(self) -> Array:
        return np.zeros(2)


class Shearing(BaseAffineOperation):
    """Apply the dLux two-axis coordinate-shear convention."""

    shear: Array

    def __init__(self, shear):
        shear = np.asarray(shear, dtype=float)
        if shear.shape != (2,):
            raise ValueError("shear must have shape (2,).")
        self.shear = shear

    @property
    def matrix(self) -> Array:
        return np.array([[1, self.shear[0]], [self.shear[1], 1]])

    @property
    def offset(self) -> Array:
        return np.zeros(2)


class Affine(BaseAffineOperation):
    """Ordered composition of semantically parameterised affine operations."""

    operations: dict

    def __init__(self, operations=()):
        if isinstance(operations, BaseAffineOperation):
            operations = [operations]
        elif isinstance(operations, dict):
            operations = list(operations.items())
        else:
            operations = list(operations)
        self.operations = dlu.list2dictionary(
            operations, True, allowed_types=(BaseAffineOperation,)
        )

    def __getattr__(self, key):
        """Resolve named affine operations and their parameters."""
        if key in self.operations:
            return self.operations[key]
        for operation in self.operations.values():
            if hasattr(operation, key):
                return getattr(operation, key)
        raise dlu.missing_attribute_error(self, key, list(self.operations))

    def _combine(self) -> tuple[Array, Array]:
        matrix = np.eye(2)
        offset = np.zeros(2)
        for operation in self.operations.values():
            operation_matrix = operation.matrix
            operation_offset = operation.offset
            offset = operation_matrix @ offset + operation_offset
            matrix = operation_matrix @ matrix
        return matrix, offset

    @property
    def matrix(self) -> Array:
        return self._combine()[0]

    @property
    def offset(self) -> Array:
        return self._combine()[1]

    def __call__(self, coords: Array) -> Array:
        matrix, offset = self._combine()
        shift = offset.reshape((2,) + (1,) * (coords.ndim - 1))
        return np.einsum("ij,j...->i...", matrix, coords) + shift

    def __matmul__(self, other: BaseAffineOperation) -> Affine:
        """Compose operations, applying ``other`` before ``self``."""
        if not isinstance(other, BaseAffineOperation):
            return NotImplemented
        left = list(self.operations.items())
        right = list(other.operations.items()) if isinstance(other, Affine) else [other]
        return Affine(right + left)

    def insert(
        self,
        operation: BaseAffineOperation | tuple[str, BaseAffineOperation],
        index: int,
    ) -> Affine:
        """Return a copy with an affine operation inserted at ``index``."""
        operations = dlu.insert_layer(
            self.operations.copy(), operation, index, BaseAffineOperation
        )
        return self.set("operations", operations)

    def remove(self, key: str) -> Affine:
        """Return a copy with the named affine operation removed."""
        operations = dlu.remove_layer(self.operations.copy(), key)
        return self.set("operations", operations)

    @classmethod
    def translate(cls, translation) -> Affine:
        return cls([Translation(translation)])

    @classmethod
    def rotate(cls, angle) -> Affine:
        return cls([Rotation(angle)])

    @classmethod
    def scale(cls, scale) -> Affine:
        return cls([Scaling(scale)])

    @classmethod
    def shear(cls, shear) -> Affine:
        return cls([Shearing(shear)])
