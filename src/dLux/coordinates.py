"""Coordinate-transformation contracts and ordered composition."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

__all__ = [
    "BaseCoordTransform",
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
