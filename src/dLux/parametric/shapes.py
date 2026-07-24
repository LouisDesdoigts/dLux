"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..coordinates import CoordTransform
from .parametrics import BaseParametric

__all__ = [
    "Shape",
    "SoftShape",
    "RadialShape",
    "Circle",
    "Square",
    "Rectangle",
    "RegularPolygon",
    "Spider",
    "Complement",
    "TransformedShape",
]


class Shape(BaseParametric):
    """Base geometry that evaluates to a transmission array."""

    @property
    def extent(self) -> Array | None:
        """Return a finite bounding radius, or ``None`` when undefined."""
        return None


class SoftShape(Shape):
    """Base geometry with a differentiably softened edge."""

    softening: Array

    def __init__(self, softening=1.0):
        self.softening = np.asarray(softening, dtype=float)
        if self.softening <= 0:
            raise ValueError("softening must be greater than zero.")

    def clip(self, pixel_scale) -> Array:
        """Return the physical half-width used to soften the boundary."""
        return np.asarray(pixel_scale) * self.softening / 2


class RadialShape(SoftShape):
    """Base softened geometry parameterised by a bounding diameter."""

    diameter: Array

    def __init__(self, diameter, softening=1.0):
        super().__init__(softening)
        self.diameter = np.asarray(diameter, dtype=float)
        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    @property
    def extent(self) -> Array:
        return self.diameter / 2


class Circle(RadialShape):
    """A circular transmissive aperture described by its diameter."""

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_circle(coordinates, self.diameter, self.clip(pixel_scale))


class Square(SoftShape):
    """A square transmissive aperture."""

    width: Array

    def __init__(self, width, softening=1.0):
        super().__init__(softening)
        self.width = np.asarray(width, dtype=float)
        if self.width <= 0:
            raise ValueError("width must be greater than zero.")

    @property
    def extent(self) -> Array:
        return self.width / np.sqrt(2)

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_square(coordinates, self.width, self.clip(pixel_scale))


class Rectangle(SoftShape):
    """A rectangular transmissive aperture."""

    width: Array
    height: Array

    def __init__(self, width, height, softening=1.0):
        super().__init__(softening)
        self.width = np.asarray(width, dtype=float)
        self.height = np.asarray(height, dtype=float)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than zero.")

    @property
    def extent(self) -> Array:
        return np.hypot(self.width, self.height) / 2

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_rectangle(
            coordinates,
            self.width,
            self.height,
            self.clip(pixel_scale),
        )


class RegularPolygon(RadialShape):
    """A regular polygon described by its circumscribed-circle diameter."""

    nsides: int

    def __init__(self, nsides, diameter, softening=1.0):
        super().__init__(diameter, softening)
        self.nsides = int(nsides)
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_reg_polygon(
            coordinates,
            self.diameter,
            self.nsides,
            self.clip(pixel_scale),
        )


class Spider(SoftShape):
    """A general set of occulting radial support arms with angles in degrees."""

    width: Array
    angles: Array

    def __init__(self, width, angles, softening=1.0):
        super().__init__(softening)
        self.width = np.asarray(width, dtype=float)
        self.angles = np.asarray(angles, dtype=float)
        if self.width <= 0:
            raise ValueError("width must be greater than zero.")
        if self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_spider(
            coordinates,
            self.width,
            self.angles,
            self.clip(pixel_scale),
            invert=True,
        )


class Complement(Shape):
    """Invert any shape transmission independently of its edge model."""

    shape: Shape

    def __init__(self, shape):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape

    @property
    def extent(self) -> Array | None:
        return self.shape.extent

    def evaluate(self, **context) -> Array:
        return 1 - self.shape.evaluate(**context)


class TransformedShape(Shape):
    """Evaluate a shape in a transformed local coordinate frame."""

    shape: Shape
    transformation: CoordTransform

    def __init__(self, shape, transformation):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        self.shape = shape
        self.transformation = transformation

    @property
    def extent(self) -> Array | None:
        return self.shape.extent

    def evaluate(self, *, coordinates, **context) -> Array:
        return self.shape.evaluate(
            coordinates=self.transformation(coordinates), **context
        )
