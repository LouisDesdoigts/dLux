"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..grids import CoordTransform
from .parametrics import Parametric

__all__ = [
    "Shape",
    "InvertibleShape",
    "Soft",
    "Circle",
    "Square",
    "Rectangle",
    "RegularPolygon",
    "Spider",
    "Complement",
    "TransformedShape",
]


class Shape(Parametric):
    """Base geometry that evaluates to a transmission array."""

    @property
    def extent(self) -> Array | None:
        """Return a finite bounding radius, or ``None`` when undefined."""
        return None


class Soft(zdx.Base):
    """Differentiate a shape boundary over a width measured in pixels."""

    pixels: Array

    def __init__(self, pixels=1.0):
        self.pixels = dlu.to_value(pixels)

        if self.pixels <= 0:
            raise ValueError("pixels must be greater than zero.")

    def clip(self, pixel_scale) -> Array:
        """Return the physical half-width used to soften the boundary."""
        if pixel_scale is None:
            raise ValueError("pixel_scale is required for a softened edge.")
        pixel_scale = np.asarray(pixel_scale)
        if pixel_scale.ndim:
            pixel_scale = pixel_scale.max(-1)
        return pixel_scale * self.pixels / 2


class InvertibleShape(Shape):
    """Geometry with optional edge softening and transmission inversion.

    ``edge`` may be a ``Soft`` object, a numeric pixel width converted to ``Soft``,
    or ``None`` for a hard edge.
    """

    edge: Soft | None
    invert: bool

    def __init__(self, edge=None, invert=False):
        if edge is not None and not isinstance(edge, Soft):
            edge = Soft(edge)

        self.edge = edge
        self.invert = bool(invert)

    def evaluate(self, *, coordinates, pixel_scale=None, **kwargs) -> Array:
        if self.edge is None:
            transmission = self.evaluate_hard(coordinates)
        else:
            transmission = self.evaluate_soft(coordinates, self.edge.clip(pixel_scale))
        return 1 - transmission if self.invert else transmission

    @abstractmethod
    def evaluate_hard(self, coordinates):
        """Evaluate the hard-edged shape on Cartesian coordinates."""

    @abstractmethod
    def evaluate_soft(self, coordinates, clip):
        """Evaluate the softened shape on Cartesian coordinates."""


class Circle(InvertibleShape):
    """A circular transmissive aperture described by its diameter."""

    diameter: Array

    def __init__(self, diameter, edge=None, invert=False):
        super().__init__(edge, invert)
        self.diameter = dlu.to_value(diameter)

        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    @property
    def extent(self) -> Array:
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        return dlu.circle(coordinates, self.diameter)

    def evaluate_soft(self, coordinates, clip):
        return dlu.soft_circle(coordinates, self.diameter, clip)


class Square(InvertibleShape):
    """A square transmissive aperture."""

    width: Array

    def __init__(self, width, edge=None, invert=False):
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)

        if self.width <= 0:
            raise ValueError("width must be greater than zero.")

    @property
    def extent(self) -> Array:
        return self.width / np.sqrt(2)

    def evaluate_hard(self, coordinates):
        return dlu.square(coordinates, self.width)

    def evaluate_soft(self, coordinates, clip):
        return dlu.soft_square(coordinates, self.width, clip)


class Rectangle(InvertibleShape):
    """A rectangular transmissive aperture."""

    width: Array
    height: Array

    def __init__(self, width, height, edge=None, invert=False):
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)
        self.height = dlu.to_value(height)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than zero.")

    @property
    def extent(self) -> Array:
        return np.hypot(self.width, self.height) / 2

    def evaluate_hard(self, coordinates):
        return dlu.rectangle(coordinates, self.width, self.height)

    def evaluate_soft(self, coordinates, clip):
        return dlu.soft_rectangle(coordinates, self.width, self.height, clip)


class RegularPolygon(InvertibleShape):
    """A regular polygon described by its circumscribed-circle diameter."""

    diameter: Array
    nsides: int

    def __init__(self, nsides, diameter, edge=None, invert=False):
        super().__init__(edge, invert)
        self.diameter = dlu.to_value(diameter)
        self.nsides = int(nsides)

        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

    @property
    def extent(self) -> Array:
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        return dlu.reg_polygon(coordinates, self.diameter, self.nsides)

    def evaluate_soft(self, coordinates, clip):
        return dlu.soft_reg_polygon(coordinates, self.diameter, self.nsides, clip)


class Spider(InvertibleShape):
    """A general set of occulting radial support arms with angles in degrees."""

    width: Array
    angles: Array

    def __init__(self, width, angles, edge=None, invert=False):
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)
        self.angles = dlu.to_value(angles)

        if self.width <= 0:
            raise ValueError("width must be greater than zero.")
        if self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

    def evaluate_hard(self, coordinates):
        return 1 - dlu.spider(coordinates, self.width, self.angles)

    def evaluate_soft(self, coordinates, clip):
        return dlu.soft_spider(coordinates, self.width, self.angles, clip)


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
