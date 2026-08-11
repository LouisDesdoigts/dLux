"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import CoordTransform
from .parametrics import Parametric

__all__ = [
    "Shape",
    "InvertibleShape",
    "Hard",
    "Soft",
    "Circle",
    "Square",
    "Rectangle",
    "RegularPolygon",
    "ConvexPolygon",
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


class Hard(Base):
    """Evaluate an exact hard boundary without dynamic edge softening."""


class Soft(Base):
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

    ``edge`` may be ``Hard``, ``Soft``, a numeric pixel width converted to ``Soft``,
    or ``None`` converted to the default ``Hard`` definition.
    """

    edge: Hard | Soft
    invert: bool

    def __init__(self, edge=None, invert=False):
        if edge is None:
            edge = Hard()
        elif not isinstance(edge, (Hard, Soft)):
            edge = Soft(edge)

        self.edge = edge
        self.invert = bool(invert)

    def evaluate(self, *, coordinates, pixel_scale=None, **kwargs) -> Array:
        """Evaluate the hard or softened transmission and apply inversion."""
        if isinstance(self.edge, Hard):
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
    """A circular transmission described by its diameter."""

    diameter: Array

    def __init__(self, diameter, edge=None, invert=False):
        super().__init__(edge, invert)
        self.diameter = dlu.to_value(diameter)

        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return the circular bounding radius."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        """Evaluate a hard circular boundary."""
        return dlu.circle(coordinates, self.diameter)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened circular boundary."""
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
        """Return the radius of the square bounding circle."""
        return self.width / np.sqrt(2)

    def evaluate_hard(self, coordinates):
        """Evaluate a hard square boundary."""
        return dlu.square(coordinates, self.width)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened square boundary."""
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
        """Return the radius of the rectangular bounding circle."""
        return np.hypot(self.width, self.height) / 2

    def evaluate_hard(self, coordinates):
        """Evaluate a hard rectangular boundary."""
        return dlu.rectangle(coordinates, self.width, self.height)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened rectangular boundary."""
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
        """Return the polygon circumradius."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        """Evaluate a hard regular-polygon boundary."""
        return dlu.reg_polygon(coordinates, self.diameter, self.nsides)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened regular-polygon boundary."""
        return dlu.soft_reg_polygon(coordinates, self.diameter, self.nsides, clip)


class ConvexPolygon(InvertibleShape):
    """A convex polygon described by sequential ``(x, y)`` vertices.

    Clockwise and anticlockwise winding are both accepted. Vertices may carry natural
    leading batch dimensions, while the final dimensions must have shape
    ``(nvertices, 2)``. The boundary is included by hard evaluation; numeric or
    ``Soft`` edges use the signed distance to the nearest edge. Call ``validate()``
    explicitly to check the current vertices outside compiled calculations.

    Parameters
    ----------
    vertices : Array
        At least three distinct vertices describing a convex boundary in order.
    edge : Hard, Soft, float, or None
        Shared hard or softened edge definition.
    invert : bool
        Whether to return the complement of the polygon transmission.
    """

    vertices: Array

    def __init__(self, vertices, edge=None, invert=False):
        super().__init__(edge, invert)
        self.vertices = dlu.to_value(vertices, name="vertices")

    @property
    def extent(self) -> Array:
        """Return the largest vertex radius."""
        return np.linalg.norm(self.vertices, axis=-1).max(-1)

    def evaluate_hard(self, coordinates):
        """Evaluate a hard convex-polygon boundary."""
        return dlu.convex_polygon(coordinates, self.vertices)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened convex-polygon boundary."""
        return dlu.soft_convex_polygon(coordinates, self.vertices, clip)

    def validate(self) -> None:
        """Validate the current vertices as an ordered convex polygon."""
        dlu.validate_convex(self.vertices)


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
        """Evaluate hard radial support arms."""
        return 1 - dlu.spider(coordinates, self.width, self.angles)

    def evaluate_soft(self, coordinates, clip):
        """Evaluate softened radial support arms."""
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
        """Return the wrapped shape extent."""
        return self.shape.extent

    def evaluate(self, **context) -> Array:
        """Evaluate and invert the wrapped shape transmission."""
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
        """Return the untransformed wrapped-shape extent."""
        return self.shape.extent

    def evaluate(self, *, coordinates, **context) -> Array:
        """Evaluate the wrapped shape in its transformed coordinate frame."""
        return self.shape.evaluate(
            coordinates=self.transformation(coordinates), **context
        )
