"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

import jax.numpy as np
from jax import Array, vmap

import dLux.utils as dlu

from ..coordinates import BaseCoordTransform
from .bases import BaseParametric

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
    "Intersection",
    "Union",
    "ApertureArray",
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
    """Base softened geometry parameterised by a bounding radius."""

    radius: Array

    def __init__(self, radius, softening=1.0):
        super().__init__(softening)
        self.radius = np.asarray(radius, dtype=float)
        if self.radius <= 0:
            raise ValueError("radius must be greater than zero.")

    @property
    def extent(self) -> Array:
        return self.radius


class Circle(RadialShape):
    """A circular transmissive aperture."""

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_circle(coordinates, self.radius, self.clip(pixel_scale))


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
    """A regular polygon described by side count and circumradius."""

    nsides: int

    def __init__(self, nsides, radius, softening=1.0):
        super().__init__(radius, softening)
        self.nsides = int(nsides)
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

    def evaluate(self, *, coordinates, pixel_scale, **kwargs) -> Array:
        return dlu.soft_reg_polygon(
            coordinates,
            self.radius,
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
    transformation: BaseCoordTransform

    def __init__(self, shape, transformation):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.shape = shape
        self.transformation = transformation

    @property
    def extent(self) -> Array | None:
        return self.shape.extent

    def evaluate(self, *, coordinates, **context) -> Array:
        return self.shape.evaluate(
            coordinates=self.transformation(coordinates), **context
        )


class CompositeShape(Shape):
    """Base class for ordered collections of shapes."""

    shapes: dict

    def __init__(self, shapes):
        self.shapes = dlu.list2dictionary(list(shapes), True, Shape)

    @property
    def extent(self) -> Array | None:
        extents = [
            shape.extent for shape in self.shapes.values() if shape.extent is not None
        ]
        return None if not extents else np.max(np.asarray(extents))

    def transmissions(self, **context) -> Array:
        return np.asarray([shape.evaluate(**context) for shape in self.shapes.values()])


class Intersection(CompositeShape):
    """Combine shape transmissions by multiplication."""

    def evaluate(self, **context) -> Array:
        return self.transmissions(**context).prod(0)


class Union(CompositeShape):
    """Combine shape transmissions by clipped addition."""

    def evaluate(self, **context) -> Array:
        return np.clip(self.transmissions(**context).sum(0), 0.0, 1.0)


class ApertureArray(Shape):
    """Vectorised copies of one finite shape at a set of aperture centres."""

    shape: Shape
    positions: Array

    def __init__(self, shape, positions):
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        if shape.extent is None:
            raise TypeError("ApertureArray shapes must have a finite extent.")
        self.shape = shape
        self.positions = np.asarray(positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape (n_apertures, 2).")

    @property
    def extent(self) -> Array:
        return np.max(np.linalg.norm(self.positions, axis=-1)) + self.shape.extent

    def local_coordinates(self, coordinates) -> Array:
        return vmap(dlu.translate_coords, in_axes=(None, 0))(
            coordinates, self.positions
        )

    def evaluate(self, *, coordinates, pixel_scale, **context) -> Array:
        local_coordinates = self.local_coordinates(coordinates)
        evaluate = lambda coords: self.shape.evaluate(
            coordinates=coords,
            pixel_scale=pixel_scale,
            **context,
        )
        transmissions = vmap(evaluate)(local_coordinates)
        return np.clip(transmissions.sum(0), 0.0, 1.0)
