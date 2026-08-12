"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import BaseCoordTransform
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
        """Return a finite physical bounding radius, or ``None`` when undefined.

        Builders use this value to determine compact sampling and OPD support sizes.
        """
        return None


class Hard(Base):
    """Evaluate an exact hard boundary without dynamic edge softening."""


class Soft(Base):
    """Differentiate a shape boundary over a width measured in pixels."""

    pixels: Array

    def __init__(self, pixels=1.0):
        """Initialise a differentiable edge width.

        Parameters
        ----------
        pixels : float or Array
            Positive full transition width measured in sampled pixels.
        """
        self.pixels = dlu.to_value(pixels)

        if self.pixels <= 0:
            raise ValueError("pixels must be greater than zero.")

    def clip(self, pixel_scale) -> Array:
        """Return the physical half-width used to soften the boundary.

        ``pixel_scale`` is per-axis sampling in the same unit as the shape. The
        largest axis scale is multiplied by half the configured pixel width.
        """
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
        """Initialise shared edge and inversion behaviour.

        Parameters
        ----------
        edge : Hard, Soft, float, or None
            Edge definition. Numeric values construct `Soft`; ``None`` constructs
            `Hard`.
        invert : bool
            Return the complementary transmission when true.
        """
        if edge is None:
            edge = Hard()
        elif not isinstance(edge, (Hard, Soft)):
            edge = Soft(edge)

        self.edge = edge
        self.invert = bool(invert)

    def evaluate(self, *, coordinates, pixel_scale=None, **kwargs) -> Array:
        """Evaluate the shape transmission on Cartesian coordinates.

        ``coordinates`` has shape ``(..., 2, ny, nx)`` and uses the same physical
        unit as the shape parameters. Hard shapes ignore ``pixel_scale``; softened
        shapes require per-axis sampling in that unit. The returned real array
        preserves broadcast leading and final spatial axes and is complemented when
        ``invert=True``.
        """
        if isinstance(self.edge, Hard):
            transmission = self.evaluate_hard(coordinates)
        else:
            transmission = self.evaluate_soft(coordinates, self.edge.clip(pixel_scale))
        return 1 - transmission if self.invert else transmission

    @abstractmethod
    def evaluate_hard(self, coordinates):
        """Evaluate a hard boundary on ``(..., 2, ny, nx)`` coordinates.

        Subclasses return a real transmission preserving coordinate leading and
        final spatial axes.
        """

    @abstractmethod
    def evaluate_soft(self, coordinates, clip):
        """Evaluate a softened boundary with physical half-width ``clip``.

        Subclasses return a real transmission preserving coordinate leading and
        final spatial axes.
        """


class Circle(InvertibleShape):
    """A circular transmission described by its diameter."""

    diameter: Array

    def __init__(self, diameter, edge=None, invert=False):
        """Initialise a circular transmission.

        Parameters
        ----------
        diameter : float or Array
            Positive diameter in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.diameter = dlu.to_value(diameter)

        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return half the circle diameter in its physical input unit."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        """Return a binary circle on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.circle(coordinates, self.diameter)

    def evaluate_soft(self, coordinates, clip):
        """Return a circle softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.soft_circle(coordinates, self.diameter, clip)


class Square(InvertibleShape):
    """A square transmissive aperture."""

    width: Array

    def __init__(self, width, edge=None, invert=False):
        """Initialise a square transmission.

        Parameters
        ----------
        width : float or Array
            Positive full width in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)

        if self.width <= 0:
            raise ValueError("width must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return the square circumradius in its physical input unit."""
        return self.width / np.sqrt(2)

    def evaluate_hard(self, coordinates):
        """Return a binary square on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.square(coordinates, self.width)

    def evaluate_soft(self, coordinates, clip):
        """Return a square softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.soft_square(coordinates, self.width, clip)


class Rectangle(InvertibleShape):
    """A rectangular transmissive aperture."""

    width: Array
    height: Array

    def __init__(self, width, height, edge=None, invert=False):
        """Initialise a rectangular transmission.

        Parameters
        ----------
        width, height : float or Array
            Positive full dimensions in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)
        self.height = dlu.to_value(height)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return the rectangle circumradius in its physical input unit."""
        return np.hypot(self.width, self.height) / 2

    def evaluate_hard(self, coordinates):
        """Return a binary rectangle on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.rectangle(coordinates, self.width, self.height)

    def evaluate_soft(self, coordinates, clip):
        """Return a rectangle softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        return dlu.soft_rectangle(coordinates, self.width, self.height, clip)


class RegularPolygon(InvertibleShape):
    """A regular polygon described by its circumscribed-circle diameter."""

    diameter: Array
    nsides: int

    def __init__(self, nsides, diameter, edge=None, invert=False):
        """Initialise a regular polygon.

        Parameters
        ----------
        nsides : int
            Number of sides, at least three.
        diameter : float or Array
            Positive circumscribed-circle diameter.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.diameter = dlu.to_value(diameter)
        self.nsides = int(nsides)

        if self.diameter <= 0:
            raise ValueError("diameter must be greater than zero.")
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

    @property
    def extent(self) -> Array:
        """Return the regular-polygon circumradius in its physical input unit."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates):
        """Return a binary regular polygon on Cartesian coordinates.

        Coordinates follow ``(..., 2, ny, nx)`` and all non-component axes remain.
        """
        return dlu.reg_polygon(coordinates, self.diameter, self.nsides)

    def evaluate_soft(self, coordinates, clip):
        """Return a regular polygon softened across physical half-width ``clip``.

        Coordinates follow ``(..., 2, ny, nx)`` and all non-component axes remain.
        """
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
        """Initialise a convex polygon without enforcing convexity at runtime.

        Parameters
        ----------
        vertices : Array
            Ordered vertices with trailing shape ``(nvertices, 2)``. Call
            `validate` explicitly when diagnostic validation is required.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.vertices = dlu.to_value(vertices, name="vertices")

    @property
    def extent(self) -> Array:
        """Return the largest vertex radius in the polygon coordinate unit."""
        return np.linalg.norm(self.vertices, axis=-1).max(-1)

    def evaluate_hard(self, coordinates):
        """Return a binary convex polygon on Cartesian coordinates.

        Vertex ordering and convexity are assumed; call `validate` explicitly when
        construction-time diagnostics are required.
        """
        return dlu.convex_polygon(coordinates, self.vertices)

    def evaluate_soft(self, coordinates, clip):
        """Return a convex polygon softened across physical half-width ``clip``.

        Vertex ordering and convexity are assumed during evaluation.
        """
        return dlu.soft_convex_polygon(coordinates, self.vertices, clip)

    def validate(self) -> None:
        """Validate the current vertices as a strictly convex ordered polygon.

        Raises `ValueError` when the vertices do not form a non-degenerate clockwise
        or anticlockwise convex boundary. The method returns ``None`` and is opt-in:
        later immutable parameter updates may change the vertices again.
        """
        dlu.validate_convex(self.vertices)


class Spider(InvertibleShape):
    """A general set of occulting radial support arms with angles in degrees."""

    width: Array
    angles: Array

    def __init__(self, width, angles, edge=None, invert=False):
        """Initialise radial support arms.

        Parameters
        ----------
        width : float or Array
            Positive support width in the coordinate unit.
        angles : Array
            One-dimensional support angles in degrees.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        """
        super().__init__(edge, invert)
        self.width = dlu.to_value(width)
        self.angles = dlu.to_value(angles)

        if self.width <= 0:
            raise ValueError("width must be greater than zero.")
        if self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

    def evaluate_hard(self, coordinates):
        """Return binary radial support arms on Cartesian coordinates.

        Width uses the coordinate physical unit and angles are measured in degrees.
        """
        return 1 - dlu.spider(coordinates, self.width, self.angles)

    def evaluate_soft(self, coordinates, clip):
        """Return radial support arms softened across half-width ``clip``.

        Width uses the coordinate physical unit and angles are measured in degrees.
        """
        return dlu.soft_spider(coordinates, self.width, self.angles, clip)


class Complement(Shape):
    """Invert any shape transmission independently of its edge model."""

    shape: Shape

    def __init__(self, shape):
        """Initialise the complement of another shape.

        Parameters
        ----------
        shape : Shape
            Shape whose evaluated transmission is inverted.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape

    @property
    def extent(self) -> Array | None:
        """Return the wrapped shape's physical bounding radius unchanged."""
        return self.shape.extent

    def evaluate(self, **context) -> Array:
        """Evaluate and complement the wrapped shape transmission.

        All named context is forwarded and the result is ``1 - transmission``.
        """
        return 1 - self.shape.evaluate(**context)


class TransformedShape(Shape):
    """Evaluate a shape in a transformed local coordinate frame."""

    shape: Shape
    transformation: BaseCoordTransform

    def __init__(self, shape, transformation):
        """Initialise a shape in a transformed coordinate frame.

        Parameters
        ----------
        shape : Shape
            Geometry evaluated after transforming the input coordinates.
        transformation : BaseCoordTransform
            Coordinate transformation into the shape's local frame.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.shape = shape
        self.transformation = transformation

    @property
    def extent(self) -> Array | None:
        """Return the wrapped shape's untransformed bounding radius.

        This does not enlarge the bound for translation, scale, or distortion.
        """
        return self.shape.extent

    def evaluate(self, *, coordinates, **context) -> Array:
        """Evaluate the wrapped shape in its transformed coordinate frame.

        ``coordinates`` follows ``(..., 2, ny, nx)`` and remaining context is
        forwarded to the wrapped shape after transformation.
        """
        return self.shape.evaluate(
            coordinates=self.transformation(coordinates), **context
        )
