"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

from abc import abstractmethod

import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..utils.grids import BaseCoordTransform
from .parametrics import SpatialExpression, _spatial_context

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


class Shape(SpatialExpression):
    """Base geometry that evaluates to a transmission array.

    Call a shape with a `GridSpec` to sample it directly, or use `resolve` with
    explicit coordinates and pixel scales. Scalar-per-profile dimensions may have
    leading batch axes; these broadcast against coordinate batches before the final
    two spatial axes. Spider angles remain one shared one-dimensional list.
    Concrete constructor values are checked before JAX tracing; use immutable
    updates or expression leaves during evaluation.
    """

    @property
    def extent(self) -> Array | None:
        """Return the underlying geometry's radius, or None when undefined.

        Builders use this local, untransformed size for OPD support. It is not a
        bound on nonzero transmission: inversion and complements retain the same
        geometry size, and coordinate wrappers do not transform it.
        """
        return None


class Hard(zdx.Module):
    """Select exact binary evaluation for a shape boundary.

    Hard edges ignore pixel scale and are generally nondifferentiable with respect to
    geometric boundary parameters at sampled transition pixels.
    """


class Soft(zdx.Module):
    """Soften a shape boundary over a transition width measured in pixels.

    The pixel width is converted to physical distance from the evaluation sampling,
    allowing gradients with respect to geometric parameters near the boundary.
    Coordinate transformations leave that physical width unchanged; the transition
    need not span the same number of output pixels after a transformation.
    """

    pixels: Array | zdx.Expression

    def __init__(self, pixels=1.0, *, alias=None):
        """Initialise a differentiable edge width.

        Parameters
        ----------
        pixels : float, Array, or zodiax.Expression
            Positive full transition width measured in sampled pixels.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        self.pixels = zdx.as_array(pixels, dtype=float)

        if not isinstance(self.pixels, zdx.Expression) and np.any(self.pixels <= 0):
            raise ValueError("pixels must be greater than zero.")

        self.alias = alias

    def clip(self, pixel_scale=None, **context) -> Array:
        """Return the physical half-width used to soften the boundary.

        ``pixel_scale`` is per-axis sampling in the same unit as the shape. The
        largest axis scale is multiplied by half the configured pixel width.
        Coordinate transforms leave this scale unchanged. This is a smoothing
        convention in the shape coordinates, not integration over pixel footprints.
        Missing pixel scale raises Unresolved, allowing the owning shape to defer.
        """
        pixels = zdx.resolve(self.pixels, **context)
        if pixel_scale is None:
            raise zdx.Unresolved("pixel_scale is required for a softened edge.")
        pixel_scale = np.asarray(pixel_scale)
        if pixel_scale.ndim:
            pixel_scale = pixel_scale.max(-1)
        return pixel_scale * pixels / 2


class InvertibleShape(Shape):
    """Geometry with optional edge softening and transmission inversion.

    ``edge`` may be ``Hard``, ``Soft``, a numeric pixel width converted to ``Soft``,
    or ``None`` converted to the default ``Hard`` definition. Geometric leaves may
    be expressions and are resolved in the shape's local coordinate context.
    """

    edge: Hard | Soft
    invert: bool

    def __init__(self, edge=None, invert=False, *, alias=None):
        """Initialise shared edge and inversion behaviour.

        Parameters
        ----------
        edge : Hard, Soft, float, or None
            Edge definition. Numeric values construct `Soft`; ``None`` constructs
            `Hard`.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        if edge is None:
            edge = Hard()
        elif not isinstance(edge, (Hard, Soft)):
            edge = Soft(edge)

        self.edge = edge
        self.invert = bool(invert)

        self.alias = alias

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate transmission with the configured edge and inversion.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            Transmission with broadcast leading axes and final shape (ny, nx).
            Softening uses the source pixel scale; Hard ignores sampling metadata.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        pixel_scale = context.pop("pixel_scale")
        if isinstance(self.edge, Hard):
            transmission = self.evaluate_hard(coordinates, **context)
        else:
            clip = self.edge.clip(pixel_scale, coordinates=coordinates, **context)
            clip = clip[..., None, None]
            transmission = self.evaluate_soft(coordinates, clip, **context)
        return 1 - transmission if self.invert else transmission

    @abstractmethod
    def evaluate_hard(self, coordinates, **context):
        """Evaluate a hard boundary on ``(..., 2, ny, nx)`` coordinates.

        Subclasses return a real transmission preserving coordinate leading and
        final spatial axes.
        """

    @abstractmethod
    def evaluate_soft(self, coordinates, clip, **context):
        """Evaluate a softened boundary with physical half-width ``clip``.

        Subclasses return a real transmission preserving coordinate leading and
        final spatial axes.
        """


class Circle(InvertibleShape):
    """Represent a centred circular transmission by its full diameter.

    Diameter uses the coordinate unit supplied during evaluation. Edge softening and
    inversion follow the common `InvertibleShape` contract.
    """

    edge: Hard | Soft
    invert: bool
    diameter: Array | zdx.Expression

    def __init__(self, diameter, edge=None, invert=False, *, alias=None):
        """Initialise a circular transmission.

        Parameters
        ----------
        diameter : float, Array, or zodiax.Expression
            Positive diameter in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.diameter = zdx.as_array(diameter, dtype=float)

        if not isinstance(self.diameter, zdx.Expression) and np.any(self.diameter <= 0):
            raise ValueError("diameter must be greater than zero.")

        self.alias = alias

    @property
    def extent(self) -> Array:
        """Return half the circle diameter in its physical input unit."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates, **context):
        """Return a binary circle on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        diameter = zdx.resolve(self.diameter, coordinates=coordinates, **context)
        diameter = np.asarray(diameter)[..., None, None]
        return dlu.circle(coordinates, diameter)

    def evaluate_soft(self, coordinates, clip, **context):
        """Return a circle softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        diameter = zdx.resolve(self.diameter, coordinates=coordinates, **context)
        diameter = np.asarray(diameter)[..., None, None]
        return dlu.soft_circle(coordinates, diameter, clip)


class Square(InvertibleShape):
    """Represent a centred axis-aligned square transmission.

    Width uses the coordinate unit supplied during evaluation. Edge softening and
    inversion follow the common `InvertibleShape` contract.
    """

    edge: Hard | Soft
    invert: bool
    width: Array | zdx.Expression

    def __init__(self, width, edge=None, invert=False, *, alias=None):
        """Initialise a square transmission.

        Parameters
        ----------
        width : float, Array, or zodiax.Expression
            Positive full width in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.width = zdx.as_array(width, dtype=float)

        if not isinstance(self.width, zdx.Expression) and np.any(self.width <= 0):
            raise ValueError("width must be greater than zero.")

        self.alias = alias

    @property
    def extent(self) -> Array:
        """Return the square circumradius in its physical input unit."""
        return self.width / np.sqrt(2)

    def evaluate_hard(self, coordinates, **context):
        """Return a binary square on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        width = zdx.resolve(self.width, coordinates=coordinates, **context)
        width = np.asarray(width)[..., None, None]
        return dlu.square(coordinates, width)

    def evaluate_soft(self, coordinates, clip, **context):
        """Return a square softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        width = zdx.resolve(self.width, coordinates=coordinates, **context)
        width = np.asarray(width)[..., None, None]
        return dlu.soft_square(coordinates, width, clip)


class Rectangle(InvertibleShape):
    """Represent a centred axis-aligned rectangular transmission.

    Width and height use the coordinate unit supplied during evaluation and follow
    physical ``(x, y)`` orientation. Use `TransformedShape` to rotate or translate it.
    """

    edge: Hard | Soft
    invert: bool
    width: Array | zdx.Expression
    height: Array | zdx.Expression

    def __init__(self, width, height, edge=None, invert=False, *, alias=None):
        """Initialise a rectangular transmission.

        Parameters
        ----------
        width, height : float, Array, or zodiax.Expression
            Positive full dimensions in the coordinate unit.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.width = zdx.as_array(width, dtype=float)
        self.height = zdx.as_array(height, dtype=float)

        for value in (self.width, self.height):
            if not isinstance(value, zdx.Expression) and np.any(value <= 0):
                raise ValueError("width and height must be greater than zero.")

        self.alias = alias

    @property
    def extent(self) -> Array:
        """Return the rectangle circumradius in its physical input unit."""
        return np.hypot(self.width, self.height) / 2

    def evaluate_hard(self, coordinates, **context):
        """Return a binary rectangle on ``(..., 2, ny, nx)`` coordinates.

        Leading coordinate axes and final spatial axes are preserved.
        """
        width, height = zdx.resolve(
            (self.width, self.height), coordinates=coordinates, **context
        )
        return dlu.rectangle(
            coordinates, width[..., None, None], height[..., None, None]
        )

    def evaluate_soft(self, coordinates, clip, **context):
        """Return a rectangle softened across physical half-width ``clip``.

        Leading coordinate axes and final spatial axes are preserved.
        """
        width, height = zdx.resolve(
            (self.width, self.height), coordinates=coordinates, **context
        )
        return dlu.soft_rectangle(
            coordinates, width[..., None, None], height[..., None, None], clip
        )


class RegularPolygon(InvertibleShape):
    """Represent a centred regular polygon by its circumscribed-circle diameter.

    The first vertex follows the utility geometry convention. Diameter uses the
    evaluation coordinate unit, with common edge softening and inversion behaviour.
    """

    edge: Hard | Soft
    invert: bool
    diameter: Array | zdx.Expression
    nsides: int

    def __init__(self, nsides, diameter, edge=None, invert=False, *, alias=None):
        """Initialise a regular polygon.

        Parameters
        ----------
        nsides : int
            Number of sides, at least three.
        diameter : float, Array, or zodiax.Expression
            Positive circumscribed-circle diameter.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.diameter = zdx.as_array(diameter, dtype=float)
        self.nsides = int(nsides)

        if not isinstance(self.diameter, zdx.Expression) and np.any(self.diameter <= 0):
            raise ValueError("diameter must be greater than zero.")
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

        self.alias = alias

    @property
    def extent(self) -> Array:
        """Return the regular-polygon circumradius in its physical input unit."""
        return self.diameter / 2

    def evaluate_hard(self, coordinates, **context):
        """Return a binary regular polygon on Cartesian coordinates.

        Coordinates follow ``(..., 2, ny, nx)`` and all non-component axes remain.
        """
        diameter = zdx.resolve(self.diameter, coordinates=coordinates, **context)
        diameter = np.asarray(diameter)[..., None, None]
        return dlu.reg_polygon(
            coordinates / diameter[..., None, :, :], 1.0, self.nsides
        )

    def evaluate_soft(self, coordinates, clip, **context):
        """Return a regular polygon softened across physical half-width ``clip``.

        Coordinates follow ``(..., 2, ny, nx)`` and all non-component axes remain.
        """
        diameter = zdx.resolve(self.diameter, coordinates=coordinates, **context)
        diameter = np.asarray(diameter)[..., None, None]
        return dlu.soft_reg_polygon(
            coordinates / diameter[..., None, :, :], 1.0, self.nsides, clip / diameter
        )


class ConvexPolygon(InvertibleShape):
    """A convex polygon described by sequential ``(x, y)`` vertices.

    Clockwise and anticlockwise winding are both accepted. Vertices may carry natural
    leading batch dimensions, while the final dimensions must have shape
    ``(nvertices, 2)``. The boundary is included by hard evaluation; numeric or
    ``Soft`` edges use the signed distance to the nearest edge. Call ``validate()``
    explicitly to check the current vertices outside compiled calculations.

    """

    edge: Hard | Soft
    invert: bool
    vertices: Array | zdx.Expression

    def __init__(self, vertices, edge=None, invert=False, *, alias=None):
        """Initialise a convex polygon without enforcing convexity at runtime.

        Parameters
        ----------
        vertices : Array or zodiax.Expression
            Ordered vertices with trailing shape ``(nvertices, 2)``. Call
            `validate` explicitly when diagnostic validation is required.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.vertices = zdx.as_array(vertices, dtype=float)

        self.alias = alias

    @property
    def extent(self) -> Array:
        """Return the largest vertex radius in the polygon coordinate unit."""
        return np.linalg.norm(self.vertices, axis=-1).max(-1)

    def evaluate_hard(self, coordinates, **context):
        """Return a binary convex polygon on Cartesian coordinates.

        Vertex ordering and convexity are assumed; call `validate` explicitly when
        construction-time diagnostics are required.
        """
        vertices = zdx.resolve(self.vertices, coordinates=coordinates, **context)
        return dlu.convex_polygon(coordinates, vertices)

    def evaluate_soft(self, coordinates, clip, **context):
        """Return a convex polygon softened across physical half-width ``clip``.

        Vertex ordering and convexity are assumed during evaluation.
        """
        vertices = zdx.resolve(self.vertices, coordinates=coordinates, **context)
        return dlu.soft_convex_polygon(coordinates, vertices, clip)

    def validate(self) -> None:
        """Validate the current vertices as a strictly convex ordered polygon.

        Raises `ValueError` when the vertices do not form a non-degenerate clockwise
        or anticlockwise convex boundary. The method returns ``None`` and is opt-in:
        later immutable parameter updates may change the vertices again.
        """
        dlu.validate_convex(self.vertices)


class Spider(InvertibleShape):
    """Represent centred radial support arms as a transmissive spider mask.

    Width uses the coordinate unit and angles are measured in degrees. By default the
    returned transmission is one away from the occulting arms; inversion complements
    that convention.
    """

    edge: Hard | Soft
    invert: bool
    width: Array | zdx.Expression
    angles: Array | zdx.Expression

    def __init__(self, width, angles, edge=None, invert=False, *, alias=None):
        """Initialise radial support arms.

        Parameters
        ----------
        width : float, Array, or zodiax.Expression
            Positive support width in the coordinate unit.
        angles : Array or zodiax.Expression
            One-dimensional support angles in degrees.
        edge : Hard, Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        super().__init__(edge, invert)
        self.width = zdx.as_array(width, dtype=float)
        self.angles = zdx.as_array(angles, dtype=float)

        if not isinstance(self.width, zdx.Expression) and np.any(self.width <= 0):
            raise ValueError("width must be greater than zero.")
        if not isinstance(self.angles, zdx.Expression) and self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

        self.alias = alias

    def evaluate_hard(self, coordinates, **context):
        """Return binary radial support arms on Cartesian coordinates.

        Width uses the coordinate physical unit and angles are measured in degrees.
        """
        width, angles = zdx.resolve(
            (self.width, self.angles), coordinates=coordinates, **context
        )
        return 1 - dlu.spider(coordinates, width[..., None, None], angles)

    def evaluate_soft(self, coordinates, clip, **context):
        """Return radial support arms softened across half-width ``clip``.

        Width uses the coordinate physical unit and angles are measured in degrees.
        """
        width, angles = zdx.resolve(
            (self.width, self.angles), coordinates=coordinates, **context
        )
        return dlu.soft_spider(coordinates, width[..., None, None], angles, clip)


class Complement(Shape):
    """Return one minus another shape's evaluated transmission.

    This composes with every `Shape`, including those without an intrinsic `invert`
    option, while retaining the wrapped shape's sampling and differentiability.
    """

    shape: Shape

    def __init__(self, shape, *, alias=None):
        """Initialise the complement of another shape.

        Parameters
        ----------
        shape : Shape
            Shape whose evaluated transmission is inverted.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape

        self.alias = alias

    @property
    def extent(self) -> Array | None:
        """Return the underlying geometry size, even when transmission is unbounded."""
        return self.shape.extent

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate and complement the wrapped shape transmission.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            One minus the child transmission, retaining its sampled shape.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        return 1 - zdx.resolve(self.shape, **context)


class TransformedShape(Shape):
    """Evaluate a shape after mapping coordinates into a transformed local frame.

    The coordinate transform is applied before shape evaluation, so geometric
    parameters remain defined in the shape's local coordinates. No sampled array is
    interpolated or transformed after evaluation.
    """

    shape: Shape
    transformation: BaseCoordTransform

    def __init__(self, shape, transformation, *, alias=None):
        """Initialise a shape in a transformed coordinate frame.

        Parameters
        ----------
        shape : Shape
            Geometry evaluated after transforming the input coordinates.
        transformation : BaseCoordTransform
            Coordinate transformation into the shape's local frame.
        alias : mapping or None
            Optional parameter aliases; default None.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.shape = shape
        self.transformation = transformation

        self.alias = alias

    @property
    def extent(self) -> Array | None:
        """Return the wrapped geometry's local, untransformed radius.

        This does not enlarge the bound for translation, scale, or distortion.
        """
        return self.shape.extent

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate a shape in its stored local coordinate frame.

        The optional call-time transform is applied first, then the stored
        transformation. No sampled array is resampled.


        Parameters
        ----------
        grid : GridSpec or None
            Spatial sampling in SI units, or native unitless units. A one-dimensional
            grid is promoted to a square. Mutually exclusive with coordinates.
        coordinates : Array or None
            Samples with shape (..., 2, ny, nx), in physical (x, y) order and the same
            units as the geometric parameters. Supply exactly one spatial input.
        transform : BaseCoordTransform or None
            Optional coordinate mapping applied before this definition's calculation.
        **context
            Inputs for stored expressions. Explicit coordinates require pixel_scale
            for soft edges; a grid supplies its own scale. Coordinate transforms
            preserve that scale. Missing spatial inputs raise Unresolved and defer
            during resolution; ambiguous inputs raise ValueError.


        Returns
        -------
        Array
            Wrapped shape transmission on the transformed coordinates, retaining
            the original pixel scale for soft edges.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        pixel_scale = context.pop("pixel_scale")
        transform = zdx.resolve(
            self.transformation,
            coordinates=coordinates,
            pixel_scale=pixel_scale,
            **context,
        )
        coordinates = transform(coordinates)
        return zdx.resolve(
            self.shape,
            coordinates=coordinates,
            pixel_scale=pixel_scale,
            **context,
        )
