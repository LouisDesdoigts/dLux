"""Parametric analytic, composite, and sparse aperture geometries."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array, core, lax, vmap

import dLux.utils as dlu

from ..sampled import Sampled

__all__ = [
    "shift_and_scale",
    "soften",
    "circ_distance",
    "square_distance",
    "rectangle_distance",
    "spider_distance",
    "line_distance",
    "reg_polygon_edges",
    "reg_polygon_distance",
    "convex_polygon_distance",
    "validate_convex",
    "Shape",
    "InvertibleShape",
    "Soft",
    "Circle",
    "Square",
    "Rectangle",
    "RegularPolygon",
    "ConvexPolygon",
    "Spider",
    "Complement",
]


def shift_and_scale(array: Array) -> Array:
    """Rescale each image over its final two axes to the range zero to one."""
    minimum = array.min(axis=(-2, -1), keepdims=True)
    maximum = array.max(axis=(-2, -1), keepdims=True)
    return dlu.nandiv(array - minimum, maximum - minimum, fill=0)


def soften(distances: Array, clip_dist: float, invert: bool = False) -> Array:
    """Soften positive-inside distances over a physical half-width.

    Distances are clipped to ``[-clip_dist, clip_dist]`` then rescaled using each
    image's sampled minimum and maximum. This preserves a full zero-to-one range
    when only part of the transition is sampled. Constant images return one inside
    and zero outside. ``invert`` reverses the distance before softening.
    """
    if invert:
        distances *= -1
    # Normalise by the local width before clipping; widths may vary over a grid.
    distances = np.clip(distances / clip_dist, -1.0, 1.0)
    maximum = distances.max(axis=(-2, -1), keepdims=True)
    minimum = distances.min(axis=(-2, -1), keepdims=True)
    span = maximum - minimum
    scaled = dlu.nandiv(distances - minimum, span, fill=0)
    constant_support = (distances > 0).astype(distances.dtype)
    return np.where(span == 0, constant_support, scaled)


def circ_distance(coords: Array, radius: float) -> Array:
    """Return radial distance minus radius on ``(..., 2, ny, nx)`` coordinates."""
    return dlu.cart2polar(coords)[..., 0, :, :] - radius


def square_distance(coords: Array, width: float) -> Array:
    """Return the maximum signed edge distance for a square of full width."""
    return np.max(np.abs(coords), axis=-3) - width / 2


def rectangle_distance(coords: Array, width: float, height: float) -> Array:
    """Return the maximum signed edge distance for full x/y dimensions."""
    dist_from_vert = np.abs(coords[..., 0, :, :]) - width / 2
    dist_from_horz = np.abs(coords[..., 1, :, :]) - height / 2
    return np.maximum(dist_from_vert, dist_from_horz)


def spider_distance(coords: Array, width: float, angle: float) -> Array:
    """Return the signed edge distance to one radial arm.

    Width uses the coordinate unit and angle is in degrees. Negative distances
    lie inside the arm; zero degrees extends along the negative y axis.
    """
    coords = dlu.Affine(rotation=np.deg2rad(angle))(coords)
    dist_from_vert = np.abs(coords[..., 0, :, :]) - width / 2
    dist_from_horz = coords[..., 1, :, :]
    return np.maximum(dist_from_vert, dist_from_horz)


def line_distance(coords: Array, m: float, xy: Array) -> Array:
    """Return signed perpendicular distance to a line with slope ``m``.

    ``xy`` is one point on the line. The negative side faces the coordinate origin;
    horizontal and vertical lines are handled explicitly.
    """
    # Extract x and y coordinates from the input array
    x, y = coords[..., 0, :, :], coords[..., 1, :, :]

    # First determine the case: horizontal, vertical or regular
    case = (m == 0) * 0 + (m == np.inf) * 1 + (m != 0) * (m != np.inf) * 2

    # Now we determine the sign on the y-intersection so we know when to flip
    # the sign of the distance
    horiz_fn = lambda: np.sign(xy[1])
    vert_fn = lambda: np.sign(xy[0])
    base_fn = lambda: np.sign(xy[1] - m * xy[0])
    sgn = lax.switch(case, [horiz_fn, vert_fn, base_fn])

    # Now we calculate the distance from the point to the line
    horiz_fn = lambda: xy[1] - y
    vert_fn = lambda: xy[0] - x
    base_fn = lambda: (m * x - y + xy[1] - m * xy[0]) / np.sqrt(m**2 + 1)
    dist = lax.switch(case, [horiz_fn, vert_fn, base_fn])

    # Finally, return the distances with the correct sign
    return -sgn * dist


def reg_polygon_edges(n: int, radius: float) -> Array:
    """Return slopes and points defining a regular polygon's edges.

    ``n`` is the number of vertices, on a circle of the supplied radius. The first
    vertex lies on the negative x axis.
    """
    # Calculate the x and y coordinates of the vertices of the polygon
    x = radius * np.cos(np.linspace(-np.pi, np.pi, n, endpoint=False))
    y = radius * np.sin(np.linspace(-np.pi, np.pi, n, endpoint=False))

    # Generate the indexes of the vertices and the next vertex
    ivals = np.arange(n)
    jvals = np.roll(ivals, -1)

    # Calculate the gradient and point on each edge of the polygon
    calc_m = vmap(lambda i, j: dlu.nandiv(y[j] - y[i], x[j] - x[i]), (0, 0))
    calc_xy = vmap(lambda i, j: np.array([x[i], y[i]]), (0, 0))
    return calc_m(ivals, jvals), calc_xy(ivals, jvals)


def reg_polygon_distance(coords: Array, nsides: int, radius: float) -> Array:
    """Return the maximum signed edge distance for a regular polygon.

    ``radius`` is its circumradius. Coordinates have shape ``(..., 2, ny, nx)``;
    distances are negative inside. This is a half-plane boundary measure, not the
    exact Euclidean distance outside a vertex.
    """
    m, xy = reg_polygon_edges(nsides, radius)
    return vmap(line_distance, (None, 0, 0))(coords, m, xy).max(0)


def convex_polygon_distance(coords: Array, vertices: Array) -> Array:
    """Return the maximum signed perpendicular distance to polygon edges.

    Parameters
    ----------
    coords : Array
        Coordinates with shape ``(..., 2, ny, nx)``.
    vertices : Array
        Sequential polygon vertices with shape ``(..., nvertices, 2)``.

    Returns
    -------
    distances : Array
        Maximum signed edge distance, negative inside and positive outside. This
        half-plane measure is not the exact Euclidean distance outside a vertex.
    """
    vertices = np.asarray(vertices)

    # Generate oriented edges and determine the polygon winding
    ends = np.roll(vertices, -1, axis=-2)
    edges = ends - vertices
    area = np.sum(vertices[..., 0] * ends[..., 1], -1)
    area -= np.sum(vertices[..., 1] * ends[..., 0], -1)
    winding = np.where(area >= 0, 1.0, -1.0)

    # Calculate the signed distance from every point to every oriented edge
    x, y = coords[..., 0, :, :], coords[..., 1, :, :]
    x0, y0 = vertices[..., 0], vertices[..., 1]
    dx, dy = edges[..., 0], edges[..., 1]
    cross = dx[..., :, None, None] * (y[..., None, :, :] - y0[..., :, None, None])
    cross -= dy[..., :, None, None] * (x[..., None, :, :] - x0[..., :, None, None])
    lengths = np.linalg.norm(edges, axis=-1)
    distances = -winding[..., None, None, None] * cross
    distances /= lengths[..., :, None, None]

    # A convex polygon is bounded by its most restrictive edge
    return distances.max(-3)


def validate_convex(vertices: Array) -> None:
    """Validate one or more sequential convex-polygon vertex arrays.

    Parameters
    ----------
    vertices : Array
        Vertices with shape ``(..., nvertices, 2)`` in sequential boundary order.
        Clockwise and anticlockwise winding are accepted.

    Raises
    ------
    ValueError
        If the vertices have an invalid shape, repeat, enclose zero area, or do not
        describe a convex polygon in boundary order.
    """
    vertices = np.asarray(vertices)
    if vertices.ndim < 2 or vertices.shape[-1] != 2:
        raise ValueError("vertices must have shape (..., nvertices, 2).")
    if vertices.shape[-2] < 3:
        raise ValueError("vertices must contain at least three points.")

    # Calculate vertex separations and the signed polygon area
    ends = np.roll(vertices, -1, axis=-2)
    offsets = vertices[..., :, None, :] - vertices[..., None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    distances += np.eye(vertices.shape[-2])
    area = np.sum(vertices[..., 0] * ends[..., 1], -1)
    area -= np.sum(vertices[..., 1] * ends[..., 0], -1)

    # Check that every consecutive edge turns in a consistent direction
    edges = ends - vertices
    following = np.roll(edges, -1, axis=-2)
    cross = edges[..., 0] * following[..., 1]
    cross -= edges[..., 1] * following[..., 0]
    if np.any(distances == 0) or np.any(area == 0):
        raise ValueError("vertices must be distinct and define a non-zero area.")
    if np.any(np.any(cross > 0, axis=-1) & np.any(cross < 0, axis=-1)):
        raise ValueError("vertices must define a convex polygon in boundary order.")


class Shape(Sampled):
    """Base geometry that evaluates to a transmission array.

    Call a shape with a `Grid` to sample it directly, or use `resolve` with
    explicit coordinates and pixel scales. Scalar-per-profile dimensions may have
    leading batch axes; these broadcast against coordinate batches before the final
    two spatial axes. Spider angles remain one shared one-dimensional list.
    Concrete constructor values are checked before JAX tracing; use immutable
    updates or expression leaves during evaluation.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    @property
    def extent(self) -> Array | None:
        """Return the underlying geometry's radius, or None when undefined.

        Builders use this local, untransformed size for OPD support. It is not a
        bound on nonzero transmission: inversion and complements retain the same
        geometry size, and coordinate transforms do not change it.
        """
        return None


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

        if not isinstance(self.pixels, (zdx.Expression, core.Tracer)):
            if np.any(self.pixels <= 0):
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
        self = self.resolve(**context)
        if pixel_scale is None:
            raise zdx.Unresolved("pixel_scale is required for a softened edge.")
        pixel_scale = np.asarray(pixel_scale)
        if pixel_scale.ndim:
            pixel_scale = pixel_scale.max(-1)
        pixels = zdx.resolve(self.pixels, **context)
        return pixel_scale * pixels / 2


class InvertibleShape(Shape):
    """Geometry with optional edge softening and transmission inversion.

    ``edge`` may be ``Soft``, a numeric pixel width converted to ``Soft``,
    or ``None`` for an exact hard boundary. Shapes return one on
    their support; inversion always returns the exact complement. Hard boundaries
    are excluded, except for explicit ``ConvexPolygon`` vertices whose boundary is
    included. Geometric leaves resolve in the shape's local coordinate context.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    edge: Soft | None = None
    invert: bool = False

    def __init__(
        self,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise shared edge and inversion behaviour.

        Parameters
        ----------
        edge : Soft, float, or None
            Edge definition. Numeric values construct `Soft`; ``None`` constructs
            None (hard edges).
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        if edge is not None and not isinstance(edge, Soft):
            edge = Soft(edge)

        self.edge = edge
        self.invert = bool(invert)

        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate transmission with the configured edge and inversion.


        Parameters
        ----------
        grid : Grid or None
            Two-dimensional sampling in SI units, or native unitless units.
            Mutually exclusive with coordinates.
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
            Softening uses the source pixel scale; Hard edges ignore sampling metadata.
        """
        # Resolve shape parameters on the selected local coordinates and pixel scale.
        context = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        self = self.resolve_children(**context)
        coordinates = context.pop("coordinates")
        pixel_scale = context.pop("pixel_scale")

        # Choose a hard boundary or a pixel-scaled soft edge, then optionally invert.
        if self.edge is None:
            transmission = self.evaluate_hard(coordinates, **context)
        else:
            clip = self.edge.clip(pixel_scale, coordinates=coordinates, **context)
            clip = clip[..., None, None]
            transmission = self.evaluate_soft(coordinates, clip, **context)
        return 1 - transmission if self.invert else transmission

    def distance(self, coordinates, **context):
        """Return a boundary distance, negative inside and positive outside.

        Dimensions are resolved by evaluate before this method is called.
        Subclasses preserve coordinate leading axes and final spatial axes. Polygon
        distances use bounding half-planes,
        rather than exact Euclidean distances outside their vertices. Alternatively,
        subclasses may supply their own ``evaluate_hard`` and ``evaluate_soft``.
        """
        raise NotImplementedError("Implement distance or hard and soft evaluation.")

    def evaluate_hard(self, coordinates, **context):
        """Return binary support, excluding samples exactly on the boundary."""
        return (self.distance(coordinates, **context) < 0).astype(float)

    def evaluate_soft(self, coordinates, clip, **context):
        """Soften support using physical half-width ``clip``.

        The clipped distances are rescaled over each sampled image to span zero to
        one. A constant image is one inside the shape and zero outside it.
        """
        distances = self.distance(coordinates, **context)
        return soften(-distances, clip)


class Circle(InvertibleShape):
    """Represent a centred circular transmission by its full diameter.

    Diameter uses the coordinate unit supplied during evaluation. Edge softening and
    inversion follow the common `InvertibleShape` contract.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    diameter: Array | zdx.Expression

    def __init__(
        self,
        diameter,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a circular transmission.

        Parameters
        ----------
        diameter : float, Array, or zodiax.Expression
            Positive diameter in the coordinate unit.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.diameter = zdx.as_array(diameter, dtype=float)

        if not isinstance(self.diameter, (zdx.Expression, core.Tracer)):
            if np.any(self.diameter <= 0):
                raise ValueError("diameter must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return half the circle diameter in its physical input unit."""
        return self.diameter / 2

    def distance(self, coordinates, **context):
        """Return the signed distance to the circular boundary."""
        diameter = zdx.resolve(self.diameter, **context)
        diameter = np.asarray(diameter)[..., None, None]
        return circ_distance(coordinates, diameter / 2)


class Square(InvertibleShape):
    """Represent a centred axis-aligned square transmission.

    Width uses the coordinate unit supplied during evaluation. Edge softening and
    inversion follow the common `InvertibleShape` contract.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    width: Array | zdx.Expression

    def __init__(
        self,
        width,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a square transmission.

        Parameters
        ----------
        width : float, Array, or zodiax.Expression
            Positive full width in the coordinate unit.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.width = zdx.as_array(width, dtype=float)

        if not isinstance(self.width, zdx.Expression) and np.any(self.width <= 0):
            raise ValueError("width must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return the square circumradius in its physical input unit."""
        return self.width / np.sqrt(2)

    def distance(self, coordinates, **context):
        """Return the maximum signed distance to the square's edges."""
        width = zdx.resolve(self.width, **context)
        width = np.asarray(width)[..., None, None]
        return square_distance(coordinates, width)


class Rectangle(InvertibleShape):
    """Represent a centred axis-aligned rectangular transmission.

    Width and height use the coordinate unit supplied during evaluation and follow
    physical ``(x, y)`` orientation. An optional stored transform rotates or
    translates the shape.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    width: Array | zdx.Expression
    height: Array | zdx.Expression

    def __init__(
        self,
        width,
        height,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a rectangular transmission.

        Parameters
        ----------
        width, height : float, Array, or zodiax.Expression
            Positive full dimensions in the coordinate unit.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.width = zdx.as_array(width, dtype=float)
        self.height = zdx.as_array(height, dtype=float)

        for value in (self.width, self.height):
            if not isinstance(value, (zdx.Expression, core.Tracer)):
                if np.any(value <= 0):
                    raise ValueError("width and height must be greater than zero.")

    @property
    def extent(self) -> Array:
        """Return the rectangle circumradius in its physical input unit."""
        return np.hypot(self.width, self.height) / 2

    def distance(self, coordinates, **context):
        """Return the maximum signed distance to the rectangle's edges."""
        width, height = zdx.resolve((self.width, self.height), **context)
        return rectangle_distance(
            coordinates, width[..., None, None], height[..., None, None]
        )


class RegularPolygon(InvertibleShape):
    """Represent a centred regular polygon by its circumscribed-circle diameter.

    The first vertex follows the utility geometry convention. Diameter uses the
    evaluation coordinate unit, with common edge softening and inversion behaviour.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    diameter: Array | zdx.Expression
    nsides: int

    def __init__(
        self,
        nsides,
        diameter,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a regular polygon.

        Parameters
        ----------
        nsides : int
            Number of sides, at least three.
        diameter : float, Array, or zodiax.Expression
            Positive circumscribed-circle diameter.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.diameter = zdx.as_array(diameter, dtype=float)
        self.nsides = int(nsides)

        if not isinstance(self.diameter, (zdx.Expression, core.Tracer)):
            if np.any(self.diameter <= 0):
                raise ValueError("diameter must be greater than zero.")
        if self.nsides < 3:
            raise ValueError("nsides must be at least three.")

    @property
    def extent(self) -> Array:
        """Return the regular-polygon circumradius in its physical input unit."""
        return self.diameter / 2

    def distance(self, coordinates, **context):
        """Return the maximum signed edge distance in the coordinate unit."""
        diameter = zdx.resolve(self.diameter, **context)
        diameter = np.asarray(diameter)[..., None, None]
        unit_coordinates = coordinates / diameter[..., None, :, :]
        return reg_polygon_distance(unit_coordinates, self.nsides, 0.5) * diameter


class ConvexPolygon(InvertibleShape):
    """A convex polygon described by sequential ``(x, y)`` vertices.

    Clockwise and anticlockwise winding are both accepted. Vertices may carry natural
    leading batch dimensions, while the final dimensions must have shape
    ``(nvertices, 2)``. The boundary is included by hard evaluation; numeric or
    ``Soft`` edges use the signed distance to the nearest edge. Call ``validate()``
    explicitly to check the current vertices outside compiled calculations.

    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    vertices: Array | zdx.Expression

    def __init__(
        self,
        vertices,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise a convex polygon without enforcing convexity at runtime.

        Parameters
        ----------
        vertices : Array or zodiax.Expression
            Ordered vertices with trailing shape ``(nvertices, 2)``. Call
            `validate` explicitly when diagnostic validation is required.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.vertices = zdx.as_array(vertices, dtype=float)

    @property
    def extent(self) -> Array:
        """Return the largest vertex radius in the polygon coordinate unit."""
        return np.linalg.norm(self.vertices, axis=-1).max(-1)

    def distance(self, coordinates, **context):
        """Return signed edge distances, assuming sequential convex vertices."""
        vertices = zdx.resolve(self.vertices, **context)
        return convex_polygon_distance(coordinates, vertices)

    def evaluate_hard(self, coordinates, **context):
        """Return polygon support, including samples exactly on its boundary."""
        return (self.distance(coordinates, **context) <= 0).astype(float)

    def validate(self) -> None:
        """Validate the current vertices as a strictly convex ordered polygon.

        Raises `ValueError` when the vertices do not form a non-degenerate clockwise
        or anticlockwise convex boundary. The method returns ``None`` and is opt-in:
        later immutable parameter updates may change the vertices again.
        """
        validate_convex(self.vertices)


class Spider(InvertibleShape):
    """Represent centred radial support arms as a transmissive spider mask.

    Width uses the coordinate unit and angles are measured in degrees. The result
    is one on the arms, like every shape on its geometric support; inversion returns
    the unobstructed transmission. The legacy ``spider`` function uses the opposite
    convention.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)
    edge: dlu.Soft | None = None
    invert: bool = False

    width: Array | zdx.Expression
    angles: Array | zdx.Expression

    def __init__(
        self,
        width,
        angles,
        edge=None,
        invert=False,
        *,
        grid=None,
        transform=None,
        strict=False,
        alias=None,
    ):
        """Initialise radial support arms.

        Parameters
        ----------
        width : float, Array, or zodiax.Expression
            Positive support width in the coordinate unit.
        angles : Array or zodiax.Expression
            One-dimensional support angles in degrees.
        edge : Soft, float, or None
            Hard or softened boundary definition.
        invert : bool
            Return the complementary transmission when true.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        super().__init__(
            edge, invert, grid=grid, transform=transform, strict=strict, alias=alias
        )
        self.width = zdx.as_array(width, dtype=float)
        self.angles = zdx.as_array(angles, dtype=float)

        if not isinstance(self.width, (zdx.Expression, core.Tracer)):
            if np.any(self.width <= 0):
                raise ValueError("width must be greater than zero.")
        if not isinstance(self.angles, zdx.Expression) and self.angles.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")

    def distance(self, coordinates, **context):
        """Return the minimum signed distance across the radial support arms."""
        width, angles = zdx.resolve((self.width, self.angles), **context)
        width = np.asarray(width)[..., None, None]
        distances = vmap(lambda angle: spider_distance(coordinates, width, angle))
        return distances(angles).min(0)


class Complement(Shape):
    """Return one minus another shape's evaluated transmission.

    This composes with every `Shape`, including those without an intrinsic `invert`
    option, while retaining the wrapped shape's sampling and differentiability.
    """

    # Inherited fields
    grid: dlu.UnitSpec | None = None
    transform: Callable | None = None
    strict: bool = eqx.field(static=True, default=False)

    shape: Shape

    def __init__(self, shape, *, grid=None, transform=None, strict=False, alias=None):
        """Initialise the complement of another shape.

        Parameters
        ----------
        shape : Shape
            Shape whose evaluated transmission is inverted.
        alias : mapping or None
            Optional parameter aliases; default None.
        grid : Grid or None
            Optional stored sampling; caller context supplies sampling when it is
            absent.
        transform : callable or None
            Map evaluation coordinates into the input frame. None leaves coordinates
            unchanged.
        strict : bool
            Reject conflicting runtime sampling or transforms when true; false
            allows runtime overrides.
        """
        if not isinstance(shape, Shape):
            raise TypeError("shape must be a Shape.")
        self.shape = shape

        super().__init__(grid=grid, transform=transform, strict=strict, alias=alias)

    @property
    def extent(self) -> Array | None:
        """Return the underlying geometry size, even when transmission is unbounded."""
        return self.shape.extent

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate and complement the wrapped shape transmission.


        Parameters
        ----------
        grid : Grid or None
            Two-dimensional sampling in SI units, or native unitless units.
            Mutually exclusive with coordinates.
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
        sampling = (grid, coordinates, transform, self.grid, self.transform)
        if all(value is None for value in sampling):
            values = zdx.resolve(self.shape, **context)
            if isinstance(values, zdx.Expression):
                raise zdx.Unresolved("The complemented shape needs spatial sampling.")
            return 1 - values
        context = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        shape = zdx.resolve(self.shape, **context)
        return 1 - shape
