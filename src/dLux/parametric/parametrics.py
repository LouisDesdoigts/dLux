"""Contextual interpolation and composition of numerical expressions."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from math import isfinite
from numbers import Real

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from ..utils.grids import BaseCoordTransform, GridSpec

__all__ = [
    "Parametric",
    "SpatialExpression",
    "Interpolation",
    "DynamicParametric",
    "Combination",
]


def _extrapolate(value):
    """Canonicalise Interpax's lower/upper extrapolation settings for JIT."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("extrapolate must contain lower and upper settings.")
        return tuple(_extrapolate(item) for item in value)
    if isinstance(value, bool):
        return value
    if not isinstance(value, Real) or not isfinite(float(value)):
        raise TypeError("extrapolate must be boolean or a finite real scalar.")
    return float(value)


def _spatial_context(
    grid=None,
    *,
    coordinates=None,
    transform=None,
    pixel_scale=None,
    **context,
):
    """Obtain one local coordinate frame and its sampling for spatial operations."""
    if grid is not None and coordinates is not None:
        raise ValueError("Supply exactly one of grid or coordinates, not both.")
    if coordinates is None:
        if grid is None:
            raise zdx.Unresolved("Spatial evaluation requires grid or coordinates.")
        if not isinstance(grid, GridSpec):
            raise TypeError("grid must be a GridSpec.")
        grid = grid.resolve(**context)
        if grid.ndim == 1:
            grid = grid.broadcast(2)
        if grid.ndim != 2:
            raise ValueError("grid must define one or two spatial dimensions.")
        coordinates, pixel_scale = grid.coordinates, grid.d_si
    coordinates = BaseCoordTransform.get_coordinates(coordinates)

    # Transform evaluation coordinates; retain the source grid softening scale.
    local = dict(
        context,
        coordinates=coordinates,
        pixel_scale=pixel_scale,
    )
    if transform is not None:
        if not isinstance(transform, BaseCoordTransform):
            raise TypeError("transform must be a BaseCoordTransform.")
        transform = zdx.resolve(transform, **local)
        local["coordinates"] = transform(coordinates)
    return local


class Parametric(zdx.Expression):
    """A physical value defined by stored parameters and evaluation context.

    ``evaluate`` computes this definition; ``resolve`` prepares definitions within
    a model and defers those whose inputs are missing. Subclasses resolve the stored
    operands they consume. A numerical operand may itself be a Zodiax Transform,
    whose methods remain accessible through Zodiax's normal attribute raising.
    Those methods operate on the child, not on the containing physical definition.

    General definitions implement ``evaluate``. Spatial definitions inherit
    SpatialExpression and implement ``evaluate`` directly. Basis families additionally
    provide ``solve`` for physical coefficients. Calling a Parametric with named
    context delegates to resolution through the inherited Expression convention.
    """

    @abstractmethod
    def evaluate(self, **context):
        """Compute the physical value from stored parameters and supplied context."""


class SpatialExpression(Parametric):
    """A physical definition evaluated on a two-dimensional coordinate field.

    Concrete classes implement ``evaluate`` directly with explicit named inputs.
    Supply exactly one of grid or coordinates; missing both raises Unresolved,
    while supplying both raises ValueError. Each calculation prepares its spatial
    inputs and resolves the parameters it uses.
    Grids provide SI coordinates and pixel sampling. Explicit coordinate arrays
    use shape ``(..., 2, ny, nx)`` and the same units as physical parameters.
    Coordinate transforms preserve the original pixel scale used for soft edges.
    Soft edges require pixel sampling; evaluation does not normalise flux.
    """

    def __call__(self, grid=None, *, transform=None, **context):
        """Resolve on an optional grid or with named coordinate context."""
        return self.resolve(grid=grid, transform=transform, **context)


class Interpolation(Parametric):
    """Interpolate stored samples at coordinates supplied during resolution.

    Knots and coordinates use the same unit. The output has coordinate shape
    followed by trailing sample axes. Knots and values may be expressions; their
    ready children resolve even when query coordinates are not yet available.
    """

    knots: Array | zdx.Expression
    values: Array | zdx.Expression
    method: str = eqx.field(static=True)
    extrapolate: bool | float | tuple[bool | float, bool | float] = eqx.field(
        static=True
    )
    period: float | None = eqx.field(static=True)

    def __init__(
        self,
        knots,
        values,
        method="linear",
        extrapolate=0.0,
        *,
        period=None,
        alias=None,
    ):
        """Store a one-dimensional table and its interpolation settings.

        Parameters
        ----------
        knots : Array or zodiax.Expression
            Finite, strictly increasing real coordinates with at least two samples.
        values : Array or zodiax.Expression
            Samples with a leading knot axis and arbitrary trailing value axes.
        method : str
            Interpax interpolation method, defaulting to linear.
        extrapolate : bool, float, or pair
            Zero fills by default. False gives NaN, True extrapolates, and a number
            fills. A pair configures the lower and upper boundaries separately.
        period : positive float or None
            Optional period in the knot unit, fixed during JAX tracing.
        alias : mapping or None
            User-defined parameter aliases.
        """
        self.knots = zdx.as_array(knots)
        self.values = zdx.as_array(values)
        self.alias = alias

        # Interpolation settings determine how the sampled table is evaluated.
        if not isinstance(method, str) or not method.strip():
            raise ValueError("method must be a non-empty interpolation method name.")
        self.method = method.strip().lower()
        self.extrapolate = _extrapolate(extrapolate)
        if period is not None:
            if isinstance(period, bool) or not isinstance(period, Real):
                raise TypeError("period must be a positive real scalar.")
            if not isfinite(float(period)) or period <= 0:
                raise ValueError("period must be finite and positive.")
        self.period = None if period is None else float(period)

        # Check concrete tables at construction; expression-valued tables wait.
        if isinstance(self.knots, Array):
            if np.iscomplexobj(self.knots) or self.knots.ndim != 1:
                raise ValueError("knots must be a real one-dimensional array.")
            if self.knots.size < 2:
                raise ValueError("knots must contain at least two samples.")
            if np.any(~np.isfinite(self.knots)) or np.any(np.diff(self.knots) <= 0):
                raise ValueError("knots must be finite and strictly increasing.")
        if isinstance(self.values, Array):
            if self.values.ndim == 0:
                raise ValueError("values must have a leading knot axis.")
            if isinstance(self.knots, Array):
                if self.values.shape[0] != self.knots.size:
                    raise ValueError("values must start with an axis matching knots.")

    def evaluate(self, *, coordinates, **context) -> Array:
        """Interpolate a resolved table at explicit query coordinates.

        Parameters
        ----------
        coordinates : Array or zodiax.Expression
            Query values in the knot unit, with arbitrary shape Q.
        **context
            Named inputs used to resolve the query, knots and stored values.

        Returns
        -------
        Array
            Shape Q + V for stored values of shape (K, *V). Interpolation method,
            extrapolation and period follow the stored settings. Missing coordinates
            defers during resolution; direct evaluation requires the argument.
        """
        coordinates = zdx.resolve(coordinates, **context)
        knots = zdx.resolve(self.knots, coordinates=coordinates, **context)
        values = zdx.resolve(self.values, coordinates=coordinates, **context)
        return dlu.interpolate(
            coordinates,
            knots,
            values,
            method=self.method,
            extrapolate=self.extrapolate,
            period=self.period,
        )


class DynamicParametric(SpatialExpression):
    """Evaluate an expression in coordinates transformed by the stored mapping.

    The transformation acts on incoming ``(..., 2, ny, nx)`` coordinates before
    the child is resolved. This samples the child's definition in the new frame.
    """

    parametric: zdx.Expression
    transformation: BaseCoordTransform

    def __init__(self, parametric, transformation, *, alias=None):
        """Initialise coordinate transformation of an expression.

        Parameters
        ----------
        parametric : zodiax.Expression
            Coordinate-dependent value generator.
        transformation : BaseCoordTransform
            Mapping applied to the incoming coordinates, using the same units.
        alias : mapping or sequence of pairs, optional
            Parameter paths into the stored child or coordinate transformation.
        """
        if not isinstance(parametric, zdx.Expression):
            raise TypeError("parametric must be a zodiax.Expression.")
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.parametric = parametric
        self.transformation = transformation
        self.alias = alias

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        """Evaluate the child in its stored local coordinate frame.

        The optional call-time transform is applied first, then the stored
        transformation. The child must consume coordinates for the mapping to
        affect its result; stored sampled arrays are not resampled.


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
            The child value evaluated in local coordinates. Shape and units follow
            the child contract; the source pixel scale is preserved.
        """
        context = _spatial_context(
            grid, coordinates=coordinates, transform=transform, **context
        )
        coordinates = context.pop("coordinates")
        pixel_scale = context.pop("pixel_scale")
        transformation = zdx.resolve(
            self.transformation,
            coordinates=coordinates,
            pixel_scale=pixel_scale,
            **context,
        )
        coordinates = transformation(coordinates)
        return zdx.resolve(
            self.parametric,
            coordinates=coordinates,
            pixel_scale=pixel_scale,
            **context,
        )


class Combination(Parametric):
    """Resolve named expressions in shared context and combine their arrays.

    Sum and product are arithmetic reductions. Union clips the sum to [0, 1],
    while intersection multiplies transmissions. Child outputs broadcast together.
    """

    parametrics: dict[str, zdx.Expression]
    operation: str = eqx.field(static=True)

    def __init__(self, parametrics, operation="sum", *, alias=None):
        """Store named or unnamed expressions and their combination operation.

        Parameters
        ----------
        parametrics : mapping or sequence of zodiax.Expression
            At least one expression, optionally paired with a name.
        operation : str
            One of sum, product, union, or intersection.
        alias : mapping or sequence of pairs, optional
            Parameter paths into the stored expressions.
        """
        if isinstance(parametrics, dict):
            parametrics = list(parametrics.items())
        else:
            parametrics = list(parametrics)
        self.parametrics = dlu.list2dictionary(parametrics, True, zdx.Expression)
        if not self.parametrics:
            raise ValueError("Combination requires at least one parameterisation.")

        operation = str(operation).lower()
        if operation not in ("sum", "product", "union", "intersection"):
            raise ValueError("operation must be sum, product, union, or intersection.")
        self.operation = operation
        self.alias = alias

    def evaluate(self, **context) -> Array:
        """Resolve child values and combine broadcast-compatible arrays.

        Inputs required by each child are forwarded through context. Sum and product
        are arithmetic reductions; union clips a sum to [0, 1], while intersection
        multiplies transmissions. The output has the common broadcast shape. Children
        must produce arrays with scientifically compatible units for the operation.
        """
        values = [zdx.resolve(value, **context) for value in self.parametrics.values()]
        values = np.stack(np.broadcast_arrays(*values))
        if self.operation in ("product", "intersection"):
            return values.prod(0)
        output = values.sum(0)
        return np.clip(output, 0.0, 1.0) if self.operation == "union" else output
