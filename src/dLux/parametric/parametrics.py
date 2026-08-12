"""Core contextual parameterisation and composition classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as np
import jax.tree as jtu
from jax import Array

import dLux.utils as dlu

from ..base import Base
from ..grids import BaseCoordTransform

__all__ = [
    "Parametric",
    "ParametricHolder",
    "resolve",
    "Transform",
    "Interpolation",
    "DynamicParametric",
    "Combination",
]


def resolve(value: Any, dtype: Any = None, **context: Any) -> Any:
    """Evaluate a parameterisation and optionally cast the result.

    Parameters
    ----------
    value : Any or Parametric
        Fixed value or parametric evaluated with ``context``.
    dtype : dtype or None
        Optional dtype applied to the resolved value.
    **context
        Named physical and model values available during evaluation.
    """
    value = value.evaluate(**context) if isinstance(value, Parametric) else value
    return value if value is None or dtype is None else dlu.to_value(value, dtype)


class ParametricHolder(Base):
    """Base class for objects containing context-dependent parameters."""

    def resolve(self, **context):
        """Return a copy with every `Parametric` leaf evaluated.

        Named ``context`` values are forwarded unchanged to each leaf's `evaluate`
        method. Non-parametric leaves are preserved and the original object is not
        mutated.
        """
        is_parametric = lambda value: isinstance(value, Parametric)
        evaluate = lambda value: resolve(value, **context)
        return jtu.map(evaluate, self, is_leaf=is_parametric)


class Parametric(ParametricHolder):
    """A contextual parameterisation consumed by another dLux object."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the parameterisation in the supplied named context.

        Subclasses document which context keys they consume and the shape and units
        of the returned value. Implementations must remain compatible with the JAX
        transformations promised by the consuming dLux object.
        """

    def map(self, transformation) -> Parametric:
        """Return a parametric applying ``transformation`` after evaluation.

        ``transformation`` receives the wrapped realised value. Context and JAX
        compatibility are inherited from the wrapped parametric and callable.
        """
        return Transform(self, transformation)

    def integrate(self, lower, upper, **context) -> Array:
        """Integrate the parameterisation between matching lower and upper bounds.

        Bounds and output units are defined by the concrete parametric. The base
        implementation raises unless the subclass provides an integration contract.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define spectral integration."
        )


class Transform(Parametric):
    """Apply a callable transformation to a realised parameterisation."""

    parametric: Parametric
    transformation: Any

    def __init__(self, parametric, transformation):
        """Initialise a transformation of another parametric.

        Parameters
        ----------
        parametric : Parametric
            Value generator evaluated before the transformation.
        transformation : callable
            Function applied to the resolved value.
        """
        if not isinstance(parametric, Parametric):
            raise TypeError("parametric must be a Parametric.")
        if not callable(transformation):
            raise TypeError("transformation must be callable.")
        self.parametric = parametric
        self.transformation = transformation

    def evaluate(self, **context):
        """Evaluate the wrapped parameterisation and transform its value.

        All named context is forwarded before the stored callable is applied. The
        callable defines the returned shape and units.
        """
        return self.transformation(self.parametric.evaluate(**context))


class Interpolation(Parametric):
    """A one-dimensional parameterisation defined by sampled values."""

    knots: Array
    values: Array
    method: str = eqx.field(static=True)
    extrapolate: bool | float = eqx.field(static=True)

    def __init__(self, knots, values, method="linear", extrapolate=0.0):
        """Initialise one-dimensional parametric interpolation.

        Parameters
        ----------
        knots : Array
            Strictly increasing one-dimensional sample coordinates.
        values : Array
            Values whose leading axis matches ``knots``.
        method : str
            Interpolation method accepted by Interpax.
        extrapolate : bool or float
            Interpax extrapolation behaviour or fill value.
        """
        knots = dlu.to_value(knots)
        values = dlu.to_value(values)
        if knots.ndim != 1:
            raise ValueError("knots must be one-dimensional.")
        if knots.size < 2:
            raise ValueError("knots must contain at least two samples.")
        if values.shape[0] != knots.size:
            raise ValueError("values leading axis must match knots.")
        if not bool(np.all(np.diff(knots) > 0)):
            raise ValueError("knots must be strictly increasing.")
        self.knots = knots
        self.values = values
        self.method = str(method)
        self.extrapolate = extrapolate

    def evaluate(self, *, variables, **context) -> Array:
        """Interpolate values at ``variables`` using the configured method.

        ``variables`` uses the same coordinate unit as ``knots``. Its shape becomes
        the leading output shape ahead of any trailing value dimensions.
        """
        return ipx.interp1d(
            variables,
            self.knots,
            self.values,
            method=self.method,
            extrap=self.extrapolate,
        )

    def integrate(self, lower, upper, **context) -> Array:
        """Exactly integrate a scalar piecewise-linear interpolation.

        Broadcastable ``lower`` and ``upper`` bounds use the knot coordinate unit.
        Integration currently requires ``method="linear"``, scalar values, and zero
        extrapolation.
        """
        if self.method != "linear":
            raise NotImplementedError(
                "Exact Interpolation integration currently requires method='linear'."
            )
        if self.values.ndim != 1:
            raise NotImplementedError(
                "Interpolation integration currently requires scalar values."
            )
        if self.extrapolate not in (False, 0, 0.0):
            raise NotImplementedError(
                "Interpolation integration currently requires zero extrapolation."
            )

        lower, upper = np.broadcast_arrays(
            np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        )
        x0, x1 = self.knots[:-1], self.knots[1:]
        y0, y1 = self.values[:-1], self.values[1:]
        lower = lower[..., None]
        upper = upper[..., None]
        start = np.maximum(lower, x0)
        stop = np.minimum(upper, x1)
        active = stop > start
        start = np.where(active, start, x0)
        stop = np.where(active, stop, x0)
        slope = (y1 - y0) / (x1 - x0)
        integral = y0 * (stop - start)
        integral += slope * ((stop - x0) ** 2 - (start - x0) ** 2) / 2
        return np.where(active, integral, 0.0).sum(-1)


class DynamicParametric(Parametric):
    """Evaluate any coordinate-dependent parameterisation in a transformed frame."""

    parametric: Parametric
    transformation: BaseCoordTransform

    def __init__(self, parametric, transformation):
        """Initialise coordinate transformation of a parametric.

        Parameters
        ----------
        parametric : Parametric
            Coordinate-dependent value generator.
        transformation : BaseCoordTransform
            Map applied to context coordinates before evaluation.
        """
        if not isinstance(parametric, Parametric):
            raise TypeError("parametric must be a Parametric.")
        if not isinstance(transformation, BaseCoordTransform):
            raise TypeError("transformation must be a BaseCoordTransform.")
        self.parametric = parametric
        self.transformation = transformation

    def evaluate(self, *, coordinates, **context) -> Array:
        """Evaluate the wrapped parameterisation in transformed coordinates.

        ``coordinates`` must follow ``(..., 2, ny, nx)``. The configured coordinate
        transformation is applied first and forwarded under the same context key.
        """
        return self.parametric.evaluate(
            coordinates=self.transformation(coordinates), **context
        )


class Combination(Parametric):
    """Combine an ordered collection of parameterisations with one operation."""

    parametrics: dict
    operation: str = eqx.field(static=True)

    def __init__(self, parametrics, operation="sum"):
        """Initialise a combination of parametric values.

        Parameters
        ----------
        parametrics : mapping or sequence of Parametric
            Named or unnamed values evaluated in a shared context.
        operation : str
            Supported reduction operation applied in insertion order.
        """
        if isinstance(parametrics, dict):
            parametrics = list(parametrics.items())
        else:
            parametrics = list(parametrics)
        self.parametrics = dlu.list2dictionary(parametrics, True, Parametric)
        self.operation = self.validate_operation(operation)

    @staticmethod
    def validate_operation(operation):
        """Return a lower-case supported combination operation.

        Accepted values are ``"sum"``, ``"product"``, ``"union"``, and
        ``"intersection"``; other values raise `ValueError`.
        """
        operation = str(operation).lower()
        valid = ("sum", "product", "union", "intersection")
        if operation not in valid:
            raise ValueError(f"operation must be one of {valid}.")
        return operation

    @staticmethod
    def combine(values, operation):
        """Combine the leading axis of ``values`` with ``operation``.

        Sum and product reduce arithmetically. Union and intersection combine
        transmission-like values while preserving every remaining axis.
        """
        if operation in ("product", "intersection"):
            return values.prod(0)
        output = values.sum(0)
        return np.clip(output, 0.0, 1.0) if operation == "union" else output

    def values(self, **context) -> Array:
        """Evaluate every contained parameterisation and stack the results.

        Named context is shared by every child. Outputs must have compatible shapes
        and are stacked on a new leading axis.
        """
        return np.asarray(
            [parametric.evaluate(**context) for parametric in self.parametrics.values()]
        )

    def evaluate(self, **context) -> Array:
        """Evaluate all children and reduce them with the configured operation.

        Named context is shared by every child and the result retains their common
        sampled shape.
        """
        return self.combine(self.values(**context), self.operation)
