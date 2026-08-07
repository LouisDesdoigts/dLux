"""Core contextual parameterisation and composition classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as np
import jax.tree as jtu
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from ..grids import CoordTransform

__all__ = [
    "Parametric",
    "ParametricHolder",
    "resolve",
    "Transform",
    "Interpolation",
    "DynamicParametric",
    "Combination",
]


class ParametricHolder(zdx.Base):
    """Base class for objects containing context-dependent parameters."""

    def resolve(self, **context):
        """Return a copy with every parametric leaf evaluated in ``context``."""
        is_parametric = lambda value: isinstance(value, Parametric)
        evaluate = lambda value: resolve(value, **context)
        return jtu.map(evaluate, self, is_leaf=is_parametric)


class Parametric(zdx.Base):
    """A contextual parameterisation consumed by another dLux object."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Array:
        """Evaluate the parameterisation in the supplied context."""

    def map(self, transformation) -> Parametric:
        """Apply a callable transformation to the realised value."""
        return Transform(self, transformation)

    def integrate(self, lower, upper, **context) -> Array:
        """Integrate the realised parameterisation between two bounds."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define spectral integration."
        )


def resolve(value: Any, dtype: Any = None, **context: Any) -> Any:
    """Evaluate a parameterisation and optionally cast the result."""
    value = value.evaluate(**context) if isinstance(value, Parametric) else value
    return value if value is None or dtype is None else dlu.to_value(value, dtype)


class Transform(Parametric):
    """Apply a callable transformation to a realised parameterisation."""

    parametric: Parametric
    transformation: Any

    def __init__(self, parametric, transformation):
        if not isinstance(parametric, Parametric):
            raise TypeError("parametric must be a Parametric.")
        if not callable(transformation):
            raise TypeError("transformation must be callable.")
        self.parametric = parametric
        self.transformation = transformation

    def evaluate(self, **context):
        """Evaluate the wrapped parameterisation and transform its value."""
        return self.transformation(self.parametric.evaluate(**context))


class Interpolation(Parametric):
    """A one-dimensional parameterisation defined by sampled values."""

    knots: Array
    values: Array
    method: str = eqx.field(static=True)
    extrapolate: bool | float = eqx.field(static=True)

    def __init__(self, knots, values, method="linear", extrapolate=0.0):
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
        """Interpolate at the supplied variables."""
        return ipx.interp1d(
            variables,
            self.knots,
            self.values,
            method=self.method,
            extrap=self.extrapolate,
        )

    def integrate(self, lower, upper, **context) -> Array:
        """Exactly integrate a piecewise-linear interpolation."""
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
    transformation: CoordTransform

    def __init__(self, parametric, transformation):
        if not isinstance(parametric, Parametric):
            raise TypeError("parametric must be a Parametric.")
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        self.parametric = parametric
        self.transformation = transformation

    def evaluate(self, *, coordinates, **context) -> Array:
        """Evaluate the wrapped parameterisation in transformed coordinates."""
        return self.parametric.evaluate(
            coordinates=self.transformation(coordinates), **context
        )


class Combination(Parametric):
    """Combine an ordered collection of parameterisations with one operation."""

    parametrics: dict
    operation: str = eqx.field(static=True)

    def __init__(self, parametrics, operation="sum"):
        if isinstance(parametrics, dict):
            parametrics = list(parametrics.items())
        else:
            parametrics = list(parametrics)
        self.parametrics = dlu.list2dictionary(parametrics, True, Parametric)
        self.operation = self.validate_operation(operation)

    @staticmethod
    def validate_operation(operation):
        """Validate and standardize a supported combination operation."""
        operation = str(operation).lower()
        valid = ("sum", "product", "union", "intersection")
        if operation not in valid:
            raise ValueError(f"operation must be one of {valid}.")
        return operation

    @staticmethod
    def combine(values, operation):
        """Combine a stack of realised values with one operation."""
        if operation in ("product", "intersection"):
            return values.prod(0)
        output = values.sum(0)
        return np.clip(output, 0.0, 1.0) if operation == "union" else output

    def values(self, **context) -> Array:
        """Evaluate every contained parameterisation into one stack."""
        return np.asarray(
            [parametric.evaluate(**context) for parametric in self.parametrics.values()]
        )

    def evaluate(self, **context) -> Array:
        """Evaluate and combine the contained parameterisations."""
        return self.combine(self.values(**context), self.operation)
