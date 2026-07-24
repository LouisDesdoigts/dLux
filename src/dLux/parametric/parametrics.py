"""Core contextual parameterisation and composition classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu
from ..coordinates import CoordTransform

__all__ = [
    "BaseParametric",
    "TransformedParametric",
    "Combination",
]


class BaseParametric(zdx.Base):
    """A contextual parameterisation consumed by another dLux object."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> Array:  # pragma: no cover
        """Evaluate the parameterisation in the supplied context."""


class TransformedParametric(BaseParametric):
    """Evaluate any coordinate-dependent parameterisation in a transformed frame."""

    parametric: BaseParametric
    transformation: CoordTransform

    def __init__(self, parametric, transformation):
        if not isinstance(parametric, BaseParametric):
            raise TypeError("parametric must be a BaseParametric.")
        if not isinstance(transformation, CoordTransform):
            raise TypeError("transformation must be a CoordTransform.")
        self.parametric = parametric
        self.transformation = transformation

    def evaluate(self, *, coordinates, **context) -> Array:
        return self.parametric.evaluate(
            coordinates=self.transformation(coordinates), **context
        )


class Combination(BaseParametric):
    """Combine an ordered collection of parameterisations with one operation."""

    parametrics: dict
    operation: str = eqx.field(static=True)

    def __init__(self, parametrics, operation="sum"):
        if isinstance(parametrics, dict):
            parametrics = list(parametrics.items())
        else:
            parametrics = list(parametrics)
        self.parametrics = dlu.list2dictionary(parametrics, True, BaseParametric)
        self.operation = self.validate_operation(operation)

    @staticmethod
    def validate_operation(operation):
        operation = str(operation).lower()
        valid = ("sum", "product", "union", "intersection")
        if operation not in valid:
            raise ValueError(f"operation must be one of {valid}.")
        return operation

    @staticmethod
    def combine(values, operation):
        if operation in ("product", "intersection"):
            return values.prod(0)
        output = values.sum(0)
        return np.clip(output, 0.0, 1.0) if operation == "union" else output

    def values(self, **context) -> Array:
        return np.asarray(
            [parametric.evaluate(**context) for parametric in self.parametrics.values()]
        )

    def evaluate(self, **context) -> Array:
        return self.combine(self.values(**context), self.operation)
