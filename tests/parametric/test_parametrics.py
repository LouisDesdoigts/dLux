"""Tests for dLux.parametric.parametrics."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class CoordinateValue(dl.BaseParametric):
    """Small concrete parametric used to exercise composition."""

    def evaluate(self, *, coordinates, **kwargs):
        return coordinates[0]


@pytest.fixture
def coordinates():
    return np.linspace(0.1, 0.4, 32).reshape(2, 4, 4)


def test_transformed_parametric_contract(coordinates):
    parametric = dl.TransformedParametric(
        CoordinateValue(),
        dl.Affine(translation=[0.1, 0.0]),
    )

    assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates),
        parametric,
    )
    assert_differentiable(
        lambda translation: parametric.set(
            "transformation.translation",
            translation,
        ).evaluate(coordinates=coordinates),
        parametric.transformation.translation,
    )


@pytest.mark.parametrize(
    "operation",
    ["sum", "product", "union", "intersection"],
)
def test_combination_contract(operation, coordinates):
    parametric = dl.Combination(
        [
            CoordinateValue(),
            dl.TransformedParametric(
                CoordinateValue(),
                dl.Affine(scale=[1.1, 0.9]),
            ),
        ],
        operation,
    )

    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates),
        parametric,
    )
    assert output.shape == coordinates.shape[-2:]
    assert_differentiable(
        lambda value: parametric.evaluate(coordinates=value),
        coordinates,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.TransformedParametric(np.ones(2), dl.Affine()),
        lambda: dl.TransformedParametric(CoordinateValue(), np.eye(2)),
        lambda: dl.Combination([CoordinateValue()], "invalid"),
        lambda: dl.Combination([np.ones(2)]),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
