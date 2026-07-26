"""Tests for dLux.parametric.parametrics."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class CoordinateValue(dl.Parametric):
    """Small concrete parametric used to exercise composition."""

    def evaluate(self, *, coordinates, **kwargs):
        return coordinates[0]


class Square:
    def __call__(self, value):
        return value**2


@pytest.fixture
def coordinates():
    return np.linspace(0.1, 0.4, 32).reshape(2, 4, 4)


def test_dynamic_parametric_contract(coordinates):
    parametric = dl.DynamicParametric(
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


def test_value_transform_contract(coordinates):
    parametric = CoordinateValue().map(Square())
    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates),
        parametric,
    )
    assert np.allclose(output, coordinates[0] ** 2)


def test_interpolation_evaluation_and_integration():
    interpolation = dl.Interpolation(
        knots=np.asarray([0.0, 1.0, 2.0]),
        values=np.asarray([0.0, 1.0, 0.0]),
    )
    variables = np.asarray([0.5, 1.5])

    values = assert_jittable(
        lambda model: model.evaluate(variables=variables),
        interpolation,
    )
    integral = assert_jittable(
        lambda model: model.integrate(0.0, 2.0),
        interpolation,
    )

    assert np.allclose(values, 0.5)
    assert np.allclose(integral, 1.0)
    assert_differentiable(
        lambda samples: interpolation.set(values=samples).integrate(0.25, 1.75),
        interpolation.values,
    )


@pytest.mark.parametrize(
    "operation",
    ["sum", "product", "union", "intersection"],
)
def test_combination_contract(operation, coordinates):
    parametric = dl.Combination(
        [
            CoordinateValue(),
            dl.DynamicParametric(
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
        lambda: dl.DynamicParametric(np.ones(2), dl.Affine()),
        lambda: dl.DynamicParametric(CoordinateValue(), np.eye(2)),
        lambda: dl.Combination([CoordinateValue()], "invalid"),
        lambda: dl.Combination([np.ones(2)]),
        lambda: dl.Interpolation([0.0], [1.0]),
        lambda: dl.Interpolation([0.0, 1.0], [1.0]),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
