import jax.numpy as np
import pytest

from dLux import Affine
from dLux.parametric import (
    BaseParametric,
    Combination,
    TransformedParametric,
)


class CoordinateValue(BaseParametric):
    def evaluate(self, *, coordinates, **kwargs):
        return coordinates[0]


def test_transformed_parametric():
    coordinates = np.zeros((2, 3, 3))
    transformed = TransformedParametric(
        CoordinateValue(), Affine(translation=[1.0, 0.0])
    )

    assert np.allclose(transformed.evaluate(coordinates=coordinates), -1)


def test_transformed_validation():
    with pytest.raises(TypeError, match="BaseParametric"):
        TransformedParametric(np.ones(2), Affine())
    with pytest.raises(TypeError, match="CoordTransform"):
        TransformedParametric(CoordinateValue(), np.eye(2))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("sum", np.array([[1.0, 3.0], [1.0, 3.0]])),
        ("product", np.array([[0.0, 2.0], [0.0, 2.0]])),
        ("union", np.array([[1.0, 1.0], [1.0, 1.0]])),
        ("intersection", np.array([[0.0, 2.0], [0.0, 2.0]])),
    ],
)
def test_combination(operation, expected):
    coordinates = np.stack(np.meshgrid(np.arange(2.0), np.arange(2.0), indexing="xy"))
    x = CoordinateValue()
    shifted_x = TransformedParametric(x, Affine(translation=[-1.0, 0.0]))
    composite = Combination([x, shifted_x], operation)

    assert composite.values(coordinates=coordinates).shape == (2, 2, 2)
    assert np.allclose(composite.evaluate(coordinates=coordinates), expected)


def test_combination_validation():
    with pytest.raises(ValueError, match="operation"):
        Combination([CoordinateValue()], "invalid")
    composite = Combination({"x": CoordinateValue()})
    assert tuple(composite.parametrics) == ("x",)
