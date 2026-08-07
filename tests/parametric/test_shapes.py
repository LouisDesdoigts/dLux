"""Tests for dLux.parametric.shapes."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


class UnboundedShape(dl.Shape):
    def evaluate(self, **context):
        return np.asarray(1.0)


@pytest.fixture
def context():
    return {"coordinates": dlu.pixel_coords(12, 2.0), "pixel_scale": 2.0 / 12}


@pytest.mark.parametrize(
    ("shape", "parameter"),
    [
        (dl.Circle(0.8), "diameter"),
        (dl.Square(0.8), "width"),
        (dl.Rectangle(0.8, 0.6), "width"),
        (dl.RegularPolygon(6, 0.8), "diameter"),
        (dl.Spider(0.1, [0.0, 90.0]), None),
    ],
)
def test_shape_contract(shape, parameter, context):
    output = assert_jittable(lambda value: value.evaluate(**context), shape)

    assert output.shape == context["coordinates"].shape[-2:]
    assert np.all((output >= 0) & (output <= 1))
    if parameter is not None:
        assert_differentiable(
            lambda value: shape.set(parameter, value).evaluate(**context),
            getattr(shape, parameter),
        )


@pytest.mark.parametrize(
    "shape",
    [
        dl.Complement(dl.Circle(0.8)),
        dl.TransformedShape(dl.Circle(0.8), dl.Affine(translation=[0.1, 0.0])),
    ],
)
def test_composed_shape_contract(shape, context):
    output = assert_jittable(lambda value: value.evaluate(**context), shape)
    assert output.shape == context["coordinates"].shape[-2:]


def test_edge_contract(context):
    default = dl.Circle(0.8)
    hard = dl.Circle(0.8, edge=dl.Hard())
    soft = dl.Circle(0.8, edge=1.5)

    assert isinstance(default.edge, dl.Hard)
    assert isinstance(hard.edge, dl.Hard)
    assert isinstance(soft.edge, dl.Soft)
    assert np.allclose(default.evaluate(**context), hard.evaluate(**context))


def test_transformed_shape_gradient(context):
    shape = dl.TransformedShape(dl.Circle(0.8), dl.Affine(translation=[0.1, 0.0]))

    assert_differentiable(
        lambda value: shape.set("transformation.translation", value).evaluate(
            **context
        ),
        shape.transformation.translation,
    )
    assert np.isclose(shape.get("diameter"), 0.8)
    assert np.array_equal(shape.get("translation"), [0.1, 0.0])


def test_shape_extents():
    circle = dl.Circle(0.8)
    square = dl.Square(0.8)
    rectangle = dl.Rectangle(0.8, 0.6)

    assert UnboundedShape().extent is None
    assert np.isclose(circle.extent, 0.4)
    assert np.isclose(square.extent, 0.8 / np.sqrt(2))
    assert np.isclose(rectangle.extent, 0.5)
    assert np.isclose(dl.Complement(circle).extent, circle.extent)
    assert np.isclose(dl.TransformedShape(circle, dl.Affine()).extent, circle.extent)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Circle(0.0),
        lambda: dl.Square(0.0),
        lambda: dl.Rectangle(1.0, 0.0),
        lambda: dl.RegularPolygon(2, 1.0),
        lambda: dl.Spider(0.0, [0.0]),
        lambda: dl.Spider(1.0, [[0.0]]),
        lambda: dl.Complement(object()),
        lambda: dl.TransformedShape(dl.Circle(1.0), object()),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
