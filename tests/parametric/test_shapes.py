import jax.numpy as np
import pytest

import dLux.utils as dlu
from dLux import Affine
from dLux.parametric.shapes import (
    Circle,
    Complement,
    Intersection,
    Rectangle,
    RegularPolygon,
    Shape,
    SoftShape,
    Spider,
    Square,
    TransformedShape,
    Union,
)


class InfiniteShape(Shape):
    def evaluate(self, *, coordinates, **kwargs):
        return np.ones(coordinates.shape[-2:])


class InfiniteSoftShape(SoftShape):
    def evaluate(self, *, coordinates, **kwargs):
        return np.ones(coordinates.shape[-2:])


@pytest.fixture
def context():
    return {"coordinates": dlu.pixel_coords(16, 2.0), "pixel_scale": 1 / 8}


def test_shape_base_and_softening():
    assert InfiniteShape().extent is None
    soft = InfiniteSoftShape(2)
    assert soft.clip(0.5) == 0.5
    with pytest.raises(ValueError, match="softening"):
        InfiniteSoftShape(0)


@pytest.mark.parametrize(
    "shape",
    [
        Circle(0.5),
        Square(1.0),
        Rectangle(1.0, 0.5),
        RegularPolygon(6, 0.5),
        Spider(0.1, [0, 90]),
    ],
)
def test_analytic_shapes(shape, context):
    transmission = shape.evaluate(**context)
    assert transmission.shape == (16, 16)
    assert np.all((transmission >= 0) & (transmission <= 1))
    assert shape.extent is None or shape.extent > 0


@pytest.mark.parametrize(
    "constructor,args,message",
    [
        (Circle, (0,), "diameter"),
        (Square, (0,), "width"),
        (Rectangle, (0, 1), "width and height"),
        (Rectangle, (1, 0), "width and height"),
        (RegularPolygon, (2, 1), "nsides"),
        (Spider, (0, [0]), "width"),
        (Spider, (1, [[0]]), "one-dimensional"),
    ],
)
def test_shape_validation(constructor, args, message):
    with pytest.raises(ValueError, match=message):
        constructor(*args)


def test_complement_and_transformation(context):
    circle = Circle(0.5)
    complement = Complement(circle)
    transformed = TransformedShape(circle, Affine(translation=[0.1, 0]))
    assert complement.extent == circle.extent
    assert transformed.extent == circle.extent
    assert np.allclose(complement.evaluate(**context), 1 - circle.evaluate(**context))
    assert transformed.evaluate(**context).shape == (16, 16)
    with pytest.raises(TypeError, match="Shape"):
        Complement(object())
    with pytest.raises(TypeError, match="Shape"):
        TransformedShape(object(), Affine())
    with pytest.raises(TypeError, match="CoordTransform"):
        TransformedShape(circle, object())


def test_composite_shapes(context):
    circle = Circle(0.5)
    square = Square(1.0)
    intersection = Intersection([circle, square])
    union = Union([circle, square])
    assert intersection.extent == square.extent
    assert intersection.transmissions(**context).shape == (2, 16, 16)
    assert intersection.evaluate(**context).shape == (16, 16)
    assert union.evaluate(**context).shape == (16, 16)
    assert Intersection([InfiniteShape()]).extent is None


def test_sparse_union(context):
    circle = Circle(0.2)
    aperture = Union(
        [
            TransformedShape(circle, Affine(translation=[-0.3, 0])),
            TransformedShape(circle, Affine(translation=[0.3, 0])),
        ]
    )
    assert aperture.extent == circle.extent
    assert aperture.evaluate(**context).shape == (16, 16)
