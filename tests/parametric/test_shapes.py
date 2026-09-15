"""Tests for dLux.parametric.shapes."""

import jax.numpy as np
import zodiax as zdx
import pytest
from jax import grad

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
        (
            dl.ConvexPolygon([[-0.4, -0.4], [0.4, -0.4], [0.0, 0.4]], edge=1.0),
            None,
        ),
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


@pytest.mark.parametrize("edge", [None, 1.0])
@pytest.mark.parametrize(
    "shape_type,args",
    [
        (dl.Circle, (0.8,)),
        (dl.Square, (0.8,)),
        (dl.Rectangle, (0.8, 0.6)),
        (dl.RegularPolygon, (6, 0.8)),
        (dl.Spider, (0.1, [0.0, 90.0])),
    ],
)
def test_shape_preserves_coordinate_batches(shape_type, args, edge, context):
    offsets = np.array([[0.0, 0.0], [0.123, -0.077], [-0.119, 0.217]])
    coordinates = context["coordinates"][None] + offsets[:, :, None, None]
    shape = shape_type(*args, edge=edge)
    evaluate = lambda coords: shape.evaluate(
        coordinates=coords,
        pixel_scale=context["pixel_scale"],
    )

    output = assert_jittable(evaluate, coordinates)
    expected = np.stack([evaluate(coords) for coords in coordinates])

    assert output.shape == coordinates.shape[:-3] + coordinates.shape[-2:]
    assert np.allclose(output, expected)


def test_shape_grid_contract():
    shape = dl.Circle(0.8)
    grid = dl.GridSpec(n=12, diam=1.0, unit="m")

    assert shape(grid).shape == (12, 12)

    with pytest.raises(ValueError, match="one or two spatial dimensions"):
        shape(grid.broadcast(3))


@pytest.mark.parametrize("npix", [(4, 3), (2, 3)])
def test_soft_shape_batched_pixel_scales_match_individual_grids(npix):
    # nx=2 also catches accidental pairing of the batch width with image columns.
    spacing = np.asarray([[0.2, 0.2], [0.1, 0.1]])
    grid = dl.GridSpec(n=npix, d=spacing)
    shape = dl.Circle(0.5, edge=dl.Soft(1.0))

    output = assert_jittable(lambda model, sampling: model(sampling), shape, grid)
    expected = np.stack([shape(dl.GridSpec(n=npix, d=d)) for d in spacing])

    assert output.shape == (2,) + npix[::-1]
    assert np.allclose(output, expected)


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


@pytest.mark.parametrize(
    "vertices",
    [
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.4, 0.4], [0.0, 1.0]],
    ],
)
def test_convex_polygon_validation(vertices):
    polygon = dl.ConvexPolygon(vertices)
    with pytest.raises(ValueError):
        polygon.validate()


def test_soft_convex_polygon_has_vertex_gradients(context):
    vertices = np.array([[-0.4, -0.4], [0.4, -0.4], [0.0, 0.4]])

    def area(scale):
        polygon = dl.ConvexPolygon(vertices * scale, edge=1.0)
        return polygon.evaluate(**context).sum()

    assert not np.isclose(grad(area)(1.0), 0.0)


def test_shape_resolves_state_in_its_transformed_coordinate_frame(context):
    transform = dl.Affine().set(translation=zdx.StateRef("shift"))
    shape = dl.TransformedShape(
        dl.Circle(zdx.StateRef("diameter"), edge=1.0), transform
    )
    state = zdx.State(diameter=0.7, shift=[0.1, -0.05])
    result = assert_jittable(lambda model: model.resolve(state=state, **context), shape)
    expected = dl.TransformedShape(
        dl.Circle(0.7, edge=1.0), dl.Affine(translation=[0.1, -0.05])
    )
    assert np.allclose(result, expected.resolve(**context))


@pytest.mark.parametrize("edge", [None, 1.0])
@pytest.mark.parametrize(
    "constructor",
    [
        lambda size, edge: dl.Circle(size, edge),
        lambda size, edge: dl.Square(size, edge),
        lambda size, edge: dl.Rectangle(size, size * 0.8, edge),
        lambda size, edge: dl.RegularPolygon(6, size, edge),
        lambda size, edge: dl.Spider(size / 10, [0.0, 90.0], edge),
    ],
)
def test_shape_dimension_batches_match_independent_models(constructor, edge):
    grid = dl.GridSpec(n=(7, 5), d=0.2, unit="m")
    sizes = np.array([0.6, 0.9, 1.2])
    shape = constructor(sizes, edge)
    output = assert_jittable(lambda model: model.evaluate(grid=grid), shape)
    expected = np.stack([constructor(size, edge).evaluate(grid=grid) for size in sizes])
    assert output.shape == (3, 5, 7)
    assert np.allclose(output, expected, atol=1e-6)
