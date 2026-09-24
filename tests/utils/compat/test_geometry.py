"""Tests for dLux.utils.geometry."""

import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable

COORDS = dlu.pixel_coords(16, diameter=2)
BATCH_OFFSETS = np.array([[0.0, 0.0], [0.2, -0.1], [-0.15, 0.25]])
BATCHED_COORDS = COORDS[None] + BATCH_OFFSETS[:, :, None, None]


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (dlu.circle, (1.0,)),
        (dlu.square, (1.0,)),
        (dlu.rectangle, (1.0, 0.5)),
        (dlu.reg_polygon, (1.0, 6)),
        (dlu.convex_polygon, (np.array([[-0.5, -0.5], [0.5, -0.5], [0, 0.5]]),)),
        (dlu.spider, (0.1, (0, 90))),
    ],
)
def test_hard_shape_contract(operation, args):
    output = assert_jittable(operation, COORDS, *args)
    assert output.shape == COORDS.shape[1:]
    assert np.all((output == 0) | (output == 1))


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (dlu.soft_circle, (1.0,)),
        (dlu.soft_square, (1.0,)),
        (dlu.soft_rectangle, (1.0, 0.5)),
        (dlu.soft_reg_polygon, (1.0, 6)),
        (
            dlu.soft_convex_polygon,
            (np.array([[-0.5, -0.5], [0.5, -0.5], [0, 0.5]]),),
        ),
        (dlu.soft_spider, (0.1, (0, 90))),
    ],
)
def test_soft_shape_contract(operation, args):
    output = assert_jittable(operation, COORDS, *args)
    assert output.shape == COORDS.shape[1:]
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0) & (output <= 1))


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (dlu.circle, (1.0,)),
        (dlu.square, (1.0,)),
        (dlu.rectangle, (1.0, 0.5)),
        (dlu.reg_polygon, (1.0, 6)),
        (dlu.spider, (0.1, (0, 90))),
        (dlu.soft_circle, (1.0,)),
        (dlu.soft_square, (1.0,)),
        (dlu.soft_rectangle, (1.0, 0.5)),
        (dlu.soft_reg_polygon, (1.0, 6)),
        (dlu.soft_spider, (0.1, (0, 90))),
    ],
)
def test_shapes_preserve_coordinate_batches(operation, args):
    sample = jax.vmap(lambda coordinates: operation(coordinates, *args))
    output = assert_jittable(sample, BATCHED_COORDS)
    expected = np.stack([operation(coords, *args) for coords in BATCHED_COORDS])

    assert output.shape == BATCHED_COORDS.shape[:-3] + COORDS.shape[-2:]
    assert np.allclose(output, expected)


def test_combine():
    arrays = np.stack((dlu.circle(COORDS, 1), dlu.square(COORDS, 1)))
    assert np.array_equal(dlu.combine(arrays), np.prod(arrays, axis=0))
    assert dlu.combine(arrays, use_sum=True).shape == COORDS.shape[1:]


@pytest.mark.parametrize(
    ("operation", "args"), [(dlu.square, (1.0,)), (dlu.reg_polygon, (1.0, 6))]
)
def test_inverted_shapes(operation, args):
    regular = operation(COORDS, *args)
    inverted = operation(COORDS, *args, invert=True)

    assert np.array_equal(inverted, 1 - regular)


def test_convex_polygon_winding_boundary_and_batching():
    vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    coordinates = np.array([[[0.0, 0.5, 0.6]], [[0.0, 0.0, 0.0]]])

    forward = dlu.convex_polygon(coordinates, vertices)
    reverse = dlu.convex_polygon(coordinates, vertices[::-1])
    batched = dlu.convex_polygon(COORDS, np.stack((vertices, vertices * 0.5)))

    assert np.array_equal(forward, np.asarray([[1.0, 1.0, 0.0]]))
    assert np.array_equal(reverse, forward)
    assert batched.shape == (2, *COORDS.shape[-2:])
    assert np.array_equal(batched[0], dlu.convex_polygon(COORDS, vertices))
    assert np.array_equal(batched[1], dlu.convex_polygon(COORDS, vertices * 0.5))


def test_legacy_circle_boundary_and_shape_complement():
    coordinates = np.array([[[0.0, 0.5, 0.75]], [[0.0, 0.0, 0.0]]])

    # Released functions exclude the exact boundary with either inversion choice.
    assert np.array_equal(dlu.circle(coordinates, 0.5), np.asarray([[1.0, 0.0, 0.0]]))
    assert np.array_equal(
        dlu.circle(coordinates, 0.5, invert=True), np.asarray([[0.0, 0.0, 1.0]])
    )

    # Classes always complement the evaluated support, including boundary samples.
    shape = dlu.Circle(1.0, invert=True)
    assert np.array_equal(
        shape.resolve(coordinates=coordinates), np.asarray([[0.0, 1.0, 1.0]])
    )


def test_legacy_spider_polarity_is_preserved():
    coordinates = np.array([[[0.0, 0.3, 0.0]], [[-0.3, -0.3, 0.3]]])
    on_arms = np.array([[1.0, 0.0, 0.0]])

    assert np.array_equal(dlu.spider(coordinates, 0.2, [0.0]), 1 - on_arms)
    assert np.array_equal(dlu.soft_spider(coordinates, 0.2, [0.0]), on_arms)
    assert np.array_equal(
        dlu.Spider(0.2, [0.0]).resolve(coordinates=coordinates), on_arms
    )
