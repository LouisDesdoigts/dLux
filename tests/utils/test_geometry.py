"""Tests for dLux.utils.geometry."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable

COORDS = dlu.pixel_coords(16, diameter=2)


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (dlu.circle, (1.0,)),
        (dlu.square, (1.0,)),
        (dlu.rectangle, (1.0, 0.5)),
        (dlu.reg_polygon, (1.0, 6)),
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
        (dlu.soft_spider, (0.1, (0, 90))),
    ],
)
def test_soft_shape_contract(operation, args):
    output = assert_jittable(operation, COORDS, *args)
    assert output.shape == COORDS.shape[1:]
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0) & (output <= 1))


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
