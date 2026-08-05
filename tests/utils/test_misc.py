"""Tests for dLux.utils.misc."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.ones((6, 6)), 1.0),
        (-np.ones((6, 6)), 0.0),
    ],
)
def test_soft_binarise_constant_regions(array, expected):
    output = assert_jittable(dlu.soft_binarise, array, 3)
    assert output.shape == (2, 2)
    assert np.all(output == expected)


def test_soft_binarise_boundary():
    x = np.linspace(-1, 1, 6)
    output = assert_jittable(dlu.soft_binarise, np.broadcast_to(x, (6, 6)), 3)
    assert output.shape == (2, 2)
    assert np.all((output >= 0) & (output <= 1))


def test_soft_binarise_validation():
    with pytest.raises(ValueError):
        dlu.soft_binarise(np.ones((5, 5)), 3)
