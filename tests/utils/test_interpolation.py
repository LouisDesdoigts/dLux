"""Tests for dLux.utils.interpolation."""

import jax.numpy as np
import pytest
from jax import random

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.fixture
def image():
    array = random.normal(random.PRNGKey(0), (8, 8))
    return array + 1j * np.flip(array, axis=0)


@pytest.mark.parametrize("complex", [True, False])
def test_interpolation_contract(image, complex):
    knots = dlu.pixel_coords(8, diameter=1.0)
    samples = dlu.pixel_coords(12, diameter=1.0)
    output = assert_jittable(dlu.interp, image, knots, samples, complex=complex)

    assert output.shape == (12, 12)
    assert np.iscomplexobj(output)


@pytest.mark.parametrize("complex", [True, False])
def test_scale_contract(image, complex):
    output = assert_jittable(dlu.scale, image, 12, 1.5, complex=complex)
    assert output.shape == (12, 12)


@pytest.mark.parametrize("complex", [True, False])
def test_rotate_contract(image, complex):
    output = assert_jittable(dlu.rotate, image, np.pi / 4, complex=complex)
    assert output.shape == image.shape


def test_rectangular_scale_and_rotate():
    image = np.arange(48.0).reshape((6, 8))

    scaled = assert_jittable(dlu.scale, image, (10, 12), (0.8, 1.2))
    rotated = assert_jittable(dlu.rotate, image, 0.2)

    assert scaled.shape == (12, 10)
    assert rotated.shape == image.shape
