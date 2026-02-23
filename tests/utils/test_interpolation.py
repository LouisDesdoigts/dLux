import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)
from jax import random
from dLux.utils.interpolation import interp, scale, rotate
from dLux.utils import pixel_coords


@pytest.fixture
def array():
    key = random.PRNGKey(0)
    return random.normal(key, (10, 10))


@pytest.fixture
def knot_coords():
    return pixel_coords(10, 1)


@pytest.fixture
def sample_coords():
    return pixel_coords(20, 1)


@pytest.fixture
def npixels():
    return 20


@pytest.fixture
def ratio():
    return 2.0


@pytest.fixture
def angle():
    return np.pi / 4


@pytest.fixture
def method():
    return "linear"


@pytest.fixture
def fill():
    return 0.0


def test_interp(array, knot_coords, sample_coords, method, fill):
    result = interp(array, knot_coords, sample_coords, method, fill)
    assert result.shape == (20, 20)


def test_scale(array, npixels, ratio, method):
    result = scale(array, npixels, ratio, method)
    assert result.shape == (npixels, npixels)


def test_rotate(array, angle, method):
    result = rotate(array, angle, method)
    assert result.shape == (10, 10)
