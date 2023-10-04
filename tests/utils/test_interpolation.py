import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)
from jax import random
from dLux.utils.interpolation import generate_coordinates, scale, rotate


@pytest.fixture
def npixels_in():
    return 10


@pytest.fixture
def npixels_out():
    return 20


@pytest.fixture
def sampling_ratio():
    return np.array(2.0)


@pytest.fixture
def x_shift():
    return np.array(1.0)


@pytest.fixture
def y_shift():
    return np.array(-1.0)


@pytest.fixture
def array():
    key = random.PRNGKey(0)
    return random.normal(key, (10, 10))


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
def order():
    return 1


def test_generate_coordinates(
    npixels_in, npixels_out, sampling_ratio, x_shift, y_shift
):
    result = generate_coordinates(
        npixels_in, npixels_out, sampling_ratio, x_shift, y_shift
    )
    assert result.shape == (2, npixels_out, npixels_out)


def test_scale(array, npixels, ratio, order):
    result = scale(array, npixels, ratio, order)
    assert result.shape == (npixels, npixels)


def test_rotate(array, angle, order):
    result = rotate(array, angle, order)
    assert result.shape == (10, 10)
