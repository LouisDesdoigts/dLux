from jax import numpy as np, config

config.update("jax_debug_nans", True)
import pytest
from dLux.utils.array_ops import pad_to, crop_to, resize, downsample


@pytest.mark.parametrize(
    "array, npixels",
    [
        (np.ones((10, 10)), 8),
        (np.ones((11, 11)), 9),
        (np.ones((10, 10)), 11),
        (np.ones((11, 11)), 12),
    ],
)
def test_pad_to_invalid(array, npixels):
    with pytest.raises(ValueError):
        pad_to(array, npixels)


@pytest.mark.parametrize(
    "array, npixels",
    [(np.ones((10, 10)), 12), (np.ones((11, 11)), 13)],
)
def test_pad_to(array, npixels):
    actual = pad_to(array, npixels)
    expected = (npixels, npixels)
    assert actual.shape == expected


@pytest.mark.parametrize(
    "array, npixels",
    [
        (np.ones((10, 10)), 12),
        (np.ones((11, 11)), 13),
        (np.ones((10, 10)), 9),
        (np.ones((11, 11)), 10),
    ],
)
def test_crop_to_invalid(array, npixels):
    with pytest.raises(ValueError):
        crop_to(array, npixels)


@pytest.mark.parametrize(
    "array, npixels",
    [(np.ones((10, 10)), 8), (np.ones((11, 11)), 9)],
)
def test_crop_to(array, npixels):
    actual = crop_to(array, npixels)
    expected = array[:npixels, :npixels]
    assert actual.shape == expected.shape


@pytest.mark.parametrize(
    "array, shape",
    [
        (np.ones((10, 10)), 12),
        (np.ones((10, 10)), 8),
        (np.ones((11, 11)), 13),
        (np.ones((11, 11)), 9),
        (np.ones((10, 10)), 10),
    ],
)
def test_resize(array, shape):
    actual = resize(array, shape)
    assert actual.shape == (shape, shape)


@pytest.mark.parametrize(
    "array, n, mean",
    [
        (np.ones((10, 9)), 2, True),
        (np.ones((10, 9)), 2, False),
        (np.ones((10, 10)), 3, True),
        (np.ones((10, 10)), 3, False),
    ],
)
def test_downsample_invalid(array, n, mean):
    with pytest.raises(ValueError):
        downsample(array, n, mean)


@pytest.mark.parametrize(
    "array, n, mean",
    [
        (np.ones((10, 10)), 2, True),
        (np.ones((9, 9)), 3, True),
        (np.ones((10, 10)), 5, True),
        (np.ones((15, 15)), 3, True),
        (np.ones((10, 10)), 2, False),
        (np.ones((9, 9)), 3, False),
        (np.ones((10, 10)), 5, False),
        (np.ones((15, 15)), 3, False),
    ],
)
def test_downsample(array, n, mean):
    actual = downsample(array, n, mean)
    assert actual.shape == (array.shape[0] // n, array.shape[1] // n)
