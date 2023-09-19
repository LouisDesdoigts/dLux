import pytest
import jax.numpy as np
from dLux.utils.coordinates import (
    cart2polar,
    nd_coords,
    pixel_coords,
    polar2cart,
    translate_coords,
    compress_coords,
    shear_coords,
    rotate_coords,
)

rtol = 1e-5
atol = 1e-7


def test_translate_coords():
    coords = np.array([[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]])
    centre = np.array([0.5, 0.5])
    expected = np.array([[[-0.5, 0.0, 0.5]], [[-0.5, 0.0, 0.5]]])
    assert np.allclose(translate_coords(coords, centre), expected)


def test_compress_coords():
    coords = np.array([[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]])
    compress = np.array([0.5, 1.0])
    expected = np.array([[[0.0, 0.25, 0.5]], [[0.0, 0.5, 1.0]]])
    assert np.allclose(compress_coords(coords, compress), expected)


def test_shear_coords():
    # Shear _requires_ a square coordinate array
    coords = np.array(
        [
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
        ]
    )
    shear = np.array([0.0, 0.5])
    expected = np.array(
        [
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
            [[0.0, 0.5, 1.0], [0.25, 0.75, 1.25], [0.5, 1.0, 1.5]],
        ]
    )
    assert np.allclose(shear_coords(coords, shear), expected)


def test_rotate_coords():
    coords = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    rotation = np.pi
    expected = np.array([[0.0, -0.5, -1.0], [0.0, -0.5, -1.0]])
    assert np.allclose(rotate_coords(coords, rotation), expected)


def test_cart2polar():
    # Test case 1
    coordinates = np.array([1, 0])
    expected = np.array([1, 0])
    actual = cart2polar(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 2
    coordinates = np.array([0, 1])
    expected = np.array([1, np.pi / 2])
    actual = cart2polar(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 3
    coordinates = np.array([-1, 0])
    expected = np.array([1, np.pi])
    actual = cart2polar(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 4
    coordinates = np.array([0, -1])
    expected = np.array([1, -np.pi / 2])
    actual = cart2polar(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)


def test_polar2cart():
    # Test case 1
    coordinates = np.array([1, 0])
    expected = np.array([1, 0])
    actual = polar2cart(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 2
    coordinates = np.array([1, np.pi / 2])
    expected = np.array([0, 1])
    actual = polar2cart(coordinates)
    print(actual, expected)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 3
    coordinates = np.array([1, np.pi])
    expected = np.array([-1, 0])
    actual = polar2cart(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)

    # Test case 4
    coordinates = np.array([1, -np.pi / 2])
    expected = np.array([0, -1])
    actual = polar2cart(coordinates)
    assert np.allclose(actual, expected, rtol=rtol, atol=atol)


def test_pixel_coords():
    # Test case 1
    npixels = 10
    diameter = 1.0
    polar = False
    expected_shape = (2, 10, 10)
    actual = pixel_coords(npixels, diameter, polar)
    assert actual.shape == expected_shape

    # Test case 2
    npixels = 10
    diameter = 1.0
    polar = True
    expected_shape = (2, 10, 10)
    actual = pixel_coords(npixels, diameter, polar)
    assert actual.shape == expected_shape

    # Test case 3
    npixels = 20
    diameter = 2.0
    polar = False
    expected_shape = (2, 20, 20)
    actual = pixel_coords(npixels, diameter, polar)
    assert actual.shape == expected_shape

    # Test case 4
    npixels = 20
    diameter = 2.0
    polar = True
    expected_shape = (2, 20, 20)
    actual = pixel_coords(npixels, diameter, polar)
    assert actual.shape == expected_shape


def test_nd_coords():
    # Test case 1
    npixels = 10
    pixel_scales = 1.0
    offsets = 0.0
    indexing = "xy"
    expected_shape = (10,)
    actual = nd_coords(npixels, pixel_scales, offsets, indexing)
    assert actual.shape == expected_shape

    # Test case 2
    npixels = (10, 20)
    pixel_scales = (1.0, 2.0)
    offsets = (0.0, 1.0)
    indexing = "ij"
    expected_shape = (2, 10, 20)
    actual = nd_coords(npixels, pixel_scales, offsets, indexing)
    assert actual.shape == expected_shape

    # Test case 3
    npixels = (10, 20, 30)
    pixel_scales = (1.0, 2.0, 3.0)
    offsets = (0.0, 1.0, 2.0)
    expected_shape = (3, 10, 20, 30)
    actual = nd_coords(npixels, pixel_scales, offsets, indexing)
    assert actual.shape == expected_shape

    # Test case 4
    with pytest.raises(ValueError):
        nd_coords(npixels, pixel_scales, offsets, "xi")
