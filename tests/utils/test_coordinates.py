"""Tests for dLux.utils.coordinates."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    ("operation", "parameter"),
    [
        (dlu.translate_coords, np.asarray((0.1, -0.2))),
        (dlu.compress_coords, np.asarray((0.5, 2.0))),
        (dlu.rotate_coords, 0.3),
    ],
)
def test_coordinate_transform_contract(operation, parameter):
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    output = assert_jittable(operation, coordinates, parameter)
    assert output.shape == coordinates.shape


def test_shear_contract():
    coordinates = dlu.nd_coords((6, 6), (0.1, 0.2))
    output = assert_jittable(dlu.shear_coords, coordinates, np.asarray((0.1, -0.2)))
    assert output.shape == coordinates.shape


def test_shear_non_square():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    assert dlu.shear_coords(coordinates, np.asarray((0.1, -0.2))).shape == (
        coordinates.shape
    )


def test_coordinate_system_roundtrip():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    polar = assert_jittable(dlu.cart2polar, coordinates)
    assert np.allclose(dlu.polar2cart(polar), coordinates, atol=1e-6)


def test_polynomial_distortion():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    powers = dlu.polynomial_powers(2, 2)
    coefficients = np.ones((2, powers.shape[-1])) * 0.01
    output = assert_jittable(dlu.distort_coords, coordinates, coefficients, powers)
    assert output.shape == coordinates.shape


@pytest.mark.parametrize(
    ("npixels", "scales", "shape"),
    [
        ((8,), (0.1,), (8,)),
        ((8, 6), (0.1, 0.2), (2, 6, 8)),
        ((8, 6, 4), (0.1, 0.2, 0.3), (3, 6, 8, 4)),
    ],
)
def test_nd_coordinate_generation(npixels, scales, shape):
    axes = dlu.nd_axes(npixels, scales)
    coordinates = dlu.nd_coords(npixels, scales)

    assert tuple(axis.size for axis in axes) == npixels
    assert coordinates.shape == shape


@pytest.mark.parametrize("scale", ["diameter", "radius", "pixel_scale"])
def test_pixel_coordinates(scale):
    output = dlu.pixel_coords(8, **{scale: 1.0})
    assert output.shape == (2, 8, 8)


def test_fft_and_polar_pixel_coordinates():
    centered = dlu.pixel_coords(8, diameter=1.0)
    fft_centered = dlu.pixel_coords(8, diameter=1.0, fft_style=True)
    polar = dlu.pixel_coords(8, diameter=1.0, polar=True)

    assert np.allclose(fft_centered - centered, -1 / 16)
    assert np.allclose(polar, dlu.cart2polar(centered))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.pixel_coords(8),
        lambda: dlu.pixel_coords(8, diameter=1.0, radius=0.5),
        lambda: dlu.nd_coords((4, 6), indexing="bad"),
    ],
)
def test_validation(operation):
    with pytest.raises(ValueError):
        operation()
