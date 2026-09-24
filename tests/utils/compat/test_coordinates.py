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


@pytest.mark.parametrize("parameter", ("translation", "scale", "shear"))
def test_batched_coordinate_transform_contract(parameter):
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    coordinates = np.stack((coordinates, coordinates + 0.1))
    transform = dlu.Affine(**{parameter: np.asarray((0.1, -0.2))})
    assert assert_jittable(transform, coordinates).shape == coordinates.shape


@pytest.mark.parametrize("operation", [dlu.translate_coords, dlu.compress_coords])
def test_nd_coordinate_transform_contract(operation):
    coordinates = dlu.nd_coords((6, 4, 3), (0.1, 0.2, 0.3), indexing="ij")
    output = assert_jittable(operation, coordinates, np.asarray((0.1, 0.2, 0.3)))
    assert output.shape == coordinates.shape


def test_shear_contract():
    coordinates = dlu.nd_coords((6, 6), (0.1, 0.2))
    output = assert_jittable(dlu.shear_coords, coordinates, np.asarray((0.1, -0.2)))
    assert output.shape == coordinates.shape


def test_shear_non_square():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    assert dlu.Affine(shear=np.asarray((0.1, -0.2)))(coordinates).shape == (
        coordinates.shape
    )


def test_polynomial_distortion():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
    coordinates = np.stack((coordinates, coordinates + 0.1))
    powers = dlu.polynomial_powers(2, 2)
    coeffs = np.ones((2, powers.shape[-1])) * 0.01
    output = assert_jittable(
        dlu.Distortion(powers=powers, distortion=coeffs), coordinates
    )
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
    expected = np.squeeze(np.stack(np.meshgrid(*axes, indexing="xy")))
    assert np.allclose(coordinates, expected)


def test_nd_coordinate_sampling_retains_metadata_batches_under_jit():
    spacing = np.asarray([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
    offsets = np.asarray([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    sample = lambda d, c: dlu.nd_coords((4, 3, 2), d, c, indexing="xy")
    actual = assert_jittable(sample, spacing, offsets)
    expected = np.stack([sample(d, c) for d, c in zip(spacing, offsets)])
    assert actual.shape == (2, 3, 3, 4, 2)
    assert np.allclose(actual, expected)


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
