"""Tests for dLux.layers.unified_layers."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.fixture
def targets(make_wavefront, make_psf):
    return (make_wavefront(), make_psf())


@pytest.mark.parametrize(
    "layer",
    [
        dl.ResizeLayer(12),
        dl.DownsampleLayer(2),
        dl.FlipLayer(0),
        dl.FlipLayer((0, 1)),
        dl.InterpolateLayer(
            dlu.Affine(translation=[0.01, -0.02], rotation=0.1), method="linear"
        ),
        dl.NormaliseLayer(),
        dl.NormaliseLayer(mode="peak", value=2),
        dl.LambdaLayer(),
    ],
)
def test_unified_layer_contract(layer, targets):
    assert isinstance(layer, dl.OpticalLayer)
    assert isinstance(layer, dl.DetectorLayer)

    for target in targets:
        output = assert_jittable(layer, target)
        applied = assert_jittable(layer.apply, target)

        assert type(output) is type(target)
        assert np.allclose(output.field, applied.field)


def test_lambda_identity(targets):
    layer = dl.LambdaLayer()
    for target in targets:
        assert layer(target) is target


def test_normalise_modes(targets):
    wavefront, psf = targets

    assert np.allclose(dl.NormaliseLayer("power", 2)(wavefront).power, 2)
    assert np.allclose(dl.NormaliseLayer("peak", 2)(psf).data.max(), 2)


@pytest.mark.parametrize(
    ("layer", "shape", "n", "d"),
    [
        (dl.ResizeLayer((10, 6)), (2, 3, 6, 10), (10, 6), (0.1, 0.1)),
        (dl.DownsampleLayer((2, 4)), (2, 3, 2, 4), (4, 2), (0.2, 0.4)),
        (dl.FlipLayer((-2, -1)), (2, 3, 8, 8), (8, 8), (0.1, 0.1)),
        (
            dl.InterpolateLayer(dlu.Affine(translation=[0.01, -0.02])),
            (2, 3, 8, 8),
            (8, 8),
            (0.1, 0.1),
        ),
    ],
)
def test_unified_layers_preserve_leading_axes(layer, shape, n, d, make_grid):
    grid = make_grid(n=(8, 8), d=(0.1, 0.1), c=(0.2, -0.1))
    targets = (
        dl.Wavefront(1e-6, grid, np.ones((2, 3, 8, 8), complex)),
        dl.Intensity(np.ones((2, 3, 8, 8)), grid),
    )

    for target in targets:
        output = assert_jittable(layer, target)
        assert output.field.shape == shape
        assert output.n == n
        assert np.allclose(output.d, np.asarray(d))
        assert np.allclose(output.c, target.c)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.DownsampleLayer(0),
        lambda: dl.FlipLayer((0.0, 1)),
        lambda: dl.InterpolateLayer("rotate"),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_interpolate_defaults_to_shared_cubic_interpolator(make_psf):
    intensity = make_psf(data=np.arange(64.0).reshape(8, 8) ** 2)
    transform = dlu.Affine(rotation=0.2)
    layer = dl.InterpolateLayer(transform)
    assert layer.method == "cubic2"
    assert np.allclose(layer(intensity).data, intensity.interpolate(transform).data)
