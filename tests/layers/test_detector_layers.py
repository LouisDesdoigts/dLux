"""Tests for dLux.layers.detector_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def psf(make_psf):
    data = np.arange(64, dtype=float).reshape(8, 8) + 1
    return make_psf(data=data)


@pytest.mark.parametrize(
    "layer",
    [
        dl.ApplyPixelResponse(np.ones((8, 8))),
        dl.ApplyJitter(0.5, kernel_size=5),
        dl.ApplySaturation(32),
        dl.AddConstant(1),
    ],
)
def test_detector_layer_contract(layer, psf):
    output = assert_jittable(layer, psf)
    assert isinstance(output, dl.PSF)
    assert output.data.shape == psf.data.shape


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (dl.ApplyPixelResponse(np.ones((8, 8))), "pixel_response"),
        (dl.ApplyJitter(0.5, kernel_size=5), "sigma"),
        (dl.ApplySaturation(32), "threshold"),
        (dl.AddConstant(1), "value"),
    ],
)
def test_detector_layer_gradients(layer, path, psf):
    assert_differentiable(
        lambda value: layer.set(path, value)(psf),
        layer.get(path),
    )


def test_jitter_kernel():
    layer = dl.ApplyJitter(0.5, kernel_size=5, oversample=3)
    assert layer.kernel.shape == (5, 5)
    assert np.allclose(layer.kernel.sum(), 1)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.ApplyPixelResponse(np.ones(8)),
        lambda: dl.ApplyJitter(0.5, kernel_size=0),
    ],
)
def test_validation(constructor):
    with pytest.raises(ValueError):
        constructor()
