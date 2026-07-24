"""Tests for dLux.layers.unified_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_jittable


@pytest.fixture
def targets(make_wavefront, make_psf):
    return (make_wavefront(), make_psf())


@pytest.mark.parametrize(
    "layer",
    [
        dl.Resize(12),
        dl.Downsample(2),
        dl.Flip(0),
        dl.Flip((0, 1)),
        dl.Interpolate(
            dl.Affine(translation=[0.01, -0.02], rotation=0.1),
            method="linear",
        ),
        dl.Normalise(),
        dl.Normalise(mode="peak", value=2),
        dl.Lambda(),
    ],
)
def test_unified_layer_contract(layer, targets):
    assert isinstance(layer, dl.OpticalLayer)
    assert isinstance(layer, dl.DetectorLayer)

    for target in targets:
        output = assert_jittable(layer, target)
        assert type(output) is type(target)


def test_lambda_identity(targets):
    layer = dl.Lambda()
    for target in targets:
        assert layer(target) is target


def test_normalise_modes(targets):
    wavefront, psf = targets

    assert np.allclose(dl.Normalise("power", 2)(wavefront).power, 2)
    assert np.allclose(dl.Normalise("peak", 2)(psf).data.max(), 2)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Downsample(0),
        lambda: dl.Flip((0.0, 1)),
        lambda: dl.Interpolate("rotate"),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
