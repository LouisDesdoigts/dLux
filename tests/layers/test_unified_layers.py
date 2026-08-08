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
            dl.Affine(translation=[0.01, -0.02], rotation=0.1), method="linear"
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
    ("layer", "shape", "n", "d"),
    [
        (dl.Resize((10, 6)), (2, 3, 6, 10), (10, 6), (0.1, 0.1)),
        (dl.Downsample((2, 4)), (2, 3, 2, 4), (4, 2), (0.2, 0.4)),
        (dl.Flip((-2, -1)), (2, 3, 8, 8), (8, 8), (0.1, 0.1)),
        (
            dl.Interpolate(dl.Affine(translation=[0.01, -0.02])),
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
        dl.PSF(np.ones((2, 3, 8, 8)), grid),
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
        lambda: dl.Downsample(0),
        lambda: dl.Flip((0.0, 1)),
        lambda: dl.Interpolate("rotate"),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
