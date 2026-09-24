"""Tests for deterministic detector layers."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def intensity(make_psf):
    data = np.arange(64, dtype=float).reshape(8, 8) + 1
    return make_psf(data=data)


@pytest.mark.parametrize(
    "layer",
    [
        dl.SensitivityLayer(np.ones((8, 8))),
        dl.ConvolveLayer(np.ones((3, 3)) / 9),
        dl.ConvolveLayer(dlu.Jitter(0.5, kernel_size=5)),
        dl.BiasLayer(1),
        dl.GainLayer(2),
        dl.SaturationLayer(32),
    ],
)
def test_detector_layer_contract(layer, intensity):
    output = assert_jittable(layer, intensity, rtol=1e-5, atol=1e-5)
    applied = assert_jittable(layer.apply, intensity, rtol=1e-5, atol=1e-5)

    assert isinstance(output, dl.Intensity)
    assert output.data.shape == intensity.data.shape
    assert np.allclose(output.data, applied.data)


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (dl.SensitivityLayer(np.ones((8, 8))), "response"),
        (dl.ConvolveLayer(np.ones((3, 3)) / 9), "kernel"),
        (dl.ConvolveLayer(dlu.Jitter(0.5, kernel_size=5)), "kernel.profile.std"),
        (dl.BiasLayer(1), "bias"),
        (dl.GainLayer(2), "gain"),
        (dl.SaturationLayer(32), "limit"),
    ],
)
def test_detector_layer_gradients(layer, path, intensity):
    function = lambda value: layer.set(path, value)(intensity)

    # XLA convolution changes float32 accumulation order for convolution layers
    rtol = 3e-5 if isinstance(layer, dl.ConvolveLayer) else 1e-6
    gradient = assert_differentiable(function, layer.get(path), rtol=rtol)
    assert np.any(gradient != 0)


def test_parametric_detector_layers(intensity):
    spatial = dlu.Basis(np.ones((1, 8, 8)), [2.0])
    kernel = dlu.Basis(np.ones((1, 3, 3)) / 9, [1.0])
    scalar = dlu.Basis(np.ones(1), [2.0])
    std = dlu.Basis(np.ones(1), [0.5])
    polynomial = dlu.Polynomial(degree=1, coeffs=[1.0, 0.5])

    layers = [
        dl.SensitivityLayer(spatial),
        dl.ConvolveLayer(kernel),
        dl.ConvolveLayer(dlu.Jitter(dlu.Gaussian(std), kernel_size=5)),
        dl.BiasLayer(spatial),
        dl.GainLayer(polynomial),
        dl.SaturationLayer(scalar),
    ]
    outputs = [assert_jittable(layer, intensity, rtol=1e-5) for layer in layers]

    assert all(isinstance(output, dl.Intensity) for output in outputs)
    assert np.allclose(outputs[0].data, 2 * intensity.data)
    assert np.allclose(outputs[1].data, intensity.convolve(kernel.evaluate()).data)
    assert np.allclose(outputs[3].data, intensity.data + 2)
    gain = 1 + 0.5 * intensity.data
    assert np.allclose(outputs[4].data, intensity.data * gain)
    assert np.allclose(outputs[5].data, np.minimum(intensity.data, 2))


@pytest.mark.parametrize(
    ("profile", "shape"),
    [
        (0.5, (5, 5)),
        ([0.5, 1.0], (5, 7)),
        ([[0.25, 0.1], [0.1, 1.0]], (5, 7)),
    ],
)
def test_jitter_kernel(profile, shape):
    layer = dlu.Jitter(profile, kernel_size=shape[::-1], oversample=(2, 3))
    kernel = assert_jittable(lambda value: value.resolve(), layer)

    assert kernel.shape == shape
    assert np.allclose(kernel.sum(), 1)

    coordinates = dlu.nd_coords(shape[::-1])
    assert np.allclose((kernel * coordinates).sum((-2, -1)), 0, atol=1e-6)


def test_zero_jitter(intensity):
    layer = dlu.Jitter(0.0, kernel_size=5)
    expected = np.zeros((5, 5)).at[2, 2].set(1.0)

    assert np.allclose(layer.resolve(), expected)
    assert np.allclose(
        assert_jittable(dl.ConvolveLayer(layer), intensity).data, intensity.data
    )


@pytest.mark.parametrize(
    "layer",
    [
        dl.SensitivityLayer(np.linspace(0.5, 1.0, 64).reshape(8, 8)),
        dl.ConvolveLayer(np.ones((3, 3)) / 9),
        dl.ConvolveLayer(dlu.Jitter(0.5, kernel_size=5)),
        dl.BiasLayer(1),
        dl.GainLayer(2),
        dl.SaturationLayer(32),
    ],
)
def test_detector_layers_preserve_leading_axes(layer, make_psf):
    intensity = make_psf(data=np.arange(2 * 3 * 64).reshape(2, 3, 8, 8) + 1)
    output = assert_jittable(layer, intensity, rtol=1e-5, atol=1e-5)

    apply = lambda image: layer(make_psf(data=image)).data
    rows = [np.stack([apply(image) for image in row]) for row in intensity.data]
    expected = np.stack(rows)

    assert output.data.shape == intensity.data.shape
    assert output.grid == intensity.grid
    assert np.allclose(output.data, expected)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.ConvolveLayer(np.ones(8)),
        lambda: dlu.Jitter(0.5, kernel_size=0),
        lambda: dlu.Jitter(0.5, kernel_size=4),
        lambda: dlu.Jitter([0.5]),
        lambda: dlu.Jitter(0.5, oversample=0),
    ],
)
def test_validation(constructor):
    with pytest.raises(ValueError):
        constructor()


def test_expression_kernel_is_checked_after_resolution(intensity):
    import zodiax as zdx

    layer = dl.ConvolveLayer(zdx.StateRef("kernel"))
    with pytest.raises(ValueError, match="2d"):
        layer(intensity, state=zdx.State(kernel=np.ones(3)))
