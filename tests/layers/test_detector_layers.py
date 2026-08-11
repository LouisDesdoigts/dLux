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
        dl.Sensitivity(np.ones((8, 8))),
        dl.Convolve(np.ones((3, 3)) / 9),
        dl.Jitter(0.5, kernel_size=5),
        dl.Bias(1),
        dl.Gain(2),
        dl.Saturation(32),
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
        (dl.Sensitivity(np.ones((8, 8))), "response"),
        (dl.Convolve(np.ones((3, 3)) / 9), "kernel"),
        (dl.Jitter(0.5, kernel_size=5), "sigma"),
        (dl.Bias(1), "bias"),
        (dl.Gain(2), "gain"),
        (dl.Saturation(32), "limit"),
    ],
)
def test_detector_layer_gradients(layer, path, intensity):
    function = lambda value: layer.set(path, value)(intensity)

    # XLA convolution changes float32 accumulation order for convolution layers
    rtol = 3e-5 if isinstance(layer, (dl.Convolve, dl.Jitter)) else 1e-6
    gradient = assert_differentiable(function, layer.get(path), rtol=rtol)
    assert np.any(gradient != 0)


def test_parametric_detector_layers(intensity):
    spatial = dl.Basis(np.ones((1, 8, 8)), [2.0])
    kernel = dl.Basis(np.ones((1, 3, 3)) / 9, [1.0])
    scalar = dl.Basis(np.ones(1), [2.0])
    sigma = dl.Basis(np.ones(1), [0.5])
    polynomial = dl.Polynomial(degree=1, coeffs=[1.0, 0.5])

    layers = [
        dl.Sensitivity(spatial),
        dl.Convolve(kernel),
        dl.Jitter(sigma, kernel_size=5),
        dl.Bias(spatial),
        dl.Gain(polynomial),
        dl.Saturation(scalar),
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
    ("sigma", "shape"),
    [
        (0.5, (5, 5)),
        ([0.5, 1.0], (5, 7)),
        ([[0.25, 0.1], [0.1, 1.0]], (5, 7)),
    ],
)
def test_jitter_kernel(sigma, shape):
    layer = dl.Jitter(sigma, kernel_size=shape[::-1], oversample=(2, 3))
    kernel = assert_jittable(layer.kernel)

    assert kernel.shape == shape
    assert np.allclose(kernel.sum(), 1)

    coordinates = dlu.nd_coords(shape[::-1])
    assert np.allclose((kernel * coordinates).sum((-2, -1)), 0, atol=1e-6)


def test_zero_jitter(intensity):
    layer = dl.Jitter(0.0, kernel_size=5)
    expected = np.zeros((5, 5)).at[2, 2].set(1.0)

    assert np.allclose(layer.kernel(), expected)
    assert np.allclose(assert_jittable(layer, intensity).data, intensity.data)


@pytest.mark.parametrize(
    "layer",
    [
        dl.Sensitivity(np.linspace(0.5, 1.0, 64).reshape(8, 8)),
        dl.Convolve(np.ones((3, 3)) / 9),
        dl.Jitter(0.5, kernel_size=5),
        dl.Bias(1),
        dl.Gain(2),
        dl.Saturation(32),
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
        lambda: dl.Convolve(np.ones(8)),
        lambda: dl.Jitter(0.5, kernel_size=0),
        lambda: dl.Jitter(0.5, kernel_size=4),
        lambda: dl.Jitter([0.5]),
        lambda: dl.Jitter(0.5, oversample=0),
    ],
)
def test_validation(constructor):
    with pytest.raises(ValueError):
        constructor()
