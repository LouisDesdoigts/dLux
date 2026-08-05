"""Tests for dLux.layers.polarised_layers."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def wavefront(make_wavefront):
    return make_wavefront(polarised=True)


@pytest.mark.parametrize(
    "layer",
    [
        dl.PolarisingOptic(np.eye(2)),
        dl.UniformPolarisingOptic(dlu.horizontal_polariser(), 0.1),
        dl.LinearPolariser(0.2),
        dl.Retarder(np.pi / 2, 0.1),
        dl.PolarisationLayer([dl.LinearPolariser(0.2), dl.Retarder(np.pi / 2, 0.1)]),
    ],
)
def test_polarisation_contract(layer, wavefront):
    output = assert_jittable(layer, wavefront)
    assert isinstance(output, dl.PolarisedWavefront)
    assert output.phasor.shape == wavefront.phasor.shape


def test_polarisation_layer_composition(wavefront):
    optics = [dl.LinearPolariser(0.2), dl.Retarder(np.pi / 2, 0.1)]
    layer = dl.PolarisationLayer(optics)
    sequential = optics[1](optics[0](wavefront))

    assert np.allclose(layer(wavefront).phasor, sequential.phasor)
    assert dl.PolarisationLayer()(wavefront) is wavefront


def test_promotes_scalar_wavefront(make_wavefront):
    output = dl.LinearPolariser()(make_wavefront())

    assert isinstance(output, dl.PolarisedWavefront)
    assert output.phasor.shape[-4:-2] == (2, 2)


def test_derived_jones_matrices():
    polariser = dl.LinearPolariser(0.2)
    retarder = dl.Retarder(np.pi / 2, 0.1)

    assert np.allclose(polariser.jones, dlu.linear_polariser(polariser.angle))
    assert np.allclose(
        retarder.jones, dlu.retarder(retarder.retardance, retarder.angle)
    )


@pytest.mark.parametrize(
    ("layer", "path"),
    [
        (dl.PolarisingOptic(np.eye(2)), "jones"),
        (dl.UniformPolarisingOptic(dlu.horizontal_polariser(), 0.1), "orientation"),
        (dl.LinearPolariser(0.2), "angle"),
        (dl.Retarder(np.pi / 2, 0.1), "retardance"),
        (dl.Retarder(np.pi / 2, 0.1), "angle"),
    ],
)
def test_polarisation_gradients(layer, path, wavefront):
    assert_differentiable(
        lambda value: layer.set(path, value)(wavefront), layer.get(path)
    )


@pytest.mark.parametrize(
    "layer",
    [
        dl.LinearPolariser(dl.CoordinatePolynomial(1, coefficients=[0.1, 0.01, -0.01])),
        dl.Retarder(
            dl.CoordinatePolynomial(0, coefficients=[np.pi / 2]),
            dl.CoordinatePolynomial(1, coefficients=[0.1, 0.01, -0.01]),
        ),
    ],
)
def test_parametric_polarisation(layer, wavefront):
    assert_jittable(layer, wavefront)


def test_uniform_validation():
    with pytest.raises(ValueError, match="\\(2, 2\\)"):
        dl.UniformPolarisingOptic(np.ones((2, 2, 4)))
