"""Tests for dLux.layers.optical_layers."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def wavefront(make_wavefront):
    return make_wavefront()


@pytest.mark.parametrize(
    "layer",
    [
        dl.TransmissiveLayer(),
        dl.TransmissiveLayer(0.5),
        dl.TransmissiveLayer(0.5, normalise=True),
        dl.AberratedLayer(),
        dl.AberratedLayer(opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, opd=1e-7, phase=0.2),
        dl.Optic(transmission=0.5, normalise=True),
        dl.Tilt([0.1, -0.2], unit="arcsec"),
    ],
)
def test_optical_layer_contract(layer, wavefront):
    output = assert_jittable(layer, wavefront)
    assert output.phasor.shape == wavefront.phasor.shape


def test_layer_alias_and_inheritance(wavefront):
    layer = dl.Optic(transmission=0.5, opd=1e-7, phase=0.2)

    assert isinstance(layer, dl.TransmissiveLayer)
    assert isinstance(layer, dl.AberratedLayer)
    assert np.allclose(layer.apply(wavefront).phasor, layer(wavefront).phasor)


def test_transmissive_layer_contract(wavefront):
    unchanged = dl.TransmissiveLayer()(wavefront)
    attenuated = dl.TransmissiveLayer(0.5)(wavefront)
    normalised = dl.TransmissiveLayer(0.5, normalise=True)(wavefront)

    assert np.allclose(unchanged.phasor, wavefront.phasor)
    assert np.allclose(attenuated.power, 0.25 * wavefront.power)
    assert np.allclose(normalised.power, 1)


def test_aberrated_layer_contract(wavefront):
    layer = dl.AberratedLayer(opd=1e-7, phase=0.2)
    expected = wavefront.add_opd(1e-7).add_phase(0.2)

    assert np.allclose(layer(wavefront).phasor, expected.phasor)


def test_optic_phasor_contract(wavefront):
    optic = dl.Optic(transmission=0.5, opd=1e-7, phase=0.2)
    params = optic.params(wavefront)

    assert optic.context(wavefront) == {"wavefront": wavefront}
    assert optic.phasor(wavefront).shape == (1, 1)
    assert np.allclose(
        optic.phasor(wavefront),
        optic.phasor(wavefront, params),
    )


@pytest.mark.parametrize(
    ("layer", "attribute"),
    [
        (dl.TransmissiveLayer(0.5), "transmission"),
        (dl.AberratedLayer(opd=1e-7), "opd"),
        (dl.AberratedLayer(phase=0.2), "phase"),
        (dl.Optic(transmission=0.5), "transmission"),
        (dl.Optic(opd=1e-7), "opd"),
        (dl.Optic(phase=0.2), "phase"),
        (dl.Tilt([0.1, -0.2]), "angles"),
    ],
)
def test_optical_layer_gradients(layer, attribute, wavefront):
    def apply(value):
        output = layer.set(attribute, value)(wavefront)
        if attribute in ("opd", "phase", "angles"):
            return np.real(output.phasor)
        return output

    assert_differentiable(
        apply,
        getattr(layer, attribute),
    )


def test_parametric_optic(wavefront):
    optic = dl.Optic(
        transmission=0.5,
        opd=dl.DynamicZernikeBasis(
            js=[4],
            coefficients=[1e-7],
            diameter=0.5,
        ),
    )

    assert_jittable(optic, wavefront)
    assert_differentiable(
        lambda coefficients: np.real(
            optic.set(
                "opd.coefficients",
                coefficients,
            )(wavefront).phasor
        ),
        optic.opd.coefficients,
    )


def test_tilt_validation():
    with pytest.raises(ValueError, match="shape"):
        dl.Tilt([1])
