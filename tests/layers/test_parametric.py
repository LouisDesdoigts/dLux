import jax.numpy as np
from jax import Array

from dLux import Wavefront
from dLux.layers import AberratedLayer, Optic, TransmissiveLayer
from dLux.parametric import ExplicitBasis, Parametric

wavefront = Wavefront(1e-6, 8, diameter=1.0)


class WavefrontParametric(Parametric):
    value: Array

    def __init__(self, value):
        self.value = np.asarray(value)

    def evaluate(self, *, wavefront, **kwargs):
        return np.ones_like(wavefront.amplitude) * self.value


def test_transmissive_layer_resolves_explicit_basis():
    transmission = ExplicitBasis(np.ones((1, 8, 8)), coefficients=np.array([0.5]))
    output = TransmissiveLayer(transmission)(wavefront)

    assert np.allclose(output.amplitude, wavefront.amplitude * 0.5)


def test_transmissive_layer_passes_wavefront_context():
    output = TransmissiveLayer(WavefrontParametric(0.25))(wavefront)
    assert np.allclose(output.amplitude, wavefront.amplitude * 0.25)


def test_aberrated_layer_resolves_opd_and_phase():
    opd = ExplicitBasis(np.ones((1, 8, 8)), coefficients=np.array([1e-9]))
    phase = ExplicitBasis(np.ones((1, 8, 8)), coefficients=np.array([0.2]))

    assert np.allclose(
        AberratedLayer(opd=opd, phase=phase)(wavefront).phase,
        wavefront.add_opd(1e-9).add_phase(0.2).phase,
    )


def test_optic_resolves_all_parametric_leaves():
    transmission = WavefrontParametric(0.5)
    opd = WavefrontParametric(1e-9)
    phase = WavefrontParametric(0.2)

    output = Optic(transmission=transmission, opd=opd, phase=phase)(wavefront)
    expected = (wavefront * 0.5).add_opd(1e-9).add_phase(0.2)

    assert np.allclose(output.phasor, expected.phasor)
