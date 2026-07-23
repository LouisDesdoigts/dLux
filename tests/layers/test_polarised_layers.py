from jax import numpy as np, config
import pytest

config.update("jax_debug_nans", True)

from dLux import PolarisedWavefront, Wavefront
from dLux.parametric import ExplicitBasis, FourierBasis
from dLux.parametric.polynomials import PolynomialBasis
from dLux.layers.polarised_layers import (
    LinearPolariser,
    PolarisationLayer,
    PolarisingOptic,
    Retarder,
    UniformPolarisingOptic,
)
import dLux.utils as dlu


def wavefront(npixels=8):
    return PolarisedWavefront(1.0e-6, npixels=npixels, diameter=1.0)


def chromatic_wavefront(npixels=8):
    wavelengths = np.array([1.0e-6, 1.5e-6, 2.0e-6])
    return PolarisedWavefront(wavelengths, npixels=npixels, diameter=1.0)


def explicit_basis(npixels=8):
    y, x = np.mgrid[:npixels, :npixels]
    return np.array([np.ones((npixels, npixels)), x / npixels, y / npixels])


def fourier_basis(npixels=8, n_modes=3):
    return dlu.fourier_kernels(n_modes, npixels)


def test_polarisation_layer():
    wf = Wavefront(1.0e-6, npixels=8, diameter=1.0)
    assert PolarisationLayer()(wf) is wf
    layer = PolarisationLayer(
        [
            UniformPolarisingOptic(dlu.horizontal_polariser(), 0.1),
            PolarisingOptic(np.eye(2)),
            LinearPolariser(0.0),
        ]
    )
    assert layer.evaluate_jones(wf).shape == (2, 2)
    assert layer(wf).phasor.shape == (2, 2, 8, 8)


class TestUniformPolarisingLayers:
    def test_polarising_optic_applies_jones(self):
        optic = PolarisingOptic(dlu.horizontal_polariser())
        out = optic(wavefront())

        expected = np.zeros((4, 8, 8))
        expected = expected.at[0].set(0.5 / 8**4)
        expected = expected.at[1].set(0.5 / 8**4)
        assert np.allclose(out.stokes(), expected)

    def test_polarising_optic_promotes_wavefront(self):
        optic = PolarisingOptic(dlu.horizontal_polariser())
        out = optic(Wavefront(1.0e-6, npixels=8, diameter=1.0))

        assert isinstance(out, PolarisedWavefront)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_uniform_polarising_optic_rotates_jones(self):
        optic = UniformPolarisingOptic(dlu.horizontal_polariser(), np.pi / 2)
        out = optic(wavefront())

        assert np.allclose(optic.jones, dlu.horizontal_polariser())
        assert np.allclose(
            dlu.rotate_jones(optic.jones, optic.orientation),
            dlu.vertical_polariser(),
            atol=1e-6,
        )
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_uniform_polarising_optic_invalid_shape(self):
        with pytest.raises(ValueError, match="\\(2, 2\\)"):
            UniformPolarisingOptic(np.ones((2, 2, 4)))

    def test_linear_polariser(self):
        optic = LinearPolariser(np.pi / 4)
        out = optic(wavefront())

        assert optic.jones.shape == (2, 2)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_retarder(self):
        optic = Retarder(np.pi / 2, np.pi / 4)
        out = optic(wavefront())

        assert optic.jones.shape == (2, 2)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_chromatic_wavefront(self):
        optic = LinearPolariser(0.0)
        out = optic(chromatic_wavefront())

        assert out.phasor.shape == (3, 2, 2, 8, 8)
        assert out.stokes().shape == (3, 4, 8, 8)


class TestParametricPolarisingLayers:
    def test_linear_polariser_scalar_angle(self):
        optic = LinearPolariser(0.0)

        assert optic.angle == 0.0
        assert optic.angle.shape == ()
        assert optic.jones.shape == (2, 2)
        assert np.allclose(optic.jones, dlu.horizontal_polariser())

    def test_linear_polariser_array_angle(self):
        angle = np.zeros((8, 8))
        optic = LinearPolariser(angle)
        out = optic(wavefront())

        assert np.allclose(optic.angle, angle)
        assert optic.angle.shape == (8, 8)
        assert optic.jones.shape == (2, 2, 8, 8)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_linear_polariser_explicit_basis_angle(self):
        basis = explicit_basis()
        coefficients = np.array([0.0, 0.1, 0.2])
        parameter = ExplicitBasis(basis, coefficients)
        optic = LinearPolariser(parameter)

        assert isinstance(optic.angle, ExplicitBasis)
        assert optic.jones.shape == (2, 2, 8, 8)
        assert np.allclose(optic.evaluate_angle(), dlu.eval_basis(basis, coefficients))

    def test_linear_polariser_fourier_angle(self):
        coefficients = np.ones((3, 3))
        parameter = FourierBasis(8, 3, coefficients)
        optic = LinearPolariser(parameter)

        assert isinstance(optic.angle, FourierBasis)
        assert optic.jones.shape == (2, 2, 8, 8)
        assert np.allclose(optic.evaluate_angle(), parameter.evaluate())

    def test_linear_polariser_dynamic_basis(self):
        parameter = PolynomialBasis(1, np.array([0.1, 0.2, -0.3]))
        optic = LinearPolariser(parameter)
        wf = wavefront()
        angle = parameter.evaluate(wavefront=wf)
        out = optic(wf)

        assert np.allclose(optic.evaluate_angle(wf), angle)
        assert out.phasor.shape == (2, 2, 8, 8)
        with pytest.raises(ValueError, match="wavefront or coordinates"):
            _ = optic.jones

    def test_retarder_scalar_inputs(self):
        optic = Retarder(np.pi / 2, 0.0)

        assert optic.retardance.shape == ()
        assert optic.angle.shape == ()
        assert optic.jones.shape == (2, 2)

    def test_retarder_array_inputs(self):
        retardance = np.ones((8, 8)) * np.pi / 2
        angle = np.zeros((8, 8))
        optic = Retarder(retardance, angle)

        assert np.allclose(optic.evaluate_retardance(), retardance)
        assert np.allclose(optic.evaluate_angle(), angle)
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_retarder_explicit_basis_retardance(self):
        basis = explicit_basis()
        coefficients = np.array([0.1, 0.2, 0.3])
        parameter = ExplicitBasis(basis, coefficients)
        optic = Retarder(parameter, 0.0)

        assert isinstance(optic.retardance, ExplicitBasis)
        assert np.allclose(
            optic.evaluate_retardance(), dlu.eval_basis(basis, coefficients)
        )
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_retarder_mixed_fourier_and_explicit_basis(self):
        angle_basis = explicit_basis()
        retardance_coefficients = np.ones((3, 3))
        angle_coefficients = np.array([0.0, 0.1, 0.2])
        retardance = FourierBasis(8, 3, retardance_coefficients)
        angle = ExplicitBasis(angle_basis, angle_coefficients)
        optic = Retarder(
            retardance,
            angle,
        )

        assert np.allclose(optic.evaluate_retardance(), retardance.evaluate())
        assert np.allclose(
            optic.evaluate_angle(), dlu.eval_basis(angle_basis, angle_coefficients)
        )
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_retarder_dynamic_bases(self):
        retardance = PolynomialBasis(0, np.array([np.pi / 2]))
        angle = PolynomialBasis(1, np.array([0.0, 0.1, -0.1]))
        optic = Retarder(retardance, angle)
        wf = wavefront()
        out = optic(wf)

        assert np.allclose(
            optic.evaluate_retardance(wf), retardance.evaluate(wavefront=wf)
        )
        assert np.allclose(optic.evaluate_angle(wf), angle.evaluate(wavefront=wf))
        assert out.phasor.shape == (2, 2, 8, 8)
        with pytest.raises(ValueError, match="wavefront or coordinates"):
            _ = optic.jones

    def test_retarder_applies_to_chromatic_wavefront(self):
        optic = Retarder(np.ones((8, 8)) * np.pi / 2, np.zeros((8, 8)))
        out = optic(chromatic_wavefront())

        assert out.phasor.shape == (3, 2, 2, 8, 8)
        assert out.stokes().shape == (3, 4, 8, 8)
