from jax import numpy as np, config
import pytest

config.update("jax_debug_nans", True)

from dLux import PolarisedWavefront, Wavefront
from dLux.layers.polarised_layers import (
    Basis,
    Constant,
    FourierParameter,
    LinearPolariser,
    Parameter,
    PolarisingOptic,
    Retarder,
    SVLinearPolariser,
    SVRetarder,
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


class TestParameters:
    def test_constant(self):
        scalar = Constant(0.3)
        array = Constant(np.ones((4, 4)))

        assert isinstance(scalar, Parameter)
        assert scalar().shape == ()
        assert array().shape == (4, 4)
        assert np.allclose(array(), 1.0)

    def test_explicit_basis(self):
        basis = explicit_basis()
        coefficients = np.array([0.1, 0.2, 0.3])
        parameter = Basis(basis, coefficients)

        assert parameter().shape == (8, 8)
        assert np.allclose(parameter(), dlu.eval_basis(basis, coefficients))

    def test_explicit_basis_default_coefficients(self):
        parameter = Basis(explicit_basis())

        assert parameter.coefficients.shape == (3,)
        assert np.allclose(parameter(), 0.0)

    def test_explicit_basis_invalid_coefficients(self):
        with pytest.raises(ValueError, match="basis vectors"):
            Basis(explicit_basis(), np.ones(4))

    def test_fourier_basis(self):
        coefficients = np.ones((3, 5))
        parameter = FourierParameter(npix=8, n_modes=(3, 5), coefficients=coefficients)

        assert parameter().shape == (8, 8)
        assert np.allclose(
            parameter(), dlu.eval_fourier_basis(coefficients, *parameter.kernels)
        )

    def test_fourier_basis_default_coefficients(self):
        parameter = FourierParameter(npix=8, n_modes=3)

        assert parameter.coefficients.shape == (3, 3)
        assert np.allclose(parameter(), 0.0)

    def test_fourier_basis_invalid_coefficients(self):
        with pytest.raises(ValueError, match="Fourier coefficient"):
            FourierParameter(npix=8, n_modes=3, coefficients=np.ones((2, 3)))

    def test_fourier_basis_update_kernels(self):
        parameter = FourierParameter(npix=8, n_modes=3)
        updated = parameter.update_kernels(10)

        assert updated().shape == (10, 10)
        assert updated.coefficients.shape == parameter.coefficients.shape


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

        assert np.allclose(optic.jones, dlu.horizontal_polariser())
        assert np.allclose(
            dlu.rotate_jones(optic.jones, optic.orientation),
            dlu.vertical_polariser(),
            atol=1e-6,
        )

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


class TestSpatiallyVaryingPolarisingLayers:
    def test_sv_linear_polariser_scalar_angle(self):
        optic = SVLinearPolariser(0.0)

        assert isinstance(optic.angle, Constant)
        assert optic.jones.shape == (2, 2)
        assert np.allclose(optic.jones, dlu.horizontal_polariser())

    def test_sv_linear_polariser_array_angle(self):
        angle = np.zeros((8, 8))
        optic = SVLinearPolariser(angle)
        out = optic(wavefront())

        assert optic.jones.shape == (2, 2, 8, 8)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_sv_linear_polariser_explicit_basis_angle(self):
        basis = explicit_basis()
        coefficients = np.array([0.0, 0.1, 0.2])
        angle = Basis(basis, coefficients)
        optic = SVLinearPolariser(angle)

        assert optic.angle is angle
        assert optic.jones.shape == (2, 2, 8, 8)
        assert np.allclose(optic.angle(), dlu.eval_basis(basis, coefficients))

    def test_sv_linear_polariser_fourier_angle(self):
        angle = FourierParameter(npix=8, n_modes=3)
        optic = SVLinearPolariser(angle)

        assert optic.angle is angle
        assert optic.jones.shape == (2, 2, 8, 8)

    @pytest.mark.parametrize(
        "retardance, angle",
        [
            (np.pi / 2, 0.0),
            (np.ones((8, 8)) * np.pi / 2, np.zeros((8, 8))),
            (Basis(explicit_basis(), np.array([0.1, 0.2, 0.3])), 0.0),
            (FourierParameter(npix=8, n_modes=3), Basis(explicit_basis())),
        ],
        ids=["scalar", "array", "basis-retardance", "fourier-and-basis"],
    )
    def test_sv_retarder_parameter_inputs(self, retardance, angle):
        optic = SVRetarder(retardance, angle)
        jones = optic.jones

        assert isinstance(optic.retardance, Parameter)
        assert isinstance(optic.angle, Parameter)
        assert jones.shape[:2] == (2, 2)

    def test_sv_retarder_applies_to_chromatic_wavefront(self):
        optic = SVRetarder(np.ones((8, 8)) * np.pi / 2, np.zeros((8, 8)))
        out = optic(chromatic_wavefront())

        assert out.phasor.shape == (3, 2, 2, 8, 8)
        assert out.stokes().shape == (3, 4, 8, 8)
