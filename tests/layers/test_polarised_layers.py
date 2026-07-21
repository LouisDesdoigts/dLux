from jax import numpy as np, config
import pytest

config.update("jax_debug_nans", True)

from dLux import PolarisedWavefront, Wavefront
import dLux.layers as dl
from dLux.layers.optical_layers import FourierBasis as OpticalFourierBasis
from dLux.layers.polarised_layers import (
    ExplicitBasis,
    FieldDict,
    FourierBasis,
    LinearPolariser,
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


def fourier_basis(npixels=8, n_modes=3):
    return dlu.fourier_kernels(n_modes, npixels)


class TestFieldDict:
    def test_accessors(self):
        fields = FieldDict({"angle": 0.1}, retardance=0.2)

        assert fields["angle"] == 0.1
        assert fields.angle == 0.1
        assert fields.get("retardance") == 0.2
        assert fields.get("missing", None) is None
        assert "angle" in fields

    def test_missing_attribute(self):
        with pytest.raises(AttributeError):
            FieldDict().angle


def test_layers_fourier_basis_export_is_optical_layer():
    assert dl.FourierBasis is OpticalFourierBasis
    assert dl.FourierBasis is not FourierBasis


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

        assert optic.parameters.angle == 0.0
        assert optic.angle.shape == ()
        assert optic.jones.shape == (2, 2)
        assert np.allclose(optic.jones, dlu.horizontal_polariser())

    def test_sv_linear_polariser_array_angle(self):
        angle = np.zeros((8, 8))
        optic = SVLinearPolariser(angle)
        out = optic(wavefront())

        assert optic.parameters.angle is angle
        assert optic.angle.shape == (8, 8)
        assert optic.jones.shape == (2, 2, 8, 8)
        assert out.phasor.shape == (2, 2, 8, 8)

    def test_sv_linear_polariser_explicit_basis_angle(self):
        basis = explicit_basis()
        coefficients = np.array([0.0, 0.1, 0.2])
        optic = SVLinearPolariser(
            ExplicitBasis(),
            basis={"angle": basis},
            coefficients={"angle": coefficients},
        )

        assert isinstance(optic.parameters.angle, ExplicitBasis)
        assert optic.basis.angle is basis
        assert optic.coefficients.angle is coefficients
        assert optic.jones.shape == (2, 2, 8, 8)
        assert np.allclose(optic.angle, dlu.eval_basis(basis, coefficients))

    def test_sv_linear_polariser_fourier_angle(self):
        basis = fourier_basis()
        coefficients = np.ones((3, 3))
        optic = SVLinearPolariser(
            FourierBasis(),
            basis={"angle": basis},
            coefficients={"angle": coefficients},
        )

        assert isinstance(optic.parameters.angle, FourierBasis)
        assert optic.basis.angle is basis
        assert optic.coefficients.angle is coefficients
        assert optic.jones.shape == (2, 2, 8, 8)
        assert np.allclose(optic.angle, dlu.eval_fourier_basis(coefficients, *basis))

    def test_sv_retarder_scalar_inputs(self):
        optic = SVRetarder(np.pi / 2, 0.0)

        assert optic.retardance.shape == ()
        assert optic.angle.shape == ()
        assert optic.jones.shape == (2, 2)

    def test_sv_retarder_array_inputs(self):
        retardance = np.ones((8, 8)) * np.pi / 2
        angle = np.zeros((8, 8))
        optic = SVRetarder(retardance, angle)

        assert np.allclose(optic.retardance, retardance)
        assert np.allclose(optic.angle, angle)
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_sv_retarder_explicit_basis_retardance(self):
        basis = explicit_basis()
        coefficients = np.array([0.1, 0.2, 0.3])
        optic = SVRetarder(
            ExplicitBasis(),
            0.0,
            basis={"retardance": basis},
            coefficients={"retardance": coefficients},
        )

        assert isinstance(optic.parameters.retardance, ExplicitBasis)
        assert optic.basis.retardance is basis
        assert optic.coefficients.retardance is coefficients
        assert np.allclose(optic.retardance, dlu.eval_basis(basis, coefficients))
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_sv_retarder_mixed_fourier_and_explicit_basis(self):
        retardance_basis = fourier_basis()
        angle_basis = explicit_basis()
        retardance_coefficients = np.ones((3, 3))
        angle_coefficients = np.array([0.0, 0.1, 0.2])
        optic = SVRetarder(
            FourierBasis(),
            ExplicitBasis(),
            basis={"retardance": retardance_basis, "angle": angle_basis},
            coefficients={
                "retardance": retardance_coefficients,
                "angle": angle_coefficients,
            },
        )

        assert np.allclose(
            optic.retardance,
            dlu.eval_fourier_basis(retardance_coefficients, *retardance_basis),
        )
        assert np.allclose(optic.angle, dlu.eval_basis(angle_basis, angle_coefficients))
        assert optic.jones.shape == (2, 2, 8, 8)

    def test_missing_explicit_basis_inputs(self):
        with pytest.raises(ValueError, match="Explicit basis"):
            optic = SVLinearPolariser(ExplicitBasis())
            optic.angle

    def test_missing_fourier_basis_inputs(self):
        with pytest.raises(ValueError, match="Fourier basis"):
            optic = SVLinearPolariser(FourierBasis())
            optic.angle

    def test_sv_retarder_applies_to_chromatic_wavefront(self):
        optic = SVRetarder(np.ones((8, 8)) * np.pi / 2, np.zeros((8, 8)))
        out = optic(chromatic_wavefront())

        assert out.phasor.shape == (3, 2, 2, 8, 8)
        assert out.stokes().shape == (3, 4, 8, 8)
