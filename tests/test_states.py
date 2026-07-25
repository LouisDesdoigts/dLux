"""Tests for dLux.states."""

import jax.numpy as np
import pytest

import dLux as dl

from .helpers import assert_differentiable, assert_jittable


class TestWavefront:
    def test_construction_and_properties(self, make_wavefront):
        wavefront = make_wavefront()
        restored = dl.Wavefront.from_phasor(
            wavefront.phasor * np.exp(0.2j),
            wavefront.wavelength,
            wavefront.spec,
        )

        assert restored.spatial_shape == restored.spec.shape
        assert restored.batch_ndim == 0
        assert restored.is_chromatic is False
        assert restored._mapped_axis is None
        assert np.allclose(restored.complex[0], restored.real)
        assert np.allclose(restored.complex[1], restored.imaginary)
        assert np.allclose(restored.polar[0], restored.amplitude)
        assert np.allclose(restored.polar[1], restored.phase)
        assert np.allclose(restored.wavenumber, 2 * np.pi / restored.wavelength)
        assert np.allclose(restored.power, restored.psf.sum())

    def test_chromatic_contract(self, make_spec):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        wavefront = dl.Wavefront(wavelengths, make_spec())

        assert wavefront.phasor.shape == (2, 8, 8)
        assert wavefront.batch_ndim == 1
        assert wavefront.is_chromatic
        assert wavefront._mapped_axis is not None

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront.add_phase(0.2),
            lambda wavefront: wavefront.add_opd(1e-8),
            lambda wavefront: wavefront.tilt(np.asarray((1e-7, -2e-7))),
            lambda wavefront: wavefront.normalise(),
            lambda wavefront: wavefront.normalise("peak", 2.0),
            lambda wavefront: wavefront.interpolate(dl.Affine(rotation=0.1)),
            lambda wavefront: wavefront.rotate(0.1),
            lambda wavefront: wavefront.scale_to(6, 0.08),
            lambda wavefront: wavefront.resize(6),
            lambda wavefront: wavefront.downsample(2),
            lambda wavefront: wavefront.flip((0, 1)),
        ],
    )
    def test_wavefront_operation_contract(self, operation, make_wavefront):
        assert_jittable(operation, make_wavefront(), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront + 2.0,
            lambda wavefront: wavefront - 2.0,
            lambda wavefront: wavefront * np.exp(0.2j),
            lambda wavefront: wavefront / 2.0,
            lambda wavefront: wavefront + wavefront,
        ],
    )
    def test_wavefront_arithmetic(self, operation, make_wavefront):
        assert_jittable(operation, make_wavefront())

    def test_phase_gradients(self, make_wavefront):
        wavefront = make_wavefront()
        assert_differentiable(lambda phase: wavefront.add_phase(phase), np.asarray(0.2))

    def test_polarisation_promotion(self, make_wavefront):
        wavefront = make_wavefront()
        polarised = wavefront.apply_jones(np.eye(2))
        assert isinstance(polarised, dl.PolarisedWavefront)
        assert polarised.phasor.shape == (2, 2, 8, 8)
        assert np.allclose(wavefront.psf_from_stokes(), wavefront.psf)
        assert np.allclose(
            wavefront.psf_from_stokes(np.asarray((2.0, 0, 0, 0))),
            2 * wavefront.psf,
        )

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront.tilt(np.zeros(3)),
            lambda wavefront: wavefront.normalise("invalid"),
            lambda wavefront: wavefront.interpolate("invalid"),
            lambda wavefront: wavefront * "invalid",
            lambda wavefront: wavefront / wavefront,
        ],
    )
    def test_validation(self, operation, make_wavefront):
        with pytest.raises((TypeError, ValueError)):
            operation(make_wavefront())


class TestPolarisedWavefront:
    def test_construction_and_stokes(self, make_wavefront):
        wavefront = make_wavefront(polarised=True)
        promoted = dl.PolarisedWavefront.from_wavefront(make_wavefront())

        assert wavefront.batch_ndim == 0
        assert wavefront.phasor.shape == promoted.phasor.shape == (2, 2, 8, 8)
        assert wavefront.psf.shape == (8, 8)
        assert wavefront.stokes().shape == (4, 8, 8)
        assert wavefront.psf_from_stokes().shape == (8, 8)

    def test_jones_application(self, make_wavefront):
        wavefront = make_wavefront(polarised=True)
        output = assert_jittable(
            lambda value: value.apply_jones(np.eye(2)),
            wavefront,
        )
        assert output.phasor.shape == wavefront.phasor.shape


class TestPSF:
    def test_construction(self, make_psf, make_wavefront):
        psf = make_psf()
        converted = dl.PSF.from_wavefront(make_wavefront())

        assert psf.data.shape == psf.spec.shape
        assert psf.batch_ndim == 0
        assert converted.data.shape == converted.spec.shape

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf.normalise(),
            lambda psf: psf.normalise("peak", 2.0),
            lambda psf: psf.convolve(np.ones((3, 3)), method="direct"),
            lambda psf: psf.convolve(np.ones((3, 3)), method="fft"),
            lambda psf: psf.interpolate(dl.Affine(rotation=0.1)),
            lambda psf: psf.rotate(0.1),
            lambda psf: psf.scale_to(6, 0.08),
            lambda psf: psf.resize(6),
            lambda psf: psf.downsample(2),
            lambda psf: psf.flip((0, 1)),
        ],
    )
    def test_spatial_operation_contract(self, operation, make_psf):
        data = np.arange(64.0).reshape(8, 8) + 1
        assert_jittable(operation, make_psf(data=data), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf + 2.0,
            lambda psf: psf - 2.0,
            lambda psf: psf * 2.0,
            lambda psf: psf / 2.0,
        ],
    )
    def test_arithmetic_contract(self, operation, make_psf):
        assert_jittable(operation, make_psf())

    @pytest.mark.parametrize(
        "operation",
        [
            lambda psf: psf.normalise(),
            lambda psf: psf.convolve(np.ones((3, 3))),
            lambda psf: psf.interpolate(dl.Affine(rotation=0.1)),
        ],
    )
    def test_gradients(self, operation, make_psf):
        psf = make_psf(data=np.arange(64.0).reshape(8, 8) + 1)

        def apply(data):
            return operation(psf.set(data=data)).data

        assert_differentiable(apply, psf.data)

    def test_sampling_updates(self, make_psf):
        psf = make_psf().downsample(2)

        assert psf.spec.n == (4, 4)
        assert np.allclose(psf.spec.d, 0.2)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda spec: dl.PSF(np.ones(8), spec),
            lambda spec: dl.PSF(np.ones((4, 4)), spec),
            lambda spec: dl.PSF(np.ones((8, 8)), "invalid"),
        ],
    )
    def test_construction_validation(self, constructor, make_spec):
        with pytest.raises((TypeError, ValueError)):
            constructor(make_spec())

    def test_operation_validation(self, make_psf):
        psf = make_psf()

        with pytest.raises(ValueError, match="mode"):
            psf.normalise("invalid")
        with pytest.raises(TypeError, match="transformation"):
            psf.interpolate("invalid")
        with pytest.raises(TypeError, match="Unsupported type"):
            psf + "invalid"
