"""Tests for dLux.fields."""

import jax.numpy as np
import jax.random as jr
import pytest

import dLux as dl

from .helpers import assert_differentiable, assert_jittable


class TestWavefront:
    def test_construction_and_properties(self, make_wavefront):
        wavefront = make_wavefront()
        restored = dl.Wavefront.from_phasor(
            wavefront.phasor * np.exp(0.2j), wavefront.wavelength, wavefront.spec
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

    def test_chromatic_phasor_broadcasting(self, make_spec):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        phasor = np.ones((8, 8), dtype=complex)
        wavefront = dl.Wavefront(wavelengths, make_spec(), phasor)
        polarised = dl.PolarisedWavefront(wavelengths, make_spec(), phasor)

        assert wavefront.phasor.shape == (2, 8, 8)
        assert polarised.phasor.shape == (2, 2, 2, 8, 8)

    def test_chromatic_phase_broadcasting(self, make_spec):
        wavefront = dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), make_spec())
        phase = np.asarray((0.1, 0.2))
        spatial = np.linspace(0.0, 0.1, 64).reshape((8, 8))

        chromatic = assert_jittable(lambda value: wavefront.add_phase(value), phase)
        output = assert_jittable(lambda value: wavefront.add_phase(value), spatial)

        assert np.allclose(
            chromatic.phasor[:, 0, 0], wavefront.phasor[:, 0, 0] * np.exp(1j * phase)
        )
        assert np.allclose(
            output.phasor[:, 0], wavefront.phasor[:, 0] * np.exp(1j * spatial[0])
        )

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

    def test_mixed_wavefront_arithmetic(self, make_wavefront):
        wavefront = make_wavefront()
        chromatic = dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), wavefront.spec)
        polarised = dl.PolarisedWavefront.from_wavefront(wavefront)

        mixed = assert_jittable(lambda left, right: left * right, wavefront, polarised)
        reverse = assert_jittable(
            lambda left, right: left * right, polarised, wavefront
        )
        broadcast = assert_jittable(
            lambda left, right: left * right, chromatic, wavefront
        )

        assert isinstance(mixed, dl.PolarisedWavefront)
        assert mixed.phasor.shape == polarised.phasor.shape
        assert reverse.phasor.shape == polarised.phasor.shape
        assert broadcast.phasor.shape == chromatic.phasor.shape

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
            wavefront.psf_from_stokes(np.asarray((2.0, 0, 0, 0))), 2 * wavefront.psf
        )

    def test_polarised_from_phasor(self, make_wavefront):
        wavefront = make_wavefront()
        polarised = dl.PolarisedWavefront.from_phasor(
            wavefront.phasor, wavefront.wavelength, wavefront.spec
        )

        assert polarised.phasor.shape == (2, 2) + wavefront.phasor.shape

    @pytest.mark.parametrize(
        "operation",
        [
            lambda wavefront: wavefront.tilt(np.zeros(3)),
            lambda wavefront: wavefront.normalise("invalid"),
            lambda wavefront: wavefront.interpolate("invalid"),
            lambda wavefront: wavefront * "invalid",
            lambda wavefront: wavefront / wavefront,
            lambda wavefront: (
                wavefront + dl.Wavefront(1e-6, wavefront.spec.set(n=(6, 6)))
            ),
            lambda wavefront: (
                wavefront + dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), wavefront.spec)
            ),
            lambda wavefront: wavefront + np.ones((2, 3, 8, 8)),
        ],
    )
    def test_validation(self, operation, make_wavefront):
        with pytest.raises((TypeError, ValueError)):
            operation(make_wavefront())

    @pytest.mark.parametrize(
        "operation",
        [
            lambda spec: dl.Wavefront(1e-6, "invalid"),
            lambda spec: dl.Wavefront(1e-6, dl.GridSpec(d=0.1, unit="m")),
            lambda spec: dl.Wavefront(1e-6, spec, np.ones(8)),
            lambda spec: dl.Wavefront(1e-6, spec) + dl.PSF(np.ones((8, 8)), spec),
            lambda spec: dl.Wavefront(np.ones(2), spec) + np.ones((3, 8, 8)),
        ],
    )
    def test_construction_and_operand_validation(self, operation, make_spec):
        with pytest.raises((TypeError, ValueError)):
            operation(make_spec())


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
        output = assert_jittable(lambda value: value.apply_jones(np.eye(2)), wavefront)
        assert output.phasor.shape == wavefront.phasor.shape

    def test_chromatic_jones_phasor(self, make_spec):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        phasor = np.broadcast_to(np.eye(2)[:, :, None, None], (2, 2, 8, 8))
        wavefront = dl.PolarisedWavefront(wavelengths, make_spec(), phasor)

        assert wavefront.is_polarised
        assert wavefront.phasor.shape == (2, 2, 2, 8, 8)
        assert wavefront.psf_from_stokes(np.asarray((1.0, 0, 0, 0))).shape == (2, 8, 8)


class TestPSF:
    def test_construction(self, make_psf, make_wavefront):
        psf = make_psf()
        converted = dl.PSF.from_wavefront(make_wavefront())

        assert psf.data.shape == psf.spec.shape
        assert psf.batch_ndim == 0
        assert converted.data.shape == converted.spec.shape

    def test_sampling_contract(self):
        spec = dl.GridSpec(d=(0.1, 0.2), c=(0.3, -0.4), unit="m")
        psf = dl.PSF(np.ones((8, 8)), spec)

        assert psf.spec.n == (8, 8)
        assert psf.npixels == 8
        assert len(psf.axes) == 2
        assert psf.coordinates.shape == (2, 8, 8)
        assert np.allclose(psf.xs[0].mean(), 0.3)
        assert np.allclose(psf.pixel_scale, np.asarray((0.1, 0.2)))
        assert np.allclose(psf.center, np.asarray((0.3, -0.4)))
        assert np.allclose(psf.diameter, np.asarray((0.8, 1.6)))

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
            lambda psf: psf + psf,
        ],
    )
    def test_arithmetic_contract(self, operation, make_psf):
        assert_jittable(operation, make_psf())

    def test_inplace_and_none_arithmetic(self, make_psf):
        psf = make_psf()

        assert psf + None is psf
        assert np.allclose(psf.__iadd__(1).data, psf.data + 1)
        assert np.allclose(psf.__isub__(1).data, psf.data - 1)
        assert np.allclose(psf.__imul__(2).data, psf.data * 2)
        assert np.allclose(psf.__itruediv__(2).data, psf.data / 2)

    def test_forwarded_attribute_validation(self, make_psf):
        with pytest.raises(AttributeError, match="not_an_attribute"):
            _ = make_psf().not_an_attribute

        psf = dl.PSF(np.ones((8, 8)), dl.GridSpec(n=8).broadcast(2))
        with pytest.raises(ValueError, match="spec.d"):
            _ = psf.pixel_scale

    def test_coordinate_batch_validation(self):
        spec = dl.GridSpec(n=8, d=0.1, c=np.zeros((2, 2)), unit="m").broadcast(2)
        psf = dl.PSF(np.ones((8, 8)), spec)

        with pytest.raises(ValueError, match="Coordinate batch"):
            psf.interpolate(dl.Affine())

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

    def test_rectangular_sampling_operations(self):
        spec = dl.GridSpec(n=(8, 6), d=(0.1, 0.2), unit="m")
        psf = dl.PSF(np.arange(48.0).reshape(6, 8) + 1, spec)

        downsampled = assert_jittable(lambda value: value.downsample((2, 3)), psf)
        scaled = assert_jittable(
            lambda value: value.scale_to((10, 12), (0.08, 0.15)), psf
        )
        rotated = assert_jittable(lambda value: value.rotate(0.1), psf)

        assert downsampled.data.shape == (2, 4)
        assert downsampled.spec.n == (4, 2)
        assert np.allclose(downsampled.spec.d, np.asarray((0.2, 0.6)))
        assert scaled.data.shape == (12, 10)
        assert scaled.spec.n == (10, 12)
        assert np.allclose(scaled.spec.d, np.asarray((0.08, 0.15)))
        assert rotated.data.shape == psf.data.shape

    def test_batched_convolution(self):
        spec = dl.GridSpec(n=(8, 6), d=(0.1, 0.2), unit="m")
        data = np.arange(96.0).reshape(2, 6, 8)
        psf = dl.PSF(data, spec)
        kernel = np.ones((3, 3)) / 9

        output = assert_jittable(lambda value: value.convolve(kernel), psf)
        expected = np.stack(
            tuple(dl.PSF(image, spec).convolve(kernel).data for image in data)
        )

        assert output.data.shape == data.shape
        assert np.allclose(output.data, expected)

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


class TestImage:
    def test_discrete_field_contract(self, make_spec):
        image = dl.Image(np.full((8, 8), 10.0), make_spec())
        poisson = assert_jittable(
            lambda value: value.add_poisson_noise(jr.key(0)), image
        )
        noisy = assert_jittable(
            lambda value: value.add_read_noise(jr.key(1), 2.0), poisson
        )

        assert isinstance(image, dl.DiscreteField)
        assert not isinstance(image, dl.ContinuousField)
        assert poisson.variance.shape == image.data.shape
        assert noisy.error.shape == image.data.shape
        assert np.allclose(noisy.read_noise, 2.0)

    def test_noise_variance_accumulates(self, make_spec):
        image = dl.Image(
            np.full((8, 8), 10.0), make_spec(), variance=np.full((8, 8), 4.0)
        )
        poisson = image.add_poisson_noise(jr.key(0))
        noisy = poisson.add_read_noise(jr.key(1), 2.0)

        assert np.allclose(poisson.variance, 14.0)
        assert np.allclose(noisy.variance, 18.0)

    def test_fourier_spectra(self, make_spec):
        image = dl.Image(np.eye(8), make_spec())

        transformed = image.fourier_transform
        amplitude = image.amplitude_spectrum
        power = image.power_spectrum

        assert transformed.shape == image.data.shape
        assert np.allclose(amplitude, np.abs(transformed))
        assert np.allclose(power, amplitude**2)

    def test_likelihood_contract(self, make_spec):
        model = dl.PSF(np.full((8, 8), 10.0), make_spec())
        image = dl.Image(
            model.data, model.spec, variance=np.full((8, 8), 4.0), read_noise=2.0
        )

        gaussian = assert_jittable(
            lambda value: value.log_likelihood(model, "gaussian"), image
        )
        poisson = assert_jittable(
            lambda value: value.log_likelihood(model, "poisson"), image
        )

        assert np.isfinite(gaussian)
        assert np.isfinite(poisson)

    def test_leading_axis_contract(self, make_spec):
        data = np.full((2, 3, 8, 8), 10.0)
        image = dl.Image(data, make_spec(), variance=2.0)
        poisson = assert_jittable(
            lambda value: value.add_poisson_noise(jr.key(0)), image
        )
        noisy = assert_jittable(
            lambda value: value.add_read_noise(jr.key(1), 2.0), poisson
        )

        assert noisy.data.shape == noisy.variance.shape == data.shape
        assert noisy.fourier_transform.shape == data.shape
        assert noisy.power_spectrum.shape == data.shape
        assert np.isfinite(noisy.log_likelihood(image, "gaussian"))

    @pytest.mark.parametrize(
        "operation",
        [
            lambda spec: dl.Image(np.ones(8), spec),
            lambda spec: dl.Image(np.ones((4, 4)), spec),
            lambda spec: dl.Image(np.ones((8, 8)), "invalid"),
            lambda spec: dl.Image(np.ones((8, 8)), spec).log_likelihood(
                np.ones((4, 4))
            ),
            lambda spec: dl.Image(np.ones((8, 8)), spec).log_likelihood(
                np.ones((8, 8)), "gaussian"
            ),
            lambda spec: dl.Image(np.ones((8, 8)), spec, variance=1.0).log_likelihood(
                np.ones((8, 8)), "invalid"
            ),
        ],
    )
    def test_validation(self, operation, make_spec):
        with pytest.raises((TypeError, ValueError)):
            operation(make_spec())
