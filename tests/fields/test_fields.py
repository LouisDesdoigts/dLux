"""Tests for dLux.fields."""

import jax
import jax.numpy as np
import jax.random as jr
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from dLux.fields import PSF

from tests.helpers import assert_differentiable, assert_jittable


@pytest.mark.parametrize("field_type", [dl.Intensity, PSF, dl.Image])
class TestIntensityOperations:
    @pytest.mark.parametrize("mode", ["power", "peak"])
    def test_normalisation_keeps_global_measure(self, field_type, mode):
        data = np.arange(1.0, 49.0).reshape(2, 4, 6)
        field = field_type(data, dlu.Grid(n=(6, 4), d=0.1))

        normalised = assert_jittable(lambda value: value.normalise(mode, 7.0), field)

        measure = data.sum() if mode == "power" else data.max()
        assert type(normalised) is field_type
        assert np.allclose(normalised.array, data * (7.0 / measure))
        assert np.array_equal(field.array, data)

    def test_downsampling_integrates_counts(self, field_type):
        data = np.arange(1.0, 49.0).reshape(2, 4, 6)
        field = field_type(data, dlu.Grid(n=(6, 4), d=0.1, c=(0.2, -0.1)))

        binned = assert_jittable(lambda value: value.downsample((3, 2)), field)
        averaged = field.downsample((3, 2), mean=True)

        assert type(binned) is field_type
        assert binned.array.shape == (2, 2, 2)
        assert np.allclose(binned.array.sum((-2, -1)), data.sum((-2, -1)))
        assert np.allclose(averaged.array * 6, binned.array)
        assert np.allclose(binned.grid.c, field.grid.c)
        assert np.allclose(binned.grid.fov, field.grid.fov)


class TestWavefront:
    def test_construction_and_properties(self, make_wavefront):
        wavefront = make_wavefront()
        restored = dl.Wavefront(
            wavefront.wavelength, wavefront.grid, wavefront.phasor * np.exp(0.2j)
        )

        assert restored.spatial_shape == restored.grid.shape
        assert restored.batch_ndim == 0
        assert restored.is_chromatic is False
        assert np.allclose(restored.complex[0], restored.real)
        assert np.allclose(restored.complex[1], restored.imaginary)
        assert np.allclose(restored.polar[0], restored.amplitude)
        assert np.allclose(restored.polar[1], restored.phase)
        assert np.allclose(restored.wavenumber, 2 * np.pi / restored.wavelength)
        assert np.allclose(restored.power, restored.psf.sum())

    def test_chromatic_contract(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        wavefront = dl.Wavefront(wavelengths, make_grid())
        normalised = wavefront.normalise()

        assert wavefront.phasor.shape == (2, 8, 8)
        assert wavefront.batch_ndim == 1
        assert wavefront.is_chromatic
        assert normalised.power.shape == wavelengths.shape
        assert np.allclose(normalised.power, 1)

    def test_chromatic_grid_batch_contract(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        centers = np.asarray(((-0.2, 0.0), (0.0, 0.2), (0.2, 0.0)))
        wavefront = dl.Wavefront(wavelengths, make_grid().set(c=centers[:, None]))
        normalised = wavefront.normalise()

        assert wavefront.phasor.shape == (3, 2, 8, 8)
        assert wavefront.power.shape == (3, 2)
        assert np.allclose(normalised.power, 1)

    @pytest.mark.parametrize("batch_size", [2, 3])
    def test_chromatic_grid_batch_tilt_axes_remain_independent(
        self, batch_size, make_grid
    ):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        centres = np.zeros((batch_size, 2))
        wavefront = dl.Wavefront(wavelengths, make_grid().set(c=centres[:, None]))

        tilted = assert_jittable(
            lambda value: value.tilt(np.asarray((1e-7, -2e-7))),
            wavefront,
        )

        assert tilted.phasor.shape == (batch_size, 2, 8, 8)
        assert tilted.grid.c.shape == (batch_size, 1, 2)

    def test_psf_conversion_preserves_unweighted_samples(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        phasor = np.ones((3, 2, 8, 8), dtype=complex)
        wavefront = dl.Wavefront(wavelengths, make_grid(), phasor).normalise()
        psf = assert_jittable(lambda value: value.to_psf(), wavefront)
        intensity = wavefront.to_intensity()
        assert psf.data.shape == phasor.shape
        assert np.allclose(psf.data.sum((-2, -1)), np.ones((3, 2)))
        assert np.allclose(intensity.data, psf.data)
        assert np.allclose(wavefront.materialise().intensity, wavefront.intensity)

    def test_generated_wavefront_has_unit_power(self, make_grid):
        wavefront = dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), make_grid())

        assert np.allclose(wavefront.power, np.ones(2))

    def test_chromatic_phasor_broadcasting(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        phasor = np.ones((8, 8), dtype=complex)
        wavefront = dl.Wavefront(wavelengths, make_grid(), phasor)
        polarised = dl.Wavefront(wavelengths, make_grid(), phasor, jones=True)
        normalised = polarised.normalise()

        assert wavefront.phasor.shape == (2, 8, 8)
        assert polarised.phasor.shape == (2, 2, 2, 8, 8)
        assert normalised.power.shape == wavelengths.shape
        assert np.allclose(normalised.power, 1)

    def test_chromatic_polarised_normalisation(self, make_grid):
        wavelengths = np.asarray((0.8e-6, 1.0e-6, 1.2e-6))
        amplitude = np.asarray((1.0, 2.0, 4.0))[:, None, None]
        wavefront = dl.Wavefront(
            wavelengths, make_grid(), amplitude * np.ones((3, 8, 8))
        ).materialise()

        normalised = assert_jittable(lambda value: value.normalise(), wavefront)

        assert normalised.phasor.shape == (3, 2, 2, 8, 8)
        assert np.allclose(normalised.power, np.ones(3))

    def test_chromatic_phase_broadcasting(self, make_grid):
        wavefront = dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), make_grid())
        phase = np.asarray((0.1, 0.2))
        spatial = np.linspace(0.0, 0.1, 64).reshape((8, 8))

        chromatic = assert_jittable(
            lambda value: wavefront.add_phase(value[:, None, None]), phase
        )
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
            lambda wavefront: wavefront.interpolate(dlu.Affine(rotation=0.1)),
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
        chromatic = dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), wavefront.grid)
        polarised = wavefront.materialise()

        mixed = assert_jittable(lambda left, right: left * right, wavefront, polarised)
        reverse = assert_jittable(
            lambda left, right: left * right, polarised, wavefront
        )
        broadcast = assert_jittable(
            lambda left, right: left * right, chromatic, wavefront
        )

        assert isinstance(mixed, dl.Wavefront)
        assert mixed.phasor.shape == polarised.phasor.shape
        assert reverse.phasor.shape == polarised.phasor.shape
        assert broadcast.phasor.shape == chromatic.phasor.shape

    def test_phase_gradients(self, make_wavefront):
        wavefront = make_wavefront()
        assert_differentiable(lambda phase: wavefront.add_phase(phase), np.asarray(0.2))

    def test_polarisation_promotion(self, make_wavefront):
        wavefront = make_wavefront()
        polarised = wavefront.apply_jones(np.eye(2))
        assert isinstance(polarised, dl.Wavefront)
        assert polarised.phasor.shape == (8, 8)
        assert polarised.jones.shape == (2, 2)
        assert np.allclose(wavefront.intensity, wavefront.psf)
        assert np.allclose(wavefront.psf_from_stokes(), wavefront.psf)
        assert np.allclose(wavefront.intensity_from_stokes(), wavefront.intensity)
        assert np.allclose(
            wavefront.psf_from_stokes(np.asarray((2.0, 0, 0, 0))), 2 * wavefront.psf
        )

    def test_polarised_phasor_construction(self, make_wavefront):
        wavefront = make_wavefront()
        polarised = dl.Wavefront(
            wavefront.wavelength, wavefront.grid, wavefront.phasor, jones=True
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
                wavefront + dl.Wavefront(1e-6, wavefront.grid.set(n=(6, 6)))
            ),
            lambda wavefront: (
                wavefront + dl.Wavefront(np.asarray((0.9e-6, 1.1e-6)), wavefront.grid)
            ),
            lambda wavefront: wavefront + np.ones((3, 2, 8, 8)),
        ],
    )
    def test_validation(self, operation, make_wavefront):
        with pytest.raises((TypeError, ValueError)):
            operation(make_wavefront())

    @pytest.mark.parametrize(
        "operation",
        [
            lambda grid: dl.Wavefront(1e-6, "invalid"),
            lambda grid: dl.Wavefront(1e-6, dl.Grid(d=0.1, unit="m")),
            lambda grid: dl.Wavefront(1e-6, grid, np.ones(8)),
            lambda grid: dl.Wavefront(1e-6, grid) + dl.Intensity(np.ones((8, 8)), grid),
            lambda grid: dl.Wavefront(np.ones(2), grid) + np.ones((3, 8, 8)),
            lambda grid: dl.Wavefront(
                np.ones(2), grid, np.ones((3, 8, 8), dtype=complex)
            ),
        ],
    )
    def test_construction_and_operand_validation(self, operation, make_grid):
        with pytest.raises((TypeError, ValueError)):
            operation(make_grid())


class TestJonesWavefront:
    def test_construction_and_stokes(self, make_wavefront):
        wavefront = make_wavefront(polarised=True)
        promoted = make_wavefront().materialise()

        assert wavefront.batch_ndim == 0
        assert wavefront.phasor.shape == promoted.phasor.shape == (2, 2, 8, 8)
        assert wavefront.psf.shape == (8, 8)
        assert np.allclose(wavefront.intensity, wavefront.psf)
        assert wavefront.stokes().shape == (4, 8, 8)
        assert wavefront.psf_from_stokes().shape == (8, 8)
        assert np.allclose(
            wavefront.intensity_from_stokes(), wavefront.psf_from_stokes()
        )

    def test_jones_application(self, make_wavefront):
        wavefront = make_wavefront(polarised=True)
        output = assert_jittable(lambda value: value.apply_jones(np.eye(2)), wavefront)
        assert output.phasor.shape == wavefront.phasor.shape

    def test_chromatic_jones_phasor(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        phasor = np.broadcast_to(np.eye(2)[:, :, None, None], (2, 2, 8, 8))
        wavefront = dl.Wavefront(wavelengths, make_grid(), phasor, jones=True)

        assert wavefront.is_polarised
        assert wavefront.phasor.shape == (2, 2, 2, 8, 8)
        stokes = np.asarray((1.0, 0, 0, 0))
        assert wavefront.psf_from_stokes(stokes).shape == (2, 8, 8)
        assert np.allclose(
            wavefront.intensity_from_stokes(stokes),
            wavefront.psf_from_stokes(stokes),
        )


class TestPSF:
    def test_construction_and_intensity_conversion(self, make_grid):
        grid = make_grid()
        data = np.arange(64.0).reshape(8, 8) + 1
        psf = assert_jittable(lambda value: PSF(value, grid), data)
        intensity = assert_jittable(lambda value: value.to_intensity(), psf)
        fluxes = np.asarray((2.0, 3.0))
        batched = assert_jittable(lambda value: psf.to_intensity(value), fluxes)

        assert isinstance(psf, dlu.Field)
        assert isinstance(psf, dl.Intensity)
        assert isinstance(intensity, dl.Intensity)
        assert np.allclose(intensity.data, psf.data)
        assert intensity.grid == psf.grid
        assert batched.data.shape == fluxes.shape + data.shape
        assert np.allclose(batched.data, fluxes[:, None, None] * data)

    def test_wavefront_construction_preserves_spectral_axis(self, make_grid):
        wavelengths = np.asarray((0.9e-6, 1.1e-6))
        wavefront = dl.Wavefront(
            wavelengths,
            make_grid(),
        ).normalise()

        direct = assert_jittable(lambda value: PSF.from_wavefront(value), wavefront)
        converted = wavefront.to_psf()

        assert direct.data.shape == wavelengths.shape + wavefront.grid.shape
        assert np.allclose(direct.data, converted.data)
        assert np.allclose(direct.data.sum((-2, -1)), np.ones(2))

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda grid: PSF(np.ones(8), grid),
            lambda grid: PSF(np.ones((4, 4)), grid),
            lambda grid: PSF(np.ones((8, 8)), "invalid"),
            lambda grid: PSF.from_wavefront(np.ones((8, 8))),
        ],
    )
    def test_construction_validation(self, constructor, make_grid):
        with pytest.raises((TypeError, ValueError)):
            constructor(make_grid())


class TestIntensity:
    def test_construction(self, make_psf, make_wavefront):
        psf = make_psf()
        wavefront = make_wavefront()
        converted = dl.Intensity.from_wavefront(wavefront)
        direct = assert_jittable(lambda value: value.to_intensity(), wavefront)

        assert psf.data.shape == psf.grid.shape
        assert psf.batch_ndim == 0
        assert converted.data.shape == converted.grid.shape
        assert np.allclose(direct.data, converted.data)

    def test_sampling_contract(self):
        grid = dl.Grid(d=(0.1, 0.2), c=(0.3, -0.4), unit="m")
        psf = dl.Intensity(np.ones((8, 8)), grid)

        assert psf.grid.n == (8, 8)
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
            lambda psf: psf.interpolate(dlu.Affine(rotation=0.1)),
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

        with pytest.raises(AttributeError, match="no attribute 'var'"):
            _ = make_psf().var

        psf = dl.Intensity(np.ones((8, 8)), dl.Grid(n=8))
        with pytest.raises(ValueError, match="d must be specified"):
            _ = psf.pixel_scale

    def test_coordinate_batch_broadcasting(self):
        grid = dl.Grid(n=8, d=0.1, c=np.zeros((2, 2)), unit="m")
        psf = dl.Intensity(np.ones((8, 8)), grid)

        output = psf.interpolate(dlu.Affine())
        assert output.data.shape == (2, 8, 8)
        assert np.allclose(output.data, psf.data)

    @pytest.mark.parametrize(
        "operation, atol",
        [
            (lambda psf: psf.normalise(), 1e-6),
            (lambda psf: psf.convolve(np.ones((3, 3))), 1e-6),
            (lambda psf: psf.interpolate(dlu.Affine(rotation=0.1)), 2e-4),
        ],
    )
    def test_gradients(self, operation, atol, make_psf):
        psf = make_psf(data=np.arange(64.0).reshape(8, 8) + 1)

        def apply(data):
            return operation(psf.set(data=data)).data

        # Cubic gradients reach 305 here: float32 accumulation differs by 1.2e-4,
        # including cancellation at small entries. The x64 discrepancy is 1.2e-13.
        assert_differentiable(apply, psf.data, rtol=3e-5, atol=atol)

    def test_sampling_updates(self, make_psf):
        psf = make_psf().downsample(2)

        assert psf.grid.n == (4, 4)
        assert np.allclose(psf.grid.d, 0.2)

    def test_rectangular_sampling_operations(self):
        grid = dl.Grid(n=(8, 6), d=(0.1, 0.2), unit="m")
        psf = dl.Intensity(np.arange(48.0).reshape(6, 8) + 1, grid)

        downsampled = assert_jittable(lambda value: value.downsample((2, 3)), psf)
        # Cubic extrapolation differs by up to 2.5 ppm under float32 XLA fusion.
        scaled = assert_jittable(
            lambda value: value.scale_to((10, 12), (0.08, 0.15)), psf, rtol=3e-6
        )
        rotated = assert_jittable(lambda value: value.rotate(0.1), psf)

        assert downsampled.data.shape == (2, 4)
        assert downsampled.grid.n == (4, 2)
        assert np.allclose(downsampled.grid.d, np.asarray((0.2, 0.6)))
        assert scaled.data.shape == (12, 10)
        assert scaled.grid.n == (10, 12)
        assert np.allclose(scaled.grid.d, np.asarray((0.08, 0.15)))
        assert rotated.data.shape == psf.data.shape

    def test_batched_convolution(self):
        grid = dl.Grid(n=(8, 6), d=(0.1, 0.2), unit="m")
        data = np.arange(96.0).reshape(2, 6, 8)
        psf = dl.Intensity(data, grid)
        kernel = np.ones((3, 3)) / 9

        output = assert_jittable(lambda value: value.convolve(kernel), psf)
        expected = np.stack(
            tuple(dl.Intensity(image, grid).convolve(kernel).data for image in data)
        )

        assert output.data.shape == data.shape
        assert np.allclose(output.data, expected)

    @pytest.mark.parametrize(
        "constructor",
        [
            lambda grid: dl.Intensity(np.ones(8), grid),
            lambda grid: dl.Intensity(np.ones((4, 4)), grid),
            lambda grid: dl.Intensity(np.ones((8, 8)), "invalid"),
        ],
    )
    def test_construction_validation(self, constructor, make_grid):
        with pytest.raises((TypeError, ValueError)):
            constructor(make_grid())

    def test_operation_validation(self, make_psf):
        psf = make_psf()

        with pytest.raises(ValueError, match="mode"):
            psf.normalise("invalid")
        with pytest.raises(TypeError, match="callable"):
            psf.interpolate("invalid")
        with pytest.raises(TypeError, match="Unsupported type"):
            psf + "invalid"


class TestImage:
    def test_peak_normalisation_uses_signed_maximum(self):
        image = dl.Image(np.asarray([[-8.0, 1.0], [2.0, 4.0]]), dlu.Grid(n=2, d=1))

        normalised = assert_jittable(lambda value: value.normalise("peak", 2), image)

        assert np.allclose(normalised.array, image.array / 2)
        assert normalised.array.max() == 2

    def test_discrete_field_contract(self, make_grid):
        image = dl.Image(np.full((8, 8), 10.0), make_grid())
        poisson = assert_jittable(
            lambda value: value.add_poisson_noise(jr.key(0)), image
        )
        noisy = assert_jittable(
            lambda value: value.add_read_noise(jr.key(1), 2.0), poisson
        )
        expected = jr.poisson(jr.key(0), image.data).astype(image.data.dtype)
        expected += 2 * jr.normal(jr.key(1), image.data.shape, image.data.dtype)

        assert isinstance(image, dlu.Field)
        assert np.array_equal(noisy.data, expected)
        assert np.all(image.data == 10)

    def test_arithmetic_and_resizing_store_only_samples(self, make_grid):
        image = dl.Image(np.ones((8, 8)), make_grid())
        output = (image * 3).downsample(2)

        assert output.data.shape == (4, 4)
        assert np.allclose(output.data, 12)
        assert not hasattr(output, "std")
        assert not hasattr(output, "variance")
        assert not hasattr(output, "read_noise")

    @pytest.mark.parametrize("read_noise", [None, 2.0])
    def test_intensity_conversion_and_simulation(self, read_noise, make_grid):
        intensity = dl.Intensity(np.full((8, 8), 10.0), make_grid())
        image = intensity.to_image()
        simulated = assert_jittable(
            lambda value: value.simulate(jr.key(0), 4, read_noise=read_noise), image
        )
        frames = []
        for key in jr.split(jr.key(0), 4):
            photon_key, read_key = jr.split(key)
            frame = jr.poisson(photon_key, intensity.data).astype(image.data.dtype)
            if read_noise is not None:
                frame += read_noise * jr.normal(read_key, frame.shape, frame.dtype)
            frames.append(frame)

        assert np.allclose(simulated.data, np.stack(frames).mean(0))
        assert np.array_equal(image.data, intensity.data)
        assert isinstance(dl.Image.from_intensity(intensity), dl.Image)
        with pytest.raises(ValueError):
            dl.Image(intensity, make_grid())
        with pytest.raises(ValueError, match="n_frames"):
            image.simulate(jr.key(0), 0)

    def test_fourier_spectra(self, make_grid):
        image = dl.Image(np.eye(8), make_grid())
        transformed = image.fourier_transform
        assert transformed.shape == image.data.shape
        assert np.allclose(image.amplitude_spectrum, np.abs(transformed))
        assert np.allclose(image.power_spectrum, np.abs(transformed) ** 2)

    def test_leading_axis_contract(self, make_grid):
        data = np.full((3, 2, 8, 8), 10.0)
        image = dl.Image(data, make_grid())
        poisson = assert_jittable(
            lambda value: value.add_poisson_noise(jr.key(0)), image
        )
        noisy = assert_jittable(
            lambda value: value.add_read_noise(jr.key(1), 2.0), poisson
        )
        assert noisy.data.shape == data.shape
        assert noisy.fourier_transform.shape == data.shape
        assert noisy.power_spectrum.shape == data.shape

    @pytest.mark.parametrize(
        "operation",
        [
            lambda grid: dl.Image(np.ones(8), grid),
            lambda grid: dl.Image(np.ones((4, 4)), grid),
            lambda grid: dl.Image(np.ones((8, 8)), "invalid"),
        ],
    )
    def test_validation(self, operation, make_grid):
        with pytest.raises((TypeError, ValueError)):
            operation(make_grid())


@pytest.mark.parametrize("chromatic", [False, True])
def test_factored_jones_matches_materialised_resampling(chromatic, make_grid):
    wavelengths = np.array([5e-7, 6e-7, 7e-7]) if chromatic else np.asarray(6e-7)
    matrix = dlu.Jones.retarder(0.7, 0.2)
    if chromatic:
        matrix = np.stack([matrix, 0.8 * matrix, 0.6 * matrix])
    wavefront = dl.Wavefront(wavelengths, make_grid(), jones=matrix)
    transform = dlu.Affine(translation=[0.03, -0.02], rotation=0.1)
    compact = assert_jittable(
        lambda field: field.interpolate(transform), wavefront, rtol=3e-5
    )
    explicit = wavefront.materialise().interpolate(transform)
    assert np.allclose(
        compact.materialise().phasor, explicit.phasor, rtol=3e-5, atol=1e-7
    )
    assert np.allclose(compact.intensity, explicit.intensity, rtol=3e-5, atol=1e-7)


def test_chromatic_jones_resampling_interpolates_complete_field(make_grid):
    wavelengths = np.array([5e-7, 6e-7, 7e-7])
    matrix = np.array([1.0, 0.8, 0.5])[:, None, None] * np.eye(2)
    phasor = np.array([1.0, 2.0, 4.0])[:, None, None] * np.ones((3, 8, 8))
    wavefront = dl.Wavefront(wavelengths, make_grid(), phasor, jones=matrix)
    query = np.array([5.5e-7, 6.5e-7])
    actual = assert_jittable(
        lambda field: field.resample_wavelengths(query), wavefront, rtol=3e-5
    )
    expected = wavefront.materialise().resample_wavelengths(query)
    assert actual.jones is True
    assert np.allclose(actual.phasor, expected.phasor, rtol=3e-5)


def test_coherent_sum_consumes_only_common_non_spectral_axis(make_grid):
    field = np.stack([np.ones((8, 8)), 1j * np.ones((8, 8))])
    wavefront = dl.Wavefront(6e-7, make_grid(), field)
    output = assert_jittable(lambda value: value.coherent_sum(), wavefront)
    assert np.allclose(output.phasor, 1 + 1j)
    assert np.allclose(output.intensity, 2.0)
    with pytest.raises(ValueError, match="non-wavelength"):
        dl.Wavefront(np.array([5e-7, 6e-7]), make_grid(), field).coherent_sum()


def test_resampling_resolves_chromatic_grids_without_pairing_population_axes():
    class WavelengthSpacing(zdx.Expression):
        def evaluate(self, wavelength, **context):
            return np.stack((wavelength, wavelength), axis=-1)

    grid = dlu.Grid(9, d=1.0)
    x, y = grid.coordinates
    wavelengths = np.asarray([0.6, 0.9])
    populations = np.asarray([1.0, 2.0])
    samples = populations[:, None, None] * (1 + 0.1 * x + 0.2j * y)
    samples = np.broadcast_to(samples[:, None], (2, 2) + grid.shape)
    matrices = np.asarray([1.0, 0.7])[:, None, None] * np.eye(2)
    wavefront = dl.Wavefront(wavelengths, grid, samples, jones=matrices)
    operation = dlu.Interp(grid_out=dlu.Grid(5, d=WavelengthSpacing()))

    result = assert_jittable(lambda field: field.resample(operation), wavefront)
    assert isinstance(result.grid.d, jax.Array)
    assert result.grid.d.shape == (2, 2, 2)
    assert np.allclose(result.grid.d, wavelengths[None, :, None])
    assert np.array_equal(result.jones, matrices)
    for spectral_index, wavelength in enumerate(wavelengths):
        for population_index in range(2):
            reference = dl.Wavefront(
                wavelength,
                grid,
                samples[population_index, spectral_index],
                jones=matrices[spectral_index],
            ).resample(dlu.Grid(5, d=wavelength))
            assert np.allclose(
                result.materialise().phasor[population_index, spectral_index],
                reference.materialise().phasor,
                rtol=3e-5,
                atol=1e-6,
            )

    # Interpolation preserves sample values: this pixel is 1 + 0.1 * wavelength.
    monochromatic = dl.Wavefront(0.6, grid, 1 + 0.1 * x + 0.2j * y)

    def sampled_value(wavelength):
        field = monochromatic.set(wavelength=wavelength).resample(operation)
        return field.phasor[2, 3].real

    wavelength = np.asarray(0.6)
    assert np.allclose(sampled_value(wavelength), 1 + 0.1 * wavelength)
    derivative = assert_jittable(jax.grad(sampled_value), wavelength)
    assert np.allclose(derivative, 0.1, rtol=2e-5)
