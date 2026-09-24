"""Standalone layer systems and detector input/output contracts."""

import jax.numpy as np
import jax.random as jr
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_optical_and_detector_systems_are_simple_ordered_operations(make_wavefront):
    wavefront = make_wavefront()
    optics = dl.OpticalSystem(
        [("pupil", dl.Optic(transmission=0.5)), ("phase", dl.Optic(phase=0.2))]
    )
    actual = assert_jittable(optics, wavefront)
    assert np.allclose(actual.phasor, 0.5 * wavefront.phasor * np.exp(0.2j))
    detector = dl.DetectorSystem([("bias", dl.BiasLayer(1)), ("gain", dl.GainLayer(2))])
    result = assert_jittable(detector, actual.to_intensity())
    assert np.allclose(result.data, 2 * (actual.intensity + 1))
    with pytest.raises(TypeError):
        detector(wavefront)
    with pytest.raises(TypeError):
        optics(wavefront.to_intensity())


def test_detector_response_precedes_pixel_sum_and_gain(make_psf):
    intensity = make_psf(data=np.arange(64.0).reshape(8, 8) + 1)
    model = dl.Detector(layers=[dl.BiasLayer(2)], downsample=2, gain=3)
    result = assert_jittable(model, intensity)
    assert isinstance(result, dl.Image)
    expected = (intensity + 2).downsample(2, mean=False) * 3
    assert np.allclose(result.data, expected.data)
    assert result.grid.n == (4, 4)


def test_detector_gain_and_noise_resolve_on_the_final_grid():
    grid = dlu.Grid(n=5, d=0.4)
    intensity = dl.Intensity(np.ones((5, 5)), grid)
    response = dlu.Gaussian(0.7)
    detector = dl.Detector(gain=response, read_noise=response)
    coordinates = grid.coordinates
    expected = np.exp(-np.sum(coordinates**2, axis=0) / (2 * 0.7**2))
    result = assert_jittable(lambda model: model.readout(intensity), detector)
    noise = assert_jittable(lambda model: model.noise(intensity), detector)
    assert np.allclose(result.data, expected)
    assert np.allclose(noise, expected)


def test_detector_image_integrates_before_noise_then_applies_gain(make_psf):
    intensity = make_psf(data=np.ones((8, 8)) * 3)
    detector = dl.Detector(layers=[dl.BiasLayer(1)], gain=2, read_noise=0.3)
    key = jr.key(8)
    result = assert_jittable(
        lambda values: detector.image(values, oversample=2, key=key, n_frames=3),
        intensity,
    )
    frames = []
    for frame_key in jr.split(key, 3):
        photon_key, read_key = jr.split(frame_key)
        photons = jr.poisson(photon_key, np.full((4, 4), 16.0)).astype(float)
        frames.append(2 * (photons + 0.3 * jr.normal(read_key, (4, 4))))
    assert isinstance(result, dl.Image)
    assert result.grid.n == (4, 4)
    assert np.allclose(result.data, np.stack(frames).mean(0))
    with pytest.raises(ValueError, match="n_frames"):
        detector.image(intensity, key=key, n_frames=0)


def test_detector_resolves_native_sampling_before_oversampling():
    class WavelengthSpacing(zdx.Expression):
        def evaluate(self, wavelength, **context):
            return np.stack((wavelength, wavelength), axis=-1)

    wavefront = dl.Wavefront(0.75, dlu.Grid(10, d=0.5))
    target = dlu.Grid(3, d=WavelengthSpacing())
    detector = dl.Detector(resample=dlu.Interp(grid_out=target))
    result = assert_jittable(
        lambda field: detector.sample(field, oversample=2), wavefront
    )
    reference = wavefront.resample(dlu.Grid(6, d=0.375))
    assert result.grid.n == (6, 6)
    assert np.allclose(result.grid.d, 0.375)
    assert np.allclose(result.phasor, reference.phasor)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Detector(downsample=0),
        lambda: dl.Detector(resample=dlu.Interp(ndim=1)),
        lambda: dl.Detector(resample=dlu.Interp(complex=False)),
        lambda: dl.Detector(layers=[dl.Optic()]),
        lambda: dl.OpticalSystem([dl.GainLayer(2)]),
    ],
)
def test_detector_and_sequence_contract_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_detector_optional_sampling_and_readout(make_wavefront):
    field = make_wavefront()
    detector = dl.Detector()
    assert detector.sample(field) is field
    assert detector.noise(field.to_intensity()) is None
    assert np.allclose(detector.readout(field.to_intensity()).data, field.intensity)
    requested = dlu.Grid(d=0.08, unit="m")
    detector = dl.Detector(resample=dlu.Interp(grid_out=requested))
    actual = assert_jittable(lambda x: detector.sample(x, oversample=2), field)
    assert actual.grid.n == field.grid.n
    assert np.allclose(actual.grid.d, 0.04)
    transformed = dl.Detector(resample=dlu.Interp(rotation=0.1)).sample(field)
    assert np.allclose(transformed.phasor, field.rotate(0.1).phasor)


def test_detector_noise_units_and_sequence_insertion(make_psf):
    intensity = make_psf()
    model = dl.Detector(gain=-2.0, read_noise=0.3)
    assert np.allclose(model.noise(intensity, output=True), 0.6)
    sequence = dl.DetectorSystem([dl.BiasLayer(1)]).insert_layer(dl.GainLayer(2), 1)
    assert np.allclose(sequence(intensity).data, (intensity.data + 1) * 2)
    with pytest.raises(TypeError):
        dl.OpticalSystem().debug(intensity)
    with pytest.raises(TypeError):
        dl.Detector(resample=1)
    with pytest.raises(TypeError):
        model.image(np.ones((3, 3)))


@pytest.mark.parametrize("sigma", [-1.0, np.nan, np.inf])
def test_detector_invalid_noise_is_rejected_before_sampling(sigma, make_psf):
    with pytest.raises(ValueError, match="finite and non-negative"):
        dl.Detector(read_noise=sigma).noise(make_psf())
