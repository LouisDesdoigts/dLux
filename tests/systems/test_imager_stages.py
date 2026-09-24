"""The optical, photometric and detector stages retain independent contracts."""

import equinox as eqx
import jax
import jax.numpy as np
from jax.scipy.signal import convolve
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu


@pytest.fixture
def model():
    return dl.Imager(
        dl.Grid(n=6, diam=1, unit="m"),
        dl.Grid(n=8, d=1e-7, unit="rad"),
        source=dl.Source(),
        bandpass=dlu.BandPass(6e-7, 2e-7, n=3),
        oversample=2,
    )


class UnavailableSpectrum(zdx.Expression):
    def evaluate(self, **context):
        raise AssertionError("An optical calculation must not evaluate this spectrum.")


def test_psf_uses_astrometry_and_optics_without_evaluating_brightness(model):
    positions = np.array([[0.0, 0.0], [1e-7, 2e-8]])
    source = dl.Source(UnavailableSpectrum(), positions=positions, flux=[0.0, 10.0])
    system = model.set(filter=dlu.Filter(0.1), detector=dl.Detector(qe=0.3, gain=7))
    result = eqx.filter_jit(lambda value: value.psf(source))(system)
    expected = np.stack(
        [
            np.stack(
                [
                    model.propagate_mono(wavelength, position).data
                    for wavelength in model.bandpass.wavelengths
                ]
            )
            for position in positions
        ]
    )
    assert result.data.shape == (2, 3, 16, 16)
    assert np.allclose(result.data, expected, rtol=2e-5, atol=1e-7)
    assert np.array_equal(result.wavelengths, model.bandpass.wavelengths)
    assert np.array_equal(result.positions, positions)


def test_photometry_is_independent_of_optical_propagation_and_has_single_losses(model):
    # An unresolved pupil expression is irrelevant to the photometric stage.
    class UnavailableOptic(dl.Optic):
        def apply(self, wavefront, **context):
            raise AssertionError("Photometry must not propagate light.")

    # Use linear interpolation for the analytic bin integrals below.
    spectrum = dlu.Spectrum([1.0, 2.0, 1.0], method="linear")
    source = dl.Source(spectrum, positions=[[0, 0], [1e-7, 0]], flux=[2, 5])
    system = model.set(
        source=source,
        pupil=UnavailableOptic(),
        filter=dlu.Filter(0.4),
        detector=dl.Detector(qe=0.3, gain=100),
    )
    result = system.photometry()
    spectral = np.array([4.0, 7.0, 4.0]) / 15
    expected = np.array([2, 5])[:, None] * spectral[None] * 0.4 * 0.3
    assert np.allclose(result.value, expected)
    assert np.array_equal(result.wavelengths, model.bandpass.wavelengths)


def test_scene_photometry_does_not_require_astrometric_offsets(model):
    source = dl.Source(flux=3)
    scene = dl.Scene(
        {"inner": dl.Scene({"star": source}, position=zdx.StateRef("inner"))},
        position=zdx.StateRef("outer"),
    )
    system = model.set(source=scene, filter=dlu.Filter(0.4))
    weights = eqx.filter_jit(lambda value: value.photometry())(system)
    expected = model.photometry(source, filter=0.4)
    assert np.allclose(weights["inner"]["star"].value, expected.value)
    assert isinstance(system.source.position, zdx.StateRef)


def test_psf_and_photometry_share_equal_optical_bins(model):
    result = eqx.filter_jit(lambda value: value.psf())(model)
    assert np.allclose(result.wavelengths, model.bandpass.wavelengths)
    weights = model.photometry()
    assert np.allclose(weights.value, np.ones((1, 3)) / 3)


def test_fine_intensity_stops_before_detector_layers_pixels_and_gain(model):
    spectrum = dlu.Spectrum([1.0, 2.0, 1.0], method="linear")
    source = dl.Source(spectrum, flux=5)
    detector = dl.Detector(
        layers=[dl.SensitivityLayer(0.7)], qe=0.4, gain=3, downsample=2
    )
    system = model.set(source=source, detector=detector)
    psf = system.psf()
    spectral = np.array([4.0, 7.0, 4.0]) / 15
    expected = (psf.data[0] * (5 * 0.4 * spectral)[:, None, None]).sum(0)
    intensity = eqx.filter_jit(lambda value: value.intensity())(system)
    assert intensity.data.shape == (16, 16)
    assert np.allclose(intensity.data, expected, rtol=2e-5)
    image = system.model()
    integrated = dl.Intensity(expected * 0.7, model.propagation_grid).downsample(4)
    assert isinstance(image, dl.Image)
    assert np.allclose(image.data, integrated.data * 3, rtol=2e-5)
    derivative = jax.grad(lambda flux: system.image(source.set(flux=flux)).data.sum())(
        np.asarray(5.0)
    )
    assert np.allclose(derivative, image.data.sum() / 5, rtol=3e-5)


def test_named_channels_pair_keys_and_use_scalar_stored_defaults(model):
    sources = {"star": dl.Source(flux=2)}
    filters = {"red": dlu.Filter(0.3)}
    outputs = model.model(source=sources, filter=filters)
    assert tuple(outputs) == ("star", "red")
    assert np.allclose(outputs["star"].data, model.image(dl.Source(flux=2)).data)
    assert np.allclose(outputs["red"].data, model.image(filter=0.3).data)
    assert model.source.flux == 1
    assert model.filter is None


def test_stored_named_channels_pair_sampling_without_cartesian_product(model):
    channels = model.set(
        source={"blue": dl.Source(flux=2), "red": dl.Source(flux=3)},
        bandpass={"blue": dlu.BandPass(5e-7), "red": dlu.BandPass(7e-7)},
    )
    result = eqx.filter_jit(lambda value: value.psf(filter=0.2))(channels)
    assert tuple(result) == ("blue", "red")
    for name, wavelength in (("blue", 5e-7), ("red", 7e-7)):
        assert result[name].data.shape == (1, 1, 16, 16)
        assert np.allclose(
            result[name].data[0, 0], model.propagate_mono(wavelength).data
        )
    images = channels.model(filter={"red": 0.2})
    assert np.allclose(
        images["blue"].data,
        model.image(dl.Source(flux=2), bandpass=dlu.BandPass(5e-7)).data,
    )
    assert np.allclose(
        images["red"].data,
        model.image(dl.Source(flux=3), bandpass=dlu.BandPass(7e-7), filter=0.2).data,
    )


def test_call_time_named_bandpasses_match_independent_calculations(model):
    bands = {
        "blue": dlu.BandPass(range=(450, 550), n=2, unit="nm"),
        "red": dlu.BandPass(range=(650, 750), n=4, unit="nm"),
    }
    outputs = model.intensity(bandpass=bands, filter={"red": 0.4})
    assert tuple(outputs) == ("red", "blue")
    for name, band in bands.items():
        expected = model.intensity(bandpass=band, filter=0.4 if name == "red" else 1)
        assert np.allclose(outputs[name].data, expected.data, rtol=2e-5)
    assert model.bandpass.n == 3


def test_named_channel_errors_identify_the_failing_input(model):
    # A useful validation tail: a bad channel must identify itself to the user.
    with pytest.raises(TypeError, match="Channel 'broken'.*requires a BandPass"):
        model.intensity(
            bandpass={"valid": model.bandpass, "broken": np.array([5e-7, 6e-7])}
        )


def test_source_list_adds_intensity_and_keeps_stage_objects_separate(model):
    sources = [
        dl.Source([1, 2, 1], flux=2),
        dl.Source([2, 1, 3], positions=[1e-7, 0], flux=4),
    ]
    psfs = model.psf(sources)
    weights = model.photometry(sources)
    assert len(psfs) == len(weights) == 2
    assert all(value.data.shape == (1, 3, 16, 16) for value in psfs)
    expected = sum(model.intensity(source).data for source in sources)
    assert np.allclose(model.intensity(sources).data, expected, rtol=2e-5)
    # Named output dictionaries differ deliberately from additive source lists.
    assert set(model.intensity(source={"a": sources[0], "b": sources[1]})) == {"a", "b"}


class ChromaticProfile(zdx.Expression):
    def evaluate(self, *, coordinates, wavelengths, **context):
        width = 5e-8 * wavelengths / 6e-7
        radius2 = np.square(coordinates).sum(0)
        return np.exp(-radius2[None] / (2 * width[:, None, None] ** 2))


def test_point_and_chromatic_disk_add_before_detection(model):
    point = dl.Source(dlu.Spectrum([1.0, 2.0, 1.0], method="linear"), flux=2)
    spectrum = dlu.Spectrum([3.0, 1.0, 2.0], method="linear")
    disk = dl.ResolvedSource(spectrum, flux=4, profile=ChromaticProfile())
    scene = dl.Scene({"point": point, "disk": disk})
    waves = model.bandpass.wavelengths
    grid = dlu.Grid(n=17, d=model.propagation_grid.d_si, unit="rad")
    radius2 = np.square(grid.coordinates).sum(0)
    point_psfs = model.psf(point).data[0]
    widths = 5e-8 * waves / 6e-7
    kernels = np.exp(-radius2[None] / (2 * widths[:, None, None] ** 2))
    kernels = kernels / kernels.sum((-2, -1), keepdims=True)
    disk_psfs = np.stack(
        [convolve(psf, kernel, mode="same") for psf, kernel in zip(point_psfs, kernels)]
    )
    # Analytic integrals of each linear shape over the three optical bins.
    point_weights = np.array([4.0, 7.0, 4.0])
    disk_weights = np.array([24.0, 11.0, 16.0])
    expected = (
        point_psfs * (2 * point_weights / point_weights.sum())[:, None, None]
    ).sum(0)
    expected += (
        disk_psfs * (4 * disk_weights / disk_weights.sum())[:, None, None]
    ).sum(0)
    actual = eqx.filter_jit(lambda value: model.intensity(value))(scene)
    assert np.allclose(actual.data, expected, rtol=3e-5, atol=2e-7)


def test_named_simulations_use_independent_random_draws(model):
    key = jax.random.key(4)
    sources = {"a": dl.Source(flux=100), "b": dl.Source(flux=100)}
    images = model.simulate(key, source=sources)
    keys = jax.random.split(key, 2)
    for index, name in enumerate(sources):
        reference = model.simulate(keys[index], sources[name])
        assert np.array_equal(images[name].data, reference.data)
    assert not np.array_equal(images["a"].data, images["b"].data)


def test_channel_defaults_populate_with_the_selected_source_owner(model):
    owner = zdx.Linked(0.5, key="shared")
    system = model.set(source=dl.Source(flux=owner), filter=dlu.Filter(owner.defer()))
    sources = {"a": system.source}
    result = system.photometry(source=sources, filter={"b": 0.7})
    assert np.allclose(result["a"].value.sum(), 0.5 * 0.5)
    assert np.allclose(result["b"].value.sum(), 0.5 * 0.7)
    # A changed selected owner must update the inherited filter too.
    changed = {"a": system.source.set("flux.value", 0.8)}
    result = system.photometry(source=changed, filter={"b": 0.7})
    assert np.allclose(result["a"].value.sum(), 0.8 * 0.8)
    assert np.allclose(result["b"].value.sum(), 0.5 * 0.7)
