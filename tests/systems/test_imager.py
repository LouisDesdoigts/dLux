"""Imager owns photometry, incoherent source summation and detector ordering."""

import equinox as eqx
import jax
import jax.numpy as np
import jax.random as jr
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable, assert_tree_allclose


@pytest.fixture
def imager():
    return dl.Imager(
        dl.Grid(n=8, diam=1.0, unit="m"),
        dl.Grid(n=12, d=1e-7, unit="rad"),
        bandpass=dlu.BandPass(6e-7, 2e-7, n=3),
        source=dl.Source(),
    )


def test_default_source_and_container_routes(imager):
    field = imager.wavefront()
    assert field.phasor.shape == (1, 3, 12, 12)
    assert isinstance(imager(), dl.Image)
    assert isinstance(imager.psf(), dl.PSF)
    assert isinstance(imager.intensity(), dl.Intensity)
    assert np.allclose(imager().data, imager.intensity().data)
    assert np.allclose(imager.model().data, imager.intensity().data)
    assert np.allclose(imager.image(dl.Source(flux=7)).data, 7 * imager().data)
    assert imager.source.flux == 1


def test_flux_and_absolute_density_have_one_brightness_scale(imager):
    baseline = imager.intensity().data
    source = dl.Source(dlu.Spectrum(2e7, flux=None))
    assert np.allclose(imager.intensity(source).data, 4 * baseline, rtol=3e-6)
    assert np.allclose(imager.psf(source).data, imager.psf().data, rtol=3e-6)
    assert np.array_equal(imager.psf(dl.Source(flux=0)).data, imager.psf().data)


def test_normalisation_precedes_filter_qe_and_gain(imager):
    baseline = imager.intensity().data
    model = imager.set(filter=dlu.Filter(0.4), detector=dl.Detector(qe=0.3, gain=5.0))
    assert np.allclose(
        model.image(dl.Source(flux=7)).data, baseline * 7 * 0.4 * 0.3 * 5, rtol=3e-6
    )
    # PSFs contain optical responses only; all spectral losses belong to weights.
    assert np.allclose(model.psf(dl.Source(flux=7)).data, imager.psf().data, rtol=3e-6)
    assert np.allclose(model.photometry(dl.Source(flux=7)).value.sum(), 7 * 0.4 * 0.3)


def test_binary_uses_the_ordinary_positions_and_flux_path(imager):
    source = dl.BinarySource(
        position=[1e-8, 2e-8],
        separation=1.5e-7,
        pa=np.rad2deg(0.3),
        unit="rad",
        primary_flux=2,
        contrast=0.4,
        log=False,
    )
    angle = np.array([np.sin(0.3), np.cos(0.3)])
    positions = np.stack(
        (np.array([1e-8, 2e-8]), np.array([1e-8, 2e-8]) + 1.5e-7 * angle)
    )
    expected = imager.image(dl.Source(positions=positions, flux=[2, 0.8])).data
    actual = assert_jittable(
        lambda source: imager.image(source).data, source, rtol=2e-5
    )
    assert np.allclose(actual, expected, rtol=2e-5)


def test_nested_scenes_add_intensities_and_position_offsets(imager):
    scene = dl.Scene(
        {
            "a": dl.Source(flux=2),
            "pair": dl.Scene({"b": dl.Source(flux=3)}, position=[2e-7, 0]),
        },
        position=[0, 1e-8],
    )
    expected = (
        imager.image(dl.Source(positions=[0, 1e-8], flux=2)).data
        + imager.image(dl.Source(positions=[2e-7, 1e-8], flux=3)).data
    )
    actual = assert_jittable(lambda source: imager.image(source).data, scene, rtol=2e-5)
    assert np.allclose(actual, expected, rtol=2e-5)
    psfs = imager.psf(scene)
    assert psfs["a"].data.shape == (1, 3, 12, 12)
    assert psfs["pair"]["b"].data.shape == (1, 3, 12, 12)
    fields = imager.wavefront(scene)
    assert fields["a"].phasor.shape == (1, 3, 12, 12)
    assert fields["pair"]["b"].phasor.shape == (1, 3, 12, 12)


def test_scene_is_summed_before_nonlinear_detector_response(imager):
    scene = dl.Scene({"a": dl.Source(flux=2), "b": dl.Source(flux=3)})
    detector = dl.Detector(layers=[dl.SaturationLayer(0.2)])
    expected = np.minimum(imager.intensity(scene).data, 0.2)
    assert np.allclose(imager.image(scene, detector=detector).data, expected)


def test_scene_links_populate_from_the_containing_model(imager):
    flux = zdx.Linked(10.0, key="shared_flux")
    scene = dl.Scene(
        {"a": dl.Source(flux=flux), "b": dl.Source(flux=zdx.Map(x=flux.defer(), s=0.1))}
    )
    model = imager.set(source=scene)
    actual = assert_jittable(lambda model: model.intensity().data, model)
    expected = imager.intensity(dl.Source(flux=11.0)).data
    assert np.allclose(actual, expected)
    grad = jax.jit(
        jax.grad(
            lambda value: model.set("source.sources.a.flux.value", value)
            .intensity()
            .data.sum()
        )
    )(np.asarray(10.0))
    assert np.allclose(grad, expected.sum() / 10, rtol=2e-5)


@pytest.mark.parametrize("factor", [2, (2, 3)])
def test_oversampling_matches_explicit_fine_propagation(imager, factor):
    model = imager.set(oversample=dlu.as_size(factor, 2))
    fine = imager.set(image_grid=imager.image_grid.oversample(factor))
    assert np.allclose(model.wavefront().phasor, fine.wavefront().phasor)
    expected = fine.intensity()
    actual = assert_jittable(lambda model: model.intensity(), model)
    assert_tree_allclose(actual, expected, rtol=2e-5)
    assert np.allclose(actual.data.sum(), fine.intensity().data.sum())
    assert np.allclose(imager.intensity(oversample=factor).data, actual.data)


def test_detector_runs_response_integration_noise_then_gain(imager):
    detector = dl.Detector(
        layers=[dl.SensitivityLayer(0.7)], downsample=2, read_noise=2, gain=3
    )
    model = imager.set(detector=detector, oversample=(2, 2))
    fine = imager.set(image_grid=imager.image_grid.oversample(2)).intensity()
    counts = (fine * 0.7).downsample(4, mean=False)
    assert np.allclose(model.image().data, counts.data * 3)
    assert np.allclose(model.intensity().data, fine.data)
    key = jr.key(7)
    photon_key, read_key = jr.split(jr.split(key, 1)[0])
    expected = (
        jr.poisson(photon_key, counts.data) + 2 * jr.normal(read_key, counts.data.shape)
    ) * 3
    actual = assert_jittable(lambda value: value.simulate(key), model)
    assert np.allclose(actual.data, expected)
    assert model.psf().data.shape == (1, 3, 24, 24)
    assert actual.data.shape == (6, 6)


def test_explicit_detector_grid_is_native_sampling(imager):
    target = dl.Grid(n=(10, 8), d=(1.1e-7, 1.2e-7), unit="rad")
    operation = dlu.Interp(grid_out=target)
    model = imager.set(oversample=(2, 3), detector=dl.Detector(resample=operation))
    fine = imager.set(
        image_grid=imager.image_grid.oversample((2, 3)),
        detector=dl.Detector(
            resample=operation.set(grid_out=target.oversample((2, 3)))
        ),
    )
    expected = fine.intensity()
    assert_tree_allclose(model.intensity(), expected, rtol=3e-5)
    assert model.intensity().grid.n == target.oversample((2, 3)).n
    assert model.psf().grid.n == model.propagation_grid.n


def test_array_jitter_offsets_stay_in_native_pixels(imager):
    kernel = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 0.0]])
    model = imager.set(jitter=kernel, oversample=(2, 3))
    fine_kernel = np.zeros((7, 5)).at[::3, ::2].set(kernel)
    fine = imager.set(
        image_grid=imager.image_grid.oversample((2, 3)), jitter=fine_kernel
    )
    assert np.allclose(model.intensity().data, fine.intensity().data, rtol=3e-5)


def test_pixel_and_angular_jitter_widths_agree(imager):
    pixel = dlu.Jitter(0.7, kernel_size=5)
    physical = dlu.Jitter(0.7e-7, kernel_size=5, unit="rad")
    model = imager.set(oversample=(2, 2), jitter=pixel)
    assert np.allclose(
        model.intensity().data, model.set(jitter=physical).intensity().data, rtol=3e-5
    )


def test_resolved_source_expression_and_sampled_field_agree(imager):
    model = imager.set(oversample=(2, 2))
    profile = dlu.Gaussian(8e-8)
    grid = dl.Grid(n=25, d=5e-8, unit="rad")
    samples = profile.resolve(grid=grid)
    parametric = dl.ResolvedSource(profile=profile, flux=2)
    sampled = dl.ResolvedSource(profile=samples, profile_grid=grid, flux=2)
    field = dl.ResolvedSource(profile=dlu.Field(samples, grid), flux=2)
    actual = assert_jittable(
        lambda source: model.intensity(source).data, parametric, rtol=3e-5
    )
    assert np.allclose(actual, model.intensity(sampled).data, rtol=3e-5)
    assert np.allclose(actual, model.intensity(field).data, rtol=3e-5)
    point = model.set(oversample=(1, 1), image_grid=model.propagation_grid).intensity()
    expected = point.convolve(samples / samples.sum()).data * 2
    assert np.allclose(actual, expected, rtol=3e-5)


def test_independent_scene_spectra_use_their_own_sampling(imager):
    model = imager.set(bandpass=None)
    scene = dl.Scene(
        {
            "a": dl.Source(dlu.Spectrum(bandpass=dlu.BandPass(5e-7), flux=2), flux=2),
            "b": dl.Source(dlu.Spectrum(bandpass=dlu.BandPass(7e-7), flux=3), flux=3),
        }
    )
    expected = sum(model.intensity(source).data for source in scene.sources.values())
    assert np.allclose(model.intensity(scene).data, expected)


def test_sparse_imager_includes_aperture_centres_in_source_tilt(imager):
    pupil = dl.SparseOptic([[-0.6, 0.0], [0.6, 0.0]], transmission=dlu.Circle(0.5))
    model = imager.set(pupil=pupil)
    source = dl.Source(positions=[1e-7, 0.0])
    actual = assert_jittable(lambda value: value.wavefront(source), model, rtol=2e-5)
    incident = dl.Wavefront(model.bandpass.wavelengths, model.pupil_grid)
    incident = incident.set(phasor=incident.phasor / np.sqrt(2))
    local = pupil(incident).tilt(np.array([1e-7, 0.0]))
    expected = local.propagate(model.propagator).coherent_sum()
    assert np.allclose(actual.phasor[0], expected.phasor, rtol=2e-5, atol=2e-7)


def test_coronagraph_identity_matches_imager_and_rejects_resolved_profile(imager):
    model = dl.Coronagraph(
        imager.pupil_grid,
        dl.Grid(n=8, d=1e-7, unit="rad"),
        imager.image_grid,
        focal=dl.Optic(transmission=1),
        source=dl.Source(),
        bandpass=imager.bandpass,
    )
    assert np.allclose(model.image().data, imager.image().data, rtol=2e-5)
    with pytest.raises(ValueError, match="resolved source"):
        model.image(dl.Scene({"disk": dl.ResolvedSource(profile=dlu.Gaussian(1e-7))}))


def test_position_and_flux_gradients_are_observable(imager):
    source = dl.BinarySource(
        separation=1.5e-7, pa=np.rad2deg(0.3), unit="rad", contrast=0.4, log=False
    )
    fn = lambda source: imager.image(source).data[5, 7]
    derivative = eqx.filter_jit(eqx.filter_grad(fn))(source)
    assert np.isfinite(derivative.positions.separation)
    assert np.abs(derivative.positions.separation) > 1
    assert np.abs(derivative.flux.contrast) > 1e-5


def test_focal_length_metadata_converts_length_to_angular_grid(imager):
    focal = 2.0
    physical = dl.Grid(n=12, d=2e-7, unit="m")
    model = imager.set(image_grid=physical, pupil=dl.Optic(focal_length=focal))
    assert np.allclose(model.psf().data, imager.psf().data, rtol=2e-5)
    with pytest.raises(ValueError, match="focal_length"):
        dl.Imager(imager.pupil_grid, physical)


def test_interferometer_and_sparse_imager_share_incident_flux(imager):
    aperture = dl.SparseOptic([[-0.6, 0.0], [0.6, 0.0]], transmission=dlu.Circle(0.5))
    source = dl.Source(
        dlu.Spectrum(bandpass=dlu.BandPass(6e-7)), positions=[1e-7, 0], flux=3
    )
    model = imager.set(pupil=aperture, bandpass=None, source=source)
    interferometer = dl.Interferometer(imager.pupil_grid, imager.image_grid, aperture)
    output = assert_jittable(
        lambda system: system.model(source), interferometer, rtol=3e-5
    )
    assert np.allclose(output.data, model.intensity().data, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize(
    "kwargs,exception,message",
    [
        ({"source": object()}, TypeError, "Source, Scene"),
        ({"detector": object()}, TypeError, "must be a Detector"),
        ({"bandpass": np.ones((2, 2))}, ValueError, "one-dimensional"),
        ({"bandpass": np.array([])}, ValueError, "at least one wavelength"),
        ({"jitter": np.ones(3)}, ValueError, "two-dimensional kernel"),
    ],
)
def test_imager_validates_constructor_and_method_inputs(
    imager, kwargs, exception, message
):
    with pytest.raises(exception, match=message):
        dl.Imager(imager.pupil_grid, imager.image_grid, **kwargs)
    with pytest.raises(exception, match=message):
        imager.intensity(**kwargs)


def test_imager_array_overrides_match_constructor_conversion(imager):
    expected = imager.set(filter=dlu.Filter(0.4)).intensity().data
    assert np.allclose(imager.intensity(filter=0.4).data, expected)
    assert np.allclose(
        imager.psf(bandpass=[5e-7, 6e-7, 7e-7]).data,
        imager.psf(bandpass=np.array([5e-7, 6e-7, 7e-7])).data,
    )


def test_interferometer_inherits_full_imager_detector_route(imager):
    aperture = dl.SparseOptic([[-0.6, 0.0], [0.6, 0.0]], transmission=dlu.Circle(0.5))
    detector = dl.Detector(qe=0.7, gain=3)
    model = dl.Interferometer(
        imager.pupil_grid,
        imager.image_grid,
        aperture,
        focal_plane=dl.Optic(transmission=0.6),
        source=dl.BinarySource(
            separation=1e-7, unit="rad", contrast=0.5, log=False, primary_flux=2
        ),
        bandpass=imager.bandpass,
        filter=0.4,
        detector=detector,
        oversample=2,
    )
    reference = dl.Imager(
        imager.pupil_grid,
        imager.image_grid,
        aperture,
        source=model.source,
        bandpass=model.bandpass,
        filter=0.4,
        detector=detector,
        oversample=2,
    )
    assert isinstance(model, dl.Imager)
    output = assert_jittable(lambda value: value.image(), model, rtol=3e-5)
    assert np.allclose(
        output.data, reference.image().data * 0.6**2, rtol=3e-5, atol=1e-7
    )
    assert np.allclose(model.model().data, model.image().data)


def test_point_propagation_wrappers_preserve_explicit_weight_scale(imager):
    wavelengths = np.array([5e-7, 7e-7])
    weights = np.array([0.3, 0.9])
    reference = sum(
        weight * imager.propagate_mono(wavelength).data
        for wavelength, weight in zip(wavelengths, weights)
    )
    assert np.allclose(imager.propagate(wavelengths, weights=weights).data, reference)
    assert np.allclose(
        imager.propagate_mono(wavelengths[0]).data,
        imager.propagate(wavelengths[:1]).data,
    )


def test_interferometer_custom_mft_keeps_physical_grid_and_oversampling(imager):
    aperture = dl.SparseOptic([[-0.6, 0.0], [0.6, 0.0]], transmission=dlu.Circle(0.5))
    physical = dl.Grid(n=12, d=2e-7, unit="m")
    relay = dlu.Propagator(physical, focal_length=2.0)
    model = dl.Interferometer(
        imager.pupil_grid,
        physical,
        aperture,
        propagator=relay,
        source=imager.source,
        bandpass=imager.bandpass,
        oversample=2,
    )
    angular = imager.set(pupil=aperture, oversample=(2, 2))
    assert np.allclose(model.image().data, angular.image().data, rtol=3e-5, atol=1e-7)
    assert model.propagator.grid_out.n == (24, 24)


def test_scalar_wavelength_is_optical_only(imager):
    reference = imager.psf(bandpass=dlu.BandPass(6e-7)).data
    assert np.allclose(imager.psf(bandpass=6e-7).data, reference)
    with pytest.raises(TypeError, match="requires a BandPass"):
        imager.intensity(bandpass=6e-7)


def test_angular_log_binary_matches_cartesian_source(imager):
    binary = dl.BinarySource(
        position=[0.01, -0.02], separation=0.05, pa=60, contrast=-2, primary_flux=10
    )
    primary = np.array([0.01, -0.02])
    companion = primary + 0.05 * np.array([np.sin(np.pi / 3), np.cos(np.pi / 3)])
    cartesian = dl.Source(
        positions=dlu.Positions(np.stack([primary, companion])), flux=[10, 0.1]
    )
    actual = assert_jittable(lambda source: imager.image(source).data, binary)
    assert np.allclose(actual, imager.image(cartesian).data, rtol=2e-5)
