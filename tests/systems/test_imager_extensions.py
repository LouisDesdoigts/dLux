"""Optical stages can be reused and overridden independently of source modelling."""

import equinox as eqx
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu


def make_imager(cls=dl.Imager):
    return cls(
        dlu.Grid(6, diam=1.0, unit="m"),
        dlu.Grid(6, d=1e-7, unit="rad"),
        pupil=dl.Optic(transmission=0.8),
        bandpass=dlu.BandPass(range=(500.0, 700.0), n=3, unit="nm"),
    )


def test_imager_requires_bins_only_for_photometric_calculations():
    model = make_imager()
    wavelengths = model.bandpass.wavelengths
    assert np.allclose(model.psf(bandpass=wavelengths).data, model.psf().data)
    for operation in (model.photometry, model.intensity, model.image):
        with pytest.raises(TypeError, match="requires a BandPass"):
            operation(bandpass=wavelengths)
    source = dl.Source(dlu.Spectrum(2.0, bandpass=wavelengths, flux=None))
    with pytest.raises(TypeError, match="requires a BandPass"):
        model.set(bandpass=None).intensity(source)


def test_pupil_field_can_feed_two_branches_without_reapplying_the_pupil():
    model = make_imager()
    incident = model.initialise_wavefront(model.bandpass.wavelengths)
    pupil = model.apply_pupil(incident)
    science = model.propagate_image(pupil)
    assert np.allclose(science.phasor, model.apply(incident).phasor)

    # Reimaging is an independent consumer of the same immutable pupil field.
    other_grid = dlu.Grid(8, d=2e-7, unit="rad")
    other = pupil.propagate(dlu.Propagator(other_grid))
    assert other.grid.n == (8, 8)
    assert np.allclose(pupil.phasor, incident.phasor * 0.8)
    assert np.allclose(model.propagate_image(pupil).phasor, science.phasor)


def test_initial_tilt_matches_offset_propagation_under_jit():
    model = make_imager()
    offset = np.array([1e-7, -2e-8])
    wavelength = 6e-7

    def propagate(system, position):
        incident = system.initialise_wavefront(wavelength, offset=position)
        return system.apply(incident)

    actual = eqx.filter_jit(propagate)(model, offset)
    expected = model.propagate_mono(wavelength, offset).data
    assert np.allclose(actual.intensity, expected, rtol=2e-5, atol=1e-8)
    assert not np.allclose(actual.intensity, model.propagate_mono(wavelength).data)


@pytest.mark.parametrize("stage", ["psf", "intensity", "image"])
def test_grid_and_pupil_overrides_match_a_separately_constructed_model(stage):
    original = make_imager()
    pupil_grid = dlu.Grid(8, diam=1.2, unit="m")
    image_grid = dlu.Grid(10, d=8e-8, unit="rad")
    pupil = dl.Optic(transmission=0.5)
    expected = dl.Imager(pupil_grid, image_grid, pupil, bandpass=original.bandpass)

    actual = getattr(original, stage)(
        pupil_grid=pupil_grid, image_grid=image_grid, optics=pupil
    )
    assert actual.data.shape[-2:] == (10, 10)
    assert np.allclose(actual.data, getattr(expected, stage)().data, rtol=2e-5)
    assert original.pupil_grid.n == (6, 6)
    assert original.image_grid.n == (6, 6)
    assert original.pupil.transmission == 0.8


def test_overridden_pupil_stage_is_used_by_psf_intensity_and_direct_propagation():
    class HalfAmplitude(dl.Imager):
        def apply_pupil(self, wavefront, offset=None, *, unit_input=False, **context):
            field = super().apply_pupil(
                wavefront, offset, unit_input=unit_input, **context
            )
            return field.set(phasor=field.phasor * 0.5)

    original, changed = make_imager(), make_imager(HalfAmplitude)
    for operation in ("psf", "intensity", "image"):
        actual = eqx.filter_jit(lambda m: getattr(m, operation)().data)(changed)
        assert np.allclose(
            actual, getattr(original, operation)().data * 0.25, rtol=2e-5
        )
    assert np.allclose(
        changed.propagate_mono(6e-7).data,
        original.propagate_mono(6e-7).data * 0.25,
        rtol=2e-5,
    )


def test_coronagraph_stages_reproduce_soummer_and_retain_branch_fields():
    model = make_imager()
    coronagraph = dl.Coronagraph(
        model.pupil_grid,
        dlu.Grid(4, d=2e-7, unit="rad"),
        model.image_grid,
        pupil=model.pupil,
        focal=dl.Optic(transmission=0.4),
        lyot=dl.Optic(transmission=0.9),
        bandpass=model.bandpass,
    )
    incident = coronagraph.initialise_wavefront(model.bandpass.wavelengths)
    pupil = coronagraph.apply_pupil(incident)
    focal = coronagraph.propagate_focal(pupil)
    masked = coronagraph.apply_focal(focal)
    lyot = coronagraph.propagate_lyot(pupil, focal, masked)
    reference = dl.SoummerFPMLayer(coronagraph.focal, coronagraph.focal_propagator)(
        pupil
    )
    assert np.allclose(lyot.phasor, reference.phasor, rtol=2e-5, atol=1e-7)
    science = coronagraph.propagate_image(coronagraph.apply_lyot(lyot))
    actual = eqx.filter_jit(lambda m: m.wavefront().phasor)(coronagraph)
    assert np.allclose(actual[0], science.phasor, rtol=2e-5, atol=1e-7)
    assert np.allclose(masked.phasor, focal.phasor * 0.4)


def test_psf_and_photometry_overrides_feed_the_shared_image_calculation():
    class ModifiedStages(dl.Imager):
        def _psf(self, wavefront, positions, stokes=None):
            return super()._psf(wavefront, positions, stokes) * 2

        def _photometry(self, source, bandpass, filter=None, qe=None, **context):
            return super()._photometry(source, bandpass, filter, qe, **context) * 0.25

    original, changed = make_imager(), make_imager(ModifiedStages)
    assert np.allclose(changed.psf().data, original.psf().data * 2)
    assert np.allclose(changed.photometry().value, original.photometry().value * 0.25)
    actual = eqx.filter_jit(lambda m: m.image().data)(changed)
    assert np.allclose(actual, original.image().data * 0.5, rtol=2e-5)


def test_independent_coronagraph_stages_populate_links_from_the_full_owner():
    model = make_imager()
    shared = zdx.Linked(np.asarray(0.8), key="shared_transmission")
    coronagraph = dl.Coronagraph(
        model.pupil_grid,
        dlu.Grid(4, d=2e-7, unit="rad"),
        model.image_grid,
        pupil=dl.Optic(transmission=shared),
        focal=dl.Optic(transmission=shared.defer()),
        lyot=dl.Optic(transmission=shared.defer()),
        bandpass=model.bandpass,
    )

    def stages(system):
        incident = system.initialise_wavefront(system.bandpass.wavelengths)
        pupil = system.apply_pupil(incident)
        focal = system.propagate_focal(pupil)
        masked = system.apply_focal(focal)
        lyot = system.propagate_lyot(pupil, focal, masked)
        return system.propagate_image(system.apply_lyot(lyot))

    actual = eqx.filter_jit(stages)(coronagraph)
    expected = coronagraph.wavefront()
    assert np.allclose(actual.phasor, expected.phasor[0], rtol=2e-5, atol=1e-7)
