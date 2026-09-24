"""The familiar system entry point uses the same rendering route as Imager."""

import equinox as eqx
import jax
import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu


@pytest.fixture
def setup():
    pupil = dl.Grid(8, diam=1, unit="m")
    image = dl.Grid(8, d=0.1, unit="arcsec")
    optic = dl.Optic(opd=np.arange(64.0).reshape(8, 8) * 1e-9)
    system = dl.OpticalSystem([("pupil", optic), dlu.Propagator(image)], pupil)
    source = dl.PointSource(
        spectrum=dlu.Spectrum([1, 2, 1], dlu.BandPass(6e-7, 2e-7, n=3), flux=5),
        position=[1e-8, -2e-8],
    )
    imager = dl.Imager(pupil, image, optic, source=source)
    return system, source, imager


def test_propagation_returns_and_weighting(setup):
    system, source, _ = setup
    wavelengths = source.spectrum.bandpass.wavelengths
    weights = np.array([1.0, 2.0, 3.0])
    expected = sum(
        weight * system.propagate_mono(wavelength)
        for wavelength, weight in zip(wavelengths, weights)
    )
    psf = system.propagate(wavelengths, weights=weights, return_psf=True)
    assert isinstance(psf, dl.PSF)
    assert np.allclose(psf.data, expected, rtol=2e-6, atol=1e-8)
    field = system.propagate(wavelengths, weights=weights, return_wf=True)
    assert np.allclose(field.intensity.sum(0), expected, rtol=2e-6, atol=1e-8)
    assert np.allclose(
        system.propagate(wavelengths),
        sum(system.propagate_mono(w) for w in wavelengths) / 3,
        rtol=2e-6,
        atol=1e-8,
    )
    final, stages = system.debug_propagate_mono(wavelengths[0])
    assert tuple(stages) == ("input", "pupil", "Propagator")
    assert np.allclose(
        final.phasor, system.propagate_mono(wavelengths[0], return_wf=True).phasor
    )


def test_model_matches_imager_and_has_flux_gradient(setup):
    system, source, imager = setup
    expected = imager.intensity().data
    actual = eqx.filter_jit(lambda optics, star: optics.model(star))(system, source)
    assert np.allclose(actual, expected, rtol=2e-6, atol=1e-8)
    result = system.model(source, return_all=True)
    assert np.allclose(result["PSF"].data, expected, rtol=2e-6, atol=1e-8)
    field = system.model(source, return_wf=True)
    assert field.phasor.shape == (3, 8, 8)
    assert np.allclose(field.intensity.sum(0), expected, rtol=2e-6, atol=1e-8)
    derivative = jax.grad(
        lambda flux: system.model(source.set("spectrum.flux", flux)).sum()
    )(np.array(5.0))
    assert np.allclose(derivative, expected.sum() / 5, rtol=2e-6)


def test_scene_and_resolved_profile_share_imager_rendering(setup):
    system, source, imager = setup
    disk = dl.ResolvedSource(source.spectrum, profile=dlu.Gaussian(std=1e-7))
    scene = dl.Scene({"star": source, "disk": disk}, position=[1e-8, 0])
    expected = imager.intensity(scene).data
    assert np.allclose(system.model(scene), expected, rtol=2e-6, atol=1e-8)
    result = system.model(scene, return_all=True)
    assert set(result["Wavefront"]) == {"star", "disk"}
    assert np.allclose(result["PSF"].data, expected, rtol=2e-6, atol=1e-8)


def test_jones_weights_and_reduced_optical_sampling(setup):
    system, source, imager = setup
    optic = system.layers["pupil"].set(jones=dlu.Jones.HalfWavePlate(angle=0.2))
    system = system.set("layers.pupil", optic)
    imager = imager.set(pupil=optic)
    bandpass = dlu.BandPass(6e-7, 2e-7, n=1)
    field = system.model(source, bandpass=bandpass, return_wf=True)
    assert np.allclose(
        field.intensity.sum(0),
        imager.intensity(bandpass=bandpass).data,
        rtol=2e-6,
        atol=1e-8,
    )
    # Explicit Jones axes also keep the wavelength weight outside those axes.
    jones = np.broadcast_to(dlu.Jones.half_wave_plate(angle=0.2), (8, 8, 2, 2))
    system = system.set("layers.pupil.jones", jones)
    field = system.propagate([5e-7, 6e-7], weights=[2, 3], return_wf=True)
    assert field.jones is True
    expected = system.propagate([5e-7, 6e-7], weights=[2, 3])
    assert np.allclose(field.intensity.sum(0), expected, rtol=2e-6, atol=1e-8)


def test_sampling_and_return_errors(setup):
    system, source, _ = setup
    with pytest.raises(ValueError, match="grid"):
        dl.OpticalSystem([]).propagate([6e-7])
    with pytest.raises(ValueError, match="weights"):
        system.propagate([6e-7], weights=[1, 2])
    with pytest.raises(ValueError, match="Select"):
        system.model(source, return_psf=True, return_wf=True)


def test_propagation_validation_tails(setup):
    system, _, _ = setup
    with pytest.raises(TypeError, match="grid must be a Grid"):
        dl.OpticalSystem([], grid=np.ones((2, 2)))
    with pytest.raises(ValueError, match="wavelength must be scalar"):
        system.propagate_mono(np.array([5e-7, 6e-7]))
    with pytest.raises(ValueError, match="Select"):
        system.propagate_mono(6e-7, return_wf=True, return_psf=True)
    with pytest.raises(ValueError, match="Select"):
        system.propagate([6e-7], return_all=True, return_psf=True)
    with pytest.raises(ValueError, match="non-empty vector"):
        system.propagate([])


def test_zero_weight_retains_linear_intensity_gradient(setup):
    system, source, _ = setup
    wavelengths = source.spectrum.bandpass.wavelengths
    derivative = jax.grad(
        lambda weights: system.propagate(wavelengths, weights=weights).sum()
    )(np.array([0.0, 1.0, 1.0]))
    expected = np.stack([system.propagate_mono(w).sum() for w in wavelengths])
    assert np.allclose(derivative, expected, rtol=2e-6, atol=1e-8)


def test_nested_sequence_and_weighted_return_all_match_flat_system(setup):
    system, source, _ = setup
    nested = dl.OpticalSystem([("relay", dl.Sequential(system.layers))], system.grid)
    wavelengths = source.spectrum.bandpass.wavelengths
    weights = np.array([0.0, 2.0, 3.0])
    expected_fields = np.stack(
        [system.propagate_mono(w, return_wf=True).phasor for w in wavelengths]
    )

    result = eqx.filter_jit(
        lambda value: value.propagate(wavelengths, weights=weights, return_all=True)
    )(nested)
    assert nested.propagator is system.propagator
    assert set(result) == {"Wavefront", "PSF"}
    expected_weighted = expected_fields * np.sqrt(weights)[:, None, None]
    eager = nested.propagate(wavelengths, weights=weights, return_all=True)
    assert np.array_equal(eager["Wavefront"].phasor, expected_weighted)
    # Fused float32 propagation changes rounding near diffraction zeros. Compare
    # complex-field error relative to total field amplitude, not individual zeros.
    error = np.linalg.norm(result["Wavefront"].phasor - expected_weighted)
    assert error < 2e-6 * np.linalg.norm(expected_weighted)
    expected_intensity = (abs(expected_fields) ** 2 * weights[:, None, None]).sum(0)
    assert np.allclose(result["PSF"].data, expected_intensity, rtol=2e-6, atol=1e-8)
    assert np.allclose(nested.model(source), system.model(source), rtol=2e-6)
