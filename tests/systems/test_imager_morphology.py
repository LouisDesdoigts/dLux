"""Resolved morphology broadens optical amplitudes before detector resampling."""

import equinox as eqx
import jax
import jax.numpy as np
from jax.scipy.signal import convolve
import pytest

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def make_imager(pupil=None, detector=None):
    return dl.Imager(
        dl.Grid(n=8, diam=1.0, unit="m"),
        dl.Grid(n=10, d=1e-7, unit="rad"),
        pupil=pupil,
        source=dl.Source(),
        bandpass=dlu.BandPass(6e-7),
        detector=detector,
        oversample=2,
    )


@pytest.mark.parametrize("uniform_jones", [False, True])
def test_morphology_rescales_amplitude_before_detector_transform(uniform_jones):
    matrix = np.array([[0.8, 0.1j], [0, 0.5]])
    pupil = dl.Optic(jones=matrix) if uniform_jones else None
    detector = dl.Detector(resample=dlu.Interp(rotation=0.27, scale=(0.9, 1.1)), gain=2)
    model = make_imager(pupil, detector)
    grid = dl.Grid(n=21, d=5e-8, unit="rad")
    x, y = grid.coordinates
    profile = np.exp(-0.5 * ((x / 9e-8) ** 2 + (y / 4e-8) ** 2))
    source = dl.ResolvedSource(profile=profile, profile_grid=grid, flux=3)

    # Independently convolve one fine point-response intensity, retain its phase,
    # and pass the resulting complex amplitudes through the detector transform.
    field = model.wavefront()
    scalar = field.phasor[0, 0]
    broadened = convolve(np.abs(scalar) ** 2, profile / profile.sum(), mode="same")
    amplitude = np.sqrt(np.maximum(broadened, 0))
    phasor = amplitude * np.exp(1j * np.angle(scalar))
    expected_field = field.set(phasor=phasor[None, None])
    expected = detector.sample(expected_field, oversample=2).to_intensity()
    expected = expected.set(data=expected.data.sum((0, 1)) * 3)
    expected = detector(expected.downsample(2, mean=False))
    actual = assert_jittable(lambda source: model.image(source), source)
    assert np.allclose(actual.data, expected.data, rtol=4e-5, atol=2e-7)


class FixedFieldImager(dl.Imager):
    """Supply a known point field to isolate amplitude and zero-phase handling."""

    samples: jax.Array

    def __init__(self, samples):
        grid = dl.Grid(n=5, d=1e-7, unit="rad")
        super().__init__(
            dl.Grid(n=5, diam=1, unit="m"), grid, bandpass=dlu.BandPass(6e-7)
        )
        self.samples = samples

    def _wavefront(self, wavelengths, positions, **context):
        shape = (wavelengths.size, positions.shape[0]) + self.samples.shape
        return dl.Wavefront(
            wavelengths, self.image_grid, np.broadcast_to(self.samples, shape)
        )


def test_resolved_amplitude_fills_dark_pixels_with_finite_gradients():
    samples = np.zeros((5, 5), dtype=complex).at[2, 2].set(1j)
    model = FixedFieldImager(samples).set(detector=dl.Detector(resample=dlu.Interp()))
    profile = np.array([[0.0, 0.1, 0.0], [0.2, 0.4, 0.2], [0.0, 0.1, 0.0]])
    source = dl.ResolvedSource(
        profile=profile, profile_grid=dl.Grid(n=3, d=1e-7, unit="rad")
    )
    expected = np.zeros((5, 5)).at[1:4, 1:4].set(profile)
    actual = assert_jittable(lambda value: value.intensity(source).data, model)
    assert np.allclose(actual, expected, atol=2e-7)
    gradient = eqx.filter_jit(
        eqx.filter_grad(lambda value: value.intensity(source).data.sum())
    )(model)
    assert np.isfinite(gradient.samples).all()
    assert np.abs(gradient.samples[2, 2]) > 0


def spatial_jones():
    """A varying pupil response whose point PSF retains polarisation cross terms."""
    x, y = dl.Grid(n=8, diam=1, unit="m").coordinates
    return np.stack(
        (
            np.stack((0.65 + 0.1 * x, 0.2 * np.exp(2j * np.pi * x)), -1),
            np.stack((0.1 + 0.1j * y, 0.45 * np.ones_like(x)), -1),
        ),
        -2,
    )


@pytest.mark.parametrize("stokes", [None, np.array([1.0, 0.2, 0.5, -0.3])])
def test_pupil_jones_profile_convolves_the_full_stokes_response(stokes):
    detector = dl.Detector(qe=0.7, gain=2)
    model = make_imager(dl.Optic(jones=spatial_jones()), detector)
    grid = dl.Grid(n=21, d=5e-8, unit="rad")
    x, y = grid.coordinates
    profile = np.exp(-0.5 * ((x / 9e-8) ** 2 + (y / 4e-8) ** 2))
    source = dl.ResolvedSource(profile=profile, profile_grid=grid, flux=3)

    # Form J C J* directly, including off-diagonal input coherency. Squaring the
    # matrix entries independently would lose these terms for polarised input.
    intensity, q, u, v = np.array([1.0, 0.0, 0.0, 0.0]) if stokes is None else stokes
    coherency = 0.5 * np.array(
        [[intensity + q, u + 1j * v], [u - 1j * v, intensity - q]]
    )
    matrix = model.wavefront().phasor[0, 0]
    response = np.einsum("abxy,bc,acxy->xy", matrix, coherency, matrix.conj()).real
    if stokes is not None:
        diagonal = np.diag(np.diag(coherency))
        without_cross_terms = np.einsum(
            "abxy,bc,acxy->xy", matrix, diagonal, matrix.conj()
        ).real
        assert not np.allclose(response, without_cross_terms, rtol=1e-3, atol=1e-7)
    broadened = convolve(response, profile / profile.sum(), mode="same")
    expected = dl.PSF(broadened, model.propagation_grid).downsample(2, mean=False)
    expected = expected.data * 3 * 0.7 * 2
    actual = assert_jittable(
        lambda source: model.image(source, stokes=stokes).data, source
    )
    assert np.allclose(actual, expected, rtol=4e-5, atol=2e-7)


def test_explicit_jones_reconstruction_requires_a_detector_resampling_decision():
    detector = dl.Detector(resample=dlu.Interp(rotation=0.2))
    model = make_imager(dl.Optic(jones=spatial_jones()), detector)
    source = dl.ResolvedSource(profile=dlu.Gaussian(1e-7))
    with pytest.raises(ValueError, match="[Rr]esampl"):
        model.intensity(source)


def test_resolved_profile_uses_custom_interferometer_relay_scale():
    pupil_grid = dlu.Grid(n=6, diam=0.5, unit="m")
    image_grid = dlu.Grid(n=8, d=2e-7, unit="m")
    aperture = dl.SparseOptic([[-0.3, 0], [0.3, 0]], transmission=dlu.Circle(0.2))
    source = dl.ResolvedSource(profile=dlu.Gaussian(1e-7), flux=3)
    model = dl.Interferometer(
        pupil_grid,
        image_grid,
        aperture,
        propagator=dlu.Propagator(image_grid, focal_length=2.0),
        source=source,
        bandpass=dlu.BandPass(6e-7),
    )
    # This relay maps angles to distances by x = focal_length * angle.
    angular = dl.Interferometer(
        pupil_grid,
        dlu.Grid(n=8, d=1e-7, unit="rad"),
        aperture,
        source=source,
        bandpass=dlu.BandPass(6e-7),
    )
    actual = assert_jittable(lambda value: value.intensity(), model)
    assert np.allclose(actual.data, angular.intensity().data, rtol=3e-5, atol=1e-7)
