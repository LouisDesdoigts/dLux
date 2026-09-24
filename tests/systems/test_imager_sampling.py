"""Independent optical sampling and fine photometric integration."""

import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


class ChromaticJones(zdx.Expression):
    """A uniform, wavelength-dependent amplitude and phase response."""

    def evaluate(self, *, wavelengths, **context):
        ratio = wavelengths / 6e-7
        first = 0.8 * np.exp(0.3j * ratio)
        second = 0.4 * ratio * np.exp(-0.5j * ratio)
        zero = np.zeros_like(ratio)
        return np.stack((np.stack((first, zero), -1), np.stack((zero, second), -1)), -2)


def make_imager(n_optical, pupil):
    native = dlu.BandPass(wavelengths=[500, 550, 600, 650, 700], unit="nm")
    # The independent integral below assumes piecewise-linear input curves.
    spectrum = dlu.Spectrum(
        [1.0, 2.0, 4.0, 2.0, 1.0], native, flux=7.0, method="linear"
    )
    source = dl.Source(spectrum)
    return dl.Imager(
        dl.Grid(n=8, diam=1.0, unit="m"),
        dl.Grid(n=10, d=1e-7, unit="rad"),
        pupil=pupil,
        source=source,
        filter=dlu.Filter([0.3, 0.6, 1.0, 0.8, 0.2], native, method="linear"),
        bandpass=dlu.BandPass(6e-7, 2e-7, n=n_optical),
        detector=dl.Detector(qe=0.7),
    )


def detected_counts():
    """Integrate the product of two linear curves independently of dLux."""
    wavelengths = np.linspace(5e-7, 7e-7, 5)
    density = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    response = np.array([0.3, 0.6, 1.0, 0.8, 0.2]) * 0.7
    # The product is quadratic within each interval, so Simpson's rule is exact.
    midpoint_density = (density[:-1] + density[1:]) / 2
    midpoint_response = (response[:-1] + response[1:]) / 2
    product = density * response
    samples = product[:-1] + 4 * midpoint_density * midpoint_response + product[1:]
    detected = (np.diff(wavelengths) * samples / 6).sum()
    return 7 * detected / np.trapezoid(density, wavelengths)


@pytest.mark.parametrize("polarised", [False, True])
def test_single_optical_sample_is_a_constant_field_approximation(polarised):
    pupil = dl.Optic(jones=ChromaticJones()) if polarised else dl.Optic()
    model = make_imager(1, pupil)
    optical = model.wavefront()
    assert optical.wavelength.shape == (1,)
    if polarised:
        assert optical.jones.shape == (1, 1, 2, 2)

    # Propagate the central wavelength independently, then integrate only the
    # fine source/filter/QE response. No wavelength-dependent PSF is invented.
    entrance = dl.Wavefront(6e-7, model.pupil_grid)
    central = dl.PropagatorLayer(model.image_grid)(pupil(entrance)).intensity
    expected = central * detected_counts()
    output = assert_jittable(lambda system: system.intensity().data, model, rtol=3e-5)
    assert np.allclose(output, expected, rtol=3e-5, atol=2e-7)


def test_fine_curves_integrate_onto_coarse_optical_psfs():
    pupil = dl.Optic(opd=4e-8, jones=ChromaticJones())
    model = make_imager(3, pupil)
    psf = model.psf()
    assert psf.wavelengths.shape == (3,)
    weights = model.photometry().value[0]
    expected = (psf.array[0] * weights[:, None, None]).sum(0)
    output = assert_jittable(lambda system: system.intensity().data, model, rtol=3e-5)
    assert np.allclose(output, expected, rtol=3e-5, atol=2e-7)
    # Optical resolution does not change integrated photometry on a fixed fine mesh.
    for n in (1, 5, 9):
        band = dlu.BandPass(6e-7, 2e-7, n=n)
        assert np.allclose(model.photometry(bandpass=band).integrate(), weights.sum())
