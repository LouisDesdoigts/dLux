"""Integrated photons are allocated before spectral response and optical losses."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_photon_flux_preserves_shape_and_total():
    wavelengths = np.asarray([5e-7, 7e-7])
    flux = dlu.Flux(2)
    weights = assert_jittable(
        lambda f: f.photons(wavelengths, np.array([1.0, 3.0])), flux
    )
    assert np.allclose(weights, np.asarray([0.5, 1.5]))
    derivative = jax.jit(
        jax.grad(
            lambda value: flux.set(value=value).photons(wavelengths, np.ones(2)).sum()
        )
    )(flux.value)
    assert np.allclose(derivative, 1.0)


@pytest.mark.parametrize("unit", ["photons/s", "photons/s/m^2", "W/m^2", "unknown"])
def test_flux_rejects_units_other_than_photons(unit):
    with pytest.raises(ValueError, match="Only flux unit 'photons'"):
        dlu.Flux(1.0, unit=unit)


def test_blackbody_flux_context_and_losses_share_the_spectrum_contract():
    band = dlu.BandPass(600e-9, 200e-9, n=5)
    flux = dlu.Flux(zdx.StateRef("brightness"))
    spectrum = dlu.Blackbody(5500, bandpass=band, flux=flux)
    model = dlu.Photometry(spectrum, filter=0.5, qe=0.2)
    state = zdx.State(brightness=10.0)
    actual = assert_jittable(lambda m: m.integrate(state=state), model)
    assert np.allclose(actual, 1.0)
    assert np.allclose(model.integrate(state=state.set(brightness=20.0)), 2.0)


def test_source_stores_spectrum_brightness_and_imager_accepts_flux():
    spectrum = dlu.Blackbody(5500, flux=dlu.Flux(2.0))
    source = dl.PointSource(spectrum)
    assert "flux" not in vars(source)
    assert source.spectrum is spectrum
    imager = dl.Imager(
        dlu.Grid(4, diam=1.0, unit="m"),
        dlu.Grid(4, d=1e-6, unit="rad"),
        source=source,
        bandpass=dlu.BandPass(600e-9, 200e-9, n=3),
        filter=0.5,
    )
    weights = assert_jittable(lambda m: m.photometry().value, imager)
    assert np.allclose(weights.sum(), 1.0)
    assert np.allclose(dl.PointSource(flux=4.0).spectrum.flux, 4.0)


def test_changing_spectrum_flux_switches_absolute_density_to_photon_counts():
    band = dlu.BandPass(600e-9, 200e-9, n=3)
    spectrum = dlu.Spectrum(2.0, bandpass=band, flux=None)
    assert np.allclose(spectrum.integrate() / 1e-9, 400.0)
    assert np.allclose(spectrum.set(flux=np.asarray(3.0)).integrate(), 3.0)


def test_blackbody_rejects_nonphysical_concrete_temperature():
    with pytest.raises(ValueError, match="temperature"):
        dlu.Blackbody(0.0)
