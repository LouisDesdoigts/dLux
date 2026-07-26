"""Tests for dLux.sources."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


def test_array_spectrum_contract():
    spectrum = dl.Spectrum([0.9e-6, 1.1e-6], [0.25, 0.75])

    wavelengths, weights = assert_jittable(
        lambda value: value.spectrum_params(), spectrum
    )
    assert np.allclose(wavelengths, spectrum.wavelengths)
    assert np.allclose(weights, spectrum.weights)


def test_parametric_spectrum_contract():
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 5)
    weights = dl.Polynomial(1, [1.0, 2e5])
    spectrum = dl.Spectrum(wavelengths, weights)

    resolved_wavelengths, resolved_weights = assert_jittable(
        lambda value: value.spectrum_params(), spectrum
    )
    assert np.allclose(resolved_wavelengths, wavelengths)
    assert np.allclose(resolved_weights, 1 + 2e5 * wavelengths)
    assert_differentiable(
        lambda coefficients: spectrum.set(
            "weights.coefficients", coefficients
        ).spectrum_params()[1],
        weights.coefficients,
    )
    resolved = spectrum.resolve(variables=wavelengths)
    assert isinstance(resolved, dl.Spectrum)
    assert not isinstance(resolved.weights, dl.Parametric)


def test_basis_weight_spectrum():
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 5)
    basis = np.stack([np.ones(5), np.linspace(-1, 1, 5)])
    spectrum = dl.Spectrum(
        wavelengths, dl.ExplicitBasis(basis, coefficients=[1.0, 0.2])
    )

    _, weights = spectrum.spectrum_params()
    assert weights.shape == wavelengths.shape


def test_spectrum_and_source_parameters():
    spectrum = dl.Spectrum([900, 1100], units={"wavelengths": "nm"})
    source = dl.Source(
        [900, 1100],
        position=[1, -2],
        flux=2,
        units={"wavelengths": "nm", "position": "arcsec", "flux": "kphoton"},
    )

    wavelengths, weights = spectrum.spectrum_params()
    parameters = source.params()
    assert np.allclose(wavelengths, np.array([0.9e-6, 1.1e-6]))
    assert np.allclose(weights, np.ones(2))
    assert np.allclose(parameters["position"], np.array([1, -2]) * dlu.arcsec2rad(1))
    assert np.allclose(parameters["flux"], 2000)
    assert parameters["distribution"] is None


def test_default_source():
    source = dl.Source([1e-6])
    parameters = source.params()

    assert np.allclose(parameters["position"], np.zeros(2))
    assert np.allclose(parameters["flux"], 1)
    assert parameters["distribution"] is None


def test_binary_source_parameters():
    source = dl.BinarySource(
        [1e-6],
        centre=[0.1, -0.2],
        separation=0.4,
        position_angle=0.3,
        contrast=3.0,
        flux=2.0,
    )

    parameters = source.params()
    positions = parameters["position"]
    fluxes = parameters["flux"]
    assert positions.shape == (2, 2)
    assert fluxes.shape == (2,)
    assert np.allclose(fluxes.mean(), 2)
    assert np.allclose(fluxes[0] / fluxes[1], 3)
    assert parameters["distribution"] is None

    assert_differentiable(
        lambda separation: source.set("separation", separation).params()["position"],
        source.separation,
    )


def test_parametric_distribution():
    distribution = dl.ExplicitBasis(np.ones((2, 3, 3)), coefficients=[1.0, 2.0])
    source = dl.BinarySource([1e-6], distribution=distribution)

    assert source.params()["distribution"].shape == (3, 3)


def test_log_flux_units():
    source = dl.Source(
        [1e-6],
        flux=np.log(1000),
        distribution=np.log(np.full((3, 3), 2.0)),
        units={"flux": "log_photon", "distribution": "log"},
    )
    assert np.allclose(source.params()["flux"], 1000)
    assert np.allclose(source.params()["distribution"], 2)


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dl.Spectrum([[1e-6]]).spectrum_params(),
        lambda: dl.Spectrum([1e-6], [[1.0, 1.0], [1.0, 1.0]]).spectrum_params(),
        lambda: dl.Spectrum(dl.Polynomial(0, [1])),
        lambda: dl.Spectrum([1, 2], np.ones((2, 2, 2))).spectrum_params(),
        lambda: dl.Spectrum([1e-6], units={"unknown": "m"}),
        lambda: dl.Source([1e-6], position=np.zeros(3)).params(),
        lambda: dl.Source([1e-6], flux=np.ones(2)).params(),
        lambda: dl.Source([1e-6], distribution=np.ones(3)).params(),
        lambda: dl.BinarySource([1e-6], centre=np.zeros(3)).params(),
    ],
)
def test_source_validation(operation):
    with pytest.raises((TypeError, ValueError)):
        operation()
