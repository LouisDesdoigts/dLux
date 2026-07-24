"""Tests for dLux.spectra."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


def test_array_spectrum_contract():
    spectrum = dl.Spectrum([0.9e-6, 1.1e-6], [0.25, 0.75])

    wavelengths, weights = assert_jittable(lambda value: value.params(), spectrum)
    assert np.allclose(wavelengths, spectrum.wavelengths)
    assert np.allclose(weights, spectrum.weights)


def test_parametric_spectrum_contract():
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 5)
    weights = dl.Polynomial([1.0, 2e5])
    spectrum = dl.Spectrum(wavelengths, weights)

    resolved_wavelengths, resolved_weights = assert_jittable(
        lambda value: value.params(),
        spectrum,
    )
    assert np.allclose(resolved_wavelengths, wavelengths)
    assert np.allclose(resolved_weights, 1 + 2e5 * wavelengths)
    assert_differentiable(
        lambda coefficients: spectrum.set(
            "weights.coefficients",
            coefficients,
        ).params()[1],
        weights.coefficients,
    )


def test_basis_weight_spectrum():
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 5)
    basis = np.stack([np.ones(5), np.linspace(-1, 1, 5)])
    spectrum = dl.Spectrum(
        wavelengths,
        dl.ExplicitBasis(basis, coefficients=[1.0, 0.2]),
    )

    _, weights = spectrum.params()
    assert weights.shape == wavelengths.shape


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Spectrum(np.ones((2, 2))),
        lambda: dl.Spectrum([1, 2], [1]),
        lambda: dl.Spectrum(dl.Polynomial([1])),
        lambda: dl.Spectrum([1, 2], np.ones((2, 2, 2))),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
