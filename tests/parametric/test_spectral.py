"""Tests for parametric spectral-weight models."""

import jax
import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


def test_spectral_polynomial_contract():
    wavelengths = np.linspace(600e-9, 700e-9, 5)
    polynomial = dl.SpectralPolynomial(degree=1, coeffs=[0.5])

    weights = assert_jittable(
        lambda value: value.evaluate(wavelengths=wavelengths), polynomial
    )
    expected = np.linspace(0.75, 1.25, 5)
    expected /= expected.sum()

    assert np.allclose(weights, expected)
    assert np.isclose(weights.sum(), 1.0)
    assert_differentiable(
        lambda coeffs: polynomial.set(coeffs=coeffs).evaluate(wavelengths=wavelengths),
        polynomial.coeffs,
    )

    raw = dl.SpectralPolynomial(
        degrees=1, coeffs=[0.5], normalise=False
    ).evaluate(wavelengths=wavelengths)
    assert np.allclose(raw, np.linspace(0.75, 1.25, 5))


def test_vectorized_spectral_models():
    wavelengths = np.linspace(0.8e-6, 1.2e-6, 7)
    polynomial = dl.SpectralPolynomial(
        degree=1, coeffs=[[0.2], [-0.2]]
    )
    basis = dl.SpectralBasis(
        np.stack([np.ones(7), np.linspace(-0.2, 0.2, 7)]),
        coeffs=[[1.0, 0.5], [1.0, -0.5]],
        shape=(2,),
    )
    blackbody = dl.Blackbody([4000.0, 8000.0])

    for model in (polynomial, basis, blackbody):
        weights = assert_jittable(
            lambda value: value.evaluate(wavelengths=wavelengths), model
        )
        assert weights.shape == (2, 7)
        assert np.all(weights > 0)
        assert np.allclose(weights.sum(-1), np.ones(2))

    scalar_blackbodies = np.stack(
        [
            dl.Blackbody(temperature).evaluate(wavelengths=wavelengths)
            for temperature in (4000.0, 8000.0)
        ]
    )
    assert np.allclose(blackbody.evaluate(wavelengths=wavelengths), scalar_blackbodies)


def test_blackbody_temperature_derivatives():
    wavelengths = np.linspace(0.5e-6, 1.0e-6, 8)
    blackbody = dl.Blackbody(5800.0)

    assert_differentiable(
        lambda temperature: blackbody.set(temperature=temperature).evaluate(
            wavelengths=wavelengths
        ),
        blackbody.temperature,
    )
    evaluate = lambda temperature: blackbody.set(
        temperature=temperature
    ).evaluate(wavelengths=wavelengths)
    jacobian = jax.jacrev(evaluate)(blackbody.temperature)
    hessian = jax.jacrev(jax.jacrev(evaluate))(blackbody.temperature)
    assert np.all(np.isfinite(jacobian))
    assert np.all(np.isfinite(hessian))
    assert np.isclose(jacobian.sum(), 0.0, atol=1e-8)
    assert np.isclose(hessian.sum(), 0.0, atol=1e-10)

    raw = dl.Blackbody(5800.0, normalise=False).evaluate(wavelengths=wavelengths)
    assert not np.isclose(raw.sum(), 1.0)


def test_spectral_source_unit_invariance():
    nanometres = dl.Source(
        np.linspace(600, 700, 5),
        weights=dl.SpectralPolynomial(degree=1, coeffs=[0.5]),
        units={"wavelengths": "nm"},
    )
    metres = dl.Source(
        np.linspace(600e-9, 700e-9, 5),
        weights=dl.SpectralPolynomial(degree=1, coeffs=[0.5]),
    )

    assert np.allclose(nanometres.params()["weights"], metres.params()["weights"])
    assert dl.Source(650e-9).params()["wavelengths"].shape == (1,)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.SpectralPolynomial(degrees=0, coeffs=[0.1]),
        lambda: dl.Blackbody(0.0),
    ],
)
def test_spectral_validation(constructor):
    with pytest.raises(ValueError):
        constructor()
