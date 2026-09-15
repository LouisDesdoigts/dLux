"""Tests for dLux.parametric.refractive."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.mark.parametrize(
    ("index", "parameter"),
    [
        (dl.CauchyIndex([1.5, 0.01, 0.001]), "coeffs"),
        (dl.PolynomialIndex([1.5, 0.01, -0.001]), "coeffs"),
        (
            dl.InterpolatedIndex(
                [0.8e-6, 1.0e-6, 1.2e-6],
                [1.52, 1.50, 1.49],
            ),
            "values",
        ),
    ],
)
def test_refractive_index_contract(index, parameter, make_wavefront):
    wavefront = make_wavefront(wavelength=np.asarray([0.9e-6, 1.0e-6, 1.1e-6]))

    output = assert_jittable(
        lambda value: value.resolve(wavelengths=wavefront.wavelength),
        index,
    )
    assert output.shape == wavefront.wavelength.shape
    assert_differentiable(
        lambda value: index.set(parameter, value).resolve(
            wavelengths=wavefront.wavelength
        ),
        getattr(index, parameter),
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.CauchyIndex([]),
        lambda: dl.CauchyIndex([[1.5]]),
        lambda: dl.PolynomialIndex([1.5], scale=0.0),
        lambda: dl.InterpolatedIndex([1e-6], [1.5]),
        lambda: dl.InterpolatedIndex([1e-6, 2e-6], [1.5]),
        lambda: dl.InterpolatedIndex([2e-6, 1e-6], [1.5, 1.6]),
    ],
)
def test_validation(constructor):
    with pytest.raises(ValueError):
        constructor()


@pytest.mark.parametrize(
    "index",
    [
        dl.PolynomialIndex([1.5, 0.01]),
        dl.CauchyIndex([1.5, 0.01]),
        dl.InterpolatedIndex([500e-9, 700e-9], [1.51, 1.50]),
    ],
)
def test_refractive_models_require_explicit_wavelengths(index, make_wavefront):
    wavefront = make_wavefront(wavelength=600e-9)
    assert isinstance(index.resolve(wavefront=wavefront), type(index))
    with pytest.raises(TypeError, match="wavelengths"):
        index.evaluate(wavefront=wavefront)
    with pytest.raises(ValueError, match="explicitly"):
        index.evaluate(wavelengths=None, wavefront=wavefront)
    expected = index.evaluate(wavelengths=600e-9)
    result = assert_jittable(
        lambda model: model.resolve(wavelengths=600e-9, wavefront=wavefront), index
    )
    assert np.allclose(result, expected)
