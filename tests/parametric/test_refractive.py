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
            "indices",
        ),
    ],
)
def test_refractive_index_contract(index, parameter, make_wavefront):
    wavefront = make_wavefront(wavelength=np.asarray([0.9e-6, 1.0e-6, 1.1e-6]))

    output = assert_jittable(
        lambda value: value.evaluate(wavefront=wavefront),
        index,
    )
    assert output.shape == wavefront.wavelength.shape
    assert_differentiable(
        lambda value: index.set(parameter, value).evaluate(wavefront=wavefront),
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
