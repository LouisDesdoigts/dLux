"""Tests for numerical helpers and retained array calculations."""

import jax.numpy as np
import pytest
from jax import random

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_gaussian_contract():
    output = assert_jittable(
        dlu.gaussian,
        mean=np.asarray((0.0, 0.0)),
        std=np.asarray((1.0, 2.0)),
        npixels=(8, 6),
    )

    # Released Gaussian sampling uses xy indexing: the array ends in (y, x).
    assert output.shape == (6, 8)
    assert np.isclose(output.sum(), 1.0)


def test_gaussian_zero_width_and_axis_extents():
    output = assert_jittable(
        dlu.gaussian,
        mean=np.zeros(2),
        std=np.asarray((0.0, 1.0)),
        npixels=(9, 7),
        extent=np.asarray((2.0, 3.0)),
    )

    assert np.isclose(output.sum(), 1.0)
    assert output.shape == (7, 9)
    # Zero x width confines the profile to one column, across all sampled y values.
    assert np.count_nonzero(output.sum(0)) == 1


def test_correlated_gaussian_profile_contract():
    profile = dlu.Profile(
        dlu.CorrelatedGaussian(stds=[1.0, 2.0], corr=0.3, mean=[0.2, -0.1])
    )
    grid = dlu.Grid(n=(6, 8), d=0.2)
    output = assert_jittable(lambda profile: profile.sample(grid), profile)
    assert output.shape == (8, 6)
    assert np.isclose(output.sum(), 1.0)


def test_basis_roundtrip():
    basis = random.normal(random.PRNGKey(0), (2, 3, 5, 4))
    coeffs = np.arange(6.0).reshape((2, 3))
    array = assert_jittable(dlu.eval_basis, basis, coeffs)
    recovered = dlu.solve_basis(array, basis)

    assert array.shape == (5, 4)
    assert np.allclose(recovered, coeffs, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [float, complex])
def test_zero_basis_has_zero_minimum_norm_coefficients(dtype):
    basis = np.zeros((2, 3), dtype=dtype)
    values = np.ones((4, 3), dtype=dtype)
    coefficients = assert_jittable(
        lambda data: dlu.solve_basis(data, basis, axes=1), values
    )

    assert np.array_equal(coefficients, np.zeros((4, 2)))


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [(dlu.factorial, 5.0, 120.0), (dlu.triangular_number, 4, 10)],
)
def test_integer_sequences(function, value, expected):
    assert np.isclose(function(value), expected)


def test_safe_division():
    output = assert_jittable(
        dlu.nandiv, np.asarray((1.0, 2.0)), np.asarray((1.0, 0.0)), 0.0
    )
    assert np.allclose(output, np.asarray((1.0, 0.0)))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.eval_basis(np.ones((2, 3, 4)), np.ones((3, 2))),
        lambda: dlu.solve_basis(np.ones((3, 4)), np.ones((2, 2, 6))),
        lambda: dlu.CorrelatedGaussian(stds=[1.0, 2.0], mean=np.zeros((2, 1))),
        lambda: dlu.CorrelatedGaussian(stds=[1.0, 2.0, 3.0]),
    ],
)
def test_validation(operation):
    with pytest.raises(ValueError):
        operation()


@pytest.mark.parametrize("cartesian", [True, False])
@pytest.mark.parametrize("axis", [0, -1, 1])
def test_complex_channels_roundtrip_and_axis_order(cartesian, axis):
    values = np.array([[1 + 2j, -2 + 1j], [3 - 1j, 2 + 4j]])
    channels, restore = dlu.from_complex(values, cartesian, axis=axis)
    expected = (
        (values.real, values.imag) if cartesian else (np.abs(values), np.angle(values))
    )
    assert np.allclose(channels, np.stack(expected, axis=axis))
    assert np.allclose(restore(channels), values, atol=1e-6)
    assert np.allclose(
        dlu.to_complex(channels, cartesian, axis=axis), values, atol=1e-6
    )
