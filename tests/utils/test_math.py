"""Tests for dLux.utils.math."""

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

    assert output.shape == (8, 6)
    assert np.isclose(output.sum(), 1.0)


def test_basis_roundtrip():
    basis = random.normal(random.PRNGKey(0), (2, 3, 5, 4))
    coeffs = np.arange(6.0).reshape((2, 3))
    array = assert_jittable(dlu.eval_basis, basis, coeffs)
    recovered = dlu.solve_basis(array, basis)

    assert array.shape == (5, 4)
    assert np.allclose(recovered, coeffs, rtol=1e-5, atol=1e-5)


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
        lambda: dlu.mv_gaussian(np.zeros(2), np.eye(2)),
    ],
)
def test_validation(operation):
    with pytest.raises((ValueError, NotImplementedError)):
        operation()
