"""Tests for deprecated public compatibility contracts."""

import jax.numpy as np
import pytest

import dLux as dl


def test_coefficients_constructor_alias():
    basis = np.ones((2, 3, 3))

    with pytest.warns(DeprecationWarning) as record:
        parametric = dl.Basis(basis, coefficients=[1.0, 2.0])

    message = str(record[0].message)
    assert "removed in dLux 0.16.2" in message
    assert "Class(coefficients=value)` -> `Class(coeffs=value)" in message
    assert np.allclose(parametric.coeffs, np.array([1.0, 2.0]))


def test_coefficients_attribute_alias():
    basis = dl.Basis(np.ones((2, 3, 3)), coeffs=[1.0, 2.0])

    with pytest.warns(DeprecationWarning) as record:
        coeffs = basis.coefficients

    message = str(record[0].message)
    assert "removed in dLux 0.16.2" in message
    assert "basis.coefficients` -> `basis.coeffs" in message
    assert np.allclose(coeffs, basis.coeffs)
