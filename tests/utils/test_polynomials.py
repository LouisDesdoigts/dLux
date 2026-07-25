"""Tests for dLux.utils.polynomials."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    ("degree", "ndim", "nterms"), [(2, 1, 3), (2, 2, 6), (2, 3, 10)]
)
def test_polynomial_basis_contract(degree, ndim, nterms):
    powers = dlu.polynomial_powers(degree, ndim)
    variables = np.ones((ndim, 4, 3))
    basis = assert_jittable(dlu.polynomial_basis, variables, powers)
    assert powers.shape == (ndim, nterms)
    assert basis.shape == (nterms, 4, 3)


@pytest.mark.parametrize(("index", "orders"), [(1, (0, 0)), (2, (1, 1)), (4, (2, 0))])
def test_noll_indices(index, orders):
    assert dlu.noll_indices(index) == orders


def test_zernike_metadata():
    assert dlu.radial_orders_to_indices([0, 2]) == [1, 4, 5, 6]
    assert dlu.zernike_name(4) == "Defocus"
    assert dlu.zernike_name(100) == "Zernike 100"
    with pytest.raises(ValueError):
        dlu.zernike_factors(0)


@pytest.mark.parametrize("index", [1, 2, 4, 7])
def test_zernike_fast_matches_direct(index):
    coordinates = dlu.pixel_coords(16, diameter=2)
    n, m = dlu.noll_indices(index)
    c, k = dlu.zernike_factors(index)
    expected = dlu.zernike(index, coordinates)
    output = assert_jittable(dlu.zernike_fast, n, m, c, k, coordinates)
    assert np.allclose(output, expected)


def test_polynomial_family_bases():
    coordinates = dlu.pixel_coords(16, diameter=2)
    zernikes = dlu.zernike_basis((1, 2, 4), coordinates)
    polikes = dlu.polike_basis(6, (1, 2, 4), coordinates)
    assert zernikes.shape == polikes.shape == (3, 16, 16)
    assert np.all(np.isfinite(zernikes))
    assert np.all(np.isfinite(polikes))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: dlu.polynomial_powers(-1, 2),
        lambda: dlu.polynomial_powers(2, 0),
        lambda: dlu.radial_orders_to_indices([-1]),
    ],
)
def test_validation(operation):
    with pytest.raises(ValueError):
        operation()
