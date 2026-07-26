"""Tests for dLux.parametric.polynomials."""

import jax.numpy as np
import pytest

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def coordinates():
    return dlu.pixel_coords(10, 2.0)


def test_general_polynomial_contract():
    variables = np.linspace(-1, 1, 8)
    polynomial = dl.Polynomial(2, [1.0, 2.0, 3.0])

    output = assert_jittable(
        lambda value: value.evaluate(variables=variables),
        polynomial,
    )
    assert np.allclose(output, 1 + 2 * variables + 3 * variables**2)
    assert_differentiable(
        lambda coefficients: polynomial.set(coefficients=coefficients).evaluate(
            variables=variables
        ),
        polynomial.coefficients,
    )


def test_multivariate_polynomial(coordinates):
    powers = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    polynomial = dl.Polynomial(1, [1.0, 2.0, 3.0, 4.0], powers=powers)
    x, y = coordinates

    output = polynomial.evaluate(variables=coordinates)
    assert np.allclose(output, 1 + 2 * x + 3 * y + 4 * x * y)


@pytest.mark.parametrize("coordinate_source", ["array", "spec"])
def test_explicit_polynomial_contract(coordinate_source, coordinates):
    if coordinate_source == "spec":
        coordinate_source = dl.GridSpec(n=10, d=0.2, unit="m").broadcast(2)
    else:
        coordinate_source = coordinates
    polynomial = dl.ExplicitPolynomial(
        coordinate_source,
        1,
        coefficients=[1.0, 2.0, 3.0],
        ndim=2,
    )

    output = assert_jittable(lambda value: value.evaluate(), polynomial)
    recovered = polynomial.solve_basis(output)
    assert output.shape == coordinates.shape[-2:]
    assert np.allclose(recovered, polynomial.coefficients, atol=1e-5)


@pytest.mark.parametrize("nsides", [0, 6])
def test_dynamic_zernike_contract(nsides, coordinates):
    zernike = dl.DynamicZernike(4)
    output = assert_jittable(
        lambda value: value.calculate(coordinates, nsides=nsides),
        zernike,
    )
    assert output.shape == coordinates.shape[-2:]


@pytest.mark.parametrize(
    "make_basis",
    [
        lambda coordinates: dl.ZernikeBasis(
            coordinates,
            radial_orders=[2],
            coefficients=np.linspace(-0.2, 0.2, 3),
        ),
        lambda coordinates: dl.DynamicZernikeBasis(
            radial_orders=[2],
            coefficients=np.linspace(-0.2, 0.2, 3),
        ),
        lambda coordinates: dl.DynamicZernikeBasis(
            js=[4, 5],
            coefficients=np.asarray([0.1, -0.1]),
            nsides=6,
        ),
        lambda coordinates: dl.CoordinatePolynomial(
            2,
            coefficients=np.linspace(-0.2, 0.2, 6),
        ),
    ],
)
def test_polynomial_basis_contract(make_basis, coordinates):
    basis = make_basis(coordinates)
    context = {} if isinstance(basis, dl.ZernikeBasis) else {"coordinates": coordinates}

    output = assert_jittable(lambda value: value.evaluate(**context), basis)
    assert output.shape == coordinates.shape[-2:]
    assert_differentiable(
        lambda coefficients: basis.set(coefficients=coefficients).evaluate(**context),
        basis.coefficients,
    )


@pytest.mark.parametrize(
    "basis",
    [
        dl.DynamicZernikeBasis(radial_orders=[2]),
        dl.CoordinatePolynomial(2),
    ],
)
def test_dynamic_basis_context(basis, coordinates, make_wavefront):
    calculated = assert_jittable(
        lambda value: value.calculate_basis(coordinates=coordinates),
        basis,
    )
    assert calculated.shape[-2:] == coordinates.shape[-2:]

    wavefront = make_wavefront()
    assert_jittable(lambda value: value.evaluate(wavefront=wavefront), basis)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda coordinates: dl.DynamicZernike(0),
        lambda coordinates: dl.ZernikeBasis(coordinates),
        lambda coordinates: dl.ZernikeBasis(coordinates, js=[]),
        lambda coordinates: dl.DynamicZernikeBasis(js=[1], nsides=2),
        lambda coordinates: dl.DynamicZernikeBasis(js=[1], diameter=0.0),
        lambda coordinates: dl.CoordinatePolynomial(-1),
        lambda coordinates: dl.CoordinatePolynomial(2, np.ones(5)),
        lambda coordinates: dl.Polynomial(0, []),
        lambda coordinates: dl.Polynomial(1, [1, 2], powers=np.ones((2, 3))),
        lambda coordinates: dl.Polynomial(1, [1, 2], powers=[0, -1]),
    ],
)
def test_validation(constructor, coordinates):
    with pytest.raises((TypeError, ValueError)):
        constructor(coordinates)


@pytest.mark.parametrize(
    "basis",
    [dl.DynamicZernikeBasis(js=[1]), dl.CoordinatePolynomial(1)],
)
def test_coordinate_context_validation(basis):
    with pytest.raises(ValueError, match="wavefront or coordinates"):
        basis.evaluate()
