"""Tests for dLux.utils.parametrics.polynomials."""

import jax.numpy as np
import jax
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
        lambda value: value.resolve(coordinates=variables), polynomial
    )
    assert np.allclose(output, 1 + 2 * variables + 3 * variables**2)
    assert np.allclose(
        polynomial.solve(output, coordinates=variables),
        np.asarray([1, 2, 3]),
    )
    assert_differentiable(
        lambda coeffs: polynomial.set(coeffs=coeffs).evaluate(coordinates=variables),
        polynomial.coeffs,
    )


def test_selected_polynomial_degrees():
    variables = np.linspace(-1, 1, 8)

    linear = dl.Polynomial(degrees=1, coeffs=[2.0])
    selected = dl.Polynomial(degrees=[0, 2], coeffs=[1.0, 3.0])

    assert np.allclose(linear.resolve(coordinates=variables), 2 * variables)
    assert np.allclose(selected.resolve(coordinates=variables), 1 + 3 * variables**2)


def test_multivariate_polynomial(coordinates):
    powers = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    polynomial = dl.Polynomial(coeffs=[1.0, 2.0, 3.0, 4.0], powers=powers)
    x, y = coordinates

    output = polynomial.resolve(coordinates=coordinates)
    assert np.allclose(output, 1 + 2 * x + 3 * y + 4 * x * y)


@pytest.mark.parametrize("coordinate_source", ["array", "grid"])
def test_explicit_polynomial_contract(coordinate_source, coordinates):
    if coordinate_source == "grid":
        coordinate_source = dl.GridSpec(n=10, d=0.2, unit="m").broadcast(2).coordinates
    else:
        coordinate_source = coordinates
    terms = dlu.polynomial_basis(coordinate_source, dlu.polynomial_powers(1, 2))
    polynomial = dl.Basis(terms, coeffs=[1.0, 2.0, 3.0])

    output = assert_jittable(lambda value: value.evaluate(), polynomial)
    recovered = polynomial.solve(output)
    assert output.shape == coordinates.shape[-2:]
    assert np.allclose(recovered, polynomial.coeffs, atol=1e-5)


def test_explicit_polynomial_samples_promoted_grid():
    grid = dl.GridSpec(n=8, d=0.25, unit="m")
    terms = dlu.polynomial_basis(
        grid.broadcast(2).coordinates, dlu.polynomial_powers(1, 2)
    )
    polynomial = dl.Basis(terms, coeffs=[1.0, 2.0, 3.0])

    output = assert_jittable(lambda value: value.evaluate(), polynomial)

    assert output.shape == (8, 8)
    assert np.allclose(polynomial.solve(output), polynomial.coeffs, atol=1e-5)


@pytest.mark.parametrize("nsides", [0, 6])
def test_dynamic_zernike_contract(nsides, coordinates):
    zernike = dl.DynamicZernike(4)
    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates, nsides=nsides), zernike
    )
    assert output.shape == coordinates.shape[-2:]


def test_zernike_order_selection(coordinates):
    explicit = dl.ZernikeBasis(coordinates, order=3)
    dynamic = dl.DynamicZernikeBasis(order=3)

    assert explicit.coeffs.shape == (10,)
    assert dynamic.coeffs.shape == (10,)

    with pytest.raises(ValueError, match="exactly one"):
        dl.DynamicZernikeBasis(order=3, orders=[2, 3])


@pytest.mark.parametrize("npix", [(6, 4), 6])
def test_zernike_basis_accepts_grid_or_its_coordinate_array(npix):
    grid = dl.GridSpec(n=npix, d=0.2)
    options = {"js": [1, 4], "coeffs": [0.2, -0.1], "diameter": 1.2}
    sampled = dl.ZernikeBasis(grid, **options)
    explicit = dl.ZernikeBasis(grid.broadcast(2).coordinates, **options)

    output = assert_jittable(lambda model: model.resolve(), sampled)

    assert output.shape == grid.broadcast(2).shape
    assert np.allclose(output, explicit.resolve())


@pytest.mark.parametrize(
    "make_basis",
    [
        lambda coordinates: dl.ZernikeBasis(
            coordinates, orders=[2], coeffs=np.linspace(-0.2, 0.2, 3)
        ),
        lambda coordinates: dl.DynamicZernikeBasis(
            orders=[2], coeffs=np.linspace(-0.2, 0.2, 3)
        ),
        lambda coordinates: dl.DynamicZernikeBasis(
            js=[4, 5], coeffs=np.asarray([0.1, -0.1]), nsides=6
        ),
        lambda coordinates: dl.Polynomial(2, coeffs=np.linspace(-0.2, 0.2, 6), ndim=2),
    ],
)
def test_polynomial_basis_contract(make_basis, coordinates):
    basis = make_basis(coordinates)
    context = {} if isinstance(basis, dl.ZernikeBasis) else {"coordinates": coordinates}

    output = assert_jittable(lambda value: value.evaluate(**context), basis)
    assert output.shape == coordinates.shape[-2:]
    assert_differentiable(
        lambda coeffs: basis.set(coeffs=coeffs).evaluate(**context),
        basis.coeffs,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "basis", [dl.DynamicZernikeBasis(orders=[2]), dl.Polynomial(2, ndim=2)]
)
def test_dynamic_basis_context(basis, coordinates, make_wavefront):
    calculated = assert_jittable(
        lambda value: value.calculate_basis(coordinates=coordinates), basis
    )
    assert calculated.shape[-2:] == coordinates.shape[-2:]

    wavefront = make_wavefront()
    assert_jittable(
        lambda value: value.resolve(coordinates=wavefront.coordinates), basis
    )


def test_coordinate_polynomial_solution(coordinates):
    polynomial = dl.Polynomial(1, coeffs=np.asarray([1.0, 2.0, 3.0]), ndim=2)
    value = polynomial.evaluate(coordinates=coordinates)
    recovered = polynomial.solve(value, coordinates=coordinates)

    assert np.allclose(recovered, polynomial.coeffs, atol=1e-5)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda coordinates: dl.DynamicZernike(0),
        lambda coordinates: dl.ZernikeBasis(coordinates),
        lambda coordinates: dl.ZernikeBasis(coordinates, js=[]),
        lambda coordinates: dl.DynamicZernikeBasis(js=[1], nsides=2),
        lambda coordinates: dl.DynamicZernikeBasis(js=[1], diameter=0.0),
        lambda coordinates: dl.Polynomial(-1, ndim=2),
        lambda coordinates: dl.Polynomial(2, np.ones(5), ndim=2),
        lambda coordinates: dl.Polynomial(0, []),
        lambda coordinates: dl.Polynomial(1, [1, 2], powers=np.ones((2, 3))),
        lambda coordinates: dl.Polynomial(1, [1, 2], powers=[0, -1]),
    ],
)
def test_validation(constructor, coordinates):
    with pytest.raises((TypeError, ValueError)):
        constructor(coordinates)


@pytest.mark.parametrize(
    "basis", [dl.DynamicZernikeBasis(js=[1]), dl.Polynomial(1, ndim=2)]
)
def test_coordinate_context_validation(basis):
    assert isinstance(basis.resolve(), type(basis))
    with pytest.raises((TypeError, ValueError), match="coordinates"):
        basis.evaluate()


def test_polynomial_integer_powers_have_finite_zero_derivatives():
    polynomial = dl.Polynomial(3, [1.0, 2.0, 3.0, 4.0])
    evaluate = lambda coordinate: polynomial.resolve(coordinates=coordinate)
    assert np.allclose(jax.grad(evaluate)(0.0), 2.0)
    assert np.allclose(jax.hessian(evaluate)(0.0), 6.0)
    assert np.allclose(jax.grad(jax.hessian(evaluate))(0.0), 24.0)


def test_polynomial_paired_coordinate_batches_and_solve(coordinates):
    batch = np.stack((coordinates, coordinates * 0.8))
    coeffs = np.asarray([[1.0, 2.0, 3.0], [4.0, -2.0, 1.0]])
    polynomial = dl.Polynomial(1, coeffs, ndim=2)
    result = assert_jittable(lambda model: model.resolve(coordinates=batch), polynomial)
    expected = np.stack(
        [
            dl.Polynomial(1, coeff, ndim=2).resolve(coordinates=coord)
            for coeff, coord in zip(coeffs, batch)
        ]
    )
    assert np.allclose(result, expected)
    assert np.allclose(polynomial.solve(result, coordinates=batch), coeffs, atol=2e-6)


def test_dynamic_zernike_context_is_consistent_without_wavefront_fallback(
    make_wavefront,
):
    grid = dl.GridSpec(n=(12, 12), diam=1.0, unit="m")
    wavefront = make_wavefront(grid=grid)
    basis = dl.DynamicZernikeBasis(js=[4, 5], coeffs=[0.1, -0.2])
    assert isinstance(basis.resolve(wavefront=wavefront), dl.DynamicZernikeBasis)
    with pytest.raises(ValueError, match="grid or coordinates"):
        basis.calculate_basis(wavefront=wavefront)
    default = basis.evaluate(grid=grid)
    assert np.allclose(default, basis.evaluate(grid=grid, diameter=2.0))
    assert np.allclose(default, basis.evaluate(grid=grid, wavefront=wavefront))
    stored = basis.set(diameter=0.8)
    transform = dl.Affine(translation=[0.05, -0.02])
    context = dict(grid=grid, transform=transform, diameter=1.0)
    modes = stored.calculate_basis(**context)
    output = assert_jittable(lambda model: model.evaluate(**context), stored)
    assert np.allclose(output, np.einsum("i,iyx->yx", stored.coeffs, modes))
    assert np.allclose(stored.solve(output, **context), stored.coeffs, atol=1e-5)
