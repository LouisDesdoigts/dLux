"""Tests for dLux.utils.numerical.polynomials."""

import jax.numpy as np
import jax
import pytest
import zodiax as zdx

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


def test_stored_polynomial_sampling_and_mapping_are_shared_by_all_methods():
    axis = dlu.Axis(np.linspace(-1.0, 1.0, 7), unit=None)
    mapping = zdx.Map(s=2.0, b=0.25)
    polynomial = dlu.Polynomial(2, [1.0, 2.0, 3.0], grid=axis, transform=mapping)
    coordinates = 2 * axis.axes[0] + 0.25
    expected = 1 + 2 * coordinates + 3 * coordinates**2
    values = assert_jittable(lambda value: value.resolve(), polynomial)
    assert np.allclose(values, expected)
    terms = polynomial.calculate_basis()
    assert np.allclose(
        terms, np.stack((np.ones_like(coordinates), coordinates, coordinates**2))
    )
    assert np.allclose(polynomial.solve(values), polynomial.coeffs, atol=2e-5)
    override = zdx.Map(b=-0.2)
    assert np.allclose(
        polynomial.resolve(transform=override),
        1 + 2 * (axis.axes[0] - 0.2) + 3 * (axis.axes[0] - 0.2) ** 2,
    )


def test_polynomial_pairs_axis_batches_with_coefficients_across_all_methods():
    axis = dlu.Axis(
        n=3, d=np.asarray([[0.2], [0.3]]), c=np.asarray([[-0.2], [1.1]]), unit=None
    )
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    polynomial = dlu.Polynomial(1, coeffs=coeffs, grid=axis)
    coordinates = axis.axes[0]
    expected = coeffs[:, 0, None] + coeffs[:, 1, None] * coordinates

    result = assert_jittable(lambda value: value.resolve(), polynomial)
    assert result.shape == (2, 3)
    assert np.allclose(result, expected, atol=1e-6)
    unbound = polynomial.set(grid=None)
    runtime = assert_jittable(lambda grid: unbound.resolve(grid=grid), axis)
    assert np.allclose(runtime, expected, atol=1e-6)
    terms = polynomial.calculate_basis()
    assert np.allclose(
        terms, np.stack((np.ones_like(coordinates), coordinates), axis=-2)
    )
    solved = assert_jittable(lambda value: polynomial.solve(value), result)
    assert np.allclose(solved, coeffs, atol=2e-6)

    def loss(coefficients, centres):
        model = polynomial.set(coeffs=coefficients).set("grid.c", centres)
        return np.sum(model.resolve() ** 2)

    coefficient_grad, centre_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        coeffs, axis.c
    )
    expected_coefficients = np.stack(
        (2 * expected.sum(-1), 2 * (expected * coordinates).sum(-1)), axis=-1
    )
    expected_centres = 2 * (expected * coeffs[:, 1, None]).sum(-1, keepdims=True)
    assert np.allclose(coefficient_grad, expected_coefficients, rtol=1e-6)
    assert np.allclose(centre_grad, expected_centres, rtol=1e-6)


def test_polynomial_keeps_explicit_bare_query_shapes():
    coordinates = np.asarray([[-0.5, 0.0, 0.5], [0.75, 1.0, 1.5]])
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    polynomial = dlu.Polynomial(1, coeffs=coeffs)
    expected = coeffs[:, 0, None, None] + coeffs[:, 1, None, None] * coordinates

    result = assert_jittable(
        lambda value: polynomial.resolve(coordinates=value), coordinates
    )
    assert result.shape == (2, 2, 3)
    assert np.allclose(result, expected)
    assert np.allclose(polynomial.resolve(coordinates=0.5), coeffs @ np.array([1, 0.5]))


def test_spatial_polynomial_transform_is_applied_once_and_is_differentiable():
    grid = dlu.Grid(n=(4, 3), d=0.4)
    transform = dlu.Affine(translation=[0.1, -0.2])
    polynomial = dlu.Polynomial(
        1, [1.0, 2.0, 3.0], ndim=2, grid=grid, transform=transform
    )
    coordinates = transform(grid.coordinates)
    expected = 1 + 2 * coordinates[0] + 3 * coordinates[1]
    values = assert_jittable(lambda value: value.resolve(), polynomial)
    assert np.allclose(values, expected)
    assert np.allclose(polynomial.solve(values), polynomial.coeffs, atol=2e-6)
    gradient = jax.jit(
        jax.grad(
            lambda value: polynomial.set("transform.translation", value)
            .resolve()
            .mean()
        )
    )(transform.translation)
    assert np.allclose(gradient, np.array([-2.0, -3.0]), atol=1e-6)


def test_selected_polynomial_degrees():
    variables = np.linspace(-1, 1, 8)

    linear = dl.Polynomial(degrees=1, coeffs=[2.0])
    selected = dl.Polynomial(degrees=[0, 2], coeffs=[1.0, 3.0])

    assert np.allclose(linear.resolve(coordinates=variables), 2 * variables)
    assert np.allclose(selected.resolve(coordinates=variables), 1 + 3 * variables**2)


def test_selected_polynomial_degrees_construct_inside_jit():
    coordinates = np.asarray([-0.8, -0.2, 0.0, 0.4, 0.9])
    coeffs = np.asarray([1.5, -0.3])

    def evaluate(coeffs):
        polynomial = dlu.Polynomial(degrees=(3, 1, 3), coeffs=coeffs)
        return polynomial.resolve(coordinates=coordinates)

    expected = 1.5 * coordinates - 0.3 * coordinates**3
    expected_gradient = 2 * np.asarray(
        [np.sum(expected * coordinates), np.sum(expected * coordinates**3)]
    )
    assert np.allclose(evaluate(coeffs), expected)
    assert np.allclose(jax.jit(evaluate)(coeffs), expected)
    gradient = jax.jit(jax.grad(lambda values: np.sum(evaluate(values) ** 2)))(coeffs)
    assert np.allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_explicit_polynomial_powers_preserve_order_inside_jit():
    coordinates = np.asarray([[[0.2, 0.7]], [[-0.1, 0.5]]])
    x, y = coordinates
    powers = np.asarray([[1, 0, 2], [1, 0, 0]])
    coeffs = np.asarray([2.0, -0.3, 0.8])

    def evaluate(coeffs, powers):
        polynomial = dlu.Polynomial(powers=powers, coeffs=coeffs)
        return polynomial.resolve(coordinates=coordinates)

    expected = 2 * x * y - 0.3 + 0.8 * x**2
    result = jax.jit(evaluate)(coeffs, powers)
    gradient = jax.jit(jax.grad(lambda values: evaluate(values, powers).sum()))(coeffs)
    assert np.allclose(result, expected)
    assert np.allclose(gradient, np.asarray([(x * y).sum(), x.size, (x**2).sum()]))


def test_polynomial_powers_selected_degrees_and_explicit_order():
    generated = dlu.polynomial_powers(ndim=2, degrees=(2, 0, 2))
    assert np.array_equal(generated, np.asarray([[0, 2, 1, 0], [0, 0, 1, 2]]))
    powers = np.asarray([[1, 0, 2], [1, 0, 0]])
    assert np.array_equal(dlu.polynomial_powers(powers=powers), powers)


@pytest.mark.parametrize(
    "definition",
    [
        {},
        {"degree": 2, "degrees": (1, 2)},
        {"degrees": (1,), "powers": [[1]]},
    ],
)
def test_polynomial_powers_requires_one_definition(definition):
    with pytest.raises(ValueError, match="exactly one"):
        dlu.polynomial_powers(**definition)


def test_polynomial_rejects_an_empty_basis():
    powers = dlu.polynomial_powers(powers=np.empty((2, 0), dtype=int))
    assert powers.shape == (2, 0)
    with pytest.raises(ValueError):
        dlu.Polynomial(powers=powers)


def test_multivariate_polynomial(coordinates):
    powers = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    polynomial = dl.Polynomial(coeffs=[1.0, 2.0, 3.0, 4.0], powers=powers)
    x, y = coordinates

    output = polynomial.resolve(coordinates=coordinates)
    assert np.allclose(output, 1 + 2 * x + 3 * y + 4 * x * y)


@pytest.mark.parametrize("coordinate_source", ["array", "grid"])
def test_explicit_polynomial_contract(coordinate_source, coordinates):
    if coordinate_source == "grid":
        coordinate_source = dl.Grid(n=10, d=0.2, unit="m").coordinates
    else:
        coordinate_source = coordinates
    terms = dlu.polynomial_basis(coordinate_source, dlu.polynomial_powers(1, 2))
    polynomial = dl.Basis(terms, coeffs=[1.0, 2.0, 3.0])

    output = assert_jittable(lambda value: value.evaluate(), polynomial)
    recovered = polynomial.solve(output)
    assert output.shape == coordinates.shape[-2:]
    assert np.allclose(recovered, polynomial.coeffs, atol=1e-5)


def test_explicit_polynomial_samples_promoted_grid():
    grid = dl.Grid(n=8, d=0.25, unit="m")
    terms = dlu.polynomial_basis(grid.coordinates, dlu.polynomial_powers(1, 2))
    polynomial = dl.Basis(terms, coeffs=[1.0, 2.0, 3.0])

    output = assert_jittable(lambda value: value.evaluate(), polynomial)

    assert output.shape == (8, 8)
    assert np.allclose(polynomial.solve(output), polynomial.coeffs, atol=1e-5)


@pytest.mark.parametrize("nsides", [0, 6])
def test_dynamic_zernike_contract(nsides, coordinates):
    zernike = dl.Basis(dl.utils.Zernike(nolls=[4]), coeffs=[1.0])
    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates, n_sides=nsides), zernike
    )
    assert output.shape == coordinates.shape[-2:]


def test_zernike_order_selection(coordinates):
    explicit = dl.ZernikeDef(order=3)(coordinates=coordinates)
    dynamic = dl.Basis(dl.utils.Zernike(order=3))

    assert explicit.coeffs.shape == (10,)
    assert dynamic.coeffs.shape == (10,)

    with pytest.raises(ValueError, match="exactly one"):
        dl.Basis(dl.utils.Zernike(order=3, orders=[2, 3]))


@pytest.mark.parametrize("npix", [(6, 4), 6])
def test_zernike_basis_accepts_grid_or_its_coordinate_array(npix):
    grid = dl.Grid(n=npix, d=0.2)
    options = {"nolls": [1, 4], "coeffs": [0.2, -0.1], "diameter": 1.2}
    sampled = dl.ZernikeDef(**options)(grid=grid)
    explicit = dl.ZernikeDef(**options)(coordinates=grid.coordinates)

    output = assert_jittable(lambda model: model.resolve(), sampled)

    assert output.shape == grid.shape
    assert np.allclose(output, explicit.resolve())


@pytest.mark.parametrize(
    "make_basis",
    [
        lambda coordinates: dl.ZernikeDef(orders=[2], coeffs=np.linspace(-0.2, 0.2, 3))(
            coordinates=coordinates
        ),
        lambda coordinates: dl.Basis(
            dl.utils.Zernike(orders=[2]), coeffs=np.linspace(-0.2, 0.2, 3)
        ),
        lambda coordinates: dl.Basis(
            dl.utils.Zernike(nolls=[4, 5], n_sides=6), coeffs=np.asarray([0.1, -0.1])
        ),
        lambda coordinates: dl.Polynomial(2, coeffs=np.linspace(-0.2, 0.2, 6), ndim=2),
    ],
)
def test_polynomial_basis_contract(make_basis, coordinates):
    basis = make_basis(coordinates)
    context = {"coordinates": coordinates}

    output = assert_jittable(lambda value: value.evaluate(**context), basis)
    assert output.shape == coordinates.shape[-2:]
    assert_differentiable(
        lambda coeffs: basis.set(coeffs=coeffs).evaluate(**context),
        basis.coeffs,
        atol=2e-6,
    )


@pytest.mark.parametrize(
    "basis", [dl.Basis(dl.utils.Zernike(orders=[2])), dl.Polynomial(2, ndim=2)]
)
def test_dynamic_basis_context(basis, coordinates, make_wavefront):
    def sample(value):
        if isinstance(value, dl.Basis):
            return value.basis.evaluate(coordinates=coordinates)
        return value.calculate_basis(coordinates=coordinates)

    calculated = assert_jittable(sample, basis)
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
        lambda coordinates: dl.Basis(dl.utils.Zernike(nolls=[0])),
        lambda coordinates: dl.ZernikeDef()(coordinates=coordinates),
        lambda coordinates: dl.ZernikeDef(nolls=[])(coordinates=coordinates),
        lambda coordinates: dl.Basis(dl.utils.Zernike(nolls=[1], n_sides=2)),
        lambda coordinates: dl.Basis(dl.utils.Zernike(nolls=[1], diameter=0.0)),
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
    "basis", [dl.Basis(dl.utils.Zernike(nolls=[1])), dl.Polynomial(1, ndim=2)]
)
def test_coordinate_context_validation(basis):
    assert isinstance(basis.resolve(), type(basis))
    with pytest.raises(zdx.Unresolved):
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
    grid = dl.Grid(n=(12, 12), diam=1.0, unit="m")
    wavefront = make_wavefront(grid=grid)
    basis = dl.Basis(dl.utils.Zernike(nolls=[4, 5]), coeffs=[0.1, -0.2])
    assert isinstance(basis.resolve(wavefront=wavefront), dl.Basis)
    with pytest.raises(ValueError, match="grid or coordinates"):
        basis.basis.evaluate(wavefront=wavefront)
    default = basis.evaluate(grid=grid)
    assert np.allclose(default, basis.evaluate(grid=grid, diameter=2.0))
    assert np.allclose(default, basis.evaluate(grid=grid, wavefront=wavefront))
    stored = basis.set("basis.diameter", 0.8)
    transform = dl.Affine(translation=[0.05, -0.02])
    context = dict(grid=grid, transform=transform, diameter=1.0)
    modes = stored.basis.evaluate(**context)
    output = assert_jittable(lambda model: model.evaluate(**context), stored)
    assert np.allclose(output, np.einsum("i,iyx->yx", stored.coeffs, modes))
    assert np.allclose(stored.solve(output, **context), stored.coeffs, atol=1e-5)


@pytest.mark.parametrize(
    "operation,error",
    [
        (lambda: dlu.polynomial_powers(powers=np.zeros((0, 2), dtype=int)), ValueError),
        (lambda: dlu.polynomial_powers(powers=[0.0, 1.0]), TypeError),
        (lambda: dlu.polynomial_powers(powers=[-1, 0]), ValueError),
        (lambda: dlu.polynomial_powers(1, ndim=True), TypeError),
        (lambda: dlu.polynomial_powers(1, ndim=0), ValueError),
        (lambda: dlu.polynomial_powers(True), TypeError),
        (lambda: dlu.polynomial_powers(degrees=[True]), TypeError),
        (
            lambda: dlu.polynomial_basis(
                np.ones((2, 3)), np.empty((2, 0), dtype=int), axis=None
            ),
            ValueError,
        ),
        (
            lambda: dlu.polynomial_basis(np.ones((2, 3)), [0.0, 1.0], axis=None),
            TypeError,
        ),
        (
            lambda: dlu.polynomial_basis(
                np.ones((3, 2)), np.ones((2, 3), dtype=int), axis=None
            ),
            ValueError,
        ),
        (
            lambda: dlu.polynomial_basis(np.ones((2, 3)), np.empty((2, 0), dtype=int)),
            ValueError,
        ),
        (lambda: dlu.polynomial_basis(np.ones((2, 3)), [0.0, 1.0]), TypeError),
        (
            lambda: dlu.polynomial_basis(np.ones((2, 3)), [[0, 1], [1, 0]], axis=3),
            ValueError,
        ),
        (lambda: dlu.polynomial_basis(np.ones((2, 3)), [[0, 1]], axis=0), ValueError),
        (lambda: dlu.eval_polynomial(np.arange(3.0), [1.0], [0, 1]), ValueError),
        (lambda: dlu.solve_polynomial(np.ones(4), np.arange(3.0), [0, 1]), ValueError),
    ],
)
def test_polynomial_validation_tails(operation, error):
    """Coverage tails: invalid powers, dimensions and evaluation shapes."""
    with pytest.raises(error):
        operation()
