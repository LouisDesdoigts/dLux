"""Static polynomial topology with differentiable distortion coefficients."""

import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    "definition, terms",
    [
        ({"order": 2}, [1, 2, 3, 4, 5]),
        ({"orders": (3, 1, 3)}, [1, 2, 6, 7, 8, 9]),
        ({"orders": (2, 3)}, [3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_distortion_construction_under_jit_preserves_term_order(definition, terms):
    coordinates = dlu.Grid((3, 2), d=0.2, c=(0.1, -0.2)).coordinates
    expected_powers = dlu.polynomial_powers(3, 2)[:, np.asarray(terms, dtype=int)]
    coeffs = np.arange(2 * len(terms), dtype=float).reshape((2, -1)) / 100

    def calculate(coeffs):
        distortion = dlu.Distortion(distortion=coeffs, **definition)
        return distortion(coordinates), distortion.powers

    expected = dlu.distort_coords(coordinates, coeffs, expected_powers)
    eager, eager_powers = calculate(coeffs)
    compiled, compiled_powers = jax.jit(calculate)(coeffs)

    assert np.array_equal(eager_powers, expected_powers)
    assert np.array_equal(compiled_powers, expected_powers)
    assert np.allclose(eager, expected)
    assert np.allclose(compiled, expected)


def test_distortion_constructed_in_jit_has_analytic_coefficient_gradient():
    coordinates = np.asarray([[[0.2, 0.7], [-0.3, 0.4]], [[-0.1, 0.5], [0.6, 0.8]]])
    x, y = coordinates
    coeffs = np.arange(10, dtype=float).reshape((2, 5)) / 100

    def calculate(coeffs):
        return dlu.Distortion(order=2, distortion=coeffs)(coordinates)

    expected = []
    for coordinate, values in zip(coordinates, coeffs):
        a, b, c, d, e = values
        expected.append(coordinate + a * x + b * y + c * x**2 + d * x * y + e * y**2)
    expected = np.stack(expected)
    monomials = np.stack((x, y, x**2, x * y, y**2))
    expected_gradient = 2 * np.sum(expected[:, None] * monomials[None], axis=(-2, -1))

    output = jax.jit(calculate)(coeffs)
    gradient = jax.jit(jax.grad(lambda values: np.sum(calculate(values) ** 2)))(coeffs)

    assert np.allclose(output, expected)
    assert np.allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_distortion_with_no_terms_is_identity_under_jit():
    coordinates = dlu.Grid((3, 2), d=0.2, c=(0.1, -0.2)).coordinates
    powers = np.empty((2, 0), dtype=int)

    def apply(coordinates):
        return dlu.Distortion(powers=powers)(coordinates)

    assert np.array_equal(apply(coordinates), coordinates)
    assert np.array_equal(jax.jit(apply)(coordinates), coordinates)
    gradient = jax.jit(jax.grad(lambda values: np.sum(apply(values) ** 2)))(coordinates)
    assert np.allclose(gradient, 2 * coordinates)


def test_distortion_explicit_powers_preserve_order_under_jit():
    coordinates = np.asarray([[[0.2, 0.7]], [[-0.1, 0.5]]])
    x, y = coordinates
    powers = np.asarray([[1, 0, 2], [1, 0, 0]])
    coeffs = np.asarray([[0.2, 0.1, -0.3], [-0.1, 0.4, 0.2]])

    def apply(values, powers):
        distortion = dlu.Distortion(distortion=values, powers=powers)
        return distortion(coordinates), distortion.powers

    expected = coordinates + np.stack(
        (0.2 * x * y + 0.1 - 0.3 * x**2, -0.1 * x * y + 0.4 + 0.2 * x**2)
    )
    result, actual_powers = jax.jit(apply)(coeffs, powers)
    assert np.array_equal(actual_powers, powers)
    assert np.allclose(result, expected)
