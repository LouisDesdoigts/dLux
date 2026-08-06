"""Tests for dLux.parametric.bases."""

import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def bases():
    return [
        dl.Basis(
            np.arange(120.0).reshape(2, 3, 4, 5),
            np.linspace(-0.2, 0.2, 6).reshape(2, 3),
        ),
        dl.CLIMBBasis(
            np.arange(4 * 6 * 6.0).reshape(4, 6, 6),
            np.linspace(-0.2, 0.2, 4),
            oversample=3,
        ),
        dl.FourierBasis((8, 10), (3, 4), np.linspace(-0.2, 0.2, 12).reshape(3, 4)),
        dl.SplineBasis(
            (8, 10), (2, 3), np.linspace(-0.2, 0.2, 6).reshape(3, 2), method="linear"
        ),
    ]


@pytest.mark.parametrize("index", range(4))
def test_basis_evaluation_contract(index, bases):
    basis = bases[index]
    assert_jittable(lambda value: value.evaluate(), basis, rtol=1e-5, atol=1e-5)
    assert_differentiable(
        lambda coefficients: basis.set(coefficients=coefficients).evaluate(),
        basis.coefficients,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("index", [0, 2, 3])
def test_basis_solution_contract(index, bases):
    basis = bases[index]
    value = basis.evaluate()

    output = assert_jittable(lambda item, data: item.solve_basis(data), basis, value)
    assert output.shape == basis.coefficient_shape


@pytest.mark.parametrize("index", [2, 3])
def test_implicit_basis_contract(index, bases):
    basis = bases[index]
    calculated = assert_jittable(lambda value: value.calculate_basis(), basis)

    assert calculated.shape[: len(basis.coefficient_shape)] == basis.coefficient_shape


def test_basis_specific_operations(bases):
    climb = bases[1]
    latent = assert_jittable(lambda value: value.evaluate_latent(), climb)
    output = climb.evaluate()

    assert latent.shape == (6, 6)
    assert np.all((output >= climb.values[0]) & (output <= climb.values[1]))

    resized = assert_jittable(lambda value: value.resize((5, 7)), bases[2])
    assert resized.evaluate().shape == (5, 7)


def test_coefficient_aliases(bases):
    basis = bases[0]

    assert basis.coeffs is basis.coefficients
    assert basis.c is basis.coefficients
    assert basis.alpha is basis.coefficients


def test_default_explicit_coefficients():
    basis = dl.Basis(np.ones((2, 3, 4)), coefficient_shape=(2,))

    assert basis.coefficient_shape == (2,)
    assert np.allclose(basis.coefficients, 0)
    assert np.allclose(basis.evaluate(), 0)


def test_vectorised_basis_coefficients():
    shared_basis = np.arange(12.0).reshape(1, 3, 4)
    shared = dl.Basis(shared_basis, [1.0, 2.0, 3.0], coefficient_shape=(1,))
    local_basis = np.arange(48.0).reshape(2, 2, 3, 4)
    local_coefficients = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    local = dl.Basis(local_basis, local_coefficients, coefficient_shape=(2,))

    shared_output = assert_jittable(lambda value: value.evaluate(), shared)
    local_output = assert_jittable(lambda value: value.evaluate(), local)

    assert shared_output.shape == (3, 3, 4)
    assert np.allclose(
        shared_output, shared.coefficients[:, None, None] * shared_basis[0]
    )
    assert local_output.shape == (2, 3, 4)
    assert np.allclose(local_output[0], local_basis[0, 0])
    assert np.allclose(local_output[1], local_basis[1, 1])


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Basis(np.ones((2, 3, 4))),
        lambda: dl.Basis(np.ones((2, 3, 4)), np.ones((3, 2))),
        lambda: dl.CLIMBBasis(np.ones((4, 6, 6)), np.ones(4), values=(0.0, 0.5, 1.0)),
        lambda: dl.CLIMBBasis(np.ones((4, 5, 5)), np.ones(4), oversample=3),
        lambda: dl.FourierBasis(8, 3, np.ones((2, 3))),
        lambda: dl.SplineBasis((8, 10, 12), 2),
        lambda: dl.SplineBasis(8, 1),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()
