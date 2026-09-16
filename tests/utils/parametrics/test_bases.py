"""Tests for dLux.utils.parametrics.bases."""

import jax.numpy as np
import jax
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


@pytest.fixture
def bases():
    return [
        dl.Basis(
            np.arange(120.0).reshape(2, 3, 4, 5),
            np.linspace(-0.2, 0.2, 6).reshape(2, 3),
            axes=2,
        ),
        dl.CLIMBBasis(
            np.arange(4 * 6 * 6.0).reshape(4, 6, 6),
            np.linspace(-0.2, 0.2, 4),
            oversample=3,
        ),
        dl.FourierBasis((8, 10), (3, 4), np.linspace(-0.2, 0.2, 12).reshape(3, 4)),
    ]


@pytest.mark.parametrize("index", range(3))
def test_basis_evaluation_contract(index, bases):
    basis = bases[index]
    assert_jittable(lambda value: value.evaluate(), basis, rtol=1e-5, atol=1e-5)
    assert_differentiable(
        lambda coeffs: basis.set(coeffs=coeffs).evaluate(),
        basis.coeffs,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("index", [0, 2])
def test_basis_solution_contract(index, bases):
    basis = bases[index]
    value = basis.evaluate()

    output = assert_jittable(lambda item, data: item.solve(data), basis, value)
    assert output.shape == basis.coeffs.shape


@pytest.mark.parametrize("index", [2])
def test_implicit_basis_contract(index, bases):
    basis = bases[index]
    calculated = assert_jittable(lambda value: value.calculate_basis(), basis)

    assert calculated.shape[: len(basis.shape)] == basis.shape


def test_basis_specific_operations(bases):
    climb = bases[1]
    latent = assert_jittable(lambda value: value.expansion.resolve(), climb)
    output = climb.evaluate()

    assert latent.shape == (6, 6)
    assert np.all((output >= climb.values[0]) & (output <= climb.values[1]))

    resized = assert_jittable(lambda value: value.resize((5, 7)), bases[2])
    assert resized.evaluate().shape == (7, 5)


def test_omitted_explicit_coeffs_start_at_zero():
    basis = dl.Basis(np.ones((2, 3, 4)))
    assert np.array_equal(basis.coeffs, np.zeros(2))
    assert np.array_equal(basis.evaluate(), np.zeros((3, 4)))
    assert np.allclose(basis.set(coeffs=np.array([1.0, 2.0])).evaluate(), 3.0)


def test_vectorised_basis_coeffs():
    shared_basis = np.arange(12.0).reshape(1, 3, 4)
    shared = dl.Basis(shared_basis, [[1.0], [2.0], [3.0]])
    local_basis = np.arange(48.0).reshape(2, 2, 3, 4)
    local_coeffs = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    local = dl.Basis(local_basis, local_coeffs)

    shared_output = assert_jittable(lambda value: value.evaluate(), shared)
    local_output = assert_jittable(lambda value: value.evaluate(), local)

    assert shared_output.shape == (3, 3, 4)
    assert np.allclose(shared_output, shared.coeffs[:, 0, None, None] * shared_basis[0])
    assert local_output.shape == (2, 2, 3, 4)
    assert np.allclose(local_output[0], local_basis[0])
    assert np.allclose(local_output[1], local_basis[1])


@pytest.mark.parametrize("method", ["scan", "scatter"])
def test_pasted_basis_contract(method):
    spec = dl.PasteSpec(
        n=(6, 4),
        shape=(2, 2),
        starts=[[0, 0], [4, 2]],
        offsets=np.zeros((2, 2)),
        d=np.ones(2),
    )
    modes = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )
    basis = np.broadcast_to(modes, (2, 2, 2, 2))
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    pasted = dl.PastedBasis(basis, spec, coeffs, method)

    output = assert_jittable(lambda value: value.evaluate(), pasted)
    recovered = assert_jittable(lambda value: pasted.solve(value), output)
    assert_differentiable(lambda values: pasted.set(coeffs=values).evaluate(), coeffs)

    assert output.shape == (4, 6)
    assert np.allclose(recovered, coeffs)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.PastedBasis(np.ones((2, 1, 2, 2)), "invalid"),
        lambda: dl.PastedBasis(
            np.ones((2, 1, 2, 2)),
            dl.PasteSpec((4, 4), (2, 2), [[0, 0]], [[0.0, 0.0]], [1.0, 1.0]),
        ),
    ],
)
def test_pasted_basis_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Basis(np.ones((2, 3, 4)), axes=0),
        lambda: dl.Basis(np.ones((2, 3, 4)), np.ones((2, 3))),
        lambda: dl.CLIMBBasis(np.ones((4, 6, 6)), np.ones(4), values=(0.0, 0.5, 1.0)),
        lambda: dl.CLIMBBasis(np.ones((4, 5, 5)), np.ones(4), oversample=3),
        lambda: dl.FourierBasis(8, 3, np.ones((2, 3))),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_basis_rank_deficient_solve_returns_minimum_norm_representative():
    basis = dl.Basis(np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    value = np.asarray([4.0, 3.0])
    coeffs = assert_jittable(lambda model: model.solve(value), basis)
    assert np.allclose(coeffs, np.asarray([2.0, 2.0]), atol=1e-6)
    assert np.allclose(
        basis.set(coeffs=coeffs).evaluate(), np.asarray([4.0, 0.0]), atol=1e-6
    )


def test_fourier_rectangular_image_axes_and_compact_evaluation():
    coeffs = np.arange(6.0).reshape(2, 3)
    fourier = dl.FourierBasis((7, 5), (2, 3), coeffs)
    explicit = dl.Basis(fourier.calculate_basis(), coeffs, axes=2)
    assert fourier.resolve().shape == (5, 7)
    assert np.allclose(fourier.resolve(), explicit.resolve(), atol=1e-5)


def test_basis_coefficients_batch_like_independent_models():
    vectors = np.arange(24.0).reshape(2, 3, 4)
    coeffs = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    basis = dl.Basis(vectors, coeffs)
    expected = jax.vmap(lambda values: dl.Basis(vectors, values).resolve())(coeffs)
    assert np.allclose(basis.resolve(), expected)
