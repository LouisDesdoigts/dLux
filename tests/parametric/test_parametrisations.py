"""Tests for dLux.parametric.parametrisations."""

import jax
import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_jittable


@pytest.fixture
def affine_data():
    origin = np.arange(6.0).reshape(2, 3)
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [0.5, 0.0],
            [0.0, -0.5],
        ]
    )
    latent = np.asarray([0.2, -0.4])
    return origin, matrix, latent


@pytest.fixture
def selection_data():
    origin = np.arange(6.0).reshape(2, 3)
    mask = np.asarray(
        [
            [False, True, False],
            [True, False, True],
        ]
    )
    latent = np.asarray([0.5, 1.5, -2.0])
    return origin, mask, latent


def test_reparametrisation_evaluation_contract(affine_data):
    origin, matrix, latent = affine_data
    parametrisation = dl.Reparametrisation(origin, matrix, latent)
    default = dl.Reparametrisation(origin, matrix)

    output = assert_jittable(lambda value: value.evaluate(), parametrisation)
    expected = origin + (matrix @ latent).reshape(origin.shape)

    assert parametrisation.shape == origin.shape
    assert np.array_equal(parametrisation.matrix, matrix)
    assert np.allclose(output, expected)
    assert np.allclose(parametrisation.to_coeffs(), expected)
    assert np.allclose(parametrisation.resolve(), expected)
    assert np.array_equal(default.latent, np.zeros(matrix.shape[1]))
    assert np.array_equal(default.evaluate(), origin)


def test_reparametrisation_batching_and_inverse(affine_data):
    origin, matrix, _ = affine_data
    parametrisation = dl.Reparametrisation(origin, matrix)
    latent = np.linspace(-0.6, 0.5, 12).reshape(2, 3, 2)
    batched = dl.Reparametrisation(origin, matrix, latent)

    output = assert_jittable(parametrisation.to_coeffs, latent)
    evaluated = assert_jittable(lambda value: value.evaluate(), batched)
    recovered = assert_jittable(parametrisation.to_latent, output)
    scalar = np.stack(
        [parametrisation.to_coeffs(value) for value in latent.reshape(-1, 2)]
    ).reshape(latent.shape[:-1] + origin.shape)

    assert output.shape == latent.shape[:-1] + origin.shape
    assert recovered.shape == latent.shape
    assert np.allclose(evaluated, output)
    assert np.allclose(output, scalar)
    assert np.allclose(recovered, latent, rtol=1e-5, atol=1e-5)
    assert np.allclose(parametrisation.project(output), output, atol=1e-5)
    assert np.allclose(parametrisation.initialise(output).latent, latent, atol=1e-5)


def test_reparametrisation_projection_and_initialisation(affine_data):
    origin, matrix, latent = affine_data
    parametrisation = dl.Reparametrisation(origin, matrix, latent)
    coeffs = origin + np.asarray([[0.8, -0.3, 1.2], [-0.7, 0.4, 1.1]])

    projected = assert_jittable(parametrisation.project, coeffs)
    initialised = assert_jittable(
        lambda model, values: model.initialise(values), parametrisation, coeffs
    )
    residual = (coeffs - projected).ravel()

    assert isinstance(initialised, dl.Reparametrisation)
    assert np.allclose(matrix.T @ residual, 0, atol=1e-5)
    assert np.allclose(parametrisation.project(projected), projected, atol=1e-5)
    assert np.allclose(initialised.latent, parametrisation.to_latent(coeffs))
    assert np.allclose(initialised.evaluate(), projected)
    assert np.array_equal(parametrisation.latent, latent)
    assert np.array_equal(initialised.origin, parametrisation.origin)
    assert np.array_equal(initialised.matrix, parametrisation.matrix)


def test_rank_deficient_reparametrisation_uses_minimum_norm_inverse():
    origin = np.zeros(3)
    matrix = np.asarray([[1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
    parametrisation = dl.Reparametrisation(origin, matrix)
    coeffs = np.asarray([3.0, 6.0, 4.0])

    latent = assert_jittable(parametrisation.to_latent, coeffs)

    assert np.allclose(latent, np.asarray([1.5, 1.5]), atol=1e-5)
    assert np.allclose(parametrisation.project(coeffs), np.asarray([3.0, 6.0, 0.0]))


@pytest.mark.parametrize("kind", ["dense", "selection"])
def test_parametrisation_jacobian(kind, affine_data, selection_data):
    if kind == "dense":
        origin, matrix, latent = affine_data
        parametrisation = dl.Reparametrisation(origin, matrix, latent)
    else:
        origin, mask, latent = selection_data
        parametrisation = dl.Selection(origin, mask, latent)

    jacobian = jax.jacfwd(parametrisation.to_coeffs)(parametrisation.latent)
    expected = parametrisation.matrix.reshape(
        parametrisation.shape + (parametrisation.latent.shape[-1],)
    )

    assert np.allclose(jacobian, expected)


def test_selection_contract(selection_data):
    origin, mask, latent = selection_data
    selection = dl.Selection(origin, mask, latent)
    indices = np.asarray([1, 3, 5])
    matrix = np.eye(origin.size)[:, indices]
    dense = dl.Reparametrisation(origin, matrix, latent)

    output = assert_jittable(lambda value: value.evaluate(), selection)

    assert selection.shape == origin.shape
    assert np.array_equal(selection.mask, mask)
    assert np.array_equal(selection.matrix, matrix)
    assert np.allclose(output, dense.evaluate())
    assert np.array_equal(output[~mask], origin[~mask])
    assert np.allclose(selection.to_latent(output), latent)


def test_selection_batching(selection_data):
    origin, mask, _ = selection_data
    selection = dl.Selection(origin, mask)
    dense = dl.Reparametrisation(origin, selection.matrix)
    latent = np.linspace(-0.9, 0.8, 18).reshape(2, 3, 3)
    batched = dl.Selection(origin, mask, latent)

    output = assert_jittable(selection.to_coeffs, latent)
    evaluated = assert_jittable(lambda value: value.evaluate(), batched)
    recovered = assert_jittable(selection.to_latent, output)

    assert output.shape == latent.shape[:-1] + origin.shape
    assert np.allclose(evaluated, output)
    assert np.allclose(output, dense.to_coeffs(latent))
    assert np.allclose(recovered, latent, rtol=1e-6, atol=1e-6)
    assert np.allclose(selection.project(output), output)
    assert np.allclose(
        selection.initialise(output).latent, latent, rtol=1e-6, atol=1e-6
    )


def test_selection_projection_and_initialisation(selection_data):
    origin, mask, latent = selection_data
    selection = dl.Selection(origin, mask, latent)
    coeffs = origin + np.asarray([[0.8, -0.3, 1.2], [-0.7, 0.4, 1.1]])
    expected = origin.at[mask].set(coeffs[mask])

    projected = assert_jittable(selection.project, coeffs)
    initialised = assert_jittable(
        lambda model, values: model.initialise(values), selection, coeffs
    )

    assert isinstance(initialised, dl.Selection)
    assert np.allclose(projected, expected)
    assert np.allclose(initialised.evaluate(), expected)
    assert np.allclose(initialised.latent, coeffs[mask] - origin[mask])
    assert np.array_equal(selection.latent, latent)
    assert np.array_equal(initialised.origin, selection.origin)
    assert np.array_equal(initialised.mask, selection.mask)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: dl.Reparametrisation(np.asarray([]), np.ones((0, 1))),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones(6)),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones((1, 2, 3))),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones((5, 2))),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones((6, 0))),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones((6, 7))),
        lambda: dl.Reparametrisation(np.ones((2, 3)), np.ones((6, 2)), np.ones(3)),
        lambda: dl.Selection(np.asarray([]), np.asarray([], dtype=bool)),
        lambda: dl.Selection(np.ones((2, 3)), np.ones((3, 2), dtype=bool)),
        lambda: dl.Selection(np.ones((2, 3)), np.zeros((2, 3), dtype=bool)),
        lambda: dl.Selection(np.ones((2, 3)), np.ones((2, 3), dtype=bool), np.ones(5)),
    ],
)
def test_constructor_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


@pytest.mark.parametrize(
    "method",
    [
        lambda value: value.to_coeffs(np.ones(3)),
        lambda value: value.to_latent(np.ones((3, 2))),
        lambda value: value.project(np.ones((3, 2))),
        lambda value: value.initialise(np.ones((3, 2))),
    ],
)
def test_reparametrisation_method_validation(method):
    parametrisation = dl.Reparametrisation(np.ones((2, 3)), np.ones((6, 2)))

    with pytest.raises((TypeError, ValueError)):
        method(parametrisation)


@pytest.mark.parametrize(
    "method",
    [
        lambda value: value.to_coeffs(np.ones(4)),
        lambda value: value.to_latent(np.ones((3, 2))),
        lambda value: value.project(np.ones((3, 2))),
        lambda value: value.initialise(np.ones((3, 2))),
    ],
)
def test_selection_method_validation(method):
    mask = np.asarray([[False, True, False], [True, False, True]])
    selection = dl.Selection(np.ones((2, 3)), mask)

    with pytest.raises((TypeError, ValueError)):
        method(selection)
