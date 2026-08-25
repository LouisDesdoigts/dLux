"""Tests for dLux.parametric.parametrisations."""

import jax
import jax.numpy as np
import pytest

import dLux as dl

from tests.helpers import assert_differentiable, assert_jittable


class _ScaleTransform(dl.ParameterTransform):
    """Simple reversible coordinate scaling used to test composition."""

    scale: jax.Array

    def __init__(self, scale):
        self.scale = np.asarray(scale, dtype=float)

    def encode(self, value):
        return self.scale * value

    def decode(self, value):
        return value / self.scale


class _ShapeChangingTransform(dl.ParameterTransform):
    """Invalid transform used to exercise shape validation."""

    def encode(self, value):
        return value.reshape(-1)

    def decode(self, value):
        return value


class _NonInvertingTransform(dl.ParameterTransform):
    """Invalid transform used to exercise scale-aware inverse validation."""

    def encode(self, value):
        return np.zeros_like(value)

    def decode(self, value):
        return np.zeros_like(value)


class _InexactTransform(dl.ParameterTransform):
    """Almost reversible transform used to test accumulated chain error."""

    def encode(self, value):
        return value * (1 + 9e-6)

    def decode(self, value):
        return value


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
    assert isinstance(default.transform, dl.IdentityTransform)


def test_parameter_transform_composition_and_paths():
    physical = np.asarray([2.0, 4.0])
    identity = dl.IdentityTransform()
    empty = dl.CompositeTransform()
    transform = dl.CompositeTransform(
        [("log", dl.LogTransform()), ("scale", _ScaleTransform(2.0))]
    )

    encoded = assert_jittable(transform.encode, physical)
    decoded = assert_jittable(transform.decode, encoded)
    updated = transform.set("transforms.scale.scale", np.asarray(3.0))

    assert np.array_equal(identity.encode(physical), physical)
    assert np.array_equal(identity.decode(physical), physical)
    assert np.array_equal(empty.encode(physical), physical)
    assert np.array_equal(empty.decode(physical), physical)
    assert np.allclose(encoded, 2 * np.log(physical))
    assert np.allclose(decoded, physical)
    assert np.allclose(updated.encode(physical), 3 * np.log(physical))
    assert np.array_equal(transform.transforms["scale"].scale, np.asarray(2.0))


def test_transform_preserves_native_axes_and_leading_batches():
    origin = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    scale = np.asarray([[1.0, 2.0], [4.0, 8.0]])
    transform = _ScaleTransform(scale)
    parametrisation = dl.Reparametrisation(
        origin, np.eye(origin.size), transform=transform
    )
    latent = np.linspace(-0.4, 0.3, 8).reshape(2, origin.size)

    coeffs = assert_jittable(parametrisation.to_coeffs, latent)
    recovered = assert_jittable(parametrisation.to_latent, coeffs)
    expected = origin + latent.reshape((2,) + origin.shape) / scale

    assert coeffs.shape == (2,) + origin.shape
    assert np.allclose(coeffs, expected)
    assert np.allclose(recovered, latent, atol=1e-5)


def test_log_reparametrisation_mapping_batching_and_gradient():
    origin = np.asarray([2.0, 3.0, 5.0])
    matrix = np.asarray([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]])
    latent = np.asarray([0.2, -0.3])
    parametrisation = dl.Reparametrisation(
        origin, matrix, latent, transform=dl.LogTransform()
    )
    batch = np.asarray([[0.1, -0.2], [-0.3, 0.4], [0.5, 0.2]])

    output = assert_jittable(lambda value: value.evaluate(), parametrisation)
    batched = assert_jittable(parametrisation.to_coeffs, batch)
    expected = origin * np.exp(matrix @ latent)
    mapped = jax.vmap(parametrisation.to_coeffs)(batch)
    gradient = jax.grad(lambda value: parametrisation.to_coeffs(value).sum())(latent)
    assert_differentiable(parametrisation.to_coeffs, latent)

    assert np.allclose(output, expected)
    assert np.allclose(parametrisation.to_coeffs(np.zeros(2)), origin)
    assert np.allclose(parametrisation.to_latent(output), latent, atol=1e-5)
    assert np.allclose(parametrisation.to_latent(batched), batch, atol=1e-5)
    assert np.allclose(batched, mapped)
    assert np.allclose(gradient, matrix.T @ expected, atol=1e-5)


def test_log_reparametrisation_projects_in_encoded_space():
    origin = np.asarray([1.0, 2.0, 4.0])
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    parametrisation = dl.Reparametrisation(origin, matrix, transform=dl.LogTransform())
    delta = np.asarray([0.3, -0.2, 0.8])
    coeffs = origin * np.exp(delta)

    projected = assert_jittable(parametrisation.project, coeffs)
    initialised = assert_jittable(
        lambda model, values: model.initialise(values), parametrisation, coeffs
    )
    latent = np.asarray([8 / 15, 1 / 30])
    residual = np.log(coeffs) - np.log(projected)

    assert np.allclose(parametrisation.to_latent(coeffs), latent, atol=1e-5)
    assert np.allclose(matrix.T @ residual, 0, atol=1e-5)
    assert np.allclose(initialised.evaluate(), projected)
    assert isinstance(initialised.transform, dl.LogTransform)
    assert np.array_equal(initialised.origin, origin)
    assert np.array_equal(initialised.matrix, matrix)


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


def test_log_selection_transforms_only_active_coefficients():
    origin = np.asarray([-1.0, 2.0, -3.0, 8.0])
    mask = np.asarray([False, True, False, True])
    latent = np.log(np.asarray([3.0, 0.5]))
    selection = dl.Selection(origin, mask, latent, transform=dl.LogTransform())
    expected = np.asarray([-1.0, 6.0, -3.0, 4.0])
    coeffs = np.asarray([-10.0, 12.0, 0.0, 2.0])
    batch = np.log(np.asarray([[3.0, 0.5], [0.25, 2.0], [1.5, 0.75]]))

    output = assert_jittable(lambda value: value.evaluate(), selection)
    projected = assert_jittable(selection.project, coeffs)
    batched = assert_jittable(selection.to_coeffs, batch)
    mapped = jax.vmap(selection.to_coeffs)(batch)
    recovered = assert_jittable(selection.to_latent, batched)
    jacobian = jax.jacfwd(selection.to_coeffs)(latent)
    expected_jacobian = np.zeros((origin.size, latent.size))
    expected_jacobian = expected_jacobian.at[mask, np.arange(latent.size)].set(
        expected[mask]
    )
    assert_differentiable(selection.to_coeffs, latent)

    assert np.allclose(output, expected)
    assert np.array_equal(output[~mask], origin[~mask])
    assert np.allclose(selection.to_latent(output), latent)
    assert np.allclose(projected, np.asarray([-1.0, 12.0, -3.0, 2.0]))
    assert np.allclose(batched, mapped)
    assert np.allclose(recovered, batch)
    assert np.allclose(jacobian, expected_jacobian)


def test_transformed_reparametrisation_retains_nested_set_paths():
    origin = np.asarray([2.0, 4.0])
    matrix = np.eye(2)
    transform = dl.CompositeTransform(
        [("log", dl.LogTransform()), ("scale", _ScaleTransform(2.0))]
    )
    coeffs = dl.Reparametrisation(origin, matrix, transform=transform)
    basis = dl.Basis(np.arange(8.0).reshape(2, 2, 2), coeffs)
    latent = np.asarray([0.2, -0.4])

    updated = basis.set("latent", latent)
    rescaled = updated.set("coeffs.transform.transforms.scale.scale", np.asarray(4.0))
    expected = origin * np.exp(latent / 2)
    expected_rescaled = origin * np.exp(latent / 4)
    output = assert_jittable(lambda model: model.evaluate(), updated)
    gradient = assert_differentiable(
        lambda value: basis.set("latent", value).evaluate(), latent
    )

    assert np.allclose(updated.coeffs.evaluate(), expected)
    assert np.allclose(rescaled.coeffs.evaluate(), expected_rescaled)
    assert np.allclose(output, np.einsum("n,nij->ij", expected, basis.basis))
    assert np.any(np.abs(gradient) > 0)
    assert np.array_equal(basis.coeffs.latent, np.zeros(2))
    assert np.array_equal(
        updated.coeffs.transform.transforms["scale"].scale, np.asarray(2.0)
    )


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
    "constructor, exception, match",
    [
        (
            lambda: dl.Reparametrisation(
                np.ones(2), np.eye(2), transform=lambda value: value
            ),
            TypeError,
            "transform must be a ParameterTransform",
        ),
        (
            lambda: dl.Reparametrisation(
                np.asarray([0.0, 1.0]), np.eye(2), transform=dl.LogTransform()
            ),
            ValueError,
            "strictly positive",
        ),
        (
            lambda: dl.Selection(
                np.asarray([0.0, 1.0]),
                np.asarray([True, False]),
                transform=dl.LogTransform(),
            ),
            ValueError,
            "strictly positive",
        ),
        (
            lambda: dl.Reparametrisation(
                np.ones((2, 2)), np.eye(4), transform=_ShapeChangingTransform()
            ),
            ValueError,
            "encode must preserve shape",
        ),
        (
            lambda: dl.Reparametrisation(
                np.asarray([1e-9]),
                np.ones((1, 1)),
                transform=_NonInvertingTransform(),
            ),
            ValueError,
            "decode must invert transform.encode",
        ),
        (
            lambda: dl.CompositeTransform([object()]),
            TypeError,
            "not an allowed type",
        ),
        (
            lambda: dl.Reparametrisation(
                np.asarray([1.0]),
                np.ones((1, 1)),
                transform=dl.CompositeTransform(
                    [_InexactTransform(), _InexactTransform()]
                ),
            ),
            ValueError,
            "decode must invert transform.encode",
        ),
        (
            lambda: dl.CompositeTransform([("bad name", dl.IdentityTransform())]),
            ValueError,
            "Names cannot contain spaces",
        ),
        (
            lambda: dl.CompositeTransform([("bad.name", dl.IdentityTransform())]),
            ValueError,
            "cannot contain periods",
        ),
        (
            lambda: dl.CompositeTransform([("items", dl.IdentityTransform())]),
            ValueError,
            "cannot shadow mapping attributes",
        ),
    ],
)
def test_transform_constructor_validation(constructor, exception, match):
    with pytest.raises(exception, match=match):
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
