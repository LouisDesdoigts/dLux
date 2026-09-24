"""Tests for dLux.utils.sampled."""

import jax.numpy as np
import jax
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu
from dLux.utils import bases, numerical, optical, sampled

from tests.helpers import assert_differentiable, assert_jittable


@pytest.mark.parametrize(
    "namespace,name",
    [
        (bases, "Basis"),
        (sampled, "Sampled"),
        (numerical, "Interp"),
        (numerical, "Gaussian"),
        (optical, "Zernike"),
    ],
)
def test_public_utility_namespaces(namespace, name):
    """Numerical ownership is visible while convenient class exports agree."""
    assert name in namespace.__all__
    # Zernike has an old core compatibility name; its canonical API is in utils.
    if name != "Zernike":
        assert getattr(dl, name) is getattr(namespace, name)
    assert getattr(dlu, name) is getattr(namespace, name)


class CoordinateValue(dl.Sampled):
    """Small concrete parametric used to exercise composition."""

    def evaluate(self, *, grid=None, coordinates=None, transform=None, **context):
        local = self.sampling_context(
            grid, coordinates=coordinates, transform=transform, ndim=2, **context
        )
        return local["coordinates"][0]


@pytest.fixture
def coordinates():
    return np.linspace(0.1, 0.4, 32).reshape(2, 4, 4)


def test_sampled_transform_contract(coordinates):
    parametric = CoordinateValue(transform=dl.Affine(translation=[0.1, 0.0]))

    assert_jittable(lambda value: value.evaluate(coordinates=coordinates), parametric)
    assert_differentiable(
        lambda translation: parametric.set(
            "transform.translation", translation
        ).evaluate(coordinates=coordinates),
        parametric.transform.translation,
    )

    raised = parametric.set("translation", np.asarray((0.2, -0.1)))
    assert np.array_equal(parametric.get("translation"), np.asarray([0.1, 0.0]))
    assert np.array_equal(raised.transform.translation, np.asarray([0.2, -0.1]))

    resolved = assert_jittable(
        lambda value: value.resolve(coordinates=coordinates), parametric
    )
    assert np.allclose(resolved, coordinates[0] - 0.1)
    gradient = jax.grad(
        lambda shift: parametric.set("translation", shift)
        .resolve(coordinates=coordinates)
        .sum()
    )(parametric.transform.translation)
    assert np.allclose(gradient, np.asarray([-16.0, 0.0]))


def test_value_transform_contract(coordinates):
    parametric = zdx.Pow(CoordinateValue(), p=2)
    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates), parametric
    )
    assert np.allclose(output, coordinates[0] ** 2)


def test_conversion_and_resolution(coordinates):
    parametric = CoordinateValue()

    assert np.array_equal(
        zdx.resolve(parametric, coordinates=coordinates), coordinates[0]
    )


def test_interpolation_evaluation_and_sample_gradients():
    interpolation = dl.Interp(
        grid_in=np.asarray([0.0, 1.0, 2.0]),
        array=np.asarray([0.0, 1.0, 0.0]),
        ndim=1,
        method="linear",
    )
    variables = np.asarray([0.5, 1.5])

    values = assert_jittable(
        lambda model: model.evaluate(coordinates=variables), interpolation
    )

    assert np.allclose(values, 0.5)
    gradient = assert_differentiable(
        lambda samples: interpolation.set(array=samples).evaluate(
            coordinates=variables
        ),
        interpolation.array,
    )
    assert np.allclose(gradient, np.asarray([0.5, 1.0, 0.5]))


@pytest.mark.parametrize("operation", ["sum", "product", "union", "intersection"])
def test_combination_contract(operation, coordinates):
    parametric = dl.Combine(
        [
            CoordinateValue(),
            CoordinateValue(transform=dl.Affine(scale=[1.1, 0.9])),
        ],
        operation,
    )

    output = assert_jittable(
        lambda value: value.evaluate(coordinates=coordinates), parametric
    )
    assert output.shape == coordinates.shape[-2:]
    assert_differentiable(
        lambda value: parametric.evaluate(coordinates=value), coordinates
    )


def test_dictionary_combination(coordinates):
    parametric = dl.Combine(
        {
            "direct": CoordinateValue(),
            "shifted": CoordinateValue(transform=dl.Affine(translation=[0.1, 0.0])),
        }
    )

    output = parametric.evaluate(coordinates=coordinates)

    assert list(parametric.parametrics) == ["direct", "shifted"]
    assert parametric.shifted is parametric.parametrics["shifted"]
    assert np.array_equal(parametric.get("translation"), np.asarray([0.1, 0.0]))
    assert output.shape == coordinates.shape[-2:]


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: CoordinateValue(grid=np.ones(2)),
        lambda: CoordinateValue(transform=np.eye(2)),
        lambda: dl.Combine([CoordinateValue()], "invalid"),
        lambda: dl.Combine([np.ones(2)]),
        lambda: dl.Interp([1.0], [0.0], ndim=1).prepare(),
        lambda: dl.Interp([1.0], [0.0, 1.0], ndim=1).prepare(),
    ],
)
def test_validation(constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor()


def test_stored_sampling_resolves_inside_an_ordinary_module():
    class Model(zdx.Module):
        value: zdx.Expression

    grid = dl.Grid(n=(5, 3), d=0.1, c=[0.2, 0.0])
    model = Model(CoordinateValue(grid=grid))
    resolved = assert_jittable(lambda value: value.resolve(), model)
    assert isinstance(resolved, Model)
    assert np.allclose(resolved.value, grid.coordinates[0])
    derivative = jax.grad(
        lambda centre: model.set("value.grid.c", centre).resolve().value.sum()
    )(grid.c)
    assert np.allclose(derivative, np.array([15, 0]))


def test_runtime_sampling_and_mapping_replace_stored_definitions():
    stored = dl.Grid(n=(5, 3), d=0.1, c=[0.2, 0.0])
    query = dl.Grid(n=(7, 5), d=0.2, c=[-0.3, 0.0])
    first = dl.Affine(translation=[0.1, 0.0])
    replacement = dl.Affine(translation=[0.4, 0.0])
    expression = CoordinateValue(grid=stored, transform=first)
    assert np.allclose(expression.resolve(), stored.coordinates[0] - 0.1)
    output = assert_jittable(
        lambda value: value.resolve(grid=query, transform=replacement), expression
    )
    assert np.allclose(output, query.coordinates[0] - 0.4)
    assert np.allclose(
        expression.resolve(coordinates=query.coordinates), query.coordinates[0] - 0.1
    )
    assert expression.grid is stored
    assert expression.transform is first


def test_strict_sampling_rejects_runtime_overrides_but_allows_immutable_updates():
    grid = dl.Grid(n=(5, 3), d=0.1)
    expression = CoordinateValue(
        grid=grid, transform=dl.Affine(translation=[0.1, 0.0]), strict=True
    )
    assert np.allclose(expression.resolve(), grid.coordinates[0] - 0.1)
    for overrides in (
        {"grid": grid},
        {"coordinates": grid.coordinates},
        {"transform": dl.Affine()},
    ):
        with pytest.raises(ValueError, match="strict=True"):
            expression.resolve(**overrides)
    updated = expression.set("transform.translation", np.array([0.2, 0.0]))
    result = assert_jittable(lambda value: value.resolve(), updated)
    assert np.allclose(result, grid.coordinates[0] - 0.2)
    # A strict object can still receive inputs it did not store.
    unbound = CoordinateValue(strict=True)
    assert isinstance(unbound.resolve(), CoordinateValue)
    assert np.allclose(unbound(grid), grid.coordinates[0])


def test_fixed_sampled_bases_do_not_gain_sampling_leaves():
    basis = dl.Basis(np.eye(2), [1.0, 2.0])
    assert not isinstance(basis, dl.Sampled)
    assert np.array_equal(basis.resolve(), [1.0, 2.0])
    with pytest.raises(TypeError, match="grid"):
        dl.Basis(np.eye(2), grid=dl.Grid(n=2, d=1))


@pytest.mark.parametrize(
    "context,error",
    [
        ({"grid": 1}, TypeError),
        ({"grid": dlu.Grid(d=1)}, zdx.Unresolved),
        ({"grid": dlu.Grid(n=3)}, zdx.Unresolved),
        ({"coordinates": np.ones((3, 2, 2))}, ValueError),
        ({"coordinates": np.ones((2, 2, 2)), "transform": 1}, TypeError),
    ],
)
def test_sampling_context_validation_tails(context, error):
    """Coverage tails: unresolved sampling and malformed runtime context."""
    with pytest.raises(error):
        CoordinateValue().evaluate(**context)
