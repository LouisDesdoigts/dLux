"""Public Zodiax expression interoperability for dLux numerical recipes."""

from io import BytesIO

import equinox as eqx
import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl
import dLux.utils as dlu

from tests.helpers import assert_jittable


class _Model(zdx.Module):
    coefficients: object
    spectrum: object
    polynomial: object
    response: object

    def __init__(self):
        self.coefficients = zdx.Linked(
            zdx.Map(np.asarray([1.0, 2.0]), s=zdx.StateRef("scale")), key="coeffs"
        )
        self.spectrum = dlu.Basis(np.eye(2), self.coefficients.defer(), axes=1)
        self.polynomial = dl.Polynomial(1, self.coefficients.defer())
        self.response = dl.Interp(
            array=[0.0, 2.0, 0.0], grid_in=[0.0, 1.0, 2.0], ndim=1, method="linear"
        )


def test_complete_model_resolves_linked_recipes_and_explicit_state():
    model = _Model()
    state = zdx.State(scale=2.0, query=0.5)
    coordinates = np.asarray([0.0, 1.0, 2.0])
    resolved = assert_jittable(
        lambda value: value.resolve(state=state, coordinates=coordinates), model
    )
    assert np.allclose(resolved.spectrum, np.asarray([2.0, 4.0]))
    assert np.allclose(resolved.polynomial, 2 + 4 * coordinates)
    assert np.allclose(resolved.response, np.asarray([0.0, 2.0, 0.0]))
    assert isinstance(model.coefficients, zdx.Linked)
    assert isinstance(model.polynomial.coeffs, zdx.Deferred)

    partial = model.resolve(state=state)
    assert isinstance(partial.polynomial, dl.Polynomial)
    assert isinstance(partial.response, dl.Interp)
    derivative = jax.grad(
        lambda scale: model.resolve(
            state=zdx.State(scale=scale, query=0.5), coordinates=coordinates
        ).polynomial.sum()
    )(2.0)
    assert np.allclose(derivative, 9.0)


def test_nested_transform_initialisation_is_explicit():
    basis = dl.Basis(np.eye(3), zdx.Exp(np.asarray([0.0, 0.5, 1.0])), axes=1)
    values = np.asarray([1.0, 2.0, 4.0])
    physical = basis.solve(values)
    initialised = basis.set(coeffs=basis.coeffs.initialise(physical))
    assert np.allclose(initialised.resolve(), values)
    assert np.allclose(initialised.coeffs.x, np.log(values))
    assert np.allclose(basis.coeffs.x, np.asarray([0.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    "model,context",
    [
        (dl.Basis(np.eye(2), zdx.Map([1.0, 2.0], s=2.0), axes=1), {}),
        (dl.Polynomial(2, [1.0, 2.0, 3.0]), {"coordinates": np.asarray([0.0, 1.0])}),
        (
            dl.Interp(array=[2.0, 4.0], grid_in=[0.0, 1.0], ndim=1, method="linear"),
            {"coordinates": [0.25, 0.75]},
        ),
        (dlu.Basis(np.eye(2), [1.0, 2.0]), {}),
    ],
)
def test_dlux_parameterisation_archive_roundtrip(model, context):
    file = BytesIO()
    zdx.save(file, model)
    file.seek(0)
    restored = zdx.load(file)
    assert eqx.tree_equal(model, restored, typematch=True)
    assert np.allclose(restored.resolve(**context), model.resolve(**context))
