"""Sampled interpolation shape, context, and boundary contracts."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux as dl

from tests.helpers import assert_jittable


def test_interpolation_appends_value_axes_after_arbitrary_query_shape():
    samples = np.asarray([[0.0, 10.0], [2.0, 12.0], [4.0, 14.0]])
    interpolation = dl.Interpolation([0.0, 1.0, 2.0], samples)
    query = np.asarray([[0.0, 0.5], [1.0, 1.5]])
    result = assert_jittable(lambda q: interpolation.evaluate(coordinates=q), query)
    assert result.shape == (2, 2, 2)
    assert np.allclose(result, np.stack((2 * query, 10 + 2 * query), axis=-1))
    assert isinstance(interpolation.resolve(), dl.Interpolation)
    assert np.allclose(interpolation.resolve(coordinates=query), result)
    assert isinstance(interpolation.resolve(variables=query), dl.Interpolation)
    assert np.allclose(
        interpolation.resolve(coordinates=query, variables=query + 1), result
    )


def test_interpolation_contextual_query_and_values_have_gradients():
    interpolation = dl.Interpolation([0.0, 1.0, 2.0], zdx.StateRef("samples"))
    state = zdx.State(time=0.25, samples=[0.0, 4.0, 0.0])
    assert np.allclose(
        interpolation.resolve(coordinates=zdx.StateRef("time"), state=state), 1.0
    )
    gradient = jax.grad(
        lambda time: interpolation.resolve(coordinates=time, state=state)
    )(0.25)
    assert np.allclose(gradient, 4.0)
    sample_gradient = jax.jit(
        jax.grad(
            lambda samples: interpolation.resolve(
                coordinates=state.time, state=state.set(samples=samples)
            )
        )
    )(state.samples)
    assert np.allclose(sample_gradient, np.asarray([0.75, 0.25, 0.0]))


def test_interpolation_extrapolation_sides_and_period():
    fill = dl.Interpolation([0.0, 1.0], [0.0, 1.0], extrapolate=(-2.0, 3.0))
    assert np.allclose(
        fill.evaluate(coordinates=[-1.0, 0.5, 2.0]), np.asarray([-2.0, 0.5, 3.0])
    )
    periodic = dl.Interpolation([0.0, 1.0, 2.0], [0.0, 1.0, 0.0], period=2.0)
    assert np.allclose(
        periodic.evaluate(coordinates=[-0.5, 0.5, 2.5]), np.asarray([0.5, 0.5, 0.5])
    )


@pytest.mark.parametrize(
    "knots,values",
    [
        ([0.0], [1.0]),
        ([0.0, 0.0], [1.0, 2.0]),
        ([1.0, 0.0], [1.0, 2.0]),
        ([0.0, np.inf], [1.0, 2.0]),
        ([0.0, 1.0], [1.0]),
    ],
)
def test_interpolation_rejects_invalid_sample_data(knots, values):
    with pytest.raises(ValueError):
        dl.Interpolation(knots, values)
