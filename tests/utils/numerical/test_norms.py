"""Array norm measurements and numerical normalisation."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize("mode", ["sum", "mean", "rms", "l1", "l2", "max", "p2v"])
def test_normalisation_preserves_batches_and_support_measure(mode):
    values = np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    weights = np.asarray([1.0, 1.0, 0.0])
    output = assert_jittable(
        lambda x: dlu.normalise(x, mode=mode, weights=weights, scale=2.0, axis=-1),
        values,
    )
    measures = {
        "sum": 3.0,
        "mean": 1.5,
        "rms": np.sqrt(2.5),
        "l1": 3.0,
        "l2": np.sqrt(5.0),
        "max": 2.0,
        "p2v": 1.0,
    }
    expected = np.asarray([1.0, 2.0, 3.0]) * 2 / measures[mode]
    assert np.allclose(output, np.broadcast_to(expected, values.shape))
    assert np.all(output[:, -1] != 0)


def test_complex_rms_and_l2_use_absolute_squared_amplitudes():
    values = np.asarray([1j, -1j, 2.0])
    assert np.allclose(dlu.l2_norm(values), np.sqrt(6.0))
    assert np.allclose(dlu.rms_norm(values), np.sqrt(2.0))
    rms = dlu.normalise(values, mode="rms")
    assert np.allclose(np.mean(np.abs(rms) ** 2), 1.0)


def test_signed_sum_differs_from_l1_and_normalisation_is_differentiable():
    values = np.asarray([-1.0, 3.0])
    summed = dlu.normalise(values, mode="sum")
    absolute = dlu.normalise(values, mode="l1")
    assert np.allclose(summed, np.asarray([-0.5, 1.5]))
    assert np.allclose(absolute, np.asarray([-0.25, 0.75]))
    evaluate = lambda x: dlu.normalise(zdx.Exp(x).resolve(), mode="sum")
    jacobian = jax.jacrev(evaluate)(values)
    hessian = jax.jacfwd(jax.jacrev(evaluate))(values)
    assert np.all(np.isfinite(hessian))
    assert np.any(np.abs(jacobian) > 0)
    assert np.allclose(jacobian.sum(0), 0.0, atol=1e-6)


def test_normalisation_uses_explicit_weights_and_scale():
    output = assert_jittable(
        lambda scale: dlu.normalise(
            np.asarray([1.0, 2.0, 3.0]),
            mode="sum",
            scale=scale,
            weights=np.asarray([1.0, 0.0, 1.0]),
        ),
        np.asarray(2.0),
    )
    assert np.allclose(output, np.asarray([0.5, 1.0, 1.5]))


@pytest.mark.parametrize("axis", [1, [1], (1,)])
def test_norm_expression_resolves_children_and_preserves_batches(axis):
    values = np.array([[1.0, -2.0, 9.0], [3.0, -4.0, 7.0]])
    model = dlu.Norm(
        zdx.Map(x=values, s=2), "rms", weights=[1.0, 1.0, 0.0], axis=axis, keepdims=True
    )
    result = assert_jittable(lambda m: m.resolve(), model)
    assert np.allclose(result, np.sqrt(np.array([[10.0], [50.0]])))
    assert dlu.Norm(values, "sum").resolve() == values.sum()


def test_signed_peak_and_nan_semantics():
    values = np.array([-4.0, -2.0, 99.0])
    assert dlu.norm(values, "peak", weights=[1, 1, 0]) == -2
    assert dlu.norm(values, "max", weights=[1, 1, 0]) == 4
    assert np.isnan(dlu.norm(np.array([1.0, np.nan]), "peak"))
    assert dlu.norm(np.array([1.0, np.nan]), "l1") == 1


@pytest.mark.parametrize(
    "operation, error",
    [
        (lambda: dlu.norm([1.0], "unknown"), ValueError),
        (lambda: dlu.Norm([1.0], "unknown"), ValueError),
        (lambda: dlu.norm([1.0], weights=[1j]), TypeError),
        (lambda: dlu.norm([1j], "peak"), TypeError),
        (lambda: dlu.norm([1j], "p2v"), TypeError),
        (lambda: dlu.normalise(np.ones(2), scale=np.ones((2, 2))), ValueError),
    ],
)
def test_norm_validation_tails(operation, error):
    """Coverage tails: invalid reductions, complex support and shape-changing scale."""
    with pytest.raises(error):
        operation()
