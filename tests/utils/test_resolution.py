"""Numerical expressions defer missing operands and retain independent progress."""

import jax
import jax.numpy as np
import pytest
import zodiax as zdx

import dLux.utils as dlu
from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    "make, value",
    [
        (lambda x: dlu.Norm(x), [1.0, 2.0]),
        (lambda x: dlu.Norm([1.0, 2.0], weights=x), [1.0, 0.5]),
        (lambda x: dlu.Combine([zdx.Map(x=x), zdx.Map(x=2.0)]), 1.0),
        (lambda x: dlu.Gaussian(x), 0.4),
        (lambda x: dlu.CorrelatedGaussian([0.5, 0.3], x), 0.2),
        (lambda x: dlu.EllipticalGaussian(0.5, angle=x), 0.2),
        (lambda x: dlu.Jitter(dlu.Gaussian(x), kernel_size=3), 0.4),
        (lambda x: dlu.Zernike(nolls=[1, 2, 4], oversize=x), 0.1),
        (lambda x: dlu.Circle(x), 0.8),
        (lambda x: dlu.Square(x), 0.8),
        (lambda x: dlu.Rectangle(0.8, x), 0.5),
        (lambda x: dlu.RegularPolygon(6, x), 0.8),
        (lambda x: dlu.ConvexPolygon(x), [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]),
        (lambda x: dlu.Spider(0.1, x), [0.0, 90.0]),
        (lambda x: dlu.Circle(0.8, edge=dlu.Soft(x)), 1.0),
        (lambda x: dlu.Complement(dlu.Circle(x)), 0.8),
        (lambda x: dlu.ABCD(free_space=x), 0.3),
        (lambda x: dlu.ABCD(lens=x), 0.3),
        (lambda x: dlu.ABCD(radius=x), 0.3),
        (lambda x: dlu.ABCD(power=x), 0.3),
        (lambda x: dlu.ABCD(fraunhofer=x), 0.3),
        (lambda x: dlu.ABCD(x), [[1.0, 0.3], [0.0, 1.0]]),
        (lambda x: dlu.ABCDChain([dlu.ABCD(free_space=x), dlu.ABCD(lens=2)]), 0.3),
    ],
)
def test_missing_operand_defers_until_state_is_available(make, value):
    grid = dlu.Grid(5, diam=1, unit="m")
    expression = make(zdx.StateRef("operand"))

    pending = expression.resolve(grid=grid)
    assert isinstance(pending, type(expression))

    value = np.asarray(value)
    expected = make(value).resolve(grid=grid)
    actual = assert_jittable(
        lambda x: pending.resolve(grid=grid, state=zdx.State(operand=x)), value
    )
    assert np.allclose(actual, expected, atol=1e-6)


def test_partial_combination_retains_resolved_siblings():
    expression = dlu.Combine(
        {"pending": zdx.StateRef("operand"), "ready": zdx.Map(x=2)}
    )
    pending = expression.resolve()
    assert isinstance(pending.parametrics["pending"], zdx.StateRef)
    assert pending.parametrics["ready"] == 2
    assert pending.resolve(state=zdx.State(operand=3)) == 5


def test_resolved_gaussian_width_keeps_its_gradient():
    grid = dlu.Grid(5, diam=1)
    definition = dlu.Gaussian(zdx.StateRef("std"), grid=grid)
    loss = lambda std: definition.resolve(state=zdx.State(std=std)).sum()
    direct = lambda std: dlu.log_gaussian(grid.coordinates, 0.0, std, 2)
    expected = jax.grad(lambda std: np.exp(direct(std)).sum())(0.4)
    actual = assert_jittable(jax.grad(loss), 0.4)
    assert np.allclose(actual, expected)
    assert actual > 0
