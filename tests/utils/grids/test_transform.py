"""Coordinate-dependent operands follow the affine and polynomial sequence."""

import jax
import pytest
import jax.numpy as np
import zodiax as zdx
from jax import Array

import dLux.utils as dlu

from tests.helpers import assert_jittable


class LocalOffset(zdx.Expression):
    scale: Array | zdx.Expression

    def evaluate(self, *, coordinates):
        self = self.resolve_children(coordinates=coordinates)
        offset = self.scale * np.mean(coordinates[0])
        return np.array([[offset], [0.0]])


def test_transform_resolves_polynomial_on_affine_coordinates():
    coordinates = dlu.Grid((3, 2), d=0.2, c=(0.4, -0.1)).coordinates
    polynomial = dlu.Distortion(
        powers=np.zeros((2, 1), dtype=int), distortion=LocalOffset(np.asarray(0.3))
    )
    transform = dlu.Transform(
        affine=dlu.Affine(translation=[0.2, 0.0]), polynomial=polynomial
    )

    expected = coordinates - np.array([0.2, 0.0])[:, None, None]
    expected = expected.at[0].add(0.3 * np.mean(expected[0]))
    result = assert_jittable(lambda value: value(coordinates), transform)
    chain = dlu.TransformChain([transform.affine, polynomial])
    assert np.allclose(result, expected)
    assert np.allclose(result, chain(coordinates))

    def shifted(translation):
        current = transform.set("affine.translation", np.array([translation, 0.0]))
        return current(coordinates)[0].sum()

    derivative = jax.jit(jax.grad(shifted))(np.asarray(0.2))
    assert np.allclose(derivative, -coordinates[0].size * 1.3)
    assert isinstance(transform.polynomial.distortion, LocalOffset)


def test_transform_populates_links_before_applying_its_children():
    coordinates = dlu.Grid((3, 2), d=0.2, c=(0.4, -0.1)).coordinates
    shared = zdx.Linked(2.0, key="scale")
    transform = dlu.Transform(
        affine=dlu.Affine(scale=shared),
        polynomial=dlu.Distortion(
            powers=np.zeros((2, 1), dtype=int),
            distortion=LocalOffset(shared.defer()),
        ),
    )

    expected = coordinates / 2
    expected = expected.at[0].add(np.mean(coordinates[0]))
    result = assert_jittable(lambda value: value(coordinates), transform)
    assert np.allclose(result, expected)
    assert isinstance(transform.affine.scale, zdx.Linked)
    assert isinstance(transform.polynomial.distortion.scale, zdx.Deferred)


def test_bound_transform_chain_resolves_to_coordinates():
    grid = dlu.Grid(3, 1)
    chain = dlu.TransformChain([dlu.Affine(translation=[0.5, 0.0])], grid=grid)
    assert np.allclose(
        chain.resolve(), grid.coordinates - np.array([0.5, 0.0])[:, None, None]
    )


def test_distortion_order_validation_tail():
    """Coverage tail: invalid polynomial order selection."""
    with pytest.raises(ValueError, match="positive integers"):
        dlu.Distortion(orders=(0,))
