"""Tests for grid-aware optical-component builders."""

import jax.numpy as np
import jax.random as jr

import dLux as dl

from tests.helpers import assert_jittable


def test_aperture_builder_contract():
    grid = dl.GridSpec(n=32, d=0.1, unit="m")
    builder = dl.SimpleCircular(
        diameter=2.0,
        secondary_diameter=0.4,
        opd=dl.ZernikeDef(orders=1),
        oversample=2,
    )

    transmission, basis, support = builder.build(grid, return_support=True)
    optic = builder(grid, key=jr.key(0))
    plain = dl.SimpleCircular(diameter=2.0, oversample=2)
    assert_jittable(lambda value: value(grid), plain)

    assert transmission.shape == support.shape == (32, 32)
    assert basis.shape == (2, 32, 32)
    assert np.all(basis[:, ~support] == 0)
    assert optic.transmission.shape == (32, 32)
    assert optic.opd.coefficients.shape == (2,)
    assert optic.normalise
    assert not builder(grid, normalise=False).normalise


def test_sparse_builder_contract():
    grid = dl.GridSpec(n=24, d=0.1, unit="m")
    builder = dl.NRMLike(
        centers=[[-0.5, 0.0], [0.5, 0.0]],
        hole=dl.Circle(0.4),
        opd=dl.ZernikeDef(nolls=1),
        oversample=2,
    )

    global_optic = builder(grid)
    sparse_optic = builder(grid, sparse=True, coefficients=[[0.1], [-0.1]])

    assert isinstance(global_optic, dl.Optic)
    assert isinstance(sparse_optic, dl.SparseOptic)
    assert global_optic.normalise and sparse_optic.normalise
    assert sparse_optic.centers.shape == (2, 2)
    assert sparse_optic.opd.coefficients.shape == (2, 1)
