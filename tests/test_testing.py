"""Tests for the shared testing infrastructure."""

import jax.numpy as np
import pytest

from .helpers import (
    assert_differentiable,
    assert_finite_tree,
    assert_jittable,
)


def test_finite_tree_rejects_nan_and_infinity():
    with pytest.raises(AssertionError, match="NaN"):
        assert_finite_tree({"value": np.asarray(np.nan)})
    with pytest.raises(AssertionError, match="Non-finite"):
        assert_finite_tree({"value": np.asarray(np.inf)})


def test_jittable_contract():
    output = assert_jittable(lambda value: {"value": value**2}, np.arange(3.0))
    assert_finite_tree(output)


def test_differentiable_contract():
    gradient = assert_differentiable(lambda value: value**2, np.arange(3.0))
    assert_finite_tree(gradient)


def test_shared_factories(make_grid, make_wavefront, make_psf):
    grid = make_grid()
    assert_finite_tree(grid)
    assert_finite_tree(make_wavefront(grid=grid))
    assert_finite_tree(make_wavefront(grid=grid, polarised=True))
    assert_finite_tree(make_psf(grid=grid))
