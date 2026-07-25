"""Tests for dLux.utils.abcd."""

import jax.numpy as np
import pytest

import dLux.utils as dlu


@pytest.mark.parametrize(
    "matrix",
    [
        dlu.abcd_lens(2.0),
        dlu.abcd_mirror(2.0),
        dlu.abcd_free_space(2.0),
        dlu.abcd_fraunhofer(2.0),
    ],
)
def test_matrix_contract(matrix):
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert np.allclose(dlu.abcd_unimodularity(matrix), 1)


def test_composition_and_classification():
    free_space = dlu.abcd_free_space(2.0)
    lens = dlu.abcd_lens(3.0)
    system = dlu.compose_abcd([free_space, lens])
    assert system.shape == (2, 2)
    assert dlu.is_free_space(free_space)
    assert dlu.is_surface(lens)
