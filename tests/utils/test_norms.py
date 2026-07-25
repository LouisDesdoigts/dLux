"""Tests for dLux.utils.norms."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (dlu.l1_norm, 10.0),
        (dlu.l2_norm, np.sqrt(30.0)),
        (dlu.max_norm, 4.0),
        (dlu.rms_norm, np.sqrt(7.5)),
        (dlu.p2v_norm, 7.0),
    ],
)
def test_norm_contract(function, expected):
    array = np.asarray((1.0, -2.0, 3.0, -4.0))
    output = assert_jittable(function, array)

    assert np.isclose(output, expected)


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        (dlu.l1_norm, 3.0),
        (dlu.l2_norm, np.sqrt(5.0)),
        (dlu.max_norm, 2.0),
        (dlu.rms_norm, np.sqrt(2.5)),
        (dlu.p2v_norm, 3.0),
    ],
)
def test_masked_norms(function, expected):
    array = np.asarray((1.0, -2.0, 3.0, -4.0))
    mask = np.asarray((True, True, False, False))
    assert np.isclose(function(array, mask), expected)


def test_axis_and_keepdims():
    output = dlu.l2_norm(np.ones((2, 3, 4)), axis=(1, 2), keepdims=True)
    assert output.shape == (2, 1, 1)


def test_invalid_mask():
    with pytest.raises(ValueError):
        dlu.l1_norm(np.ones(2), np.ones((1, 1)))
