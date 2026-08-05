"""Tests for dLux.utils.source."""

import jax.numpy as np

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_fluxes_from_contrast():
    output = assert_jittable(dlu.fluxes_from_contrast, 2.0, 0.25)
    assert np.allclose(output, 2 * np.asarray((0.5, 2.0)) / 1.25)


def test_positions_from_separation():
    center = np.asarray((0.1, -0.2))
    output = assert_jittable(
        dlu.positions_from_sep,
        center,
        2.0,
        np.pi / 4,
    )

    assert np.allclose(output.mean(0), center)
    assert np.isclose(np.linalg.norm(output[0] - output[1]), 2.0)
