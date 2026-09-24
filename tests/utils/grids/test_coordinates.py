"""Cartesian coordinate arrays use physical components before spatial axes."""

import jax.numpy as np
import dLux.utils as dlu
from tests.helpers import assert_jittable


def test_coordinate_system_roundtrip():
    coordinates = dlu.Grid((6, 4), (0.1, 0.2)).coordinates
    coordinates = np.stack((coordinates, coordinates + 0.1))
    polar = assert_jittable(dlu.cart2polar, coordinates)
    assert np.allclose(dlu.polar2cart(polar), coordinates, atol=1e-6)


def test_coordinate_grid_broadcasts_independent_axis_batches():
    x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])[:, None, :]
    y = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    coordinates = assert_jittable(dlu.coordinate_grid, (x, y))
    assert coordinates.shape == (2, 3, 2, 2, 3)
    for i in range(2):
        for j in range(3):
            assert np.array_equal(
                coordinates[i, j, 0], np.broadcast_to(x[i, 0], (2, 3))
            )
            assert np.array_equal(
                coordinates[i, j, 1], np.broadcast_to(y[j, :, None], (2, 3))
            )
