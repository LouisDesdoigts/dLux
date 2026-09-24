"""Tests for dLux.utils.optical.phase."""

import jax
import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_phase_opd_roundtrip():
    opd = np.asarray((1e-9, 2e-9, 3e-9))
    phase = assert_jittable(dlu.opd2phase, opd, 500e-9)

    assert np.allclose(dlu.phase2opd(phase, 500e-9) / 1e-9, opd / 1e-9, atol=1e-6)
    assert np.isclose(dlu.wavenumber(500e-9), 2 * np.pi / 500e-9)


def test_fringe_size():
    assert np.isclose(dlu.fringe_size(1e-6, 2.0), 0.5e-6)
    assert np.isclose(dlu.fringe_size(1e-6, 2.0, 10.0), 5e-6)


def test_tilt_contract():
    coordinates = dlu.Grid((6, 4), d=(0.1, 0.2)).coordinates
    field = np.ones((4, 6), dtype=complex)
    output = assert_jittable(
        dlu.tilt,
        field,
        coordinates,
        np.asarray((0.1, -0.2)),
        0.5,
    )

    assert np.allclose(np.abs(output), 1.0)
    assert np.allclose(
        np.angle(output),
        dlu.opd2phase(
            dlu.tilt_opd(coordinates, np.asarray((0.1, -0.2))),
            0.5,
        ),
    )


def test_phase_conversion_broadcasts_and_has_the_expected_gradient():
    wavelengths = np.array([0.5, 0.8, 1.2])[:, None, None]
    opd = np.array([[0.02, -0.03], [0.01, 0.04]])
    expected = 2 * np.pi * opd / wavelengths
    result = assert_jittable(dlu.opd2phase, opd, wavelengths)
    assert np.allclose(result, expected)
    assert np.allclose(
        dlu.phase2opd(result, wavelengths), np.broadcast_to(opd, result.shape)
    )
    gradient = jax.grad(lambda value: dlu.opd2phase(value, wavelengths).sum())(opd)
    assert np.allclose(gradient, np.full_like(opd, (2 * np.pi / wavelengths).sum()))


def test_tilt_batches_units_and_phase_sign():
    grid = dlu.Grid((4, 3), d=0.2)
    x, y = grid.coordinates
    angles = np.array([[0.1, -0.2], [-0.3, 0.4]])
    expected = angles[:, 0, None, None] * x + angles[:, 1, None, None] * y
    opd = assert_jittable(dlu.tilt_opd, grid.coordinates, angles)
    assert np.allclose(opd, expected)
    degrees = angles * 180 / np.pi
    assert np.allclose(dlu.tilt_opd(grid.coordinates, degrees, unit="deg"), expected)

    phasor = np.full(grid.shape, 0.5 + 0.2j)
    result = assert_jittable(dlu.tilt, phasor, grid.coordinates, angles, 0.7)
    assert np.allclose(result, phasor * np.exp(2j * np.pi * expected / 0.7))
    restored = dlu.tilt(result, grid.coordinates, -angles, 0.7)
    assert np.allclose(restored, np.broadcast_to(phasor, result.shape), atol=1e-6)


def test_tilt_angle_validation_tail():
    """Coverage tail: malformed angle vectors fail with a useful shape error."""
    with pytest.raises(ValueError, match="final shape"):
        dlu.tilt_opd(dlu.Grid(2, d=1).coordinates, [0.1, 0.2, 0.3])
