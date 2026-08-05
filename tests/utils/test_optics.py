"""Tests for dLux.utils.optics."""

import jax.numpy as np

import dLux.utils as dlu

from tests.helpers import assert_jittable


def test_phase_opd_roundtrip():
    opd = np.asarray((1e-9, 2e-9, 3e-9))
    phase = assert_jittable(dlu.opd2phase, opd, 500e-9)

    assert np.allclose(dlu.phase2opd(phase, 500e-9), opd)
    assert np.isclose(dlu.wavenumber(500e-9), 2 * np.pi / 500e-9)


def test_fringe_size():
    assert np.isclose(dlu.fringe_size(1e-6, 2.0), 0.5e-6)
    assert np.isclose(dlu.fringe_size(1e-6, 2.0, 10.0), 5e-6)


def test_tilt_contract():
    coordinates = dlu.nd_coords((6, 4), (0.1, 0.2))
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
