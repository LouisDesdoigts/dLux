"""Tests for dLux.utils.polarisation."""

import jax.numpy as np
import pytest

import dLux.utils as dlu

from tests.helpers import assert_jittable


@pytest.mark.parametrize(
    "constructor",
    [
        dlu.horizontal_polariser,
        dlu.vertical_polariser,
        dlu.rhc_polariser,
        dlu.lhc_polariser,
    ],
)
def test_constant_polarisers(constructor):
    output = constructor()
    assert output.shape == (2, 2)
    assert np.allclose(output @ output, output)


@pytest.mark.parametrize(
    "constructor", [dlu.linear_polariser, dlu.quarter_wave_plate, dlu.half_wave_plate]
)
@pytest.mark.parametrize("shape", [(), (3,), (3, 4)])
def test_angular_jones_contract(constructor, shape):
    angle = np.zeros(shape)
    output = assert_jittable(constructor, angle)
    assert output.shape == (2, 2) + shape


def test_retarder_contract():
    retardance = np.ones((3, 4)) * np.pi / 2
    angle = np.zeros((3, 4))
    output = assert_jittable(dlu.retarder, retardance, angle)

    assert output.shape == (2, 2, 3, 4)
    assert np.allclose(output, dlu.quarter_wave_plate(angle))


def test_jones_application():
    phasor = np.ones((2, 2, 4, 6), dtype=complex)
    output = assert_jittable(dlu.apply_jones, dlu.horizontal_polariser(), phasor)

    assert np.allclose(output[0], phasor[0])
    assert np.allclose(output[1], 0)


def test_jones_rotation_and_stokes():
    horizontal = dlu.horizontal_polariser()
    vertical = dlu.rotate_jones(horizontal, np.pi / 2)
    stokes = assert_jittable(dlu.jones_to_stokes, vertical)

    assert np.allclose(vertical, dlu.vertical_polariser(), atol=1e-6)
    assert np.allclose(stokes, np.asarray((0.5, -0.5, 0.0, 0.0)), atol=1e-6)


def test_unrotated_jones_identity():
    jones = dlu.horizontal_polariser()

    assert dlu.rotate_jones(jones, None) is jones
