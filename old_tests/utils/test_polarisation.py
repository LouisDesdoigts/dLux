from jax import numpy as np, config
import pytest

config.update("jax_debug_nans", True)

import dLux.utils as utils

SHAPES = [(), (3,), (3, 4)]


def angles(shape):
    if shape == ():
        return np.array(0.0)
    return np.linspace(0.0, np.pi / 2, int(np.prod(np.array(shape)))).reshape(shape)


def trailing_ones(shape):
    return np.ones(shape, dtype=complex)


def assert_jones_shape(jones, shape):
    assert jones.shape == (2, 2) + shape


def expand_jones(jones, shape):
    return jones.reshape((2, 2) + (1,) * len(shape)) * trailing_ones(shape)


class TestSimplePolarisers:
    def test_horizontal_polariser(self):
        expected = np.array([[1, 0], [0, 0]])
        actual = utils.horizontal_polariser()
        assert actual.shape == (2, 2)
        assert np.allclose(actual, expected)

    def test_vertical_polariser(self):
        expected = np.array([[0, 0], [0, 1]])
        actual = utils.vertical_polariser()
        assert actual.shape == (2, 2)
        assert np.allclose(actual, expected)

    def test_rhc_polariser(self):
        expected = np.array([[0.5, -0.5j], [0.5j, 0.5]])
        actual = utils.rhc_polariser()
        assert actual.shape == (2, 2)
        assert np.allclose(actual, expected)

    def test_lhc_polariser(self):
        expected = np.array([[0.5, 0.5j], [-0.5j, 0.5]])
        actual = utils.lhc_polariser()
        assert actual.shape == (2, 2)
        assert np.allclose(actual, expected)


class TestPolariserConstructors:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_linear_polariser_shape(self, shape):
        actual = utils.linear_polariser(angles(shape))
        assert_jones_shape(actual, shape)

    def test_linear_polariser_basis_angles(self):
        assert np.allclose(utils.linear_polariser(0.0), utils.horizontal_polariser())
        assert np.allclose(
            utils.linear_polariser(np.pi / 2), utils.vertical_polariser(), atol=1e-6
        )

    @pytest.mark.parametrize("shape", [(3,), (3, 4)])
    def test_linear_polariser_trailing_values(self, shape):
        angle = np.zeros(shape)
        actual = utils.linear_polariser(angle)
        expected = expand_jones(utils.horizontal_polariser(), shape)
        assert np.allclose(actual, expected)


class TestRetarderConstructors:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_quarter_wave_plate_shape(self, shape):
        actual = utils.quarter_wave_plate(angles(shape))
        assert_jones_shape(actual, shape)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_half_wave_plate_shape(self, shape):
        actual = utils.half_wave_plate(angles(shape))
        assert_jones_shape(actual, shape)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_retarder_shape(self, shape):
        actual = utils.retarder(np.ones(shape) * np.pi / 2, angles(shape))
        assert_jones_shape(actual, shape)

    def test_basis_retarders(self):
        assert np.allclose(
            utils.retarder(np.pi / 2, 0.0), utils.quarter_wave_plate(0.0)
        )
        assert np.allclose(utils.retarder(np.pi, 0.0), utils.half_wave_plate(0.0))

    @pytest.mark.parametrize("shape", [(3,), (3, 4)])
    def test_retarder_trailing_values(self, shape):
        actual = utils.retarder(np.ones(shape) * np.pi / 2, np.zeros(shape))
        expected = expand_jones(utils.quarter_wave_plate(0.0), shape)
        assert np.allclose(actual, expected)

    def test_retarder_broadcasts_retardance_and_angle(self):
        actual = utils.retarder(np.pi / 2, np.zeros((3, 4)))
        expected = expand_jones(utils.quarter_wave_plate(0.0), (3, 4))
        assert actual.shape == (2, 2, 3, 4)
        assert np.allclose(actual, expected)


class TestJonesOperations:
    def test_rotate_jones_none(self):
        jones = utils.horizontal_polariser()
        assert utils.rotate_jones(jones, None) is jones

    @pytest.mark.parametrize("shape", SHAPES)
    def test_rotate_jones_shape(self, shape):
        jones = utils.horizontal_polariser().reshape((2, 2) + (1,) * len(shape))
        jones = jones * trailing_ones(shape)
        actual = utils.rotate_jones(jones, angles(shape))
        assert_jones_shape(actual, shape)

    def test_rotate_jones_right_angle(self):
        actual = utils.rotate_jones(utils.horizontal_polariser(), np.pi / 2)
        assert np.allclose(actual, utils.vertical_polariser(), atol=1e-6)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_apply_jones_shape(self, shape):
        phasor = np.ones((2, 2) + shape, dtype=complex)
        jones = utils.linear_polariser(np.zeros(shape))
        actual = utils.apply_jones(jones, phasor)
        assert_jones_shape(actual, shape)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_apply_horizontal_jones(self, shape):
        phasor = np.ones((2, 2) + shape, dtype=complex)
        actual = utils.apply_jones(utils.horizontal_polariser(), phasor)
        expected = phasor.at[1].set(0)
        assert np.allclose(actual, expected)


class TestStokes:
    def test_identity(self):
        actual = utils.jones_to_stokes(np.eye(2))
        assert actual.shape == (4,)
        assert np.allclose(actual, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_linear_polarisers(self):
        assert np.allclose(
            utils.jones_to_stokes(utils.horizontal_polariser()),
            np.array([0.5, 0.5, 0.0, 0.0]),
        )
        assert np.allclose(
            utils.jones_to_stokes(utils.vertical_polariser()),
            np.array([0.5, -0.5, 0.0, 0.0]),
        )
        assert np.allclose(
            utils.jones_to_stokes(utils.linear_polariser(np.pi / 4)),
            np.array([0.5, 0.0, 0.5, 0.0]),
            atol=1e-6,
        )

    def test_circular_polarisers(self):
        assert np.allclose(
            utils.jones_to_stokes(utils.rhc_polariser()),
            np.array([0.5, 0.0, 0.0, -0.5]),
        )
        assert np.allclose(
            utils.jones_to_stokes(utils.lhc_polariser()),
            np.array([0.5, 0.0, 0.0, 0.5]),
        )

    def test_input_stokes(self):
        stokes = np.array([1.0, 1.0, 0.0, 0.0])
        actual = utils.jones_to_stokes(utils.horizontal_polariser(), stokes)
        assert np.allclose(actual, stokes)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_jones_to_stokes_shape(self, shape):
        jones = utils.linear_polariser(np.zeros(shape))
        actual = utils.jones_to_stokes(jones)
        assert actual.shape == (4,) + shape

    @pytest.mark.parametrize("shape", [(3,), (3, 4)])
    def test_jones_to_stokes_trailing_values(self, shape):
        jones = utils.linear_polariser(np.zeros(shape))
        actual = utils.jones_to_stokes(jones)
        assert np.allclose(actual[0], 0.5)
        assert np.allclose(actual[1], 0.5)
        assert np.allclose(actual[2:], 0.0)
