import pytest
import jax.numpy as np
from dLux.utils.optics import wavenumber, opd2phase, phase2opd, fringe_size


@pytest.fixture
def wavelength():
    return 1.0


@pytest.fixture
def opd():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def phase():
    return np.array([np.pi / 2, np.pi, 3 * np.pi / 2])


@pytest.fixture
def diameter():
    return 1.0


@pytest.fixture
def focal_length():
    return 2.0


def test_wavenumber(wavelength):
    result = wavenumber(wavelength)
    assert result == 2 * np.pi / wavelength


def test_opd2phase(opd, wavelength):
    result = opd2phase(opd, wavelength)
    expected = wavenumber(wavelength) * opd
    assert np.allclose(result, expected)


def test_phase2opd(phase, wavelength):
    result = phase2opd(phase, wavelength)
    expected = phase / wavenumber(wavelength)
    assert np.allclose(result, expected)


def test_fringe_size_no_focal_length(wavelength, diameter):
    result = fringe_size(wavelength, diameter)
    assert result == wavelength / diameter


def test_fringe_size_with_focal_length(wavelength, diameter, focal_length):
    result = fringe_size(wavelength, diameter, focal_length)
    assert result == wavelength * focal_length / diameter
