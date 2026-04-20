import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import optics as optics_utils


# ============================================================================
# Fixtures
# ============================================================================
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


# ============================================================================
# Tests for wavenumber
# ============================================================================
class TestWavenumber:
    """Tests for scalar wavelength-to-wavenumber conversion."""

    def test_formula(self, wavelength):
        """Wavenumber equals 2π divided by wavelength."""
        result = optics_utils.wavenumber(wavelength)
        assert result == 2 * np.pi / wavelength


# ============================================================================
# Tests for opd2phase
# ============================================================================
class TestOPD2Phase:
    """Tests for optical path difference to phase conversion."""

    def test_formula(self, opd, wavelength):
        """Phase equals wavenumber times OPD."""
        result = optics_utils.opd2phase(opd, wavelength)
        expected = optics_utils.wavenumber(wavelength) * opd
        assert np.allclose(result, expected)


# ============================================================================
# Tests for phase2opd
# ============================================================================
class TestPhase2OPD:
    """Tests for phase to optical path difference conversion."""

    def test_formula(self, phase, wavelength):
        """OPD equals phase divided by wavenumber."""
        result = optics_utils.phase2opd(phase, wavelength)
        expected = phase / optics_utils.wavenumber(wavelength)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for fringe_size
# ============================================================================
class TestFringeSize:
    """Tests for diffraction fringe size calculation."""

    def test_without_focal_length(self, wavelength, diameter):
        """Without focal length the fringe size is angular."""
        result = optics_utils.fringe_size(wavelength, diameter)
        assert result == wavelength / diameter

    def test_with_focal_length(self, wavelength, diameter, focal_length):
        """With focal length the fringe size is linear in the focal plane."""
        result = optics_utils.fringe_size(wavelength, diameter, focal_length)
        assert result == wavelength * focal_length / diameter
