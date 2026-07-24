import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import units as units_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def values():
    return np.array([0.0, np.pi / 4, np.pi / 2])


# ============================================================================
# Tests for direct unit conversions
# ============================================================================
class TestAngleConversions:
    """Tests for direct angular unit conversion helpers."""

    def test_rad2arcsec(self, values):
        """Radians convert to arcseconds using the standard scale factor."""
        result = units_utils.rad2arcsec(values)
        expected = values * 3600 * 180 / np.pi
        assert np.allclose(result, expected)

    def test_rad2deg(self, values):
        """Radians convert to degrees using the standard scale factor."""
        result = units_utils.rad2deg(values)
        expected = values * 180 / np.pi
        assert np.allclose(result, expected)

    def test_rad2arcmin(self, values):
        """Radians convert to arcminutes using the standard scale factor."""
        result = units_utils.rad2arcmin(values)
        expected = values * 60 * 180 / np.pi
        assert np.allclose(result, expected)

    def test_deg2rad(self, values):
        """Degrees convert to radians using the standard scale factor."""
        result = units_utils.deg2rad(values)
        expected = values * np.pi / 180
        assert np.allclose(result, expected)

    def test_deg2arcmin(self, values):
        """Degrees convert to arcminutes by multiplying by 60."""
        result = units_utils.deg2arcmin(values)
        expected = values * 60
        assert np.allclose(result, expected)

    def test_deg2arcsec(self, values):
        """Degrees convert to arcseconds by multiplying by 3600."""
        result = units_utils.deg2arcsec(values)
        expected = values * 3600
        assert np.allclose(result, expected)

    def test_arcmin2rad(self, values):
        """Arcminutes convert to radians using the inverse scale factor."""
        result = units_utils.arcmin2rad(values)
        expected = values * np.pi / (60 * 180)
        assert np.allclose(result, expected)

    def test_arcmin2deg(self, values):
        """Arcminutes convert to degrees by dividing by 60."""
        result = units_utils.arcmin2deg(values)
        expected = values / 60
        assert np.allclose(result, expected)

    def test_arcmin2arcsec(self, values):
        """Arcminutes convert to arcseconds by multiplying by 60."""
        result = units_utils.arcmin2arcsec(values)
        expected = values * 60
        assert np.allclose(result, expected)

    def test_arcsec2rad(self, values):
        """Arcseconds convert to radians using the inverse scale factor."""
        result = units_utils.arcsec2rad(values)
        expected = values * np.pi / (3600 * 180)
        assert np.allclose(result, expected)

    def test_arcsec2deg(self, values):
        """Arcseconds convert to degrees by dividing by 3600."""
        result = units_utils.arcsec2deg(values)
        expected = values / 3600
        assert np.allclose(result, expected)

    def test_arcsec2arcmin(self, values):
        """Arcseconds convert to arcminutes by dividing by 60."""
        result = units_utils.arcsec2arcmin(values)
        expected = values / 60
        assert np.allclose(result, expected)


# ============================================================================
# Tests for unit_factor_to_rad
# ============================================================================
class TestUnitFactorToRad:
    """Tests for canonical angular unit resolution."""

    def test_non_string_raises(self):
        """Non-string unit names raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            units_utils.unit_factor_to_rad(123)

    def test_empty_raises(self):
        """Empty unit names raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            units_utils.unit_factor_to_rad("")

    def test_prefixed_unit(self):
        """SI-prefixed angular units resolve to the correct factor."""
        factor = units_utils.unit_factor_to_rad("mrad")
        assert np.isclose(factor, 1e-3)

    def test_unknown_unit_raises(self):
        """Unknown unit names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown angular unit"):
            units_utils.unit_factor_to_rad("xyz")
