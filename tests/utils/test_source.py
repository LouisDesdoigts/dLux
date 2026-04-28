import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import source as source_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def mean_flux():
    return 1.0


@pytest.fixture
def contrast():
    return 0.5


@pytest.fixture
def position():
    return np.array([0.0, 0.0])


@pytest.fixture
def separation():
    return 1.0


@pytest.fixture
def position_angle():
    return np.pi / 4


# ============================================================================
# Tests for fluxes_from_contrast
# ============================================================================
class TestFluxesFromContrast:
    """Tests for binary-source flux splitting."""

    def test_formula(self, mean_flux, contrast):
        """Returned fluxes follow the documented contrast formula."""
        result = source_utils.fluxes_from_contrast(mean_flux, contrast)
        expected = 2 * np.array([contrast * mean_flux, mean_flux]) / (1 + contrast)
        assert np.allclose(result, expected)


# ============================================================================
# Tests for positions_from_sep
# ============================================================================
class TestPositionsFromSep:
    """Tests for binary-source sky-position construction."""

    def test_formula(self, position, separation, position_angle):
        """
        Returned positions are centered on the input position and separated correctly.
        """
        result = source_utils.positions_from_sep(position, separation, position_angle)
        r, phi = separation / 2, position_angle
        sep_vec = np.array([r * np.sin(phi), r * np.cos(phi)])
        expected = np.array([position + sep_vec, position - sep_vec])
        assert np.allclose(result, expected)
