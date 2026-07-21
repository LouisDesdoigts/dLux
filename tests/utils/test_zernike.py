import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux import utils
from dLux.utils import zernikes as zernike_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def coordinates():
    return utils.pixel_coords(16, 2)


# ============================================================================
# Tests for metadata helpers
# ============================================================================
class TestZernikeName:
    """Tests for human-readable Zernike naming."""

    def test_known_names(self):
        """Known low-order Noll indices map to the expected names."""
        assert zernike_utils.zernike_name(1) == "Piston"
        assert zernike_utils.zernike_name(2) == "Tilt X"
        assert zernike_utils.zernike_name(3) == "Tilt Y"
        assert zernike_utils.zernike_name(4) == "Defocus"
        assert zernike_utils.zernike_name(5) == "Astig X"
        assert zernike_utils.zernike_name(6) == "Astig Y"
        assert zernike_utils.zernike_name(7) == "Coma X"
        assert zernike_utils.zernike_name(8) == "Coma Y"
        assert zernike_utils.zernike_name(9) == "Trefoil X"
        assert zernike_utils.zernike_name(10) == "Trefoil Y"


class TestNollIndices:
    """Tests for Noll index mapping."""

    def test_known_indices(self):
        """Known Noll indices map to the expected (n, m) pairs."""
        assert zernike_utils.noll_indices(1) == (0, 0)
        assert zernike_utils.noll_indices(2) == (1, 1)
        assert zernike_utils.noll_indices(3) == (1, -1)
        assert zernike_utils.noll_indices(4) == (2, 0)
        assert zernike_utils.noll_indices(5) == (2, -2)
        assert zernike_utils.noll_indices(6) == (2, 2)
        assert zernike_utils.noll_indices(7) == (3, -1)
        assert zernike_utils.noll_indices(8) == (3, 1)
        assert zernike_utils.noll_indices(9) == (3, -3)
        assert zernike_utils.noll_indices(10) == (3, 3)


# ============================================================================
# Tests for polynomial builders
# ============================================================================
class TestZernike:
    """Tests for direct Zernike polynomial evaluation."""

    def test_no_nans(self, coordinates):
        """Evaluating a Zernike polynomial returns finite values."""
        assert not np.isnan(zernike_utils.zernike(5, coordinates)).any()


class TestZernikeBasis:
    """Tests for Zernike basis evaluation."""

    def test_no_nans(self, coordinates):
        """Evaluating a Zernike basis returns finite values."""
        basis = zernike_utils.zernike_basis((1, 2, 3, 4, 5), coordinates)
        assert not np.isnan(basis).any()


class TestPolike:
    """Tests for polygonal Zernike-like polynomials."""

    def test_no_nans(self, coordinates):
        """Evaluating a valid polike returns finite values."""
        assert not np.isnan(zernike_utils.polike(6, 5, coordinates)).any()

    def test_invalid_nsides_raises(self, coordinates):
        """Polikes require at least three polygon sides."""
        with pytest.raises(ValueError):
            zernike_utils.polike(2, 5, coordinates)


class TestPolikeBasis:
    """Tests for polygonal basis generation."""

    def test_no_nans(self, coordinates):
        """Evaluating a polike basis returns finite values."""
        basis = zernike_utils.polike_basis(6, (1, 2, 3, 4, 5), coordinates)
        assert not np.isnan(basis).any()


# ============================================================================
# Tests for factor helpers and fast evaluators
# ============================================================================
class TestZernikeFactors:
    """Tests for precomputed Zernike factors."""

    def test_invalid_index_raises(self):
        """Noll indices start at one."""
        with pytest.raises(ValueError):
            zernike_utils.zernike_factors(0)


class TestZernikeFast:
    """Tests for fast Zernike evaluation from precomputed factors."""

    def test_no_nans(self, coordinates):
        """Fast Zernike evaluation returns finite values."""
        j = 5
        indices = zernike_utils.noll_indices(j)
        factors = zernike_utils.zernike_factors(j)
        result = zernike_utils.zernike_fast(*indices, *factors, coordinates)
        assert not np.isnan(result).any()

    @pytest.mark.parametrize("diameter", [2.0, 1.5])
    def test_matches_zernike(self, coordinates, diameter):
        j = 5
        indices = zernike_utils.noll_indices(j)
        factors = zernike_utils.zernike_factors(j)

        result = zernike_utils.zernike_fast(*indices, *factors, coordinates, diameter)
        expected = zernike_utils.zernike(j, coordinates, diameter)

        assert np.allclose(result, expected)


class TestPolikeFast:
    """Tests for fast polygonal Zernike-like evaluation."""

    def test_no_nans(self, coordinates):
        """Fast polike evaluation returns finite values for valid polygons."""
        j = 5
        indices = zernike_utils.noll_indices(j)
        factors = zernike_utils.zernike_factors(j)
        result = zernike_utils.polike_fast(6, *indices, *factors, coordinates)
        assert not np.isnan(result).any()

    @pytest.mark.parametrize("diameter", [2.0, 1.5])
    def test_matches_polike(self, coordinates, diameter):
        nsides = 6
        j = 5
        indices = zernike_utils.noll_indices(j)
        factors = zernike_utils.zernike_factors(j)

        result = zernike_utils.polike_fast(
            nsides, *indices, *factors, coordinates, diameter
        )
        expected = zernike_utils.polike(nsides, j, coordinates, diameter)

        assert np.allclose(result, expected)

    def test_invalid_nsides_raises(self, coordinates):
        """Fast polike evaluation rejects polygons with fewer than three sides."""
        j = 5
        indices = zernike_utils.noll_indices(j)
        factors = zernike_utils.zernike_factors(j)
        with pytest.raises(ValueError):
            zernike_utils.polike_fast(2, *indices, *factors, coordinates)
