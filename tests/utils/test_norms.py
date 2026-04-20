import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import norms as norms_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def array_1d():
    return np.array([1.0, -2.0, 3.0, -4.0])


@pytest.fixture
def array_2d():
    return np.array([[1.0, -2.0], [3.0, -4.0]])


@pytest.fixture
def mask_1d():
    return np.array([1.0, 1.0, 0.0, 0.0])


@pytest.fixture
def mask_2d():
    return np.array([[1.0, 0.0], [1.0, 0.0]])


# ============================================================================
# Tests for _resolve_mask
# ============================================================================
class TestResolveMask:
    """Tests for the internal mask resolver."""

    def test_none_mask_returns_ones(self, array_2d):
        """None mask returns ones with same shape as input."""
        mask = norms_utils._resolve_mask(array_2d, None)
        assert mask.shape == array_2d.shape
        assert np.all(mask == 1.0)

    def test_lower_dim_mask_is_broadcast(self, array_2d):
        """Mask with fewer dims than array has a leading dimension prepended."""
        mask_1d = np.array([1.0, 0.0])
        mask = norms_utils._resolve_mask(array_2d, mask_1d)
        # expand_dims prepends axes, so (2,) → (1, 2) for a 2D array
        assert mask.ndim == array_2d.ndim

    def test_same_dim_mask_is_returned(self, array_2d, mask_2d):
        """Mask with same dims as array is returned directly."""
        mask = norms_utils._resolve_mask(array_2d, mask_2d)
        assert mask.shape == array_2d.shape

    def test_higher_dim_mask_raises(self, array_1d):
        """Mask with more dims than array raises ValueError."""
        mask_3d = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="more dimensions"):
            norms_utils._resolve_mask(array_1d, mask_3d)


# ============================================================================
# Tests for l1_norm
# ============================================================================
class TestL1Norm:
    """Tests for L1 norm."""

    def test_basic(self, array_1d):
        """L1 norm is sum of absolute values."""
        result = norms_utils.l1_norm(array_1d)
        assert np.isclose(result, 10.0)

    def test_with_mask(self, array_1d, mask_1d):
        """Mask suppresses contributions from masked-out elements."""
        result = norms_utils.l1_norm(array_1d, mask=mask_1d)
        assert np.isclose(result, 3.0)  # |1| + |-2| = 3

    def test_with_axis(self, array_2d):
        """Axis reduces along specified dimension."""
        result = norms_utils.l1_norm(array_2d, axis=0)
        assert result.shape == (2,)

    def test_keepdims(self, array_2d):
        """keepdims preserves reduced axes."""
        result = norms_utils.l1_norm(array_2d, axis=0, keepdims=True)
        assert result.shape == (1, 2)


# ============================================================================
# Tests for l2_norm
# ============================================================================
class TestL2Norm:
    """Tests for L2 norm."""

    def test_basic(self, array_1d):
        """L2 norm is sqrt(sum of squares)."""
        result = norms_utils.l2_norm(array_1d)
        expected = np.sqrt(1.0 + 4.0 + 9.0 + 16.0)
        assert np.isclose(result, expected)

    def test_with_mask(self, array_1d, mask_1d):
        """Mask suppresses contributions from masked-out elements."""
        result = norms_utils.l2_norm(array_1d, mask=mask_1d)
        expected = np.sqrt(1.0 + 4.0)
        assert np.isclose(result, expected)

    def test_with_axis(self, array_2d):
        """Axis reduces along specified dimension."""
        result = norms_utils.l2_norm(array_2d, axis=1)
        assert result.shape == (2,)

    def test_keepdims(self, array_2d):
        """keepdims preserves reduced axes."""
        result = norms_utils.l2_norm(array_2d, axis=1, keepdims=True)
        assert result.shape == (2, 1)


# ============================================================================
# Tests for max_norm
# ============================================================================
class TestMaxNorm:
    """Tests for maximum norm."""

    def test_basic(self, array_1d):
        """max_norm returns the maximum absolute value."""
        result = norms_utils.max_norm(array_1d)
        assert np.isclose(result, 4.0)

    def test_with_mask(self, array_1d, mask_1d):
        """Mask restricts max to unmasked elements."""
        result = norms_utils.max_norm(array_1d, mask=mask_1d)
        assert np.isclose(result, 2.0)  # |-2| is max over first two elements

    def test_with_axis(self, array_2d):
        """Axis reduces along specified dimension."""
        result = norms_utils.max_norm(array_2d, axis=0)
        assert result.shape == (2,)

    def test_keepdims(self, array_2d):
        """keepdims preserves reduced axes."""
        result = norms_utils.max_norm(array_2d, axis=0, keepdims=True)
        assert result.shape == (1, 2)


# ============================================================================
# Tests for rms_norm
# ============================================================================
class TestRMSNorm:
    """Tests for RMS norm."""

    def test_basic(self, array_1d):
        """RMS norm is sqrt(mean of squares)."""
        result = norms_utils.rms_norm(array_1d)
        expected = np.sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
        assert np.isclose(result, expected)

    def test_with_mask(self, array_1d, mask_1d):
        """Mask restricts mean calculation to unmasked elements."""
        result = norms_utils.rms_norm(array_1d, mask=mask_1d)
        expected = np.sqrt((1.0 + 4.0) / 2.0)
        assert np.isclose(result, expected)

    def test_with_axis(self, array_2d):
        """Axis reduces along specified dimension."""
        result = norms_utils.rms_norm(array_2d, axis=0)
        assert result.shape == (2,)

    def test_keepdims(self, array_2d):
        """keepdims preserves reduced axes."""
        result = norms_utils.rms_norm(array_2d, axis=0, keepdims=True)
        assert result.shape == (1, 2)


# ============================================================================
# Tests for p2v_norm
# ============================================================================
class TestP2VNorm:
    """Tests for peak-to-valley norm."""

    def test_basic(self, array_1d):
        """P2V norm is max minus min."""
        result = norms_utils.p2v_norm(array_1d)
        assert np.isclose(result, 7.0)  # 3 - (-4) = 7

    def test_with_mask(self, array_1d, mask_1d):
        """Mask restricts range calculation to unmasked elements."""
        result = norms_utils.p2v_norm(array_1d, mask=mask_1d)
        assert np.isclose(result, 3.0)  # 1 - (-2) = 3

    def test_with_axis(self, array_2d):
        """Axis reduces along specified dimension."""
        result = norms_utils.p2v_norm(array_2d, axis=0)
        assert result.shape == (2,)

    def test_keepdims(self, array_2d):
        """keepdims preserves reduced axes."""
        result = norms_utils.p2v_norm(array_2d, axis=0, keepdims=True)
        assert result.shape == (1, 2)
