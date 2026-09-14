import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import fourier as fourier_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def npixels():
    return 16


@pytest.fixture
def n_modes():
    return 5  # odd, so tests the odd branch


@pytest.fixture
def wavelength():
    return 500e-9


@pytest.fixture
def focal_length():
    return 1.0


@pytest.fixture
def pixel_scale():
    return 10e-6


@pytest.fixture
def shift():
    return 1.0e-6


@pytest.fixture
def coefficients(n_modes):
    return np.ones((n_modes, n_modes))


@pytest.fixture
def kernels(n_modes, npixels):
    return fourier_utils.fourier_kernels(n_modes, npixels)


# ============================================================================
# Tests for _to_xy (private helper)
# ============================================================================
class TestToXY:
    """Tests for integer-or-tuple normalisation."""

    def test_int_input(self):
        """Integer is broadcast to (n, n)."""
        result = fourier_utils._to_xy(8, "test")
        assert result == (8, 8)

    def test_tuple_input(self):
        """Length-2 tuple is returned as-is."""
        result = fourier_utils._to_xy((4, 8), "test")
        assert result == (4, 8)

    def test_list_input(self):
        """Length-2 list is accepted."""
        result = fourier_utils._to_xy([4, 8], "test")
        assert result == (4, 8)

    def test_invalid_type_raises(self):
        """Non-int, non-tuple input raises TypeError."""
        with pytest.raises(TypeError, match="must be an int or length-2 tuple"):
            fourier_utils._to_xy("bad", "test")

    def test_invalid_length_raises(self):
        """Tuple with wrong length raises TypeError."""
        with pytest.raises(TypeError, match="must be an int or length-2 tuple"):
            fourier_utils._to_xy((1, 2, 3), "test")

    def test_nonpositive_raises(self):
        """Zero or negative values raise ValueError."""
        with pytest.raises(ValueError, match="positive integers"):
            fourier_utils._to_xy((0, 4), "test")


# ============================================================================
# Tests for _fourier_mode_order (private helper)
# ============================================================================
class TestFourierModeOrder:
    """Tests for Fourier mode ordering."""

    def test_odd_n_modes(self):
        """Odd n_modes returns correct number of modes."""
        modes = fourier_utils._fourier_mode_order(5)
        assert modes.shape == (5,)
        assert modes[0] == 0  # DC first

    def test_even_n_modes(self):
        """Even n_modes returns correct number of modes."""
        modes = fourier_utils._fourier_mode_order(4)
        assert modes.shape == (4,)
        assert modes[0] == 0  # DC first

    def test_single_mode(self):
        """Single mode (n=1) returns just DC."""
        modes = fourier_utils._fourier_mode_order(1)
        assert modes.shape == (1,)
        assert modes[0] == 0


# ============================================================================
# Tests for fourier_kernel_1d
# ============================================================================
class TestFourierKernel1D:
    """Tests for 1D Fourier kernel computation."""

    def test_output_shape(self, n_modes, npixels):
        """Kernel shape is (npix, n_modes)."""
        kernel = fourier_utils.fourier_kernel_1d(n_modes, npixels)
        assert kernel.shape == (npixels, n_modes)

    def test_real_output(self, n_modes, npixels):
        """Kernel is real-valued."""
        kernel = fourier_utils.fourier_kernel_1d(n_modes, npixels)
        assert kernel.dtype in [np.float32, np.float64]

    def test_even_modes(self, npixels):
        """Works correctly with even n_modes."""
        kernel = fourier_utils.fourier_kernel_1d(4, npixels)
        assert kernel.shape == (npixels, 4)

    def test_scale(self, n_modes, npixels):
        """Scale parameter multiplies the kernel."""
        k1 = fourier_utils.fourier_kernel_1d(n_modes, npixels, scale=1.0)
        k2 = fourier_utils.fourier_kernel_1d(n_modes, npixels, scale=2.0)
        assert np.allclose(k2, 2.0 * k1)


# ============================================================================
# Tests for fourier_kernels
# ============================================================================
class TestFourierKernels:
    """Tests for 2D Fourier kernel computation."""

    def test_int_input(self, n_modes, npixels):
        """Integer n_modes and npix are expanded to (n, n)."""
        Kx, Ky = fourier_utils.fourier_kernels(n_modes, npixels)
        assert Kx.shape == (npixels, n_modes)
        assert Ky.shape == (npixels, n_modes)

    def test_tuple_input(self):
        """Tuple inputs for asymmetric kernels."""
        Kx, Ky = fourier_utils.fourier_kernels((3, 5), (16, 24))
        assert Kx.shape == (16, 3)
        assert Ky.shape == (24, 5)

    def test_scale(self, n_modes, npixels):
        """Scale parameter multiplies both kernels."""
        Kx1, Ky1 = fourier_utils.fourier_kernels(n_modes, npixels, scale=1.0)
        Kx2, Ky2 = fourier_utils.fourier_kernels(n_modes, npixels, scale=2.0)
        assert np.allclose(Kx2, 2.0 * Kx1)
        assert np.allclose(Ky2, 2.0 * Ky1)


# ============================================================================
# Tests for eval_fourier_basis
# ============================================================================
class TestEvalFourierBasis:
    """Tests for 2D Fourier basis evaluation."""

    def test_output_shape(self, coefficients, kernels, npixels, n_modes):
        """Output shape matches (npix_x, npix_y)."""
        Kx, Ky = kernels
        result = fourier_utils.eval_fourier_basis(coefficients, Kx, Ky)
        assert result.shape == (npixels, npixels)

    def test_zero_coefficients(self, kernels, n_modes):
        """Zero coefficients return zero output."""
        Kx, Ky = kernels
        coeffs = np.zeros((n_modes, n_modes))
        result = fourier_utils.eval_fourier_basis(coeffs, Kx, Ky)
        assert np.allclose(result, 0.0)

    def test_asymmetric_kernels(self):
        """Works with asymmetric kernel shapes."""
        Kx, Ky = fourier_utils.fourier_kernels((3, 5), (16, 24))
        coeffs = np.ones((3, 5))
        result = fourier_utils.eval_fourier_basis(coeffs, Kx, Ky)
        assert result.shape == (16, 24)
