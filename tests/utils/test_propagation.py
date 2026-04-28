import pytest
from jax import numpy as np, config

config.update("jax_debug_nans", True)

from dLux.utils import propagation as propagation_utils


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def phasor():
    return np.ones((32, 32), dtype=complex)


@pytest.fixture
def wavelength():
    return 1.0


@pytest.fixture
def pixel_scale():
    return 0.1


@pytest.fixture
def focal_length():
    return 2.0


@pytest.fixture
def npixels_in():
    return 32


@pytest.fixture
def npixels_out():
    return 16


@pytest.fixture
def pixel_scale_out():
    return 0.05


@pytest.fixture
def shift():
    return np.array([1.0, -2.0])


# ============================================================================
# Tests for FFT
# ============================================================================
class TestFFT:
    """Tests for FFT-based propagation."""

    @pytest.mark.parametrize("inverse", [True, False])
    def test_output_shape(
        self,
        phasor,
        wavelength,
        pixel_scale,
        focal_length,
        npixels_in,
        inverse,
    ):
        """Padding scales the FFT output dimensions by the pad factor."""
        result, _ = propagation_utils.FFT(
            phasor, wavelength, pixel_scale, focal_length, pad=2, inverse=inverse
        )
        assert result.shape == (2 * npixels_in, 2 * npixels_in)

    def test_round_trip_no_padding(self, phasor, wavelength, pixel_scale):
        """Forward then inverse FFT recovers the input when no padding is used."""
        forward, _ = propagation_utils.FFT(
            phasor,
            wavelength,
            pixel_scale,
            focal_length=None,
            pad=1,
            inverse=False,
        )
        recovered, _ = propagation_utils.FFT(
            forward,
            wavelength,
            pixel_scale,
            focal_length=None,
            pad=1,
            inverse=True,
        )
        assert np.allclose(recovered, phasor, rtol=1e-6, atol=1e-6)


# ============================================================================
# Tests for MFT
# ============================================================================
class TestMFT:
    """Tests for matrix Fourier transform propagation."""

    @pytest.mark.parametrize(
        "focal_length,inverse,pixel",
        [
            (2.0, True, True),
            (2.0, False, True),
            (2.0, True, False),
            (2.0, False, False),
            (None, True, True),
            (None, False, True),
            (None, True, False),
            (None, False, False),
        ],
    )
    def test_output_shape_and_finiteness(
        self,
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        shift,
        focal_length,
        pixel,
        inverse,
    ):
        """MFT returns a finite array with the requested output shape."""
        result = propagation_utils.MFT(
            phasor,
            wavelength,
            pixel_scale,
            npixels_out,
            pixel_scale_out,
            focal_length,
            shift,
            pixel=pixel,
            inverse=inverse,
        )
        assert not np.isnan(result).any()
        assert result.shape == (npixels_out, npixels_out)

    def test_shift_units_are_consistent(
        self,
        phasor,
        wavelength,
        pixel_scale,
        npixels_out,
        pixel_scale_out,
        focal_length,
        shift,
    ):
        """
        Pixel and physical-unit shifts produce the same result when scaled consistently.
        """
        by_pixels = propagation_utils.MFT(
            phasor,
            wavelength,
            pixel_scale,
            npixels_out,
            pixel_scale_out,
            focal_length,
            shift,
            pixel=True,
            inverse=False,
        )
        by_scale = propagation_utils.MFT(
            phasor,
            wavelength,
            pixel_scale,
            npixels_out,
            pixel_scale_out,
            focal_length,
            shift * pixel_scale_out,
            pixel=False,
            inverse=False,
        )
        assert np.allclose(by_pixels, by_scale, rtol=1e-6, atol=1e-6)
