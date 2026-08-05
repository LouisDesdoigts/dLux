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

    def test_exports_coordinate_helpers(self):
        """FFT coordinate helpers are public dLux utilities."""
        assert "FFT_spec" in propagation_utils.__all__
        assert "FFT_ramp" in propagation_utils.__all__

    def test_spec_even_without_focal_length(self, wavelength, pixel_scale):
        """Even npixels without focal_length returns non-zero offset."""
        d_out, offset = propagation_utils.FFT_spec(16, pixel_scale, wavelength)
        assert np.isfinite(d_out)
        assert offset != 0.0

    def test_spec_odd_without_focal_length(self, wavelength, pixel_scale):
        """Odd npixels without focal_length returns zero offset."""
        d_out, offset = propagation_utils.FFT_spec(15, pixel_scale, wavelength)
        assert np.isfinite(d_out)
        assert offset == 0.0

    def test_spec_with_focal_length(self, wavelength, pixel_scale, focal_length):
        """focal_length scales the output pixel size."""
        d_out_no_fl, _ = propagation_utils.FFT_spec(16, pixel_scale, wavelength)
        d_out_fl, _ = propagation_utils.FFT_spec(
            16, pixel_scale, wavelength, focal_length=focal_length
        )
        assert np.isclose(d_out_fl, d_out_no_fl * focal_length)

    def test_phase_ramp_forward(self, wavelength, shift, pixel_scale):
        """Forward phase ramp has shape (npix, npix) and is complex."""
        xs = np.linspace(-0.5, 0.5, 16) * pixel_scale
        ramp = propagation_utils.FFT_ramp(xs, wavelength, shift[0])
        assert ramp.shape == (16, 16)
        assert np.iscomplexobj(ramp)

    def test_phase_ramp_inverse(self, wavelength, shift, pixel_scale):
        """Inverse phase ramp is conjugate of forward."""
        xs = np.linspace(-0.5, 0.5, 16) * pixel_scale
        ramp_fwd = propagation_utils.FFT_ramp(xs, wavelength, shift[0])
        ramp_inv = propagation_utils.FFT_ramp(xs, wavelength, shift[0], inverse=True)
        assert np.allclose(ramp_fwd, np.conj(ramp_inv))

    def test_phase_ramp_with_focal_length(self, wavelength, shift, pixel_scale):
        """Focal length affects the phase scaling."""
        xs = np.linspace(-0.5, 0.5, 16) * pixel_scale
        ramp_no_fl = propagation_utils.FFT_ramp(xs, wavelength, shift[0])
        ramp_fl = propagation_utils.FFT_ramp(xs, wavelength, shift[0], focal_length=2.0)
        assert not np.allclose(ramp_no_fl, ramp_fl)

    def test_centered_output(self, phasor, wavelength, pixel_scale, focal_length):
        """FFT applies the coordinate ramps when an output centre is requested."""
        result, d_out, c_out = propagation_utils.FFT(
            phasor,
            wavelength,
            pixel_scale,
            center=0.0,
            focal_length=focal_length,
            pad=2,
            output_center=0.0,
        )
        assert result.shape == (2 * phasor.shape[-1], 2 * phasor.shape[-1])
        assert np.ndim(d_out) == 0
        assert c_out == 0.0

    def test_native_output_matches_bare_fft(self, phasor, wavelength, pixel_scale):
        """Without output centering, FFT is the centred padded FFT."""
        pad = 2
        expected = np.pad(phasor, phasor.shape[-1] * (pad - 1) // 2)
        expected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(expected)))
        expected /= expected.shape[-1]

        actual, d_out, center = propagation_utils.FFT(
            phasor, wavelength, pixel_scale, pad=pad
        )
        _, native_center = propagation_utils.FFT_spec(
            phasor.shape[-1] * pad, pixel_scale, wavelength
        )
        assert np.allclose(actual, expected)
        assert center == native_center
        assert np.isclose(d_out, wavelength / (phasor.shape[-1] * pad * pixel_scale))

    def test_requested_output_center_changes_phasor_not_center(
        self, phasor, wavelength, pixel_scale
    ):
        """Requesting a centre applies ramps and returns that centre."""
        native, _, native_center = propagation_utils.FFT(
            phasor, wavelength, pixel_scale, pad=2
        )
        centered, _, center = propagation_utils.FFT(
            phasor, wavelength, pixel_scale, pad=2, output_center=0.0
        )
        assert center == 0.0
        assert native_center != center
        assert not np.allclose(centered, native)

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
        result, _, _ = propagation_utils.FFT(
            phasor,
            wavelength,
            pixel_scale,
            focal_length=focal_length,
            pad=2,
            inverse=inverse,
        )
        assert result.shape == (2 * npixels_in, 2 * npixels_in)

    def test_round_trip_no_padding(self, phasor, wavelength, pixel_scale):
        """Forward then inverse FFT recovers the input when no padding is used."""
        forward, _, _ = propagation_utils.FFT(
            phasor,
            wavelength,
            pixel_scale,
            focal_length=None,
            pad=1,
            inverse=False,
        )
        recovered, _, _ = propagation_utils.FFT(
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
