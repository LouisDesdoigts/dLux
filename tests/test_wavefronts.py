import jax.numpy as np
from jax import config
import pytest
import dLux

config.update("jax_debug_nans", True)


class TestWavefront:
    """Test the Wavefront class."""

    def test_constructor(self, create_wavefront):
        """Tests the constructor."""
        # Test constructor
        create_wavefront()

        # Test 1d array
        with pytest.raises(ValueError):
            create_wavefront(wavelength=[1e3])

        with pytest.raises(ValueError):
            create_wavefront(diameter=np.ones(2))

    def test_npixels(self, create_wavefront):
        """Tests the npixels property."""
        wf = create_wavefront()
        assert wf.npixels == wf.amplitude.shape[-1]

    def test_diameter(self, create_wavefront):
        """Tests the diameter property."""
        wf = create_wavefront()
        assert wf.diameter == wf.npixels * wf.pixel_scale

    def test_real(self, create_wavefront):
        """Tests the real property."""
        wf = create_wavefront()
        assert (wf.real == wf.amplitude * np.cos(wf.phase)).all()

    def test_imaginary(self, create_wavefront):
        """Tests the imaginary property."""
        wf = create_wavefront()
        assert (wf.imaginary == wf.amplitude * np.sin(wf.phase)).all()

    def test_phasor(self, create_wavefront):
        """Tests the phasor property."""
        wf = create_wavefront()
        assert (wf.phasor == wf.amplitude * np.exp(1j * wf.phase)).all()

    def test_psf(self, create_wavefront):
        """Tests the psf property."""
        wf = create_wavefront()
        assert (wf.psf == wf.amplitude**2).all()

    def test_coordinates(self, create_wavefront):
        """Tests the coordinates property."""
        wf = create_wavefront()
        wf.coordinates

    def test_add_phase(self, create_wavefront):
        """Tests the add_phase method."""
        wf = create_wavefront()
        wf.add_phase(np.ones(wf.npixels))
        wf.add_phase(None)

    def test_tilt(self, create_wavefront):
        """Tests the tilt method."""
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(ValueError):
            wf.tilt("some string")

        # Test wrong shapes
        with pytest.raises(ValueError):
            wf.tilt(np.ones(1))

        # Test wrong shapes
        with pytest.raises(ValueError):
            wf.tilt(np.ones(3))

        # Test basic behaviour
        wf.tilt(np.ones(2))

    def test_normalise(self, create_wavefront):
        """Tests the normalise method."""
        wf = create_wavefront()
        new_wf = wf.normalise()
        assert np.sum(new_wf.amplitude**2) == 1.0

    def test_flip(self, create_wavefront):
        """Tests the flip method."""
        wf = create_wavefront()
        wf = wf.flip(0)
        wf = wf.flip(1)
        wf = wf.flip((0, 1))

    def test_interpolate(self, create_wavefront):
        """Tests the interpolate method."""
        k = 2
        wf = create_wavefront()
        wf = wf.scale_to(wf.npixels // k, k * wf.pixel_scale)
        wf = wf.scale_to(wf.npixels // k, k * wf.pixel_scale, complex=True)

    def test_rotate(self, create_wavefront):
        """Tests the rotate method."""
        wf = create_wavefront()
        wf = dLux.CircularAperture(1.0)(wf)
        flipped_amplitude = np.flip(wf.amplitude, axis=(-1, -2))
        flipped_phase = np.flip(wf.phase, axis=(-1, -2))

        new_wf = wf.rotate(np.pi, order=1)
        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        assert np.allclose(new_wf.phase, flipped_phase)

        new_wf = wf.rotate(np.pi, complex=True, order=1)

        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        # Add small remainer to fix 0-pi instability
        assert np.allclose(
            (new_wf.phase + 1e-6) % np.pi, flipped_phase, atol=1e-5
        )

    def test_pad_to(self, create_wavefront):
        """Tests the pad_to method."""
        even_wf = create_wavefront(npixels=16)
        odd_wf = create_wavefront(npixels=15)

        # Smaller value
        with pytest.raises(ValueError):
            even_wf.pad_to(14)

        # even -> odd
        with pytest.raises(ValueError):
            even_wf.pad_to(17)

        # odd -> even
        with pytest.raises(ValueError):
            odd_wf.pad_to(16)

        assert even_wf.pad_to(20).npixels == 20

    def test_crop_to(self, create_wavefront):
        """Tests the crop_to method."""
        even_wf = create_wavefront(npixels=16)
        odd_wf = create_wavefront(npixels=15)

        # Smaller value
        with pytest.raises(ValueError):
            even_wf.crop_to(18)

        # even -> odd
        with pytest.raises(ValueError):
            even_wf.crop_to(15)

        # odd -> even
        with pytest.raises(ValueError):
            odd_wf.crop_to(14)

        assert even_wf.crop_to(12).npixels == 12

    def test_magic(self, create_wavefront, create_optic):
        """Tests the magic methods."""
        wf = create_wavefront()
        wf += create_optic()

        wf *= None
        wf *= wf.phasor

        with pytest.raises(TypeError):
            create_wavefront() + "a string"

        with pytest.raises(TypeError):
            create_wavefront() * "a string"

    def test_FFT(self, create_wavefront):
        """Tests the FFT method."""
        wf = create_wavefront()

        with pytest.raises(ValueError):
            wf = wf.set("units", "Angular")
            wf.FFT(focal_length=1.0)

        with pytest.raises(ValueError):
            wf.IFFT()

        with pytest.raises(ValueError):
            wf = wf.set("plane", "Focal")
            wf.FFT()

    def test_MFT(self, create_wavefront):
        """Tests the MFT method."""
        wf = create_wavefront()

        with pytest.raises(ValueError):
            wf = wf.set("units", "Angular")
            wf.MFT(16, 1 / 16, focal_length=1.0)

        with pytest.raises(ValueError):
            wf.IMFT(
                16,
                1 / 16,
            )

        with pytest.raises(ValueError):
            wf = wf.set("plane", "Focal")
            wf.MFT(16, 1 / 16)


class TestFresnelWavefront:
    """Test the FresnelWavefront class."""

    def test_constructor(self, create_wavefront):
        """Tests the constructor."""
        # Test constructor
        create_wavefront()

        # Test 1d array
        with pytest.raises(ValueError):
            create_wavefront(wavelength=[1e3])

        with pytest.raises(ValueError):
            create_wavefront(diameter=np.ones(2))

    def test_fresnel_prop(self, create_fresnel_wavefront):
        """Tests the fresnel_prop method."""
        wf = create_fresnel_wavefront()
        wf.fresnel_prop(16, 1 / 16, 1e2, 1e-2)

        with pytest.raises(ValueError):
            wf = wf.set("plane", "not Pupil")
            wf.fresnel_prop(16, 1 / 16, 1e2, 1e-2)
