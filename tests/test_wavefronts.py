import jax.numpy as np
from jax import config, Array
import pytest
import dLux

config.update("jax_debug_nans", True)


class TestWavefront(object):
    """
    Test the Wavefront class.
    """


    def test_constructor(self, create_wavefront):
        """
        Tests the constructor.
        """
        # Test constructor
        create_wavefront()

        # Test 1d array
        with pytest.raises(ValueError):
            create_wavefront(wavelength=[1e3])


    def test_npixels(self, create_wavefront):
        """
        Tests the npixels property.
        """
        wf = create_wavefront()
        assert wf.npixels == wf.amplitude.shape[-1]


    def test_diameter(self, create_wavefront):
        """
        Tests the diameter property.
        """
        wf = create_wavefront()
        assert wf.diameter == wf.npixels * wf.pixel_scale


    def test_real(self, create_wavefront):
        """
        Tests the real property.
        """
        wf = create_wavefront()
        assert (wf.real == wf.amplitude * np.cos(wf.phase)).all()


    def test_imaginary(self, create_wavefront):
        """
        Tests the imaginary property.
        """
        wf = create_wavefront()
        assert (wf.imaginary == wf.amplitude * np.sin(wf.phase)).all()


    def test_phasor(self, create_wavefront):
        """
        Tests the phasor property.
        """
        wf = create_wavefront()
        assert (wf.phasor == wf.amplitude * np.exp(1j*wf.phase)).all()


    def test_psf(self, create_wavefront):
        """
        Tests the psf property.
        """
        wf = create_wavefront()
        assert (wf.psf == wf.amplitude**2).all()


    def test_coordinates(self, create_wavefront):
        """
        Tests the coordinates property.
        """
        wf = create_wavefront()
        wf.coordinates


    def test_tilt(self, create_wavefront):
        """
        Tests the tilt method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(ValueError):
            wf.tilt("some string")

        # Test wrong shapes
        with pytest.raises(ValueError):
            wf.tilt(np.ones(1))

        # Test wrong shapes
        with pytest.raises(ValueError):
            wf.tilt(np.array(1.))

        # Test wrong shapes
        with pytest.raises(ValueError):
            wf.tilt(np.ones(3))

        # Test basic behaviour
        wf.tilt(np.ones(2))


    def test_normalise(self, create_wavefront):
        """
        Tests the normalise method.
        """
        wf = create_wavefront()

        new_wf = wf.normalise()
        assert np.sum(new_wf.amplitude**2) == 1.


    def test_invert_x_and_y(self, create_wavefront):
        """
        Tests the invert_x_and_y method.
        """
        wf = create_wavefront()
        wf = wf.flip(0)
        wf = wf.flip(1)
        wf = wf.flip((0, 1))



    def test_interpolate(self, create_wavefront):
        """
        Tests the interpolate method.
        """
        k = 2

        wf = create_wavefront()
        wf = wf.scale_to(wf.npixels//k, k * wf.pixel_scale)
        wf = wf.scale_to(wf.npixels//k, k * wf.pixel_scale, complex=True)


    def test_rotate(self, create_wavefront):
        """
        Tests the rotate method.
        """
        wf = create_wavefront()
        wf = dLux.CircularAperture(1.)(wf)
        flipped_amplitude = np.flip(wf.amplitude, axis=(-1, -2))
        flipped_phase = np.flip(wf.phase, axis=(-1, -2))

        new_wf = wf.rotate(np.pi, order=1)
        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        assert np.allclose(new_wf.phase, flipped_phase)

        new_wf = wf.rotate(np.pi, real_imaginary=True, order=1)

        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        # Add small remainer to fix 0-pi instability
        assert np.allclose((new_wf.phase+1e-6)%np.pi, flipped_phase, atol=1e-5)


    def test_pad_to(self, create_wavefront):
        """
        Tests the pad_to method.
        """
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
        """
        Tests the crop_to method.
        """
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
