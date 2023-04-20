import jax.numpy as np
import pytest
import dLux
from jax import config, Array
config.update("jax_debug_nans", True)


class TestWavefront(object):
    """
    Test the Wavefront class.
    """


    def test_constructor(self, create_wavefront: callable) -> None:
        """
        Tests the constructor.
        """
        # Test constructor
        create_wavefront()

        # Test empty array
        with pytest.raises(AssertionError):
            create_wavefront(wavelength=[])

        # Test 1d array
        with pytest.raises(AssertionError):
            create_wavefront(wavelength=[1e3, 1e4])


    def test_npixels(self, create_wavefront: callable) -> None:
        """
        Tests the npixels property.
        """
        wf = create_wavefront()
        assert wf.npixels == wf.amplitude.shape[-1]


    def test_diameter(self, create_wavefront: callable) -> None:
        """
        Tests the diameter property.
        """
        wf = create_wavefront()
        assert wf.diameter == wf.npixels * wf.pixel_scale


    def test_real(self, create_wavefront: callable) -> None:
        """
        Tests the real property.
        """
        wf = create_wavefront()
        assert (wf.real == wf.amplitude * np.cos(wf.phase)).all()


    def test_imaginary(self, create_wavefront: callable) -> None:
        """
        Tests the imaginary property.
        """
        wf = create_wavefront()
        assert (wf.imaginary == wf.amplitude * np.sin(wf.phase)).all()


    def test_phasor(self, create_wavefront: callable) -> None:
        """
        Tests the phasor property.
        """
        wf = create_wavefront()
        assert (wf.phasor == wf.amplitude * np.exp(1j*wf.phase)).all()


    def test_psf(self, create_wavefront: callable) -> None:
        """
        Tests the psf property.
        """
        wf = create_wavefront()
        assert (wf.psf == wf.amplitude**2).all()


    def test_coordinates(self, create_wavefront: callable) -> None:
        """
        Tests the coordinates property.
        """
        wf = create_wavefront()
        assert (wf.coordinates == \
        dLux.utils.coordinates.get_pixel_positions((wf.npixels, wf.npixels,),
            (wf.pixel_scale, wf.pixel_scale))).all()


    def test_tilt_wavefront(self, create_wavefront: callable) -> None:
        """
        Tests the tilt_wavefront method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.tilt_wavefront("some string")

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.tilt_wavefront(np.ones(1))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.tilt_wavefront(np.array(1.))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.tilt_wavefront(np.ones(3))

        # Test basic behaviour
        wf.tilt_wavefront(np.ones(2))


    def test_normalise(self, create_wavefront: callable) -> None:
        """
        Tests the normalise method.
        """
        wf = create_wavefront()

        new_wf = wf.normalise()
        assert np.sum(new_wf.amplitude**2) == 1.


    def test_invert_x_and_y(self, create_wavefront: callable) -> None:
        """
        Tests the invert_x_and_y method.
        """
        wf = create_wavefront()

        flipped_ampl = np.flip(wf.amplitude, axis=(-1, -2))
        flipped_phase = np.flip(wf.phase, axis=(-1, -2))
        new_wf = wf.invert_x_and_y()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_invert_x(self, create_wavefront: callable) -> None:
        """
        Tests the invert_x method.
        """
        wf = create_wavefront()

        flipped_ampl = np.flip(wf.amplitude, axis=-1)
        flipped_phase = np.flip(wf.phase, axis=-1)
        new_wf = wf.invert_x()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_invert_y(self, create_wavefront: callable) -> None:
        """
        Tests the invert_y method.
        """
        wf = create_wavefront()

        flipped_ampl = np.flip(wf.amplitude, axis=-2)
        flipped_phase = np.flip(wf.phase, axis=-2)
        new_wf = wf.invert_y()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_interpolate(self, create_wavefront: callable) -> None:
        """
        Tests the interpolate method.
        """
        wf = create_wavefront()
        wf = dLux.CircularAperture(1.)(wf)

        npix = wf.npixels
        pixscale = wf.pixel_scale
        k = 2
        new_wf = wf.interpolate(npix//k, pixscale*k)
        new_wf2 = wf.interpolate(npix//k, pixscale*k, real_imaginary=True)
        small_ampl = dLux.IntegerDownsample(k)(wf.amplitude)/k**2

        assert np.allclose(new_wf.amplitude[0], small_ampl)
        assert np.allclose(new_wf2.amplitude[0], small_ampl)


    def test_rotate(self, create_wavefront: callable) -> None:
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

        with pytest.raises(NotImplementedError):
            wf.rotate(np.pi, fourier=True)


    def test_pad_to(self, create_wavefront: callable) -> None:
        """
        Tests the pad_to method.
        """
        even_wf = create_wavefront(npixels=16)
        odd_wf = create_wavefront(npixels=15)
        
        # Smaller value
        with pytest.raises(AssertionError):
            even_wf.pad_to(14)

        # even -> odd
        with pytest.raises(AssertionError):
            even_wf.pad_to(17)

        # odd -> even
        with pytest.raises(AssertionError):
            odd_wf.pad_to(16)

        assert even_wf.pad_to(20).npixels == 20


    def test_crop_to(self, create_wavefront: callable) -> None:
        """
        Tests the crop_to method.
        """
        even_wf = create_wavefront(npixels=16)
        odd_wf = create_wavefront(npixels=15)

        # Smaller value
        with pytest.raises(AssertionError):
            even_wf.crop_to(18)

        # even -> odd
        with pytest.raises(AssertionError):
            even_wf.crop_to(15)

        # odd -> even
        with pytest.raises(AssertionError):
            odd_wf.crop_to(14)

        assert even_wf.crop_to(12).npixels == 12
