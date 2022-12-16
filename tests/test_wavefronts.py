from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


Array = np.ndarray


@pytest.fixture
def create_wavefront():
    """
    Returns:
    --------
    create_wavefront: callable
        A function that has all keyword arguments and can be 
        used to create a wavefront for testing.
    """
    def _create_wavefront(
            wavelength: Array = np.array(550e-09),
            pixel_scale: Array = np.array(1.),
            plane_type: int = dLux.PlaneType.Pupil,
            amplitude: Array = np.ones((1, 16, 16)),
            phase: Array = np.zeros((1, 16, 16))) -> Wavefront:
        return dLux.wavefronts.Wavefront(
            wavelength, pixel_scale, amplitude, phase, plane_type)
    return _create_wavefront


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

        # Test empty array
        with pytest.raises(AssertionError):
            create_wavefront(pixel_scale=[])

        # Test 1d array
        with pytest.raises(AssertionError):
            create_wavefront(pixel_scale=[1e3, 1e4])

        # Test non 3d amplitude array
        with pytest.raises(AssertionError):
            create_wavefront(amplitude=np.ones((3, 3)))

        # Test non 3d amplitude array
        with pytest.raises(AssertionError):
            create_wavefront(amplitude=np.ones((3, 3, 3, 3)))

        # Test non 3d phase array
        with pytest.raises(AssertionError):
            create_wavefront(phase=np.ones((3, 3)))

        # Test non 3d phase array
        with pytest.raises(AssertionError):
            create_wavefront(phase=np.ones((3, 3, 3, 3)))

        # Test different amplitude/phase array shapes
        with pytest.raises(AssertionError):
            create_wavefront(amplitude=np.ones((4, 4, 4)),
                                   phase=np.ones((3, 3, 3)))

        # Test non-planetype plane_type
        with pytest.raises(AssertionError):
            create_wavefront(plane_type=[1])


    def test_npixels(self, create_wavefront: callable) -> None:
        """
        Tests the npixels property.
        """
        wf = create_wavefront()
        assert wf.npixels == wf.amplitude.shape[-1]


    def test_nfields(self, create_wavefront: callable) -> None:
        """
        Tests the nfields property.
        """
        wf = create_wavefront()
        assert wf.nfields == wf.amplitude.shape[0]


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


    def test_pixel_coordinates(self, create_wavefront: callable) -> None:
        """
        Tests the pixel_coordinates property.
        """
        wf = create_wavefront()
        assert (wf.pixel_coordinates == \
        dLux.utils.coordinates.get_pixel_coordinates(wf.npixels,
                                                     wf.pixel_scale)).all()


    def test_set_amplitude(self, create_wavefront: callable) -> None:
        """
        Tests the set_amplitude method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_amplitude("some string")

        # Test list inputs
        with pytest.raises(AssertionError):
            wf.set_amplitude([])

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_amplitude(np.ones((16, 16)))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_amplitude(np.ones((1, 1, 16, 16)))

        # Test correct behaviour
        new_ampl = 0.5*np.ones((1, 16, 16))
        assert (wf.set_amplitude(new_ampl).amplitude == new_ampl).all()


    def test_set_phase(self, create_wavefront: callable) -> None:
        """
        Tests the set_phase method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_phase("some string")

        # Test list inputs
        with pytest.raises(AssertionError):
            wf.set_phase([])

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_phase(np.ones((16, 16)))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_phase(np.ones((1, 1, 16, 16)))

        # Test correct behaviour
        new_phase = 0.5*np.ones((1, 16, 16))
        assert (wf.set_phase(new_phase).phase == new_phase).all()


    def test_set_pixel_scale(self, create_wavefront: callable) -> None:
        """
        Tests the set_pixel_scale method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_pixel_scale("some string")

        # Test list inputs
        with pytest.raises(AssertionError):
            wf.set_pixel_scale([])

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_pixel_scale(np.array([]))

        # Test correct behaviour
        new_pixscale = np.array(1.5)
        assert wf.set_pixel_scale(new_pixscale).pixel_scale == new_pixscale


    def test_set_plane_type(self, create_wavefront: callable) -> None:
        """
        Tests the set_plane_type method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_plane_type("some string")

        # Test correct behaviour
        new_plane_type = dLux.PlaneType.Focal
        assert wf.set_plane_type(new_plane_type).plane_type == new_plane_type


    def test_set_phasor(self, create_wavefront: callable) -> None:
        """
        Tests the set_phasor method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_phasor("some string", "some_string")

        # Test list inputs
        with pytest.raises(AssertionError):
            wf.set_phasor([], [])

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_phasor(np.ones((16, 16)), np.ones((16, 16)))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_phasor(np.ones((1, 1, 16, 16)), np.ones((1, 1, 16, 16)))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.set_phasor(np.ones((1, 16, 16)), np.ones((1, 15, 15)))

        # Test correct behaviour
        new_ampl = 0.5*np.ones((1, 16, 16))
        new_phase = 0.5*np.ones((1, 16, 16))
        assert (wf.set_phasor(new_ampl, new_phase).amplitude == new_ampl).all()
        assert (wf.set_phasor(new_ampl, new_phase).phase == new_phase).all()


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


    def test_multiply_amplitude(self, create_wavefront: callable) -> None:
        """
        Tests the multiply_amplitude method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.multiply_amplitude("some string")

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.multiply_amplitude(np.ones(1))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.multiply_amplitude(np.ones((1, 1, 1, 1)))

        # Test basic behaviour
        npix = wf.npixels
        wf.multiply_amplitude(np.array(1.))
        wf.multiply_amplitude(np.ones((npix, npix)))
        wf.multiply_amplitude(np.ones((1, npix, npix)))


    def test_add_phase(self, create_wavefront: callable) -> None:
        """
        Tests the add_phase method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.add_phase("some string")

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.add_phase(np.ones(1))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.add_phase(np.ones((1, 1, 1, 1)))

        # Test basic behaviour
        npix = wf.npixels
        wf.add_phase(np.array(1.))
        wf.add_phase(np.ones((npix, npix)))
        wf.add_phase(np.ones((1, npix, npix)))


    def test_add_opd(self, create_wavefront: callable) -> None:
        """
        Tests the add_opd method.
        """
        wf = create_wavefront()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.add_opd("some string")

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.add_opd(np.ones(1))

        # Test wrong shapes
        with pytest.raises(AssertionError):
            wf.add_opd(np.ones((1, 1, 1, 1)))

        # Test basic behaviour
        npix = wf.npixels
        wf.add_opd(np.array(1.))
        wf.add_opd(np.ones((npix, npix)))
        wf.add_opd(np.ones((1, npix, npix)))


    def test_normalise(self, create_wavefront: callable) -> None:
        """
        Tests the normalise method.
        """
        wf = create_wavefront()

        new_wf = wf.normalise()
        assert np.sum(new_wf.amplitude**2) == 1.


    def test_wavefront_to_psf(self, create_wavefront: callable) -> None:
        """
        Tests the wavefront_to_psf method.
        """
        wf = create_wavefront()
        wf.wavefront_to_psf()


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
        small_ampl = dLux.IntegerDownsample(k)(wf.amplitude[0])/k**2

        assert np.allclose(new_wf.amplitude[0], small_ampl)
        assert np.allclose(new_wf2.amplitude[0], small_ampl)


    def test_rotate(self, create_wavefront: callable) -> None:
        """
        Tests the rotate method.
        """
        wf = create_wavefront()
        wf = dLux.CircularAperture(1.)(wf)
        flipped_amplitude = np.flipud(wf.amplitude)
        flipped_phase = np.flipud(wf.phase)

        new_wf = wf.rotate(np.pi)
        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        assert np.allclose(new_wf.phase, flipped_phase)

        new_wf = wf.rotate(np.pi, real_imaginary=True)
        assert np.allclose(new_wf.amplitude, flipped_amplitude, atol=1e-5)
        assert np.allclose(new_wf.phase, flipped_phase)

        with pytest.raises(NotImplementedError):
            wf.rotate(np.pi, fourier=True)


    def test_pad_to(self, create_wavefront: callable) -> None:
        """
        Tests the pad_to method.
        """
        even_wf = create_wavefront()
        odd_wf = create_wavefront(amplitude=np.ones((1, 15, 15)),
                                        phase=np.ones((1, 15, 15)))

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
        even_wf = create_wavefront()
        odd_wf = create_wavefront(amplitude=np.ones((1, 15, 15)),
                                        phase=np.ones((1, 15, 15)))

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
