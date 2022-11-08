from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


Array = np.ndarray


class WavefrontUtility(Utility):
    """
    Utility for Wavefront class.
    """
    wavelength  : Array
    pixel_scale : Array
    plane_type  : PlaneType
    amplitude   : Array
    phase       : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Wavefront Utility.
        """
        self.wavelength  = np.array(550e-09)
        self.pixel_scale = np.array(1.)
        self.plane_type  = dLux.PlaneType.Pupil
        self.amplitude   = np.ones((1, 16, 16))
        self.phase       = np.zeros((1, 16, 16))


    def construct(self        : Utility,
                  wavelength  : Array = None,
                  pixel_scale : Array = None,
                  plane_type  : dLux.wavefronts.PlaneType = None,
                  amplitude   : Array = None,
                  phase       : Array = None) -> Wavefront:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        wavelength  = self.wavelength  if wavelength  is None else wavelength
        pixel_scale = self.pixel_scale if pixel_scale is None else pixel_scale
        plane_type  = self.plane_type  if plane_type  is None else plane_type
        amplitude   = self.amplitude   if amplitude   is None else amplitude
        phase       = self.phase       if phase       is None else phase
        return dLux.wavefronts.Wavefront(wavelength, pixel_scale, amplitude,
                                         phase, plane_type)


class CartesianWavefrontUtility(WavefrontUtility):
    """
    Utility for CartesianWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianWavefront Utility.
        """
        super().__init__()


class AngularWavefrontUtility(WavefrontUtility):
    """
    Utility for AngularWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianWavefront Utility.
        """
        super().__init__()


class FarFieldFresnelWavefrontUtility(WavefrontUtility):
    """
    Utility for FarFieldFresnelWavefront class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the FarFieldFresnelWavefront Utility.
        """
        super().__init__()


class TestWavefront(UtilityUser):
    """
    Test the Wavefront class.
    """
    utility : WavefrontUtility = WavefrontUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test constructor
        self.utility.construct()

        # Test empty array
        with pytest.raises(AssertionError):
            self.utility.construct(wavelength=[])

        # Test 1d array
        with pytest.raises(AssertionError):
            self.utility.construct(wavelength=[1e3, 1e4])

        # Test empty array
        with pytest.raises(AssertionError):
            self.utility.construct(pixel_scale=[])

        # Test 1d array
        with pytest.raises(AssertionError):
            self.utility.construct(pixel_scale=[1e3, 1e4])

        # Test non 3d amplitude array
        with pytest.raises(AssertionError):
            self.utility.construct(amplitude=np.ones((3, 3)))

        # Test non 3d amplitude array
        with pytest.raises(AssertionError):
            self.utility.construct(amplitude=np.ones((3, 3, 3, 3)))

        # Test non 3d phase array
        with pytest.raises(AssertionError):
            self.utility.construct(phase=np.ones((3, 3)))

        # Test non 3d phase array
        with pytest.raises(AssertionError):
            self.utility.construct(phase=np.ones((3, 3, 3, 3)))

        # Test different amplitude/phase array shapes
        with pytest.raises(AssertionError):
            self.utility.construct(amplitude=np.ones((4, 4, 4)),
                                   phase=np.ones((3, 3, 3)))

        # Test non-planetype plane_type
        with pytest.raises(AssertionError):
            self.utility.construct(plane_type=[1])


    def test_npixels(self):
        """
        Tests the npixels property.
        """
        wf = self.utility.construct()
        assert wf.npixels == wf.amplitude.shape[-1]


    def test_nfields(self):
        """
        Tests the nfields property.
        """
        wf = self.utility.construct()
        assert wf.nfields == wf.amplitude.shape[0]


    def test_diameter(self):
        """
        Tests the diameter property.
        """
        wf = self.utility.construct()
        assert wf.diameter == wf.npixels * wf.pixel_scale


    def test_real(self):
        """
        Tests the real property.
        """
        wf = self.utility.construct()
        assert (wf.real == wf.amplitude * np.cos(wf.phase)).all()


    def test_imaginary(self):
        """
        Tests the imaginary property.
        """
        wf = self.utility.construct()
        assert (wf.imaginary == wf.amplitude * np.sin(wf.phase)).all()


    def test_phasor(self):
        """
        Tests the phasor property.
        """
        wf = self.utility.construct()
        assert (wf.phasor == wf.amplitude * np.exp(1j*wf.phase)).all()


    def test_psf(self):
        """
        Tests the psf property.
        """
        wf = self.utility.construct()
        assert (wf.psf == wf.amplitude**2).all()


    def test_pixel_coordinates(self):
        """
        Tests the pixel_coordinates property.
        """
        wf = self.utility.construct()
        assert (wf.pixel_coordinates == \
        dLux.utils.coordinates.get_pixel_coordinates(wf.npixels,
                                                     wf.pixel_scale)).all()


    def test_set_amplitude(self):
        """
        Tests the set_amplitude method.
        """
        wf = self.utility.construct()

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


    def test_set_phase(self):
        """
        Tests the set_phase method.
        """
        wf = self.utility.construct()

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


    def test_set_pixel_scale(self):
        """
        Tests the set_pixel_scale method.
        """
        wf = self.utility.construct()

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


    def test_set_plane_type(self):
        """
        Tests the set_plane_type method.
        """
        wf = self.utility.construct()

        # Test string inputs
        with pytest.raises(AssertionError):
            wf.set_plane_type("some string")

        # Test correct behaviour
        new_plane_type = dLux.PlaneType.Focal
        assert wf.set_plane_type(new_plane_type).plane_type == new_plane_type


    def test_set_phasor(self):
        """
        Tests the set_phasor method.
        """
        wf = self.utility.construct()

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


    def test_tilt_wavefront(self):
        """
        Tests the tilt_wavefront method.
        """
        wf = self.utility.construct()

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


    def test_multiply_amplitude(self):
        """
        Tests the multiply_amplitude method.
        """
        wf = self.utility.construct()

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


    def test_add_phase(self):
        """
        Tests the add_phase method.
        """
        wf = self.utility.construct()

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


    def test_add_opd(self):
        """
        Tests the add_opd method.
        """
        wf = self.utility.construct()

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


    def test_normalise(self):
        """
        Tests the normalise method.
        """
        wf = self.utility.construct()

        new_wf = wf.normalise()
        assert np.sum(new_wf.amplitude**2) == 1.


    def test_wavefront_to_psf(self):
        """
        Tests the wavefront_to_psf method.
        """
        wf = self.utility.construct()
        wf.wavefront_to_psf()


    def test_invert_x_and_y(self):
        """
        Tests the invert_x_and_y method.
        """
        wf = self.utility.construct()

        flipped_ampl = np.flip(wf.amplitude, axis=(-1, -2))
        flipped_phase = np.flip(wf.phase, axis=(-1, -2))
        new_wf = wf.invert_x_and_y()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_invert_x(self):
        """
        Tests the invert_x method.
        """
        wf = self.utility.construct()

        flipped_ampl = np.flip(wf.amplitude, axis=-1)
        flipped_phase = np.flip(wf.phase, axis=-1)
        new_wf = wf.invert_x()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_invert_y(self):
        """
        Tests the invert_y method.
        """
        wf = self.utility.construct()

        flipped_ampl = np.flip(wf.amplitude, axis=-2)
        flipped_phase = np.flip(wf.phase, axis=-2)
        new_wf = wf.invert_y()
        assert (new_wf.amplitude == flipped_ampl).all()
        assert (new_wf.phase == flipped_phase).all()


    def test_interpolate(self):
        """
        Tests the interpolate method.
        """
        wf = self.utility.construct()
        wf = dLux.CircularAperture(wf.npixels)(wf)

        npix = wf.npixels
        pixscale = wf.pixel_scale
        k = 2
        new_wf = wf.interpolate(npix//k, pixscale*k)
        new_wf2 = wf.interpolate(npix//k, pixscale*k, real_imaginary=True)
        small_ampl = dLux.IntegerDownsample(k)(wf.amplitude[0])/k**2

        assert np.allclose(new_wf.amplitude[0], small_ampl)
        assert np.allclose(new_wf2.amplitude[0], small_ampl)


    def test_rotate(self):
        """
        Tests the rotate method.
        """
        wf = self.utility.construct()
        wf = dLux.CircularAperture(wf.npixels)(wf)
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


    def test_pad_to(self):
        """
        Tests the pad_to method.
        """
        even_wf = self.utility.construct()
        odd_wf = self.utility.construct(amplitude=np.ones((1, 15, 15)),
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


    def test_crop_to(self):
        """
        Tests the crop_to method.
        """
        even_wf = self.utility.construct()
        odd_wf = self.utility.construct(amplitude=np.ones((1, 15, 15)),
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