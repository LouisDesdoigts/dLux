from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
# from jax import config
# config.update("jax_debug_nans", True)


class TestCartesianMFT():
    """
    Test the CartesianMFT class.
    """


    def test_constructor(self, create_cartesian_mft : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        create_cartesian_mft()


    def test_propagate(self, create_cartesian_mft : callable):
        """
        Tests the propagate method.
        """
        wl, npix, f_pscale, fl = 1e-6, 32, 1e-6, 5.
        p_pscale = 1/npix
        amplitude = np.ones((1, npix, npix))
        phase = np.zeros((1, npix, npix))
        plane_type = dLux.PlaneType.Pupil

        # Construct
        wf = dLux.CartesianWavefront(wl, p_pscale, amplitude, phase, plane_type)
        prop = create_cartesian_mft(npix, f_pscale, fl)
        inv_prop = create_cartesian_mft(npix, p_pscale, fl, inverse=True)

        # Prop
        focal = prop(wf)
        pupil = inv_prop(focal)

        # Test
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()

        # Shift tests
        shift_pix = np.ones(2)
        shift = f_pscale * np.ones(2)

        # Construct
        prop_shift = create_cartesian_mft(npix, f_pscale, fl, shift=shift)
        prop_shift_pix = create_cartesian_mft(npix, f_pscale, fl, 
                                            shift=shift_pix, pixel_shift=True)

        # Prop
        focal_shift = prop_shift(wf)
        focal_shift_pix = prop_shift_pix(wf)
        focal_roll = np.roll(focal.psf, (1, 1), (0, 1))[1:, 1:]

        # Test
        assert np.allclose(focal_roll, focal_shift.psf[1:, 1:])
        assert np.allclose(focal_roll, focal_shift_pix.psf[1:, 1:])


class TestAngularMFT():
    """
    Test the AngularMFT class.
    """

    def test_constructor(self, create_angular_mft : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        create_angular_mft()


    def test_propagate(self, create_angular_mft : callable):
        """
        Tests the propagate method.
        """
        wl, npix, f_pscale = 1e-6, 32, 2e-7
        p_pscale = 1/npix
        amplitude = np.ones((1, npix, npix))
        phase = np.zeros((1, npix, npix))
        plane_type = dLux.PlaneType.Pupil

        # Construct
        wf = dLux.AngularWavefront(wl, p_pscale, amplitude, phase, plane_type)
        prop = create_angular_mft(npix, f_pscale)
        inv_prop = create_angular_mft(npix, p_pscale, inverse=True)

        # Prop
        focal = prop(wf)
        pupil = inv_prop(focal)

        # Test
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()

        # Shift tests
        shift_pix = np.ones(2)
        shift = f_pscale * np.ones(2)

        # Construct
        prop_shift = create_angular_mft(npix, f_pscale, shift=shift)
        prop_shift_pix = create_angular_mft(npix, f_pscale, 
                                            shift=shift_pix, pixel_shift=True)

        # Prop
        focal_shift = prop_shift(wf)
        focal_shift_pix = prop_shift_pix(wf)
        focal_roll = np.roll(focal.psf, (1, 1), (0, 1))[1:, 1:]

        # Test
        assert np.allclose(focal_roll, focal_shift.psf[1:, 1:])
        assert np.allclose(focal_roll, focal_shift_pix.psf[1:, 1:])


class TestCartesianFFT():
    """
    Test the CartesianFFT class.
    """


    def test_constructor(self, create_cartesian_fft : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        create_cartesian_fft()


    def test_propagate(self, create_cartesian_fft : callable):
        """
        Tests the propagate method.
        """
        wl, npix, fl = 1e-6, 32, 5.
        p_pscale = 1/npix
        amplitude = np.ones((1, npix, npix))
        phase = np.zeros((1, npix, npix))
        plane_type = dLux.PlaneType.Pupil

        # Construct
        wf = dLux.CartesianWavefront(wl, p_pscale, amplitude, phase, plane_type)
        prop = create_cartesian_fft(fl)
        inv_prop = create_cartesian_fft(fl, inverse=True)

        # Prop
        focal = prop(wf.pad_to(npix * 5))
        pupil = inv_prop(focal).crop_to(npix)

        # Test
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()

class TestAngularFFT():
    """
    Test the AngularFFT class.
    """


    def test_constructor(self, create_angular_fft : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        create_angular_fft()


    def test_propagate(self, create_angular_fft : callable):
        """
        Tests the propagate method.
        """
        wl, npix = 1e-6, 32
        p_pscale = 1/npix
        amplitude = np.ones((1, npix, npix))
        phase = np.zeros((1, npix, npix))
        plane_type = dLux.PlaneType.Pupil

        # Construct
        wf = dLux.AngularWavefront(wl, p_pscale, amplitude, phase, plane_type)
        prop = create_angular_fft()
        inv_prop = create_angular_fft(inverse=True)

        # Prop
        focal = prop(wf.pad_to(npix * 5))
        pupil = inv_prop(focal).crop_to(npix)

        # Test
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()



class TestCartesianFresnel():
    """
    Test the CartesianFresnel class.
    """


    def test_constructor(self, create_cartesian_fresnel : callable):
        """
        Tests the constructor.
        """
        # Test constructor
        create_cartesian_fresnel()


    def test_propagate(self, create_cartesian_fresnel : callable):
        """
        Tests the propagate method.
        """
        wl, npix, f_pscale, fl = 1e-6, 32, 2e-7, 5.
        p_pscale = 1/npix
        amplitude = np.ones((1, npix, npix))
        phase = np.zeros((1, npix, npix))
        plane_type = dLux.PlaneType.Pupil

        # Construct
        wf = dLux.FarFieldFresnelWavefront(wl, p_pscale, amplitude, phase, 
                                         plane_type)
        fresnel_prop = create_cartesian_fresnel(npix, f_pscale, fl, 5e-5)
        focal_prop = dLux.CartesianFresnel(npix, f_pscale, fl, 0.)
        inv_prop = create_cartesian_fresnel(npix, p_pscale, fl, 1e0, inverse=True)

        # Prop
        fresnel = fresnel_prop(wf)
        focal = focal_prop(wf)
        pupil = inv_prop(focal)

        # Test
        assert not np.isnan(fresnel.psf).any()
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()
