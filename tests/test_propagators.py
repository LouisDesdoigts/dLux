from __future__ import annotations
# from utilities import Utility, 
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)


# class PropagatorUtility(Utility):
#     """
#     Utility for Propagator class.
#     """
#     dLux.propagators.Propagator.__abstractmethods__ = ()
#     inverse : bool


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the Propagator Utility.
#         """
#         self.inverse  = False


#     def construct(self : Utility, inverse : bool = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.Propagator(inverse=inverse)


# class FixedSamplingPropagatorUtility(PropagatorUtility):
#     """
#     Utility for FixedSamplingPropagator class.
#     """
#     dLux.propagators.FixedSamplingPropagator.__abstractmethods__ = ()


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the Propagator Utility.
#         """
#         super().__init__()


#     def construct(self : Utility, inverse : bool = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.FixedSamplingPropagator(inverse=inverse)


# class VariableSamplingPropagatorUtility(PropagatorUtility):
#     """
#     Utility for VariableSamplingPropagator class.
#     """
#     dLux.propagators.VariableSamplingPropagator.__abstractmethods__ = ()
#     npixels_out     : int
#     pixel_scale_out : Array
#     shift           : Array
#     pixel_shift     : bool


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the VariableSamplingPropagator Utility.
#         """
#         super().__init__()
#         self.npixels_out      = 16
#         self.pixel_scale_out  = np.array(1.)
#         self.shift            = np.zeros(2)
#         self.pixel_shift      = False


#     def construct(self            : Utility,
#                   npixels_out     : int   = None,
#                   pixel_scale_out : Array = None,
#                   shift           : Array = None,
#                   pixel_shift     : bool  = None,
#                   inverse         : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
#                                                         else pixel_scale_out
#         npixels_out = self.npixels_out if npixels_out is None else npixels_out
#         shift       = self.shift       if shift       is None else shift
#         pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
#         inverse     = self.inverse     if inverse     is None else inverse
#         return dLux.propagators.VariableSamplingPropagator(pixel_scale_out,
#                             npixels_out, shift, pixel_shift, inverse=inverse)


# class CartesianPropagatorUtility(PropagatorUtility):
#     """
#     Utility for CartesianPropagator class.
#     """
#     dLux.propagators.CartesianPropagator.__abstractmethods__ = ()
#     focal_length : Array


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the CartesianPropagator Utility.
#         """
#         super().__init__()
#         self.focal_length = np.array(1.)


#     def construct(self         : Utility,
#                   focal_length : Array = None,
#                   inverse      : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         focal_length = self.focal_length \
#                         if focal_length is None else focal_length
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.CartesianPropagator(focal_length, \
#                                                     inverse=inverse)


# class AngularPropagatorUtility(PropagatorUtility):
#     """
#     Utility for AngularPropagator class.
#     """
#     dLux.propagators.AngularPropagator.__abstractmethods__ = ()


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the AngularPropagator Utility.
#         """
#         super().__init__()
#         self.focal_length = np.array(1.)


#     def construct(self : Utility, inverse : bool = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.AngularPropagator(inverse=inverse)


# class FarFieldFresnelUtility(PropagatorUtility):
#     """
#     Utility for FarFieldFresnel class.
#     """
#     dLux.propagators.FarFieldFresnel.__abstractmethods__ = ()
#     propagation_shift : Array


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the FarFieldFresnel Utility.
#         """
#         super().__init__()
#         self.propagation_shift = np.array(1e-3)


#     def construct(self              : Utility,
#                   propagation_shift : Array = None,
#                   inverse           : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         propagation_shift = self.propagation_shift \
#                             if propagation_shift is None else propagation_shift
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.FarFieldFresnel(propagation_shift, \
#                                                 inverse=inverse)



# class CartesianMFTUtility(CartesianPropagatorUtility,
#                           VariableSamplingPropagatorUtility):
#     """
#     Utility for CartesianMFT class.
#     """


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the CartesianMFT Utility.
#         """
#         super().__init__()


#     def construct(self            : Utility,
#                   npixels_out     : int   = None,
#                   pixel_scale_out : float = None,
#                   focal_length    : Array = None,
#                   inverse         : bool  = None,
#                   shift           : Array = None,
#                   pixel_shift     : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
#                                                         else pixel_scale_out
#         focal_length = self.focal_length if focal_length is None \
#                                                         else focal_length
#         npixels_out = self.npixels_out if npixels_out is None else npixels_out
#         shift       = self.shift       if shift       is None else shift
#         pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
#         inverse     = self.inverse     if inverse     is None else inverse
#         return dLux.propagators.CartesianMFT(npixels_out, pixel_scale_out,
#                                      focal_length, inverse, shift, pixel_shift)


# class AngularMFTUtility(AngularPropagatorUtility,
#                         VariableSamplingPropagatorUtility):
#     """
#     Utility for AngularMFT class.
#     """


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the AngularMFT Utility.
#         """
#         super().__init__()


#     def construct(self            : Utility,
#                   npixels_out     : int   = None,
#                   pixel_scale_out : float = None,
#                   inverse         : bool  = None,
#                   shift           : Array = None,
#                   pixel_shift     : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
#                                                         else pixel_scale_out
#         npixels_out = self.npixels_out if npixels_out is None else npixels_out
#         shift       = self.shift       if shift       is None else shift
#         pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
#         inverse     = self.inverse     if inverse     is None else inverse
#         return dLux.propagators.AngularMFT(npixels_out, pixel_scale_out,
#                                            inverse, shift, pixel_shift)


# class CartesianFFTUtility(CartesianPropagatorUtility,
#                           FixedSamplingPropagatorUtility):
#     """
#     Utility for CartesianFFT class.
#     """


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the CartesianFFT Utility.
#         """
#         super().__init__()


#     def construct(self         : Utility,
#                   focal_length : Array = None,
#                   inverse      : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         focal_length = self.focal_length if focal_length is None \
#                                                         else focal_length
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.CartesianFFT(focal_length, inverse)


# class AngularFFTUtility(AngularPropagatorUtility,
#                         FixedSamplingPropagatorUtility):
#     """
#     Utility for AngularFFT class.
#     """


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the AngularFFT Utility.
#         """
#         super().__init__()


#     def construct(self : Utility, inverse : bool = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         inverse = self.inverse if inverse is None else inverse
#         return dLux.propagators.AngularFFT(inverse)



# class CartesianFresnelUtility(FarFieldFresnelUtility, CartesianMFTUtility):
#     """
#     Utility for CartesianFresnel class.
#     """


#     def __init__(self : Utility) -> Utility:
#         """
#         Constructor for the CartesianFresnel Utility.
#         """
#         super().__init__()


#     def construct(self              : Utility,
#                   npixels_out       : int   = None,
#                   pixel_scale_out   : float = None,
#                   focal_length      : Array = None,
#                   propagation_shift : Array = None,
#                   inverse           : bool  = None,
#                   shift             : Array = None,
#                   pixel_shift       : bool  = None) -> Propagator:
#         """
#         Safe constructor for the dLuxModule, associated with this utility.
#         """
#         pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
#                                                         else pixel_scale_out
#         focal_length = self.focal_length if focal_length is None \
#                                                         else focal_length
#         propagation_shift = self.propagation_shift if propagation_shift is None\
#                                                         else propagation_shift
#         npixels_out = self.npixels_out if npixels_out is None else npixels_out
#         shift       = self.shift       if shift       is None else shift
#         pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
#         inverse     = self.inverse     if inverse     is None else inverse
#         return dLux.propagators.CartesianFresnel(npixels_out, pixel_scale_out,
#                  focal_length, propagation_shift, inverse, shift, pixel_shift)



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


'''
class TestCartesianFFT():
    """
    Test the CartesianFFT class.
    """
    utility : CartesianFFTUtility = CartesianFFTUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test constructor
        self.utility.construct()


    def test_propagate(self):
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
        prop = self.utility.construct(fl)
        inv_prop = self.utility.construct(fl, inverse=True)

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
    utility : AngularFFTUtility = AngularFFTUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test constructor
        self.utility.construct()


    def test_propagate(self):
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
        prop = self.utility.construct()
        inv_prop = self.utility.construct(inverse=True)

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
    utility : CartesianFresnelUtility = CartesianFresnelUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test constructor
        self.utility.construct()


    def test_propagate(self):
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
        fresnel_prop = self.utility.construct(npix, f_pscale, fl, 5e-5)
        focal_prop = dLux.CartesianFresnel(npix, f_pscale, fl, 0.)
        inv_prop = self.utility.construct(npix, p_pscale, fl, 1e0, inverse=True)

        # Prop
        fresnel = fresnel_prop(wf)
        focal = focal_prop(wf)
        pupil = inv_prop(focal)

        # Test
        assert not np.isnan(fresnel.psf).any()
        assert not np.isnan(focal.psf).any()
        assert not np.isnan(pupil.psf).any()
'''