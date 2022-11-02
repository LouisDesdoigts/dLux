from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
import pytest
import dLux


class PropagatorUtility(Utility):
    """
    Utility for Propagator class.
    """
    dLux.propagators.Propagator.__abstractmethods__ = ()
    inverse : bool


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Propagator Utility.
        """
        self.inverse  = False


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.Propagator(inverse=inverse)


class FixedSamplingPropagatorUtility(PropagatorUtility):
    """
    Utility for FixedSamplingPropagator class.
    """
    dLux.propagators.FixedSamplingPropagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the Propagator Utility.
        """
        super().__init__()


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.FixedSamplingPropagator(inverse=inverse)


class VariableSamplingPropagatorUtility(PropagatorUtility):
    """
    Utility for VariableSamplingPropagator class.
    """
    dLux.propagators.VariableSamplingPropagator.__abstractmethods__ = ()
    npixels_out     : int
    pixel_scale_out : Array
    shift           : Array
    pixel_shift     : bool


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the VariableSamplingPropagator Utility.
        """
        super().__init__()
        self.npixels_out      = 16
        self.pixel_scale_out  = np.array(1.)
        self.shift            = np.zeros(2)
        self.pixel_shift      = False


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : Array = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None,
                  inverse         : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.VariableSamplingPropagator(pixel_scale_out,
                            npixels_out, shift, pixel_shift, inverse=inverse)


class CartesianPropagatorUtility(PropagatorUtility):
    """
    Utility for CartesianPropagator class.
    """
    dLux.propagators.CartesianPropagator.__abstractmethods__ = ()
    focal_length : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianPropagator Utility.
        """
        super().__init__()
        self.focal_length = np.array(1.)


    def construct(self         : Utility,
                  focal_length : Array = None,
                  inverse      : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        focal_length = self.focal_length \
                        if focal_length is None else focal_length
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.CartesianPropagator(focal_length, \
                                                    inverse=inverse)


class AngularPropagatorUtility(PropagatorUtility):
    """
    Utility for AngularPropagator class.
    """
    dLux.propagators.AngularPropagator.__abstractmethods__ = ()


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularPropagator Utility.
        """
        super().__init__()
        self.focal_length = np.array(1.)


    def construct(self : Utility, inverse : bool = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.AngularPropagator(inverse=inverse)


class FarFieldFresnelUtility(PropagatorUtility):
    """
    Utility for FarFieldFresnel class.
    """
    dLux.propagators.FarFieldFresnel.__abstractmethods__ = ()
    propagation_shift : Array


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the FarFieldFresnel Utility.
        """
        super().__init__()
        self.propagation_shift = np.array(1e-3)


    def construct(self              : Utility,
                  propagation_shift : Array = None,
                  inverse           : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        propagation_shift = self.propagation_shift \
                            if propagation_shift is None else propagation_shift
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.FarFieldFresnel(propagation_shift, \
                                                inverse=inverse)



class CartesianMFTUtility(CartesianPropagatorUtility,
                          VariableSamplingPropagatorUtility):
    """
    Utility for CartesianMFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianMFT Utility.
        """
        super().__init__()


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : float = None,
                  focal_length    : Array = None,
                  inverse         : bool  = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.CartesianMFT(npixels_out, pixel_scale_out,
                                     focal_length, inverse, shift, pixel_shift)


class AngularMFTUtility(AngularPropagatorUtility,
                        VariableSamplingPropagatorUtility):
    """
    Utility for AngularMFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularMFT Utility.
        """
        super().__init__()


    def construct(self            : Utility,
                  npixels_out     : int   = None,
                  pixel_scale_out : float = None,
                  inverse         : bool  = None,
                  shift           : Array = None,
                  pixel_shift     : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.AngularMFT(npixels_out, pixel_scale_out,
                                     focal_length, inverse, shift, pixel_shift)


class CartesianFFTUtility(CartesianPropagatorUtility,
                          FixedSamplingPropagatorUtility):
    """
    Utility for CartesianFFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianFFT Utility.
        """
        super().__init__()


    def construct(self         : Utility,
                  focal_length : Array = None,
                  inverse      : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.CartesianFFT(focal_length, inverse)


class AngularFFTUtility(AngularPropagatorUtility,
                        FixedSamplingPropagatorUtility):
    """
    Utility for AngularFFT class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the AngularFFT Utility.
        """
        super().__init__()


    def construct(self    : Utility, inverse : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        inverse = self.inverse if inverse is None else inverse
        return dLux.propagators.AngularFFT(inverse)



class CartesianFresnelUtility(FarFieldFresnelUtility, CartesianMFTUtility):
    """
    Utility for CartesianFresnel class.
    """


    def __init__(self : Utility) -> Utility:
        """
        Constructor for the CartesianFresnel Utility.
        """
        super().__init__()


    def construct(self              : Utility,
                  npixels_out       : int   = None,
                  pixel_scale_out   : float = None,
                  focal_length      : Array = None,
                  propagation_shift : Array = None,
                  inverse           : bool  = None,
                  shift             : Array = None,
                  pixel_shift       : bool  = None) -> Propagator:
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        pixel_scale_out = self.pixel_scale_out if pixel_scale_out is None \
                                                        else pixel_scale_out
        focal_length = self.focal_length if focal_length is None \
                                                        else focal_length
        propagation_shift = self.propagation_shift if propagation_shift is None\
                                                        else propagation_shift
        npixels_out = self.npixels_out if npixels_out is None else npixels_out
        shift       = self.shift       if shift       is None else shift
        pixel_shift = self.pixel_shift if pixel_shift is None else pixel_shift
        inverse     = self.inverse     if inverse     is None else inverse
        return dLux.propagators.CartesianFresnel(npixels_out, pixel_scale_out,
                 focal_length, propagation_shift, inverse, shift, pixel_shift)











# class TestPropagator(UtilityUser):
#     """
#     Tests the concrete methods of the abstract propagator class.
#     The concrete methods are:
#      - __init__()
#      - _get_pixel_grid()
#      - _get_pixel_positions()
#      - is_inverse()

#     Attributes
#     ----------
#     utility : PropagatorUtility
#     """
#     utility : Utility = PropagatorUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Tests that the constructor correctly initialialises the 
#         un-traced `inverse` attribute of the `Propagator` class.
#         """
#         false_propagator = self\
#             .get_utility()\
#             .construct(inverse = False)

#         true_propagator = self\
#             .get_utility()\
#             .construct(inverse = True)

#         assert false_propagator.is_inverse() == False
#         assert true_propagator.is_inverse() == True


#     def test_is_inverse(self : Tester) -> None:
#         """
#         Makes sure that the `is_inverse` function correctly reflects 
#         the state of the `Propagator`
#         """
#         true_propagator = self\
#             .get_utility()\
#             .construct(inverse = True)

#         false_propagator = self\
#             .get_utility()\
#             .construct(inverse = False)

#         assert true_propagator.is_inverse() == True
#         assert false_propagator.is_inverse() == False
        

# class TestVariableSamplingPropagator(UtilityUser):
#     """
#     Tests the concrete methods of the `VariableSamplingPropagator`
#     abstract class. The concrete methods are:
#      - __init__()
#      - _matrix_fourier_transform()
#      - _fourier_transform()
#      - _inverse_fourier_transform()
#      - _normalising_factor()
#      - get_pixel_scale_out()
#      - get_npixels_out()
#      - generate_twiddle_factors()

#     Attributes
#     ----------
#     utility : VariableSamplingUtility
#         A testing utility that generates safe test cases.

#     Inherits
#     --------
#     : UtilityUser
#         Provides acces to the `get_utitlity` method, broadcast 
#         accross all test classes.
#     """
#     utility : Utility = VariableSamplingUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Checks that all the appropriate fields are correctly 
#         initialised by the constructor. These fields are:
#          - pixel_scale_out : float
#          - npixels_out : int
#          - inverse : bool
#         """
#         propagator = self.get_utility().construct()

#         assert propagator.is_inverse() == \
#             self.get_utility().is_inverse()

#         assert propagator.get_npixels_out() == \
#             self.get_utility().get_npixels_out()

#         assert propagator.get_pixel_scale_out() == \
#             self.get_utility().get_pixel_scale_out()


#     def test_get_pixel_scale_out(self : Tester) -> None:
#         """
#         Tests that the `get_pixel_scale_out` method correctly 
#         reports the state of the object. 
#         """
#         OLD_PIXEL_SCALE = 0.7
#         NEW_PIXEL_SCALE = 0.5

#         old_propagator = self\
#             .get_utility()\
#             .construct(pixel_scale_out = OLD_PIXEL_SCALE)

#         new_propagator = self\
#             .get_utility()\
#             .construct(pixel_scale_out = NEW_PIXEL_SCALE)

#         assert old_propagator.get_pixel_scale_out() == OLD_PIXEL_SCALE
#         assert new_propagator.get_pixel_scale_out() == NEW_PIXEL_SCALE


#     def test_get_npixels_out(self : Tester) -> None:
#         """
#         Tests that `get_npixels_out` correctly tracks the state of 
#         the object. 
#         """
#         NEW_PIXELS = 100
#         OLD_PIXELS = 50

#         old_propagator = self\
#             .get_utility()\
#             .construct(npixels_out = OLD_PIXELS)

#         new_propagator = self\
#             .get_utility()\
#             .construct(npixels_out = NEW_PIXELS)

#         assert old_propagator.get_npixels_out() == OLD_PIXELS
#         assert new_propagator.get_npixels_out() == NEW_PIXELS


# #     def test_generate_twiddle_factors(self : Tester) -> None:
# #         """
# #         Tests that the correct twiddle factors are getting 
# #         generated. This is done by direct comparison for a 
# #         simple case.
# #         """
# #         TWIDDLE_FACTORS : Array = numpy.array([
# #             [-1.j,  1.j, -1.j, 1.j],
# #             [1.j, -1.j, 1.j, -1.j],
# #             [-1.j,  1.j, -1.j, 1.j],
# #             [1.j, -1.j, 1.j, -1.j]])
# #         OFFSETS : float = 2. 
# #         PIXELS : Array = numpy.array([4., 4.])
# #         PIXEL_SCALES : Array = numpy.array([1., 1.])
# #         SIGN : int = 1

# #         twiddle_factors = self\
# #             .get_utility()\
# #             .construct()\
# #             ._generate_twiddle_factors(OFFSETS, PIXEL_SCALES, 
# #                 PIXELS, SIGN)

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(twiddle_factors, TWIDDLE_FACTORS)\
# #             .all()

# #         assert is_correct


# class TestFixedSamplingPropagator(UtilityUser):
#     """
#     Contains the tests for the abstract FixedSamplingPropagator class.
#     The concrete methods tested:
#     - fourier_transform()
#     - inverse_fourier_transform()
#     - normalising_factor()
    
#     Attributes
#     ----------
#     utility : Utility 
#         A container of helpful things for testing.
#     """
#     utility : Utility = FixedSamplingUtility()


# #     def test_fourier_transform(self : Tester) -> None:
# #         """
# #         Tests the fourier transform of the FixedSamplingPropagator.
# #         The Fourier transform is based on the inbuilt `numpy.fft`
# #         so the tests are simple and mostly a formaility,
# #         """
# #         propagator = self.get_utility().construct()
# #         wavefront = self.get_utility().get_utility().construct()

# #         CORRECT = numpy.fft.fftshift(numpy.fft.ifft2(
# #             wavefront.get_phasor()))

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(propagator._fourier_transform(wavefront), CORRECT)\
# #             .all()

# #         assert is_correct


# #     def test_inverse_fourier_transfrom(self : Tester) -> None:
# #         """
# #         Tests the inverse fourier transform of the FixedSamplingPropagator
# #         As above this method is based on the inbuilt `numpy.fft` module so
# #         the tests are mostly a formality.
# #         """
# #         propagator = self.get_utility().construct()
# #         wavefront = self.get_utility().get_utility().construct()

# #         CORRECT = numpy.fft.fft2(numpy.fft.ifftshift(
# #             wavefront.get_phasor()))

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(propagator._inverse_fourier_transform(wavefront), 
# #                 CORRECT)\
# #             .all()

# #         assert is_correct


# #     def test_normalising_factor(self : Tester) -> None:
# #         """
# #         Tests that the correct normalising factor is generated for both 
# #         the forward and reverse modes. The correct forward mode is 
# #         `1. / wavefront.get_npixels()` and the correct inverse 
# #         mode factor is `wavefront.get_npixels()`
# #         """
# #         wavefront = self.get_utility().get_utility().construct() 
# #         forward = self.get_utility().construct(inverse = False)
# #         backward = self.get_utility().construct(inverse = True)

# #         assert forward._normalising_factor(wavefront) == \
# #             wavefront.get_npixels()
# #         assert backward._normalising_factor(wavefront) == \
# #             1. / wavefront.get_npixels()


# class TestCartesianMFT(UtilityUser):
#     """
#     Tests the concrete methods of the concrete class CartesianMFT. 
#     The concrete methods that are tested are.
#     - __init__
#     - __call__
#     - _normalising_factor()
#     - propagate()
#     - get_focal_length()
#     - get_pixel_offsets()

#     Attributes 
#     ----------
#     utility : CartesianMFTUtility
#         Provides access to safe constructors for both CartesianWavefronts
#         and CartesianMFT objects as well as other helper functions.
#     """
#     utility : CartesianMFTUtility = CartesianMFTUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Tests that the constructor correctly initialises all of the 
#         fields of the constructor. 
#         """
#         propagator = self.get_utility().construct()

#         assert propagator.get_focal_length() == \
#             self.get_utility().get_focal_length()
#         assert propagator.get_pixel_scale_out() == \
#             self.get_utility().get_pixel_scale_out() 
#         assert propagator.get_npixels_out() == \
#             self.get_utility().get_npixels_out()
#         assert propagator.is_inverse() == \
#             self.get_utility().is_inverse()


#     def test_get_nfringes(self : Tester) -> None:
#         """
#         Tests that the number of fringes is implemented correctly. 
#         Simply repeats the calculation here, mainly for formalism.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()

#         FRINGES = wavefront.get_pixel_scale() * wavefront.get_npixels() * \
#             propagator.get_pixel_scale_out() * propagator.get_npixels_out() /\
#             propagator.get_focal_length() / wavefront.get_wavelength()

#         assert propagator.get_nfringes(wavefront) == FRINGES


# #     def test_normalising_factor(self : Tester) -> None:
# #         """
# #         Tests that the normalising factor is correctly implemented. 
# #         The normalising factor is the far field normalisation factor.
# #         and is the same for the forward and backward directions. 
# #         """
# #         propagator = self.get_utility().construct()
# #         wavefront = self.get_utility().get_utility().construct()

# #         # TODO: This could be implemented in the src code
# #         NORMALISING_FACTOR = propagator.get_nfringes(wavefront) / \
# #             propagator.get_npixels_out() / \
# #             wavefront.get_npixels()

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(1., propagator\
# #                 ._normalising_factor(wavefront) / NORMALISING_FACTOR)


#     def test_get_focal_length(self : Tester) -> None:
#         """
#         Tests that the `get_focal_length` method correctly
#         tracks the state of the object.
#         """
#         SHORT = 0.01
#         LONG = 1.0

#         short_focal_length = self\
#             .get_utility()\
#             .construct(focal_length = SHORT)

#         long_focal_length = self\
#             .get_utility()\
#             .construct(focal_length = LONG)

#         assert short_focal_length.get_focal_length() == SHORT
#         assert long_focal_length.get_focal_length() == LONG


# #     def test_get_pixel_offsets(self : Tester) -> None:
# #         """
# #         Tests that the get_pixel_offsets function correctly generates
# #         the wavefront offsets in picels. 
# #         """
# #         OFFSET_RADIANS = numpy.array([0., 1.])

# #         propagator = self\
# #             .get_utility()\
# #             .construct()

# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()\
# #             .set_offset(OFFSET_RADIANS)

# #         OFFSET_PIXELS = OFFSET_RADIANS * \
# #             self.get_utility().get_focal_length() /\
# #             self.get_utility().get_pixel_scale_out()

#         # assert (propagator.get_pixel_offsets(wavefront) == \
#         #     OFFSET_PIXELS).all()


#     def test_propagate(self : Tester) -> None:
#         """
#         Checks that the propagate function is correctly assigned in 
#         the constructor.
#         """
#         wavefront = self.get_utility().get_utility().construct()

#         forwards = self.get_utility().construct(inverse = False)
#         backwards = self.get_utility().construct(inverse = True)

# #         is_forwards_correct = self\
# #             .get_utility()\
# #             .approx(forwards.propagate(wavefront), 
# #                 forwards._normalising_factor(wavefront) * \
# #                     forwards._fourier_transform(wavefront))\
# #             .all()

# #         is_backwards_correct = self\
# #             .get_utility()\
# #             .approx(backwards.propagate(wavefront),
# #                 backwards._normalising_factor(wavefront) * \
# #                     backwards._inverse_fourier_transform(wavefront))\
# #             .all()

#         # assert is_forwards_correct
#         # assert is_backwards_correct


#     def test_physical_mft(self : Tester) -> None:
#         """
#         Checks the __call__ method can be made without errors and 
#         repeats the calculation in the function namespace to check 
#         for correctness.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()
        
#         print(type(wavefront))

#         OUTPUT_FIELD = propagator.propagate(wavefront)

#         # output_field = propagator({"Wavefront": wavefront})\
#         #     ["Wavefront"].get_phasor()
#         output_field = propagator(wavefront).get_phasor()

#         is_correct = self\
#             .get_utility()\
#             .approx(1., OUTPUT_FIELD / output_field)\
#             .all()

#         assert is_correct


# class TestCartesianFFT(UtilityUser):
#     """
#     Tests the concrete methods of the CartesianMFT class. The concrete 
#     methods tested are. 
#     - __init__
#     - __call__
#     - get_focal_length()
#     - get_pixel_scale_out()

#     Attributes
#     ----------
#     utility : CartesianFFTUtility
#         Provides access to safe constructors for both CartesianWavefronts
#         and CartesianMFT objects as well as other helper functions.
#     """
#     utility : CartesianFFTUtility = CartesianFFTUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Tests that all the parameters of the CartesianFFT are correctly
#         initialised in the constructor. The parameters are:
#         - inverse : bool
#         - focal_length : float
#         """
#         propagator = self.get_utility().construct()

#         assert propagator.is_inverse() == self.get_utility().is_inverse()
#         assert propagator.get_focal_length() == \
#             self.get_utility().get_focal_length()
        

#     def test_get_focal_length(self : Tester) -> None:   
#         """
#         Tests that the `get_focal_length` method correctly
#         tracks the state of the object.
#         """
#         SHORT = 0.01
#         LONG = 1.0

#         short_focal_length = self\
#             .get_utility()\
#             .construct(focal_length = SHORT)

#         long_focal_length = self\
#             .get_utility()\
#             .construct(focal_length = LONG)

#         assert short_focal_length.get_focal_length() == SHORT
#         assert long_focal_length.get_focal_length() == LONG


#     def test_get_pixel_scale_out(self : Tester) -> None:
#         """
#         Repeats the calculation of the pixel_scale_out to make sure
#         that it is correct.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()

#         PIXEL_SCALE = wavefront.get_wavelength() * \
#             propagator.get_focal_length() / \
#             (wavefront.get_pixel_scale() * \
#                 wavefront.get_npixels())

#         assert PIXEL_SCALE == propagator.get_pixel_scale_out(wavefront)


#     def test_propagate(self : Tester) -> None:
#         """
#         Checks that `_propagate` is correctly assigned in the 
#         constructor, but does not check for correctness.
#         """
#         wavefront = self.get_utility().get_utility().construct()

#         forwards = self.get_utility().construct(inverse = False)
#         backwards = self.get_utility().construct(inverse = True)

# #         is_forwards_correct = self\
# #             .get_utility()\
# #             .approx(forwards.propagate(wavefront), 
# #                 forwards._normalising_factor(wavefront) * \
# #                     forwards._fourier_transform(wavefront))\
# #             .all()

# #         is_backwards_correct = self\
# #             .get_utility()\
# #             .approx(backwards.propagate(wavefront),
# #                 backwards._normalising_factor(wavefront) * \
# #                     backwards._inverse_fourier_transform(wavefront))\
# #             .all()

# #         assert is_forwards_correct
# #         assert is_backwards_correct


#     def test_physical_fft(self : Tester) -> None:
#         """
#         Tests that the __call__ method is made without error and 
#         returns the correct result.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()

#         OUTPUT_FIELD = propagator.propagate(wavefront)

#         # output_field = propagator({"Wavefront": wavefront})\
#         #     ["Wavefront"].get_phasor()
#         output_field = propagator(wavefront).get_phasor()

#         # Because the test wavefront has uniform electric field 
#         # the fourier transform is only non-zero in the top 
#         # left pixel. 
#         is_correct = self\
#             .get_utility()\
#             .approx(OUTPUT_FIELD, output_field)\
#             .all()

#         assert is_correct


# # class TestCartesianFresnel(UtilityUser):
# #     """
# #     Tests the concrete methods of the `CartesianFresnel` propagator.
# #     The concrete methods tested are.
# #     - __init__
# #     - __call__
# #     - propagate()
# #     - get_focal_length()
# #     - get_focal_shift()
# #     - nfringes()

# #     Some methods are not tested due to faith in the mathematical 
# #     process. These methods are:
# #     - quadratic_phase()
# #     - thin_lens()

# #     Attributes
# #     ----------
# #     utility : CartesianFresnelUtility
# #         A utility that provides access to safe testing constructors
# #         and other helpful functionalty.
# #     """
# #     utility : CartesianFresnelUtility = CartesianFresnelUtility()


# #     def test_constructor(self : Tester) -> None:
# #         """
# #         Checks that the constructor correctly initialises all of the
# #         relevant fields. The relevant fields are:
# #         - inverse : bool
# #         - focal_length : float
# #         - focal_shift : float
# #         - npixels_out : int
# #         - pixel_scale_out : float
# #         """
# #         propagator = self.get_utility().construct()
        
# #         assert propagator.is_inverse() == \
# #             self.get_utility().is_inverse()
# #         assert propagator.get_focal_length() == \
# #             self.get_utility().get_focal_length()
# #         assert propagator.get_focal_shift() == \
# #             self.get_utility().get_focal_shift() 
# #         assert propagator.get_npixels_out() == \
# #             self.get_utility().get_npixels_out() 
# #         assert propagator.get_pixel_scale_out() == \
# #             self.get_utility().get_pixel_scale_out()


# #     def test_get_focal_length(self : Tester) -> None:
# #         """
# #         Tests that the `get_focal_length` method correctly tracks 
# #         the state of the class.
# #         """
# #         SHORT = 0.01
# #         LONG = 1.0

# #         short_focal_length = self\
# #             .get_utility()\
# #             .construct(focal_length = SHORT)

# #         long_focal_length = self\
# #             .get_utility()\
# #             .construct(focal_length = LONG)

# #         assert short_focal_length.get_focal_length() == SHORT
# #         assert long_focal_length.get_focal_length() == LONG


# #     def test_get_focal_shift(self : Tester) -> None:
# #         """
# #         Tests that the `get_focal_shift` method is correctly 
# #         reporting the state of the object.
# #         """
# #         POSITIVE = .1
# #         NEGATIVE = -.1
        
# #         positive = self.get_utility().construct(focal_shift = POSITIVE)
# #         negative = self.get_utility().construct(focal_shift = NEGATIVE)

# #         assert positive.get_focal_shift() == POSITIVE
# #         assert negative.get_focal_shift() == NEGATIVE


# #     def test_nfringes(self : Tester) -> None:
# #         """
# #         Implements the full calculation in a single scope to check
# #         for correctness. This is nessecary because the implementation
# #         makes use of super.
# #         """
# #         propagator = self.get_utility().construct() 
# #         wavefront = self.get_utility().get_utility().construct()

# #         size_in = wavefront.get_npixels() * \
# #             wavefront.get_pixel_scale()
# #         size_out = propagator.get_npixels_out() * \
# #             propagator.get_pixel_scale_out()
# #         propagation_distance = propagator.get_focal_length() + \
# #             propagator.get_focal_shift()
# #         focal_ratio = propagator.get_focal_length() / \
# #             propagation_distance

# #         nfringes = focal_ratio * size_in * size_out / \
# #             propagator.get_focal_length() / wavefront.get_wavelength()

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(1., propagator\
# #                 .nfringes(wavefront) / nfringes)

# #         assert is_correct  


# #     # TODO: Forwards propagation is the only supported direction.
# #     # TODO: Change heirachy have the fresnel in its own maybe.
# #     def test_propagate(self : Tester) -> None:
# #         """
# #         Passes some simple inputs and checks for `numpy.nan` 
# #         `numpy.inf`. 
# #         """
# #         propagator = self.get_utility().construct()
# #         wavefront = self.get_utility().get_utility().construct()

# #         electric_field = propagator.propagate(wavefront)

# #         assert not numpy.isnan(electric_field).any()
# #         assert not numpy.isnan(electric_field).any()


# #     def test_physical_fresnel(self : Tester) -> None:
# #         """
# #         tests that the `__call__` method is correctly activated and
# #         makes sure that the output is instantiated in terms of
# #         the correct operations.
# #         """
# #         propagator = self.get_utility().construct()
# #         wavefront = self.get_utility().get_utility().construct()

# #         OUTPUT = propagator.propagate(wavefront)

# #         output = propagator({"Wavefront": wavefront})["Wavefront"]\
# #             .get_phasor()

# #         is_correct = self\
# #             .get_utility()\
# #             .approx(1., output / OUTPUT)\
# #             .all()         

# #         assert is_correct


# class TestAngularMFT(UtilityUser):
#     """
#     Tests the `AngularMFT` class concrete methods. These methods 
#     are.
#     - __init__()
#     - __call__()
#     - _proagate()

#     Some methods have not been tested because they are purely 
#     methematical. These functions are.
#     - nfringes()
#     - get_pixel_offsets()

#     Attributes
#     ----------
#     utility : AngularMFTUtility 
#         Contain helpful constructors and other methods.
#     """
#     utility : AngularMFTUtility = AngularMFTUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Checks that the state of the class is correctly initialised 
#         by the constructor. The states is
#         - inverse : bool
#         - npixels_out : int
#         - pixel_scale_out : float
#         """
#         propagator = self.get_utility().construct()

#         assert propagator.is_inverse() == \
#             self.get_utility().is_inverse()
#         assert propagator.get_npixels_out() == \
#             self.get_utility().get_npixels_out()
#         assert propagator.get_pixel_scale_out() == \
#             self.get_utility().get_pixel_scale_out()


#     def test_propagate(self : Tester) -> None:
#         """
#         Checks that `_propagate` is correctly assigned in the 
#         constructor, but does not check for correctness.
#         """
#         wavefront = self.get_utility().get_utility().construct()

#         forwards = self.get_utility().construct(inverse = False)
#         backwards = self.get_utility().construct(inverse = True)

# #         is_forwards_correct = self\
# #             .get_utility()\
# #             .approx(forwards.propagate(wavefront), 
# #                 forwards._normalising_factor(wavefront) * \
# #                     forwards._fourier_transform(wavefront))\
# #             .all()

# #         is_backwards_correct = self\
# #             .get_utility()\
# #             .approx(backwards.propagate(wavefront),
# #                 backwards._normalising_factor(wavefront) * \
# #                     backwards._inverse_fourier_transform(wavefront))\
# #             .all()

# #         assert is_forwards_correct
# #         assert is_backwards_correct  


#     def test_angular_mft(self : Tester) -> None:
#         """
#         tests that the `__call__` method is correctly activated and
#         makes sure that the output is instantiated in terms of
#         the correct operations.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()

#         OUTPUT = propagator.propagate(wavefront)

#         # output = propagator({"Wavefront": wavefront})["Wavefront"]\
#         #     .get_phasor()
#         output = propagator(wavefront).get_phasor()

#         is_correct = self\
#             .get_utility()\
#             .approx(1., output / OUTPUT)\
#             .all()    

#         assert is_correct


# class TestAngularFFT(UtilityUser):
#     """
#     Tests the concrete methods of the `AngularFFT` class. The methods 
#     tested are:
#     - __init__()
#     - __call__()
#     - propagate()

#     Some functions are not tested because they are purely mathematical 
#     entities. These functions are.
#     - pixel_scale_out()

#     Attributes
#     ----------
#     Utility : AngularFFT = AngularFFTUtility()
#         A collection of safe cosntructors and helpful functions.
#     """
#     utility : AngularFFTUtility = AngularFFTUtility()


#     def test_constructor(self : Tester) -> None:
#         """
#         Checks that the state of the class is correctly instantiated 
#         by the constructor. The state of the class consists of.
#         - inverse : bool
#         """
#         propagator = self.get_utility().construct()

#         assert propagator.is_inverse() == \
#             self.get_utility().is_inverse()


#     def test_propagate(self : Tester) -> None:
#         """
#         Checks that `_propagate` is correctly assigned in the 
#         constructor, but does not check for correctness.
#         """
#         wavefront = self.get_utility().get_utility().construct()

#         forwards = self.get_utility().construct(inverse = False)
#         backwards = self.get_utility().construct(inverse = True)

# #         is_forwards_correct = self\
# #             .get_utility()\
# #             .approx(forwards.propagate(wavefront), 
# #                 forwards._normalising_factor(wavefront) * \
# #                     forwards._fourier_transform(wavefront))\
# #             .all()

# #         is_backwards_correct = self\
# #             .get_utility()\
# #             .approx(backwards.propagate(wavefront),
# #                 backwards._normalising_factor(wavefront) *\
# #                     backwards._inverse_fourier_transform(wavefront))\
# #             .all()

# #         assert is_forwards_correct
# #         assert is_backwards_correct  


#     def test_angular_fft(self : Tester) -> None:
#         """
#         tests that the `__call__` method is correctly activated and
#         makes sure that the output is instantiated in terms of
#         the correct operations.
#         """
#         propagator = self.get_utility().construct()
#         wavefront = self.get_utility().get_utility().construct()

#         OUTPUT = propagator.propagate(wavefront)

#         # output = propagator({"Wavefront": wavefront})["Wavefront"]\
#         #     .get_phasor()
#         output = propagator(wavefront).get_phasor()

#         is_correct = self\
#             .get_utility()\
#             .approx(output, OUTPUT)\
#             .all()    

#         assert is_correct  


# # class TestAngularFresnel(UtilityUser):
# #     """
# #     This class will hold tests for the concrete methods of the 
# #     AngularFresnel wavefront when the class is written. For now 
# #     this is a stub.
# #     """
# #     pass


# # class TestGaussianPropagator(UtilityUser):
# #     """
# #     Holds tests for the concrete methods of the GaussianPropagator 
# #     class. These methods are:
# #     - __init__()
# #     - __call__()
# #     - propagate()
# #     - _outside_to_outside()
# #     - _outside_to_inside()
# #     - _inside_to_outside()
# #     - _inside_to_inside()

# #     Some methods have not been tested out of mathematical faith. These 
# #     methods are:
# #     - _waist_to_spherical()
# #     - _spherical_to_waist()
# #     - _planar_to_planar()

# #     Due to the increased complexity of many of these methods they
# #     are only checked for `numpy.nan` and `numpy.inf` values. With
# #     correctness testing relegated to the end to end tests.

# #     Attributes
# #     ----------
# #     utility : GaussianProapagatorUtility
# #         A collection of helpful constructors and functions for testing.
# #     """
# #     utility : GaussianPropagatorUtility = GaussianPropagatorUtility()


# #     def test_propagate(self : Tester) -> None:
# #         """
# #         Tests full branch coverage and then the boundary cases 
# #         distance == -numpy.inf, distance == numpy.inf and distance 
# #         == 0. The branches covered are:
        
# #         inside_to_inside
# #         inside_to_outside
# #         outside_to_inside
# #         outside_to_outside
# #         """
# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()

# #         rayleigh_distance = wavefront.rayleigh_distance()

# #         inside_to_inside = self\
# #             .get_utility()\
# #             .construct(distance = rayleigh_distance / 2.)\
# #                 ({"Wavefront": wavefront})["Wavefront"]\
# #             .get_phasor()

# #         inside_to_outside = self\
# #             .get_utility()\
# #             .construct(distance = rayleigh_distance + 1.)\
# #                 ({"Wavefront": wavefront})["Wavefront"]\
# #             .get_phasor()

# #         outside_to_inside = self\
# #             .get_utility()\
# #             .construct(rayleigh_distance)({"Wavefront": wavefront\
# #                 .set_position(-rayleigh_distance - 1.)})["Wavefront"]\
# #             .get_phasor()

# #         outside_to_outside = self\
# #             .get_utility()\
# #             .construct(2 * (rayleigh_distance + 1.))({"Wavefront": wavefront\
# #                 .set_position(-rayleigh_distance - 1.)})["Wavefront"]\
# #             .get_phasor()

# #         assert not numpy.isnan(inside_to_inside).any()
# #         assert not numpy.isnan(inside_to_outside).any()
# #         assert not numpy.isnan(outside_to_inside).any()
# #         assert not numpy.isnan(outside_to_outside).any()

# #         assert not numpy.isinf(inside_to_inside).any()
# #         assert not numpy.isinf(inside_to_outside).any()
# #         assert not numpy.isinf(outside_to_inside).any()
# #         assert not numpy.isinf(outside_to_outside).any()


# #     def test_outside_to_outside(self : Tester) -> None:
# #         """
# #         Tests the three boundary cases -numpy.inf, and numpy.inf as
# #         well as a negative and a positive valid input.
# #         """
# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()\
# #             .set_phase_radius(1.)

# #         rayleigh_distance = wavefront.rayleigh_distance()

# #         negative = self\
# #             .get_utility()\
# #             .construct(- 2. * rayleigh_distance - 1.)\
# #             .outside_to_outside(wavefront\
# #                 .set_position(rayleigh_distance + .5),
# #                 - 2. * rayleigh_distance - 1.)\
# #             .get_phasor()

# #         positive = self\
# #             .get_utility()\
# #             .construct(2. * rayleigh_distance + 1.)\
# #             .outside_to_outside(wavefront\
# #                 .set_position(- rayleigh_distance - .5), 
# #                 2. * rayleigh_distance + 1.)\
# #             .get_phasor()

# #         assert not numpy.isnan(negative).any()
# #         assert not numpy.isnan(positive).any()

# #         assert not numpy.isinf(negative).any()
# #         assert not numpy.isinf(positive).any()


# #     def test_outside_to_inside(self : Tester) -> None:
# #         """
# #         Tests negative and a positive valid input.
# #         """
# #         # TODO: Work out if I need to be moving the wavefront
# #         # back by the rayleigh_distance in the outside tests
# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()\
# #             .set_phase_radius(1.)

# #         rayleigh_distance = wavefront.rayleigh_distance()

# #         negative = self\
# #             .get_utility()\
# #             .construct(- rayleigh_distance - 0.01)\
# #             .outside_to_outside(wavefront\
# #                 .set_position(rayleigh_distance + 0.01),
# #                 - rayleigh_distance - 0.01)\
# #             .get_phasor()

# #         positive = self\
# #             .get_utility()\
# #             .construct(rayleigh_distance + 0.01)\
# #             .outside_to_outside(wavefront\
# #                 .set_position(- rayleigh_distance - 0.01),
# #                 rayleigh_distance + 0.01)\
# #             .get_phasor()
        
# #         assert not numpy.isnan(negative).any()
# #         assert not numpy.isnan(positive).any()

# #         assert not numpy.isinf(negative).any()
# #         assert not numpy.isinf(positive).any()
    

# #     def test_inside_to_outside(self : Tester) -> None:
# #         """
# #         Tests the three boundary cases -numpy.inf, and numpy.inf as
# #         well as a negative and a positive valid input.
# #         """
# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()\
# #             .set_phase_radius(1.)

# #         rayleigh_distance = wavefront.rayleigh_distance()

# #         negative = self\
# #             .get_utility()\
# #             .construct(- rayleigh_distance - 0.01)\
# #             .outside_to_outside(wavefront, - rayleigh_distance - 0.01)\
# #             .get_phasor()

# #         positive = self\
# #             .get_utility()\
# #             .construct(rayleigh_distance + 0.01)\
# #             .outside_to_outside(wavefront, rayleigh_distance + 0.01)\
# #             .get_phasor()

# #         assert not numpy.isnan(negative).any()
# #         assert not numpy.isnan(positive).any()
        
# #         assert not numpy.isinf(negative).any()
# #         assert not numpy.isinf(positive).any()


# #     def test_inside_to_inside(self : Tester) -> None:
# #         """
# #         Tests the three boundary cases -numpy.inf, and numpy.inf as
# #         well as a negative and a positive valid input.
# #         """
# #         wavefront = self\
# #             .get_utility()\
# #             .get_utility()\
# #             .construct()\
# #             .set_phase_radius(1.)

# #         rayleigh_distance = wavefront.rayleigh_distance()

# #         negative = self\
# #             .get_utility()\
# #             .construct(- rayleigh_distance + 0.01)\
# #             .outside_to_outside(wavefront, - rayleigh_distance + 0.01)\
# #             .get_phasor()

# #         positive = self\
# #             .get_utility()\
# #             .construct(rayleigh_distance - 0.01)\
# #             .outside_to_outside(wavefront, rayleigh_distance - 0.01)\
# #             .get_phasor()
                
# #         assert not numpy.isnan(negative).any()
# #         assert not numpy.isnan(positive).any()
        
# #         assert not numpy.isinf(negative).any()
# #         assert not numpy.isinf(positive).any()
