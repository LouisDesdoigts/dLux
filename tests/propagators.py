"""
tests/propagators.py
--------------------
Contains the unit tests for the `Propagator` objects coded in the 
`propagators.py` file. Tests are comprehensive unit tests based of 
the `UtilityUser` framework stored in the `tests/utilities.py` file.


Tests are organised one `TestX` class per `PropagatorX` class in 
`src/propagators.py` and can be run in the terminal using 
`pytest tests/propagators.py`. It is recommended that the `--quiet`
or `-q` flag is used to supress extra stack traces.
"""
__author__ = "Jordan Dennis"
__date__ = "02/07/2022"


import pytest
import jax.numpy as numpy
import dLux
import typing
from .utilities.py import *


Tester = typing.NewType("Tester", UtilityUser)


class TestPropagator(UtilityUser):
    """
    Tests the concrete methods of the abstract propagator class.
    The concrete methods are:
     - __init__()
     - _get_pixel_grid()
     - _get_pixel_positions()
     - is_inverse()

    Attributes
    ----------
    utility : PropagatorUtility
    """
    utility : PropagatorUtility = PropagatorUtility()


    def test_constructor(self : Tester) -> None:
        """
        Tests that the constructor correctly initialialises the 
        un-traced `inverse` attribute of the `Propagator` class.
        """
        false_propagator = self\
            .get_utility()\
            .construct(inverse = False)

        true_propagator = self\
            .get_utility()\
            .construct(inverse = True)

        assert false_constructor.is_inverse() == False
        assert true_propagator.is_inverse() == True


    def test_is_inverse(self : Tester) -> None:
        """
        Makes sure that the `is_inverse` function correctly reflects 
        the state of the `Propagator`
        """
        true_propagator = self\
            .get_utility()\
            .construct(inverse = True)

        false_propagator = self\
            .get_utility()\
            .construct(inverse = False)

        assert true_propagator.is_inverse() == True
        assert false_propagator.is_inverse() == False


    def test_get_pixel_grid(self : Tester) -> None:
        """
        Checks that the pixel grid is generated correctly in the 
        output plane. The implementation is considered correct if
        it has the correct maximum and minimum values and is the 
        correct shape.
        """
        utility = self.get_utility()

        pixel_grid = utility\
            .construct()\
            ._get_pixel_grid(
                utility.get_pixel_offset(), 
                utility.get_pixel_scale(), 
                utility.get_number_of_pixels())

        assert pixel_grid.max() == \
            utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.min() == \
            - utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.shape == (2, utility.get_number_of_pixels(),
            utility.get_number_of_pixels)        


    def test_get_pixel_positions(self : Tester) -> None:
        """
        Checks that the pixel positions are correctly generated.
        The pixel positions are considered correct if they have 
        the correct maximum, minimum and shape.
        """    
        utility = self.get_utility()

        pixel_positions = utility\
            .construct()\
            ._get_pixel_positions()    

        assert pixel_positions.max() == \
            utility.get_nuber_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.min() == \
            - utility.get_number_of_pixels() // 2 * \
            utility.get_pixel_scale()

        assert pixel_grid.shape == (2, utility.get_number_of_pixels(),
            utility.get_number_of_pixels)    


class TestVariableSamplingPropagator(UtilityUser):
    """
    Tests the concrete methods of the `VariableSamplingPropagator`
    abstract class. The concrete methods are:
     - __init__()
     - _matrix_fourier_transform()
     - _fourier_transform()
     - _inverse_fourier_transform()
     - _normalising_factor()
     - get_pixel_scale_out()
     - get_pixels_out()
     - generate_twiddle_factors()

    Attributes
    ----------
    utility : VariableSamplingUtility
        A testing utility that generates safe test cases.

    Inherits
    --------
    : UtilityUser
        Provides acces to the `get_utitlity` method, broadcast 
        accross all test classes.
    """
    utility : VariableSamplingUtility = VariableSamplingUtility()


    def test_constructor(self : Tester) -> None:
        """
        Checks that all the appropriate fields are correctly 
        initialised by the constructor. These fields are:
         - pixel_scale_out : float
         - pixels_out : int
         - inverse : bool
        """
        propagator = self.get_utility().construct()

        assert propagator.is_inverse() == utility.get_inverse()
        assert propagator.get_pixels_out() == utility.get_pixels_out()
        assert propagator.get_pixel_scale_out() == \
            utility.get_pixel_scale_out()


    def test_get_pixel_scale_out(self : Tester) -> None:
        """
        Tests that the `get_pixel_scale_out` method correctly 
        reports the state of the object. 
        """
        OLD_PIXEL_SCALE = 0.7
        NEW_PIXEL_SCALE = 0.5

        old_propagtor = self\
            .get_utility()\
            .construct(pixel_scale_out = OLD_PIXEL_SCALE)

        new_propagator = self\
            .get_utility()\
            .construct(pixel_scale_out = NEW_PIXEL_SCALE)

        assert old_propagator.get_pixel_scale_out() == OLD_PIXEL_SCALE
        assert new_propagator.get_pixel_scale_out() == NEW_PIXEL_SCALE


    def test_get_pixels_out(self : Tester) -> None:
        """
        Tests that `get_pixels_out` correctly tracks the state of 
        the object. 
        """
        NEW_PIXELS = 100
        OLD_PIXELS = 50

        old_propagator = self\
            .get_utility()\
            .construct(pixels_out = OLD_PIXELS)

        new_propagator = self\
            .get_utility()\
            .construct(pixels_out = NEW_PIXELS)

        assert old_propagator.get_pixels_out() == OLD_PIXELS
        assert new_propagator.get_pixels_out() == NEW_PIXELS


    def test_generate_twiddle_factors(self : Tester) -> None:
        """
        Tests that the correct twiddle factors are getting 
        generated. This is done by direct comparison for a 
        simple case.
        """
        TWIDDLE_FACTORS : Array = 0.5 * numpy.array([
            [1. + 0.j, 1. + 0.j, 1. + 0.j, 1. + 0.j],
            [1. + 0.j, 0. - 1.j, -1. + 0.j, 0. + 1.j],
            [1. + 0.j, -1. + 0.j, 1. + 0.j, -1. + 0.j],
            [1. + 0.j, 0. + 0.j, -1. + 0.j, 0. - 1.j]])
        OFFSET : Array = numpy.array([2., 2.])
        PIXELS : int = 4
        PIXEL_SCALES : float = 1.
        SIGN : int = 1

        twiddle_factors = self\
            .get_utility()\
            .construct()\
            ._generate_twiddle_factors(OFFSETS, PIXEL_SCALES, 
                PIXESL, SIGN)

         is_correct = self\
            .get_utility()\
            .approx(twiddle_factors, TWIDDLE_FACTORS)\
            .all()

        assert is_correct


    def test_matrix_fourier_transform(self : Tester) -> None:
        """
        Tests that the discrete foruier transform is correct for
        a simple case. 
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = 1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = 1))\
            .all()

        assert is_correct


    def test_fourier_transform(self : Tester) -> None:
        """
        Tests that the Fourier transform of the wavefront is 
        correctly calculated. 
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = 1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = 1))\
            .all()

        assert is_correct


    def test_inverse_fourier_transform(self : Tester) -> None:
        """
        Tests that the inverse fourier transform is correctly 
        calculated.
        """
        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\ # Not sure about the nesting here but it
            .construct()    # will do for not.

        fourier_transform = \
            propagator._matrix_fourier_transform(wavefront, sign = -1)

        is_correct = self\
            .get_utility()\
            .approx(fourier_transform,
                self.get_utility()\
                    .louis_transform(wavefront, sign = -1))\
            .all()

        assert is_correct


    def test_normalising_factor(self : Tester) -> None:
        """
        Tests that the normalising factor is correctly implemented. 
        The normalising factor is the far field normalisation factor.
        and is the same for the forward and backward directions. 
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        NORMALISING_FACTOR = numpy.exp(
            propagator._number_of_fringes() * \
            propagator.get_pixels_out() / \
            wavefront.number_of_pixels())

        assert NORMALISING_FACTOR == \
            propagator._normalising_factor(wavefront) 


    def test_number_of_fringes(self : Tester) -> None:
        """
        Tests that the number of fringes is implemented correctly. 
        Simply repeats the calculation here, mainly for formalism.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        FRINGES = wavefront.get_pixel_scale() * wavefront.number_of_pixels() * \
            propagator.get_pixel_scale_out() * propagator.get_pixels_out() /\
            propagator.get_focal_length() / wavefront.get_wavefront()

        assert propagator.number_of_fringes(wavefront) == FRINGES


class TestFixedSamplingPropagator(UtilityUser):
    """
    Contains the tests for the abstract FixedSamplingPropagator class.
    The concrete methods tested:
    - fourier_transform()
    - inverse_fourier_transform()
    - normalising_factor()
    """
    def test_fourier_transform(self : Tester) -> None:
        """
        Tests the fourier transform of the FixedSamplingPropagator.
        The Fourier transform is based on the inbuilt `numpy.fft`
        so the tests are simple and mostly a formaility,
        """
        propagtor = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        CORRECT = numpy.fft.fft2(numpy.ifftshift(
            wavefront.get_complex_form()))

        is_correct = self\
            .get_utility()\
            .approx(propagator._fourier_transform(wavefront), CORRECT)\
            .all()

        assert is_correct


    def test_inverse_fourier_transfrom(self : Tester) -> None:
        """
        Tests the inverse fourier transform of the FixedSamplingPropagator
        As above this method is based on the inbuilt `numpy.fft` module so
        the tests are mostly a formality.
        """
        propagtor = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        CORRECT = numpy.fft.fftshift(numpy.ifft2(
            wavefront.get_complex_form()))

        is_correct = self\
            .get_utility()\
            .approx(propagator._inverse_fourier_transform(wavefront), 
                CORRECT)\
            .all()

        assert is_correct


    def test_normalising_factor(self : Tester) -> None:
        """
        Tests that the correct normalising factor is generated for both 
        the forward and reverse modes. The correct forward mode is 
        `1. / wavefront.number_of_pixels()` and the correct inverse 
        mode factor is `wavefront.number_of_pixels()`
        """
        wavefront = self.get_utility().get_utility().construct() 
        forward = self.get_utility().construct(inverse = False)
        backward = self.get_utility().construct(inverse = True)

        assert forward._normalising_factor(wavefront) == \
            1. / wavefront.number_of_pixels()
        assert backward._normalising_factor(wavefront) == \
            wavefront.number_of_pixels()



class TestPhysicalMFT(UtilityUser):
    """
    Tests the concrete methods of the concrete class PhysicalMFT. 
    The concrete methods that are tested are.
    - __init__
    - __call__
    - _normalising_factor()
    - _propagate()
    - get_focal_length()
    - get_pixel_offsets()

    Attributes 
    ----------
    utility : PhysicalMFTUtility
        Provides access to safe constructors for both PhysicalWavefronts
        and PhysicalMFT objects as well as other helper functions.
    """
    utility : PhysicalMFTUtility = PhysicalMFTUtility()


    def test_constructor(self : Tester) -> None:
        """
        Tests that the constructor correctly initialises all of the 
        fields of the constructor. 
        """
        propagator = self.get_utility().construct()

        assert propagator.get_focal_length() == \
            self.get_utility().get_focal_length()
        assert propagator.get_pixel_scale() == \
            self.get_utility().get_pixel_scale() 
        assert propagator.get_pixels_out() == \
            self.get_utility().get_pixels_out()
        assert propagator.is_inside() == \
            self.get_utility().is_inverse()


    def test_get_focal_length(self : Tester) -> None:
        """
        Tests that the `get_focal_length` method correctly
        tracks the state of the object.
        """
        SHORT = 0.01
        LONG = 1.0

        short_focal_length = self\
            .get_utility()\
            .construct(focal_length = SHORT)

        long_focal_length = self\
            .get_utility()\
            .construct(focal_length = LONG)

        assert short_focal_length.get_focal_length() == SHORT
        assert long_focal_length.get_focal_length() == LONG


    def test_get_pixel_offsets(self : Tester) -> None:
        """
        Tests that the get_pixel_offsets function correctly generates
        the wavefront offsets in picels. 
        """
        OFFSET_RADIANS = numpy.array([0., 1.])

        propagator = self\
            .get_utility()\
            .construct()

        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()\
            .set_offset(OFFSET_RADIANS)

        OFFSET_PIXELS = OFFSET_RADIANS * \
            self.get_utility().get_focal_length() /\
            self.get_utility().get_pixel_scale()

        assert propagator.get_pixel_offsets() == OFFSET_PIXELS


    def test_propagate(self : Tester) -> None:
        """
        Checks that the propagate function is correctly assigned in 
        the constructor.
        """
        forwards = self.get_utility().construct(inverse = False)
        backwards = self.get_utility().construct(inverse = True)

        assert forwards._propagate == forwards._fourier_transform
        assert bakcwards._propagate == backwards._inverse_fourier_transform


    def test_physical_mft(self : Tester) -> None:
        """
        Checks the __call__ method can be made without errors and 
        repeats the calculation in the function namespace to check 
        for correctness.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        OUTPUT_FIELD = propagator._propagate(wavefront) * \
            propagator._normalising_factor(wavefront)

        output_field = propagator({"Wavefront": wavefront})\
            ["Wavefront"].get_complex_form()

        is_correct = self\
            .get_utility()\
            .approx(OUTPUT_FIELD, output_field)\
            .all()

        assert is_correct


class TestPhysicalFFT(UtilityUser):
    """
    Tests the concrete methods of the PhysicalMFT class. The concrete 
    methods tested are. 
    - __init__
    - __call__
    - get_focal_length()
    - get_pixel_scale_out()

    Attributes
    ----------
    utility : PhysicalFFTUtility
        Provides access to safe constructors for both PhysicalWavefronts
        and PhysicalMFT objects as well as other helper functions.
    """
    utility : PhysicalFFTUtility = PhysicalFFTUtility()


    def test_constructor(self : Tester) -> None:
        """
        Tests that all the parameters of the PhysicalFFT are correctly
        initialised in the constructor. The parameters are:
        - inverse : bool
        - focal_length : float
        """
        propagator = self.get_utility().construct()

        assert propagator.is_inverse() == self.get_utility().is_inverse()
        assert propagator.get_focal_length() == \
            self.get_utility().get_focal_length()
        

    def test_get_focal_length(self : Tester) -> None:   
        """
        Tests that the `get_focal_length` method correctly
        tracks the state of the object.
        """
        SHORT = 0.01
        LONG = 1.0

        short_focal_length = self\
            .get_utility()\
            .construct(focal_length = SHORT)

        long_focal_length = self\
            .get_utility()\
            .construct(focal_length = LONG)

        assert short_focal_length.get_focal_length() == SHORT
        assert long_focal_length.get_focal_length() == LONG
  
 
    def test_get_pixel_scale_out(self : Tester) -> None:
        """
        Repeats the calculation of the pixel_scale_out to make sure
        that it is correct.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        PIXEL_SCALE = wavefront.get_wavelength() * \
            propagator.get_focal_length() / \
            wavefront.get_pixel_scale() * wavefront.number_of_pixels()

        assert PIXEL_SCALE == propagator.get_pixel_scale_out(wavefront)
                   

    def test_propagate(self : Tester) -> None:
        """
        Checks that `_propagate` is correctly assigned in the 
        constructor, but does not check for correctness.
        """
        forwards = self.get_utility().construct(inverse = False)
        backwards = self.get_utility().construct(inverse = True)

        assert forwards._propagate == forwards._fourier_transform
        assert backwards._propagate == backwards._inverse_fourier_transform


    def test_physical_fft(self : Tester) -> None:
        """
        Tests that the __call__ method is made without error and 
        returns the correct result.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        OUTPUT_FIELD = propagator._normalising_factor(wavefront) * \
            propagator._propagate(wavefront)

        output_field = propagator({"Wavefront": wavefront})\
            ["Wavefront"].get_complex_form()

        is_correct = self\
            .get_utility()\
            .approx(OUTPUT_FIELD, output_field)\
            .all()

        assert is_correct


class TestPhysicalFresnel(UtilityUser):
    """
    Tests the concrete methods of the `PhysicalFresnel` propagator.
    The concrete methods tested are.
    - __init__
    - __call__
    - _propagate()
    - get_focal_length()
    - get_focal_shift()
    - number_of_fringes()

    Some methods are not tested due to faith in the mathematical 
    process. These methods are:
    - quadratic_phase()
    - thin_lens()

    Attributes
    ----------
    utility : PhysicalFresnelUtility
        A utility that provides access to safe testing constructors
        and other helpful functionalty.
    """
    utility : PhysicalFresnelUtility = PhysicalFresnelUtility()


    def test_constructor(self : Tester) -> None:
        """
        Checks that the constructor correctly initialises all of the
        relevant fields. The relevant fields are:
        - inverse : bool
        - focal_length : float
        - focal_shift : float
        - pixels_out : int
        - pixel_scale_out : float
        """
        propagator = self.get_utility().construct()
        
        assert propagator.is_inverse() == \
            self.get_utility().is_inverse()
        assert propagator.get_focal_length() == \
            self.get_utility().get_focal_length()
        assert propagator.get_focal_shift() == \
            self.get_utility().get_focal_shift() 
        assert propagator.get_pixels_out() == \
            self.get_utility().get_pixels_out() 
        assert propagator.get_pixel_scale_out() == \
            self.get_utility().get_pixel_scale_out()


    def test_get_focal_length(self : Tester) -> None:
        """
        Tests that the `get_focal_length` method correctly tracks 
        the state of the class.
        """
        SHORT = 0.01
        LONG = 1.0

        short_focal_length = self\
            .get_utility()\
            .construct(focal_length = SHORT)

        long_focal_length = self\
            .get_utility()\
            .construct(focal_length = LONG)

        assert short_focal_length.get_focal_length() == SHORT
        assert long_focal_length.get_focal_length() == LONG


    def test_get_focal_shift(self : Tester) -> None:
        """
        Tests that the `get_focal_shift` method is correctly 
        reporting the state of the object.
        """
        POSITIVE = .1
        NEGATIVE = -.1
        
        positive = self.get_utility().construct(focal_shift = POSITIVE)
        negative = self.get_utility().construct(focal_shift = NEGATIVE)

        assert positive.get_focal_shift() == POSITIVE
        assert negative.get_focal_shift() == NEGATIVE


    def test_number_of_fringes(self : Tester) -> None:
        """
        Implements the full calculation in a single scope to check
        for correctness. This is nessecary because the implementation
        makes use of super.
        """
        propagator = self.get_utility().construct() 
        wavefront = self.get_utility().get_utility().construct()

        size_in = wavefront.number_of_pixels() * \
            wavefront.get_pixel_scale()
        size_out = propagator.get_pixels_out() * \
            propagator.get_pixel_scale_out()
        propagation_distance = propagator.get_focal_length() + \
            propagator.get_focal_shift()
        focal_ratio = propagator.get_focal_length() / \
            propagation_distance

        NUMBER_OF_FRINGES = focal_ratio * size_in * size_out / \
            propagator.get_focal_length() \ wavefront.get_wavelength()

        assert NUMBER_OF_FRINGES == propagator.number_of_fringes(wavefront)        


    # TODO: Forwards propagation is the only supported direction.
    def test_propagate(self : Tester) -> None:
        """
        Passes some simple inputs and checks for `numpy.nan` 
        `numpy.inf`. 
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        electric_field = propagator._propagate(wavefront)

        assert not electric_field.isnan().any()
        assert not electric_field.isinf().any()


    def test_physical_fresnel(self : Tester) -> None:
        """
        tests that the `__call__` method is correctly activated and
        makes sure that the output is instantiated in terms of
        the correct operations.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        OUTPUT = propagator._normalising_factor(wavefront) * \
            propagator._propagate(wavefront)

        output = propagator({"Wavefront": wavefront})["Wavefront"]\
            .get_complex_form()

        assert output == OUTPUT         


class TestAngularMFT(UtilityUser):
    """
    Tests the `AngularMFT` class concrete methods. These methods 
    are.
    - __init__()
    - __call__()
    - _proagate()

    Some methods have not been tested because they are purely 
    methematical. These functions are.
    - number_of_fringes()
    - get_pixel_offsets()

    Attributes
    ----------
    utility : AngularMFTUtility 
        Contain helpful constructors and other methods.
    """
    utility : AngularMFTUtility = AngularMFTUtility()


    def test_constructor(self : Tester) -> None:
        """
        Checks that the state of the class is correctly initialised 
        by the constructor. The states is
        - inverse : bool
        - pixels_out : int
        - pixel_scale_out : float
        """
        propagator = self.get_utility().construct()

        assert propagator.is_inverse() == \
            self.get_utility().is_inverse()
        assert propagator.get_pixels_out() == \
            self.get_utility().get_pixels_out()
        assert propagtor.get_pixel_scale_out() == \
            self.get_utility().get_pixel_scale_out()


    def test_propagate(self : Tester) -> None:
        """
        Tests that `_propagate` is correctly assigned in the 
        constructor.
        """
        forwards = self.get_utility().construct(inverse = False)
        backwards = self.get_utility().construct(inverse = True)

        assert forwards._propagate == forwards._fourier_transform
        assert backwards._propagate == backwards._inverse_fourier_transform


    def test_angular_mft(self : Tester) -> None:
        """
        tests that the `__call__` method is correctly activated and
        makes sure that the output is instantiated in terms of
        the correct operations.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        OUTPUT = propagator._normalising_factor(wavefront) * \
            propagator._propagate(wavefront)

        output = propagator({"Wavefront": wavefront})["Wavefront"]\
            .get_complex_form()

        assert output == OUTPUT     


class TestAngularFFT(UtilityUser):
    """
    Tests the concrete methods of the `AngularFFT` class. The methods 
    tested are:
    - __init__()
    - __call__()
    - _propagate()

    Some functions are not tested because they are purely mathematical 
    entities. These functions are.
    - pixel_scale_out()

    Attributes
    ----------
    Utility : AngularFFT = AngularFFTUtility()
        A collection of safe cosntructors and helpful functions.
    """
    utility : AngularFFTUtility = AngularFFTUtility()


    def test_constructor(self : Tester) -> None:
        """
        Checks that the state of the class is correctly instantiated 
        by the constructor. The state of the class consists of.
        - inverse : bool
        """
        propagator = self.get_utility().construct()

        assert propagator.is_inverse() == self.get_utility().is_invser()


    def test_propagate(self : Tester) -> None:
        """
        Checks that the propagate method of this class is correctly 
        assiged during construction.
        """
        forward = self.get_utility().construct(inverse = False)
        backward = self.get_utility().construct(inverse = True)

        assert forward._propagate == forward._fourier_transform
        assert backward._propagate == backward._inverse_fourier_transform


    def test_angular_fft(self : Tester) -> None:
        """
        tests that the `__call__` method is correctly activated and
        makes sure that the output is instantiated in terms of
        the correct operations.
        """
        propagator = self.get_utility().construct()
        wavefront = self.get_utility().get_utility().construct()

        OUTPUT = propagator._normalising_factor(wavefront) * \
            propagator._propagate(wavefront)

        output = propagator({"Wavefront": wavefront})["Wavefront"]\
            .get_complex_form()

        assert output == OUTPUT    


class TestAngularFresnel(UtilityUser):
    """
    This class will hold tests for the concrete methods of the 
    AngularFresnel wavefront when the class is written. For now 
    this is a stub.
    """
    pass


class TestGaussianPropagator(object):
    """
    Holds tests for the concrete methods of the GaussianPropagator 
    class. These methods are:
    - __init__()
    - __call__()
    - _propagate()
    - _outside_to_outside()
    - _outside_to_inside()
    - _inside_to_outside()
    - _inside_to_inside()

    Some methods have not been tested out of mathematical faith. These 
    methods are:
    - _waist_to_spherical()
    - _spherical_to_waist()
    - _planar_to_planar()

    Due to the increased complexity of many of these methods they
    are only checked for `numpy.nan` and `numpy.inf` values. With
    correctness testing relegated to the end to end tests.

    Attributes
    ----------
    utility : GaussianProapagatorUtility
        A collection of helpful constructors and functions for testing.
    """
    utility : GaussianPropagatorUtility = GaussianPropagatorUtility()


    def test_propagate(self : Tester) -> None:
        """
        Tests full branch coverage and then the boundary cases 
        distance == -numpy.inf, distance == numpy.inf and distance 
        == 0. The branches covered are:
        
        inside_to_inside
        inside_to_outside
        outside_to_inside
        outside_to_outside
        """
        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()

        rayleigh_distance = wavefront.rayleigh_distance()

        negative_infinity = self\
            .get_utility()\
            .construct(distance = -numpy.inf)\
            ._propagate(wavefront)

        zero = .self\
            .get_utility()\
            .construct(distance = 0.)\
            ._propagate(wavefront)

        positive_infinity = self\
            .get_utility()\
            .construct(distance = numpy.inf)\
            ._propagate(wavefront)

        inside_to_inside = self\
            .get_utility()\
            .construct(distance = rayleigh_distance / 2.)\
            ._propagate(wavefront)

        inside_to_outside = self\
            .get_utility()\
            .construct(distance = rayleigh_distance + 1.)\
            ._propagate()

        outside_to_inside = self\
            .get_utility()\
            .construct(rayleigh_distance)\
            .propagate(wavefront\
                .set_position(-rayleigh_distance - 1.))

        outside_to_outside = self\
            .get_utility()\
            .construct(2 * (rayleigh_distance + 1.))\
            .propagate(wavefront\
                .set_position(-rayleigh_distance - 1.))

        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive_infinity).any()

        assert not numpy.isnan(inside_to_inside).any()
        assert not numpy.isnan(inside_to_outside).any()
        assert not numpy.isnan(outside_to_inside).any()
        assert not numpy.isnan(outside_to_outside).any()

        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive_infinity).any()

        assert not numpy.isinf(inside_to_inside).any()
        assert not numpy.isinf(inside_to_outside).any()
        assert not numpy.isinf(outside_to_inside).any()
        assert not numpy.isinf(outside_to_outside).any()


    def test_outside_to_outside(self : Tester) -> None:
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()

        rayleigh_distance = wavefront.rayleigh_distance()

        negative_infinity = self\
            .get_utility()\
            .construct(-numpy.inf)\
            .outside_to_outside(wavefront\
                .set_position(rayleigh_distance + 1.))

        negative = self\
            .get_utility()\
            .construct(- .2 * rayleigh_distance - 1.)\
            .outside_to_outside(wavefront\
                .set_position(rayleigh_distance + .5))

        positive = self\
            .get_utility()\
            .construct(2. * rayleigh_distance + 1.)\
            .outside_to_outside(wavefront\
                .set_position(- rayleigh_distance - .5))

        positive_infinity = self\
            .get_utility()\
            .construct(numpy.inf)\
            .outside_to_outside(wavefront\
                .set_position(- rayleigh_ditance - 1.))
        
        assert not numpy.isnan(negative_infinity).any()
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        assert not numpy.isnan(positive_infinity).any()     

        assert not numpy.isinf(negative_infinity).any()
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
        assert not numpy.isinf(positive_infinity).any()    


    def test_outside_to_inside(self : Tester) -> None:
        """
        Tests negative and a positive valid input.
        """
        # TODO: Work out if I need to be moving the wavefront
        # back by the rayleigh_distance in the outside tests
        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()

        rayleigh_distance = wavefront.rayleigh_distance()

        negative = self\
            .get_utility()\
            .construct(- rayleigh_distance - 0.01)\
            .outside_to_outside(wavefront\
                .set_position(rayleigh_distance + 0.01))

        positive = self\
            .get_utility()\
            .construct(rayleigh_distance + 0.01)\
            .outside_to_outside(wavefront\
                .set_position(- rayleigh_distance - 0.01))
        
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()

        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
    

    def test_inside_to_outside(self : Tester) -> None:
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()

        rayleigh_distance = wavefront.rayleigh_distance()

        negative = self\
            .get_utility()\
            .construct(- rayleigh_distance - 0.01)\
            .outside_to_outside(wavefront)

        positive = self\
            .get_utility()\
            .construct(rayleigh_distance + 0.01)\
            .outside_to_outside(wavefront)

        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()


    def test_inside_to_inside(self : Tester) -> None:
        """
        Tests the three boundary cases -numpy.inf, and numpy.inf as
        well as a negative and a positive valid input.
        """
        wavefront = self\
            .get_utility()\
            .get_utility()\
            .construct()

        rayleigh_distance = wavefront.rayleigh_distance()

        negative = self\
            .get_utility()\
            .construct(- rayleigh_distance + 0.01)\
            .outside_to_outside(wavefront)

        positive = self\
            .get_utility()\
            .construct(rayleigh_distance - 0.01)\
            .outside_to_outside(wavefront)
                
        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(positive).any()
        
        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(positive).any()
