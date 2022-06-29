import jax.numpy as numpy
import pytest
import jax.random as random
import typing

from dLux import Wavefront
from dLux import PhysicalWavefront
from dLux import AngularWavefront
from dLux import GaussianWavefront

from utilities import WavefrontUtility
from utilities import PhysicalWavefrontUtility
from utilities import GaussianWavefrontUtility
from utilities import AngularWavefrontUtility
from utilities import UtilityUser


Array = typing.NewType("Array", numpy.ndarray)

TestWavefront = typing.NewType("TestWavefront", UtilityUser)
TestPhysicalWavefront = typing.NewType("TestPhysicalWavefront", UtilityUser)
TestAngularWavefront = typing.NewType("TestAngularWavefront", UtilityUser)
TestGaussianWavefront = typing.NewType("TestGaussianWavefront", UtilityUser)


class TestWavefront(UtilityUser):
    """
    Tests the Wavefront class. Tests are written looking 
    for null and nan values, with only simple end to end tests.

    Attributes
    ----------
    utility : WavefrontUtility
        A helper for generating safe test cases.
    """    
    utility : WavefrontUtility = WavefrontUtility()


    def test_get_wavelength(self : TestWavefront) -> None:
        """
        Test for the accessor get_wavelength.
        """
        OUTPUT_WAVELENGTH = self\
            .get_utility()\
            .get_wavelength()

        wavefront = self\
            .get_utility()\
            .construct()

        assert wavefront.get_wavelength() == OUTPUT_WAVELENGTH
            


    def test_set_wavelength(self : TestWavefront) -> None:
        """
        Test for the mutator set_wavelength.
        """
        OLD_WAVELENGTH = self.get_utility().get_wavelength()
        NEW_WAVELENGTH = 540.e-09

        wavefront = self\
            .get_utility()\
            .construct()\
            .set_wavelength(NEW_WAVELENGTH)

        assert wavefront.get_wavelength() == NEW_WAVELENGTH
        assert wavefront.get_wavelength() != OLD_WAVELENGTH


    def test_get_offset(self : TestWavefront) -> None:
        """
        Test for the accessor method get_offset.
        """
        wavefront = self\
            .get_utility()\
            .construct()

        is_correct = (wavefront.get_offset() == self\
            .get_utility()\
            .get_offset())\
            .all()

        assert is_correct


    def test_set_offset(self : TestWavefront) -> None:
        """
        Test for the mutator method set_offset.
        """
        NEW_OFFSET = numpy.array([numpy.pi] * 2)
        OLD_OFFSET = self\
            .get_utility()\
            .get_offset()

        wavefront = self\
            .get_utility()\
            .construct()\
            .set_offset(NEW_OFFSET)

        assert (wavefront.get_offset() == NEW_OFFSET).all()
        assert (wavefront.get_offset() != OLD_OFFSET).all()
    

    def test_get_real(self : TestWavefront) -> None:
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct.         
        """
        FULL_REAL = 0.
        PART_REAL = numpy.pi / 4
        NONE_REAL = numpy.pi / 2

        FULL_REAL_OUTPUT = 1.
        PART_REAL_OUTPUT = 1. / numpy.sqrt(2.)
        NONE_REAL_OUTPUT = 0.        

        full_real_wavefront = self\
            .utility\
            .construct()

        part_real_wavefront = self\
            .utility\
            .construct()\
            .add_phase(PART_REAL)

        none_real_wavefront = self\
            .utility\
            .construct()\
            .add_phase(NONE_REAL)

        is_full_real_correct = self\
            .get_utility()\
            .approx(full_real_wavefront.get_real(), FULL_REAL_OUTPUT)\
            .all()
 
        is_part_real_correct = self\
            .get_utility()\
            .approx(part_real_wavefront.get_real(), PART_REAL_OUTPUT)\
            .all()

        is_none_real_correct = self\
            .get_utility()\
            .approx(none_real_wavefront.get_real(), NONE_REAL_OUTPUT)\
            .all()     

        print(is_full_real_correct)

        assert is_full_real_correct
        assert is_part_real_correct
        assert is_none_real_correct    
        

    def test_get_imaginary(self : TestWavefront) -> None:
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        NONE_IMAGINARY = numpy.pi
        PART_IMAGINARY = numpy.pi / 4
        FULL_IMAGINARY = numpy.pi / 2

        NONE_IMAGINARY_OUT = 0.
        PART_IMAGINARY_OUT = 1. / numpy.sqrt(2)
        FULL_IMAGINARY_OUT = 1.

        none_imaginary_wavefront = self\
            .utility\
            .construct()\
            .add_phase(NONE_IMAGINARY)

        part_imaginary_wavefront = self\
            .utility\
            .construct()\
            .add_phase(PART_IMAGINARY)

        full_imaginary_wavefront = self\
            .utility\
            .construct()\
            .add_phase(FULL_IMAGINARY)

        assert self\
            .get_utility()\
            .approx(full_imaginary_wavefront.get_imaginary(), 
                FULL_IMAGINARY_OUT)\
            .all()

        assert self\
            .get_utility()\
            .approx(part_imaginary_wavefront.get_imaginary(),
                PART_IMAGINARY_OUT)\
            .all()
  
        assert self\
            .get_utility()\
            .approx(none_imaginary_wavefront.get_imaginary(),
                NONE_IMAGINARY_OUT)\
            .all() 


    def test_multiply_amplitude(self : TestWavefront) -> None:
        """
        Checks that the amplitude array has been updated, after 
        operations
        """
        INITIAL_AMPLITUDE = 1.
        CHANGED_AMPLITUDE = 2.

        initial_wavefront = self.get_utility().construct()
        initial_amplitude = initial_wavefront.get_amplitude()
        changed_wavefront = initial_wavefront.multiply_amplitude(2)
        changed_amplitude = changed_wavefront.get_amplitude()
        
        assert (initial_amplitude == INITIAL_AMPLITUDE).all()  
        assert (changed_amplitude == CHANGED_AMPLITUDE).all()  
        

    def test_add_phase(self : TestWavefront) -> None:   
        """
        Checks that the phase array is correctly updated by the 
        operations. 
        """
        INITIAL_PHASE = 0.
        CHANGED_PHASE = numpy.pi

        initial_wavefront = self.get_utility().construct()
        initial_phase = initial_wavefront.get_phase()
        changed_wavefront = initial_wavefront.add_phase(numpy.pi)
        changed_phase = changed_wavefront.get_phase()
        
        assert (initial_phase == INITIAL_PHASE).all()
        assert (changed_phase == CHANGED_PHASE).all()


    def test_update_phasor(self : TestWavefront) -> None:
        """
        Checks that the phasor is correctly updated by changes to
        both the phase and the amplitude. 
        """
        NEW_AMPLITUDE = numpy.ones(
            (self.utility.size, self.utility.size))
        NEW_PHASE = numpy.ones(
            (self.utility.size, self.utility.size))

        wavefront = self\
            .get_utility()\
            .construct()\
            .update_phasor(NEW_AMPLITUDE, NEW_PHASE)

        assert (wavefront.get_phase() == NEW_PHASE).all() 
        assert (wavefront.get_amplitude() == NEW_AMPLITUDE).all() 
        

    def test_wavefront_to_point_spread_function(
            self : TestWavefront) -> None:
        """
        Test that the point spread function is correctly generated
        from the amplitude and phase arrays. Considered correct
        if the output is the amplitude ** 2 and modifying the phase 
        does not affect the PSF
        """
        OUTPUT_AMPLITUDE = self.utility.amplitude * 2.
        OUTPUT_PSF = OUTPUT_AMPLITUDE ** 2

        output_psf = self\
            .get_utility()\
            .construct()\
            .set_amplitude(OUTPUT_AMPLITUDE)\
            .wavefront_to_psf()
         
        assert (output_psf == OUTPUT_PSF).all() 


    def test_add_optical_path_difference(
            self : TestWavefront) -> None:
        """
        Testing for correct behaviour when optical paths length 
        distances are added. 
        """
        INITIAL_PHASE = 0.
        CHANGED_PHASE = numpy.pi
        WAVELENGTH = self.get_utility().get_wavelength()

        initial_wavefront = self.get_utility().construct()
        initial_phase = initial_wavefront.get_phase()
        changed_wavefront = initial_wavefront.add_opd(WAVELENGTH / 2)
        changed_phase = changed_wavefront.get_phase()

        assert (initial_phase == INITIAL_PHASE).all() 
        assert (changed_phase == CHANGED_PHASE).all() 


    def test_normalise(self : TestWavefront) -> None:
        """
        Checks that the normalise functionality is working. The 
        behaviour is considered functional if the maximum 
        value encountered is 1. and the minimum value encountered 
        is 0.
        """
        key = random.PRNGKey(0)
        size = self.get_utility().get_size() 
        
        INITIAL_AMPLITUDE = random.normal(key, (size, size))
        OUTPUT = INITIAL_AMPLITUDE / numpy.linalg.norm(INITIAL_AMPLITUDE)

        normalised_amplitude = self\
            .get_utility()\
            .construct()\
            .set_amplitude(INITIAL_AMPLITUDE)\
            .normalise()\
            .get_amplitude()

        is_correct = self\
            .get_utility()\
            .approx(normalised_amplitude, OUTPUT)\
            .all()

        assert is_correct 
        

    def test_get_pixel_position_vector(
            self : TestWavefront) -> None:
        """
        The get_xs_vec() function is consider correct if the 
        minimum value -(npix - 1) / 2 and the maximum value is
        (npix - 1) / 2
        """
        size = self\
            .get_utility()\
            .get_size()

        pixel_coordinates = self\
            .get_utility()\
            .construct()\
            .get_pixel_coordinates(size)
        
        assert (pixel_coordinates.max() == (size - 1) / 2).all()
        assert (pixel_coordinates.min() == -(size - 1) / 2).all()


    def test_get_pixel_grid(self : TestWavefront) -> None:
        """
        The get_XXYY function is considered correct if it produces 
        an array that is of dimensions (2, self.SIZE, self.SIZE)
        as it uses the get_xs_vec() function under the hood.
        """
        size = self\
            .get_utility()\
            .get_size()

        pixel_grid = self\
            .get_utility()\
            .construct()\
            .get_pixel_grid()
        
        assert pixel_grid.shape == (2, size, size)


    # TODO: Update the docstrings to match the new implementations
    def test_get_pixel_positions(self : TestWavefront) -> None:
        """
        The get_xycoords() function is considered correct if it 
        produces an array with the correct dimesions such that the 
        minimum and maximum values are plus and minus
        self.pixelscale * (npix - 1) / 2
        """
        size = self\
            .get_utility()\
            .get_size()

        pixel_scale = self\
            .get_utility()\
            .get_pixel_scale()

        physical_coordinates = self\
            .get_utility()\
            .construct()\
            .get_pixel_positions()
        
        assert physical_coordinates.max() == (size - 1) / 2 * pixel_scale
        assert physical_coordinates.min() == -(size - 1) / 2 * pixel_scale 
        assert physical_coordinates.shape == (2, size, size)


    def test_set_phase(self : TestWavefront) -> None:
        """
        Functionality that is not currently supported, allows the 
        state to be immutably changed and viewed from outside the 
        namespace. 
        """
        size = self.get_utility().get_size()
        key = random.PRNGKey(0)
        phase = random.normal(key, (size, size))

        wavefront = self\
            .get_utility()\
            .construct()\
            .set_phase(phase)
        
        assert (wavefront.get_phase() == phase).all() 
 

    def test_set_amplitude(self : TestWavefront) -> None:
        """
        Functionality that is not currently supported, provides 
        immutable access to the state of the wavefront. 
        """
        size = self.get_utility().get_size()
        key = random.PRNGKey(0)
        amplitude = random.normal(key, (size, size))

        wavefront = self\
            .get_utility()\
            .construct()\
            .set_amplitude(amplitude)
        
        assert (wavefront.get_amplitude() == amplitude).all() 


class TestPhysicalWavefront(UtilityUser):
    """
    Tests the dLux.PhysicalWavefront class. 

    Attributes
    ----------
    utility : PhysicalWavefrontUtility
        A helper object for generating safe test cases.
    """
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()


    def test_constructor(self : TestPhysicalWavefront) -> None:
        """
        Tests the `PhysicaleWavefront` constructor.
        """
        WAVELENGTH = self.get_utility().get_wavelength()
        OFFSET = self.get_utility().get_offset()

        wavefront = PhysicalWavefront(WAVELENGTH, OFFSET)
        
        assert wavefront.get_wavelength() == WAVELENGTH
        assert (wavefront.get_offset() == OFFSET).all()
        assert wavefront.get_amplitude() == None
        assert wavefront.get_phase() == None
        


class TestAngularWavefront(UtilityUser):
    """
    Tests the `AngularWavefront` class. As `PhysicalWavefront` is 
    just a type ecosystem extension this only tests the constructor. 

    Attributes 
    ----------
    utility : AngularWavefrontUtility
        A helper class for generating safe test cases.
    """
    utility : AngularWavefrontUtility = AngularWavefrontUtility()


    def test_constructor(self : TestAngularWavefront) -> None:
        """
        Checks that the constructor initialises all the required 
        fields.
        """
        WAVELENGTH = self.get_utility().get_wavelength()
        OFFSET = self.get_utility().get_offset()

        wavefront = AngularWavefront(WAVELENGTH, OFFSET)
        
        assert wavefront.get_wavelength() == WAVELENGTH
        assert (wavefront.get_offset() == OFFSET).all()
        assert wavefront.get_amplitude() == None
        assert wavefront.get_phase() == None


class TestGaussianWavefront(UtilityUser):
    """
    Tests the exteded functionality of the `GaussianWavefront`
    class.

    Attributes
    ----------
    utility : GaussianWavefrontUtility
        A helper class for testing the `GaussianWavefront`.
    """
    utility : GaussianWavefront = GaussianWavefrontUtility()


    def test_set_position(self : TestGaussianWavefront) -> None:
        """
        Provides immutable access to the state of the wavefront.
        Considered cocrect if the initial instance is not modified 
        and a modified wavefront is created.
        """
        initial_wavefront = self.get_utility().construct()
        changed_wavefront = initial_wavefront.set_position(1.)

        assert (initial_wavefront.get_position() == 0.)
        assert (changed_wavefront.get_position() == 1.)


    def test_set_phase_radius(self : TestGaussianWavefront) -> None:
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial instance is not modified
        and a modified wavefront is created.
        """
        initial_wavefront = self.get_utility().construct()
        changed_wavefront = initial_wavefront.set_phase_radius(1.)

        # TODO: Work out a good default value for this. 
        assert initial_wavefront.get_phase_radius() == self\
            .get_utility()\
            .get_phase_radius()

        assert changed_wavefront.get_phase_radius() == 1.
       

    # Properties 
    def test_rayleigh_distance(self : TestGaussianWavefront) -> None:
        """
        Checks that the rayleigh distance is calculated correctly
        based on the parameters that are input. 
        """
        wavefront = self.get_utility().construct()
        rayleigh_distance = wavefront.rayleigh_distance()
        correct_rayleigh_distance = numpy.pi * \
            wavefront.get_beam_radius() ** 2 / \
            wavefront.get_wavelength()        

        assert rayleigh_distance == correct_rayleigh_distance
               

    def test_location_of_waist(self : TestGaussianWavefront) -> None:
        """
        Checks that the location of the waist is correctly determined.
        Simply runs the explicit calculations externally and compares.
        """
        wavefront = self\
            .get_utility()\
            .construct()\
            .set_phase_radius(1.)

        correct_location_of_waist = - wavefront.get_phase_radius() / \
            (1 + (wavefront.get_phase_radius() / \
                wavefront.rayleigh_distance()) ** 2)

        assert (wavefront.location_of_waist() == correct_location_of_waist)


    def test_waist_radius(self : TestGaussianWavefront) -> None:
        """
        Directly confirms that the correct numerical calculations are 
        implemented, by the method
        """
        wavefront = self.get_utility().construct()
        correct_waist_radius = wavefront.get_beam_radius() / \
            numpy.sqrt(1 + (wavefront.rayleigh_distance() / \
            wavefront.get_beam_radius()) ** 2)

        assert wavefront.waist_radius() == correct_waist_radius        


    def test_transfer_function_not_nan(
            self : TestGaussianWavefront) -> None:
        """
        Check the boundary case distance = 0. and then two normal
        inputs a large and a small.
        """
        wavefront = self.get_utility().construct()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        large = wavefront.transfer_function(1.)

        # assert not numpy.isnan(zero).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()


    def test_transfer_function_not_inf(
            self : TestGaussianWavefront) -> None:
        """
        Checks that the boundary case distance == 0. and then two 
        normal inputs do not produce infinte results
        """
        wavefront = self.get_utility().construct()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        large = wavefront.transfer_function(1.)

        # assert not numpy.isinf(zero).any()
        assert not numpy.isinf(small).any()
        assert not numpy.isinf(large).any()


    def test_quadratic_phase_factor_not_nan(
            self : TestGaussianWavefront) -> None:
        """
        Checks the boundary case distance == 0. for nan inputs 
        as well as a small and a large typical use case
        """
        wavefront = self.get_utility().construct()
        # zero = wavefront.quadratic_phase_factor(0.)
        # infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        # assert not numpy.isnan(zero).any()
        # assert not numpy.isnan(infinte).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()        


    def test_quadratic_phase_factor_not_inf(
            self : TestGaussianWavefront) -> None:
        """
        Tests the boundary cases distance == numpy.inf and distance 
        == 0. as well as a small and a large typical case for 
        infinite values
        """
        wavefront = self.get_utility().construct()
        # zero = wavefront.quadratic_phase_factor(0.)
        # infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        # assert not numpy.isinf(zero).any()
        # assert not numpy.isinf(infinte).any()
        assert not numpy.isinf(small).any()
        assert not numpy.isinf(large).any()  

        
    def test_pixel_scale_not_nan(
            self : TestGaussianWavefront) -> None:
        """
        Checks that the new pixel scale is not generated to be nan 
        by a negative, zero and positive use case
        """
        wavefront = self.get_utility().construct()
        negative = wavefront.calculate_pixel_scale(-0.01)
        zero = wavefront.calculate_pixel_scale(0.)
        positive = wavefront.calculate_pixel_scale(0.01)

        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive).any()        


    def test_pixel_scale_not_inf(
            self : TestGaussianWavefront) -> None:
        """
        Tests the pixel scale for infinite values when passed a 
        negative, zero and positive position.
        """
        wavefront = self.get_utility().construct()
        negative = wavefront.calculate_pixel_scale(-0.01)
        zero = wavefront.calculate_pixel_scale(0.)
        positive = wavefront.calculate_pixel_scale(0.01)

        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive).any()       


    def test_is_inside(self : TestGaussianWavefront) -> None:
        """
        Branch coverage for a two dimensional is_inside call. That is 
        to say:
        
        false, false
        false, true
        true, false
        true, true
        """
        wavefront = self\
            .get_utility()\
            .construct()\
            .set_phase_radius(1.)

        rayleigh_distance = wavefront.rayleigh_distance()

        false_false = wavefront.is_inside(
            rayleigh_distance * numpy.ones((2, )) + 1.)

        false_true = wavefront.is_inside(numpy\
            .zeros((2, ))\
            .at[0]\
            .set(numpy.squeeze(rayleigh_distance) + 1.))

        true_false = wavefront.is_inside(numpy\
            .zeros((2, ))\
            .at[1]\
            .set(numpy.squeeze(rayleigh_distance) + 1.))

        true_true = wavefront.is_inside(numpy.array([0., 0.]))

        assert (false_false == numpy.array([False, False])).all() 
        assert (false_true == numpy.array([False, True])).all() 
        assert (true_false == numpy.array([True, False])).all() 
        assert (true_true == numpy.array([True, True])).all() 
