import dLux
import jax.numpy as numpy
import pytest
import jax.random as random
import typing
from utilities import WavefrontUtility, PhysicalWavefrontUtility, \
    GaussianWavefrontUtility


Array = typing.NewType("Array", numpy.ndarray)
Wavefront = typing.NewType("Wavefront", object)
PhysicalWavefront = typing.NewType("PhysicalWavefront", Wavefront)
GaussainWavefront = typing.NewType("GaussianWavefront", PhysicalWavefront)


class TestWavefront(object):
    """
    Tests the Wavefront class. Tests are written looking 
    for null and nan values, with only simple end to end tests.

    Attributes
    ----------
    utility : WavefrontUtility
        A helper for generating safe test cases.
    """    
    utility : WavefontUtility = WavefrontUtility()


    def test_get_wavelength(self : TestWavefront) -> None:
        """
        Test for the accessor get_wavelength.
        """
        wavefront = self.utility.construct_wavefront()
        assert wavefront.get_wavelength() == \
            self.utility.get_wavelength()


    def test_set_wavelength(self : TestWavefront) -> None:
        """
        Test for the mutator set_wavelength.
        """
        NEW_WAVELENGTH = 540.e-09
        wavefront = self\
            .utility\
            .construct_wavefront()\
            .set_wavelength(NEW_WAVELENGTH)
        assert wavefront.get_wavelength() == NEW_WAVELENGTH
        assert wavefront.get_wavelength() != self.utility.wavelength


    def test_get_offset(self : TestWavefront) -> None:
        """
        Test for the accessor method get_offset.
        """
        wavefront = self.utility.construct_wavefront()
        assert wavefront.get_offset() == utility.offset


    # TODO: Should this error check for bad inputs 
    def test_set_offset(self : TestWavefront) -> None:
        """
        Test for the mutator method set_offset.
        """
        NEW_OFFSET = [numpy.pi] * 2
        wavefront = self\
            .utility\
            .construct_wavefront()\
            .set_offset(NEW_OFFSET)
        assert wavefront.get_offset() == NEW_OFFSET
        assert wavefront.get_offset() != self.utility.offset
    

    def test_get_real(self : TestPhysicalWavefront) -> None:
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct.         
        """
        PART_REAL = numpy.pi / 4
        NONE_REAL = numpy.pi / 2

        full_real_wavefront = self\
            .utility\
            .construct_wavefront()

        part_real_wavefront = self\
            .utility\
            .construct_wavefront()\
            .add_phase(PART_REAL)

        none_real_wavefront = self\
            .utility\
            .construct_wavefront()\
            .add_phase(NONE_REAL)

        assert (full_real_wavefront.get_real() == 1.).all()
        assert (part_real_wavefront.get_real() == 1. / numpy.sqrt(2)).all()
        assert (none_real_wavefront.get_real() == 0.).all()        
        

    def test_get_imaginary(self : TestPhysicalWavefront) -> None:
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        NONE_IMAGINARY = numpy.pi
        PART_IMAGINARY = numpy.pi / 4
        FULL_IMAGINARY = numpy.pi / 2

        NONE_IMAGINARY_OUT = 0.
        PART_IMAGINARY_OUT = 1. / numpy.sqrt(2)
        FULL_IMAGINARY_OUT = -1.

        none_imaginary_wavefront = self\
            .utility\
            .construct_wavefront()\
            .add_phase(NON_IMAGINARY)

        part_imagnary_wavefront = self\
            .utility\
            .construct_wavefront()\
            .add_phase(PART_IMAGINARY)

        full_imaginary_wavefront = self\
            .utility\
            .construct_wavefront()\
            .add_phase(FULL_IMAGINARY)

        assert (full_imaginary_wavefront.get_imaginary() == \
            FULL_IMAGINARY_OUT).all() 
        assert (part_imaginary_wavefront.get_imaginary() == \
            PART_IMAGINARY_OUT).all() 
        assert (none_imaginary_wavefront.get_imaginary() == \
            NONE_IMAGINARY_OUT).all() 


    def test_multiply_amplitude(self : TestPhysicalWavefront) -> None:
        """
        Checks that the amplitude array has been updated, after 
        operations
        """
        INITIAL_AMPLITUDE = 1.
        CHANGED_AMPLITUDE = 2.

        initial_wavefront = self.utility.construct_wavefront()
        initial_amplitude = initial_wavefront.get_amplitude()
        changed_wavefront = initial_wavefront.multiply_amplitude(2)
        changed_amplitude = changed_wavefront.get_amplitude()
        
        assert (initial_amplitude == INITIAL_AMPLITUDE).all() 
        assert (changed_amplitude == CHANGED_AMPLITUDE).all() 
        

    def test_add_phase(self : TestPhysicalWavefront) -> None:   
        """
        Checks that the phase array is correctly updated by the 
        operations. 
        """
        INITIAL_PHASE = 0.
        CHANGED_PHASE = numpy.pi

        initial_wavefront = self.utility.construct_wavefront()
        initial_phase = initial_wavefront.get_phase()
        changed_wavefront = intial_wavefront.add_phase(numpy.pi)
        changed_phase = wavefront.get_phase()
        
        assert (initial_phase == INITIAL_PHASE).all()
        assert (changed_phase == CHANGED_PHASE).all()


    def test_update_phasor(self : TestingPhysicalWavefront) -> None:
        """
        Checks that the phasor is correctly updated by changes to
        both the phase and the amplitude. 
        """
        NEW_AMPLITUDE = numpy.ones(
            (self.utility.size, self.utility.size))
        NEW_PHASE = numpy.ones(
            (self.utility.size, self.utility.size))

        wavefront = self\
            .utility\
            .construct_wavefront()\
            .update_phasor(NEW_AMPLITUDE, NEW_PHASE)

        assert (wavefront.get_phase() == NEW_PHASE).all()
        assert (wavefront.get_amplitude() == NEW_AMPLITUDE).all()
        

    def test_wavefront_to_point_spread_function(
            self : TestingPhysicalWavefront) -> None:
        """
        Test that the point spread function is correctly generated
        from the amplitude and phase arrays. Considered correct
        if the output is the amplitude ** 2 and modifying the phase 
        does not affect the PSF
        """
        OUTPUT_AMPLITUDE = self.utility.amplitude * 2.
        OUTPUT_PSF = OUTPUT_AMPLITUDE ** 2

        output_psf = self\
            .utility
            .construct_wavefront()\
            .set_amplitude(OUTPUT_AMPLITUDE)\
            .wavefront_to_psf()
         
        assert (output_psf == OUTPUT_PSF).all()


    def test_add_optical_path_difference(
            self : TestingPhysicalWavefront) -> None:
        """
        Testing for correct behaviour when optical paths length 
        distances are added. 
        """
        INITIAL_PHASE = 0.
        CHANGED_PHASE = numpy.pi / 2

        initial_wavefront = self.utitlity.construct_wavefront()
        initial_phase = wavefront.get_phase()
        changed_wavefront = wavefront.add_opd(self.WAVELENGTH / 2)
        changed_phase = wavefront.get_phase()

        assert (initial_phase == INITIAL_PHASE).all()
        assert (changed_phase == CHANGED_PHASE).all()


    def test_normalise(self : TestingPhysicalWavefront) -> None:
        """
        Checks that the normalise functionality is working. The 
        behaviour is considered functional if the maximum 
        value encountered is 1. and the minimum value encountered 
        is 0.
        """
        # TODO: Implement getters for the utility. 
        # NOTE: I don't think that we need to have setters.
        key = random.PRNGKey(0).split()[0]
        size = self.utility.get_size() 
        
        INITIAL_AMPLITUDE = random.norm(key, (size, size))

        normalised_amplitude = self\
            .utility\
            .construct_wavefront(amplitude = INITIAL_AMPLITUDE)\
            .normalise()\
            .get_amplitude()

        assert (changed_amplitude.max() == 1.)
        assert (changed_amplitude.min() == 0.)
        

    def test_get_pixel_position_vector(self):
        """
        The get_xs_vec() function is consider correct if the 
        minimum value -(npix - 1) / 2 and the maximum value is
        (npix - 1) / 2
        """
        initial_wavefront = self.set_up()
        pixel_coordinates = initial_wavefront.get_xs_vec(self.SIZE)
        
        assert (pixel_coordinates.max() == (self.SIZE - 1) / 2)
        assert (pixel_coorsinates.min() == -(self.SIZE - 1) / 2)
        #TODO: implement a check that the increment between the 
        # max and min values is uniform. 


    def test_get_pixel_position_grid(self):
        """
        The get_XXYY function is considered correct if it produces 
        an array that is of dimensions (2, self.SIZE, self.SIZE)
        as it uses the get_xs_vec() function under the hood.
        """
        initial_wavefront = self.set_up()
        pixel_position_grid = initial_wavefront.get_XXYY()
        
        assert pixel_position_grid.shape() == (2, self.SIZE, self.SIZE)


    def test_get_physical_position_gird(self):
        """
        The get_xycoords() function is considered correct if it 
        produces an array with the correct dimesions such that the 
        minimum and maximum values are plus and minus
        self.pixelscale * (npix - 1) / 2
        """
        initial_wavefront = self.set_up()
        changed_wavefront = dLux.CreateWavefront(self.SIZE, 1.)\
            ({"Wavefront": initial_wavefront})["Wavefront"]
        physical_coordinates = changed_wavefront.get_xycoords()
        
        assert (physical_coordinates.max() == (self.SIZE - 1) / (2 * self.SIZE))
        assert (physical_coordinates.min() == -(self.SIZE - 1) / (2 * self.SIZE))
        assert physical_coordinates.shape() == (2, self.SIZE, self.SIZE)


    # TODO: implement accessors in the GaussianWavefront class.
    # Mutators and Accessors
    def test_set_phase(self):
        """
        Functionality that is not currently supported, allows the 
        state to be immutably changed and viewed from outside the 
        namespace. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_phase(self.GRID_ONE)
        
        assert (changed_wavefront.get_phase() == self.GRID_ONE).all()
        assert (initial_wavefront.get_phase() == self.PHASE).all()
 

    def test_set_amplitude(self):
        """
        Functionality that is not currently supported, provides 
        immutable access to the state of the wavefront. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_amplitude(self.GRID_ONE)
        
        assert (changed_wavefront.get_amplitude() == self.GRID_ONE).all()
        assert (initial_wavefront.get_amplitude() == self.AMPLITUDE).all()


    def test_set_wavelength(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial wavefront is not modified 
        and a modified wavefront is created. 
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_wavelength(2 * self.WAVELENGTH)

        assert (changed_wavefront.get_wavelength() == 2 * self.WAVELENGTH)
        assert (initial_wavefront.get_wavelength() == self.WAVELENGTH)


class TestPhysicalWavefront(object):
    """
    Tests the dLux.PhysicalWavefront class. 

    Attributes
    ----------
    utility : PhysicalWavefrontUtility
        A helper object for generating safe test cases.
    """
    utility : PhysicalWavefrontUtility = PhysicalWavefrontUtility()


class TestAngularWavefront(object):
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
        angular_wavefront = self.get_utility()
    

class TestGaussianWavefront(object):
    """
    Tests the exteded functionality of the `GaussianWavefront`
    class.

    Attributes
    ----------
    utility : GaussianWavefrontUtility
        A helper class for testing the `GaussianWavefront`.
    """
    utility : GaussianWavefront = GaussianWavefrontUtility()


    def test_set_beam_waist(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the intial state is not modfied and
        a modified clone is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_beam_waist(2.)
        
        assert (changed_wavefront.get_beam_waist() == 2.)
        assert (initial_wavefront.get_beam_waist() == 1.)




    def test_set_position(self):
        """
        Provides immutable access to the state of the wavefront.
        Considered cocrect if the initial instance is not modified 
        and a modified wavefront is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_position(1.)

        assert (initial_wavefront.get_position() = 0.)
        assert (changed_wavefront.get_position() = 1.)


    def test_set_phase_radius(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial instance is not modified
        and a modified wavefront is created.
        """
        initial_wavefront = self.set_up()
        changed_wavefront = initial_wavefront.set_phase_radius(1.)

        # TODO: Work out a good default value for this. 
        assert initial_wavefront.get_phase_radius() == 0.
        assert changed_wavefront.get_phase_radius() == 1.
       

    # Properties 
    def test_rayleigh_distance_correct(self):
        """
        Checks that the rayleigh distance is calculated correctly
        based on the parameters that are input. 
        """
        # TODO: will not work because equinox enforces immutability
        # need to check that equinox is compatible with cached
        # properties
        wavefront = self.set_up()
        rayleigh_distance = wavefront.rayleigh_distance
        correct_rayleigh_distance = numpy.pi * \
            wavefront.get_beam_radius() ** 2 / \
            wavefront.get_wavelength()        

        assert rayleigh_distance == correct_rayleigh_distance
               

    def test_location_of_waist_correct(self):
        """
        Checks that the location of the waist is correctly determined.
        Simply runs the explicit calculations externally and compares.
        """
        wavefront = self.set_up()
        correct_location_of_waist = - wavefront.get_phase_radius() / \
            (1 + (wavefront.get_phase_radius() / \
                wavefront.rayleigh_distance) ** 2)

        assert (wavefront.location_of_waist == correct_location_of_waist)


    # TODO: Implement the waist radius as a chached property 
    def test_waist_radius(self):
        """
        Directly confirms that the correct numerical calculations are 
        implemented, by the method
        """
        # TODO: fix the call of rayleigh_distance() to rayleigh_distance 
        # and also beam_radius to self.beam_radius
        wavefront = self.set_up()
        correct_waist_radius = wavefront.get_beam_radius() / \
            numpy.sqrt(1 + (wavefront.rayleigh_distance / \
            wavefront.beam_radius) ** 2)

        assert wavefront.waist_radius == correct_waist_radius        


    def test_transfer_function_not_nan(self):
        """
        Check the boundary case distance = 0. and then two normal
        inputs a large and a small.
        """
        wavefront = self.set_up()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        larger = wavefront.transfer_function(1.)

        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()


    def test_transfer_function_not_inf(self):
        """
        Checks that the boundary case distance == 0. and then two 
        normal inputs do not produce infinte results
        """
        wavefront = self.set_up()
        zero = wavefront.transfer_function(0.)
        small = wavefront.transfer_function(0.01)
        larger = wavefront.transfer_function(1.)

        assert not numpy.ininf(zero).any()
        assert not numpy.ininf(small).any()
        assert not numpy.ininf(large).any()


    def test_quadratic_phase_factor_not_nan(self):
        """
        Checks the boundary case distance == 0. for nan inputs 
        as well as a small and a large typical use case
        """
        wavefront = set_up()
        zero = wavefront.quadratic_phase_factor(0.)
        infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(infinte).any()
        assert not numpy.isnan(small).any()
        assert not numpy.isnan(large).any()        


    def test_quadratic_phase_factor_not_inf(self):
        """
        Tests the boundary cases distance == numpy.inf and distance 
        == 0. as well as a small and a large typical case for 
        infinite values
        """
        zero = wavefront.quadratic_phase_factor(0.)
        infinte = wavefront.quadratic_phase_factor(numpy.inf)
        small = wavefront.quadratic_phase_factor(0.001)
        large = wavefront.quadratic_phase_factor(1.)

        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(infinte).any()
        assert not numpy.isinf(small).any()
        assert not numpy.isinf(large).any()  

        
    def test_pixel_scale_not_nan(self):
        """
        Checks that the new pixel scale is not generated to be nan 
        by a negative, zero and positive use case
        """
        wavefront = set_up()
        negative = wavefront.pixel_scale(-0.01)
        zero = wavefront.pixel_scale(0.)
        positive = wavefront.pixel_scale(0.01)

        assert not numpy.isnan(negative).any()
        assert not numpy.isnan(zero).any()
        assert not numpy.isnan(positive).any()        


    def test_pixel_scale_not_inf(self):
        """
        Tests the pixel scale for infinite values when passed a 
        negative, zero and positive position.
        """
        wavefront = set_up()
        negative = wavefront.pixel_scale(-0.01)
        zero = wavefront.pixel_scale(0.)
        positive = wavefront.pixel_scale(0.01)

        assert not numpy.isinf(negative).any()
        assert not numpy.isinf(zero).any()
        assert not numpy.isinf(positive).any()       


    def test_is_inside(self):
        """
        Branch coverage for a two dimensional is_inside call. That is 
        to say:
        
        false, false
        false, true
        true, false
        true, true
        """
        wavefront = set_up()
        rayleigh_distance = wavefront.rayleigh_distance

        false_false = wavefront.is_inside(
            rayleigh_distance * numpy.ones((2, )) + 1.)
        false_true = wavefront.is_inside(
            numpy.array([rayleigh_distance + 1., 0.]))
        true_false = wavefront.is_inside(
            numpy.array([0., rayleigh_distance + 1.]))
        true_true = wavefront.is_inside([0., 0.])

        assert (false_false == numpy.array([False, False])).all()
        assert (false_true == numpy.array([False, True])).all()
        assert (true_false == numpy.array([True, False])).all()
        assert (true_true == numpy.array([True, True])).all()




