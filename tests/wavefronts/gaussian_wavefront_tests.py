import dLux
import jax.numpy as numpy
import jax.numpy.pi as Pi 
import pytest

# TODO: Implement type annotations for the tests. Is this nessecary 
# I am not sure and will consult stackoverflow. 
# TODO: I need to make some interesting arrays and add them as class 
# constants based on the way that this is currently shaking out. 


class TestGaussianWavefront():
    """
    Tests the GaussianWavefront class. Tests are written looking 
    for null and nan values, with only simple end to end tests. 
    Some properties are cached and the caching process is also tested
    as well as stateful updates when the dependencies are updated. 
    This may require some reimplementation to get correct. 

    A common testing pattern is that nan's and inf's are not generated
    at Runtime under a variety of circumstances. Once this is confirmed 
    the correctness is checked against some simple examples.
    """


    # Constants
    SIZE = 128
    AMPLITUDE = numpy.ones((SIZE, SIZE))
    PHASE = numpy.zeros((SIZE, SIZE))
    WAVELENGTH = 550e-09
    OFFSET = [0., 0.]
    POSITION = 0.
    BEAM_RADIUS = 1


    # Test Cases 
    ONES = numpy.ones((SIZE, SIZE))
    GRID_ONE, GRID_TWO = numpy.meshgrid(
        numpy.linspace(0., 1., SIZE), 
        numpy.linspace(0., 1., SIZE))
    


    @pytest.fixture
    def set_up(self, /, amplitude = None, phase = None, 
            beam_radius = None, offset = None, position = None,
            wavelength = None):
        """
        Convinience fixture to create a testing object. Effectively 
        adds default parameters to the GaussianWavefront constructor.
        """
        return dLux.GaussianWavefront(
            amplitude = self.AMPLITUDE if amplitude is None else amplitude,
            phase = self.PHASE if amplitude is None else phase,
            beam_radius = self.BEAM_RADIUS if amplitude is None else beam_radius,
            offset = self.OFFSET if offset is None else offset,
            position = self.POSITION if position is None else position,
            wavelength = self.WAVELENGTH if wavelength is None else wavelength)

    # TODO: These need to be moved into the superclass test file.
    def test_get_real(self):
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        # TODO: Implement more complex test cases
        full_real_wavefront = set_up()
        part_real_wavefront = set_up(phase = Pi * self.ONES / 4)
        none_real_wavefront = set_up(phase = Pi * self.ONES / 2)

        assert (full_real_wavefront.get_real() == 1.).all()
        assert (part_real_wavefront.get_real() == 1. / numpy.sqrt(2)).all()
        assert (none_real_wavefront.get_real() == 0.).all()        
        

    def test_get_imag(self):
        """
        Rotates the phasor through real -> imaginary and checks that 
        the output is correct
        """
        # TODO: Implement more complex test cases. 
        none_imaginary_wavefront = set_up()
        part_imaginary_wavefront = set_up(phase = Pi * self.ONES / 4)
        full_imaginary_wavefront = set_up(phase = Pi * self.ONES / 2)

        assert (full_imaginary_wavefront.get_real() == 1.).all() 
        assert (part_imaginary_wavefront.get_real() == numpy.cos(Pi/4)).all() 
        assert (none_imaginary_wavefront.get_real() == 0.).all() 


    def test_multiply_amplitude(self):
        """
        Checks that the amplitude array has been updated, after 
        operations
        """
        # TODO: Check modification by array valued inputs.
        initial_wavefront = set_up()
        initial_amplitude = wavefront.get_amplitude()
        changed_wavefront.multiply_ampl(Pi)
        changed_amplitude = wavefront.get_amplitude()
        
        assert (initial_amplitude == 1.).all() 
        assert (changed_amplitude == Pi).all() 
        

    def test_add_phase(self):   
        """
        Checks that the phase array is correctly updated by the 
        operations. 
        """
        # TODO: Check modification by array valued inputs.
        initial_wavefront = set_up()
        initial_phase = wavefront.get_phase()
        changed_wavefront = wavefront.add_phase(Pi)
        changed_phase = wavefront.get_phase()
        
        assert (initial_phase == 0.).all()
        assert (changed_phase == Pi).all()


    def test_update_phasor(self):
        """
        Checks that the phasor is correctly updated by changes to
        both the phase and the amplitude. 
        """
        # TODO: Assumes that the imputs are real
        # TODO: Implement more complex example arrays
        new_amplitude = self.ONES * Pi
        new_phase = self.ONES * Pi
        wavefront = set_up()
        wavefront = wavefront.update_phasor(new_amplitude, new_phase)

        assert (wavefront.get_phase() == Pi).all()
        assert (wavefront.get_amplitude() == Pi).all()
        

    def test_wavefront_to_point_spread_function(self):
        """
        Test that the point spread function is correctly generated
        from the amplitude and phase arrays. Considered correct
        if the output is the amplitude ** 2 and modifying the phase 
        does not affect the PSF
        """
        wavefront = set_up(amplitude = 2. * self.ONES)
        uniform_zero_phase_psf = wavefront.wf2psf()
        wavefront.add_phase(Pi)
        uniform_pi_phase_psf = wavefront.wf2psf()
        wavefront.multiply_ampl(0.5 * self.GRID_ONE)
        variable_pi_phase_psf = wavefront.wf2psf()
        
        assert (uniform_zero_phase_psf == 4.).all()
        # TODO: This may be overkill and is a little blackboxy 
        assert (unifrom_pi_phase_psf == 4.).all()
        assert (variable_pi_phase_psf == self.GRID_ONE ** 2)


    def test_add_optical_path_difference(self):
        """
        Testing for correct behaviour when optical paths length 
        distances are added. 
        """
        # So if I were to add some phase I would want to be able to 
        # interfere this beam with another
        # TODO: Raise a github issue about this functionality
        # TODO: Raise a github issue about immutability vs mutability 
        # get some real evidence as to the performance re-creating 
        # vs modifying the classes.  
        initial_wavefront = set_up()
        initial_phase = wavefront.get_phase()
        changed_wavefront = wavefront.add_opd(self.WAVELENGTH / 2)
        changed_phase = wavefront.get_phase()

        assert (initial_phase == 0.).all()
        assert (changed_phase == Pi / 2).all()


    def test_normalise(self):
        """
        Checks that the normalise functionality is working. The 
        behaviour is considered functional if the maximum 
        value encountered is 1. and the minimum value encountered 
        is 0.
        """
        initial_wavefront = set_up(amplitude = 2 * GRID_ONE * GRID_TWO)
        changed_wavefront = initial_wavefront.normalise()
        changed_amplitude = changed_wavefront.get_amplitude()

        assert (changed_amplitude.max() == 1.)
        assert (changed_amplitude.min() == 0.)
        

    def test_get_pixel_position_vector(self):
        """
        The get_xs_vec() function is consider correct if the 
        minimum value -(npix - 1) / 2 and the maximum value is
        (npix - 1) / 2
        """
        initial_wavefront = set_up()
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
        initial_wavefront = set_up()
        pixel_position_grid = initial_wavefront.get_XXYY()
        
        assert pixel_position_grid.shape() == (2, self.SIZE, self.SIZE)


    def test_get_physical_position_gird(self):
        """
        The get_xycoords() function is considered correct if it 
        produces an array with the correct dimesions such that the 
        minimum and maximum values are plus and minus
        self.pixelscale * (npix - 1) / 2
        """
        initial_wavefront = set_up()
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
        initial_wavefront = set_up()
        changed_wavefront = initial_wavefront.set_phase(self.GRID_ONE)
        
        assert (changed_wavefront.get_phase() == self.GRID_ONE).all()
        assert (initial_wavefront.get_phase() == self.PHASE).all()
 

    def test_set_amplitude(self):
        """
        Functionality that is not currently supported, provides 
        immutable access to the state of the wavefront. 
        """
        initial_wavefront = set_up()
        changed_wavefront = initial_wavefront.set_amplitude(self.GRID_ONE)
        
        assert (changed_wavefront.get_amplitude() == self.GRID_ONE).all()
        assert (initial_wavefront.get_amplitude() == self.AMPLITUDE).all()


    def test_set_beam_waist(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the intial state is not modfied and
        a modified clone is created.
        """
        initial_wavefront = set_up()
        changed_wavefront = initial_wavefront.set_beam_waist(2.)
        
        assert (changed_wavefront.get_beam_waist() == 2.)
        assert (initial_wavefront.get_beam_waist() == 1.)


    def test_set_wavelength(self):
        """
        Provides immutable access to the state of the wavefront. 
        Considered correct if the initial wavefront is not modified 
        and a modified wavefront is created. 
        """


    def test_set_position(self):
       

    # Properties 
    def test_rayleigh_distance_correct(self):
        """
        Checks that the rayleigh distance is calculated correctly
        based on the parameters that are input. 
        """
        # TODO: will not work because equinox enforces immutability
        # need to check that equinox is compatible with cached
        # properties
        wavefront = set_up()
        rayleigh_distance = wavefront.rayleigh_distance
        correct_rayleigh_distance = numpy.pi * \
            wavefront.get_beam_radius() ** 2 / \
            wavefront.get_wavelength()        

        assert rayleigh_distance == correct_rayleigh_distance


    def test_rayleigh_distance_cached(self):
        """
        Checks that after getting called this becomes a class 
        property stored in GaussianWavefront.__dict__
        """
        wavefront = set_up()
        wavefront.rayleigh_distance
        
        assert wavefront.__dict__["rayleigh_distance"] not None    
    

    def test_rayleigh_distance_updated(self):
        """
        Checks that if beam_radius and wavelength are changed then 
        the rayleigh distance is also updated.
        """
        # TODO: I can enforce this behaviour using setter methods
        # or mutators. These need to be implemented. 
        initial_wavefront = set_up()
        initial_wavefront.rayleigh_distance
        changed_wavefront = wavefront.set_beam_waist(2.)
        changed_wavefront.rayleigh_distance

        correct_rayleigh_distance
               


    def test_location_of_waist_correct():
    def test_location_of_waist_cached():
    def test_location_of_waist_updated():

    # TODO: Implement the waist radius as a chached property 
    def test_waist_radius_correct():
    def test_waist_radius_cached():
    def test_waist_radius_updated():

    # State modifying behaviours 
    def test_planar_to_planar_not_nan():
    def test_planar_to_planar_not_inf():
    def test_planar_to_planar_correct():

    def test_waist_to_spherical_not_nan():
    def test_waist_to_spherical_not_inf():
    def test_waist_to_spherical_correct():

    def test_spherical_to_waist_not_nan():
    def test_spherical_to_waist_not_inf():
    def test_spherical_to_waist_correct():

    # State independent behavious (static)
    def test_calculate_phase_not_nan():
    def test_calculate_phase_not_inf():
    def test_calculate_phase_correct(): 

    def test_transfer_function_not_nan():
    def test_transfer_function_not_inf():
    def test_transfer_function_correct():

    def test_quadratic_phase_factor_not_nan():
    def test_quadratic_phase_factor_not_inf():
    def test_quadratic_phase_factor_correct():
        
    def test_pixel_scale_not_nan():
    def test_pixel_scale_not_inf():
    # TODO: Implement tests for differentiability

    def test_outside_to_outside_not_nan():
    def test_outside_to_outside_not_inf():
    def test_outside_to_outside_correct():

    def test_outside_to_inside_not_nan():
    def test_outside_to_inside_not_inf():
    def test_outside_to_inside_correct():

    def test_inside_to_outside_not_nan():
    def test_inside_to_outside_not_inf():
    def test_inside_to_outside_correct():

    def test_inside_to_inside_not_nan():
    def test_inside_to_inside_not_inf():
    def test_inside_to_inside_correct():

    # TODO: Remove the Float call from is_inside
    def test_is_inside_false_false():
    def test_is_inside_false_true():
    def test_is_inside_true_false():
    def test_is_inside_true_ture():

    def test_propagate_not_nan():
    def test_propagate_not_inf():
    def test_propagate_correct():

