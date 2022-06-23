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

        assert (full_imaginary_wavefront.get_real() == 1.).all() == True
        assert (part_imaginary_wavefront.get_real() == numpy.cos(Pi/4)).all() == True
        assert (none_imaginary_wavefront.get_real() == 0.).all() == True


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
        
        assert (initial_amplitude == 1.).all() == True
        assert (changed_amplitude == Pi).all() == True
        

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
        
        assert (initial_phase == 0.).all() == True
        assert (changed_phase == Pi).all() == True


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

        assert (wavefront.get_phase() == Pi).all() == True
        assert (wavefront.get_amplitude() == Pi).all() == True
        

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
        
        assert (uniform_zero_phase_psf == 4.).all() == True
        # TODO: This may be overkill and is a little blackboxy 
        assert (unifrom_pi_phase_psf == 4.).all() == True
        assert (variable_pi_phase_psf == self.GRID_ONE ** 2) == True


    def test_add_optical_path_difference():
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


    def test_normalise():


    def test_get_pixel_position_vector():


    def test_get_pixel_position_grid():


    def test_get_physical_position_gird():
    
    # TODO: implement accessors in the GaussianWavefront class.
    # Mutators and Accessors
    def test_get_phase():
    def test_get_amplitude():
    def test_set_phase():
    def test_set_amplitude():

    # Properties 
    def test_rayleigh_distance_correct():
    def test_rayleigh_distance_cached():
    def test_rayleigh_distance_updated():

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

