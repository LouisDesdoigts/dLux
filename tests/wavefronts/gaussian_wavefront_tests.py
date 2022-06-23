import dLux
import jax.numpy as numpy 
import pytest


class TestGaussianWavefront():
    """
    Tests the GaussianWavefront class. Tests are written looking 
    for null and nan values, with only simple end to end tests. 
    """


    AMPLITUDE = numpy.ones((128, 128))
    PHASE = numpy.zeros((128, 128))
    WAVELENGTH = 550e-09
    OFFSET = [0., 0.]
    POSITION = 0.
    BEAM_RADIUS = 1


    @pytest.fixture
    def setup():
        """
        Convinience fixture to create a testing object
        """
        return dLux.GaussianWavefront(
            amplitude = self.AMPLITUDE,
            phase = self.PHASE,
            beam_radius = self.BEAM_RADIUS,
            offset = self.OFFSET,
            position = self.POSITION,
            wavelength = self.WAVELENGTH)

    def test_get_real():
    def test_get_imag():
    def test_multiply_ampl():
    def test_add_phase():
    def test_update_phasor():
    def test_wavefront_to_point_spread_function():
    def test_add_optical_path_difference():
    def test_normalise():
    def test_get_pixel_position_vector():
    def test_get_pixel_position_grid():
    def test_get_physical_position_gird():
    
    def test_get_phase():
    def test_get_amplitude():
    def test_set_phase():
    def test_set_amplitude():
    def test_rayleigh_distance_correct():
    def test_rayleigh_distance_cached():
    def test_location_of_waist_correct():
    def test_location_of_waist_cached():
    def test_waist_radius():
        
        
 

