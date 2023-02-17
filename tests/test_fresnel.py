import pytest
import jax.numpy as np 
import jax.random as jr
import jax.lax as jl

WAVELENGTH: str = "wavelength"
OFFSET: str = "offset"
PIXEL_SCALE: str = "pixel_scale"
PLANE_TYPE: str = "plane_type"
AMPLITUDE: str = "amplitude"
PHASE: str = "phase"
WAIST_RADIUS: str = "waist_radius"
RAYLEIGH_FACTOR: str = "rayleigh_factor"

class TestGaussianWavefront(object) -> None:
    """
    Only testing the new parameters of the GaussianWavefront, since the 
    remainder should be tested elsewhere?
    """
    def test_constructor_with_non_scalar_waist_radius(
            def_gauss_wave_params: fixture[dict],
            non_scalar_param: fixture[float],
        ) -> None:
        """
        Parameter
        ---------
        def_gauss_wave_params: fixture[tuple],
            A tuple of default parameters to the GaussianWavefront constuctor.
        non_scalar_param: fixture[float],
            An array (non-scalar) that can be subsituted for a parameter.
        """
        with pytest.raises(ValueError):
            dLux.GaussianWavefront(
                wavelength = def_gauss_wave_params[WAVELENGTH],
                offset = def_gauss_wave_params[OFFSET],
                pixel_scale = def_gauss_wave_params[PIXEL_SCALE],
                plane_type = def_gauss_wave_params[PLANE_TYPE],
                amplitude = def_gauss_wave_params[AMPLITUDE],
                phase = def_gauss_wave_params[PHASE],
                waist_radius = non_scalar_param, 
                rayleigh_factor = def_gauss_wave_params[RAYLEIGH_FACTOR],
            )

    def test_constructor_with_non_scalar_rayleigh_factor(
            def_gauss_wave_params: fixture[dict],
            non_scalar_param: fixture[float],
        ) -> None:
        """
        Parameter
        ---------
        def_gauss_wave_params: fixture[tuple],
            A tuple of default parameters to the GaussianWavefront constuctor.
        non_scalar_param: fixture[float],
            An array (non-scalar) that can be subsituted for a parameter.
        """
        with pytest.raises(ValueError):
            dLux.GaussianWavefront(
                wavelength = def_gauss_wave_params[WAVELENGTH],
                offset = def_gauss_wave_params[OFFSET],
                pixel_scale = def_gauss_wave_params[PIXEL_SCALE],
                plane_type = def_gauss_wave_params[PLANE_TYPE],
                amplitude = def_gauss_wave_params[AMPLITUDE],
                phase = def_gauss_wave_params[PHASE],
                waist_radius = def_gauss_wave_params[WAIST_RADIUS], 
                rayleigh_factor = non_scalar_param,
            )

    def test_constructor_zeros_position(
            def_gauss_wavefront: fixture[object],
        ) -> None:
        """
        Parameters
        ----------
        def_gauss_wavefront: fixture[object],
           A GaussianWavefront with legitimate parameters. 
        """
        assert def_gauss_wavefront.position == 0.0

    def test_constructor_zeros_waist_position(
            def_gauss_wavefront: fixture[object],
        ) -> None: # TODO: how valuable is this.
        """
        Parameters
        ----------
        def_gauss_wavefront: fixture[object],
           A GaussianWavefront with legitimate parameters. 
        """
        assert def_gauss_wavefront.waist_position == 0.0

    def test_constructor_falsifies_angularity(
            def_gauss_wavefront: fixture[object],
        ) -> None:
        """
        Parameters
        ----------
        def_gauss_wavefront: fixture[object],
           A GaussianWavefront with legitimate parameters. 
        """
        assert not def_gauss_wavefront.angular

    def test_constructor_falsifies_sphericity(
            def_gauss_wavefront: fixture[object],
        ) -> None:
        """
        Parameters
        ----------
        def_gauss_wavefront: fixture[object],
           A GaussianWavefront with legitimate parameters. 
        """
        assert not def_gauss_wavefront.spherical

    def test_constructor_infinitates_focal_length(
            def_gauss_wavefront: fixture[object],
        ) -> None:
        """
        Parameters
        ----------
        def_gauss_wavefront: fixture[object],
           A GaussianWavefront with legitimate parameters. 
        """
        assert np.isinf(def_gauss_wavefront.focal_length)

    # NOTE: It so happens that having a different pixel scale depending 
    #       on the get_pixel_scale parameter means that a lot of methods
    #       kind of need to be tested twice. Is this necessary or can I 
    #       assume that get_pixel_scale works so everything that depends 
    #       on it works. 

    def test_get_pixel_scale_when_angular() -> None:
        pass 

    def test_get_pixel_scale_when_planar() -> None:
        pass

    def test_pixel_coordinates_is_linear() -> None:
        pass

    def test_pixel_coordinates_is_increasing() -> None:
        pass 

    def test_pixel_coordinates_x_is_broadcast() -> None:
        pass

    def test_pixel_coordinates_y_is_broadcast() -> None:
        pass

    # NOTE: So I do not think that rayleigh_distance needs testing,
    #       but I could add a test to make sure that it is not producing 
    #       errors.

    @pytest.parametrize("position", [])
    def test_calculate_pixel_scale_at_when_angular() -> None:
        pass

    # NOTE: Hmmm, there is an interesting case in testing the is_inside 
    #       function. Without manually doing the calculations it will be 
    #       hard to know. However, this does lead to some interesting 
    #       thoughts about what values are allowed for the rayleigh distance
    #       and what the rayleigh distance actually means. For example,
    #       can the rayleigh distance be infinite or zero? In general,
    #       I think I need to try and work out the properties of the 
    #       rayleigh distance and is_inside function. For example, the 
    #       is_inside function should be discontinuous at rayleigh 
    #       distance and that is something that I can test right there. 

    def test_is_inside_is_discontinuous_at_the_rayleigh_distance() -> None:
        pass

    # NOTE: The quadratic phase is another great one to test. For example,
    #       I can consider the second derivative and verify that it is 
    #       approximately linear in log space. I can also test that the 
    #       gradient passes through zero at the centre along both axes. 
    #       Except I just checked and it is radially quadratic. This presents
    #       more difficulty. I can still slice it along the x and y axis 
    #       and do this. That would be a really cool usage of parametrize
    #       to test both axis separately. Then I can craft a really cool 
    #       way of testing that the function is increasing radially. 
    #       Basically, generate a grid which represents the coordinates
    #       select pixel wide bands and then calculate the average within 
    #       the band. Make sure that this sequence is increasing. This is 
    #       going to be so fucking sick.

    @pytest.mark.parametrize("axis", [0, 1])
    def test_quadratic_phase_has_linear_slope_along_axis_in_log_space() -> None:
        pass

    def test_quadratic_phase_is_radially_increasing_in_log_space() -> None:
        pass

    # NOTE: Since I know that multiplication by 1.0j can be undone by 
    #       division etc. I simply take the logarithms and multiply
    #       by i cubed.

    # NOTE: The transfer function has very similar properties. The problem
    #       is that the exponential is imaginary which transforms the properties
    #       in ways that I don't fully understand. 

    @pytest.mark.parametrize("axis", [0, 1])
    def test_transfer_function_has_linear_slope_along_axis_in_log_space() -> None:
        pass

    def test_transfer_function_is_radially_increasing_in_log_space() -> None:
        pass

    # NOTE: curvature_at is going to be a fun function to analyse. The functional
    #       form has a discontinuity and, perhaps more exitingly there are 
    #       two distinct asymptotic behaviours. Firstly, and this is my favourite 
    #       the linear behaviour dominates for very far away positions. This 
    #       will be fun to test. The problem is that the scale is in rayleigh
    #       distances. The second asymtote is towards zero (or in this case)
    #       the waist_position. This is where the discontinuity is also.

    def test_curvature_at_is_discontinuous_at_the_beam_waist() -> None:
        pass

    def test_curvature_at_is_approximately_linear_very_far_from_the_beam_waist() -> None:
        pass

    # NOTE: Again radius_at has beautiful asymptotic behaviour, tending towards
    #       a linear function as we approach infinity. This makes for excellent
    #       testing and a nice use for parametrize (sign). Also it is 
    #       guaranteed to be position and have a minimum at the waist. Hence
    #       I can test for said minimum, but I am not sure how to do the positive
    #       part.

    def test_radius_at_is_asymptotically_linear() -> None:
        pass

    # NOTE: Then there is the problem of how to tests something assymptotically
    #       is linear on a computer. There are obviously infinitely many 
    #       values that it can take. Now, if this were analysis I would do it
    #       by taking some property and at an arbitrary point. The problem,
    #       is that one a computer that doesn't really work does it. Although,
    #       there is an interesting thought about compositions of functions 
    #       lurking in here somewhere. Anyway, what I think is good enough is
    #       is it infinite at infinity? or is the gradient a constant at infinity
    #       and then pick some large numbers. Could also subtract the function 
    #       x away and does it approach zero. This is much similar to how 
    #       you would do it in analysis, then tried to prove that it was 
    #       monotonically decreasing. I think we just show it is decreasing 
    #       for some sample and assume generality. 

    def test_radius_at_has_a_minimum_at_the_waist() -> None:
        pass

