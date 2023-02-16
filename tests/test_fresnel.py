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
