import jax.numpy as np
import pytest
from jax import config

config.update("jax_debug_nans", True)


def _test_propagator_constructor(constructor):
    """Tests the constructor of a propagator."""
    constructor()
    with pytest.raises(TypeError):
        constructor(focal_length=np.array([1]))


def _test_call(constructor, wf_constructor):
    """Tests the __call__ method of a propagator."""
    wf = wf_constructor()
    wf = constructor(focal_length=None)(wf)
    wf = constructor(focal_length=None, inverse=True)(wf)
    wf = constructor(focal_length=5)(wf)
    wf = constructor(focal_length=5, inverse=True)(wf)


class TestFFT:
    """Test the CartesianFFT class."""

    def test_constructor(self, create_fft):
        """Tests the constructor."""
        _test_propagator_constructor(create_fft)

    def test_call(self, create_fft, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_fft, create_wavefront)


class TestMFT:
    """Test the CartesianMFT class."""

    def test_constructor(self, create_mft):
        """Tests the constructor."""
        _test_propagator_constructor(create_mft)
        with pytest.raises(TypeError):
            create_mft(pixel_scale=np.array([1]))

    def test_call(self, create_mft, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_mft, create_wavefront)


class TestShiftedMFT:
    """Test the ShiftedCartesianMFT class."""

    def test_constructor(self, create_shifted_mft):
        """Tests the constructor."""
        _test_propagator_constructor(create_shifted_mft)
        with pytest.raises(TypeError):
            create_shifted_mft(shift=np.array([1]))

    def test_call(self, create_shifted_mft, create_wavefront):
        """Tests the __call__ method."""
        _test_call(create_shifted_mft, create_wavefront)


class TestFarFieldFresnel:
    """Test the FarFieldFresnel class."""

    def test_constructor(self, create_far_field_fresnel):
        """Tests the constructor."""
        # Manully test here since inverse is not implemented
        create_far_field_fresnel()
        with pytest.raises(TypeError):
            create_far_field_fresnel(focal_length=np.array([1]))
        with pytest.raises(TypeError):
            create_far_field_fresnel(focal_shift=np.array([1]))
        with pytest.raises(NotImplementedError):
            create_far_field_fresnel(inverse=True)

    def test_call(self, create_far_field_fresnel, create_fresnel_wavefront):
        """Tests the __call__ method."""
        create_far_field_fresnel()(create_fresnel_wavefront())
