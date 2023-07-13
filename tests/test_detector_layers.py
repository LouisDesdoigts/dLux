import jax.numpy as np
import pytest
from jax import config

config.update("jax_debug_nans", True)


def _test_call(constructor, image_constructor):
    """Tests the __call__ method."""
    constructor()(image_constructor())


class TestApplyPixelResponse:
    """Tests the ApplyPixelResponse class."""

    def test_constructor(self, create_pixel_response):
        """Tests the constructor."""
        create_pixel_response()
        with pytest.raises(ValueError):
            create_pixel_response(pixel_response=np.ones(1))

    def test_call(self, create_pixel_response, create_image):
        """Tests the __call__ method."""
        _test_call(create_pixel_response, create_image)


class TestApplyJitter:
    """Tests the ApplyJitter class."""

    def test_constructor(self, create_jitter):
        """Tests the constructor."""
        create_jitter()
        with pytest.raises(ValueError):
            create_jitter(sigma=np.array([1.0]))

    def test_call(self, create_jitter, create_image):
        """Tests the __call__ method."""
        _test_call(create_jitter, create_image)


class TestApplySaturation:
    """Tests the ApplySaturation class."""

    def test_constructor(self, create_saturation):
        """Tests the constructor."""
        create_saturation()
        with pytest.raises(ValueError):
            create_saturation(saturation=np.array([1.0]))

    def test_call(self, create_saturation, create_image):
        """Tests the __call__ method."""
        _test_call(create_saturation, create_image)


class TestAddConstant:
    """Tests the AddConstant class."""

    def test_constructor(self, create_constant):
        """Tests the constructor."""
        create_constant()
        with pytest.raises(ValueError):
            create_constant(value=np.array([1.0]))

    def test_call(self, create_constant, create_image):
        """Tests the __call__ method."""
        _test_call(create_constant, create_image)


class TestIntegerDownsample:
    """Tests the IntegerDownsample class."""

    def test_constructor(self, create_integer_downsample):
        """Tests the constructor."""
        create_integer_downsample()

    def test_call(self, create_integer_downsample, create_image):
        """Tests the __call__ method."""
        _test_call(create_integer_downsample, create_image)


class TestRotateDetector:
    """Tests the RotateDetector class."""

    def test_constructor(self, create_rotate_detector):
        """Tests the constructor."""
        create_rotate_detector()
        with pytest.raises(ValueError):
            create_rotate_detector(angle=np.array([1.0]))

    def test_call(self, create_rotate_detector, create_image):
        """Tests the __call__ method."""
        _test_call(create_rotate_detector, create_image)
