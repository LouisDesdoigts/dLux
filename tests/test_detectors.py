from __future__ import annotations
import jax.numpy as np
import pytest
import dLux
from jax import config
config.update("jax_debug_nans", True)

Array = np.ndarray


class TestApplyPixelResponse(object):
    """
    Tests the ApplyPixelResponse class.
    """


    def test_constructor(self, create_pixel_response: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_pixel_response(pixel_response=np.ones(1))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_pixel_response(pixel_response=np.array(1.))

        # Test wrong dims
        with pytest.raises(AssertionError):
            create_pixel_response(pixel_response=np.ones((1, 1, 1)))

        # Test functioning
        create_pixel_response()


    def test_call(self, create_pixel_response: callable) -> None:
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        create_pixel_response()(image)


class TestApplyJitter(object):
    """
    Tests the ApplyJitter class.
    """


    def test_constructor(self, create_jitter: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_jitter(sigma=np.ones(1))

        # Test functioning
        create_jitter()


    def test_call(self, create_jitter: callable) -> None:
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        create_jitter()(image)


class TestApplySaturation(object):
    """
    Tests the ApplySaturation class.
    """


    def test_constructor(self, create_saturation: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_saturation(saturation=np.ones(1))

        # Test functioning
        create_saturation()


    def test_call(self, create_saturation: callable) -> None:
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        create_saturation()(image)


class TestAddConstant(object):
    """
    Tests the AddConstant class.
    """


    def test_constructor(self, create_constant: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_constant(value=np.ones(1))

        # Test functioning
        create_constant()


    def test_call(self, create_constant: callable) -> None:
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        create_constant()(image)


class TestIntegerDownsample(object):
    """
    Tests the IntegerDownsample class.
    """


    def test_constructor(self, create_integer_downsample: callable) -> None:
        """
        Tests the constructor.
        """
        # Test functioning
        create_integer_downsample()


    def test_call(self, create_integer_downsample: callable) -> None:
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        create_integer_downsample()(image)


class TestRotate(object):
    """
    Tests the Rotate class.
    """


    def test_constructor(self, create_rotate_detector: callable) -> None:
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_rotate_detector(angle=np.ones(1))

        # Test functioning
        create_rotate_detector()


    def test_call(self, create_rotate_detector: callable) -> None:
        """
        Tests the __call__ method.
        """
        # Test regular rotation
        image = np.ones((16, 16))
        create_rotate_detector()(image)

        # Test fourier
        with pytest.raises(NotImplementedError):
            create_rotate_detector(fourier=True)(image)
