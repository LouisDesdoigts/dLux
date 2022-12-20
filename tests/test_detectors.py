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


    def test_constructor(self, create_jitter: callable) -> None::
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            create_jitter(sigma=np.ones(1))

        # Test functioning
        create_jitter()


    def test_call(self, create_jitter: callable) -> None::
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


class TestIntegerDownsample(UtilityUser):
    """
    Tests the IntegerDownsample class.
    """
    utility : IntegerDownsampleUtility = IntegerDownsampleUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        image = np.ones((16, 16))
        self.utility.construct()(image)


class TestRotate(UtilityUser):
    """
    Tests the Rotate class.
    """
    utility : RotateUtility = RotateUtility()


    def test_constructor(self):
        """
        Tests the constructor.
        """
        # Test wrong dims
        with pytest.raises(AssertionError):
            self.utility.construct(angle=np.ones(1))

        # Test functioning
        self.utility.construct()


    def test_call(self):
        """
        Tests the __call__ method.
        """
        # Test regular rotation
        image = np.ones((16, 16))
        self.utility.construct()(image)

        # Test fourier
        with pytest.raises(NotImplementedError):
            self.utility.construct(fourier=True)(image)
